# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-P150 tensor-parallel decoder layer for ``google/gemma-4-31B``.

The layer boundary is a replicated BF16 residual tensor.  Attention and MLP
weights are tensor-parallel: QKV/gate/up are column parallel, O/down are row
parallel, and every device owns only its local attention and KV-cache heads.
The public prefill/decode and logical-length contracts are inherited from the
optimized single-chip decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import (
    MLP_CHUNK,
    FunctionalDecoder,
    _validate_target_config,
)
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import (
    DEFAULT_OPTIMIZATION_POLICY,
    OptimizedDecoder,
    _dram_weight_memory_config,
)
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_MAX_SEQ,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    apply_rope_decode_peruser,
    chunked_prefill_sdpa_sliding,
    effective_block_size,
    prefill_sdpa_program_config,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig

TARGET_MESH_SHAPE = (1, 4)
TP_SIZE = 4
PAGE_BLOCK_SIZE = 64
QKV_DECODE_OUTPUT_CORES = 32
MLP_DECODE_CORES = 24
MLP_PREFILL_1D_MAX_ROWS = 128


@dataclass(frozen=True)
class MultichipDecoderTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None


def _layer_state(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    for prefix in (f"model.language_model.layers.{layer_idx}.", f"model.layers.{layer_idx}."):
        local = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
        if local:
            return local
    raise KeyError(f"no Gemma 4 layer {layer_idx} weights found")


def _tp_tensor(
    source: torch.Tensor,
    mesh_device,
    *,
    mesh_dim: int,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Load a global TT-layout weight and fracture one dimension over TP=4."""
    return ttnn.from_torch(
        source.detach().contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=mesh_dim),
    )


class _TPOptimizedSharedMLP:
    """GeGLU with BFP4 TP weights and optimized local decode matmuls."""

    def __init__(self, *, mesh_device, mesh_config, ccl_manager, state, policy):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.policy = policy
        self.is_decode = False
        hidden = int(state["gate_proj.weight"].shape[1])
        intermediate = int(state["gate_proj.weight"].shape[0])
        self.local_intermediate = intermediate // TP_SIZE
        if hidden != self.local_intermediate:
            raise ValueError(
                f"Gemma 4 31B TP=4 expects local intermediate == hidden, got " f"{self.local_intermediate} and {hidden}"
            )

        gate = state["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up = state["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down = state["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        # Large-M prefill and M=1 decode use separate physical placements, as
        # in the optimized single-chip baseline.  Both retain BFP4/LoFi.
        self.gate_prefill = _tp_tensor(gate, mesh_device, mesh_dim=3, dtype=policy.mlp_gate_up_weight_dtype)
        self.up_prefill = _tp_tensor(up, mesh_device, mesh_dim=3, dtype=policy.mlp_gate_up_weight_dtype)
        self.down_prefill = _tp_tensor(down, mesh_device, mesh_dim=2, dtype=policy.mlp_down_weight_dtype)
        local_mem = _dram_weight_memory_config(mesh_device, k=hidden, n=hidden)
        self.gate_decode = _tp_tensor(
            gate,
            mesh_device,
            mesh_dim=3,
            dtype=policy.mlp_gate_up_weight_dtype,
            memory_config=local_mem,
        )
        self.up_decode = _tp_tensor(
            up,
            mesh_device,
            mesh_dim=3,
            dtype=policy.mlp_gate_up_weight_dtype,
            memory_config=local_mem,
        )
        self.down_decode = _tp_tensor(
            down,
            mesh_device,
            mesh_dim=2,
            dtype=policy.mlp_down_weight_dtype,
            memory_config=local_mem,
        )
        self.gate_up_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.mlp_gate_up_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.down_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.mlp_down_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _reduce(self, partial):
        if partial.is_sharded():
            interleaved = ttnn.sharded_to_interleaved(partial, ttnn.DRAM_MEMORY_CONFIG)
            partial.deallocate(True)
            partial = interleaved
        return ccl_allreduce(partial, self.mesh_config, self.ccl_manager)

    def _decode_memory_config(self, num_cores: int, width: int) -> ttnn.MemoryConfig:
        if width % (ttnn.TILE_SIZE * num_cores):
            raise ValueError(f"decode width {width} is not tile-divisible across {num_cores} cores")
        core_grid = ttnn.num_cores_to_corerangeset(
            num_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width // num_cores),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @staticmethod
    def _decode_program_config(*, k, n, num_cores, in0_block_w, fused_activation=None):
        k_tiles_per_core = k // (ttnn.TILE_SIZE * num_cores)
        if k_tiles_per_core % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} does not divide {k_tiles_per_core} " f"K tiles/core for K={k}")
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=fused_activation,
        )

    def __call__(self, hidden_states):
        if not self.is_decode:
            rows = hidden_states.shape[-2]
            if rows <= MLP_PREFILL_1D_MAX_ROWS:
                program_args = dict(
                    compute_with_storage_grid_size=(8, 3),
                    in0_block_w=7,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=rows // ttnn.TILE_SIZE,
                    per_core_N=self.local_intermediate // (ttnn.TILE_SIZE * MLP_DECODE_CORES),
                    fuse_batch=True,
                    mcast_in0=True,
                )
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    **program_args,
                    fused_activation=None,
                )
                gate_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    **program_args,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
                )
                up = ttnn.linear(
                    hidden_states,
                    self.up_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program,
                    compute_kernel_config=self.gate_up_compute,
                )
                gate = ttnn.linear(
                    hidden_states,
                    self.gate_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=gate_program,
                    compute_kernel_config=self.gate_up_compute,
                )
                activated = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                gate.deallocate(True)
                up.deallocate(True)
                partial = ttnn.linear(
                    activated,
                    self.down_prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program,
                    compute_kernel_config=self.down_compute,
                )
                activated.deallocate(True)
                return self._reduce(partial)
            gate = ttnn.linear(hidden_states, self.gate_prefill)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            up = ttnn.linear(hidden_states, self.up_prefill)
            activated = ttnn.mul(gate, up)
            gate.deallocate(True)
            up.deallocate(True)
            partial = ttnn.linear(activated, self.down_prefill)
            activated.deallocate(True)
            return self._reduce(partial)

        hidden = hidden_states.shape[-1]
        input_mem = self._decode_memory_config(self.policy.decode_num_cores, hidden)
        local_mem = self._decode_memory_config(self.policy.decode_num_cores, self.local_intermediate)
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        gate_program = self._decode_program_config(
            k=hidden,
            n=self.local_intermediate,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        )
        up_program = self._decode_program_config(
            k=hidden,
            n=self.local_intermediate,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
        )
        up = ttnn.linear(
            sharded_input,
            self.up_decode,
            memory_config=local_mem,
            program_config=up_program,
            compute_kernel_config=self.gate_up_compute,
        )
        gate = ttnn.linear(
            sharded_input,
            self.gate_decode,
            memory_config=local_mem,
            program_config=gate_program,
            compute_kernel_config=self.gate_up_compute,
        )
        sharded_input.deallocate(True)
        activated = ttnn.mul(gate, up, memory_config=local_mem)
        gate.deallocate(True)
        up.deallocate(True)
        down_program = self._decode_program_config(
            k=self.local_intermediate,
            n=hidden,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.down_in0_block_w,
        )
        partial = ttnn.linear(
            activated,
            self.down_decode,
            memory_config=input_mem,
            program_config=down_program,
            compute_kernel_config=self.down_compute,
        )
        activated.deallocate(True)
        return self._reduce(partial)


class MultichipDecoder(OptimizedDecoder):
    """Real Gemma 4 31B decoder layer specialized for the local 1x4 mesh."""

    baseline_cls = OptimizedDecoder
    mesh_profile = {
        "name": "gemma4_31b_p150x4_tp4_replicated_residual_v1",
        "single_chip_baseline": DEFAULT_OPTIMIZATION_POLICY.name,
        "target_mesh": "1x4 Blackhole P150b",
        "tp": TP_SIZE,
        "activation_contract": "replicated BF16 residual at layer input/output",
        "attention": "column-parallel local-head QKV/SDPA; row-parallel O plus TP sum",
        "mlp": "BFP4 column-parallel gate/up; BFP4 row-parallel down plus TP sum; 24-core local decode",
        "kv_cache": "BFP8 paged local KV heads; replicated page table and positions",
        "moe": "not applicable: dense target",
    }

    def __init__(self, **kwargs):
        # Bypass FusedDecoder/OptimizedDecoder construction-time module
        # rewrites: this class already installs its TP-aware attention and MLP.
        FunctionalDecoder.__init__(self, **kwargs)
        self.policy = DEFAULT_OPTIMIZATION_POLICY
        self.attention_compute = None
        self.decode_wqkv = None
        self.decode_wq = None
        self.decode_wk = None
        self.decode_wv = None
        self.decode_o_proj = None
        self.mesh_config = None
        self.ccl_manager = None
        self.qkv_decode_output_cores = QKV_DECODE_OUTPUT_CORES
        self.timings = MultichipDecoderTimings()

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        tensor_cache_path=None,
        optimization_policy=DEFAULT_OPTIMIZATION_POLICY,
        bounded_sliding_kv_cache=True,
        num_links=2,
        qkv_decode_output_cores=QKV_DECODE_OUTPUT_CORES,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported MultichipDecoder kwargs: {sorted(kwargs)}")
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != TP_SIZE:
            raise ValueError(
                f"MultichipDecoder requires MeshShape{TARGET_MESH_SHAPE}, got "
                f"shape={tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()}"
            )
        if qkv_decode_output_cores not in (8, QKV_DECODE_OUTPUT_CORES):
            raise ValueError("qkv_decode_output_cores must be one of the validated 8/32-core geometries")
        contract = _validate_target_config(hf_config, layer_idx)
        model_args = Gemma4ModelArgs.from_hf_config(hf_config)
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=TP_SIZE), prefill=ModeConfig(tp=TP_SIZE))
        ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
        layer = Gemma4DecoderLayer(
            mesh_device=mesh_device,
            hf_config=model_args,
            state_dict=state_dict,
            layer_idx=layer_idx,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat8_b,
            attention_dtype=optimization_policy.attention_weight_dtype,
            shared_mlp_dtype=ttnn.bfloat8_b,
            tensor_cache_path=tensor_cache_path,
            mesh_config=mesh_config,
            max_seq_len=contract.max_position_embeddings,
            max_local_batch_size=32,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        old_mlp = layer.shared_mlp
        local = _layer_state(state_dict, layer_idx)
        mlp_state = {key.removeprefix("mlp."): value for key, value in local.items() if key.startswith("mlp.")}
        mlp_policy = replace(
            optimization_policy,
            name=f"{optimization_policy.name}_tp4_square_mlp_24c",
            decode_num_cores=MLP_DECODE_CORES,
            gate_up_in0_block_w=7,
            down_in0_block_w=7,
        )
        layer.shared_mlp = _TPOptimizedSharedMLP(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            state=mlp_state,
            policy=mlp_policy,
        )
        for weight in (old_mlp.gate_proj, old_mlp.up_proj, old_mlp.down_proj):
            weight.deallocate(True)
        decoder = cls(layer=layer, contract=contract, layer_idx=layer_idx, mesh_device=mesh_device)
        decoder.policy = optimization_policy
        decoder.mesh_config = mesh_config
        decoder.ccl_manager = ccl_manager
        decoder.qkv_decode_output_cores = qkv_decode_output_cores
        decoder.attention_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=optimization_policy.attention_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Preserve the optimized baseline's decode dataflow with TP-local
        # packed QKV and row-local O copies, each width-sharded over all DRAM
        # banks.  The demo weights remain interleaved for large-M prefill.
        config = layer.self_attn.config
        attn_state = {
            key.removeprefix("self_attn."): value for key, value in local.items() if key.startswith("self_attn.")
        }
        q_weight = attn_state["q_proj.weight"]
        k_weight = attn_state["k_proj.weight"]
        v_weight = k_weight if config.use_kv_tying else attn_state["v_proj.weight"]
        packed_per_device = []
        q_chunks = torch.chunk(q_weight, TP_SIZE, dim=0)
        k_chunks = torch.chunk(k_weight, TP_SIZE, dim=0)
        v_chunks = torch.chunk(v_weight, TP_SIZE, dim=0)
        for device_idx in range(TP_SIZE):
            packed_per_device.append(
                torch.cat(
                    (
                        q_chunks[device_idx].transpose(-2, -1),
                        k_chunks[device_idx].transpose(-2, -1),
                        v_chunks[device_idx].transpose(-2, -1),
                    ),
                    dim=-1,
                )
            )
        packed_qkv = torch.cat(packed_per_device, dim=-1).unsqueeze(0).unsqueeze(0)
        local_qkv_width = packed_per_device[0].shape[-1]
        decoder.decode_wqkv = _tp_tensor(
            packed_qkv,
            mesh_device,
            mesh_dim=3,
            dtype=optimization_policy.attention_weight_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=config.hidden_size, n=local_qkv_width),
        )
        o_source = attn_state["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        local_o_k = config.num_attention_heads * config.head_dim // TP_SIZE
        decoder.decode_o_proj = _tp_tensor(
            o_source,
            mesh_device,
            mesh_dim=2,
            dtype=optimization_policy.attention_weight_dtype,
            memory_config=_dram_weight_memory_config(mesh_device, k=local_o_k, n=config.hidden_size),
        )
        return decoder

    def _decode_attention_tp(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        current_position,
        current_position_cache,
        batch_size,
    ):
        """TP-local decode with DRAM-sharded packed QKV/O projections."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        local_heads = config.num_attention_heads // TP_SIZE
        local_kv_heads = config.num_key_value_heads // TP_SIZE
        input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, hidden_states.shape[-1])
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        qkv_n = local_heads * config.head_dim + 2 * local_kv_heads * config.head_dim
        qkv_grid = ttnn.num_cores_to_corerangeset(
            self.qkv_decode_output_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        qkv_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, qkv_n // self.qkv_decode_output_cores),
            core_grid=qkv_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        qkv_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.policy.qkv_in0_block_w,
            per_core_M=1,
            per_core_N=qkv_n // (ttnn.TILE_SIZE * self.qkv_decode_output_cores),
        )
        qkv_sharded = ttnn.linear(
            sharded_input,
            self.decode_wqkv,
            memory_config=qkv_mem,
            program_config=qkv_program,
            compute_kernel_config=self.attention_compute,
        )
        sharded_input.deallocate(True)
        qkv = ttnn.sharded_to_interleaved(qkv_sharded, ttnn.L1_MEMORY_CONFIG)
        qkv_sharded.deallocate(True)
        q, k, v = split_qkv_heads_decode(qkv, config, weights.is_global, tp=TP_SIZE, kv_replicated=False)
        qkv.deallocate(True)

        q_sharded_mem = q.memory_config()
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)

        cos_cache, sin_cache = rope_mats
        cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(current_position, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(current_position, sin_cache, layout=ttnn.TILE_LAYOUT))
        if batch_size == 1:
            q = apply_rope(q, cos_pos, sin_pos, token_index=0)
            k = apply_rope(k, cos_pos, sin_pos, token_index=0)
        else:
            cos_b = ttnn.transpose(cos_pos, 1, 2)[:, :batch_size, :, :]
            sin_b = ttnn.transpose(sin_pos, 1, 2)[:, :batch_size, :, :]
            q = apply_rope_decode_peruser(q, cos_b, sin_b)
            k = apply_rope_decode_peruser(k, cos_b, sin_b)

        cache_position = current_position_cache if current_position_cache is not None else current_position
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, local_kv_heads)
        if config.cache_position_modulo is None:
            device_grid = self.mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch_size, device_grid.x)
            while batch_size % grid_x:
                grid_x -= 1
            grid_h = batch_size // grid_x
            k_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_x - 1, grid_h - 1),
                    )
                }
            )
            v_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, grid_h),
                        ttnn.CoreCoord(grid_x - 1, 2 * grid_h - 1),
                    )
                }
            )
            k_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=k_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            v_memory_config = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, config.head_dim),
                core_grid=v_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k = ttnn.to_memory_config(k, k_memory_config)
            v = ttnn.to_memory_config(v, v_memory_config)
            ttnn.experimental.paged_fused_update_cache(
                k_cache,
                k,
                v_cache,
                v,
                update_idxs_tensor=cache_position,
                page_table=page_table,
            )
        else:
            k = ttnn.to_memory_config(k, q_sharded_mem)
            v = ttnn.to_memory_config(v, q_sharded_mem)
            update_args = dict(
                update_idxs_tensor=cache_position,
                page_table=page_table,
                block_size=block_size,
                num_kv_heads=local_kv_heads,
                cache_position_modulo=config.cache_position_modulo,
            )
            ttnn.experimental.paged_update_cache(k_cache, k, **update_args)
            ttnn.experimental.paged_update_cache(v_cache, v, **update_args)
        k.deallocate(True)
        v.deallocate(True)

        sdpa_grid = (
            ttnn.CoreCoord(8, 4) if config.head_dim >= 512 else self.mesh_device.compute_with_storage_grid_size()
        )
        sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cache_position,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=(config.sliding_window if config.is_sliding else None),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program,
            block_size=block_size,
            num_kv_heads=local_kv_heads,
            **(
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            ),
        )
        q.deallocate(True)
        from models.tt_transformers.tt.model_config import num_to_corerange

        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(batch_size, grid.x)
        if batch_size >= grid_x and batch_size % grid_x:
            grid_x = max(x for x in range(grid_x, 0, -1) if batch_size % x == 0 and batch_size // x <= grid.y)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(batch_size, grid_x=grid_x, grid_y=grid.y)})
        head_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sdpa_sharded = ttnn.to_memory_config(sdpa, head_mem)
        sdpa.deallocate(True)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(sdpa_sharded, num_heads=local_heads)
        sdpa_sharded.deallocate(True)
        output = ttnn.sharded_to_interleaved(concatenated, ttnn.DRAM_MEMORY_CONFIG)
        concatenated.deallocate(True)
        if output.shape[2] != batch_size:
            padded = output
            output = padded[:, :, :batch_size, :]
            padded.deallocate(True)

        o_k = local_heads * config.head_dim
        o_input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, o_k)
        o_output_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, config.hidden_size)
        o_input = ttnn.to_memory_config(output, o_input_mem)
        output.deallocate(True)
        o_program = self._decode_matmul_program_config(
            k=o_k,
            n=config.hidden_size,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.o_proj_in0_block_w,
        )
        projected_sharded = ttnn.linear(
            o_input,
            self.decode_o_proj,
            memory_config=o_output_mem,
            program_config=o_program,
            compute_kernel_config=self.attention_compute,
        )
        o_input.deallocate(True)
        projected = ttnn.sharded_to_interleaved(projected_sharded, ttnn.DRAM_MEMORY_CONFIG)
        projected_sharded.deallocate(True)
        return ccl_allreduce(projected, self.mesh_config, self.ccl_manager)

    def init_paged_kv_cache(self, *, max_context=262_144, batch_size=1):
        config = self.layer.self_attn.config
        physical_context = config.sliding_window if config.is_sliding else max_context
        num_blocks_per_user = (physical_context + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
        paged = PagedAttentionConfig(
            block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=num_blocks_per_user * batch_size,
        )
        cache = init_kv_cache(
            self.mesh_device,
            config,
            paged_attention_config=paged,
            cache_dtype=self.policy.kv_cache_dtype,
            max_num_blocks_override=num_blocks_per_user * batch_size,
        )
        rows = torch.arange(num_blocks_per_user * batch_size, dtype=torch.int32).reshape(
            batch_size, num_blocks_per_user
        )
        page_table = ttnn.from_torch(
            rows,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return cache, page_table

    def _prefill_attention_tp(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        user_id,
        valid_seq_len,
    ):
        """TP-local prefill with the optimized BFP8 cache-fill contract."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        qkv = apply_qkv_projection(hidden_states, weights)
        q, k, v = split_qkv_heads_prefill(
            qkv,
            config,
            weights.is_global,
            tp=TP_SIZE,
            kv_replicated=weights.kv_replicated,
        )
        qkv.deallocate(True)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
        cos_cache, sin_cache = rope_mats
        q = apply_rope(q, cos_cache, sin_cache)
        k = apply_rope(k, cos_cache, sin_cache)

        local_kv_heads = config.num_key_value_heads // TP_SIZE
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, local_kv_heads)
        modulo = {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo else {}
        if config.cache_position_modulo is not None and valid_seq_len < k.shape[-2]:
            self._fill_bounded_sliding_cache_exact(
                k_cache,
                v_cache,
                k,
                v,
                page_table,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
                block_size=block_size,
                num_kv_heads=local_kv_heads,
                cache_position_modulo=config.cache_position_modulo,
            )
        else:
            k_fill = ttnn.typecast(k, k_cache.dtype) if k.dtype != k_cache.dtype else k
            v_fill = ttnn.typecast(v, v_cache.dtype) if v.dtype != v_cache.dtype else v
            ttnn.experimental.paged_fill_cache(
                k_cache, k_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, v_fill, page_table, batch_idx=user_id, block_size=block_size, **modulo
            )
            if k_fill is not k:
                k_fill.deallocate(True)
            if v_fill is not v:
                v_fill.deallocate(True)

        seq_len = q.shape[-2]
        if seq_len > PREFILL_SDPA_MAX_SEQ and config.is_sliding:
            sdpa = chunked_prefill_sdpa_sliding(q, k, v, config.sliding_window, config.head_dim, scale=1.0)
        elif seq_len > PREFILL_SDPA_MAX_SEQ:
            sdpa = self._chunked_full_attention(q, k_cache, v_cache, page_table, user_id, config.head_dim)
        else:
            sdpa = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                sliding_window_size=config.sliding_window if config.is_sliding else None,
                program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
            )
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)
        local_heads = config.num_attention_heads // TP_SIZE
        concatenated = self._concatenate_heads(sdpa, num_heads=local_heads, head_dim=config.head_dim)
        sdpa.deallocate(True)
        partial = ttnn.linear(concatenated, weights.o_proj)
        concatenated.deallocate(True)
        return ccl_allreduce(partial, self.mesh_config, self.ccl_manager)

    def _forward_device(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        is_decode,
        current_position=None,
        current_position_cache=None,
        token_index=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
    ):
        """Host-free replicated-residual composition for both layer kinds."""
        self.layer.shared_mlp.is_decode = is_decode
        residual = hidden_states
        normed = self.layer.input_layernorm.forward(hidden_states)
        attn_input = normed
        if not is_decode and batch_size > 1:
            attn_input = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        if is_decode:
            attn_output = self._decode_attention_tp(
                attn_input,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                current_position=current_position,
                current_position_cache=current_position_cache,
                batch_size=batch_size,
            )
        else:
            attn_output = self._prefill_attention_tp(
                attn_input,
                rope_mats=rope_mats,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                valid_seq_len=valid_seq_len,
            )
        normed.deallocate(True)
        attn_output = self.layer.post_attention_layernorm.forward(attn_output)
        if not is_decode and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3], -1])
        hidden_states = ttnn.add(residual, attn_output)
        attn_output.deallocate(True)

        residual = hidden_states
        normed = self.layer.pre_feedforward_layernorm.forward(hidden_states)
        if not is_decode and normed.shape[-2] > MLP_CHUNK:
            outputs = []
            for start in range(0, normed.shape[-2], MLP_CHUNK):
                end = min(start + MLP_CHUNK, normed.shape[-2])
                chunk = ttnn.slice(normed, [0, 0, start, 0], [1, 1, end, normed.shape[-1]])
                outputs.append(self.layer.shared_mlp(chunk))
                chunk.deallocate(True)
            mlp_output = ttnn.concat(outputs, dim=2)
            for output in outputs:
                output.deallocate(True)
        else:
            mlp_output = self.layer.shared_mlp(normed)
        normed.deallocate(True)
        hidden_states = self.layer.post_feedforward_layernorm.forward(mlp_output)
        mlp_output.deallocate(True)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)
        if self.layer.layer_scalar != 1.0:
            scaled = ttnn.mul(combined, self.layer.layer_scalar)
            combined.deallocate(True)
            combined = scaled
        return combined


__all__ = [
    "MultichipDecoder",
    "MultichipDecoderTimings",
    "PAGE_BLOCK_SIZE",
    "QKV_DECODE_OUTPUT_CORES",
    "TARGET_MESH_SHAPE",
    "TP_SIZE",
]
