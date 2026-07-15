# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device decoder layer for ``google/gemma-4-31B``.

The optimized stage keeps :class:`FusedDecoder`'s graph and public paged-cache
contract.  It specializes the decode weight-matmul path: weights are placed in
DRAM-sharded storage, activations stay width-sharded in L1 across each
projection group, and every material matmul has an explicit program and
compute-kernel configuration.  Prefill continues to use the fused decoder's
large-M path with the same logical-length padding and cache semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

import ttnn
from models.autoports.google_gemma_4_31b.tt.fused_decoder import FULL_ATTN_Q_CHUNK, FusedDecoder
from models.demos.gemma4.tt.attention.operations import (
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


@dataclass(frozen=True)
class DecoderOptimizationPolicy:
    """Cumulative decode optimization contract used by tests and sweeps."""

    name: str = "p150_bfp8attn_bfp4mlp_lofi_dram_sharded_v1"
    attention_weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_gate_up_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_down_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    attention_qkv_weight_dtype: ttnn.DataType | None = None
    attention_o_weight_dtype: ttnn.DataType | None = None
    attention_qkv_math_fidelity: ttnn.MathFidelity | None = None
    attention_o_math_fidelity: ttnn.MathFidelity | None = None
    mlp_gate_up_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    mlp_down_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    decode_num_cores: int = 8
    qkv_in0_block_w: int = 3
    o_proj_in0_block_w: int = 8
    gate_up_in0_block_w: int = 7
    down_in0_block_w: int = 21
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    attention_projection_topology: str = "packed"
    mlp_gate_up_topology: str = "separate"
    mlp_packed_output_dtype: ttnn.DataType = ttnn.bfloat16
    prefill_qkv_topology: str = "interleaved_2d"

    @property
    def resolved_attention_qkv_weight_dtype(self) -> ttnn.DataType:
        return self.attention_qkv_weight_dtype or self.attention_weight_dtype

    @property
    def resolved_attention_o_weight_dtype(self) -> ttnn.DataType:
        return self.attention_o_weight_dtype or self.attention_weight_dtype

    @property
    def resolved_attention_qkv_math_fidelity(self) -> ttnn.MathFidelity:
        return self.attention_qkv_math_fidelity or self.attention_math_fidelity

    @property
    def resolved_attention_o_math_fidelity(self) -> ttnn.MathFidelity:
        return self.attention_o_math_fidelity or self.attention_math_fidelity


DEFAULT_OPTIMIZATION_POLICY = DecoderOptimizationPolicy()


def _local_layer_state(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    prefixes = (f"model.language_model.layers.{layer_idx}.", f"model.layers.{layer_idx}.")
    for prefix in prefixes:
        local = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
        if local:
            return local
    raise KeyError(f"no Gemma 4 layer {layer_idx} weights found")


def _dram_weight_memory_config(mesh_device, *, k: int, n: int) -> ttnn.MemoryConfig:
    """Width-shard a decode weight across every Blackhole DRAM bank."""
    grid = mesh_device.dram_grid_size()
    num_banks = grid.x * grid.y
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * num_banks)) * ttnn.TILE_SIZE * num_banks
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, (k, padded_n // num_banks), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _load_weight(mesh_device, source: torch.Tensor, *, dtype: ttnn.DataType, dram_sharded: bool) -> ttnn.Tensor:
    source = source.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
    return ttnn.from_torch(
        source,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=(
            _dram_weight_memory_config(mesh_device, k=source.shape[-2], n=source.shape[-1])
            if dram_sharded
            else ttnn.DRAM_MEMORY_CONFIG
        ),
    )


class _OptimizedSharedMLP:
    """GeGLU MLP with a DRAM-sharded, width-sharded decode dataflow."""

    def __init__(
        self,
        *,
        mesh_device,
        gate_proj,
        up_proj,
        down_proj,
        decode_gate_proj,
        decode_up_proj,
        decode_down_proj,
        decode_packed_gate_up,
        policy: DecoderOptimizationPolicy,
    ):
        self.mesh_device = mesh_device
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.decode_gate_proj = decode_gate_proj
        self.decode_up_proj = decode_up_proj
        self.decode_down_proj = decode_down_proj
        self.decode_packed_gate_up = decode_packed_gate_up
        self.policy = policy
        self.is_decode = False
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

    def __call__(self, hidden_states):
        # Large-M prefill uses the fused stage's proven auto-selected 2D path.
        if not self.is_decode:
            gate = ttnn.linear(hidden_states, self.gate_proj)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            up = ttnn.linear(hidden_states, self.up_proj)
            hidden = ttnn.mul(gate, up)
            gate.deallocate(True)
            up.deallocate(True)
            output = ttnn.linear(hidden, self.down_proj)
            hidden.deallocate(True)
            return output

        k = hidden_states.shape[-1]
        n = (
            self.decode_packed_gate_up.shape[-1] // 2
            if self.policy.mlp_gate_up_topology == "packed"
            else self.decode_gate_proj.shape[-1]
        )
        input_mem = OptimizedDecoder._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, k)
        intermediate_mem = OptimizedDecoder._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, n)
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        gate_program = OptimizedDecoder._decode_matmul_program_config(
            k=k,
            n=n,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        )
        up_program = OptimizedDecoder._decode_matmul_program_config(
            k=k,
            n=n,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.gate_up_in0_block_w,
        )
        if self.policy.mlp_gate_up_topology == "packed":
            packed_n = 2 * n
            packed_mem = OptimizedDecoder._decode_memory_config(
                self.mesh_device, self.policy.decode_num_cores, packed_n
            )
            packed_program = OptimizedDecoder._decode_matmul_program_config(
                k=k,
                n=packed_n,
                num_cores=self.policy.decode_num_cores,
                in0_block_w=self.policy.gate_up_in0_block_w,
            )
            packed_sharded = ttnn.linear(
                sharded_input,
                self.decode_packed_gate_up,
                dtype=self.policy.mlp_packed_output_dtype,
                memory_config=packed_mem,
                program_config=packed_program,
                compute_kernel_config=self.gate_up_compute,
            )
            sharded_input.deallocate(True)
            packed = ttnn.sharded_to_interleaved(packed_sharded, ttnn.DRAM_MEMORY_CONFIG)
            packed_sharded.deallocate(True)
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, packed.shape[2], n])
            up = ttnn.slice(packed, [0, 0, 0, n], [1, 1, packed.shape[2], packed_n])
            packed.deallocate(True)
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True)
            gate = ttnn.to_memory_config(gate, intermediate_mem)
            up_sharded = ttnn.to_memory_config(up, intermediate_mem)
            up.deallocate(True)
        else:
            # Two 21,504-wide BF16 outputs plus the next matmul's static
            # circular buffers do not coexist in P150 L1. Keep ``up`` in DRAM
            # while fused-GELU ``gate`` is produced in L1.
            up_sharded = ttnn.linear(
                sharded_input,
                self.decode_up_proj,
                memory_config=intermediate_mem,
                program_config=up_program,
                compute_kernel_config=self.gate_up_compute,
            )
            up = ttnn.sharded_to_interleaved(up_sharded, ttnn.DRAM_MEMORY_CONFIG)
            up_sharded.deallocate(True)
            gate = ttnn.linear(
                sharded_input,
                self.decode_gate_proj,
                memory_config=intermediate_mem,
                program_config=gate_program,
                compute_kernel_config=self.gate_up_compute,
            )
            sharded_input.deallocate(True)
            up_sharded = ttnn.to_memory_config(up, intermediate_mem)
            up.deallocate(True)
        hidden = ttnn.mul(gate, up_sharded, memory_config=intermediate_mem)
        gate.deallocate(True)
        up_sharded.deallocate(True)
        output_mem = OptimizedDecoder._decode_memory_config(
            self.mesh_device, self.policy.decode_num_cores, self.decode_down_proj.shape[-1]
        )
        down_program = OptimizedDecoder._decode_matmul_program_config(
            k=n,
            n=self.decode_down_proj.shape[-1],
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.down_in0_block_w,
        )
        output_sharded = ttnn.linear(
            hidden,
            self.decode_down_proj,
            memory_config=output_mem,
            program_config=down_program,
            compute_kernel_config=self.down_compute,
        )
        hidden.deallocate(True)
        output = ttnn.sharded_to_interleaved(output_sharded, ttnn.DRAM_MEMORY_CONFIG)
        output_sharded.deallocate(True)
        return output


class OptimizedDecoder(FusedDecoder):
    """Fused Gemma 4 decoder with explicit P150 decode configurations."""

    # The paged-update kernels repack BF16/FP32 inputs into the configured
    # cache dtype.  Packed activations are valid between decoder layers, but
    # are not valid direct inputs to either paged update operation.
    cache_update_input_dtype = ttnn.bfloat16
    # Decode head splitting likewise accepts only BF16/FP32. Request BF16
    # directly from QKV matmul so BFP8 residual storage needs no extra copy.
    qkv_split_input_dtype = ttnn.bfloat16

    optimization_profile = {
        "name": DEFAULT_OPTIMIZATION_POLICY.name,
        "activation_dtype": "bfloat16",
        "norm_dtype": "bfloat16",
        "attention_weight_dtype": "bfloat8_b",
        "mlp_gate_up_weight_dtype": "bfloat4_b",
        "mlp_down_weight_dtype": "bfloat4_b",
        "kv_cache_dtype": "bfloat8_b",
        "attention_math_fidelity": "LoFi",
        "mlp_math_fidelity": "LoFi",
        "decode_weight_layout": "DRAM width sharded",
        "decode_activation_layout": "L1 width sharded inside projection groups",
        "projection_topology": "packed QKV; separate fused-GELU gate and up; down projection",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy = DEFAULT_OPTIMIZATION_POLICY
        self.attention_compute = None
        self.attention_qkv_compute = None
        self.attention_o_compute = None
        self.decode_wqkv = None
        self.decode_wq = None
        self.decode_wk = None
        self.decode_wv = None
        self.decode_o_proj = None

    @classmethod
    def _prepare_cache_update_input(cls, tensor):
        """Normalize only the paged-update operand, preserving residual dtype."""
        if tensor.dtype == cls.cache_update_input_dtype:
            return tensor
        converted = ttnn.typecast(tensor, cls.cache_update_input_dtype)
        tensor.deallocate(True)
        return converted

    @staticmethod
    def _decode_memory_config(mesh_device, num_cores: int, width: int) -> ttnn.MemoryConfig:
        if width % (ttnn.TILE_SIZE * num_cores):
            raise ValueError(f"decode width {width} is not tile-divisible across {num_cores} cores")
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width // num_cores),
            core_grid=ttnn.CoreGrid(x=num_cores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @staticmethod
    def _decode_matmul_program_config(*, k, n, num_cores, in0_block_w, fused_activation=None):
        k_tiles_per_core = k // (ttnn.TILE_SIZE * num_cores)
        if k_tiles_per_core % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} does not divide {k_tiles_per_core} K tiles/core for K={k}")
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=fused_activation,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        optimization_policy: DecoderOptimizationPolicy = DEFAULT_OPTIMIZATION_POLICY,
        tensor_cache_path=None,
        bounded_sliding_kv_cache=True,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported OptimizedDecoder kwargs: {sorted(kwargs)}")
        # The existing loaders accept BFP8 and establish all norm/cache/module
        # state.  Replace only the material projection tensors below so BFP4
        # cache-name limitations in the demo loader cannot silently select a
        # higher-precision runtime tensor.
        decoder = FusedDecoder.from_state_dict.__func__(
            cls,
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            tensor_cache_path=tensor_cache_path,
            dtype=ttnn.bfloat8_b,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        decoder.policy = optimization_policy
        local = _local_layer_state(state_dict, layer_idx)
        attn_state = {
            key.removeprefix("self_attn."): value for key, value in local.items() if key.startswith("self_attn.")
        }
        mlp_state = {key.removeprefix("mlp."): value for key, value in local.items() if key.startswith("mlp.")}
        config = decoder.layer.self_attn.config
        old_attn = decoder.layer.self_attn.weights
        v_weight = attn_state["k_proj.weight"] if config.use_kv_tying else attn_state["v_proj.weight"]
        qkv_source = torch.cat(
            (
                attn_state["q_proj.weight"],
                attn_state["k_proj.weight"],
                v_weight,
            ),
            dim=0,
        )
        wqkv = _load_weight(
            mesh_device,
            qkv_source,
            dtype=optimization_policy.attention_weight_dtype,
            dram_sharded=False,
        )
        o_proj = _load_weight(
            mesh_device,
            attn_state["o_proj.weight"],
            dtype=optimization_policy.attention_weight_dtype,
            dram_sharded=False,
        )
        if optimization_policy.attention_projection_topology == "packed":
            decoder.decode_wqkv = _load_weight(
                mesh_device,
                qkv_source,
                dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
                dram_sharded=True,
            )
        elif optimization_policy.attention_projection_topology == "split":
            decoder.decode_wq = _load_weight(
                mesh_device,
                attn_state["q_proj.weight"],
                dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
                dram_sharded=True,
            )
            decoder.decode_wk = _load_weight(
                mesh_device,
                attn_state["k_proj.weight"],
                dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
                dram_sharded=True,
            )
            decoder.decode_wv = _load_weight(
                mesh_device,
                v_weight,
                dtype=optimization_policy.resolved_attention_qkv_weight_dtype,
                dram_sharded=True,
            )
        else:
            raise ValueError(
                f"unsupported attention projection topology " f"{optimization_policy.attention_projection_topology!r}"
            )
        decoder.decode_o_proj = _load_weight(
            mesh_device,
            attn_state["o_proj.weight"],
            dtype=optimization_policy.resolved_attention_o_weight_dtype,
            dram_sharded=True,
        )
        decoder.layer.self_attn.weights = replace(old_attn, wqkv=wqkv, o_proj=o_proj)
        old_attn.wqkv.deallocate(True)
        old_attn.o_proj.deallocate(True)

        old_mlp = decoder.layer.shared_mlp
        gate_proj = _load_weight(
            mesh_device,
            mlp_state["gate_proj.weight"],
            dtype=optimization_policy.mlp_gate_up_weight_dtype,
            dram_sharded=False,
        )
        up_proj = _load_weight(
            mesh_device,
            mlp_state["up_proj.weight"],
            dtype=optimization_policy.mlp_gate_up_weight_dtype,
            dram_sharded=False,
        )
        down_proj = _load_weight(
            mesh_device,
            mlp_state["down_proj.weight"],
            dtype=optimization_policy.mlp_down_weight_dtype,
            dram_sharded=False,
        )
        decode_gate_proj = decode_up_proj = decode_packed_gate_up = None
        if optimization_policy.mlp_gate_up_topology == "separate":
            decode_gate_proj = _load_weight(
                mesh_device,
                mlp_state["gate_proj.weight"],
                dtype=optimization_policy.mlp_gate_up_weight_dtype,
                dram_sharded=True,
            )
            decode_up_proj = _load_weight(
                mesh_device,
                mlp_state["up_proj.weight"],
                dtype=optimization_policy.mlp_gate_up_weight_dtype,
                dram_sharded=True,
            )
        elif optimization_policy.mlp_gate_up_topology == "packed":
            decode_packed_gate_up = _load_weight(
                mesh_device,
                torch.cat((mlp_state["gate_proj.weight"], mlp_state["up_proj.weight"]), dim=0),
                dtype=optimization_policy.mlp_gate_up_weight_dtype,
                dram_sharded=True,
            )
        else:
            raise ValueError(f"unsupported MLP topology {optimization_policy.mlp_gate_up_topology!r}")
        decode_down_proj = _load_weight(
            mesh_device,
            mlp_state["down_proj.weight"],
            dtype=optimization_policy.mlp_down_weight_dtype,
            dram_sharded=True,
        )
        old_mlp.gate_proj.deallocate(True)
        old_mlp.up_proj.deallocate(True)
        old_mlp.down_proj.deallocate(True)
        decoder.layer.shared_mlp = _OptimizedSharedMLP(
            mesh_device=mesh_device,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
            decode_gate_proj=decode_gate_proj,
            decode_up_proj=decode_up_proj,
            decode_down_proj=decode_down_proj,
            decode_packed_gate_up=decode_packed_gate_up,
            policy=optimization_policy,
        )
        decoder.attention_qkv_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=optimization_policy.resolved_attention_qkv_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.attention_o_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=optimization_policy.resolved_attention_o_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.attention_compute = decoder.attention_qkv_compute
        return decoder

    def _decode_attention(
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
        """Fused attention with DRAM-sharded QKV and output projections."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, hidden_states.shape[-1])
        sharded_input = ttnn.to_memory_config(hidden_states, input_mem)
        if self.policy.attention_projection_topology == "packed":
            decode_weights = (self.decode_wqkv,)
        else:
            decode_weights = (self.decode_wq, self.decode_wk, self.decode_wv)
        qkv_parts = []
        for decode_weight in decode_weights:
            qkv_n = decode_weight.shape[-1]
            qkv_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, qkv_n)
            qkv_program = self._decode_matmul_program_config(
                k=hidden_states.shape[-1],
                n=qkv_n,
                num_cores=self.policy.decode_num_cores,
                in0_block_w=self.policy.qkv_in0_block_w,
            )
            qkv_sharded = ttnn.linear(
                sharded_input,
                decode_weight,
                memory_config=qkv_mem,
                program_config=qkv_program,
                compute_kernel_config=self.attention_qkv_compute,
                dtype=self.qkv_split_input_dtype,
            )
            qkv_parts.append(ttnn.sharded_to_interleaved(qkv_sharded, ttnn.L1_MEMORY_CONFIG))
            qkv_sharded.deallocate(True)
        sharded_input.deallocate(True)
        if len(qkv_parts) == 1:
            qkv = qkv_parts[0]
        else:
            qkv = ttnn.concat(qkv_parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            for part in qkv_parts:
                part.deallocate(True)
        q, k, v = split_qkv_heads_decode(qkv, config, weights.is_global, tp=1, kv_replicated=False)
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

        k = self._prepare_cache_update_input(k)
        v = self._prepare_cache_update_input(v)

        cache_position = current_position_cache if current_position_cache is not None else current_position
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
        if config.cache_position_modulo is None:
            device_grid = self.mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch_size, device_grid.x)
            while batch_size % grid_x:
                grid_x -= 1
            grid_h = batch_size // grid_x
            k_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_h - 1))})
            v_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, grid_h), ttnn.CoreCoord(grid_x - 1, 2 * grid_h - 1))}
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
                k_cache, k, v_cache, v, update_idxs_tensor=cache_position, page_table=page_table
            )
        else:
            k = ttnn.to_memory_config(k, q_sharded_mem)
            v = ttnn.to_memory_config(v, q_sharded_mem)
            update_args = dict(
                update_idxs_tensor=cache_position,
                page_table=page_table,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
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
            sliding_window_size=config.sliding_window if config.is_sliding else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
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
        concatenated = ttnn.experimental.nlp_concat_heads_decode(sdpa_sharded, num_heads=config.num_attention_heads)
        sdpa_sharded.deallocate(True)
        output = ttnn.sharded_to_interleaved(concatenated, ttnn.DRAM_MEMORY_CONFIG)
        concatenated.deallocate(True)

        o_k, o_n = output.shape[-1], self.decode_o_proj.shape[-1]
        o_input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, o_k)
        o_output_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, o_n)
        o_input = ttnn.to_memory_config(output, o_input_mem)
        output.deallocate(True)
        o_program = self._decode_matmul_program_config(
            k=o_k,
            n=o_n,
            num_cores=self.policy.decode_num_cores,
            in0_block_w=self.policy.o_proj_in0_block_w,
        )
        projected_sharded = ttnn.linear(
            o_input,
            self.decode_o_proj,
            memory_config=o_output_mem,
            program_config=o_program,
            compute_kernel_config=self.attention_o_compute,
        )
        o_input.deallocate(True)
        projected = ttnn.sharded_to_interleaved(projected_sharded, ttnn.DRAM_MEMORY_CONFIG)
        projected_sharded.deallocate(True)
        if projected.shape[2] != batch_size:
            padded = projected
            projected = padded[:, :, :batch_size, :]
            padded.deallocate(True)
        return projected

    def _fill_bounded_sliding_cache_exact(
        self,
        k_cache,
        v_cache,
        k,
        v,
        page_table,
        *,
        user_id,
        valid_seq_len,
        block_size,
        num_kv_heads,
        cache_position_modulo,
    ):
        """Use BFP8 bulk fills and BF16 token updates for a logical tail."""
        bulk_len = (valid_seq_len // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if bulk_len:
            k_bulk = ttnn.slice(k, [0, 0, 0, 0], [1, k.shape[1], bulk_len, k.shape[3]])
            v_bulk = ttnn.slice(v, [0, 0, 0, 0], [1, v.shape[1], bulk_len, v.shape[3]])
            k_bulk_fill = ttnn.typecast(k_bulk, k_cache.dtype) if k_bulk.dtype != k_cache.dtype else k_bulk
            v_bulk_fill = ttnn.typecast(v_bulk, v_cache.dtype) if v_bulk.dtype != v_cache.dtype else v_bulk
            ttnn.experimental.paged_fill_cache(
                k_cache,
                k_bulk_fill,
                page_table,
                batch_idx=user_id,
                block_size=block_size,
                cache_position_modulo=cache_position_modulo,
            )
            ttnn.experimental.paged_fill_cache(
                v_cache,
                v_bulk_fill,
                page_table,
                batch_idx=user_id,
                block_size=block_size,
                cache_position_modulo=cache_position_modulo,
            )
            if k_bulk_fill is not k_bulk:
                k_bulk_fill.deallocate(True)
            if v_bulk_fill is not v_bulk:
                v_bulk_fill.deallocate(True)
            k_bulk.deallocate(True)
            v_bulk.deallocate(True)
        tail_len = valid_seq_len - bulk_len
        if not tail_len:
            return
        k_tail = ttnn.slice(k, [0, 0, bulk_len, 0], [1, k.shape[1], valid_seq_len, k.shape[3]])
        v_tail = ttnn.slice(v, [0, 0, bulk_len, 0], [1, v.shape[1], valid_seq_len, v.shape[3]])
        k_tail_users = self._prepare_cache_update_input(
            ttnn.permute(k_tail, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        v_tail_users = self._prepare_cache_update_input(
            ttnn.permute(v_tail, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        single_token_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(one_core, [ttnn.TILE_SIZE, k.shape[3]], ttnn.ShardOrientation.ROW_MAJOR),
        )
        positions = ttnn.arange(
            bulk_len, valid_seq_len, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )
        user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
        for tail_idx in range(tail_len):
            k_token = ttnn.slice(k_tail_users, [0, tail_idx, 0, 0], [1, tail_idx + 1, num_kv_heads, k.shape[3]])
            v_token = ttnn.slice(v_tail_users, [0, tail_idx, 0, 0], [1, tail_idx + 1, num_kv_heads, v.shape[3]])
            k_token_sharded = ttnn.to_memory_config(k_token, single_token_mem)
            v_token_sharded = ttnn.to_memory_config(v_token, single_token_mem)
            position = ttnn.slice(positions, [tail_idx], [tail_idx + 1])
            update_args = dict(
                update_idxs_tensor=position,
                page_table=user_page_table,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                cache_position_modulo=cache_position_modulo,
            )
            ttnn.experimental.paged_update_cache(k_cache, k_token_sharded, **update_args)
            ttnn.experimental.paged_update_cache(v_cache, v_token_sharded, **update_args)
            for tensor in (k_token, v_token, k_token_sharded, v_token_sharded, position):
                tensor.deallocate(True)
        # A sliced page-table view aliases the caller-owned device buffer; do
        # not deallocate it here because later decode calls reuse the parent.
        for tensor in (k_tail, v_tail, k_tail_users, v_tail_users, positions):
            tensor.deallocate(True)

    def _prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id, valid_seq_len):
        """Fused prefill with explicit cache-dtype casts for BFP8 KV storage."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        if not config.is_sliding and hidden_states.shape[-2] * config.num_attention_heads * config.head_dim >= 2**32:
            return self._streaming_full_prefill_attention(
                hidden_states, rope_mats=rope_mats, page_table=page_table, kv_cache=kv_cache, user_id=user_id
            )
        if self.policy.prefill_qkv_topology == "dram_sharded_m32_chunks":
            if self.policy.attention_projection_topology != "packed":
                raise ValueError("DRAM-sharded prefill candidate requires packed QKV weights")
            qkv_chunks = []
            for start in range(0, hidden_states.shape[-2], ttnn.TILE_SIZE):
                end = min(start + ttnn.TILE_SIZE, hidden_states.shape[-2])
                chunk = ttnn.slice(
                    hidden_states,
                    [0, 0, start, 0],
                    [1, 1, end, hidden_states.shape[-1]],
                )
                input_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, chunk.shape[-1])
                sharded = ttnn.to_memory_config(chunk, input_mem)
                chunk.deallocate(True)
                qkv_n = self.decode_wqkv.shape[-1]
                output_mem = self._decode_memory_config(self.mesh_device, self.policy.decode_num_cores, qkv_n)
                program = self._decode_matmul_program_config(
                    k=sharded.shape[-1],
                    n=qkv_n,
                    num_cores=self.policy.decode_num_cores,
                    in0_block_w=self.policy.qkv_in0_block_w,
                )
                projected = ttnn.linear(
                    sharded,
                    self.decode_wqkv,
                    memory_config=output_mem,
                    program_config=program,
                    compute_kernel_config=self.attention_compute,
                )
                sharded.deallocate(True)
                qkv_chunks.append(ttnn.sharded_to_interleaved(projected, ttnn.DRAM_MEMORY_CONFIG))
                projected.deallocate(True)
            qkv = ttnn.concat(qkv_chunks, dim=2)
            for chunk in qkv_chunks:
                chunk.deallocate(True)
        else:
            qkv = apply_qkv_projection(hidden_states, weights)
        q, k, v = split_qkv_heads_prefill(qkv, config, weights.is_global, tp=1, kv_replicated=False)
        qkv.deallocate(True)
        q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
        k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
        cos_cache, sin_cache = rope_mats
        q = apply_rope(q, cos_cache, sin_cache)
        k = apply_rope(k, cos_cache, sin_cache)

        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
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
                num_kv_heads=config.num_key_value_heads,
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
        if seq_len > 4096 and config.is_sliding:
            sdpa = chunked_prefill_sdpa_sliding(q, k, v, config.sliding_window, config.head_dim, scale=1.0)
        elif seq_len > 4096:
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
        concatenated = self._concatenate_heads(sdpa, num_heads=config.num_attention_heads, head_dim=config.head_dim)
        sdpa.deallocate(True)
        output = ttnn.linear(concatenated, weights.o_proj)
        concatenated.deallocate(True)
        return output

    def _streaming_full_prefill_attention(self, hidden_states, *, rope_mats, page_table, kv_cache, user_id):
        """Advertised-context full attention with BFP8 paged-cache fills."""
        attention = self.layer.self_attn
        config, weights = attention.config, attention.weights
        k_cache, v_cache = kv_cache
        block_size = effective_block_size(k_cache, config.head_dim, config.num_key_value_heads)
        if FULL_ATTN_Q_CHUNK % block_size:
            raise ValueError("full-attention stream chunk must be page-block aligned")
        cos_cache, sin_cache = rope_mats
        seq_len = hidden_states.shape[-2]
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        projected_outputs = []
        for start in range(0, seq_len, FULL_ATTN_Q_CHUNK):
            end = min(start + FULL_ATTN_Q_CHUNK, seq_len)
            hidden_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, hidden_states.shape[-1]])
            qkv = apply_qkv_projection(hidden_chunk, weights)
            hidden_chunk.deallocate(True)
            q, k, v = split_qkv_heads_prefill(qkv, config, weights.is_global, tp=1, kv_replicated=False)
            qkv.deallocate(True)
            q = apply_per_head_norm(q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
            k = apply_per_head_norm(k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
            v = apply_per_head_norm(v, None, config.rms_norm_eps, with_scale=False)
            cos_chunk = ttnn.slice(cos_cache, [0, 0, start, 0], [1, 1, end, cos_cache.shape[-1]])
            sin_chunk = ttnn.slice(sin_cache, [0, 0, start, 0], [1, 1, end, sin_cache.shape[-1]])
            q = apply_rope(q, cos_chunk, sin_chunk)
            k = apply_rope(k, cos_chunk, sin_chunk)
            cos_chunk.deallocate(True)
            sin_chunk.deallocate(True)
            first_block, last_block = start // block_size, end // block_size
            page_chunk = ttnn.slice(page_table, [user_id, first_block], [user_id + 1, last_block])
            k_fill = ttnn.typecast(k, k_cache.dtype) if k.dtype != k_cache.dtype else k
            v_fill = ttnn.typecast(v, v_cache.dtype) if v.dtype != v_cache.dtype else v
            ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_chunk, batch_idx=0, block_size=block_size)
            ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_chunk, batch_idx=0, block_size=block_size)
            if k_fill is not k:
                k_fill.deallocate(True)
            if v_fill is not v:
                v_fill.deallocate(True)
            page_chunk.deallocate(True)
            k.deallocate(True)
            v.deallocate(True)
            sdpa = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, k_cache, v_cache, page_table, chunk_start_idx=start, scale=1.0, program_config=program_config
            )
            q.deallocate(True)
            concatenated = self._concatenate_heads(sdpa, num_heads=config.num_attention_heads, head_dim=config.head_dim)
            sdpa.deallocate(True)
            projected_outputs.append(ttnn.linear(concatenated, weights.o_proj))
            concatenated.deallocate(True)
        result = ttnn.concat(projected_outputs, dim=2)
        for output in projected_outputs:
            output.deallocate(True)
        return result

    def _forward_device(self, hidden_states, *, is_decode, **kwargs):
        # A tile-height tensor can mean either logical seq-32 prefill or padded
        # decode.  Carry the public phase explicitly into the shared MLP so the
        # phase-specific weight/program contract never guesses from shape.
        self.layer.shared_mlp.is_decode = is_decode
        return FusedDecoder._forward_device(self, hidden_states, is_decode=is_decode, **kwargs)


__all__ = [
    "DEFAULT_OPTIMIZATION_POLICY",
    "DecoderOptimizationPolicy",
    "OptimizedDecoder",
    "_OptimizedSharedMLP",
]
