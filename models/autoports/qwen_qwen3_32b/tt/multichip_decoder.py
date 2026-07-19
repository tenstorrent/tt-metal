# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Four-chip tensor-parallel Qwen3-32B decoder layer.

The implementation keeps the optimized decoder's packed-QKV, composite
attention, precision, and per-device sharding choices, then applies the 1x4 TP
mapping preserved in the compiler provenance.  The public layer boundary is a
hidden-width fracture: every device owns 1280 of the 5120 hidden channels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_CACHE_LENGTH,
    HF_MODEL,
    _config_value,
    _state_tensor,
)
from models.autoports.qwen_qwen3_32b.tt.optimized_decoder import (
    DecodePositionBuffers,
    OptimizedDecoder,
    PRECISION_POLICIES,
    _advisor_matmul_program_config,
    _compute_config,
    _core_grid_for_tiles,
    _dram_matmul_program_config,
    _dram_sharded_memory_config,
    _largest_divisor,
    _matmul_output_memory_config,
    _padded_width_sharded_memory_config,
    _prefill_matmul_program_config,
    _width_sharded_memory_config,
)
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_num_links,
    get_tt_ccl,
)


TARGET_MESH_SHAPE = (1, 4)
TP_DEGREE = 4
TP_AXIS = 1
PAGE_BLOCK_SIZE = 64

# Keep the isolated KV-cache control local to the multichip precision ledger;
# weight dtypes and math fidelities match the selected all-BFP4 policy.
PRECISION_POLICIES = {
    **PRECISION_POLICIES,
    "all_bfp4_lofi_kv_bf16": {
        **PRECISION_POLICIES["all_bfp4_lofi"],
        "kv_cache": ttnn.bfloat16,
    },
}


def _replicated_tensor(
    host: torch.Tensor,
    *,
    mesh_device,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        host.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tp_tensor(
    host: torch.Tensor,
    *,
    shard_dim: int,
    mesh_device,
    dtype,
    memory_config,
):
    return ttnn.from_torch(
        host.detach().to(torch.bfloat16).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
    )


def _rank_packed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack Q/K/V per rank before concatenating the four device shards."""

    q_chunks = q.T.chunk(TP_DEGREE, dim=-1)
    k_chunks = k.T.chunk(TP_DEGREE, dim=-1)
    v_chunks = v.T.chunk(TP_DEGREE, dim=-1)
    return torch.cat(
        [torch.cat((q_chunks[rank], k_chunks[rank], v_chunks[rank]), dim=-1) for rank in range(TP_DEGREE)],
        dim=-1,
    )


def _rank_packed_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pack each rank's gate/up columns adjacently before mesh sharding."""

    gate_chunks = gate.T.chunk(TP_DEGREE, dim=-1)
    up_chunks = up.T.chunk(TP_DEGREE, dim=-1)
    return torch.cat(
        [torch.cat((gate_chunks[rank], up_chunks[rank]), dim=-1) for rank in range(TP_DEGREE)],
        dim=-1,
    )


class MultichipDecoder(OptimizedDecoder):
    """Qwen3-32B decoder fixed to the full four-chip Blackhole mesh."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        precision_policy: str = "all_bfp4_lofi",
        decode_target_cores: int = 20,
        decode_down_target_cores: int = 20,
        decode_qkv_target_cores: int | None = None,
        decode_o_target_cores: int | None = None,
        decode_gate_target_cores: int | None = None,
        decode_packed_target_cores: int | None = None,
        decode_qkv_in0_block_w_limit: int | None = None,
        decode_o_in0_block_w_limit: int | None = None,
        decode_gate_in0_block_w_limit: int | None = None,
        decode_packed_in0_block_w_limit: int | None = None,
        decode_down_in0_block_w_limit: int | None = None,
        prefill_in0_block_w: int = 10,
        page_block_size: int = PAGE_BLOCK_SIZE,
        ccl_payload_dtype=ttnn.bfloat16,
        use_persistent_decode_collectives: bool = True,
        use_fused_decode_reduce_scatter: bool = False,
        use_packed_mlp: bool = True,
        packed_mlp_split_layout: str = "dram",
        keep_decode_output_sharded: bool = True,
        use_distributed_decode_norm: bool = False,
        use_fused_decode_all_gather_matmul: bool = False,
        decode_matmul_mode: str = "dram_sharded",
        prefill_matmul_input_l1: bool = False,
        residual_contract: str = "sharded",
    ) -> "MultichipDecoder":
        if precision_policy not in PRECISION_POLICIES:
            raise ValueError(f"Unknown precision policy {precision_policy!r}")
        mesh_shape = tuple(int(value) for value in mesh_device.shape)
        if mesh_shape != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != TP_DEGREE:
            raise ValueError(
                f"MultichipDecoder is fixed to the full 1x4 mesh, got shape={mesh_shape}, "
                f"devices={mesh_device.get_num_devices()}"
            )
        if batch != EMITTED_BATCH:
            raise ValueError(f"The compiler-derived public contract requires batch={EMITTED_BATCH}, got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")
        if page_block_size < 32 or page_block_size % 32:
            raise ValueError("page_block_size must be a positive multiple of 32")
        if ccl_payload_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError("ccl_payload_dtype must be bfloat16 or bfloat8_b")
        if residual_contract not in ("sharded", "replicated_provenance"):
            raise ValueError("residual_contract must be sharded or replicated_provenance")
        if packed_mlp_split_layout not in ("dram", "l1"):
            raise ValueError("packed_mlp_split_layout must be dram or l1")
        if decode_matmul_mode not in ("dram_sharded", "shard_advisor"):
            raise ValueError("decode_matmul_mode must be dram_sharded or shard_advisor")
        if use_fused_decode_all_gather_matmul and not use_distributed_decode_norm:
            raise ValueError("fused decode all-gather matmul requires distributed decode RMSNorm")
        if use_fused_decode_all_gather_matmul and not use_packed_mlp:
            raise ValueError("fused decode all-gather matmul requires packed gate/up")
        if residual_contract == "replicated_provenance" and use_persistent_decode_collectives:
            raise ValueError("the measured replicated-provenance candidate uses non-persistent all-reduces")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        advertised_context = int(_config_value(hf_config, "max_position_embeddings"))
        attention_width = num_heads * head_dim
        expected = (hidden_size, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size)
        if expected != (5120, 64, 64, 8, 128, 25600):
            raise ValueError(f"{HF_MODEL} config does not match the translated IR: {expected}")
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx must be in [0, {num_layers}), got {layer_idx}")
        if max_cache_len > advertised_context:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds max_position_embeddings={advertised_context}")
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError("The translated IR requires hidden_act='silu'")
        if num_heads % TP_DEGREE or num_kv_heads % TP_DEGREE or hidden_size % TP_DEGREE:
            raise ValueError("heads, KV heads, and hidden size must divide TP=4")

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        q_norm = _state_tensor(state_dict, layer_idx, "self_attn.q_norm.weight").to(torch.bfloat16)
        k_norm = _state_tensor(state_dict, layer_idx, "self_attn.k_norm.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(
            state_dict, layer_idx, "post_attention_layernorm.weight"
        ).to(torch.bfloat16)

        shape_contract = {
            "q": (q, (attention_width, hidden_size)),
            "k": (k, (num_kv_heads * head_dim, hidden_size)),
            "v": (v, (num_kv_heads * head_dim, hidden_size)),
            "o": (o, (hidden_size, attention_width)),
            "gate": (gate, (intermediate_size, hidden_size)),
            "up": (up, (intermediate_size, hidden_size)),
            "down": (down, (hidden_size, intermediate_size)),
        }
        for name, (tensor, shape) in shape_contract.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} weight has shape {tuple(tensor.shape)}, expected {shape}")

        self = object.__new__(cls)
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)
        self.tp_degree = TP_DEGREE
        self.tp_axis = TP_AXIS
        self.local_hidden_size = hidden_size // TP_DEGREE
        self.local_num_heads = num_heads // TP_DEGREE
        self.local_num_kv_heads = num_kv_heads // TP_DEGREE
        self.local_attention_width = attention_width // TP_DEGREE
        self.local_intermediate_size = intermediate_size // TP_DEGREE
        self.page_block_size = page_block_size
        self.precision_policy_name = precision_policy
        self.precision_policy = dict(PRECISION_POLICIES[precision_policy])
        self.kv_cache_dtype = self.precision_policy["kv_cache"]
        self.ccl_payload_dtype = ccl_payload_dtype
        self.use_persistent_decode_collectives = use_persistent_decode_collectives
        self.use_fused_decode_reduce_scatter = use_fused_decode_reduce_scatter
        self.use_packed_mlp = use_packed_mlp
        self.packed_mlp_split_layout = packed_mlp_split_layout
        self.keep_decode_output_sharded = keep_decode_output_sharded
        self.use_distributed_decode_norm = use_distributed_decode_norm
        self.use_fused_decode_all_gather_matmul = use_fused_decode_all_gather_matmul
        self.decode_matmul_mode = decode_matmul_mode
        self.prefill_matmul_input_l1 = prefill_matmul_input_l1
        self.residual_contract = residual_contract
        self.decode_target_cores = decode_target_cores
        self.decode_down_target_cores = decode_down_target_cores
        self.decode_qkv_target_cores = decode_target_cores if decode_qkv_target_cores is None else decode_qkv_target_cores
        # The O projection is small enough that spreading it over the default
        # 16-core divisor costs more multicast/synchronization than it saves.
        # Three paired 200-replay trials selected the 8-core program.
        self.decode_o_target_cores = 8 if decode_o_target_cores is None else decode_o_target_cores
        self.decode_gate_target_cores = (
            decode_target_cores if decode_gate_target_cores is None else decode_gate_target_cores
        )
        self.decode_packed_target_cores = (
            10 if decode_packed_target_cores is None else decode_packed_target_cores
        )
        self.decode_qkv_in0_block_w_limit = decode_qkv_in0_block_w_limit
        self.decode_o_in0_block_w_limit = decode_o_in0_block_w_limit
        self.decode_gate_in0_block_w_limit = decode_gate_in0_block_w_limit
        self.decode_packed_in0_block_w_limit = (
            8 if decode_packed_in0_block_w_limit is None else decode_packed_in0_block_w_limit
        )
        self.decode_down_in0_block_w_limit = decode_down_in0_block_w_limit
        self.prefill_in0_block_w = prefill_in0_block_w
        self.topology = ttnn.Topology.Ring
        self.num_links = get_num_links(mesh_device)
        self.tt_ccl = get_tt_ccl(mesh_device)
        self._eager_position_buffers = None

        self.input_norm = _replicated_tensor(input_norm, mesh_device=mesh_device, dtype=ttnn.bfloat16)
        self.post_attention_norm = _replicated_tensor(
            post_attention_norm, mesh_device=mesh_device, dtype=ttnn.bfloat16
        )
        self.input_norm_distributed = self.post_attention_norm_distributed = None
        if use_distributed_decode_norm:
            self.input_norm_distributed = ttnn.from_torch(
                input_norm.reshape(1, 1, hidden_size // 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
            )
            self.post_attention_norm_distributed = ttnn.from_torch(
                post_attention_norm.reshape(1, 1, hidden_size // 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
            )
        self.q_norm = _replicated_tensor(q_norm, mesh_device=mesh_device, dtype=ttnn.bfloat16)
        self.k_norm = _replicated_tensor(k_norm, mesh_device=mesh_device, dtype=ttnn.bfloat16)

        rotary = Qwen3RotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        self.rotary_cos = _replicated_tensor(
            cos.to(torch.bfloat16).unsqueeze(1), mesh_device=mesh_device, dtype=ttnn.bfloat16
        )
        self.rotary_sin = _replicated_tensor(
            sin.to(torch.bfloat16).unsqueeze(1), mesh_device=mesh_device, dtype=ttnn.bfloat16
        )
        self.rotary_cos_row_major = ttnn.to_layout(self.rotary_cos, ttnn.ROW_MAJOR_LAYOUT)
        self.rotary_sin_row_major = ttnn.to_layout(self.rotary_sin, ttnn.ROW_MAJOR_LAYOUT)

        self.attention_compute_config = _compute_config(
            mesh_device, self.precision_policy["attention_fidelity"]
        )
        self.mlp_compute_config = _compute_config(mesh_device, self.precision_policy["mlp_fidelity"])
        self.norm_compute_config = _compute_config(mesh_device, ttnn.MathFidelity.HiFi2)
        self._build_multichip_configs()
        self._decode_ag_buffer_index = 0
        self._decode_rs_buffer_index = 0
        self._decode_ag_persistent_buffers = []
        self._decode_rs_persistent_buffers = []
        self._decode_fused_rs_persistent_buffers = []
        if use_persistent_decode_collectives:
            for _ in range(2):
                self._decode_ag_persistent_buffers.append(
                    _replicated_tensor(
                        torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=ccl_payload_dtype,
                    )
                )
                if not use_fused_decode_reduce_scatter:
                    rs_intermediate = _replicated_tensor(
                        torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=ccl_payload_dtype,
                    )
                    rs_output = _replicated_tensor(
                        torch.zeros((1, 1, batch, self.local_hidden_size), dtype=torch.bfloat16),
                        mesh_device=mesh_device,
                        dtype=ccl_payload_dtype,
                        memory_config=self.local_residual_memory_config,
                    )
                    self._decode_rs_persistent_buffers.append([rs_intermediate, rs_output])
        if use_fused_decode_reduce_scatter:
            if ccl_payload_dtype != ttnn.bfloat16:
                raise ValueError("fused decode reduce-scatter currently requires BF16 CCL payloads")
            for _ in range(2):
                self._decode_fused_rs_persistent_buffers.append(
                    [
                        _replicated_tensor(
                            torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
                            mesh_device=mesh_device,
                            dtype=ttnn.bfloat16,
                        ),
                        _replicated_tensor(
                            torch.zeros((1, 1, batch, self.local_hidden_size), dtype=torch.bfloat16),
                            mesh_device=mesh_device,
                            dtype=ttnn.bfloat16,
                        ),
                    ]
                )

        qkv_host = _rank_packed_qkv(q, k, v)
        self.qkv_weight = _tp_tensor(
            qkv_host,
            shard_dim=-1,
            mesh_device=mesh_device,
            dtype=self.precision_policy["attention"],
            memory_config=_dram_sharded_memory_config(mesh_device, hidden_size, self.local_qkv_size),
        )
        self.output_weight = _tp_tensor(
            o.T,
            shard_dim=-2,
            mesh_device=mesh_device,
            dtype=self.precision_policy["attention"],
            memory_config=_dram_sharded_memory_config(
                mesh_device, self.local_attention_width, hidden_size
            ),
        )
        if use_packed_mlp:
            gate_up_host = _rank_packed_gate_up(gate, up)
            self.gate_up_weight = _tp_tensor(
                gate_up_host,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=_dram_sharded_memory_config(
                    mesh_device, hidden_size, 2 * self.local_intermediate_size
                ),
            )
            self.gate_weight = self.up_weight = None
        else:
            self.gate_up_weight = None
            self.gate_weight = _tp_tensor(
                gate.T,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=_dram_sharded_memory_config(
                    mesh_device, hidden_size, self.local_intermediate_size
                ),
            )
            self.up_weight = _tp_tensor(
                up.T,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=_dram_sharded_memory_config(
                    mesh_device, hidden_size, self.local_intermediate_size
                ),
            )
        self.down_weight = _tp_tensor(
            down.T,
            shard_dim=-2,
            mesh_device=mesh_device,
            dtype=self.precision_policy["mlp_down"],
            memory_config=_dram_sharded_memory_config(
                mesh_device, self.local_intermediate_size, hidden_size
            ),
        )
        # The optimized decode kernels consume DRAM-width-sharded weights, while
        # the sequence-parallel prefill kernels require interleaved weights.
        # Keeping both formats avoids an every-call device reshard.  The exact
        # full-stack context consequence is measured by the capacity gate.
        self.qkv_prefill_weight = _tp_tensor(
            qkv_host,
            shard_dim=-1,
            mesh_device=mesh_device,
            dtype=self.precision_policy["attention"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.output_prefill_weight = _tp_tensor(
            o.T,
            shard_dim=-2,
            mesh_device=mesh_device,
            dtype=self.precision_policy["attention"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if use_packed_mlp:
            self.gate_up_prefill_weight = _tp_tensor(
                gate_up_host,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.gate_prefill_weight = self.up_prefill_weight = None
        else:
            self.gate_up_prefill_weight = None
            self.gate_prefill_weight = _tp_tensor(
                gate.T,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.up_prefill_weight = _tp_tensor(
                up.T,
                shard_dim=-1,
                mesh_device=mesh_device,
                dtype=self.precision_policy["mlp_gate_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.down_prefill_weight = _tp_tensor(
            down.T,
            shard_dim=-2,
            mesh_device=mesh_device,
            dtype=self.precision_policy["mlp_down"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self

    @property
    def local_qkv_size(self) -> int:
        return (self.local_num_heads + 2 * self.local_num_kv_heads) * self.head_dim

    def _build_multichip_configs(self) -> None:
        padded_rows = 32 * math.ceil(self.batch / 32)
        hidden_tiles = self.hidden_size // 32
        local_hidden_tiles = self.local_hidden_size // 32
        local_attention_tiles = self.local_attention_width // 32
        local_qkv_tiles = self.local_qkv_size // 32
        local_mlp_tiles = self.local_intermediate_size // 32

        self.local_residual_grid = _core_grid_for_tiles(
            local_hidden_tiles,
            local_hidden_tiles,
            target_cores=self.decode_target_cores,
            device=self.mesh_device,
        )
        self.full_residual_grid = _core_grid_for_tiles(
            hidden_tiles,
            hidden_tiles,
            target_cores=self.decode_target_cores,
            device=self.mesh_device,
        )
        self.qkv_grid = _core_grid_for_tiles(
            hidden_tiles,
            local_qkv_tiles,
            target_cores=self.decode_qkv_target_cores,
            device=self.mesh_device,
        )
        self.o_grid = _core_grid_for_tiles(
            local_attention_tiles,
            hidden_tiles,
            target_cores=self.decode_o_target_cores,
            device=self.mesh_device,
        )
        self.mlp_gate_grid = _core_grid_for_tiles(
            hidden_tiles,
            local_mlp_tiles,
            target_cores=self.decode_gate_target_cores,
            device=self.mesh_device,
        )
        self.packed_mlp_grid = _core_grid_for_tiles(
            hidden_tiles,
            2 * local_mlp_tiles,
            target_cores=self.decode_packed_target_cores,
            device=self.mesh_device,
        )
        self.mlp_down_grid = _core_grid_for_tiles(
            local_mlp_tiles,
            hidden_tiles,
            target_cores=self.decode_down_target_cores,
            device=self.mesh_device,
        )

        self.local_residual_memory_config = _width_sharded_memory_config(
            padded_rows, self.local_hidden_size, self.local_residual_grid
        )
        self.full_residual_memory_config = _width_sharded_memory_config(
            padded_rows, self.hidden_size, self.full_residual_grid
        )
        self.qkv_input_memory_config = _width_sharded_memory_config(
            padded_rows, self.hidden_size, self.qkv_grid
        )
        self.qkv_output_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_qkv_size, self.qkv_grid, self.mesh_device
        )
        self.o_input_memory_config = _width_sharded_memory_config(
            padded_rows, self.local_attention_width, self.o_grid
        )
        self.o_partial_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.o_grid, self.mesh_device
        )
        self.mlp_gate_input_memory_config = _width_sharded_memory_config(
            padded_rows, self.hidden_size, self.mlp_gate_grid
        )
        self.mlp_gate_memory_config = _matmul_output_memory_config(
            padded_rows, self.local_intermediate_size, self.mlp_gate_grid, self.mesh_device
        )
        self.packed_mlp_memory_config = _matmul_output_memory_config(
            padded_rows, 2 * self.local_intermediate_size, self.packed_mlp_grid, self.mesh_device
        )
        self.mlp_down_input_memory_config = _width_sharded_memory_config(
            padded_rows, self.local_intermediate_size, self.mlp_down_grid
        )
        self.mlp_down_partial_memory_config = _matmul_output_memory_config(
            padded_rows, self.hidden_size, self.mlp_down_grid, self.mesh_device
        )

        norm_block_w = hidden_tiles // self.full_residual_grid.num_cores
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.full_residual_grid.x, self.full_residual_grid.y],
            subblock_w=_largest_divisor(norm_block_w, limit=4),
            block_h=padded_rows // 32,
            block_w=norm_block_w,
            inplace=False,
        )
        local_norm_block_w = self.local_hidden_size // 32 // self.local_residual_grid.num_cores
        self.distributed_norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[self.local_residual_grid.x, self.local_residual_grid.y],
            subblock_w=_largest_divisor(local_norm_block_w, limit=4),
            block_h=padded_rows // 32,
            block_w=local_norm_block_w,
            inplace=False,
        )
        self.distributed_stats_memory_config = ttnn.create_sharded_memory_config(
            shape=[1, 1, 32, 32 * self.tp_degree],
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
        )
        self.fused_qkv_agmm_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=20,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=self.local_qkv_size // 32 // 8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        self.fused_packed_agmm_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=20,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2 * self.local_intermediate_size // 32 // 8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        self.qkv_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            self.local_qkv_size,
            self.qkv_grid,
            in0_block_w_limit=self.decode_qkv_in0_block_w_limit,
        )
        self.o_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.local_attention_width,
            self.hidden_size,
            self.o_grid,
            in0_block_w_limit=self.decode_o_in0_block_w_limit,
        )
        self.gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            self.local_intermediate_size,
            self.mlp_gate_grid,
            in0_block_w_limit=self.decode_gate_in0_block_w_limit,
        )
        self.packed_gate_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.hidden_size,
            2 * self.local_intermediate_size,
            self.packed_mlp_grid,
            in0_block_w_limit=self.decode_packed_in0_block_w_limit,
        )
        self.down_decode_program_config = _dram_matmul_program_config(
            padded_rows,
            self.local_intermediate_size,
            self.hidden_size,
            self.mlp_down_grid,
            in0_block_w_limit=self.decode_down_in0_block_w_limit,
        )
        # Exact TP=4 rank-local layouts/programs emitted by the fresh
        # shard-advisor captures under doc/optimized_multichip_decoder.
        self.advisor_packed_input_memory_config = _padded_width_sharded_memory_config(
            padded_rows, self.hidden_size, 80, self.mesh_device
        )
        self.advisor_o_input_memory_config = _padded_width_sharded_memory_config(
            padded_rows, self.local_attention_width, 32, self.mesh_device
        )
        self.advisor_down_input_memory_config = _padded_width_sharded_memory_config(
            padded_rows, self.local_intermediate_size, 50, self.mesh_device
        )
        self.advisor_qkv_output_memory_config = _padded_width_sharded_memory_config(
            padded_rows, self.local_qkv_size, 80, self.mesh_device
        )
        self.advisor_packed_output_memory_config = _padded_width_sharded_memory_config(
            padded_rows, 2 * self.local_intermediate_size, 100, self.mesh_device
        )
        self.advisor_row_output_memory_config = _padded_width_sharded_memory_config(
            padded_rows, self.hidden_size, 80, self.mesh_device
        )
        self.advisor_qkv_program_config = _advisor_matmul_program_config(
            grid=(11, 8), in0_block_w=8, per_core_n=1, out_subblock_w=1
        )
        self.advisor_packed_program_config = _advisor_matmul_program_config(
            grid=(11, 10), in0_block_w=2, per_core_n=4, out_subblock_w=4
        )
        self.advisor_o_program_config = _advisor_matmul_program_config(
            grid=(11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2
        )
        self.advisor_down_program_config = _advisor_matmul_program_config(
            grid=(11, 8), in0_block_w=2, per_core_n=2, out_subblock_w=2
        )
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        if int(device_grid.x) < 8 or int(device_grid.y) < 4:
            raise ValueError(f"The fixed batch-32 decode grid requires at least 8x4 cores, got {device_grid}")
        batch_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}
        )
        # Let create_qkv_heads choose its output grid from the logical local
        # head count.  An explicit 16-core grid is too small for its batch-32
        # work split, while forcing 32 cores pads the logical head dimension
        # and breaks concat_heads' num_heads=16 contract.
        self.decode_heads_mem_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        self.decode_kv_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(batch_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_concat_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(batch_grid, [32, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def mesh_plan_summary(self) -> dict:
        def grid(grid) -> dict:
            return {"x": int(grid.x), "y": int(grid.y), "cores": int(grid.num_cores)}

        def matmul_role(*, k: int, n: int, role_grid, in0_limit: int | None, fidelity) -> dict:
            cores = int(role_grid.num_cores)
            k_tiles_per_core = k // 32 // cores
            in0_block_w = _largest_divisor(k_tiles_per_core, limit=in0_limit)
            dram_banks = 8
            return {
                "grid": grid(role_grid),
                "activation_memory": "L1 WIDTH_SHARDED",
                "activation_shard_shape": [32, k // cores],
                "weight_memory": "DRAM WIDTH_SHARDED",
                "weight_shard_shape": [k, n // dram_banks],
                "output_memory": "L1 WIDTH_SHARDED",
                "output_shard_shape": [32, n // cores],
                "input_logical_shape": [1, 1, self.batch, k],
                "weight_logical_shape": [k, n],
                "output_logical_shape": [1, 1, self.batch, n],
                "padding": {"m_rows": 0, "k_channels": 0, "n_channels": 0},
                "program_config": {
                    "type": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                    "in0_block_w": in0_block_w,
                    "per_core_M": 1,
                    "per_core_N": n // 32 // cores,
                    "output_subblock": "not exposed by DRAM-sharded program config",
                },
                "math_fidelity": str(fidelity),
            }

        decode_local_bytes = self.batch * self.local_hidden_size * 2
        decode_full_bytes = self.batch * self.hidden_size * 2
        ring_bytes = decode_full_bytes - decode_local_bytes

        return {
            "baseline": "OptimizedDecoder/all_bfp4_lofi",
            "mesh_shape": list(TARGET_MESH_SHAPE),
            "tp_degree": self.tp_degree,
            "tp_axis": self.tp_axis,
            "topology": "Ring",
            "num_links": self.num_links,
            "residual_contract": (
                "replicated full-hidden input/output; compiler-provenance measurement candidate"
                if self.residual_contract == "replicated_provenance"
                else "hidden-width-sharded input/output; 1280 channels per rank"
            ),
            "precision_policy": self.precision_policy_name,
            "ccl_payload_dtype": str(self.ccl_payload_dtype),
            "persistent_decode_collectives": self.use_persistent_decode_collectives,
            "fused_decode_reduce_scatter": self.use_fused_decode_reduce_scatter,
            "packed_gate_up": self.use_packed_mlp,
            "packed_gate_up_split_layout": self.packed_mlp_split_layout,
            "decode_output_sharded": self.keep_decode_output_sharded,
            "distributed_decode_norm": self.use_distributed_decode_norm,
            "fused_decode_all_gather_matmul": self.use_fused_decode_all_gather_matmul,
            "decode_matmul_mode": self.decode_matmul_mode,
            "prefill_matmul_input_l1": self.prefill_matmul_input_l1,
            "kv_cache_dtype": str(self.kv_cache_dtype),
            "local_shapes": {
                "residual": self.local_hidden_size,
                "q_heads": self.local_num_heads,
                "kv_heads": self.local_num_kv_heads,
                "qkv": self.local_qkv_size,
                "attention": self.local_attention_width,
                "intermediate": self.local_intermediate_size,
            },
            "decode_grids": {
                "local_residual": grid(self.local_residual_grid),
                "full_residual_norm": grid(self.full_residual_grid),
                "qkv": grid(self.qkv_grid),
                "o": grid(self.o_grid),
                "gate_up": grid(self.mlp_gate_grid),
                "packed_gate_up": grid(self.packed_mlp_grid),
                "down": grid(self.mlp_down_grid),
            },
            "decode_shards_and_programs": {
                "local_residual": {
                    "memory": "L1 WIDTH_SHARDED",
                    "logical_shape": [1, 1, self.batch, self.local_hidden_size],
                    "shard_shape": [32, self.local_hidden_size // self.local_residual_grid.num_cores],
                    "grid": grid(self.local_residual_grid),
                },
                "full_residual_norm": {
                    "memory": "L1 WIDTH_SHARDED",
                    "logical_shape": [1, 1, self.batch, self.hidden_size],
                    "shard_shape": [32, self.hidden_size // self.full_residual_grid.num_cores],
                    "grid": grid(self.full_residual_grid),
                    "program_config": {
                        "block_h": 1,
                        "block_w": self.hidden_size // 32 // self.full_residual_grid.num_cores,
                        "subblock_w": _largest_divisor(
                            self.hidden_size // 32 // self.full_residual_grid.num_cores, limit=4
                        ),
                    },
                },
                "qkv": matmul_role(
                    k=self.hidden_size,
                    n=self.local_qkv_size,
                    role_grid=self.qkv_grid,
                    in0_limit=self.decode_qkv_in0_block_w_limit,
                    fidelity=self.precision_policy["attention_fidelity"],
                ),
                "o": matmul_role(
                    k=self.local_attention_width,
                    n=self.hidden_size,
                    role_grid=self.o_grid,
                    in0_limit=self.decode_o_in0_block_w_limit,
                    fidelity=self.precision_policy["attention_fidelity"],
                ),
                "gate_up": matmul_role(
                    k=self.hidden_size,
                    n=self.local_intermediate_size,
                    role_grid=self.mlp_gate_grid,
                    in0_limit=self.decode_gate_in0_block_w_limit,
                    fidelity=self.precision_policy["mlp_fidelity"],
                ),
                "packed_gate_up": matmul_role(
                    k=self.hidden_size,
                    n=2 * self.local_intermediate_size,
                    role_grid=self.packed_mlp_grid,
                    in0_limit=self.decode_packed_in0_block_w_limit,
                    fidelity=self.precision_policy["mlp_fidelity"],
                ),
                "down": matmul_role(
                    k=self.local_intermediate_size,
                    n=self.hidden_size,
                    role_grid=self.mlp_down_grid,
                    in0_limit=self.decode_down_in0_block_w_limit,
                    fidelity=self.precision_policy["mlp_fidelity"],
                ),
                "decode_heads": {
                    "q": [1, self.batch, 32, self.head_dim],
                    "kv": [1, self.batch, self.local_num_kv_heads, self.head_dim],
                    "physical_q_padding": "16 logical Q heads duplicated to 32 for Blackhole SDPA; selected back to 16",
                    "height_shard_shape": [32, self.head_dim],
                    "height_shard_grid": [8, 4],
                },
                "kv_cache": {
                    "contiguous_per_rank": [self.batch, self.local_num_kv_heads, self.max_cache_len, self.head_dim],
                    "paged_per_rank": [
                        self.batch * math.ceil(self.max_cache_len / self.page_block_size),
                        self.local_num_kv_heads,
                        self.page_block_size,
                        self.head_dim,
                    ],
                    "dtype": str(self.kv_cache_dtype),
                    "memory": "DRAM INTERLEAVED",
                },
            },
            "collectives_per_layer": {
                "all_gather": 2,
                "reduce_scatter": 2,
                "semantic_collectives": (
                    "2 Ring all-reduces, each implemented as reduce-scatter + all-gather"
                    if self.residual_contract == "replicated_provenance"
                    else "2 hidden all-gathers + 2 row-parallel reduce-scatters"
                ),
                "layer_boundary_gather": 0,
                "decode_all_gather_input_bytes_per_rank": decode_local_bytes,
                "decode_all_gather_output_bytes_per_rank": decode_full_bytes,
                "decode_reduce_scatter_input_bytes_per_rank": decode_full_bytes,
                "decode_reduce_scatter_output_bytes_per_rank": decode_local_bytes,
                "estimated_ring_wire_bytes_per_rank_per_collective": ring_bytes,
                "estimated_total_ring_wire_bytes_per_rank_per_layer": 4 * ring_bytes,
            },
        }

    def allocate_kv_cache(
        self,
        max_cache_len: int | None = None,
        *,
        paged: bool = False,
        page_block_size: int | None = None,
        num_blocks: int | None = None,
    ):
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        block_size = self.page_block_size if page_block_size is None else int(page_block_size)
        if paged:
            if block_size < 32 or block_size % 32:
                raise ValueError("page_block_size must be a multiple of 32")
            required_blocks = self.batch * math.ceil(cache_len / block_size)
            blocks = required_blocks if num_blocks is None else int(num_blocks)
            if blocks < required_blocks:
                raise ValueError(f"num_blocks={blocks} is smaller than required {required_blocks}")
            shape = (blocks, self.local_num_kv_heads, block_size, self.head_dim)
        else:
            shape = (self.batch, self.local_num_kv_heads, cache_len, self.head_dim)
        host = torch.zeros(shape, dtype=torch.bfloat16)
        return (
            _replicated_tensor(host, mesh_device=self.mesh_device, dtype=self.kv_cache_dtype),
            _replicated_tensor(host, mesh_device=self.mesh_device, dtype=self.kv_cache_dtype),
        )

    def allocate_page_table(
        self,
        max_cache_len: int | None = None,
        *,
        page_block_size: int | None = None,
        permutation: torch.Tensor | None = None,
    ):
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        block_size = self.page_block_size if page_block_size is None else int(page_block_size)
        blocks_per_user = math.ceil(cache_len / block_size)
        total_blocks = self.batch * blocks_per_user
        if permutation is None:
            physical = torch.arange(total_blocks, dtype=torch.int32)
        else:
            physical = permutation.to(torch.int32).flatten()
            if physical.numel() != total_blocks or sorted(physical.tolist()) != list(range(total_blocks)):
                raise ValueError("permutation must contain each physical block exactly once")
        table = physical.reshape(self.batch, blocks_per_user)
        return _replicated_tensor(
            table,
            mesh_device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _validate_hidden_states(self, hidden_states, expected_seq_len: int | None = None) -> int:
        shape = tuple(hidden_states.shape)
        expected_prefix = (1, self.batch)
        if len(shape) != 4 or shape[:2] != expected_prefix or shape[3] != self.local_hidden_size:
            raise ValueError(
                f"hidden_states must be a TP-fractured [1,{self.batch},seq,{self.local_hidden_size}] "
                f"mesh tensor, got {shape}"
            )
        seq_len = int(shape[2])
        if expected_seq_len is not None and seq_len != expected_seq_len:
            raise ValueError(f"expected seq_len={expected_seq_len}, got {seq_len}")
        if seq_len < 1 or seq_len > self.max_cache_len:
            raise ValueError(f"seq_len must be in [1,{self.max_cache_len}], got {seq_len}")
        return seq_len

    def _cache_mode(self, key_cache, value_cache, page_table) -> str:
        if tuple(key_cache.shape) != tuple(value_cache.shape):
            raise ValueError("key and value cache shapes differ")
        shape = tuple(key_cache.shape)
        contiguous = (self.batch, self.local_num_kv_heads, self.max_cache_len, self.head_dim)
        if shape == contiguous:
            if page_table is not None:
                raise ValueError("page_table was supplied with a contiguous cache")
            return "contiguous"
        if len(shape) != 4 or shape[1] != self.local_num_kv_heads or shape[3] != self.head_dim:
            raise ValueError(f"invalid local KV-cache shape {shape}")
        if page_table is None:
            raise ValueError("paged cache requires a replicated page_table")
        if shape[2] < 32 or shape[2] % 32:
            raise ValueError(f"invalid page block size {shape[2]}")
        if tuple(page_table.shape)[0] != self.batch:
            raise ValueError(f"page_table must have {self.batch} rows, got {tuple(page_table.shape)}")
        return "paged"

    def _all_gather_hidden(self, tensor, *, memory_config):
        if self.ccl_payload_dtype != tensor.dtype:
            tensor = ttnn.typecast(tensor, self.ccl_payload_dtype)
        persistent_output = None
        if self.use_persistent_decode_collectives and tuple(tensor.shape) == (
            1,
            1,
            self.batch,
            self.local_hidden_size,
        ):
            persistent_output = self._decode_ag_persistent_buffers[self._decode_ag_buffer_index]
            self._decode_ag_buffer_index = (self._decode_ag_buffer_index + 1) % 2
        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=persistent_output,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        if gathered.dtype != ttnn.bfloat16:
            gathered = ttnn.typecast(gathered, ttnn.bfloat16)
        return gathered

    def _reduce_scatter_hidden(self, tensor, *, memory_config, decode: bool):
        if tensor.is_sharded():
            tensor = ttnn.sharded_to_interleaved(
                tensor, ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
            )
        if self.ccl_payload_dtype != tensor.dtype:
            tensor = ttnn.typecast(tensor, self.ccl_payload_dtype)
        persistent_outputs = None
        if self.use_persistent_decode_collectives and decode:
            persistent_outputs = self._decode_rs_persistent_buffers[self._decode_rs_buffer_index]
            self._decode_rs_buffer_index = (self._decode_rs_buffer_index + 1) % 2
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=persistent_outputs,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        if reduced.dtype != ttnn.bfloat16:
            reduced = ttnn.typecast(reduced, ttnn.bfloat16)
        return reduced

    def _all_reduce_hidden(self, tensor):
        """Compiler-provenance Ring all-reduce candidate (RS then AG)."""

        if tensor.is_sharded():
            tensor = ttnn.sharded_to_interleaved(tensor, ttnn.L1_MEMORY_CONFIG)
        if tensor.dtype != ttnn.bfloat16:
            tensor = ttnn.typecast(tensor, ttnn.bfloat16)
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        gathered = ttnn.experimental.all_gather_async(
            reduced,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        return ttnn.to_memory_config(gathered, self.full_residual_memory_config)

    def _row_parallel_collective(self, partial, *, decode: bool):
        if self.residual_contract == "replicated_provenance":
            return self._all_reduce_hidden(partial)
        return self._reduce_scatter_hidden(
            partial,
            memory_config=self.local_residual_memory_config,
            decode=decode,
        )

    def _decode_norm(self, residual, weight):
        if self.use_distributed_decode_norm and self.residual_contract == "sharded":
            distributed_weight = (
                self.input_norm_distributed
                if weight is self.input_norm
                else self.post_attention_norm_distributed
            )
            stats = ttnn.rms_norm_pre_all_gather(
                residual,
                compute_kernel_config=self.norm_compute_config,
                program_config=self.distributed_norm_program_config,
                dtype=ttnn.bfloat16,
            )
            gathered_stats = ttnn.experimental.all_gather_async(
                stats,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=self.num_links,
                topology=self.topology,
                memory_config=self.distributed_stats_memory_config,
                chunks_per_sync=CCL_CHUNKS_PER_SYNC,
                num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
                num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
            )
            ttnn.deallocate(stats, True)
            normalized = ttnn.rms_norm_post_all_gather(
                residual,
                gathered_stats,
                epsilon=self.rms_norm_eps,
                weight=distributed_weight,
                compute_kernel_config=self.norm_compute_config,
                program_config=self.distributed_norm_program_config,
                dtype=ttnn.bfloat16,
                memory_config=self.local_residual_memory_config,
            )
            ttnn.deallocate(gathered_stats, True)
            if self.use_fused_decode_all_gather_matmul:
                return normalized
            gathered = self._all_gather_hidden(normalized, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.to_memory_config(gathered, self.full_residual_memory_config)
        if self.residual_contract == "replicated_provenance":
            gathered = residual
        else:
            gathered = self._all_gather_hidden(residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gathered = ttnn.to_memory_config(gathered, self.full_residual_memory_config)
        return ttnn.rms_norm(
            gathered,
            epsilon=self.rms_norm_eps,
            weight=weight,
            program_config=self.norm_program_config,
            compute_kernel_config=self.norm_compute_config,
            memory_config=self.full_residual_memory_config,
        )

    def _fused_decode_column_parallel(self, tensor, weight, *, role: str, compute_kernel_config):
        if role == "qkv":
            program_config = self.fused_qkv_agmm_program_config
        elif role == "packed_gate_up":
            program_config = self.fused_packed_agmm_program_config
        else:
            raise ValueError(f"unsupported fused column-parallel role {role!r}")
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        if len(tuple(weight.shape)) == 2:
            weight = ttnn.reshape(weight, [1, 1, int(weight.shape[-2]), int(weight.shape[-1])])
        persistent_output = None
        if self.use_persistent_decode_collectives:
            persistent_output = self._decode_ag_persistent_buffers[self._decode_ag_buffer_index]
            self._decode_ag_buffer_index = (self._decode_ag_buffer_index + 1) % 2
        _, output = ttnn.experimental.all_gather_matmul_async(
            tensor,
            weight,
            persistent_output_buffer=persistent_output,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            all_gather_core_grid_offset=(0, 6),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            # The experimental AGMM fusion is validated and exercised only
            # with one fabric link; standalone collectives retain two links.
            num_links=1,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        return output

    def _fused_decode_row_parallel(self, tensor, weight, *, role: str, compute_kernel_config):
        if role == "o":
            local_k = self.local_attention_width
            pair = self._decode_fused_rs_persistent_buffers[0]
        elif role == "down":
            local_k = self.local_intermediate_size
            pair = self._decode_fused_rs_persistent_buffers[1]
        else:
            raise ValueError(f"unsupported fused row-parallel role {role!r}")
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=local_k // 32 // 8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=self.hidden_size // 32 // 8,
            out_block_w=self.hidden_size // 32 // 8 // 2,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        _, reduced = ttnn.experimental.matmul_reduce_scatter_async(
            tensor,
            weight,
            persistent_intermediate_buffer=pair[0],
            persistent_output_buffer=pair[1],
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            reduce_scatter_core_grid_offset=(0, 6),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        return ttnn.to_memory_config(reduced, self.local_residual_memory_config)

    def _advisor_decode_matmul(self, tensor, weight, *, role: str, compute_kernel_config):
        """Execute the exact rank-local 1-D plan emitted by ttnn-advise."""

        if role == "qkv":
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = self.advisor_qkv_output_memory_config
            program_config = self.advisor_qkv_program_config
        elif role == "packed_gate_up":
            input_memory_config = self.advisor_packed_input_memory_config
            output_memory_config = self.advisor_packed_output_memory_config
            program_config = self.advisor_packed_program_config
        elif role == "o":
            input_memory_config = self.advisor_o_input_memory_config
            output_memory_config = self.advisor_row_output_memory_config
            program_config = self.advisor_o_program_config
        elif role == "down":
            input_memory_config = self.advisor_down_input_memory_config
            output_memory_config = self.advisor_row_output_memory_config
            program_config = self.advisor_down_program_config
        else:
            raise ValueError(f"unsupported shard-advisor role {role!r}")
        tensor = ttnn.to_memory_config(tensor, input_memory_config)
        output = ttnn.matmul(
            tensor,
            weight,
            dtype=ttnn.bfloat16,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_memory_config,
        )
        # Each role report selected an output revert to DRAM interleaved.  Keep
        # it in the executable candidate so the plan is tested as advised.
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)

    def allocate_decode_position_buffers(self, current_pos: int) -> DecodePositionBuffers:
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0,{self.max_cache_len}), got {current_pos}")
        rope_index = _replicated_tensor(
            torch.full((1, 32), current_pos, dtype=torch.int64),
            mesh_device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cos_embedding_output = ttnn.embedding(
            rope_index,
            self.rotary_cos_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_embedding_output = ttnn.embedding(
            rope_index,
            self.rotary_sin_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        update_indices = _replicated_tensor(
            torch.full((self.batch,), current_pos, dtype=torch.int32),
            mesh_device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return DecodePositionBuffers(
            cos=ttnn.reshape(cos_embedding_output, [1, 1, 32, self.head_dim]),
            sin=ttnn.reshape(sin_embedding_output, [1, 1, 32, self.head_dim]),
            cos_embedding_output=cos_embedding_output,
            sin_embedding_output=sin_embedding_output,
            rope_index=rope_index,
            update_indices=update_indices,
            current_pos=current_pos,
        )

    def prepare_decode_position_buffers(
        self, buffers: DecodePositionBuffers, current_pos: int
    ) -> DecodePositionBuffers:
        if current_pos == buffers.current_pos:
            return buffers
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0,{self.max_cache_len}), got {current_pos}")
        rope_host = ttnn.from_torch(
            torch.full((1, 32), current_pos, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        update_host = ttnn.from_torch(
            torch.full((self.batch,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(rope_host, buffers.rope_index)
        ttnn.copy_host_to_device_tensor(update_host, buffers.update_indices)
        ttnn.embedding(
            buffers.rope_index,
            self.rotary_cos_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=buffers.cos_embedding_output.memory_config(),
            output_tensor=buffers.cos_embedding_output,
        )
        ttnn.embedding(
            buffers.rope_index,
            self.rotary_sin_row_major,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=buffers.sin_embedding_output.memory_config(),
            output_tensor=buffers.sin_embedding_output,
        )
        buffers.current_pos = current_pos
        return buffers

    def _eager_decode_position_buffers(self, current_pos: int) -> DecodePositionBuffers:
        if self._eager_position_buffers is None:
            self._eager_position_buffers = self.allocate_decode_position_buffers(current_pos)
        else:
            self.prepare_decode_position_buffers(self._eager_position_buffers, current_pos)
        return self._eager_position_buffers

    def _prefill_linear(self, tensor, weight, *, k: int, n: int, compute_kernel_config):
        rows = int(tensor.shape[-2])
        max_chunk_rows = 640
        if rows <= max_chunk_rows:
            matmul_input = tensor
            if self.prefill_matmul_input_l1:
                matmul_input = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
            output = ttnn.matmul(
                matmul_input,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    rows,
                    k,
                    n,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if matmul_input is not tensor:
                ttnn.deallocate(matmul_input, True)
            return output
        chunks = []
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            chunk = ttnn.slice(tensor, [0, 0, start, 0], [1, 1, end, k])
            matmul_input = chunk
            if self.prefill_matmul_input_l1:
                matmul_input = ttnn.to_memory_config(chunk, ttnn.L1_MEMORY_CONFIG)
            projected = ttnn.matmul(
                matmul_input,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    end - start,
                    k,
                    n,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if matmul_input is not chunk:
                ttnn.deallocate(matmul_input, True)
            ttnn.deallocate(chunk, True)
            chunks.append(projected)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _prefill_row_parallel(self, tensor, weight, *, k: int, compute_kernel_config):
        """Project row-parallel chunks and reduce-scatter before concatenation."""

        rows = int(tensor.shape[-2])
        max_chunk_rows = 640
        chunks = []
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            chunk = tensor
            if rows > max_chunk_rows:
                chunk = ttnn.slice(
                    tensor,
                    [0, 0, start, 0],
                    [1, 1, end, k],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            matmul_input = chunk
            if self.prefill_matmul_input_l1:
                matmul_input = ttnn.to_memory_config(chunk, ttnn.L1_MEMORY_CONFIG)
            partial = ttnn.matmul(
                matmul_input,
                weight,
                dtype=ttnn.bfloat16,
                program_config=_prefill_matmul_program_config(
                    self.mesh_device,
                    end - start,
                    k,
                    self.hidden_size,
                    in0_block_w=self.prefill_in0_block_w,
                ),
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if matmul_input is not chunk:
                ttnn.deallocate(matmul_input, True)
            if chunk is not tensor:
                ttnn.deallocate(chunk, True)
            reduced = self._reduce_scatter_hidden(
                partial, memory_config=ttnn.DRAM_MEMORY_CONFIG, decode=False
            )
            ttnn.deallocate(partial, True)
            chunks.append(reduced)
        if len(chunks) == 1:
            return chunks[0]
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _prefill_mlp_chunk(self, residual):
        rows = int(residual.shape[-2])
        gathered = self._all_gather_hidden(residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normed = ttnn.rms_norm(
            gathered,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered, True)
        packed = None
        if self.use_packed_mlp:
            packed = self._prefill_linear(
                normed,
                self.gate_up_prefill_weight,
                k=self.hidden_size,
                n=2 * self.local_intermediate_size,
                compute_kernel_config=self.mlp_compute_config,
            )
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, 1, rows, self.local_intermediate_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.local_intermediate_size],
                [1, 1, rows, 2 * self.local_intermediate_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gate = self._prefill_linear(
                normed,
                self.gate_prefill_weight,
                k=self.hidden_size,
                n=self.local_intermediate_size,
                compute_kernel_config=self.mlp_compute_config,
            )
            up = self._prefill_linear(
                normed,
                self.up_prefill_weight,
                k=self.hidden_size,
                n=self.local_intermediate_size,
                compute_kernel_config=self.mlp_compute_config,
            )
        ttnn.deallocate(normed, True)
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate, True)
        ttnn.deallocate(up, True)
        if packed is not None:
            ttnn.deallocate(packed, True)
        down = self._prefill_row_parallel(
            gated,
            self.down_prefill_weight,
            k=self.local_intermediate_size,
            compute_kernel_config=self.mlp_compute_config,
        )
        ttnn.deallocate(gated, True)
        output = ttnn.add(residual, down, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(down, True)
        return output

    def _prefill_mlp(self, residual):
        rows = int(residual.shape[-2])
        if rows <= 640:
            return self._prefill_mlp_chunk(residual)
        chunks = []
        for start in range(0, rows, 640):
            end = min(start + 640, rows)
            chunk = ttnn.slice(
                residual,
                [0, 0, start, 0],
                [1, 1, end, self.local_hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            chunks.append(self._prefill_mlp_chunk(chunk))
            ttnn.deallocate(chunk, True)
        output = ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for chunk in chunks:
            ttnn.deallocate(chunk, True)
        return output

    def _fill_prefill_cache(self, cache, tensor, *, page_table, mode: str):
        for user_id in range(self.batch):
            user_view = ttnn.slice(
                tensor,
                [user_id, 0, 0, 0],
                [user_id + 1, self.local_num_kv_heads, tensor.shape[2], self.head_dim],
            )
            # An aligned slice may be an alias.  Materialize it before cache
            # conversion/padding so releasing the per-user temporary cannot
            # invalidate the parent K/V tensor for the next user.
            user = ttnn.clone(user_view, dtype=cache.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owns_user_buffer = True
            if mode == "contiguous":
                ttnn.fill_cache(cache, user, user_id)
            else:
                page_len = int(page_table.shape[1]) * int(cache.shape[2])
                if int(user.shape[2]) < page_len:
                    padded = ttnn.pad(
                        user,
                        [(0, 0), (0, 0), (0, page_len - int(user.shape[2])), (0, 0)],
                        0.0,
                    )
                    if owns_user_buffer:
                        ttnn.deallocate(user, True)
                    user = padded
                    owns_user_buffer = True
                elif int(user.shape[2]) > page_len:
                    sliced = ttnn.slice(
                        user,
                        [0, 0, 0, 0],
                        [1, self.local_num_kv_heads, page_len, self.head_dim],
                    )
                    if owns_user_buffer:
                        ttnn.deallocate(user, True)
                    user = sliced
                    owns_user_buffer = True
                ttnn.experimental.paged_fill_cache(cache, user, page_table, batch_idx=user_id)
            if owns_user_buffer:
                ttnn.deallocate(user, True)

    def prefill_forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        page_table=None,
    ):
        if self.residual_contract != "sharded":
            raise ValueError("prefill is supported only for the selected sharded residual contract")
        seq_len = self._validate_hidden_states(hidden_states)
        cache_mode = self._cache_mode(key_cache, value_cache, page_table)
        rows = self.batch * seq_len
        residual = ttnn.reshape(hidden_states, [1, 1, rows, self.local_hidden_size])
        gathered = self._all_gather_hidden(residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normed = ttnn.rms_norm(
            gathered,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered, True)
        fused_qkv = self._prefill_linear(
            normed,
            self.qkv_prefill_weight,
            k=self.hidden_size,
            n=self.local_qkv_size,
            compute_kernel_config=self.attention_compute_config,
        )
        ttnn.deallocate(normed, True)
        fused_qkv = ttnn.reshape(fused_qkv, [self.batch, seq_len, self.local_qkv_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Exact tile-aligned sequence lengths may return split views backed by
        # fused_qkv.  Keep the parent alive until Q/K/V consumers finish; a
        # forced release here invalidates the aligned value view.
        normalized_query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(query, True)
        query = normalized_query
        normalized_key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(key, True)
        key = normalized_key
        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        rotated_query = ttnn.experimental.rotary_embedding(
            query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        rotated_key = ttnn.experimental.rotary_embedding(
            key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(query, True)
        ttnn.deallocate(key, True)
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        query = ttnn.slice(
            rotated_query,
            [0, 0, 0, 0],
            [self.batch, self.local_num_heads, seq_len, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            rotated_key,
            [0, 0, 0, 0],
            [self.batch, self.local_num_kv_heads, seq_len, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Full-tile slices can alias the rotated parents.  Retain both parents
        # through cache fill/SDPA so aligned logical lengths remain valid.
        self._fill_prefill_cache(
            key_cache,
            key,
            page_table=page_table,
            mode=cache_mode,
        )
        self._fill_prefill_cache(
            value_cache,
            value,
            page_table=page_table,
            mode=cache_mode,
        )
        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            ),
            compute_kernel_config=self.attention_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(query, True)
        ttnn.deallocate(key, True)
        ttnn.deallocate(value, True)
        concatenated_attention = ttnn.transformer.concatenate_heads(
            attention, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attention, True)
        attention = concatenated_attention
        attention = ttnn.reshape(attention, [1, 1, rows, self.local_attention_width])
        attention_shard = self._prefill_row_parallel(
            attention,
            self.output_prefill_weight,
            k=self.local_attention_width,
            compute_kernel_config=self.attention_compute_config,
        )
        ttnn.deallocate(attention, True)
        attention_residual = ttnn.add(
            residual, attention_shard, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attention_shard, True)
        residual = self._prefill_mlp(attention_residual)
        ttnn.deallocate(attention_residual, True)
        return ttnn.reshape(residual, [1, self.batch, seq_len, self.local_hidden_size])

    def _decode_mlp(self, residual):
        normed = self._decode_norm(residual, self.post_attention_norm)
        if self.use_packed_mlp:
            if self.use_fused_decode_all_gather_matmul:
                packed = self._fused_decode_column_parallel(
                    normed,
                    self.gate_up_prefill_weight,
                    role="packed_gate_up",
                    compute_kernel_config=self.mlp_compute_config,
                )
            elif self.decode_matmul_mode == "shard_advisor":
                packed = self._advisor_decode_matmul(
                    normed,
                    self.gate_up_prefill_weight,
                    role="packed_gate_up",
                    compute_kernel_config=self.mlp_compute_config,
                )
            else:
                normed = ttnn.to_memory_config(normed, self.mlp_gate_input_memory_config)
                packed = ttnn.matmul(
                    normed,
                    self.gate_up_weight,
                    dtype=ttnn.bfloat16,
                    program_config=self.packed_gate_decode_program_config,
                    compute_kernel_config=self.mlp_compute_config,
                    memory_config=self.packed_mlp_memory_config,
                )
            split_memory_config = (
                self.mlp_gate_memory_config
                if self.packed_mlp_split_layout == "l1"
                else ttnn.DRAM_MEMORY_CONFIG
            )
            if self.packed_mlp_split_layout == "dram":
                packed = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
            gate = ttnn.slice(
                packed,
                [0, 0, 0, 0],
                [1, 1, self.batch, self.local_intermediate_size],
                memory_config=split_memory_config,
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.local_intermediate_size],
                [1, 1, self.batch, 2 * self.local_intermediate_size],
                memory_config=split_memory_config,
            )
            elementwise_memory_config = split_memory_config
        else:
            normed = ttnn.to_memory_config(normed, self.mlp_gate_input_memory_config)
            gate = ttnn.matmul(
                normed,
                self.gate_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_memory_config,
            )
            up = ttnn.matmul(
                normed,
                self.up_weight,
                dtype=ttnn.bfloat16,
                program_config=self.gate_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_gate_memory_config,
            )
            elementwise_memory_config = self.mlp_gate_memory_config
        gated = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat16,
            memory_config=elementwise_memory_config,
        )
        if self.use_fused_decode_reduce_scatter:
            down = self._fused_decode_row_parallel(
                gated,
                self.down_prefill_weight,
                role="down",
                compute_kernel_config=self.mlp_compute_config,
            )
        elif self.decode_matmul_mode == "shard_advisor":
            partial = self._advisor_decode_matmul(
                gated,
                self.down_prefill_weight,
                role="down",
                compute_kernel_config=self.mlp_compute_config,
            )
            down = self._row_parallel_collective(partial, decode=True)
        else:
            gated = ttnn.to_memory_config(gated, self.mlp_down_input_memory_config)
            partial = ttnn.matmul(
                gated,
                self.down_weight,
                dtype=ttnn.bfloat16,
                program_config=self.down_decode_program_config,
                compute_kernel_config=self.mlp_compute_config,
                memory_config=self.mlp_down_partial_memory_config,
            )
            down = self._row_parallel_collective(partial, decode=True)
        return ttnn.add(
            residual,
            down,
            dtype=ttnn.bfloat16,
            memory_config=(
                self.full_residual_memory_config
                if self.residual_contract == "replicated_provenance"
                else self.local_residual_memory_config
            ),
        )

    def decode_forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        current_pos: int,
        position_buffers: DecodePositionBuffers | None = None,
        page_table=None,
    ):
        if self.residual_contract == "replicated_provenance":
            shape = tuple(hidden_states.shape)
            if shape != (1, self.batch, 1, self.hidden_size):
                raise ValueError(
                    "replicated_provenance hidden_states must be a replicated "
                    f"[1,{self.batch},1,{self.hidden_size}] mesh tensor, got {shape}"
                )
        else:
            self._validate_hidden_states(hidden_states, expected_seq_len=1)
        cache_mode = self._cache_mode(key_cache, value_cache, page_table)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0,{self.max_cache_len}), got {current_pos}")
        boundary_width = (
            self.hidden_size if self.residual_contract == "replicated_provenance" else self.local_hidden_size
        )
        boundary_memory_config = (
            self.full_residual_memory_config
            if self.residual_contract == "replicated_provenance"
            else self.local_residual_memory_config
        )
        residual = ttnn.reshape(hidden_states, [1, 1, self.batch, boundary_width])
        residual = ttnn.to_memory_config(residual, boundary_memory_config)
        normed = self._decode_norm(residual, self.input_norm)
        if self.use_fused_decode_all_gather_matmul:
            fused_qkv = self._fused_decode_column_parallel(
                normed,
                self.qkv_prefill_weight,
                role="qkv",
                compute_kernel_config=self.attention_compute_config,
            )
        elif self.decode_matmul_mode == "shard_advisor":
            fused_qkv = self._advisor_decode_matmul(
                normed,
                self.qkv_prefill_weight,
                role="qkv",
                compute_kernel_config=self.attention_compute_config,
            )
        else:
            normed = ttnn.to_memory_config(normed, self.qkv_input_memory_config)
            fused_qkv = ttnn.matmul(
                normed,
                self.qkv_weight,
                dtype=ttnn.bfloat16,
                program_config=self.qkv_decode_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=self.qkv_output_memory_config,
            )
        fused_qkv = ttnn.to_memory_config(fused_qkv, ttnn.L1_MEMORY_CONFIG)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.local_num_heads,
            num_kv_heads=self.local_num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )
        value = ttnn.to_memory_config(value, self.decode_kv_mem_config)
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if position_buffers is None:
            position_buffers = self._eager_decode_position_buffers(current_pos)
        elif position_buffers.current_pos != current_pos:
            raise ValueError(
                f"position_buffers hold {position_buffers.current_pos}, expected {current_pos}"
            )
        query = ttnn.experimental.rotary_embedding(
            query, position_buffers.cos, position_buffers.sin, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        key = ttnn.experimental.rotary_embedding(
            key, position_buffers.cos, position_buffers.sin, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        key = ttnn.to_memory_config(key, self.decode_kv_mem_config)
        update_indices = position_buffers.update_indices
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=page_table,
        )
        # SDPA decode writes one 32-row head tile per batch.  On Blackhole its
        # multi-KV writer leaves the second GQA group invalid when the logical
        # Q count is 16 even though the physical head tile is 32 rows.  Present
        # a full logical tile by duplicating each eight-head GQA group, then
        # select one copy of each group from the result.  All work stays on
        # device, and the local 16Q/2KV ownership is unchanged.
        query_group_0 = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [1, self.batch, 8, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query_group_1 = ttnn.slice(
            query,
            [0, 0, 8, 0],
            [1, self.batch, 16, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sdpa_query = ttnn.concat(
            [query_group_0, query_group_0, query_group_1, query_group_1],
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if cache_mode == "paged":
            attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                sdpa_query,
                key_cache,
                value_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attention = ttnn.transformer.scaled_dot_product_attention_decode(
                sdpa_query,
                key_cache,
                value_cache,
                is_causal=True,
                cur_pos_tensor=update_indices,
                scale=self.scale,
                program_config=self.decode_sdpa_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attention_group_0 = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, self.batch, 8, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_group_1 = ttnn.slice(
            attention,
            [0, 0, 16, 0],
            [1, self.batch, 24, self.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.concat(
            [attention_group_0, attention_group_1],
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention, num_heads=self.local_num_heads
        )
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.local_attention_width],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_fused_decode_reduce_scatter:
            attention_shard = self._fused_decode_row_parallel(
                attention,
                self.output_prefill_weight,
                role="o",
                compute_kernel_config=self.attention_compute_config,
            )
        elif self.decode_matmul_mode == "shard_advisor":
            partial = self._advisor_decode_matmul(
                attention,
                self.output_prefill_weight,
                role="o",
                compute_kernel_config=self.attention_compute_config,
            )
            attention_shard = self._row_parallel_collective(partial, decode=True)
        else:
            attention = ttnn.to_memory_config(attention, self.o_input_memory_config)
            partial = ttnn.matmul(
                attention,
                self.output_weight,
                dtype=ttnn.bfloat16,
                program_config=self.o_decode_program_config,
                compute_kernel_config=self.attention_compute_config,
                memory_config=self.o_partial_memory_config,
            )
            attention_shard = self._row_parallel_collective(partial, decode=True)
        residual = ttnn.add(
            residual,
            attention_shard,
            dtype=ttnn.bfloat16,
            memory_config=boundary_memory_config,
        )
        residual = self._decode_mlp(residual)
        if not self.keep_decode_output_sharded:
            residual = ttnn.to_memory_config(residual, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(residual, [1, self.batch, 1, boundary_width])

    def forward(
        self,
        hidden_states,
        key_cache,
        value_cache,
        *,
        mode: str,
        current_pos: int | None = None,
        page_table=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(
                hidden_states,
                key_cache,
                value_cache,
                page_table=page_table,
            )
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(
                hidden_states,
                key_cache,
                value_cache,
                current_pos=current_pos,
                page_table=page_table,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
