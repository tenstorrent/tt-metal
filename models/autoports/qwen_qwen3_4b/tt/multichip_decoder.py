# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import RMS_NORM_EPS, Qwen3DecoderConfig, _get_layer_tensor
from models.autoports.qwen_qwen3_4b.tt.optimized_decoder import OptimizedDecoder, PagedKVConfig
from models.common.lightweightmodule import LightweightModule

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional outside profiling runs.

    def signpost(header):
        return None


DEFAULT_MULTICHIP_KV_CONFIG = PagedKVConfig(max_num_blocks=2560, block_size=16)


@dataclass(frozen=True)
class MultichipDecoderTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None


class MultichipDecoder(LightweightModule):
    """Qwen3-4B dense decoder layer with 1x4 tensor parallel execution.

    This stage intentionally starts from the optimized single-chip decoder's
    math and precision policy, but shards projection weights and KV heads across
    a four-device ring. Layer inputs and outputs are replicated hidden-state
    tensors so stacked decoder layers can be composed without boundary gathers.
    """

    baseline_cls = OptimizedDecoder
    mesh_profile = {
        "name": "qwen3_4b_multichip_decoder_1x4_tp4_v1",
        "single_chip_baseline": OptimizedDecoder.optimization_profile["name"],
        "target_mesh": "1x4 Blackhole ring",
        "tp": 4,
        "activation_contract": "replicated hidden state at layer input/output; local TP shards inside attention/MLP",
        "attention": "QKV column parallel with per-device Q/K/V packing; local paged SDPA; row-parallel WO + ring all_reduce",
        "mlp": "gate/up column parallel; down row parallel; ring all_reduce",
        "kv_cache": "paged BF16 cache with local KV heads per device",
        "weight_dtype": "bfloat4_b",
        "math_fidelity": "LoFi",
    }

    def __init__(
        self,
        *,
        cfg: Qwen3DecoderConfig,
        layer_idx: int,
        mesh_device,
        qkv_prefill_weight: ttnn.Tensor,
        qkv_decode_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        packed_gate_up_proj_weight: ttnn.Tensor | None,
        down_proj_weight: ttnn.Tensor,
        down_proj_weight_dram_sharded: ttnn.Tensor,
        q_norm_weight: ttnn.Tensor,
        k_norm_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        max_seq_len: int,
        paged_kv_config: PagedKVConfig,
        attention_math_fidelity: ttnn.MathFidelity,
        mlp_math_fidelity: ttnn.MathFidelity,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        num_links: int = 1,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.tp = mesh_device.get_num_devices()
        if tuple(mesh_device.shape) != (1, 4) or self.tp != 4:
            raise ValueError(f"MultichipDecoder targets a 1x4 mesh, got shape={tuple(mesh_device.shape)}")
        if cfg.num_attention_heads % self.tp != 0 or cfg.num_key_value_heads % self.tp != 0:
            raise ValueError("attention and KV heads must divide evenly across TP=4")
        if cfg.intermediate_size % self.tp != 0:
            raise ValueError("MLP intermediate size must divide evenly across TP=4")

        self.local_num_attention_heads = cfg.num_attention_heads // self.tp
        self.local_num_key_value_heads = cfg.num_key_value_heads // self.tp
        self.local_q_width = self.local_num_attention_heads * cfg.head_dim
        self.local_kv_width = self.local_num_key_value_heads * cfg.head_dim
        self.local_qkv_width = self.local_q_width + 2 * self.local_kv_width
        self.local_intermediate_size = cfg.intermediate_size // self.tp

        self.qkv_prefill_weight = qkv_prefill_weight
        self.qkv_decode_weight = qkv_decode_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.packed_gate_up_proj_weight = packed_gate_up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.down_proj_weight_dram_sharded = down_proj_weight_dram_sharded
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.position_cos = position_cos
        self.position_sin = position_sin
        self.attention_mask = attention_mask
        self.max_seq_len = max_seq_len
        self.paged_kv_config = paged_kv_config
        self.topology = topology
        self.num_links = num_links
        self.geometry_mode = os.environ.get(
            "QWEN3_4B_MULTICHIP_GEOMETRY", "qkv_1d_dram_24c_i5_s2,gate_up_1d_l1_20c_i10_s4"
        )
        self.geometry_modes = {mode for mode in self.geometry_mode.split(",") if mode}
        self._persistent_all_reduce_cache = {}
        self._persistent_all_reduce_index = {}
        self.timings = MultichipDecoderTimings()

        self.compute_kernel_config_hifi2 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.mlp_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if mlp_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.attention_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if attention_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.auxiliary_compute_kernel_config = self.compute_kernel_config_lofi
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    @staticmethod
    def _pad_to(value: int, multiple: int) -> int:
        remainder = value % multiple
        return value if remainder == 0 else value + multiple - remainder

    @staticmethod
    def _decode_dram_matmul_num_cores(mesh_device, k: int | None = None) -> int:
        max_cores = min(8, mesh_device.compute_with_storage_grid_size().x)
        if k is None:
            return max_cores
        k_tiles = k // 32
        for num_cores in range(max_cores, 0, -1):
            if k_tiles % num_cores == 0:
                return num_cores
        return 1

    @classmethod
    def _decode_dram_core_range(cls, mesh_device, k: int | None = None) -> ttnn.CoreRangeSet:
        num_cores = cls._decode_dram_matmul_num_cores(mesh_device, k)
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    @classmethod
    def _decode_dram_weight_memory_config(cls, mesh_device, k: int, n: int) -> ttnn.MemoryConfig:
        dram_banks = mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                )
            }
        )
        shard_shape = [k, cls._pad_to(n, 32 * dram_banks) // dram_banks]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _decode_dram_num_cores(self, k: int) -> int:
        if "down_dram_2c_i38_n40" in self.geometry_modes and k == self.local_intermediate_size:
            return 2
        return self._decode_dram_matmul_num_cores(self.mesh_device, k)

    def _decode_dram_core_range_for_k(self, k: int) -> ttnn.CoreRangeSet:
        num_cores = self._decode_dram_num_cores(k)
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    def _decode_dram_input_memory_config(self, k: int) -> ttnn.MemoryConfig:
        num_cores = self._decode_dram_num_cores(k)
        shard_shape = [32, k // num_cores]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self._decode_dram_core_range_for_k(k), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _decode_dram_output_memory_config(self, k: int, n: int) -> ttnn.MemoryConfig:
        num_cores = self._decode_dram_num_cores(k)
        shard_shape = [32, self._pad_to(n, 32 * num_cores) // num_cores]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self._decode_dram_core_range_for_k(k), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _decode_dram_matmul_program_config(
        self, k: int, n: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        num_cores = self._decode_dram_num_cores(k)
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=max(1, k // (32 * num_cores)),
            per_core_M=1,
            per_core_N=math.ceil(n / (32 * num_cores)),
            fused_activation=None,
        )

    def _decode_width_l1_memory_config(self, width: int, num_cores: int) -> ttnn.MemoryConfig:
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
                [32, self._pad_to(width, 32 * num_cores) // num_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    @staticmethod
    def _decode_1d_matmul_program_config(
        *, grid_x: int, grid_y: int, in0_block_w: int, per_core_n: int, out_subblock_w: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _qkv_decode_geometry(self):
        if "qkv_1d_dram_24c_i5_s2" in self.geometry_modes:
            return None, self._decode_1d_matmul_program_config(
                grid_x=8, grid_y=3, in0_block_w=5, per_core_n=2, out_subblock_w=2
            )
        if "qkv_width_l1_10c_i8_s5" in self.geometry_modes:
            return self._decode_width_l1_memory_config(self.cfg.hidden_size, 10), self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=1, in0_block_w=8, per_core_n=5, out_subblock_w=5
            )
        if "qkv_width_l1_8c_i10_s6" in self.geometry_modes:
            return self._decode_width_l1_memory_config(self.cfg.hidden_size, 8), self._decode_1d_matmul_program_config(
                grid_x=8, grid_y=1, in0_block_w=10, per_core_n=6, out_subblock_w=6
            )
        return None, None

    def _qkv_prefill_geometry(self):
        if "prefill_qkv_1d_dram_24c_i5_s2" in self.geometry_modes:
            return self._decode_1d_matmul_program_config(
                grid_x=8, grid_y=3, in0_block_w=5, per_core_n=2, out_subblock_w=2
            )
        return None

    def _gate_up_decode_geometry(self):
        if "packed_gate_up_1d_l1_20c_i10_s4" in self.geometry_modes:
            return None, self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=2, in0_block_w=10, per_core_n=8, out_subblock_w=4
            )
        if "gate_up_1d_l1_20c_i10_s4" in self.geometry_modes:
            return None, self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=2, in0_block_w=10, per_core_n=4, out_subblock_w=4
            )
        if "gate_up_1d_l1_10c_i8_s8" in self.geometry_modes:
            return None, self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=1, in0_block_w=8, per_core_n=8, out_subblock_w=8
            )
        if "gate_up_width_l1_10c_i8_s8" in self.geometry_modes:
            return self._decode_width_l1_memory_config(self.cfg.hidden_size, 10), self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=1, in0_block_w=8, per_core_n=8, out_subblock_w=8
            )
        return None, None

    def _gate_up_prefill_geometry(self):
        if "prefill_gate_up_1d_dram_20c_i10_s4" in self.geometry_modes:
            return self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=2, in0_block_w=10, per_core_n=4, out_subblock_w=4
            )
        return None

    def _down_prefill_geometry(self):
        if "prefill_down_1d_dram_20c_i10_s4" in self.geometry_modes:
            return self._decode_1d_matmul_program_config(
                grid_x=10, grid_y=2, in0_block_w=10, per_core_n=4, out_subblock_w=4
            )
        return None

    def _wo_decode_geometry(self):
        if "wo_1d_explicit_8c_i4_s4" in self.geometry_modes:
            return self._decode_1d_matmul_program_config(
                grid_x=8, grid_y=1, in0_block_w=4, per_core_n=4, out_subblock_w=4
            )
        return None

    def _all_reduce_l1_memory_config(self, m: int, width: int) -> ttnn.MemoryConfig:
        num_cores = min(8, self.mesh_device.compute_with_storage_grid_size().x)
        padded_m = self._pad_to(m, 32)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
                [padded_m, self._pad_to(width, 32 * num_cores) // num_cores],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _persistent_all_reduce_resources(self, m: int, width: int, dtype: ttnn.DataType):
        key = (m, width, dtype)
        if key not in self._persistent_all_reduce_cache:
            mesh_cores = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(
                            self.mesh_device.compute_with_storage_grid_size().x - 1,
                            self.mesh_device.compute_with_storage_grid_size().y - 1,
                        ),
                    )
                }
            )
            buffer_mem = self._all_reduce_l1_memory_config(m, width * self.tp)
            resources = []
            for _ in range(2):
                buffer = ttnn.from_torch(
                    torch.zeros(1, 1, m, width * self.tp, dtype=torch.bfloat16),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=buffer_mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                semaphore = ttnn.create_global_semaphore(self.mesh_device, mesh_cores, 0)
                resources.append((buffer, semaphore))
            self._persistent_all_reduce_cache[key] = resources
            self._persistent_all_reduce_index[key] = 0
        resources = self._persistent_all_reduce_cache[key]
        index = self._persistent_all_reduce_index[key]
        self._persistent_all_reduce_index[key] = (index + 1) % len(resources)
        return resources[index]

    @staticmethod
    def _mesh_replicated_tensor(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor.detach().contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def _mesh_sharded_tensor(
        tensor: torch.Tensor,
        mesh_device,
        *,
        dim: int,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    ):
        return ttnn.from_torch(
            tensor.detach().contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

    @classmethod
    def _build_rope_tables(cls, cfg: Qwen3DecoderConfig, seq_len: int, mesh_device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        angles = torch.einsum("d,s->sd", inv_freq, positions)
        emb = torch.cat((angles, angles), dim=-1).reshape(1, 1, seq_len, cfg.head_dim)
        return (
            cls._mesh_replicated_tensor(torch.cos(emb), mesh_device),
            cls._mesh_replicated_tensor(torch.sin(emb), mesh_device),
        )

    @classmethod
    def _build_causal_mask(cls, seq_len: int, mesh_device) -> ttnn.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
        return cls._mesh_replicated_tensor(torch.triu(mask, diagonal=1), mesh_device)

    @staticmethod
    def _pack_qkv_by_device(
        q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, *, tp: int, prefill: bool
    ):
        q_shards = torch.chunk(q_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        k_shards = torch.chunk(k_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        v_shards = torch.chunk(v_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        packed = []
        for q, k, v in zip(q_shards, k_shards, v_shards):
            packed.append(torch.cat((v, q, k), dim=1) if prefill else torch.cat((q, k, v), dim=1))
        return torch.cat(packed, dim=1)

    @staticmethod
    def _pack_down_for_dram_sharded_by_device(down_proj: torch.Tensor, *, tp: int) -> torch.Tensor:
        return torch.cat([shard.transpose(0, 1).contiguous() for shard in torch.chunk(down_proj, tp, dim=1)], dim=0)

    @staticmethod
    def _pack_gate_up_by_device(gate_proj: torch.Tensor, up_proj: torch.Tensor, *, tp: int) -> torch.Tensor:
        gate_shards = torch.chunk(gate_proj.detach().contiguous(), tp, dim=0)
        up_shards = torch.chunk(up_proj.detach().contiguous(), tp, dim=0)
        return torch.cat([torch.cat((gate, up), dim=0) for gate, up in zip(gate_shards, up_shards)], dim=0)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_seq_len: int = DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
        paged_kv_config: PagedKVConfig | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        attention_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        mlp_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        **kwargs,
    ) -> "MultichipDecoder":
        if kwargs:
            raise TypeError(f"unsupported MultichipDecoder kwargs: {sorted(kwargs)}")
        cfg = Qwen3DecoderConfig.from_hf_config(hf_config)
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")
        paged_kv_config = paged_kv_config or DEFAULT_MULTICHIP_KV_CONFIG
        if max_seq_len > paged_kv_config.max_seq_len:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds paged KV capacity {paged_kv_config.max_seq_len}; "
                "increase PagedKVConfig.max_num_blocks or lower max_seq_len"
            )
        attention_weight_dtype = attention_weight_dtype or weight_dtype
        mlp_weight_dtype = mlp_weight_dtype or weight_dtype
        tp = mesh_device.get_num_devices()
        if tp != 4:
            raise ValueError(f"MultichipDecoder expects TP=4, got {tp}")

        q_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        qkv_prefill = cls._pack_qkv_by_device(q_proj, k_proj, v_proj, tp=tp, prefill=True)
        qkv_decode = cls._pack_qkv_by_device(q_proj, k_proj, v_proj, tp=tp, prefill=False)
        down_proj = _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
        down_dram_sharded = cls._pack_down_for_dram_sharded_by_device(down_proj, tp=tp)
        gate_proj = _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
        up_proj = _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
        geometry_modes = {mode for mode in os.environ.get("QWEN3_4B_MULTICHIP_GEOMETRY", "").split(",") if mode}
        packed_gate_up = (
            cls._pack_gate_up_by_device(gate_proj, up_proj, tp=tp)
            if "packed_gate_up_1d_l1_20c_i10_s4" in geometry_modes
            else None
        )

        position_cos, position_sin = cls._build_rope_tables(cfg, max_seq_len, mesh_device)
        attention_mask = cls._build_causal_mask(max_seq_len, mesh_device)

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            qkv_prefill_weight=cls._mesh_sharded_tensor(qkv_prefill, mesh_device, dim=1, dtype=attention_weight_dtype),
            qkv_decode_weight=cls._mesh_sharded_tensor(qkv_decode, mesh_device, dim=1, dtype=attention_weight_dtype),
            o_proj_weight=cls._mesh_sharded_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight"),
                mesh_device,
                dim=1,
                dtype=attention_weight_dtype,
            ),
            gate_proj_weight=cls._mesh_sharded_tensor(
                gate_proj,
                mesh_device,
                dim=0,
                dtype=mlp_weight_dtype,
            ),
            up_proj_weight=cls._mesh_sharded_tensor(
                up_proj,
                mesh_device,
                dim=0,
                dtype=mlp_weight_dtype,
            ),
            packed_gate_up_proj_weight=(
                cls._mesh_sharded_tensor(
                    packed_gate_up,
                    mesh_device,
                    dim=0,
                    dtype=mlp_weight_dtype,
                )
                if packed_gate_up is not None
                else None
            ),
            down_proj_weight=cls._mesh_sharded_tensor(
                down_proj,
                mesh_device,
                dim=1,
                dtype=mlp_weight_dtype,
            ),
            down_proj_weight_dram_sharded=ttnn.from_torch(
                down_dram_sharded,
                dtype=mlp_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=cls._decode_dram_weight_memory_config(
                    mesh_device, cfg.intermediate_size // tp, cfg.hidden_size
                ),
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            ),
            q_norm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.q_norm.weight"), mesh_device
            ),
            k_norm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.k_norm.weight"), mesh_device
            ),
            input_layernorm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "input_layernorm.weight"), mesh_device
            ),
            post_attention_layernorm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
            max_seq_len=max_seq_len,
            paged_kv_config=paged_kv_config,
            attention_math_fidelity=attention_math_fidelity,
            mlp_math_fidelity=mlp_math_fidelity,
            topology=topology,
            num_links=num_links,
        )

    def init_paged_kv_cache(self) -> list[ttnn.Tensor]:
        cache_shape = (
            self.paged_kv_config.max_num_blocks,
            self.local_num_key_value_heads,
            self.paged_kv_config.block_size,
            self.cfg.head_dim,
        )
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
        return [
            self._mesh_replicated_tensor(
                zeros,
                self.mesh_device,
                dtype=self.paged_kv_config.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
            )
            for _ in range(2)
        ]

    def make_identity_page_table(self, batch_size: int = 1) -> ttnn.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.paged_kv_config.max_num_blocks % batch_size != 0:
            raise ValueError(
                f"max_num_blocks={self.paged_kv_config.max_num_blocks} must divide evenly across batch_size={batch_size}"
            )
        blocks_per_user = self.paged_kv_config.max_num_blocks // batch_size
        pages = torch.arange(self.paged_kv_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_size, blocks_per_user
        )
        return ttnn.from_torch(
            pages,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def make_current_pos(self, positions: list[int] | torch.Tensor) -> ttnn.Tensor:
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.int32)
        return ttnn.from_torch(
            positions.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def position_tables_for_decode(self, position: int, *, batch_size: int = 1) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(f"decode position must be in [0, {self.max_seq_len}), got {position}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > 1:
            positions = torch.full((batch_size,), position, dtype=torch.float32)
            inv_freq = 1.0 / (
                self.cfg.rope_theta ** (torch.arange(0, self.cfg.head_dim, 2, dtype=torch.float32) / self.cfg.head_dim)
            )
            angles = torch.einsum("d,s->sd", inv_freq, positions)
            emb = torch.cat((angles, angles), dim=-1).reshape(1, 1, batch_size, self.cfg.head_dim)
            return (
                self._mesh_replicated_tensor(torch.cos(emb), self.mesh_device),
                self._mesh_replicated_tensor(torch.sin(emb), self.mesh_device),
            )
        start = [0, 0, position, 0]
        end = [1, 1, position + 1, self.cfg.head_dim]
        return (
            ttnn.slice(self.position_cos, start, end, [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.slice(self.position_sin, start, end, [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

    def _all_reduce_hidden(
        self, partial: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_persistent: bool = False
    ) -> ttnn.Tensor:
        if use_persistent:
            m = int(partial.shape[-2])
            width = int(partial.shape[-1])
            l1_mem = self._all_reduce_l1_memory_config(m, width)
            partial = ttnn.to_memory_config(partial, l1_mem)
            buffer, semaphore = self._persistent_all_reduce_resources(m, width, partial.get_dtype())
            reduced = ttnn.experimental.all_reduce_async(
                partial,
                buffer,
                1,
                self.mesh_device,
                semaphore,
                dtype=partial.get_dtype(),
                memory_config=l1_mem,
                topology=self.topology,
                num_links=self.num_links,
            )
            return ttnn.to_memory_config(reduced, memory_config)
        return ttnn.experimental.all_reduce_async(
            partial,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            num_links=self.num_links,
            math_op=ttnn.ReduceType.Sum,
            topology=self.topology,
            memory_config=memory_config,
        )

    def _fill_paged_kv_cache(self, k, v, kv_cache, page_table, *, user_id: int = 0) -> None:
        k_cache, v_cache = kv_cache
        if k.dtype != k_cache.dtype:
            k = ttnn.typecast(k, k_cache.dtype)
        if v.dtype != v_cache.dtype:
            v = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=user_id)

    def _prefill_attention(self, hidden_states, seq_len, cos, sin, mask, kv_cache=None, page_table=None, user_id=0):
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=RMS_NORM_EPS,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        qkv = ttnn.matmul(
            normed,
            self.qkv_prefill_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._qkv_prefill_geometry(),
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        v = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, seq_len, self.local_kv_width], [1, 1, 1, 1])
        q = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_kv_width],
            [1, 1, seq_len, self.local_kv_width + self.local_q_width],
            [1, 1, 1, 1],
        )
        k = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_kv_width + self.local_q_width],
            [1, 1, seq_len, self.local_qkv_width],
            [1, 1, 1, 1],
        )

        v = ttnn.reshape(v, [1, seq_len, self.local_num_key_value_heads, self.cfg.head_dim])
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, [1, seq_len, self.local_num_key_value_heads, self.cfg.head_dim])
        k = ttnn.rms_norm(k, epsilon=RMS_NORM_EPS, weight=self.k_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(k, [0, 0, 0, 0], [1, self.local_num_key_value_heads, seq_len, self.cfg.head_dim], [1, 1, 1, 1])

        if kv_cache is not None:
            if page_table is None:
                raise ValueError("page_table is required when prefill_forward fills paged kv_cache")
            self._fill_paged_kv_cache(k, v, kv_cache, page_table, user_id=user_id)

        q = ttnn.reshape(q, [1, seq_len, self.local_num_attention_heads, self.cfg.head_dim])
        q = ttnn.rms_norm(q, epsilon=RMS_NORM_EPS, weight=self.q_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(q, [0, 0, 0, 0], [1, self.local_num_attention_heads, seq_len, self.cfg.head_dim], [1, 1, 1, 1])

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(attn, [1, 1, seq_len, self.local_q_width], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _mlp(self, post_norm, *, use_dram_sharded_down: bool = False):
        if use_dram_sharded_down:
            post_norm = ttnn.to_memory_config(post_norm, ttnn.L1_MEMORY_CONFIG)
            gate_up_memcfg, gate_up_program_config = self._gate_up_decode_geometry()
            if gate_up_memcfg is not None:
                post_norm = ttnn.to_memory_config(post_norm, gate_up_memcfg)
        else:
            gate_up_program_config = self._gate_up_prefill_geometry()
        if use_dram_sharded_down and "packed_gate_up_1d_l1_20c_i10_s4" in self.geometry_modes:
            if self.packed_gate_up_proj_weight is None:
                raise RuntimeError("packed gate/up geometry selected but packed_gate_up_proj_weight was not loaded")
            packed = ttnn.matmul(
                post_norm,
                self.packed_gate_up_proj_weight,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=gate_up_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            gate = ttnn.slice(
                packed, [0, 0, 0, 0], [1, 1, post_norm.shape[-2], self.local_intermediate_size], [1, 1, 1, 1]
            )
            up = ttnn.slice(
                packed,
                [0, 0, 0, self.local_intermediate_size],
                [1, 1, post_norm.shape[-2], 2 * self.local_intermediate_size],
                [1, 1, 1, 1],
            )
            gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            gate = ttnn.matmul(
                post_norm,
                self.gate_proj_weight,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=gate_up_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up = ttnn.matmul(
                post_norm,
                self.up_proj_weight,
                transpose_b=True,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=gate_up_program_config,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if use_dram_sharded_down:
            gated = ttnn.to_memory_config(gated, self._decode_dram_input_memory_config(self.local_intermediate_size))
            down_partial = ttnn.matmul(
                gated,
                self.down_proj_weight_dram_sharded,
                dtype=ttnn.bfloat16,
                memory_config=self._decode_dram_output_memory_config(
                    self.local_intermediate_size, self.cfg.hidden_size
                ),
                program_config=self._decode_dram_matmul_program_config(
                    self.local_intermediate_size, self.cfg.hidden_size
                ),
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            return self._all_reduce_hidden(down_partial, use_persistent=True)
        down_partial = ttnn.matmul(
            gated,
            self.down_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._down_prefill_geometry(),
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        return self._all_reduce_hidden(down_partial)

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
        page_table: ttnn.Tensor | None = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        signpost("PERF_MULTICHIP_PREFILL")
        start = time.perf_counter()
        seq_len = hidden_states.shape[-2]
        if seq_len > self.max_seq_len and (position_cos is None or position_sin is None):
            raise ValueError(
                f"prefill seq_len {seq_len} exceeds setup max_seq_len {self.max_seq_len}; "
                "provide matching RoPE tables or rebuild the decoder"
            )
        cos = position_cos if position_cos is not None else self.position_cos
        sin = position_sin if position_sin is not None else self.position_sin
        attn = self._prefill_attention(hidden_states, seq_len, cos, sin, attention_mask, kv_cache, page_table, user_id)
        attn_partial = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attn_out = self._all_reduce_hidden(attn_partial)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=RMS_NORM_EPS,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        output = ttnn.add(
            self._mlp(post_norm), attn_residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_PREFILL_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=elapsed_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def _decode_head_memory_config(self, batch_size: int) -> ttnn.MemoryConfig:
        if batch_size <= 0:
            raise ValueError(f"decode batch_size must be positive, got {batch_size}")
        return ttnn.create_sharded_memory_config(
            shape=(32, self.cfg.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(
                batch_size, self.mesh_device.compute_with_storage_grid_size(), row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_qkv(self, hidden_states, position_cos, position_sin, batch_size):
        decode_head_memcfg = self._decode_head_memory_config(batch_size)
        qkv_input_memcfg, qkv_program_config = self._qkv_decode_geometry()
        if qkv_input_memcfg is not None:
            hidden_states = ttnn.to_memory_config(hidden_states, qkv_input_memcfg)
        qkv = ttnn.matmul(
            hidden_states,
            self.qkv_decode_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_program_config,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch_size, self.local_q_width], [1, 1, 1, 1])
        k = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_q_width],
            [1, 1, batch_size, self.local_q_width + self.local_kv_width],
            [1, 1, 1, 1],
        )
        v = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_q_width + self.local_kv_width],
            [1, 1, batch_size, self.local_qkv_width],
            [1, 1, 1, 1],
        )

        q = ttnn.reshape(q, [1, batch_size, self.local_num_attention_heads, self.cfg.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.rms_norm(q, epsilon=RMS_NORM_EPS, weight=self.q_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(
            q, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(
            q, [0, 0, 0, 0], [1, batch_size, self.local_num_attention_heads, self.cfg.head_dim], [1, 1, 1, 1]
        )
        q = ttnn.to_memory_config(q, decode_head_memcfg)

        k = ttnn.reshape(k, [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=RMS_NORM_EPS, weight=self.k_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(
            k, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            k, [0, 0, 0, 0], [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim], [1, 1, 1, 1]
        )
        k = ttnn.to_memory_config(k, decode_head_memcfg)

        v = ttnn.reshape(v, [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim])
        v = ttnn.to_memory_config(v, decode_head_memcfg)
        return q, k, v

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        signpost("PERF_MULTICHIP_DECODE")
        start = time.perf_counter()
        batch_size = hidden_states.shape[-2]
        if hidden_states.shape[-3] != 1:
            raise ValueError(f"decode expects one logical token per user, got shape {hidden_states.shape}")
        q, k, v = self._decode_qkv(hidden_states, position_cos, position_sin, batch_size)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table)

        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            program_config=self.sdpa_decode_program_config,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sdpa = ttnn.to_memory_config(sdpa, self._decode_head_memory_config(batch_size))
        attn = ttnn.experimental.nlp_concat_heads_decode(sdpa, num_heads=self.local_num_attention_heads)
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, batch_size, self.local_q_width], [1, 1, 1, 1])
        attn_partial = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._wo_decode_geometry(),
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attn_out = self._all_reduce_hidden(attn_partial, use_persistent=True)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=RMS_NORM_EPS,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        output = ttnn.add(
            self._mlp(post_norm, use_dram_sharded_down=True),
            attn_residual,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_DECODE_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=elapsed_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def trace_decode_once(
        self,
        hidden_states: ttnn.Tensor,
        *,
        replay_hidden_states: torch.Tensor | None = None,
        return_capture_output: bool = False,
        **kwargs,
    ) -> tuple[int, ttnn.Tensor] | tuple[int, ttnn.Tensor, list[torch.Tensor] | None]:
        warmup = self.decode_forward(hidden_states, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self.decode_forward(hidden_states, **kwargs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        capture_output = None
        if return_capture_output:
            capture_output = [
                ttnn.to_torch(tensor).to(torch.float32).clone() for tensor in ttnn.get_device_tensors(output)
            ]
        if replay_hidden_states is not None:
            replay_input_host = ttnn.from_torch(
                replay_hidden_states,
                dtype=hidden_states.get_dtype(),
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(replay_input_host, hidden_states)
            ttnn.synchronize_device(self.mesh_device)
        signpost("PERF_MULTICHIP_TRACE_DECODE")
        start = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_TRACE_DECODE_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=elapsed_ms,
        )
        if return_capture_output:
            return trace_id, output, capture_output
        return trace_id, output

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported MultichipDecoder mode: {mode!r}")
