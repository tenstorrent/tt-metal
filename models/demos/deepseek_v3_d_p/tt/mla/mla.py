# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG, MLA_SDPA_CONFIG
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


class ttMLA:
    MLA_WEIGHT_NAMES = [
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "wkv_b1",
        "wkv_b2",
        "o_proj",
    ]

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if all 8 MLA weight cache files exist."""
        for name in ttMLA.MLA_WEIGHT_NAMES:
            if not list(cache_path.glob(f"{cache_name_prefix}.{name}*.tensorbin")):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{name}")
                return False
        return True

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        seq_len: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        """Build TTNN cache for MLA weights using device=None (no device copy)."""

        num_heads = config.num_attention_heads
        kv_lora_rank = config.kv_lora_rank
        qk_nope_head_dim = config.qk_nope_head_dim
        v_head_dim = config.v_head_dim

        def _cache_name(name):
            return str(cache_path / f"layer_{layer_idx}.mla.{name}") if cache_path else None

        # q_a_layernorm
        q_a_ln_weight = state_dict["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        ttnn.as_tensor(
            q_a_ln_weight,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cache_name("q_a_layernorm"),
        )

        # kv_a_layernorm
        kv_a_ln_weight = state_dict["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        ttnn.as_tensor(
            kv_a_ln_weight,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cache_name("kv_a_layernorm"),
        )

        # q_a_proj
        q_a_proj = state_dict["q_a_proj.weight"].transpose(-2, -1)
        shard_dims = [None, None]
        shard_dims[tp_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
        ttnn.as_tensor(
            q_a_proj,
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            cache_file_name=_cache_name("q_a_proj"),
        )

        # q_b_proj
        shard_dims[tp_axis] = 1
        shard_dims[sp_axis] = None
        mesh_mapper_q_b_proj = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
        ttnn.as_tensor(
            state_dict["q_b_proj.weight"].transpose(-2, -1),
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=_cache_name("q_b_proj"),
        )

        # kv_a_proj_with_mqa
        shard_dims[tp_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
        ttnn.as_tensor(
            state_dict["kv_a_proj_with_mqa.weight"].transpose(-2, -1),
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            cache_file_name=_cache_name("kv_a_proj_with_mqa"),
        )

        # kv_b_proj - split into wkv_b1 and wkv_b2
        kv_b_proj_weights = state_dict["kv_b_proj.weight"].reshape(
            1, num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )
        torch_weights_k = kv_b_proj_weights[..., :qk_nope_head_dim, :].transpose(-2, -1)
        torch_weights_v = kv_b_proj_weights[..., qk_nope_head_dim:, :]

        shard_dims[tp_axis] = 1
        shard_dims[sp_axis] = None
        ttnn.as_tensor(
            torch_weights_k.transpose(-2, -1),
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=_cache_name("wkv_b1"),
        )
        ttnn.as_tensor(
            torch_weights_v.transpose(-2, -1),
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=_cache_name("wkv_b2"),
        )

        # o_proj
        shard_dims[tp_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
        ttnn.as_tensor(
            state_dict["o_proj.weight"].transpose(-2, -1),
            device=None,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            cache_file_name=_cache_name("o_proj"),
        )

    def __init__(
        self,
        config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        mesh_device: ttnn.MeshDevice,
        layer_idx: int = 0,
        seq_len: int = 1024,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        topology=ttnn.Topology.Linear,
        weight_cache_path: Optional[Path] = None,
    ):
        self.config = config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.is_balanced = is_balanced
        self.weight_cache_path = weight_cache_path

        self.sp_axis = sp_axis
        self.tp_axis = tp_axis

        # Store per-matmul and SDPA config dicts keyed by local seq_len for runtime lookup
        self.mm_configs = {
            name: MLA_MATMUL_CONFIG.get(name, {})
            for name in [
                "q_a_proj",
                "q_b_proj",
                "wkv_b1",
                "kv_a_proj_with_mqa",
                "wkv_b2",
                "o_proj",
            ]
        }
        self.sdpa_configs = MLA_SDPA_CONFIG

        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        rope_factor = config.rope_scaling["factor"]
        mscale = config.rope_scaling["mscale"]

        self.scale = self.qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale

        self.default_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.hifi4_fp32_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.ring_sdpa_compute_grid = (
            mesh_device.compute_with_storage_grid_size().x - 1,
            mesh_device.compute_with_storage_grid_size().y,
        )

        # Create CCL object for semaphore management
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.tp_factor = mesh_device.shape[self.tp_axis]
        self.sp_factor = mesh_device.shape[self.sp_axis]

        self.ccl_num_links = 2 if is_blackhole() else 1
        self.ccl_topology = topology

        # ring attention setup
        persistent_v_shard_dims = [None, None]
        persistent_v_shard_dims[self.tp_axis] = 1  # TP heads
        persistent_k_shard_dims = [None, None]

        ag_output_shape_k = (1, 1, seq_len, self.kv_lora_rank + self.qk_rope_head_dim)
        ag_output_shape_v = (1, self.num_heads, seq_len, self.v_head_dim)

        self.persistent_k_output_buffer = ttnn.from_torch(
            torch.zeros(ag_output_shape_k),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )

        self.persistent_v_output_buffer = ttnn.from_torch(
            torch.zeros(ag_output_shape_v),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=persistent_v_shard_dims
            ),
        )

        # Pre-allocate dummy joint tensors for ring_joint_scaled_dot_product_attention (seq_len=0)
        num_heads_local = self.num_heads // self.tp_factor
        joint_shard_dims = [None, None]
        joint_shard_dims[self.tp_axis] = 1  # shard on head dimension

        self.joint_q = ttnn.from_torch(
            torch.zeros(1, num_heads_local, 0, self.qk_head_dim),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=joint_shard_dims
            ),
        )

        self.joint_kv = ttnn.from_torch(
            torch.zeros(1, 1, 0, self.kv_lora_rank + self.qk_rope_head_dim),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.joint_v = ttnn.from_torch(
            torch.zeros(1, num_heads_local, 0, self.v_head_dim),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=joint_shard_dims
            ),
        )

        # Load weights to TT device
        self._load_weights(state_dict)

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None:
            return None
        return str(self.weight_cache_path / f"layer_{self.layer_idx}.mla.{name}")

    def _load_weights(self, state_dict: dict[str, torch.Tensor]):
        """
        Load weights from state dict and convert to TT tensors.

        Expected keys in state_dict (when weights provided):
        - q_a_proj.weight, q_a_layernorm.weight
        - q_b_proj.weight
        - kv_a_proj_with_mqa.weight, kv_a_layernorm.weight
        - kv_b_proj.weight
        - o_proj.weight

        If state_dict is empty, loads from cache using dummy tensors.
        """

        # Check if weights are provided (following TtMoe pattern)
        if state_dict and "q_a_layernorm.weight" in state_dict:
            # Weights provided - use direct access (all keys should exist)
            q_a_ln_weight = state_dict["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            kv_a_ln_weight = state_dict["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            q_a_proj = state_dict["q_a_proj.weight"].transpose(-2, -1)
            q_b_proj_weight = state_dict["q_b_proj.weight"].transpose(-2, -1)
            kv_a_proj_weight = state_dict["kv_a_proj_with_mqa.weight"].transpose(-2, -1)
            kv_b_proj_weights = state_dict["kv_b_proj.weight"].reshape(
                1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            o_proj_weight = state_dict["o_proj.weight"].transpose(-2, -1)
        else:
            # Cache-only mode - create dummy tensors with transposed dimensions (ignored when cache exists)
            q_a_ln_weight = torch.empty(1, 1, self.qk_rope_head_dim, ttnn.TILE_SIZE)
            kv_a_ln_weight = torch.empty(1, 1, self.kv_lora_rank, ttnn.TILE_SIZE)
            q_a_proj = torch.empty(self.hidden_size, self.q_lora_rank)
            q_b_proj_weight = torch.empty(self.q_lora_rank, self.qk_rope_head_dim * self.num_heads)
            kv_a_proj_weight = torch.empty(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
            kv_b_proj_weights = torch.empty(
                1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            o_proj_weight = torch.empty(self.num_heads * self.qk_rope_head_dim, self.hidden_size)

        # Mesh Device = (sp x tp)
        # Convert q_a_layernorm
        self.q_a_layernorm_weight = ttnn.as_tensor(
            q_a_ln_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self._cache_name("q_a_layernorm"),
        )

        # Convert kv_a_layernorm
        self.kv_a_layernorm_weight = ttnn.as_tensor(
            kv_a_ln_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=self._cache_name("kv_a_layernorm"),
        )

        # Convert q_a_proj
        shard_dims = [None, None]
        shard_dims[self.tp_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
        )
        self.q_a_proj_weight = ttnn.as_tensor(
            q_a_proj,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=self._cache_name("q_a_proj"),
        )

        # Convert q_b_proj
        shard_dims[self.tp_axis] = 1
        shard_dims[self.sp_axis] = None
        mesh_mapper_q_b_proj = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
        )
        self.q_b_proj_weight = ttnn.as_tensor(
            q_b_proj_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=self._cache_name("q_b_proj"),
        )

        # Convert kv_a_proj_with_mqa
        self.kv_a_proj_with_mqa_weight = ttnn.as_tensor(
            kv_a_proj_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=self._cache_name("kv_a_proj_with_mqa"),
        )

        # Convert kv_b_proj (split into k and v)
        torch_weights_k = kv_b_proj_weights[..., : self.qk_nope_head_dim, :].transpose(-2, -1)
        torch_weights_v = kv_b_proj_weights[..., self.qk_nope_head_dim :, :]

        shard_dims[self.tp_axis] = 1
        shard_dims[self.sp_axis] = None
        self.wkv_b1_weight = ttnn.as_tensor(
            torch_weights_k.transpose(-2, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=self._cache_name("wkv_b1"),
        )
        self.wkv_b2_weight = ttnn.as_tensor(
            torch_weights_v.transpose(-2, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper_q_b_proj,
            cache_file_name=self._cache_name("wkv_b2"),
        )

        # Convert o_proj
        self.o_proj_weight = ttnn.as_tensor(
            o_proj_weight,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=self._cache_name("o_proj"),
        )

        logger.info(f"✓ Loaded {len(state_dict)} weights in MLA layer {self.layer_idx} to TT device")

    @staticmethod
    def kv_cache_to_host(kvpe_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int = 0):
        """Read KVPE cache from device to host tensor [1, 1, seq_total, kv_lora_rank + qk_rope_head_dim]."""
        return ttnn.to_torch(
            kvpe_cache,
            mesh_composer=ttnn.create_mesh_composer(
                mesh_device,
                config=ttnn.MeshComposerConfig(
                    dims=(2, -1),
                    mesh_shape_override=ttnn.MeshShape(
                        mesh_device.shape[sp_axis],  # concat SP shards
                        1,  # collapse TP replicas
                    ),
                ),
            ),
        ).to(torch.bfloat16)

    def get_weight_shapes(self) -> dict[str, tuple]:
        return {
            "q_a_proj.weight": tuple(self.q_a_proj_weight.shape),
            "q_a_layernorm.weight": tuple(self.q_a_layernorm_weight.shape),
            "q_b_proj.weight": tuple(self.q_b_proj_weight.shape),
            "kv_a_proj_with_mqa.weight": tuple(self.kv_a_proj_with_mqa_weight.shape),
            "kv_a_layernorm.weight": tuple(self.kv_a_layernorm_weight.shape),
            "wkv_b1_weight": tuple(self.wkv_b1_weight.shape),
            "wkv_b2_weight": tuple(self.wkv_b2_weight.shape),
            "o_proj.weight": tuple(self.o_proj_weight.shape),
        }

    # Default output dtypes per weight, used when no tuned config exists for the seq_len_local
    MM_DEFAULT_DTYPES = {
        "q_a_proj": ttnn.bfloat16,
        "q_b_proj": ttnn.bfloat16,
        "wkv_b1": ttnn.bfloat16,
        "kv_a_proj_with_mqa": ttnn.bfloat16,
        "wkv_b2": ttnn.bfloat8_b,
        "o_proj": ttnn.bfloat16,
    }

    # Matmul dimensions for batched matmuls (wkv_b1 / wkv_b2) keyed by weight name.
    # Each entry: (K_attr, N_attr) where values are attribute names on self.
    _BATCHED_MM_DIMS = {
        "wkv_b1": ("qk_nope_head_dim", "kv_lora_rank"),
        "wkv_b2": ("kv_lora_rank", "v_head_dim"),
    }

    def _get_mm_kwargs(self, weight_name: str, seq_len_local: int) -> dict:
        """Get matmul kwargs from config, falling back to defaults."""
        cfg = self.mm_configs[weight_name].get(seq_len_local) if is_blackhole() else None
        if cfg is None:
            if weight_name in self._BATCHED_MM_DIMS:
                return self._make_batched_mm_kwargs(weight_name, seq_len_local)
            return {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": self.MM_DEFAULT_DTYPES[weight_name]}
        return {
            "memory_config": cfg["out_mem_config"],
            "program_config": cfg["program_config"],
            "dtype": cfg["out_dtype"],
        }

    def _make_batched_mm_kwargs(self, weight_name: str, seq_len_local: int) -> dict:
        """Build MatmulMultiCoreReuseMultiCast1DProgramConfig for batched matmuls (wkv_b1/wkv_b2).

        These matmuls require fuse_batch=False and mcast_in0=False to support
        batch broadcasting (in0 batch=1, in1 batch=num_heads).
        """
        k_attr, n_attr = self._BATCHED_MM_DIMS[weight_name]
        K_tiles = getattr(self, k_attr) // ttnn.TILE_SIZE
        N_tiles = getattr(self, n_attr) // ttnn.TILE_SIZE
        M_tiles = seq_len_local // ttnn.TILE_SIZE

        num_cores = self.ring_sdpa_compute_grid[0] * self.ring_sdpa_compute_grid[1]
        per_core_M = max(1, -(-M_tiles // num_cores))  # ceil division
        while M_tiles % per_core_M != 0:
            per_core_M += 1

        # out_subblock: h * w <= 8, h divides per_core_M, w divides N_tiles
        out_subblock_w = min(N_tiles, 8)
        while N_tiles % out_subblock_w != 0:
            out_subblock_w -= 1
        out_subblock_h = min(per_core_M, 8 // out_subblock_w)
        while per_core_M % out_subblock_h != 0:
            out_subblock_h -= 1

        # in0_block_w: factor of K_tiles, capped for L1 pressure
        in0_block_w = min(4, K_tiles)
        while K_tiles % in0_block_w != 0:
            in0_block_w -= 1

        return {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "dtype": self.MM_DEFAULT_DTYPES[weight_name],
            "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_M,
                per_core_N=N_tiles,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=False,
            ),
        }

    def _get_sdpa_program_config(self, seq_len_local: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA program config, falling back to default chunk sizes."""
        cfg = self.sdpa_configs.get(seq_len_local)
        q_chunk_size = cfg["q_chunk_size"] if cfg else 32
        k_chunk_size = cfg["k_chunk_size"] if cfg else 32
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

    # Expects ativation in form of:
    # [1, batch_size == 1, seq_len // sp_factor, hidden_size // tp_factor]
    def forward(
        self, hidden_states: ttnn.Tensor, rope_tensors: dict, kvpe_cache: ttnn.Tensor, cache_user_idx: int = 0
    ) -> ttnn.Tensor:
        num_heads_local = self.num_heads // self.tp_factor
        seq_len_local = hidden_states.shape[2]

        # q_projection
        tt_q = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_a_proj", seq_len_local),
        )

        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            tt_q = ttnn.experimental.reduce_scatter_minimal_async(
                tt_q,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
            tt_q = ttnn.experimental.all_gather_async(
                tt_q,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )

        # rmsnorm
        tt_q = ttnn.rms_norm(
            tt_q,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        tt_q = ttnn.linear(
            tt_q,
            self.q_b_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("q_b_proj", seq_len_local),
        )

        # convert to
        # [batch (1), num_heads_local, seq_len_local, qk_head_dim]
        tt_q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            tt_q,
            num_heads=num_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # TODO: split rope and nope, workaround remove with ttnn.narrow or fusion
        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads_local, seq_len_local, self.qk_nope_head_dim])
        tt_q_rope = ttnn.slice(
            tt_q, [0, 0, 0, self.qk_nope_head_dim], [1, num_heads_local, seq_len_local, self.qk_head_dim]
        )
        ttnn.deallocate(tt_q)

        tt_q_nope = ttnn.linear(
            tt_q_nope,
            self.wkv_b1_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b1", seq_len_local),
        )

        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)
        ttnn.deallocate(tt_q_nope)
        ttnn.deallocate(tt_q_rope)

        # kv
        tt_kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("kv_a_proj_with_mqa", seq_len_local),
        )

        # All reduce (skip for single-device TP)
        if self.tp_factor > 1:
            tt_kv = ttnn.experimental.all_gather_async(
                tt_kv,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
            tt_kv = ttnn.experimental.fast_reduce_nc(
                tt_kv, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )

        # TODO: split rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len_local, self.kv_lora_rank])
        tt_kv_rope = ttnn.slice(
            tt_kv, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim]
        )
        ttnn.deallocate(tt_kv)

        tt_kv_nope = ttnn.rms_norm(
            tt_kv_nope,
            weight=self.kv_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # TODO: concat rope and nope, workaround remove with ttnn.narrow or fusion
        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        tt_kvpe = ttnn.typecast(tt_kvpe, dtype=ttnn.bfloat8_b)

        # Update KV cache with compressed latent representation
        ttnn.kv_cache.fill_cache_for_user_(kvpe_cache, tt_kvpe, cache_user_idx)

        tt_v_embedding = ttnn.linear(
            tt_kv_nope,
            self.wkv_b2_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("wkv_b2", seq_len_local),
        )

        attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_kvpe,
            tt_v_embedding,
            self.joint_q,
            self.joint_kv,
            self.joint_v,
            persistent_output_buffer_k=self.persistent_k_output_buffer,
            persistent_output_buffer_v=self.persistent_v_output_buffer,
            joint_strategy="rear",
            logical_n=seq_len_local * self.sp_factor,
            program_config=self._get_sdpa_program_config(seq_len_local),
            compute_kernel_config=self.default_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=self.ccl_num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh_device,
            topology=self.ccl_topology,
            subdevice_id=self.tt_ccl.worker_sub_device_id,
            ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=self.scale,
            is_balanced=self.is_balanced,
        )

        v_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_out = ttnn.linear(
            v_out,
            self.o_proj_weight,
            compute_kernel_config=self.default_compute_kernel_config,
            **self._get_mm_kwargs("o_proj", seq_len_local),
        )
        if self.tp_factor > 1:
            out = ttnn.experimental.reduce_scatter_minimal_async(
                v_out,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_topology,
                cluster_axis=self.tp_axis,
            )
        else:
            out = v_out
        return out
