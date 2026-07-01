# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Shared Expert module with multi-chip sharding and CCL.

This module demonstrates:
- Multi-chip tensor parallelism with proper weight sharding
- Collective communication operations (all-gather, reduce-scatter)
- SiLU activation fusion
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def get_bh_program_configs(per_core_M: int, gate_n_tiles: int, down_n_tiles: int):
    """Program configs for the gate / up / down matmuls on Blackhole."""
    grid = ttnn.CoreCoord(11, 9)
    gate = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=gate_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    )
    up = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=gate_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )
    down = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=down_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )
    return gate, up, down


def get_wh_program_configs(per_core_M: int, gate_n_tiles: int, down_n_tiles: int):
    """Program configs for the gate / up / down matmuls on Wormhole."""
    grid = ttnn.CoreCoord(8, 7)
    gate = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=gate_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    )
    up = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=gate_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )
    down = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=per_core_M,
        per_core_N=down_n_tiles,
        fuse_batch=False,
        mcast_in0=False,
    )
    return gate, up, down


class TtSharedExpert(LightweightModule):
    """
    TTNN implementation of Shared Expert MLP with multi-chip sharding.

    Architecture with multi-chip CCL:
        Input: x [batch, seq_len, emb_dim] (replicated across mesh columns)
        1. gate_out = x @ gate_proj → [batch, seq_len, hidden_dim / num_devices]
        2. up_out = x @ up_proj → [batch, seq_len, hidden_dim / num_devices]
        3. activated = silu(gate_out) * up_out → [batch, seq_len, hidden_dim / num_devices]
        4. output_full = activated @ down_proj → [batch, seq_len, emb_dim]
        5. Reduce-scatter output across mesh columns → [batch, seq_len, emb_dim / num_devices]

    Weight Sharding (across mesh columns):
        - gate_proj, up_proj: Shard on output dimension (-1) across mesh columns
          Shape: [emb_dim, hidden_dim / num_devices]
          mesh_mapper dims=(None, -1)
        - down_proj: Shard on input dimension (-2) across mesh columns
          Shape: [hidden_dim / num_devices, emb_dim]
          mesh_mapper dims=(None, -2)
    """

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if all shared expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if not pattern_exists(f"{cache_name_prefix}.{proj}*.tensorbin", "SharedExpert"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{proj}")
                return False
        return True

    @staticmethod
    def _convert_and_cache_weights(
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
    ):
        """
        Shared logic for converting gate/up/down projections to ttnn with caching.

        Args:
            torch_weights: Dict with 'gate_proj', 'up_proj', 'down_proj' [out_features, in_features]
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension
            mesh_device: Mesh device reference (for mesh_mapper)
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load

        Returns:
            Dict of ttnn.Tensor if device is not None, else None
        """

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        # Prepare post-transpose tensors
        if torch_weights is not None:
            gate_w = torch_weights["gate_proj"].T.contiguous()
            up_w = torch_weights["up_proj"].T.contiguous()
            down_w = torch_weights["down_proj"].T.contiguous()
        else:
            gate_w = torch.empty(emb_dim, hidden_dim)
            up_w = torch.empty(emb_dim, hidden_dim)
            down_w = torch.empty(hidden_dim, emb_dim)

        def _to_ttnn(tensor, dims, name):
            mesh_mapper = ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=mesh_device.shape,
                dims=dims,
            )
            return ttnn.as_tensor(
                tensor,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
                cache_file_name=_cache_name(name),
            )

        gate_tt = _to_ttnn(gate_w, (None, -1), "gate_proj")
        up_tt = _to_ttnn(up_w, (None, -1), "up_proj")
        down_tt = _to_ttnn(down_w, (None, -2), "down_proj")

        if device is None:
            # Cache built, free host tensors
            del gate_tt, up_tt, down_tt
            return None
        else:
            # Return device tensors for __init__
            return {"gate": gate_tt, "up": up_tt, "down": down_tt}

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for shared expert without device copy."""
        TtSharedExpert._convert_and_cache_weights(
            torch_weights, emb_dim, hidden_dim, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

    def __init__(
        self,
        mesh_device,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        subdevice_id: Optional[ttnn.SubDeviceId] = None,
        subdevice_cores: Optional[ttnn.CoreRangeSet] = None,
    ):
        """
        Initialize TtSharedExpert module.

        Args:
            mesh_device: TTNN mesh device
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden dimension (default: 2048)
            torch_weights: Optional dict with keys 'gate_proj', 'up_proj', 'down_proj' containing torch tensors
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            activations_dtype: Data type for activations (default: bfloat16)
            weights_dtype: Data type for weights (default: bfloat8_b)
            compute_kernel_config: Compute kernel configuration
            weight_cache_path: Optional path for caching TTNN weight tensors
            cache_name_prefix: Optional prefix for cache file names
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.subdevice_id = subdevice_id
        self.subdevice_cores = subdevice_cores
        self.weight_cache_path = weight_cache_path
        # Hold the reduce-scatter INPUT alive until the next forward so it is not freed while the
        # async reduce_scatter is still reading it and reused by the concurrent dispatch op (the
        # catastrophic use-after-free). See forward().
        self._rs_input_keepalive = None
        # Shared per-mesh CCL handle. Drives reduce_scatter_minimal_async and owns the shared,
        # stable-address reduce_scatter INTERMEDIATE buffer (one per mesh, reused by all layers'
        # shared experts) — see forward() and TT_CCL.get_shared_rs_intermediate.
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.cache_name_prefix = cache_name_prefix

        logger.debug(f"Initializing TtSharedExpert with emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: num_links={num_links}, topology={topology}")

        # Create sharded weights
        if torch_weights is not None:
            logger.debug("Creating weights from provided torch tensors")
            weights = self._convert_and_cache_weights(
                torch_weights,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        elif weight_cache_path is not None:
            logger.debug("Loading weights from cache")
            weights = self._convert_and_cache_weights(
                None,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        else:
            logger.debug("Creating random sharded weights")
            self.gate_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="gate_proj", dtype=self.weights_dtype
            )
            self.up_proj = self._create_random_sharded_weight(
                shape=(emb_dim, hidden_dim), dims=(None, -1), name="up_proj", dtype=self.weights_dtype
            )
            self.down_proj = self._create_random_sharded_weight(
                shape=(hidden_dim, emb_dim), dims=(None, -2), name="down_proj", dtype=self.weights_dtype
            )

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor in TTNN format [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper (e.g., (None, -1) or (-2, None))
            name: Weight name for logging
            dtype: Data type for the weight tensor

        Returns:
            Sharded ttnn tensor
        """
        logger.debug(f"Creating sharded weight {name} with dims={dims}, shape={torch_weight.shape}")

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=dims,
        )

        tt_weight = ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._cache_name(name),
        )

        logger.debug(f"Created {name}: {tt_weight.shape}")
        return tt_weight

    def _create_sharded_weight_from_torch(
        self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType
    ) -> ttnn.Tensor:
        """
        Convert HuggingFace torch weight to sharded ttnn tensor.

        HF/PyTorch nn.Linear weights are [out_features, in_features], but TTNN matmul(x, W)
        expects [in_features, out_features], so we transpose before sharding.
        """
        torch_weight = torch_weight.T.contiguous()
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def _create_random_sharded_weight(self, shape: tuple, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Create random sharded weight in TTNN format [in_features, out_features].
        """
        torch_weight = torch.randn(*shape, dtype=torch.float32)
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with multi-chip sharding and CCL.

        Args:
            x: Input tensor [batch, seq_len, emb_dim] (replicated across mesh columns)

        Returns:
            Output tensor [batch, seq_len, emb_dim / num_devices]
        """
        batch_size = x.shape[0]
        logger.debug(f"Forward pass: input shape={x.shape}, batch_size={batch_size}")

        # Verify input is replicated (full emb_dim) when multiple mesh columns
        if self.mesh_device.shape[1] > 1:
            assert x.shape[-1] == self.emb_dim, (
                f"Input must be replicated (full emb_dim={self.emb_dim}), "
                f"but got sharded input with shape[-1]={x.shape[-1]}"
            )

        # Convert input to activations dtype if needed
        if x.dtype != self.activations_dtype:
            logger.warning(f"{x.dtype=} typecasting {self.activations_dtype}")
            x = ttnn.typecast(x, self.activations_dtype)

        assert (
            x.shape[-1] == self.gate_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != gate_proj[-2]={self.gate_proj.shape[-2]}"
        assert (
            x.shape[-1] == self.up_proj.shape[-2]
        ), f"Matmul shape mismatch: x[-1]={x.shape[-1]} != up_proj[-2]={self.up_proj.shape[-2]}"
        assert (
            self.gate_proj.shape[-1] == self.down_proj.shape[-2]
        ), f"Matmul shape mismatch: gate_proj[-1]={self.gate_proj.shape[-1]} != down_proj[-2]={self.down_proj.shape[-2]}"

        # ===== Inlined shared expert FFN — height-sharded sub-device matmuls =====
        # Available compute grid: BH = 11x9, WH = 8x7.
        TILE = 32
        max_cores = 11 * 9 if is_blackhole() else 8 * 7

        # Pick the largest divisor of M_tiles that fits in the sub-device — gives
        # max parallelism with no padding waste.
        m_tiles = x.padded_shape[-2] // TILE
        num_cores = m_tiles
        while num_cores > max_cores or m_tiles % num_cores != 0:
            num_cores -= 1
        per_core_M = m_tiles // num_cores

        gate_n_tiles = self.gate_proj.padded_shape[-1] // TILE
        down_n_tiles = self.down_proj.padded_shape[-1] // TILE
        if is_blackhole():
            gate_program_config, up_program_config, down_program_config = get_bh_program_configs(
                per_core_M, gate_n_tiles, down_n_tiles
            )
        else:
            gate_program_config, up_program_config, down_program_config = get_wh_program_configs(
                per_core_M, gate_n_tiles, down_n_tiles
            )

        # 1) Compute gate and up projections
        gate_out = ttnn.matmul(
            x,
            self.gate_proj,
            program_config=gate_program_config,
            compute_kernel_config=self.compute_kernel_config,
            sub_device_id=self.subdevice_id,
        )
        up_out = ttnn.matmul(
            x,
            self.up_proj,
            program_config=up_program_config,
            compute_kernel_config=self.compute_kernel_config,
            sub_device_id=self.subdevice_id,
        )

        # 2) Multiply gate and up projection
        ttnn.multiply_(gate_out, up_out, sub_core_grids=self.subdevice_cores)
        ttnn.deallocate(up_out)

        # 3) Compute down projection
        output_full = ttnn.matmul(
            gate_out,
            self.down_proj,
            program_config=down_program_config,
            compute_kernel_config=self.compute_kernel_config,
            sub_device_id=self.subdevice_id,
        )
        ttnn.deallocate(gate_out)

        # 4) Reduce-scatter across mesh columns when TP > 1.
        if self.mesh_device.shape[1] > 1:
            # This reduce_scatter runs overlapped with the MoE dispatch op (on a separate
            # sub-device, overlap_shared_expert_with_dispatch). The op is async: its kernels keep
            # accessing its DRAM buffers after this Python call returns, while the concurrent
            # dispatch allocates its own (large) buffers. Any buffer this op frees before its
            # kernels finish can be re-handed to dispatch and overwritten mid-flight — corrupting
            # the result non-deterministically. Both the reduce_scatter INPUT and its INTERMEDIATE
            # accumulator were observed (via DRAM address tracing) re-handed to the dispatch op's
            # `metadata` / `dispatched_buffer` at the exact same address: the input alias produced
            # the catastrophic period-2 failure, the intermediate alias the residual non-determinism.
            #
            # Own the intermediate (reduce_scatter_minimal_async with an explicit persistent buffer),
            # reusing one shared, stable-address buffer owned by tt_ccl (shared across all layers,
            # which run sequentially). This keeps it alive across the overlap (so dispatch can't reuse
            # its slot) AND fixes its DRAM address every iteration, so the op's fabric reduction order
            # is identical each iteration — giving bit-exact determinism. The op overwrites the
            # intermediate before reading it (the line-reduction compute skips the first slice), so no
            # per-iteration re-zeroing is needed.
            rs_intermediate = self.tt_ccl.get_shared_rs_intermediate(output_full)
            output = ttnn.experimental.reduce_scatter_minimal_async(
                output_full,
                persistent_output_buffers=[rs_intermediate],
                dim=-1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                num_links=self.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.topology,
                cluster_axis=1,
                subdevice_id=self.subdevice_id,
            )
            # Keep the (fresh-per-iter) RS input alive until the next forward so the concurrent
            # dispatch cannot reuse its slot mid-flight. (The intermediate is kept alive by the cache.)
            self._rs_input_keepalive = output_full
        else:
            output = output_full
        logger.debug(f"After shared_expert_ffn: {output.shape}")

        return output
