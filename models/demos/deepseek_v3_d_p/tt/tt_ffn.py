# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of FFN (Feed-Forward Network) module for DeepSeek V3 dense layers.

TtFfn (TP) inherits from TtSharedExpert for weight construction and caching, but
overrides `forward()` with the original simple flow (plain matmuls + ttnn.mul with
fused SiLU + reduce_scatter) so the dense FFN path is decoupled from the sub-device-
aware optimizations applied to TtSharedExpert on this branch.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import matmul_grid
from models.demos.deepseek_v3_d_p.tt.kda.kda import tp_gather_resources
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import (
    ACTIVATION_CLAMPED_SILU_GLU,
    ACTIVATION_SILU,
    ACTIVATION_SITU,
    COMPUTE_KERNEL_CONFIG_HIFI2,
    TtSharedExpert,
    clamped_silu_glu,
    situ_glu,
)

# DeepSeek 671B FFN dimensions
EMB_DIM = 7168
HIDDEN_DIM = 18432

# fuse_tp_collectives: the gathered gate|up matmul's blocking, swept on 8x4 at 640 rows per chip with bf8 weights. The
# K block is the widest that divides the local K shard (a ring needs it); the M block is capped so more rows take more
# blocks, not more L1; 24 mux buffers overflow L1 under K3.1's l1_small and are no faster than 8.
FUSED_GATE_UP_K_BLOCKS = (8, 4, 2, 1)
FUSED_GATE_UP_M_BLOCK = 2
FUSED_GATE_UP_N_BLOCK = 24
FUSED_GATE_UP_SUBBLOCK = (1, 4)
FUSED_GATE_UP_MAX_CORES = 110  # full-height columns, leaving one to the fabric workers
FUSED_GATE_UP_MUX_BUFFERS = 8


class TtFfn(TtSharedExpert):
    """
    FFN module for DeepSeek V3 dense layers.

    Reuses TtSharedExpert's weight construction / caching but applies the original
    simple multi-chip forward flow (no sub-device coupling, no height-sharded
    matmul tuning):

        gate_out = x @ gate_proj
        up_out   = x @ up_proj
        activated = silu(gate_out) * up_out          (fused via ttnn.mul)
                    -- or SiTU-GLU (ttnn.situ_glu) when activation == "situ"
        output_full = activated @ down_proj
        output = reduce_scatter(output_full)         (only when TP > 1)
    """

    def __init__(
        self,
        mesh_device,
        torch_weights: dict = None,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype=ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        activation: str = ACTIVATION_SILU,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        clamped_silu_glu_limit: Optional[float] = None,
        tp_axis: int = 1,
        fuse_tp_collectives: bool = False,
    ):
        """Initialize TtFfn — same signature as before plus the GLU activation and tp_axis, no sub-device
        parameters.

        fuse_tp_collectives: take the TP-sharded input and run the TP gather and reduce-scatter inside the gate|up and
        down matmuls. Where that cannot run, fuses_tp_collectives is False and the caller gathers."""
        super().__init__(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            torch_weights=torch_weights,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            compute_kernel_config=compute_kernel_config,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            clamped_silu_glu_limit=clamped_silu_glu_limit,
            tp_axis=tp_axis,
            # subdevice_id / subdevice_cores intentionally left as defaults (None) —
            # TtFfn's overridden forward() does not use them.
        )
        tp = mesh_device.shape[tp_axis]
        grid = mesh_device.compute_with_storage_grid_size()
        fused_grid = matmul_grid(grid, FUSED_GATE_UP_MAX_CORES)
        # The gather's fabric workers need a core column outside the matmul grid; its buffer is bf16.
        self.fuses_tp_collectives = (
            fuse_tp_collectives
            and tp > 1
            and ttnn.get_fabric_config(mesh_device).name.startswith("FABRIC_2D")
            and fused_grid.x < grid.x
            and activations_dtype == ttnn.bfloat16
        )
        if self.fuses_tp_collectives:
            local_k_tiles = emb_dim // tp // ttnn.TILE_SIZE
            self._fused_k_block = next(b for b in FUSED_GATE_UP_K_BLOCKS if local_k_tiles % b == 0)
            # One gate|up weight per chip; the gathered matmul splits its output back into the two (chunks=2).
            self.gate_up_proj = ttnn.concat([self.gate_proj, self.up_proj], dim=-1)
            ttnn.deallocate(self.gate_proj)
            ttnn.deallocate(self.up_proj)
            self.gate_proj = self.up_proj = None
            self._fused_topology = ttnn.get_usable_topology(self.down_proj, topology, tp_axis)
            self._fused_gate_up_compute_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=self.compute_kernel_config.math_fidelity,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self._fused_grid = fused_grid
            self._fused_gate_up_kwargs = {}  # per row count; see prepare_fused
            self._fused_down_compute_config = ttnn.ComputeConfigDescriptor(
                math_fidelity=self.compute_kernel_config.math_fidelity
            )

    def prepare_fused(self, rows: int) -> dict:
        """The fused gather's arguments for `rows` rows per chip. The first call allocates its buffer and semaphores
        (shared with KDA's fused gather), so callers make it at construction."""
        kwargs = self._fused_gate_up_kwargs.get(rows)
        if kwargs is None:
            grid = self._fused_grid
            links = max(1, min(self.num_links, grid.y // 2))
            kwargs = dict(
                tp_gather_resources(self.mesh_device, rows, self.emb_dim),
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=min(FUSED_GATE_UP_M_BLOCK, -(-rows // ttnn.TILE_SIZE // grid.y)),
                    K_block_size=self._fused_k_block,
                    N_block_size=FUSED_GATE_UP_N_BLOCK,
                    subblock_h=FUSED_GATE_UP_SUBBLOCK[0],
                    subblock_w=FUSED_GATE_UP_SUBBLOCK[1],
                    compute_with_storage_grid_size=grid,
                ),
                topology=self._fused_topology,
                cluster_axis=self.tp_axis,
                num_links=links,
                num_workers_per_link=-(-grid.y // links),
                num_buffers_per_channel=FUSED_GATE_UP_MUX_BUFFERS,
                force_transpose=False,
                use_mcast=True,
                chunks=2,
            )
            self._fused_gate_up_kwargs[rows] = kwargs
        return kwargs

    def _forward_fused(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """TP-sharded input [..., rows, emb_dim / TP] to the TP-sharded output of the same shape."""
        out_shape = list(x.shape)
        tp = self.mesh_device.shape[self.tp_axis]
        assert out_shape[-1] == self.emb_dim // tp, f"expected the TP-sharded width {self.emb_dim // tp}, got {x.shape}"
        rows = x.logical_volume() // out_shape[-1]
        if x.dtype != self.activations_dtype:
            x = ttnn.typecast(x, self.activations_dtype)
        gate_out, up_out = ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=ttnn.reshape(x, (1, rows, self.emb_dim // tp)),
            weight_tensor=self.gate_up_proj,
            compute_kernel_config=self._fused_gate_up_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self.prepare_fused(rows),
        )
        activated = self._activate(gate_out, up_out)
        output = ttnn.experimental.matmul_reduce_scatter_prefill(
            activated,
            self.down_proj,
            cluster_axis=self.tp_axis,
            scatter_dim=-1,
            topology=self._fused_topology,
            num_links=min(self.num_links, 2),
            compute_kernel_config=self._fused_down_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(activated)
        return ttnn.reshape(output, out_shape)

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict,
        mesh_device: ttnn.MeshDevice,
        cache_path: Path,
        cache_name_prefix: str,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        tp_axis: int = 1,
    ):
        """Build TTNN cache for dense FFN (delegates to TtSharedExpert)."""
        TtSharedExpert.build_ttnn_cache(
            torch_weights=torch_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            mesh_device=mesh_device,
            weights_dtype=weights_dtype,
            cache_path=cache_path,
            cache_name_prefix=cache_name_prefix,
            tp_axis=tp_axis,
        )

    def _activate(self, gate_out: ttnn.Tensor, up_out: ttnn.Tensor) -> ttnn.Tensor:
        if self.activation == ACTIVATION_SITU:
            # No sub-device here, so ttnn.situ_glu's L1 fast path is in play -- but it is bounded on
            # the per-chip width, and K3's 33792 is 8448 at TP=4, past the 3072 bound. So the
            # intermediates land in DRAM anyway. Correctness first; #53625 tracks perf.
            return situ_glu(gate_out, up_out, self.situ_beta, self.situ_linear_beta)
        if self.activation == ACTIVATION_CLAMPED_SILU_GLU:
            return clamped_silu_glu(gate_out, up_out, self.clamped_silu_glu_limit)
        return ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Original simple forward pass — plain matmuls, fused SiLU via ttnn.mul, and
        reduce_scatter over the TP axis. Independent of the sub-device-aware path
        in TtSharedExpert.forward().
        """
        if self.fuses_tp_collectives:
            return self._forward_fused(x)
        batch_size = x.shape[0]
        logger.debug(f"Forward pass: input shape={x.shape}, batch_size={batch_size}")

        # Verify input is replicated (full emb_dim) when TP > 1
        if self.mesh_device.shape[self.tp_axis] > 1:
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

        # Step 1: Gate projection
        gate_out = ttnn.matmul(x, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After gate_proj matmul: {gate_out.shape}")

        # Step 2: Up projection
        up_out = ttnn.matmul(x, self.up_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After up_proj matmul: {up_out.shape}")

        # Step 3: GLU activation and element-wise multiplication
        activated = self._activate(gate_out, up_out)
        logger.debug(f"After {self.activation} activation: {activated.shape}")

        # Step 4: Down projection
        output_full = ttnn.matmul(activated, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        logger.debug(f"After down_proj matmul: {output_full.shape}")

        # Step 5: Reduce-scatter output across the TP axis when TP > 1
        if self.mesh_device.shape[self.tp_axis] > 1:
            output = ttnn.reduce_scatter(
                output_full,
                dim=-1,
                cluster_axis=self.tp_axis,
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            output = output_full
        logger.debug(f"After reduce_scatter: {output.shape}")

        return output
