# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B SwiGLU MLP for BH GLX 8×4 mesh — TP-sharded over rows.

Computes:  down_proj(silu(gate_proj(x)) * up_proj(x))

Parallelization plan (8 mesh rows = cluster_axis=0, TP factor 8)
----------------------------------------------------------------
gate / up (column-parallel — shard OUTPUT dim across rows):
    w_gate, w_up: [H=5120, I_per_row=2176] per row  (I=17408 / 8)
    out: [B, T, 2176] per row  (each row owns a slice of the intermediate dim)

down (row-parallel — shard INPUT dim across rows):
    w_down: [I_per_row=2176, H=5120] per row
    Each row's matmul produces [B, T, H] PARTIAL.  All-gather over rows
    (dim=0, cluster_axis=0) + fast_reduce_nc(dims=[0]) sums them.

Across mesh cols (cluster_axis=1) the weights are REPLICATED — same shard on
each of the 4 cols.  That's correct: MLP doesn't TP over cols here (the norms
TP over cols, which is orthogonal).

Precedent: ``models/demos/qwen3_6_galaxy/tt/qwen36_deltanet.py::_output_proj_and_reduce``
uses the same all_gather + fast_reduce_nc pattern.

Per-chip weight DRAM (BF16, after sharding /8):
    gate:  5120 × 2176 × 2 = 22.3 MB
    up:    5120 × 2176 × 2 = 22.3 MB
    down:  2176 × 5120 × 2 = 22.3 MB
    Total per layer per chip: ~67 MB  (vs 534 MB if replicated)

DRAM input contract: ``forward(x)`` takes ``x`` REPLICATED [B, T, H] and returns
[B, T, H] REPLICATED.  Identical I/O contract to the previous (replicated) MLP,
so the calling decoder layer needs no change.
"""
from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36MLP(LightweightModule):
    """SwiGLU MLP for Qwen3.6-27B on BH GLX 8×4 mesh, TP-sharded over rows.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 mesh.
    args : TtQwen36ModelArgs
        Provides ``cluster_shape`` ([8,4]) and ``intermediate_dim_per_tp_native``
        (= 2176 = 17408/8) used for sharding metadata.
    state_dict : dict
        Weight dict with keys (any of these prefixes):
          mlp.gate_proj.weight  [intermediate, H]
          mlp.up_proj.weight    [intermediate, H]
          mlp.down_proj.weight  [H, intermediate]
    dtype : ttnn.DataType
        Weight dtype (default bfloat16).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args,
        state_dict: dict,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype
        self.cluster_shape = list(args.cluster_shape)  # [8, 4]
        assert self.cluster_shape == [8, 4], f"TtQwen36MLP expects 8×4 cluster shape; got {self.cluster_shape}"

        # TP across rows: 8 rows hold disjoint slices of the intermediate dim.
        self.n_rows = self.cluster_shape[0]  # 8
        self.n_cols = self.cluster_shape[1]  # 4
        self.intermediate_per_row = args.intermediate_dim // self.n_rows  # 17408/8 = 2176

        # Compute kernel: HiFi4 + fp32 accumulation for accurate MLP linears.
        # The reduce_scatter/all_gather across rows accumulates 8 BF16 partials, so
        # higher input fidelity helps preserve PCC > 0.99 on real activations.
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._build_weights(state_dict)

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(self, sd: dict):
        """Upload gate, up, down weights with row-sharding for TP.

        Convention: ttnn.linear(x[..., K], W[K, N]) = out[..., N].
        We store weights as [K, N] (input_dim first).

        Sharding via ``ShardTensor2dMesh(dims=(d_row, d_col))``:
          - d_row: which torch dim is split across the 8 mesh rows
          - d_col: which torch dim is split across the 4 mesh cols (None = replicate)
        """
        gate_w = sd["mlp.gate_proj.weight"]  # [I=17408, H=5120]
        up_w = sd["mlp.up_proj.weight"]  # [I, H]
        down_w = sd["mlp.down_proj.weight"]  # [H, I]

        # ---- gate / up: column-parallel ----
        # Transpose to [H, I] then shard I (dim=1) across rows; cols replicate.
        gate_w_T = gate_w.T.contiguous()  # [H=5120, I=17408]
        up_w_T = up_w.T.contiguous()  # [H, I]

        # ShardTensor2dMesh(dims=(d_row, d_col)) — d_row is split across the 8 mesh
        # rows, d_col across the 4 cols.  For column-parallel we shard the output
        # dim (=I) across rows and replicate across cols.
        col_parallel = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(1, None), mesh_shape=self.cluster_shape)

        self.w_gate = ttnn.from_torch(
            gate_w_T,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_parallel,
        )  # per-chip: [H, I/8] = [5120, 2176]
        self.w_up = ttnn.from_torch(
            up_w_T,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_parallel,
        )

        # ---- down: row-parallel (matches DeltaNet out_proj pattern) ----
        # Transpose [H, I] → [I, H], shard I (dim=0) across rows.
        down_w_T = down_w.T.contiguous()  # [I=17408, H=5120]
        row_parallel = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)
        self.w_down = ttnn.from_torch(
            down_w_T,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_parallel,
        )  # per-chip: [I/8, H] = [2176, 5120]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Compute SwiGLU MLP forward.

        Args:
            x: [B, T, H] bfloat16, REPLICATED across mesh.

        Returns:
            [B, T, H] bfloat16, REPLICATED across mesh (after all_gather + reduce).
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        ck = self.compute_kernel

        # gate_proj(x): [B,T,H] @ per-row[H, I/8] → [B,T,I/8] per row
        gate_out = ttnn.linear(x, self.w_gate, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        gate_act = ttnn.silu(gate_out, memory_config=mem)
        gate_out.deallocate(True)

        # up_proj(x): [B,T,H] @ per-row[H, I/8] → [B,T,I/8] per row
        up_out = ttnn.linear(x, self.w_up, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)

        # Element-wise multiply: SwiGLU
        ff = ttnn.multiply(gate_act, up_out, memory_config=mem)
        gate_act.deallocate(True)
        up_out.deallocate(True)

        # down_proj(ff): per-row[B,T,I/8] @ per-row[I/8, H] → [B,T,H] PARTIAL per row
        partial = ttnn.linear(ff, self.w_down, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        ff.deallocate(True)

        # All-reduce over rows: gather + sum (same pattern as DeltaNet._output_proj_and_reduce)
        gathered = ttnn.all_gather(
            partial,
            dim=0,
            num_links=1,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            memory_config=mem,
        )  # stacks 8 partials on dim=0
        partial.deallocate(True)

        reduced = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[0],
            output=None,
            compute_kernel_config=None,
            memory_config=mem,
        )  # [B, T, H] replicated
        gathered.deallocate(True)

        return reduced
