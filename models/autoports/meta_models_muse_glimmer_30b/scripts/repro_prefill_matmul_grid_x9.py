# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Repro: 2D multicast matmul + DRAM width-sharded weights silently returns NaN.

Found while optimizing the Muse-Glimmer-30B decoder prefill path.

``MatmulMultiCoreReuseMultiCastProgramConfig`` (the 2D multicast matmul used for large
prefill matmuls) silently produces **NaN** when its ``in1`` is DRAM *width-sharded* and
``per_core_N`` is not exactly the in1 DRAM shard width in tiles, i.e. unless
``grid_x == num_dram_banks``. With a DRAM-interleaved ``in1`` every grid is correct.

The layer's QKV projection is ``[b, 1, seq, 6656] x [6656, 4608]``. 4608 = 144 tiles, so
a 9-wide compute grid divides the tiled N *exactly* (``per_core_N = 16``) and is a
legal-looking way to use 72 of the 110 Blackhole cores instead of 64 - but the weights
are DRAM width-sharded over the 8 Blackhole DRAM banks (18 tiles per bank), so 9 wide
returns NaN. Nothing is raised. The failure was silent enough to survive a whole-layer
wall-clock sweep looking ~5%% *faster* than the legal 8-wide grid before a PCC test
caught it: whole-layer prefill PCC 0.765, paged K cache entirely NaN, at every sequence
length.

The table below isolates it: the same shapes, program configs and dtypes pass with an
interleaved ``in1`` and fail with a DRAM width-sharded ``in1`` for every
``grid_x != 8``, including ``grid_x = 9`` where ``per_core_N * grid_x`` covers N exactly.

Secondary observation: ``per_core_N * grid_x > N_tiles`` (``grid_x = 10`` ->
``per_core_N = 15``, ``150 > 144``) is also accepted rather than rejected; with an
interleaved ``in1`` it happens to be correct.

Both deserve a validation check in the matmul program-config validator: the DRAM-sharded
one is a correctness bug, the over-provisioned one is a missing guard.

``OptimizedDecoder._prefill_grid`` works around this by pinning ``grid_x`` to the DRAM
bank count whenever the weights are DRAM width-sharded.

Run::

    python models/autoports/meta_models_muse_glimmer_30b/scripts/repro_prefill_matmul_grid_x9.py
"""

from __future__ import annotations

import math

import torch

import ttnn

TILE = 32
K, N = 6656, 4608  # the Muse-Glimmer QKV projection
MAX_ROWS = 8192


def _largest(value: int, cap: int) -> int:
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _pcc(golden: torch.Tensor, got: torch.Tensor) -> float:
    if not torch.isfinite(got).all():
        return float("nan")
    a = golden.reshape(-1).to(torch.float64)
    b = got.reshape(-1).to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> int:
    torch.manual_seed(0)
    activation = torch.randn(1, 1, MAX_ROWS, K, dtype=torch.float32) * 0.05
    weight = torch.randn(K, N, dtype=torch.float32) * 0.02
    golden = activation.reshape(MAX_ROWS, K) @ weight

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        dram = mesh.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
        dram_sharded_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (K, N // dram.x), ttnn.ShardOrientation.ROW_MAJOR),
        )
        weights = {}
        for name, memcfg in (("interleaved", ttnn.DRAM_MEMORY_CONFIG), ("dram_sharded", dram_sharded_cfg)):
            host = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            weights[name] = ttnn.to_device(host, mesh, memory_config=memcfg)
        print(f"DRAM banks={dram.x}, in1 shard width={N // dram.x} elements = {N // dram.x // TILE} tiles")
        kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        print(f"K={K} N={N}  (N_tiles={N // TILE}, K_tiles={K // TILE})")
        print(
            f"{'in1 / input shape':>28}  {'grid':>8}  {'per_core_N':>10}  "
            f"{'subblock_w':>10}  {'covers N':>9}  {'PCC':>10}"
        )
        for w_kind, batch_dim, m_rows in (
            ("interleaved", 1, 512),
            ("dram_sharded", 1, 32),
            ("dram_sharded", 1, 512),
            ("dram_sharded", 16, 512),
        ):
            w_tt = weights[w_kind]
            rows = batch_dim * m_rows
            act_tt = ttnn.from_torch(
                activation[:, :, :rows, :].reshape(1, batch_dim, m_rows, K),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for grid_x, subblock_cap in ((8, 8), (9, 8), (10, 8), (11, 8)):
                grid_y = _largest(m_rows // TILE, 8)
                per_core_n = math.ceil((N // TILE) / grid_x)
                per_core_m = math.ceil((m_rows // TILE) / grid_y)
                out_subblock_w = max(w for w in range(1, subblock_cap + 1) if per_core_n % w == 0)
                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(grid_x, grid_y),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=out_subblock_w,
                    per_core_M=per_core_m,
                    per_core_N=per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                out = ttnn.linear(
                    act_tt,
                    w_tt,
                    program_config=program_config,
                    compute_kernel_config=kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                got = ttnn.to_torch(out).to(torch.float32).reshape(rows, N)
                out.deallocate(True)
                covers = "exact" if per_core_n * grid_x == N // TILE else "OVER"
                shape = f"{w_kind}[1,{batch_dim},{m_rows}]"
                pcc = _pcc(golden[:rows], got)
                print(
                    f"{shape:>28}  {grid_x:>3}x{grid_y:<4}  {per_core_n:>10}  "
                    f"{out_subblock_w:>10}  {covers:>9}  {pcc:>10.6f}"
                )
            act_tt.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
