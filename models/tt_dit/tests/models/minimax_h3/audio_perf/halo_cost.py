"""What does one halo exchange cost, and how many does the decode pay?

Sharded decode times are nearly independent of chip count -- t_factor=4 measures 0.9469 s and
t_factor=8 measures 0.898 s against a single-chip 0.9304 s. Work that divides by the factor cannot
produce that. A per-conv cost that is the *same* at every factor can, and the halo exchange is exactly
that shape: the decode has ~126 depthwise convs (7 stages x 3 branches x 6 convs) and each one takes a
`_t_neighbor_pad` round trip whose count does not change when chips are added.

This times `_t_neighbor_pad` alone, at the shapes the decode actually uses, with a single synchronize
at the end of N calls so host round-trip is not folded into each sample (the mistake op_floor.py made).

  ~40 us/call  -> halo is not the problem; look elsewhere for the ~500 ms
  ~4 ms/call   -> 126 x 4 ms = ~500 ms, and the whole sharded deficit is accounted for

Run:  T_FACTOR=4 MESH_AXIS=0 python halo_cost.py
"""

import os
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _all_gather_t, _partition_t, _t_neighbor_pad
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

T_FACTOR = int(os.environ.get("T_FACTOR", "4"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "0"))
N = int(os.environ.get("HALO_N", "20"))
REPS = int(os.environ.get("HALO_REPS", "5"))

# (global_T, C) per decoder stage, from row_model.py. Local T is global_T / factor.
STAGES = [
    (1035, 512),
    (5175, 256),
    (10350, 128),
    (20700, 64),
    (41400, 32),
    (82800, 16),
    (165600, 8),
]
# Widest dilation in resblock_dilation_sizes with kernel 11 -> pad 5*5 = 25; kernel 3 dilation 1 -> 1.
PADS = [1, 25]

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    pc = ParallelFactor(factor=T_FACTOR, mesh_axis=MESH_AXIS)
    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    print(f"t_factor={T_FACTOR} axis={MESH_AXIS}  N={N} calls/sample, {REPS} samples", flush=True)
    print(f"\n{'global_T':>9} {'local_T':>8} {'C':>5} {'pad':>4} {'us/call':>9}")
    print("-" * 42)
    total_at_pad = {p: 0.0 for p in PADS}
    for global_T, C in STAGES:
        local_T = global_T // T_FACTOR
        for pad in PADS:
            x = ttnn.from_torch(
                torch.randn(2, local_T, C) * 0.3,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=d,
            )

            def once():
                y = _t_neighbor_pad(
                    x,
                    pad_left=pad,
                    pad_right=pad,
                    parallel_config=pc,
                    ccl_manager=ccl,
                    padding_mode="zeros",
                )
                return y

            once()
            ttnn.synchronize_device(d)
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                for _ in range(N):
                    once()
                ttnn.synchronize_device(d)
                ts.append((time.perf_counter() - t0) * 1e6 / N)
            us = statistics.median(ts)
            total_at_pad[pad] += us
            print(f"{global_T:>9} {local_T:>8} {C:>5} {pad:>4} {us:>9.1f}", flush=True)
            ttnn.deallocate(x)

    print("\nDecode-level projection: ~18 halo-taking convs per stage (3 branches x 6 convs).")
    for pad in PADS:
        per_stage_avg = total_at_pad[pad] / len(STAGES)
        print(
            f"  pad={pad:>2}: mean {per_stage_avg:8.1f} us/call -> 126 calls = "
            f"{126 * per_stage_avg / 1e3:7.1f} ms of halo per decode"
        )

    # The other factor-independent cost: each of the 7 ups all-gathers T to full on every chip, runs
    # unsharded, then re-partitions -- with a TILE/ROW_MAJOR conversion on each side. The gathered
    # tensor is full-T no matter how many chips, so this does not shrink with the factor either.
    print(f"\n{'global_T':>9} {'C':>5} {'gather+partition+4x to_layout us':>34}")
    print("-" * 52)
    ups_total = 0.0
    for global_T, C in STAGES:
        # `_partition_t` slices a TILE-layout tensor, and that requires a tile-aligned begin index, so
        # the gathered T must be a multiple of 32*factor. The real decode gets this from `_upload_BCT`'s
        # padding; a synthetic shape has to round it here, or the slice raises "Can only slice tilized
        # tensor with height begin index aligned to tiles" (measured: it does).
        align = 32 * T_FACTOR
        local_T = (((global_T + align - 1) // align) * align) // T_FACTOR
        x = ttnn.from_torch(
            torch.randn(2, local_T, C) * 0.3, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d
        )

        def ups_roundtrip():
            t = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            g = _all_gather_t(ccl, t, pc)
            g = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
            # the inner conv would run here on full T, on every chip
            t2 = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
            p = _partition_t(t2, pc)
            return ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT)

        ups_roundtrip()
        ttnn.synchronize_device(d)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for _ in range(N):
                ups_roundtrip()
            ttnn.synchronize_device(d)
            ts.append((time.perf_counter() - t0) * 1e6 / N)
        us = statistics.median(ts)
        ups_total += us
        print(f"{global_T:>9} {C:>5} {us:>34.1f}", flush=True)
        ttnn.deallocate(x)
    print(f"\n  7 ups stages total: {ups_total / 1e3:.1f} ms per decode (factor-independent)")

    print("\nCompare: the sharded deficit to account for is ~500 ms (0.947 s measured at factor 4")
    print("against row_model.py's ~441 ms prediction, and the single-chip floor is ~260 ms).")
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
