"""Does `_t_neighbor_pad` return correct data, and is conv_pre's input already wrong?

stage_bisect.py put the first divergence at `conv_pre` -- the first conv in the graph -- at -191 dB,
i.e. the sharded values are ~1e9 times the reference magnitude. That is the signature of uninitialized
memory or a read that races the CCL, not of padding arithmetic being subtly off. halo_cost.py timed
`_t_neighbor_pad` but never checked what it produced, so that is the gap to close.

Two checks, cheapest first.

1. **Halo correctness.** Partition a ramp whose value *is* its global row index, halo it, and verify
   every shard sees exactly `[start - pad, start + local_T + pad)` with zeros clamped at the global
   edges. Any other content localizes the bug to the halo primitive and explains everything downstream.

2. **conv_pre's input.** `_forward_device` does to_layout(TILE) -> _partition_t -> to_layout(ROW_MAJOR)
   -> partition_channel before conv_pre. If the reassembled sharded input already differs from the
   unsharded one, the bug is in that prologue rather than in the conv.
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _partition_t, _t_neighbor_pad
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

FACTOR = int(os.environ.get("T_FACTOR", "8"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "1"))
T = int(os.environ.get("NUM_T", "256"))
C = int(os.environ.get("NUM_C", "2048"))
PADS = [int(x) for x in os.environ.get("PADS", "3,1,25").split(",")]

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    pc = ParallelFactor(factor=FACTOR, mesh_axis=MESH_AXIS)
    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    per = T // FACTOR
    print(f"factor={FACTOR} axis={MESH_AXIS} T={T} C={C} local_T={per}\n", flush=True)

    # Value == global row index, so any row's provenance is readable straight off the tensor.
    ramp = torch.arange(T, dtype=torch.float32).reshape(1, T, 1).expand(2, T, C).contiguous()
    full = ttnn.from_torch(ramp, device=d, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    tiled = ttnn.to_layout(full, ttnn.TILE_LAYOUT)
    parted = ttnn.to_layout(_partition_t(tiled, pc), ttnn.ROW_MAJOR_LAYOUT)

    # First: did the partition itself land the right rows on each device?
    print("partition check (each shard's first/last row value vs expected global index):")
    part_ok = True
    for s in range(FACTOR):
        got = ttnn.to_torch(ttnn.get_device_tensors(parted)[s]).float()
        lo, hi = float(got[0, 0, 0]), float(got[0, -1, 0])
        exp_lo, exp_hi = s * per, s * per + per - 1
        ok = abs(lo - exp_lo) < 0.5 and abs(hi - exp_hi) < 0.5
        part_ok &= ok
        print(f"  shard {s}: rows {lo:8.1f}..{hi:8.1f}  expected {exp_lo:5d}..{exp_hi:5d}  {'OK' if ok else 'WRONG'}")
    print(f"  partition: {'OK' if part_ok else 'BROKEN'}\n", flush=True)

    for pad in PADS:
        haloed = _t_neighbor_pad(
            parted,
            pad_left=pad,
            pad_right=pad,
            parallel_config=pc,
            ccl_manager=ccl,
            padding_mode="zeros",
        )
        print(f"halo check, pad={pad} (expect local_T + 2*pad = {per + 2 * pad} rows):")
        all_ok = True
        for s in range(FACTOR):
            got = ttnn.to_torch(ttnn.get_device_tensors(haloed)[s]).float()[0, :, 0]
            rows = got.shape[0]
            start = s * per
            expected = torch.zeros(per + 2 * pad)
            for j in range(per + 2 * pad):
                g = start - pad + j
                expected[j] = float(g) if 0 <= g < T else 0.0
            if rows != expected.shape[0]:
                print(f"  shard {s}: got {rows} rows, expected {expected.shape[0]} -- shape wrong")
                all_ok = False
                continue
            bad = (got - expected).abs() > 0.5
            nbad = int(bad.sum())
            all_ok &= nbad == 0
            if nbad:
                idx = torch.nonzero(bad).flatten()[:6].tolist()
                print(
                    f"  shard {s}: {nbad}/{rows} rows wrong; first bad at {idx} "
                    f"got {[round(float(got[i]), 1) for i in idx]} expected {[float(expected[i]) for i in idx]}"
                )
            else:
                print(f"  shard {s}: all {rows} rows correct")
        print(f"  halo pad={pad}: {'OK' if all_ok else 'BROKEN'}\n", flush=True)
        ttnn.deallocate(haloed)
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
