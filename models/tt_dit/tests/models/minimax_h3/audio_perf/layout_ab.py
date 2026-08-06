"""Same op, same data, ROW_MAJOR vs TILE. Closes the confound in compute_intensity.py.

There, sin on a TILE tensor beat add on a ROW_MAJOR one by 5x at C=8 -- but those differ in op *and*
layout, so the gap could be either. The distinction matters a lot: the entire audio stack is
ROW_MAJOR, because `snake_beta` demanded TILE and everything around it stayed RM to avoid converting.
If narrow ROW_MAJOR is intrinsically several times slower than TILE, that is a stack-wide lever and
nobody has tested it.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        for C, rows in ((8, 331212), (224, 11829)):
            x = torch.randn(2, rows, C) * 0.3
            rm = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            tl = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
            print(f"\n=== C={C} rows={rows} ===")
            for name, tensor in (("ROW_MAJOR", rm), ("TILE", tl)):
                for opname in ("add", "sin"):
                    if opname == "add":
                        call = lambda v=tensor: ttnn.add(v, v)
                    else:
                        call = lambda v=tensor: ttnn.sin(v)
                    try:
                        call()
                        ttnn.synchronize_device(device)
                        ts = []
                        for _ in range(10):
                            s = time.perf_counter()
                            call()
                            ttnn.synchronize_device(device)
                            ts.append((time.perf_counter() - s) * 1e3)
                        print(f"  {opname:<5} {name:<10} {statistics.median(ts):>8.3f} ms")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {opname:<5} {name:<10} FAILED {str(exc).splitlines()[0][:44]}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
