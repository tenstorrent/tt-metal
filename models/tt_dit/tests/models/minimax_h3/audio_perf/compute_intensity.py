"""Does per-op cost depend on how much arithmetic the op does?

This is the assumption the whole fused-band projection rests on. AUDIO_FUSION_PLAN.md sizes the kernel
at the measured ~142 us/op, but a fused band does perhaps 30x the arithmetic per element of the
elementwise add that number came from. If cost tracks FLOPs, the projection collapses; if cost tracks
rows regardless of arithmetic, it holds and the kernel is worth building.

Answering it does not need the kernel. Time ops of very different arithmetic intensity at identical
shape:

    add          1 flop/element
    multiply     1 flop/element
    sin          transcendental, many flops
    snake_beta   multiply + sin + multiply + reciprocal + add, plus a broadcast

If these land within a small factor of each other, arithmetic is free at this scale and the fused op
will cost what any other op costs. If sin is several times add, compute is the binding term and the
band must be sized on FLOPs instead.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

SHAPES = [("narrow s6", 8, 331212), ("wide  folded", 224, 11829)]
ITERS = 10


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        for label, C, rows in SHAPES:
            print(f"\n=== {label}: rows={rows} C={C} ({rows * C} elements) ===")
            torch.manual_seed(0)
            x = torch.randn(2, rows, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            xt = ttnn.to_layout(xd, ttnn.TILE_LAYOUT)
            a = ttnn.from_torch(torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.from_torch(torch.rand(1, 1, C) + 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

            cases = {
                "add       (1 flop)": lambda: ttnn.add(xd, xd),
                "multiply  (1 flop)": lambda: ttnn.multiply(xd, xd),
                "sin       (transcendental)": lambda: ttnn.sin(xt),
                "snake_beta(mul+sin+mul+recip+add)": lambda: ttnn.snake_beta(xt, a, b),
            }
            base = None
            for name, fn in cases.items():
                try:
                    fn()
                    ttnn.synchronize_device(device)
                    ts = []
                    for _ in range(ITERS):
                        s = time.perf_counter()
                        fn()
                        ttnn.synchronize_device(device)
                        ts.append((time.perf_counter() - s) * 1e3)
                    msv = statistics.median(ts)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:<36} FAILED {str(exc).splitlines()[0][:40]}")
                    continue
                if base is None:
                    base = msv
                print(f"  {name:<36} {msv:>8.3f} ms  {msv * 1e6 / (2 * rows):>7.1f} ns/row   x{msv / base:.2f} vs add")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
