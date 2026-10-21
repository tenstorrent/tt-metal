#!/usr/bin/env python3
"""Scratch, not for merge. md5 of topk_large_indices outputs (values quantised to force ties) per case: "BITS <case> <md5> <ok>"."""
import hashlib
import sys

import torch
import ttnn

CASES = []
for k in (512, 1024, 2048):
    for rows in (1, 7, 640):
        for chunks in (2, 3, 4, 6, 8, 25, 32):
            CASES.append((rows if chunks < 8 else min(rows, 64), chunks * k, k, None))
    for rows, tail in ((7, 100), (64, 600)):
        for chunks in (2, 4, 6):
            CASES.append((rows, chunks * k - tail, k, None))
for k in (2048, 1024):
    for rows in (1, 7):
        for chunks in (33, 64, 65):
            CASES.append((rows, chunks * k, k, None))
CASES.append((1, 250 * 2048, 2048, None))
CASES.append((7, 250 * 2048 - 600, 2048, None))
CASES.append((7, 65 * 2048, 2048, 100000))
CASES.append((7, 64 * 1024, 1024, 50000))
# production shapes: GLM, bounded cache, indexer past 64K, CSA, MSA, prefill
for rows, n, k in (
    (160, 524288, 2048),
    (160, 102400, 2048),
    (640, 73728, 2048),
    (640, 3328, 1024),
    (640, 2560, 1024),
    (640, 8192, 1024),
    (160, 32768, 1024),
    (1280, 8192, 128),
    (1280, 8192, 16),
    (640, 51200, 1536),
):
    CASES.append((rows, n, k, None))
CASES.append((64, 4 * 2048, 2048, 5000))
CASES.append((64, 6 * 1024, 1024, 4000))


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        for rows, n, k, valid in CASES:
            name = f"r{rows}_n{n}_k{k}" + (f"_v{valid}" if valid else "")
            try:
                g = torch.Generator().manual_seed(rows * 7919 + n + k)
                x = (torch.randint(-300, 300, (rows, n), generator=g).float() / 16).to(torch.bfloat16)
                t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                o = None
                for _ in range(2):
                    o = ttnn.to_torch(ttnn.experimental.topk_large_indices(t, k=k, valid_length=valid)).to(torch.int64)
                ref = torch.topk(x[:, : (valid or n)].float(), k, dim=-1).values
                got = torch.gather(x.float(), 1, o.clamp(min=0))
                ok = bool(((got == ref) | (o < 0)).all())
                print(f"BITS {name} {hashlib.md5(o.numpy().tobytes()).hexdigest()[:12]} {ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"BITS {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
