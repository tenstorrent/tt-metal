#!/usr/bin/env python3
"""Does the already-implemented ``chunked`` Gumbel mode fix the canvas-position axis? (#48291)

``sample_gumbel_noise_by_vocab_chunks`` draws each vocab chunk with a DISTINCT seed. If that
also decorrelates the canvas-position axis, the fix is a mode that already exists and is already
wired (``GUMBEL_MODES`` includes ``chunked``, serving default chunk size 1024) rather than new
code. Measured on the same metrics as the gate: exact duplicates and flat-logit winner
multiplicity across the 256 canvas positions, against the host IID control.
"""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.environ.get("TT_METAL_HOME", "/home/zni/tt-metal"))

import ttnn  # noqa: E402
from models.experimental.diffusion_gemma.tt import sampling as TS  # noqa: E402

CANVAS = 256
VOCAB = int(os.environ.get("PROBE_VOCAB", "16384"))
SEED = 48291


def report(positions: torch.Tensor, label: str) -> None:
    n = positions.shape[0]
    seen, owner = {}, [-1] * n
    for i in range(n):
        key = hash(positions[i].numpy().tobytes())
        if key in seen:
            owner[i] = seen[key]
        else:
            seen[key] = i
    dups = [i for i in range(n) if owner[i] >= 0]
    winners = positions.argmax(dim=-1)
    counts = torch.bincount(winners)
    corr = torch.nan_to_num(torch.corrcoef(positions.double()), nan=0.0)
    off = corr[~torch.eye(n, dtype=torch.bool)].abs()
    print(
        f"[{label}] unique={len(seen)}/{n} duplicated={len(dups)} "
        f"offsets={Counter(i - owner[i] for i in dups).most_common(2)} "
        f"distinct_winners={int((counts > 0).sum())}/{n} max_mult={int(counts.max())} "
        f"max|r|={float(off.max()):.5f} mean|r|={float(off.mean()):.5f}"
    )


def main() -> int:
    device = ttnn.open_device(device_id=0)
    try:
        print(f"=== chunked-vocab Gumbel arms (canvas={CANVAS}, vocab={VOCAB}) ===")
        for chunk in (1024, 2048):
            noise = TS.sample_gumbel_noise_by_vocab_chunks(
                (1, 1, CANVAS, VOCAB), device=device, seed=SEED, vocab_chunk_size=chunk
            )
            host = ttnn.to_torch(noise).float().reshape(-1, VOCAB)[:CANVAS]
            noise.deallocate(True)
            report(host, f"chunked(chunk={chunk}, {VOCAB // chunk} calls)")

        noise = TS.sample_gumbel_noise_with_permuted_vocab((1, 1, CANVAS, VOCAB), device=device, seed=SEED)
        host = ttnn.to_torch(noise).float().reshape(-1, VOCAB)[:CANVAS]
        noise.deallocate(True)
        report(host, "permuted (production default)")

        generator = torch.Generator().manual_seed(SEED)
        uniform = torch.rand((CANVAS, VOCAB), generator=generator, dtype=torch.float32)
        report(-torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float32).tiny))), "host-IID")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
