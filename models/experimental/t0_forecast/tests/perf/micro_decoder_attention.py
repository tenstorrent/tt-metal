# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-3 micro-benchmark gate: the decoder-stream time attention of one layer at the paper shape.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/micro_decoder_attention.py \
        --out models/experimental/t0_forecast/doc/optimize/iter_01/r1_3_micro.json

Split streams (b3): the context stream holds patches 0..62 plus one pad slot (64 per row), the decoder stream the
read-out patches 63 and 64. 16 rows' decoder queries fill one 32-row tile; each block attends over its 16 rows'
context keys (16 x 64) and decoder keys (16 x 2) = 1056 keys, through a constant cross mask (own row only, the context
pad slot masked, patch 63 does not read patch 64).

Both paths run on the same q, k, v (normal, the scale of q and k after QK-norm and RoPE):
    current   SDPA causal on (R, H, 96, 128), the read-out queries 63 and 64 taken from it
    decoder   concat of the context and decoder K and V, SDPA on (H, R / 16, 32, 128) x (.., 1056, 128)
Gate (D2): decoder path <= 2.0 ms per layer, and its PCC against torch >= the current path's.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

R, H, S, D, CTX, LO, N = 1024, 8, 66, 128, 64, 63, 2
BLOCK = 32 // N  # rows per decoder block


def cross_mask() -> torch.Tensor:
    """(32, BLOCK * (CTX + N)) boolean: decoder query (row i, patch LO + t) -> own row's context 0..LO-1 and decoder
    patches <= its own."""
    m = torch.zeros(BLOCK, N, BLOCK, CTX + 0, dtype=torch.bool)
    for i in range(BLOCK):
        m[i, :, i, :LO] = True
    ctx = m.reshape(BLOCK * N, BLOCK * CTX)
    dec = torch.zeros(BLOCK, N, BLOCK, N, dtype=torch.bool)
    for i in range(BLOCK):
        dec[i, :, i] = torch.tril(torch.ones(N, N, dtype=torch.bool))
    return torch.cat([ctx, dec.reshape(BLOCK * N, BLOCK * N)], dim=1)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def time_op(device, fn, batches=5, calls=10):
    import ttnn

    for _ in range(2):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        outs = [fn() for _ in range(calls)]
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) / calls)
        for o in outs:
            ttnn.deallocate(o)
    return statistics.median(samples)


def main():
    import ttnn

    from models.experimental.t0_forecast.tt.attention import sdpa_chunk
    from models.experimental.t0_forecast.tt.common import TtT0Precision

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(R, H, S, D, generator=g) for _ in range(3))
    q, k = q * 1.5, k * 1.5  # scores of a few units, like the trained QK-norm gains
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)[:, :, LO : LO + N]

    device = ttnn.open_device(device_id=0)
    p = TtT0Precision.accuracy()
    kernel = p.compute_kernel_config(fp32_acc=True)
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = {}
    try:
        # Current path: single stream padded to 96.
        pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, 96 - S))
        qc, kc, vc = up(pad(q)), up(pad(k)), up(pad(v))
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=sdpa_chunk(96),
            k_chunk_size=sdpa_chunk(96),
            exp_approx_mode=False,
        )
        current = lambda: ttnn.transformer.scaled_dot_product_attention(
            qc, kc, vc, is_causal=True, scale=1 / math.sqrt(D), program_config=cfg, compute_kernel_config=kernel
        )
        out = ttnn.to_torch(current()).float()[:, :, LO : LO + N]
        result["current"] = {"ms": time_op(device, current) * 1e3, "pcc_vs_torch": pcc(ref, out)}

        # Decoder path: head-major context K/V (patches 0..62 + a zero pad slot) and decoder q/k/v (patches 63, 64).
        def head_major(t, lo, hi, width):
            part = torch.zeros(R, H, width, D)
            part[:, :, : hi - lo] = t[:, :, lo:hi]
            return part.permute(1, 0, 2, 3).reshape(H, R // BLOCK, BLOCK * width, D)

        k_ctx, v_ctx = up(head_major(k, 0, LO, CTX)), up(head_major(v, 0, LO, CTX))
        q_dec, k_dec, v_dec = (up(head_major(t, LO, LO + N, N)) for t in (q, k, v))
        mask = torch.zeros(32, BLOCK * (CTX + N)).masked_fill(~cross_mask(), float("-inf"))
        mask_dev = up(mask.reshape(1, 1, 32, -1))
        keys = BLOCK * (CTX + N)
        cfg_d = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=sdpa_chunk(keys),
            exp_approx_mode=False,
        )

        def decoder():
            kk = ttnn.concat([k_ctx, k_dec], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            vv = ttnn.concat([v_ctx, v_dec], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.transformer.scaled_dot_product_attention(
                q_dec,
                kk,
                vv,
                attn_mask=mask_dev,
                is_causal=False,
                scale=1 / math.sqrt(D),
                program_config=cfg_d,
                compute_kernel_config=kernel,
            )
            ttnn.deallocate(kk)
            ttnn.deallocate(vv)
            return out

        got = ttnn.to_torch(decoder()).float()  # (H, R / BLOCK, BLOCK * N, D) -> (R, H, N, D)
        got = got.reshape(H, R, N, D).permute(1, 0, 2, 3)
        result["decoder"] = {"ms": time_op(device, decoder) * 1e3, "pcc_vs_torch": pcc(ref, got), "keys": keys}
        sdpa_only = lambda: ttnn.transformer.scaled_dot_product_attention(
            q_dec,
            k_ctx,
            v_ctx,
            is_causal=False,
            scale=1 / math.sqrt(D),
            program_config=cfg_d,
            compute_kernel_config=kernel,
        )
        result["decoder_sdpa_without_concat_ms"] = time_op(device, sdpa_only) * 1e3
    finally:
        ttnn.close_device(device)
    result["gate"] = {
        "ms_limit": 2.0,
        "pass": result["decoder"]["ms"] <= 2.0
        and result["decoder"]["pcc_vs_torch"] >= result["current"]["pcc_vs_torch"],
    }
    print(f"[micro] {json.dumps(result)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
