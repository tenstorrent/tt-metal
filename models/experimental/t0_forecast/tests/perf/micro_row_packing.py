# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R2-6 micro-benchmark: what a compact token stream (R * 66 rows, no per-row padding) costs at the time attention.

    ttq run --model t0-gamma --part 3 --kind bench -- python models/experimental/t0_forecast/tests/perf/micro_row_packing.py \
        --out models/experimental/t0_forecast/doc/optimize/iter_02/r2_6_micro.json

The token-wise ops (matmuls, norms) would run on R * 66 instead of R * 96 rows (-31 %); every time attention then needs
the compact qkv (1, 1, R * 66, 3d) as (R, 1, 96, 3d) and its context back. Measured: those two conversions, and one
compact and one padded QKV matmul for scale. Gate: conversions well under the token-wise saving per time layer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

R, S, SP, DM = 1024, 66, 96, 1024


def time_op(device, fn, batches=5, calls=5):
    import ttnn

    for _ in range(2):
        ttnn.deallocate(fn())
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

    from models.experimental.t0_forecast.tt.attention import QKV_BLOCKS
    from models.experimental.t0_forecast.tt.common import TtT0Precision, linear_minimal, upload

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    p = TtT0Precision.accuracy()
    result = {}
    gen = torch.Generator().manual_seed(1)
    try:
        qkv = upload(device, torch.randn(1, 1, R * S, 3 * DM, generator=gen))
        ctx = upload(device, torch.randn(R, 1, SP, DM, generator=gen))
        x_c = upload(device, torch.randn(1, 1, R * S, DM, generator=gen))
        x_p = upload(device, torch.randn(1, 1, R * SP, DM, generator=gen))
        w = upload(device, torch.randn(DM, 3 * DM, generator=gen) / 32)

        def to_padded():
            t = ttnn.reshape(qkv, (R, S, 3 * DM))
            out = ttnn.pad(t, [(0, 0), (0, SP - S), (0, 0)], 0.0)
            return ttnn.reshape(out, (R, 1, SP, 3 * DM))

        def to_compact():
            t = ttnn.slice(ctx, (0, 0, 0, 0), (R, 1, S, DM), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(t, (1, 1, R * S, DM))

        runs = {
            "qkv_to_padded": to_padded,
            "ctx_to_compact": to_compact,
            "qkv_matmul_compact": lambda: linear_minimal(x_c, w, p, role="qkv", blocks=QKV_BLOCKS),
            "qkv_matmul_padded": lambda: linear_minimal(x_p, w, p, role="qkv", blocks=QKV_BLOCKS),
        }
        for name, fn in runs.items():
            try:
                result[name] = {"ms": time_op(device, fn) * 1e3}
            except Exception as e:  # noqa: BLE001
                result[name] = {"error": str(e)[:300]}
            print(name, result[name], flush=True)
    finally:
        ttnn.close_device(device)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
