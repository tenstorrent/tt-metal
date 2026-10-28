# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""G2-r2 item 1: fused SwiGLU (minimal_matmul, fuse_swiglu=True) against the R1-1 path (w1 + silu, w2, multiply),
at paper (M = 4096 x 64 rows), one job.

    FLOWSTATE_SWEEP_JSON=out.json pytest -q -s models/experimental/flowstate_forecast/tests/perf/sweep_swiglu.py

Per config: synchronized wall ms (median of 5 after 2 warm-ups) and the max abs difference to the R1-1 output."""

import itertools
import json
import os

import pytest
import torch
import ttnn

from models.experimental.flowstate_forecast.tests.perf.check_gates import session_stamp
from models.experimental.flowstate_forecast.tests.perf.sweep_matmul import _time
from models.experimental.flowstate_forecast.tt import common
from models.experimental.flowstate_forecast.tt.common import policy, swiglu_interleave

STAMP = session_stamp()  # v1.23: taken at import, the session start (D5 item 5)

M = 4096 * 64
K, N = 512, 1024


def configs():
    yield "auto", None
    for mb, kb, nb, (sh, sw) in itertools.product((4, 8, 16), (4, 8, 16), (4, 8, 16), ((2, 2), (1, 4), (2, 4), (4, 2))):
        if mb % sh or nb % sw or (K // 32) % kb:
            continue
        yield f"m{mb}_k{kb}_n{nb}_sb{sh}x{sw}", ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=(11, 10),
        )


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_sweep_swiglu(device, precision):
    p = policy(device, precision)
    wd = p.dense_weight_dtype
    g = torch.Generator().manual_seed(0)
    put = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    a = put(torch.randn(1, 1, M, K, generator=g), ttnn.bfloat16)
    w1, w2 = torch.randn(K, N, generator=g) / K**0.5, torch.randn(K, N, generator=g) / K**0.5
    b1, b2 = torch.randn(1, N, generator=g), torch.randn(1, N, generator=g)
    tw1, tw2, tb1, tb2 = put(w1, wd), put(w2, wd), put(b1, wd), put(b2, wd)
    w12, b12 = put(swiglu_interleave(w1, w2), wd), put(swiglu_interleave(b1, b2), wd)

    def base():
        x = ttnn.linear(
            a,
            tw1,
            bias=tb1,
            program_config=common.mm_config(device, "w1", precision, M, K, N, "silu"),
            compute_kernel_config=p.dense,
            memory_config=p.mem,
        )
        y = ttnn.linear(
            a,
            tw2,
            bias=tb2,
            program_config=common.mm_config(device, "w2", precision, M, K, N),
            compute_kernel_config=p.dense,
            memory_config=p.mem,
        )
        out = ttnn.multiply(x, y)
        ttnn.deallocate(x)
        ttnn.deallocate(y)
        return out

    base.device = device
    ref = ttnn.to_torch(base()).float()
    rows = [{"config": "r1_1", "ms": _time(base), "max_abs_diff": 0.0}]
    print(f"swiglu {precision} r1_1: {rows[0]['ms']:.3f} ms")
    for name, cfg in configs():

        def fn():
            return ttnn.experimental.minimal_matmul(
                a,
                w12,
                bias_tensor=b12,
                config=cfg,
                compute_kernel_config=p.dense,
                dtype=ttnn.bfloat16,
                memory_config=p.mem,
                fuse_swiglu=True,
            )

        fn.device = device
        try:
            out = ttnn.to_torch(fn()).float()
            ms = _time(fn)
        except Exception as e:  # L1 overflow, invalid config
            rows.append({"config": name, "error": str(e).splitlines()[0][:160]})
            continue
        diff = float((out - ref).abs().max())
        rows.append({"config": name, "ms": ms, "max_abs_diff": diff})
        print(f"swiglu {precision} {name}: {ms:.3f} ms, diff {diff:.3g}")
    ok = [r for r in rows if "ms" in r]
    best = min(ok[1:], key=lambda r: r["ms"])
    print(f"BEST swiglu {precision} {best['config']} {best['ms']:.3f} ms vs r1_1 {ok[0]['ms']:.3f}")
    if path := os.environ.get("FLOWSTATE_SWEEP_JSON"):
        with open(path, "a") as f:
            f.write(json.dumps({"role": "swiglu", "precision": precision, "rows": rows, **STAMP}) + "\n")
