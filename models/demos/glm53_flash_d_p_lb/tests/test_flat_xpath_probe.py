# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Probe: how much of flat_routed_expert's per-row cost is the x path (indexed reads of the gathered tokens)?

One chip, the model's call (C++ op, indexed mode: token_index into T = 5120 gathered tokens, clamped_silu, bfp4, 36
local experts of GLM layer 4, y row-major, fp32 down). Three chip loads bracketing the measured range (2200 / 5100 /
9500 routed rows, per-expert counts from a skewed Dirichlet over the 36 experts, scattered token ids). Per variant:
time per call (10 calls between syncs) at each load, and the linear fit ms = a + b x padded rows.
  P1     x_pages_per_row 1 (the model: one 8 KB page per token row)
  P8     x_pages_per_row 8 (a row spread over the 8 DRAM banks)
Process-level switches (set before the op is built; run one per process): MIMO_FL_XRD_SKIP=32 (no x DRAM reads,
garbage tiles), MIMO_FL_XRD_BATCH=4 (chunks per read barrier). GLM_XPP_VARIANTS (default "P1,P8")."""

import os
import time

import pytest
import torch

import ttnn

LAYER, E, NG, H, I, T = 4, 36, 288, 4096, 2048, 5120
LOADS = (2200, 5100, 9500)
YRM = os.environ.get("GLM_XPP_YRM", "1") == "1"  # 0: y as bfp8 tiles (no pack-untilize on the down cores)
ITERS = 10


def _counts(total, gen):
    p = torch.distributions.Dirichlet(torch.ones(E) * 0.8).sample()
    c = torch.floor(p * total).long()
    c[torch.argmax(p)] += total - int(c.sum())
    return c


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_xpath_probe(device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.common.bringup.testing.harness import spec
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert

    S = spec()
    loader, _ = hooks._loader_cfg(S)
    W = []
    for e in range(E):
        gw, uw, dw = PackedExpert(loader, LAYER, e).weights(torch.float32)
        W.append(tuple(w.T.contiguous() for w in (gw, uw, dw)))
    op = FlatRoutedExpert(
        device, [W], m=8192, H=H, I=I, gids=[list(range(E))], n_global=NG, wdtype="bf4", act="clamped_silu", pin=1
    )
    gen = torch.manual_seed(0)
    x_host = torch.randn(T, H).to(torch.bfloat16)
    tag = (
        ",".join(
            f"{k}={os.environ[k]}"
            for k in ("MIMO_FL_XRD_SKIP", "MIMO_FL_XRD_BATCH", "MIMO_FL_X_RESIDENT", "GLM_XPP_YRM")
            if os.environ.get(k)
        )
        or "-"
    )
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for var in os.environ.get("GLM_XPP_VARIANTS", "P1,P8").split(","):
        P = int(var[1:])
        x = rm(x_host.reshape(T * P, H // P), ttnn.bfloat16)
        pts = []
        for total in LOADS:
            c = _counts(total, gen)
            pad = ((c + 31) // 32) * 32
            off = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), pad[:-1]]), 0)
            rows = int(pad.sum())
            counts = torch.zeros(1, NG, dtype=torch.int32)
            regions = torch.zeros(1, NG, dtype=torch.int32)
            counts[0, :E], regions[0, :E] = c.int(), off.int()
            tidx = torch.randint(0, T, (1, rows), dtype=torch.int32)  # scattered tokens, like the routing
            cd, rd, td = rm(counts, ttnn.uint32), rm(regions, ttnn.uint32), rm(tidx, ttnn.uint32)
            call = lambda: op(  # noqa: E731
                x, cd, rd, token_index=td, x_pages_per_row=P, y_row_major=YRM, down_fp32=True
            )
            ttnn.deallocate(call())
            ttnn.synchronize_device(device)
            t0 = time.time()
            for _ in range(ITERS):
                ttnn.deallocate(call())
            ttnn.synchronize_device(device)
            ms = (time.time() - t0) / ITERS * 1e3
            pts.append((rows, ms))
            print(f"[xpp] {tag:22s} {var}: routed {total:5d} padded {rows:5d} rows -> {ms:7.3f} ms", flush=True)
            for t_ in (cd, rd, td):
                ttnn.deallocate(t_)
        xs = torch.tensor([[1.0, r] for r, _ in pts], dtype=torch.float64)
        ys = torch.tensor([[m] for _, m in pts], dtype=torch.float64)
        a, b = torch.linalg.lstsq(xs, ys).solution.squeeze().tolist()
        print(f"[xpp] {tag:22s} {var}: fit ms = {a:.3f} + {b * 1e3:.4f} us/row", flush=True)
        ttnn.deallocate(x)
