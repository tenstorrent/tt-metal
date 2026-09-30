# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused V4.1 mHC collapse / hc_post (tt/v41/mhc.py, bead 8y7.9.5) vs the composite ttnn path and fp64 torch.

The fused ops run the composite path's SFPU multiply / addcmul sequence in the same order, so they must be
bit-identical to ``tt_mhc._mix`` / ``TtMHCWrap.hc_post``; the fp64 comparison guards against both being wrong
the same way. The block's bf16 sublayer interface (bead 8y7.9.9) is checked against the typecasts it replaces:
the bf16 collapse equals the fp32 collapse typecast to bf16, and hc_post of a bf16 ``h`` equals hc_post of that
``h`` typecast to fp32, bit for bit. ``test_mhc_fused_traced_time`` logs the traced per-call time at the production per-chip shape
(chunk 5120 on 2x4: 2560 tokens, hidden slice 1280) against the DRAM bound as ``V41_MHC_PERF`` JSON lines.
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, _cols, _mix, _streams
from models.demos.deepseek_v3_d_p.tt.v41.mhc import fused_collapse, fused_hc_post

N = 4
DRAM_GBPS = 512.0  # Blackhole p150 DRAM peak (G2 bound)
REPLAYS = 20
SHAPES = {"small": (128, 256), "two_rows_odd": (64, 96), "production": (2560, 1280)}  # (tokens, hidden/tp)


def _inputs(tokens, hidden, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(1, 1, tokens, N * hidden, generator=gen),
        h=torch.randn(1, 1, tokens, hidden, generator=gen),
        pre=torch.rand(1, 1, tokens, N, generator=gen),
        post=2 * torch.rand(1, 1, tokens, N, generator=gen),
        comb=torch.rand(1, 1, tokens, N * N, generator=gen) / N,
    )


def _upload(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _composite(op, d):
    if op == "collapse":
        return _mix(_streams(d["x"], N), _cols(d["pre"], N))
    return TtMHCWrap.hc_post(SimpleNamespace(n=N), d["h"], d["x"], d["post"], d["comb"])


def _fused(op, d):
    if op == "collapse":
        return fused_collapse(d["x"], d["pre"], N)
    return fused_hc_post(d["h"], d["x"], d["post"], d["comb"], N)


def _reference(op, t):
    """fp64 torch: V4.1 ``Block.hc_pre`` / ``hc_post`` on the packed layout."""
    s, hidden = t["x"].shape[-2], t["h"].shape[-1]
    x = t["x"].double().reshape(s, N, hidden)
    if op == "collapse":
        return (t["pre"].double().reshape(s, N, 1) * x).sum(1).reshape(1, 1, s, hidden)
    comb = t["comb"].double().reshape(s, N, N)  # [i, j]: input stream i -> output stream j
    y = t["post"].double().reshape(s, N, 1) * t["h"].double().reshape(s, 1, hidden)
    y = y + torch.einsum("sij,sid->sjd", comb, x)
    return y.reshape(1, 1, s, N * hidden)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("op", ["collapse", "hc_post"])
def test_mhc_fused_matches_composite(device, op, shape):
    tokens, hidden = SHAPES[shape]
    t = _inputs(tokens, hidden)
    d = {k: _upload(device, v) for k, v in t.items()}
    t0 = time.perf_counter()
    fused = [ttnn.to_torch(_fused(op, d)) for _ in range(2)]
    logger.info(f"fused {op} {shape}: 2 calls {time.perf_counter() - t0:.2f}s (incl. compile)")
    composite = ttnn.to_torch(_composite(op, d))
    assert torch.equal(fused[0], fused[1]), "fused mHC op is not deterministic"
    ref = _reference(op, t)
    err = (fused[0].double() - ref).abs().max().item()
    mismatch = (fused[0] != composite).sum().item()
    logger.info(f"{op} {shape}: max |fused - fp64| {err:.3e}, elements != composite {mismatch}")
    assert err < 1e-5 * ref.abs().max().item(), err
    assert mismatch == 0, f"{mismatch} elements differ from the composite path"


def _traced_us(device, fn):
    fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / REPLAYS * 1e6
    ttnn.release_trace(device, tid)
    out.deallocate()
    return us


def _fused_bf16(op, d):
    """The block's bf16 sublayer interface: collapse -> bf16, hc_post of a bf16 ``h``."""
    if op == "collapse":
        return fused_collapse(d["x"], d["pre"], N, ttnn.bfloat16)
    return fused_hc_post(d["h_bf16"], d["x"], d["post"], d["comb"], N)


def _typecast_bf16(op, d):
    """What the bf16 interface replaces: the fp32 fused op with a typecast on the sublayer side."""
    if op == "collapse":
        return ttnn.typecast(fused_collapse(d["x"], d["pre"], N), ttnn.bfloat16)
    return fused_hc_post(ttnn.typecast(d["h_bf16"], ttnn.float32), d["x"], d["post"], d["comb"], N)


def _with_bf16_h(device, d, t):
    d["h_bf16"] = ttnn.from_torch(
        t["h"].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return d


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("op", ["collapse", "hc_post"])
def test_mhc_fused_bf16_sublayer_io(device, op, shape):
    tokens, hidden = SHAPES[shape]
    t = _inputs(tokens, hidden)
    d = _with_bf16_h(device, {k: _upload(device, v) for k, v in t.items()}, t)
    out = [_fused_bf16(op, d) for _ in range(2)]
    assert out[0].dtype == (ttnn.bfloat16 if op == "collapse" else ttnn.float32), out[0].dtype
    got = [ttnn.to_torch(o) for o in out]
    want = ttnn.to_torch(_typecast_bf16(op, d))
    assert torch.equal(got[0], got[1]), "fused mHC op is not deterministic"
    mismatch = (got[0] != want).sum().item()
    logger.info(f"{op} {shape} bf16 sublayer io: elements != typecast path {mismatch}")
    assert mismatch == 0, f"{mismatch} elements differ from the typecast path"


@pytest.mark.parametrize("impl", ["fused", "composite", "fused_bf16", "typecast_bf16"])
@pytest.mark.parametrize("op", ["collapse", "hc_post"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
def test_mhc_fused_traced_time(device, op, impl):
    tokens, hidden = SHAPES["production"]
    t = _inputs(tokens, hidden)
    d = _with_bf16_h(device, {k: _upload(device, v) for k, v in t.items()}, t)
    impls = {"fused": _fused, "composite": _composite, "fused_bf16": _fused_bf16, "typecast_bf16": _typecast_bf16}
    us = _traced_us(device, lambda: impls[impl](op, d))
    stream_bytes = 4 * tokens * N * hidden
    sub = 0.5 if impl.endswith("bf16") else 1.0  # the sublayer-side tensor (collapse output / h) in bf16
    # collapse: read the streams, write one stream; hc_post: read the streams + h, write the streams
    moved = stream_bytes * (1 + sub / N) if op == "collapse" else stream_bytes * (2 + sub / N)
    bound_us = moved / (DRAM_GBPS * 1e3)
    record = {
        "op": op,
        "impl": impl,
        "tokens": tokens,
        "hidden_per_chip": hidden,
        "traced_us": round(us, 1),
        "dram_bound_us": round(bound_us, 1),
        "dram_util": round(bound_us / us, 3),
    }
    line = "V41_MHC_PERF " + json.dumps(record)
    logger.info(line)
    with open(os.environ.get("V41_MHC_PERF_OUT", "generated/v41_mhc_perf.log"), "a") as f:
        f.write(line + "\n")
