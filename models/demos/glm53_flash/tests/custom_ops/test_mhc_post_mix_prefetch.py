# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the post-mix v2 reader's access pattern (P1 of the decode roofline: ``MhcPostMix(prefetch=)``,
``GLM53_MHC_POST_PREFETCH``; README "Prefetch reader"), on one chip, at the model's decode shape: random fp32 streams
``x [1,1,32,16384]``, the sublayer output ``y [1,1,32,4096]`` (bf16 as the model passes it, fp32 too), the real
coefficient shapes (lane layout ``post [32,32]`` / ``comb [32,128]`` as the fused pre-block emits them; the raw
Sinkhorn tiles ``[32,4]`` / ``[32,16]`` of the composite path), both work splits, the streams / collapse outputs and
the pre-mix mode.

* ``test_post_mix_prefetch_bit_identical``: ``prefetch="mcast"`` == ``prefetch="off"`` bit for bit (``torch.equal``:
  the same core computes every output tile from the same operands), both within 1e-6 of the float64 golden;
* ``test_post_mix_prefetch_traced_timing``: each pattern captured as 16 calls in one trace and replayed 5 times (host
  wall time of the blocking replay / 16 = us per call, the dispatch gaps included); the replay output == the eager
  output of the other pattern. Expected: ``off`` ~29-31 us, ``mcast`` ~15 us (README).

    TT_VISIBLE_DEVICES=<chip> pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_post_mix_prefetch.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import PREFETCH_MODES, MhcPostMix
from models.demos.glm53_flash.tests.custom_ops.test_mhc_post_mix import _err, _random_inputs, _up, expand_x8

N, C, T = 4, 4096, 32
CALLS_PER_TRACE, REPLAYS = 16, 5


def _coef(device, post: torch.Tensor, comb: torch.Tensor, layout: str):
    """The coefficient tensors in the ``lane`` layout (the fused pre-block's output) or the ``raw`` padded tiles."""
    if layout == "lane":
        post, comb = expand_x8(post, comb, N)
    return _up(device, post), _up(device, comb)


def _inputs(device, seed: int, y_dtype, layout: str):
    x, y, post, comb = _random_inputs(T, N, C, seed=seed)
    if y_dtype == ttnn.bfloat16:
        y = y.to(torch.bfloat16).float()
    xd, yd = _up(device, x.reshape(1, 1, T, N * C)), _up(device, y.reshape(1, 1, T, C), y_dtype)
    pd, cd = _coef(device, post, comb, layout)
    return (x, y, post, comb), (xd, yd, pd, cd)


CASES = [
    # (split, coefficient layout, y dtype, mode): the model's decode call first
    ("extras", "lane", ttnn.bfloat16, "streams"),
    ("tile", "lane", ttnn.bfloat16, "streams"),
    ("extras", "raw", ttnn.float32, "streams"),
    ("tile", "raw", ttnn.bfloat16, "collapse"),
    ("tile", "lane", ttnn.float32, "collapse"),
    ("tile", "lane", ttnn.float32, "hmode"),
    ("tile", "raw", ttnn.float32, "hmode"),
]


def _run_mode(op: MhcPostMix, dev, mode: str, xd, yd, pd, cd):
    if mode == "hmode":
        h = op.allocate_h(xd)
        op.h_mix(xd, pd, h)
        return h
    collapse = mode == "collapse"
    out = op.allocate_output(xd, collapse)
    op(xd, yd, pd, cd, out, collapse)
    return out


def _golden(op: MhcPostMix, host, mode: str):
    x, y, post, comb = host
    if mode == "hmode":
        return (post.double()[:, :, None] * x.double().reshape(T, N, C)).sum(1)
    return op.golden(x, y, post, comb, N, mode == "collapse")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "split,layout,y_dtype,mode",
    CASES,
    ids=[f"{s}-{l}-{'bf16' if d == ttnn.bfloat16 else 'fp32'}-{m}" for s, l, d, m in CASES],
)
def test_post_mix_prefetch_bit_identical(mesh_device, split, layout, y_dtype, mode):
    host, (xd, yd, pd, cd) = _inputs(mesh_device, seed=11, y_dtype=y_dtype, layout=layout)
    outs = {}
    for prefetch in PREFETCH_MODES:
        op = MhcPostMix(N, version=2, split=split, prefetch=prefetch)
        outs[prefetch] = ttnn.to_torch(_run_mode(op, mesh_device, mode, xd, yd, pd, cd))[0, 0]
        rel = _err(f"prefetch={prefetch} {split} {layout} {mode}", _golden(op, host, mode), outs[prefetch])
        assert rel < 1e-6
    assert torch.equal(outs["off"], outs["mcast"]), "the prefetch reader changed an output bit"
    # a second call with the program cached (the semaphores re-armed by the launch, the plan cached)
    op = MhcPostMix(N, version=2, split=split, prefetch="mcast")
    again = ttnn.to_torch(_run_mode(op, mesh_device, mode, xd, yd, pd, cd))[0, 0]
    assert torch.equal(again, outs["off"])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_post_mix_prefetch_refuses_prefill_shapes(mesh_device, expect_error):
    """One token-tile only: a 256-row call with the multicast reader fails loudly (the plain reader serves it)."""
    x, y, post, comb = _random_inputs(256, N, C, seed=5)
    xd, yd = _up(mesh_device, x.reshape(1, 1, 256, N * C)), _up(mesh_device, y.reshape(1, 1, 256, C))
    pd, cd = _up(mesh_device, post), _up(mesh_device, comb)
    op = MhcPostMix(N, version=2, prefetch="mcast")
    out = op.allocate_output(xd, False)
    with expect_error(ValueError, "one token-tile"):
        op(xd, yd, pd, cd, out, False)
    MhcPostMix(N, version=2, prefetch="off")(xd, yd, pd, cd, out, False)
    assert _err("T=256 off", op.golden(x, y, post, comb, N, False), ttnn.to_torch(out)[0, 0]) < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 << 20}], indirect=True)
@pytest.mark.parametrize("split", ["extras", "tile"])
def test_post_mix_prefetch_traced_timing(mesh_device, split):
    """Both patterns at the model's decode call (lane coefficients, bf16 y, streams): 16 calls per trace, 5 replays."""
    host, (xd, yd, pd, cd) = _inputs(mesh_device, seed=23, y_dtype=ttnn.bfloat16, layout="lane")
    ref = None
    us = {}
    for prefetch in PREFETCH_MODES:
        op = MhcPostMix(N, version=2, split=split, prefetch=prefetch)
        out = op.allocate_output(xd, False)
        op(xd, yd, pd, cd, out, False)  # warm: compiles, caches the program and the plan
        eager = ttnn.to_torch(out)[0, 0]
        if ref is None:
            ref = eager
            assert _err(f"{split} {prefetch} eager", _golden(op, host, "streams"), eager) < 1e-6
        assert torch.equal(eager, ref)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(CALLS_PER_TRACE):
            op(xd, yd, pd, cd, out, False)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # one warm replay
        times = []
        for _ in range(REPLAYS):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            times.append((time.perf_counter() - t0) * 1e6 / CALLS_PER_TRACE)
        assert torch.equal(ttnn.to_torch(out)[0, 0], ref), "the traced replay changed an output bit"
        ttnn.release_trace(mesh_device, tid)
        us[prefetch] = (min(times), sorted(times)[len(times) // 2])
        logger.info(
            f"post-mix v2 {split} prefetch={prefetch}: {us[prefetch][0]:.1f} us/call min, {us[prefetch][1]:.1f} median"
        )
    logger.info(
        f"post-mix v2 {split}: off {us['off'][1]:.1f} -> mcast {us['mcast'][1]:.1f} us per call (median of {REPLAYS})"
    )
