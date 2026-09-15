# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Quantify the three remaining ``is_blackhole()``-gated Wormhole changes at 27B / TP=8.

Closes the "win unquantified at TP=8" gap in the Wormhole gating audit for:

  item 5   attention/tp.py _qk_norm   -- fused rms_norm(weight=) vs rms_norm + separate multiply.
                                        Cited measurement was N150, S=2048 HD=256:
                                        q (NH=8) 182.0 -> 78.3 us, k (NH=2) 55.1 -> 25.1 us.
  item 7   attention/rope_tp.py rot_mats_prefill -- device table slice vs host trig + DMA.
                                        NO cited measurement on any config.
  item 11  model.py _rope_from_idx    -- device table gather vs host trig + pack + DMA.
                                        NO cited measurement on any config.

WHY THESE NEED A DIFFERENT METRIC FROM ITEM 6
---------------------------------------------
Item 6 was a matmul: device kernel time was the whole story. Items 7 and 11 move work from the
HOST to the device, so device time alone would make them look like pure regressions -- they ADD
device ops (a table slice / two embedding gathers) to REMOVE host trig and a multi-MB DMA. Both
numbers are therefore reported, and for 7/11 the host column is the one the change is about.

HOW THE "BEFORE" BRANCH IS REACHED ON WORMHOLE
----------------------------------------------
Item 7 needs no patching: ``rot_mats_prefill`` takes the device-slice path only when
``position_ids is None and not is_blackhole()``, so passing an explicit
``position_ids=arange(seq_len)`` -- numerically the exact same positions -- falls through to the
host branch. Same inputs, same expected output, both paths on the same hardware.

Item 11 has no such seam (``rot_mats_decode`` branches on ``is_blackhole()`` alone), so the host
"before" is reconstructed here from the statements in that branch plus ``pack_rope_host``. That
reconstruction is a COPY and can drift -- it is asserted against the device path's values, so a
drift shows up as a value mismatch rather than a silently wrong timing.

Run (needs a device; skipped unless the env var is set)::

    QWEN_QKROPE_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_qknorm_rope_sweep.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

S = 2048
ITERS = 5
DECODE_B = [1, 32]
_SKIP = os.environ.get("QWEN_QKROPE_SWEEP") != "1"

try:
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]


def _timed(mesh_device, fn, label, iters=ITERS):
    """Wall clock per call, with the region signposted so tracy can give the device half."""
    fn()  # warm
    ttnn.synchronize_device(mesh_device)
    if _SP is not None:
        _SP(f"{label}_start")
    t0 = time.time()
    for _ in range(iters):
        out = fn()
        del out
    ttnn.synchronize_device(mesh_device)
    us = (time.time() - t0) / iters * 1e6
    if _SP is not None:
        _SP(f"{label}_stop")
    return us


@pytest.mark.skipif(_SKIP, reason="set QWEN_QKROPE_SWEEP=1 to run the qk-norm / RoPE sweep")
@pytest.mark.timeout(2400)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_qknorm_rope_sweep(mesh_device, device_params):
    del device_params
    from models.demos.blackhole.qwen36.tests.test_factory import model_path
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import _rope_dev_tables, rot_mats_prefill
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    mesh_device.enable_program_cache()
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=32, max_seq_len=4096)
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    NH, NKV = args.n_local_heads, args.n_local_kv_heads
    HD, RD = args.head_dim, args.rope_head_dim
    logger.info(f"qk/rope sweep: NH={NH} NKV={NKV} HD={HD} rope_dim={RD} S={S} mesh={MESH_SHAPE}")

    def _mk(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh_device, **({"mesh_mapper": rep} if rep else {}))

    # ---------------------------------------------------------------- item 5
    logger.info("--- item 5: _qk_norm fused vs unfused (prefill head shapes) ---")
    torch.manual_seed(0)
    item5 = []
    for name, nh in (("q", NH), ("k", NKV)):
        x = _mk(torch.randn(1, nh, S, HD, dtype=torch.bfloat16))
        w = _mk(torch.randn(1, 1, 1, HD, dtype=torch.bfloat16) * 0.1 + 1.0)
        _L1 = ttnn.L1_MEMORY_CONFIG

        def unfused():
            return ttnn.multiply(ttnn.rms_norm(x, epsilon=1e-6, memory_config=_L1), w, memory_config=_L1)

        def fused():
            return ttnn.rms_norm(x, weight=w, epsilon=1e-6, memory_config=_L1)

        a, b = unfused(), fused()
        ttnn.synchronize_device(mesh_device)
        ha = ttnn.to_torch(ttnn.get_device_tensors(a)[0] if _MULTI else a).float()
        hb = ttnn.to_torch(ttnn.get_device_tensors(b)[0] if _MULTI else b).float()
        maxdiff = float((ha - hb).abs().max())
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        u = _timed(mesh_device, unfused, f"qknorm_{name}_unfused")
        f = _timed(mesh_device, fused, f"qknorm_{name}_fused")
        logger.info(
            f"  {name} (nh={nh})  unfused {u:8.1f} us  fused {f:8.1f} us  wall {100*(f-u)/u:+6.1f}%  "
            f"max|diff| {maxdiff:.4g}"
        )
        item5.append((name, nh, u, f, maxdiff))
        ttnn.deallocate(x)
        ttnn.deallocate(w)

    # ---------------------------------------------------------------- item 7
    logger.info("--- item 7: rot_mats_prefill device-slice vs host-trig ---")
    theta, fhd = args.rope_theta, getattr(args, "rope_full_head_dim", None)
    pos_explicit = torch.arange(S).view(1, -1)  # forces the host branch; same positions as None

    def rope_pf_dev():
        return rot_mats_prefill(mesh_device, RD, S, theta, full_head_dim=fhd)

    def rope_pf_host():
        return rot_mats_prefill(mesh_device, RD, S, theta, position_ids=pos_explicit, full_head_dim=fhd)

    cd, sd = rope_pf_dev()
    ch, sh = rope_pf_host()
    ttnn.synchronize_device(mesh_device)

    def _h(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0] if _MULTI else t).float()

    d7 = max(float((_h(cd) - _h(ch)).abs().max()), float((_h(sd) - _h(sh)).abs().max()))
    for t in (cd, sd, ch, sh):
        ttnn.deallocate(t)
    dev7 = _timed(mesh_device, rope_pf_dev, "rope_prefill_device")
    host7 = _timed(mesh_device, rope_pf_host, "rope_prefill_host")
    dma_mb = 2 * S * (fhd or RD) * 2 / 1e6
    logger.info(
        f"  prefill S={S}  device-slice {dev7:9.1f} us   host-trig {host7:9.1f} us   "
        f"{100*(dev7-host7)/host7:+6.1f}%   max|diff| {d7:.4g}   (host path DMAs ~{dma_mb:.2f} MB)"
    )

    # ---------------------------------------------------------------- item 11
    logger.info("--- item 11: decode rope, device gather vs host trig+pack+DMA ---")
    item11 = []
    for B in DECODE_B:
        idx_host = torch.arange(B, dtype=torch.int32).reshape(1, B)
        idx = _mk(idx_host, dt=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        w_rope = fhd or RD
        tbl_c, tbl_s = _rope_dev_tables(mesh_device, RD, args.max_seq_len, theta, full_head_dim=fhd)

        def dev_gather():
            def g(tbl):
                r = ttnn.embedding(idx, tbl)
                r = ttnn.reshape(r, (1, B, 1, w_rope))
                return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

            return g(tbl_c), g(tbl_s)

        def host_trig():
            # COPY of rot_mats_decode's is_blackhole() branch (+ pack_rope_host's DMA shape).
            inv = 1.0 / (theta ** (torch.arange(0, RD, 2).float() / RD))
            fr = torch.outer(idx_host.float().reshape(-1), inv)
            emb = torch.cat([fr, fr], dim=-1)
            c, s = emb.cos(), emb.sin()
            if w_rope != RD:  # to_full_width_rot_mats' effect on size, without importing it
                c = torch.nn.functional.pad(c, (0, w_rope - RD))
                s = torch.nn.functional.pad(s, (0, w_rope - RD))
            packed = torch.cat([c.reshape(1, B, 1, w_rope), s.reshape(1, B, 1, w_rope)], dim=0).to(torch.bfloat16)
            return _mk(packed)

        a, b = dev_gather(), host_trig()
        ttnn.synchronize_device(mesh_device)
        gc = _h(a[0])[0, :, 0, :RD]
        hc = _h(b)[0, :, 0, :RD]
        d11 = float((gc - hc).abs().max())
        for t in (a[0], a[1], b):
            ttnn.deallocate(t)
        dg = _timed(mesh_device, dev_gather, f"rope_decode_dev_b{B}")
        hh = _timed(mesh_device, host_trig, f"rope_decode_host_b{B}")
        logger.info(
            f"  decode B={B:<3} device-gather {dg:8.1f} us   host-trig {hh:8.1f} us   "
            f"{100*(dg-hh)/hh:+6.1f}%   max|diff| {d11:.4g}"
        )
        item11.append((B, dg, hh, d11))
        ttnn.deallocate(idx)

    logger.info("=== summary (wall clock; device half from tracy signposts) ===")
    for name, nh, u, f, md in item5:
        logger.info(f"  item5  qk_norm {name} nh={nh:<3} unfused {u:8.1f} -> fused {f:8.1f} us  max|diff| {md:.4g}")
    logger.info(f"  item7  rope prefill S={S}   host {host7:8.1f} -> device {dev7:8.1f} us  max|diff| {d7:.4g}")
    for B, dg, hh, d11 in item11:
        logger.info(f"  item11 rope decode B={B:<3}   host {hh:8.1f} -> device {dg:8.1f} us  max|diff| {d11:.4g}")
