# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the flow model's frame time goes at B users: traced timing of each sub-graph of one
block at the folded row count 2*B*3, and of the whole solve, so the shares add up.

    sub-graphs per block (x 3 layers x 7 Euler steps = 21 per frame):
      attn   : _split_heads (3 slices, 3 reshapes, 3 permutes) + sdpa + permute + reshape
      mm     : the five matmuls (wqkv, wo, w1, w3, w2) at this row count
      norm   : the two RMSNorms
    per step (x 7): concat x2, typecast, p0 matmul, slices, CFG combine, Euler add
    whole  : fl._solve, for the total these must explain

Env: ABL_BATCH (32), ABL_REPLAYS (16), VOXTRAL_DEVICE_ID (0).
"""

import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    CFG_ALPHA,
    FM_HEAD_DIM,
    FM_INPUT_DIM,
    FM_N_HEADS,
    N_ACOUSTIC_CODEBOOK,
    N_DECODING_STEPS,
)
from models.experimental.voxtral_tts.tests.reference_helpers import needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as fm  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow, _split_heads  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

B = int(os.environ.get("ABL_BATCH", "32"))
REPLAYS = int(os.environ.get("ABL_REPLAYS", "16"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
B2 = 2 * B
ROWS = B2 * 3


def _traced_ms(dev, fn):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    try:
        out = fn()
    finally:
        ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    try:
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / REPLAYS * 1e3
    finally:
        ttnn.release_trace(dev, tid)
        del out


def test_flow_ablation():
    dev = open_device(device_id=DEVICE_ID)
    try:
        fl = TtVoxtralFlow(dev)
        w = fl.layers[0]
        prg = fl._prg(ROWS)
        dv = lambda t, d=None: ttnn.from_torch(t.contiguous(), dtype=d or fl.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        x = dv(torch.randn(1, ROWS, FM_INPUT_DIM) * 0.02)
        x_l1 = ttnn.to_memory_config(x, fm._L1)
        qkv = dv(torch.randn(1, ROWS, fm._QKV_WIDTH) * 0.02)
        h = dv(torch.randn(1, ROWS, FM_INPUT_DIM) * 0.02)
        cc = fm.COMPUTE_CONFIG

        def attn():
            qh, kh, vh = _split_heads(qkv, B2)
            a = ttnn.transformer.scaled_dot_product_attention(
                qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=cc
            )
            return ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, ROWS, FM_N_HEADS * FM_HEAD_DIM])

        def split_only():
            return _split_heads(qkv, B2)

        def sdpa_only(qkv_heads=[None]):
            if qkv_heads[0] is None:
                qkv_heads[0] = _split_heads(qkv, B2)
            qh, kh, vh = qkv_heads[0]
            return ttnn.transformer.scaled_dot_product_attention(
                qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=cc
            )

        def mm():
            q = ttnn.linear(h, w["wqkv"], program_config=prg["wqkv"], compute_kernel_config=cc)
            a = ttnn.slice(q, [0, 0, 0], [1, ROWS, FM_N_HEADS * FM_HEAD_DIM], memory_config=fm._L1)
            o = ttnn.linear(a, w["wo"], program_config=prg["wo"], compute_kernel_config=cc, memory_config=fm._L1)
            g = ttnn.linear(h, w["w1"], program_config=prg["w1"], compute_kernel_config=cc, memory_config=fm._L1)
            u = ttnn.multiply_(
                g, ttnn.linear(h, w["w3"], program_config=prg["w3"], compute_kernel_config=cc, memory_config=fm._L1)
            )
            return o, ttnn.linear(u, w["w2"], program_config=prg["w2"], compute_kernel_config=cc, memory_config=fm._L1)

        def norm():
            return fl._norm(x_l1, w["an"]), fl._norm(x_l1, w["fn"])

        def block():
            return fl._block(ttnn.clone(x_l1), w, B2)

        x0 = dv(torch.randn(B, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
        pair = dv(torch.randn(B2, 1, FM_INPUT_DIM) * 0.02)

        def whole():
            return fl._solve(x0, pair, B, N_DECODING_STEPS, CFG_ALPHA)

        res = {}
        for name, fn in (
            ("split_only", split_only),
            ("sdpa_only", sdpa_only),
            ("attn", attn),
            ("mm", mm),
            ("norm", norm),
            ("block", block),
            ("whole", whole),
        ):
            res[name] = _traced_ms(dev, fn)
        per_frame = {
            k: v * (21 if k in ("split_only", "sdpa_only", "attn", "mm", "norm", "block") else 1)
            for k, v in res.items()
        }
        print(f"\n[ablation] B={B} (rows {ROWS}), traced, {REPLAYS} replays")
        print(f"{'sub-graph':12s} {'ms each':>9s} {'per frame':>10s}")
        for k in ("split_only", "sdpa_only", "attn", "mm", "norm", "block", "whole"):
            print(f"{k:12s} {res[k]:9.3f} {per_frame[k]:10.2f}")
        print(
            f"[ablation] block parts attn+mm+norm = {per_frame['attn'] + per_frame['mm'] + per_frame['norm']:.2f} ms/frame vs block x21 = {per_frame['block']:.2f}; whole solve = {per_frame['whole']:.2f}"
        )
        print(
            f"[ablation] attention share of the block: {100 * res['attn'] / res['block']:.0f}%  (split+permutes {100 * res['split_only'] / res['block']:.0f}%, sdpa {100 * res['sdpa_only'] / res['block']:.0f}%)"
        )
        assert res["whole"] > 0
    finally:
        ttnn.close_device(dev)
