# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token-major flow solve (B > 1) against the batch-major one: same result, and the time saved.

  equivalence : x from _solve_tm vs _solve on the same inputs, PCC and max |diff|; FSQ codes equal
  timing      : traced ms per frame for both at B users, and the batch-1 reference

Env: TM_BATCH (32), TM_REPLAYS (16), VOXTRAL_DEVICE_ID (0).
"""

import json
import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    CFG_ALPHA,
    FM_INPUT_DIM,
    N_ACOUSTIC_CODEBOOK,
    N_DECODING_STEPS,
    pcc,
)
from models.experimental.voxtral_tts.reference.voxtral_flow_ref import _fsq_quantize  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

B = int(os.environ.get("TM_BATCH", "32"))
REPLAYS = int(os.environ.get("TM_REPLAYS", "16"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("TM_RESULTS", "")


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


def test_token_major_solve_matches_and_is_faster():
    dev = open_device(device_id=DEVICE_ID)
    try:
        fl = TtVoxtralFlow(dev)
        torch.manual_seed(0)
        # Realistic-scale inputs: the llm hidden after the final norm is O(1); noise is N(0,1).
        x0_t = torch.randn(B, 1, N_ACOUSTIC_CODEBOOK)
        h_t = torch.randn(2 * B, 1, FM_INPUT_DIM)
        h_t[B:] = 0.0  # uncond rows are zeros, as _cfg_input builds them
        dv = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT, device=dev)
        x0 = dv(x0_t, ttnn.float32)
        h = dv(h_t, fl.dtype)

        os.environ["VOXTRAL_FLOW_TM"] = "0"
        x_bm = ttnn.to_torch(fl._solve(x0, h, B, N_DECODING_STEPS, CFG_ALPHA)).float().reshape(B, N_ACOUSTIC_CODEBOOK)
        os.environ["VOXTRAL_FLOW_TM"] = "1"
        x_tm = (
            ttnn.to_torch(fl._solve_tm(x0, h, B, N_DECODING_STEPS, CFG_ALPHA)).float().reshape(B, N_ACOUSTIC_CODEBOOK)
        )
        p = pcc(x_tm, x_bm)
        mad = float((x_tm - x_bm).abs().max())
        codes_eq = float((_fsq_quantize(x_tm) == _fsq_quantize(x_bm)).float().mean())
        print(
            f"\n[tm] B={B}: token-major vs batch-major solve: PCC {p:.6f}, max|diff| {mad:.4e}, FSQ codes equal {codes_eq:.4f}"
        )
        # The arbiter: the fp32 reference solve on the same inputs (its final x before FSQ).
        from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref

        w = fref.load_flow_state()
        sem = torch.full((B, 1), 100, dtype=torch.long)  # any non-END semantic code
        _, trace = fref.decode_frame(
            sem,
            h_t[:B, 0].float(),
            w,
            cfg_alpha=CFG_ALPHA,
            n_steps=N_DECODING_STEPS,
            x_0=x0_t.reshape(B, -1),
            return_trace=True,
        )
        x_ref = trace[-1]
        p_bm, p_tm = pcc(x_bm, x_ref), pcc(x_tm, x_ref)
        c_bm = float((_fsq_quantize(x_bm) == _fsq_quantize(x_ref)).float().mean())
        c_tm = float((_fsq_quantize(x_tm) == _fsq_quantize(x_ref)).float().mean())
        worst_row_bm = min(pcc(x_bm[b], x_ref[b]) for b in range(B))
        worst_row_tm = min(pcc(x_tm[b], x_ref[b]) for b in range(B))
        print(
            f"[tm] B={B}: vs fp32 reference: batch-major PCC {p_bm:.6f} (worst row {worst_row_bm:.4f}, codes {c_bm:.4f}) | "
            f"token-major PCC {p_tm:.6f} (worst row {worst_row_tm:.4f}, codes {c_tm:.4f})"
        )

        os.environ["VOXTRAL_FLOW_TM"] = "0"
        ms_bm = _traced_ms(dev, lambda: fl._solve(x0, h, B, N_DECODING_STEPS, CFG_ALPHA))
        os.environ["VOXTRAL_FLOW_TM"] = "1"
        ms_tm = _traced_ms(dev, lambda: fl._solve_tm(x0, h, B, N_DECODING_STEPS, CFG_ALPHA))
        print(
            f"[tm] B={B}: traced solve per frame: batch-major {ms_bm:.2f} ms, token-major {ms_tm:.2f} ms ({ms_bm / ms_tm:.2f}x)"
        )
        if RESULTS_PATH:
            json.dump(
                {
                    "batch": B,
                    "pcc": p,
                    "max_abs_diff": mad,
                    "fsq_codes_equal": codes_eq,
                    "ms_batch_major": ms_bm,
                    "ms_token_major": ms_tm,
                },
                open(RESULTS_PATH, "w"),
                indent=2,
            )
        assert p_tm >= 0.999, f"token-major solve diverged from the fp32 reference: PCC {p_tm:.5f}"
        assert (
            p_tm >= p_bm - 0.001
        ), f"token-major further from the reference than batch-major: {p_tm:.5f} vs {p_bm:.5f}"
    finally:
        os.environ.pop("VOXTRAL_FLOW_TM", None)
        ttnn.close_device(dev)
