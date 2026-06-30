# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only: VAE chunk-trace bit-accuracy regression (not production validation).

Production ``decode_tiled`` is always eager; ``decode_chunk_traced`` is kept for
diagnosis only (see ``oobleck_vae_decoder.py``). These tests are skipped in the
default suite — opt in when investigating trace replay:

    ACE_STEP_RUN_VAE_TRACE_DIAG=1 pytest models/experimental/ace_step_v1_5/tests/test_vae_chunk_trace_equiv.py -s

For multi-shape interleave (can wedge some firmware builds):

    ACE_STEP_RUN_VAE_TRACE_DIAG=1 ACE_STEP_RUN_WEDGING_TEST=1 pytest ... -s
"""
from __future__ import annotations

import glob
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

pytestmark = pytest.mark.skipif(
    os.environ.get("ACE_STEP_RUN_VAE_TRACE_DIAG") != "1",
    reason="VAE chunk-trace diagnostics only; set ACE_STEP_RUN_VAE_TRACE_DIAG=1 to run.",
)


def _find_vae_dir() -> str | None:
    pats = [
        os.path.expanduser("~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/snapshots/*/vae"),
        os.path.expanduser("~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/vae"),
    ]
    if os.environ.get("ACE_STEP_VAE_DIR"):
        pats.insert(0, os.environ["ACE_STEP_VAE_DIR"])
    for p in pats:
        for cand in sorted(glob.glob(p)):
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
    return None


def _to_torch(t):
    return ttnn.to_torch(t).float()


def _diff(name, ref, got):
    pcc_ok, pcc_msg = comp_pcc(ref, got, pcc=0.999)
    max_abs = (ref - got).abs().max().item()
    mean_abs = (ref - got).abs().mean().item()
    print(f"  {name:28s} {pcc_msg}  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}")
    return pcc_msg, max_abs


@pytest.mark.parametrize("chunk_frames", [32])
def test_vae_chunk_trace_vs_eager(device, chunk_frames):
    vae_dir = _find_vae_dir()
    if vae_dir is None:
        pytest.skip("VAE checkpoint dir not found (set ACE_STEP_VAE_DIR)")
    print(f"\n[vae-trace-equiv] vae_dir={vae_dir}  chunk_frames={chunk_frames}")

    from models.experimental.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

    tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(vae_dir, device=device)
    dec = tt_vae._decoder
    c_lat = dec.input_channels

    torch.manual_seed(0)
    latent_t = torch.randn(1, chunk_frames, c_lat, dtype=torch.float32) * 0.5

    def fresh():
        return ttnn.from_torch(
            latent_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

    # 1) eager twice (determinism baseline)
    e1 = _to_torch(dec(fresh()))
    e2 = _to_torch(dec(fresh()))

    # 2) traced: first call captures, second replays the same fixed shape
    t_cap = _to_torch(tt_vae.decode_chunk_traced(fresh()))
    t_rep = _to_torch(tt_vae.decode_chunk_traced(fresh()))
    tt_vae.release_trace()

    print(f"[vae-trace-equiv] output shape eager={tuple(e1.shape)} traced={tuple(t_rep.shape)}")
    print("[vae-trace-equiv] comparisons (PCC floor for assert = 0.999):")
    _diff("eager vs eager", e1, e2)
    _diff("eager vs trace-capture", e1, t_cap)
    pcc_msg_rep, max_abs_rep = _diff("eager vs trace-replay", e1, t_rep)
    cap_vs_rep_msg, _ = _diff("trace-capture vs replay", t_cap, t_rep)

    # Eager must be deterministic; if not, the comparison itself is meaningless.
    det_ok, _ = comp_pcc(e1, e2, pcc=0.9999)
    assert det_ok, "eager decode is non-deterministic — cannot evaluate trace equivalence"

    # The real question: does traced replay match eager?
    rep_ok, _ = comp_pcc(e1, t_rep, pcc=0.999)
    assert rep_ok, (
        "VAE chunk trace replay diverges from eager at fixed shape "
        f"(replay {pcc_msg_rep}, max_abs={max_abs_rep:.3e}). "
        "This reproduces the bit-accuracy regression that keeps decode_tiled eager."
    )


@pytest.mark.skipif(
    os.environ.get("ACE_STEP_RUN_WEDGING_TEST") != "1",
    reason="multi-shape chunk trace can still wedge some firmware builds; opt in with " "ACE_STEP_RUN_WEDGING_TEST=1.",
)
def test_vae_chunk_trace_multishape_interleave(device):
    """Does capturing a SECOND shape corrupt the first shape's replay?

    Interior windows in ``decode_tiled`` share one shape; first/last/min-win-adjusted windows
    differ. This interleaves two shapes (capture A, capture B, replay A, replay B).
    """
    vae_dir = _find_vae_dir()
    if vae_dir is None:
        pytest.skip("VAE checkpoint dir not found (set ACE_STEP_VAE_DIR)")
    from models.experimental.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

    tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(vae_dir, device=device)
    dec = tt_vae._decoder
    c_lat = dec.input_channels

    def make(frames, seed):
        torch.manual_seed(seed)
        lt = torch.randn(1, frames, c_lat, dtype=torch.float32) * 0.5

        def fresh():
            return ttnn.from_torch(
                lt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

        return fresh

    fa, fb = make(32, 1), make(28, 2)  # two distinct window lengths
    eager_a, eager_b = _to_torch(dec(fa())), _to_torch(dec(fb()))

    tt_vae.decode_chunk_traced(fa())  # capture A
    tt_vae.decode_chunk_traced(fb())  # capture B (second shape)
    rep_a = _to_torch(tt_vae.decode_chunk_traced(fa()))  # replay A after B captured
    rep_b = _to_torch(tt_vae.decode_chunk_traced(fb()))  # replay B
    tt_vae.release_trace()

    print("\n[vae-trace-multishape] after capturing 2 shapes, replay vs eager:")
    _diff("shape A (32) replay", eager_a, rep_a)
    _diff("shape B (28) replay", eager_b, rep_b)

    a_ok, a_msg = comp_pcc(eager_a, rep_a, pcc=0.999)
    b_ok, b_msg = comp_pcc(eager_b, rep_b, pcc=0.999)
    assert a_ok and b_ok, (
        f"multi-shape trace interleave diverges (A {a_msg}, B {b_msg}) — "
        "confirms variable window lengths as the regression cause"
    )
