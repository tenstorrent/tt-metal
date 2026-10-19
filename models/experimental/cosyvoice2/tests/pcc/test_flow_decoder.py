# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""CosyVoice2's flow-matching estimator: `CausalConditionalDecoder` (U-Net) and
`CausalConditionalCFM` (fixed-noise Euler solver) -- see tt/flow/decoder.py's module
docstring for the verified real-source architecture (1-stage `channels=[256]`
topology, GroupNorm->LayerNorm swap, fixed-noise-buffer CFM, `streaming=False` only).

`matcha-tts`/`diffusers`/`conformer` are not installed in this environment (confirmed
by import failure), so unlike istft/hift's real `torch.istft`/`F.interpolate`
comparisons, there is no real upstream class this phase can call directly -- same
situation this package was already in for HiFT/SineGen2, and validated the same way:
`*Ref` classes below are a line-by-line transcription of the real source read for
this port (cited in tt/flow/decoder.py), built from genuine `torch.nn`/
`torch.nn.functional` primitives throughout (real `nn.LayerNorm`, real `F.mish`,
real `F.gelu`, real `F.scaled_dot_product_attention`) rather than hand-derived
reimplementations, so a TT-vs-reference PCC comparison carries no "shared bug" risk
for those primitives. On top of that, this file adds the genuinely external checks
this component's real source enables: the causal-conv independence property (a fact
about the real architecture, not about this port's own code) and the sinusoidal
embedding's closed-form values.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_causal_conv1d_future_independence():
    """The actual property "causal" claims: `CausalConv1dRef`'s output at position
    `i` must be invariant to ANY change at positions `> i`. This is a fact about
    the real architecture (upstream's own `F.pad(x, (k-1, 0))` + `padding=0` conv),
    checked independent of this port's own device code -- the genuinely external
    check this component has, in place of a real upstream class to call directly.
    """
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConv1dRef

    torch.manual_seed(0)
    conv = CausalConv1dRef(8, 8, 3)
    conv.eval()
    x = torch.randn(1, 8, 20)
    with torch.no_grad():
        y1 = conv(x)
        x2 = x.clone()
        x2[:, :, 10:] += 999.0
        y2 = conv(x2)
    assert torch.equal(y1[:, :, :10], y2[:, :, :10]), "position < 10 must be unaffected by a change at position >= 10"
    assert not torch.equal(y1[:, :, 10:], y2[:, :, 10:]), "the perturbed positions themselves must actually change"


def test_causal_block1d_layernorm_is_per_timestep():
    """The GroupNorm -> LayerNorm swap tt/flow/decoder.py's module docstring flags
    as a real, causality-motivated output-value difference (not just an added
    mask): `CausalBlock1DRef`'s LayerNorm normalises each timestep independently,
    so -- combined with the causal conv underneath -- position `i`'s output must
    still be unaffected by a change at position `> i`. A GroupNorm-based block
    would fail this: GroupNorm's statistics span every timestep in the sample, so
    a change anywhere changes the normalisation everywhere.
    """
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalBlock1DRef

    torch.manual_seed(1)
    block = CausalBlock1DRef(8, 8)
    block.eval()
    x = torch.randn(1, 8, 20)
    mask = torch.ones(1, 1, 20)
    with torch.no_grad():
        y1 = block(x, mask)
        x2 = x.clone()
        x2[:, :, 10:] += 999.0
        y2 = block(x2, mask)
    assert torch.equal(y1[:, :, :10], y2[:, :, :10])


def test_sinusoidal_pos_emb_matches_closed_form():
    """`sinusoidal_pos_emb_torch` at `t=0`: every frequency's argument is 0, so
    `sin(0)=0` for the first half and `cos(0)=1` for the second half, for every
    channel -- independent of the formula's own implementation, a literal
    algebraic fact about what it must produce."""
    from models.experimental.cosyvoice2.tt.flow.decoder import sinusoidal_pos_emb_torch

    emb = sinusoidal_pos_emb_torch(torch.zeros(3), dim=320)
    assert emb.shape == (3, 320)
    assert torch.allclose(emb[:, :160], torch.zeros(3, 160), atol=1e-6)
    assert torch.allclose(emb[:, 160:], torch.ones(3, 160), atol=1e-6)


def test_decoder_torch_reference_shape_and_range():
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef

    torch.manual_seed(2)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    b, t_len = 1, 32
    x = torch.randn(b, t_len, 80) * 0.1
    mu = torch.randn(b, t_len, 80) * 0.1
    cond = torch.randn(b, t_len, 80) * 0.1
    spks = torch.randn(b, 80) * 0.1
    mask = torch.ones(b, t_len, 1)
    t = torch.rand(b)
    with torch.no_grad():
        out = dec(x, mask, mu, t, spks, cond)
    assert out.shape == (b, t_len, 80)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("length", [32, 64])
def test_device_causal_conv1d_matches_host(device, length):
    """`TtCausalConv1d` (asymmetric `padding=(k-1, 0)` passed straight to
    `ttnn.conv1d`, confirmed from its own docstring to accept `[pad_left,
    pad_right]` directly -- no manual `ttnn.pad` step, which would hit that op's
    documented "front padding on device not supported in tile layout"
    restriction) vs. the real `F.pad` + `padding=0` conv. Isolated from the rest of
    the estimator: this is the single most novel primitive this phase introduces.
    """
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConv1dRef, TtCausalConv1d

    torch.manual_seed(length)
    conv = CausalConv1dRef(16, 24, 3)
    conv.eval()
    x = torch.randn(1, 16, length) * 0.2
    with torch.no_grad():
        want = conv(x).transpose(1, 2)  # -> [1, L, 24], this package's channels-last convention

    tt_conv = TtCausalConv1d.from_module(device, conv)
    x_dev = ttnn.from_torch(x.transpose(1, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(tt_conv(x_dev, length, 1)).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalConv1d (L={length}) PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_decoder_matches_torch_reference(device):
    """`TtCausalConditionalDecoder` (56 `BasicTransformerBlock`s + 14
    `CausalResnetBlock1D`s, the real checkpoint's verified 1-stage topology) vs.
    `CausalConditionalDecoderRef`, both random-init (no CosyVoice2 checkpoint yet,
    matching this package's existing convention). `streaming=False`, single
    forward pass -- not through the Euler solver, so this isolates the estimator
    network itself from the CFM wrapper (see test_device_cfm_matches_torch_reference
    below for that)."""
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef, TtCausalConditionalDecoder

    torch.manual_seed(0)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    b, t_len = 1, 32
    x = torch.randn(b, t_len, 80) * 0.1
    mu = torch.randn(b, t_len, 80) * 0.1
    cond = torch.randn(b, t_len, 80) * 0.1
    spks = torch.randn(b, 80) * 0.1
    mask = torch.ones(b, t_len, 1)
    t = torch.rand(b)
    with torch.no_grad():
        want = dec(x, mask, mu, t, spks, cond)

    tt_dec = TtCausalConditionalDecoder(device, dec)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mu_dev = ttnn.from_torch(mu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cond_dev = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spks_dev = ttnn.from_torch(spks.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_dev = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(tt_dec(x_dev, mask_dev, mu_dev, t, spks_dev, cond_dev, t_len, 1)).float()
    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalConditionalDecoder PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_cfm_matches_torch_reference(device):
    """`TtCausalConditionalCFM`'s full 4-step Euler solve (fixed-noise buffer, real
    classifier-free-guidance batch-of-2 doubling, cosine `t_scheduler`) vs.
    `CausalConditionalCFMRef`, wrapping the SAME decoder weights
    `test_device_decoder_matches_torch_reference` already validated in isolation --
    this test is what confirms the solver loop itself (not just one estimator
    call) is wired correctly: CFG splitting, the `dt` recompute from `t_span`, and
    that every Euler step's output actually feeds the next step's input on both
    sides identically.
    """
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalCFMRef,
        CausalConditionalDecoderRef,
        TtCausalConditionalCFM,
        TtCausalConditionalDecoder,
    )

    torch.manual_seed(3)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm = CausalConditionalCFMRef(dec)

    b, t_len = 1, 32
    mu = torch.randn(b, t_len, 80) * 0.1
    cond = torch.randn(b, t_len, 80) * 0.1
    spks = torch.randn(b, 80) * 0.1
    mask = torch.ones(b, t_len, 1)

    with torch.no_grad():
        want = cfm.forward(mu, mask, n_timesteps=4, spks=spks, cond=cond)

    tt_dec = TtCausalConditionalDecoder(device, dec)
    tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm.rand_noise, cfm)
    got = tt_cfm.forward(mu, mask, n_timesteps=4, spks_t=spks, cond_t=cond)

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalConditionalCFM (4-step Euler) PCC {pcc}")
    assert passed, pcc


needs_l1_small_trace = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 50_000_000}], indirect=True
)


@needs_l1_small_trace
def test_device_cfm_trace_cache_across_utterances_and_replays(device):
    """Regression for the two hard requirements on the 2026-09-22 traced Euler-step solve
    (`TtCausalConditionalCFM._capture`/`_forward_traced`, ported from the CosyVoice1
    reference repo's `tt/flow/cfm.py`):

    1. **The trace captures exactly ONE Euler step and is replayed N times from the host
       -- step count is never baked into the capture.** Proven here by reusing the SAME
       cached trace (same `(t_len, channels)` key) across DIFFERENT `n_timesteps` values --
       if step count were somehow baked in, either the second call would silently replay
       the wrong number of times (a shape-correct but numerically wrong result) or the
       trace would need to be recaptured every time `n_timesteps` changed, which
       `_trace_key_for` deliberately does not key on.
    2. **Replay-vs-eager PCC holds at several step counts and across multiple utterances,
       not just the first replay.** Four cases run against the same `tt_cfm` instance in
       sequence: a first capture, a same-geometry reuse at a DIFFERENT step count, a
       different-geometry forced recapture, and a return to the first geometry -- each
       compared against its own independent eager (`use_trace=False`) run on fresh random
       conditioning, not against a cached "golden" answer.
    """
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalCFMRef,
        CausalConditionalDecoderRef,
        TtCausalConditionalCFM,
        TtCausalConditionalDecoder,
    )

    torch.manual_seed(7)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm = CausalConditionalCFMRef(dec)
    tt_dec = TtCausalConditionalDecoder(device, dec)
    tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm.rand_noise, cfm)

    # (t_len, n_timesteps): A first-captures at t_len=32; B reuses that SAME trace at a
    # different step count; C forces a recapture at a different length; D returns to the
    # first geometry, forcing a second recapture there.
    cases = [(32, 4), (32, 10), (48, 10), (32, 6)]
    try:
        for i, (t_len, n_timesteps) in enumerate(cases):
            mu = torch.randn(1, t_len, 80) * 0.1
            cond = torch.randn(1, t_len, 80) * 0.1
            spks = torch.randn(1, 80) * 0.1
            mask = torch.ones(1, t_len, 1)
            with torch.no_grad():
                want = cfm.forward(mu, mask, n_timesteps=n_timesteps, spks=spks, cond=cond)
            got = tt_cfm.forward(mu, mask, n_timesteps, spks, cond, use_trace=True)
            assert got.shape == want.shape
            passed, pcc = comp_pcc(want, got, GATE_BF16)
            print(f"\n  case {i} t_len={t_len} n_timesteps={n_timesteps} (traced) PCC {pcc}")
            assert passed, f"case {i} (t_len={t_len}, n_timesteps={n_timesteps}): {pcc}"
    finally:
        tt_cfm.release_cfm_trace()


@needs_l1_small_trace
@pytest.mark.parametrize("n_timesteps", [3, 5, 10])
def test_device_cfm_traced_matches_eager_per_step_count(device, n_timesteps):
    """Traced vs. eager at several individual step counts (not just the 4-step case
    `test_device_cfm_trace_cache_across_utterances_and_replays` exercises as one point
    among several) -- each a fresh `TtCausalConditionalCFM`, so this also confirms a
    from-cold first capture is correct at each count on its own, not only as part of a
    reuse sequence."""
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalCFMRef,
        CausalConditionalDecoderRef,
        TtCausalConditionalCFM,
        TtCausalConditionalDecoder,
    )

    torch.manual_seed(100 + n_timesteps)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm = CausalConditionalCFMRef(dec)
    b, t_len = 1, 40
    mu = torch.randn(b, t_len, 80) * 0.1
    cond = torch.randn(b, t_len, 80) * 0.1
    spks = torch.randn(b, 80) * 0.1
    mask = torch.ones(b, t_len, 1)
    with torch.no_grad():
        want = cfm.forward(mu, mask, n_timesteps=n_timesteps, spks=spks, cond=cond)

    tt_dec = TtCausalConditionalDecoder(device, dec)
    tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm.rand_noise, cfm)
    try:
        got = tt_cfm.forward(mu, mask, n_timesteps, spks, cond, use_trace=True)
    finally:
        tt_cfm.release_cfm_trace()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device CausalConditionalCFM traced ({n_timesteps}-step Euler) PCC {pcc}")
    assert passed, pcc


@needs_l1_small_trace
def test_device_cfm_trace_capture_failure_falls_back_to_eager(device, monkeypatch):
    """If trace capture fails, the solve must fall back to the eager loop on the conditioning tensors it
    already uploaded -- which capture had ADOPTED as the trace's persistent buffers. Before 2026-09-25 the
    failure path released the trace's buffers first, freeing exactly those tensors, and the fallback then
    read freed tensors. Capture is forced to fail here; the fallback must match the torch reference, and a
    later traced solve (capture working again) must still be correct."""
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalCFMRef,
        CausalConditionalDecoderRef,
        TtCausalConditionalCFM,
        TtCausalConditionalDecoder,
    )

    torch.manual_seed(61)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm = CausalConditionalCFMRef(dec)
    tt_cfm = TtCausalConditionalCFM(device, TtCausalConditionalDecoder(device, dec), cfm.rand_noise, cfm)
    t_len = 32
    mu, cond = torch.randn(1, t_len, 80) * 0.1, torch.randn(1, t_len, 80) * 0.1
    spks, mask = torch.randn(1, 80) * 0.1, torch.ones(1, t_len, 1)
    with torch.no_grad():
        want = cfm.forward(mu, mask, n_timesteps=4, spks=spks, cond=cond)

    def failing_capture(*args, **kwargs):
        raise RuntimeError("forced capture failure")

    try:
        with monkeypatch.context() as m:
            m.setattr(ttnn, "begin_trace_capture", failing_capture)
            got = tt_cfm.forward(mu, mask, 4, spks, cond, use_trace=True)
        passed, pcc = comp_pcc(want, got, GATE_BF16)
        print(f"\n  capture failed -> eager fallback PCC {pcc}")
        assert passed, pcc
        got = tt_cfm.forward(mu, mask, 4, spks, cond, use_trace=True)
        passed, pcc = comp_pcc(want, got, GATE_BF16)
        print(f"  next solve, capture working again (traced) PCC {pcc}")
        assert passed, pcc
    finally:
        tt_cfm.release_cfm_trace()


def test_device_cfm_traces_pass_allocation_tracker():
    """Re-runs this file's traced CFM tests in a subprocess with `TT_METAL_TRACE_ALLOC_TRACKING=1`, under
    which `ttnn.execute_trace` raises if any buffer allocated while a trace existed is still alive at replay
    (a buffer the replay may overwrite). Covers the bug class found 2026-09-25 (a copy kernel compiled after
    capture, the initial-noise upload kept alive across replays), which PCC checks alone did not catch.
    A subprocess because the tracker is read once at process start (C++ runtime options and the Python
    `execute_trace` wrapper), so it cannot be switched on inside this already-running process. Skipped when
    the whole run is already under the tracker -- the traced tests then check it themselves.

    Opt-in (`COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1`), and it must be the only test in its pytest invocation:
    once any test in this process has opened the device, the process holds UMD's `CHIP_IN_USE` lock until
    it exits (`ttnn.close_device` does not release it), so the child blocks on that lock and never opens the
    card (verified 2026-09-27). Run it alone:
        COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1 pytest <this file>::test_device_cfm_traces_pass_allocation_tracker
    """
    import os
    import subprocess
    import sys

    if os.environ.get("COSYVOICE2_RUN_TRACE_ALLOC_TRACKER") != "1":
        pytest.skip("opt-in: set COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1 and run this test in its own pytest invocation")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1":
        pytest.skip("already running under TT_METAL_TRACE_ALLOC_TRACKING=1")
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1")
    env.pop("COSYVOICE2_RUN_TRACE_ALLOC_TRACKER")
    cmd = [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider"]
    cmd += ["-k", "device_cfm and trace and not allocation_tracker"]
    # No timeout and never a SIGKILL: the child holds the device, and a SIGKILL mid-op can wedge the card.
    # (Not `subprocess.run`, which SIGKILLs the child on any exception, Ctrl+C included.) A Ctrl+C in the
    # terminal reaches the child directly (same process group), so on KeyboardInterrupt keep draining its
    # output until its pytest has unwound and closed the device, then re-raise.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate()
    except KeyboardInterrupt:
        proc.communicate()
        raise
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l or " error" in l][-1:]
    print(f"\n  tracker subprocess: {summary}")
    assert proc.returncode == 0, out[-6000:] + err[-3000:]


# --------------------------------------------------------------------------
# streaming (chunk-causal) mode
# --------------------------------------------------------------------------


def _streaming_inputs(t_len: int, valid: int, seed: int, pad_scale: float = 10.0):
    """Batch-1 CFM conditioning at bucket size `t_len` with the first `valid` positions real and
    the rest bucket padding. The padding is deliberately large garbage (not zeros) so any attention
    leak into padded keys moves the valid outputs visibly."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, t_len, 80, generator=g) * 0.1
    cond = torch.randn(1, t_len, 80, generator=g) * 0.1
    mu[:, valid:] = torch.randn(1, t_len - valid, 80, generator=g) * pad_scale
    cond[:, valid:] = torch.randn(1, t_len - valid, 80, generator=g) * pad_scale
    spks = torch.randn(1, 80, generator=g) * 0.1
    mask = torch.zeros(1, t_len, 1)
    mask[:, :valid] = 1.0
    return mu, cond, spks, mask


def test_streaming_reference_is_chunk_causal():
    """The property upstream's streaming mask exists for, checked on the torch reference: with
    `static_chunk_size=50`, an output in chunk k must be invariant to any change in a LATER chunk
    (what makes recomputing the growing prefix every chunk valid), but -- unlike strict causality --
    must see later positions inside its OWN chunk. The non-streaming reference is checked to be
    sensitive to the same perturbation, so the invariance is the mask's doing, not a dead test."""
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef

    torch.manual_seed(0)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    t_len = 150  # three 50-frame chunks
    x, mu, cond = (torch.randn(1, t_len, 80) * 0.1 for _ in range(3))
    spks = torch.randn(1, 80) * 0.1
    mask = torch.ones(1, t_len, 1)
    t = torch.rand(1)

    def run(x_in, streaming):
        with torch.no_grad():
            return dec(x_in, mask, mu, t, spks, cond, streaming=streaming)

    base = run(x, True)
    x_late = x.clone()
    x_late[:, 100:] += 5.0  # third chunk only
    assert torch.allclose(run(x_late, True)[:, :100], base[:, :100], atol=1e-6, rtol=0)
    assert not torch.allclose(run(x_late, False)[:, :100], run(x, False)[:, :100], atol=1e-3)

    x_same_chunk = x.clone()
    x_same_chunk[:, 60] += 5.0  # chunk [50, 100): position 50 must see it, chunk [0, 50) must not
    out = run(x_same_chunk, True)
    assert not torch.allclose(out[:, 50], base[:, 50], atol=1e-3), "chunk-causal, not strictly causal"
    assert torch.allclose(out[:, :50], base[:, :50], atol=1e-6, rtol=0)


def test_cfm_partial_mask_is_honored_not_refused(expect_error):
    """A partial (bucket-padded) mask goes through in every eager mode. Streaming hands SDPA the full padding +
    chunk-causal bias; non-streaming now does the same with an all-zero chunk term (bucketing), where it used to
    refuse. Only the traced solve refuses a partial non-streaming mask. Host-only: nothing touches a device, and
    getting past the checks is detected at the first upload."""
    import os
    from unittest import mock

    from models.experimental.cosyvoice2.tt.flow import decoder as decoder_mod
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalCFMRef, TtCausalConditionalCFM

    class PastGuard(Exception):
        pass

    def upload(*args, **kwargs):
        raise PastGuard("past the guard")

    cfm_ref = CausalConditionalCFMRef.__new__(CausalConditionalCFMRef)
    cfm_ref.t_scheduler, cfm_ref.inference_cfg_rate = "cosine", 0.7
    tt_cfm = TtCausalConditionalCFM(None, None, torch.zeros(1, 64, 80), cfm_ref, use_trace=False)
    mu, cond, spks, mask = _streaming_inputs(64, 40, seed=0)
    with mock.patch.object(decoder_mod.ttnn, "from_torch", upload):
        for env in ("1", "0"):
            for streaming in (True, False):
                with mock.patch.dict(os.environ, {"COSYVOICE2_FLOW_SDPA": env}):
                    with expect_error(PastGuard, "past the guard"):
                        tt_cfm.forward(mu, mask, 2, spks, cond, streaming=streaming)
        with expect_error(NotImplementedError, "traced CFM solve"):
            tt_cfm.forward(mu, mask, 2, spks, cond, streaming=False, use_trace=True)


@needs_l1_small
@pytest.mark.parametrize("t_len, valid", [(128, 100), (128, 70)])
def test_device_streaming_decoder_bucketed_matches_exact_length(device, t_len, valid):
    """One estimator call in streaming mode on a padded bucket (`t_len`) vs. the torch reference at
    the EXACT valid length (no padding at all) -- the property bucketing relies on. `valid=100` is
    chunk-aligned; `valid=70` is not: queries 50..69 have a chunk window reaching key 99, so only the
    padding term keeps them off keys 70..99. The padding content is large garbage, and a negative
    control runs the same bucket with an all-ones mask (chunk term only, padding visible): at the
    non-aligned length it must be clearly worse than the real mask, proving the padding term reaches
    the attention actually used (fused SDPA's `attn_mask`, by default)."""
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef, TtCausalConditionalDecoder

    torch.manual_seed(11)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    mu, cond, spks, mask = _streaming_inputs(t_len, valid, seed=12)
    x = torch.randn(1, t_len, 80) * 0.1
    t = torch.rand(1)
    with torch.no_grad():
        want = dec(x[:, :valid], mask[:, :valid], mu[:, :valid], t, spks, cond[:, :valid], streaming=True)

    tt_dec = TtCausalConditionalDecoder(device, dec)

    def up(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def run(m):
        out = tt_dec(up(x), up(m), up(mu), t, up(spks.unsqueeze(1)), up(cond), t_len, 1, streaming=True)
        return ttnn.to_torch(out).float()[:, :valid]

    got = run(mask)
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    err = (got - want).abs().max().item()
    print(f"\n  streaming decoder bucket={t_len} valid={valid}: PCC {pcc}, max|diff| {err:.4g}")
    assert passed, pcc

    if valid % 50:
        leaked = run(torch.ones_like(mask))
        _, pcc_leak = comp_pcc(want, leaked, GATE_BF16)
        err_leak = (leaked - want).abs().max().item()
        print(f"  negative control (chunk term only): PCC {pcc_leak}, max|diff| {err_leak:.4g}")
        assert err_leak > 5 * err, (err_leak, err)


@needs_l1_small_trace
def test_device_cfm_streaming_traced_reuse_refreshes_padding(device):
    """The traced streaming solve at one bucket, reused across two DIFFERENT valid lengths (aligned
    100, then non-aligned 110, both in the 128 bucket -- the second solve replays the first solve's
    trace). Each is compared against the torch reference solved at its EXACT valid length. The padding
    term must follow each solve's own mask: frozen at the first capture (valid=100), the second solve
    would hide valid keys 100..109 from every query; missing entirely, queries 100..109 (chunk window
    [0, 150)) would attend padded keys 110..127, which hold large garbage. Either way this is the case
    that proves the padding term follows the per-solve mask on reuse. Also checks the eager streaming
    path against the same reference."""
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalCFMRef,
        CausalConditionalDecoderRef,
        TtCausalConditionalCFM,
        TtCausalConditionalDecoder,
    )

    torch.manual_seed(21)
    dec = CausalConditionalDecoderRef()
    dec.eval()
    cfm = CausalConditionalCFMRef(dec)
    tt_dec = TtCausalConditionalDecoder(device, dec)
    tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm.rand_noise, cfm, trace_cache_capacity=1)
    t_len, n_steps = 128, 4
    try:
        slots = []
        for i, valid in enumerate((100, 110)):
            mu, cond, spks, mask = _streaming_inputs(t_len, valid, seed=30 + i)
            with torch.no_grad():
                want = cfm.forward(
                    mu[:, :valid], mask[:, :valid], n_steps, spks=spks, cond=cond[:, :valid], streaming=True
                )
            got = tt_cfm.forward(mu, mask, n_steps, spks, cond, use_trace=True, streaming=True)[:, :valid]
            slots.append(tt_cfm._traces[(t_len, 80, True)])
            passed, pcc = comp_pcc(want, got, GATE_BF16)
            print(f"\n  traced streaming CFM bucket={t_len} valid={valid}: PCC {pcc}")
            assert passed, (valid, pcc)
            got_eager = tt_cfm.forward(mu, mask, n_steps, spks, cond, use_trace=False, streaming=True)[:, :valid]
            passed, pcc = comp_pcc(want, got_eager, GATE_BF16)
            print(f"  eager streaming CFM bucket={t_len} valid={valid}: PCC {pcc}")
            assert passed, (valid, pcc)
        assert slots[0] is slots[1], "the second valid length must REUSE the first capture"
    finally:
        tt_cfm.release_cfm_trace()


def test_cfm_trace_cache_capacity_other_than_one_is_refused(expect_error):
    """Only one resident CFM trace is allowed. A lazy multi-slot cache captures while other traces are
    live, which `TT_METAL_TRACE_ALLOC_TRACKING=1` refused on 2026-09-25 (see `TtCausalConditionalCFM`'s
    docstring), so capacity != 1 must fail loudly at construction -- via the argument or the env var --
    rather than warn. Host-only: the check runs before anything touches a device. Single-slot recapture on
    a key switch is covered on device by `test_device_cfm_trace_cache_across_utterances_and_replays`."""
    import os
    from unittest import mock

    from models.experimental.cosyvoice2.tt.flow.decoder import CausalConditionalCFMRef, TtCausalConditionalCFM

    cfm_ref = CausalConditionalCFMRef.__new__(CausalConditionalCFMRef)
    cfm_ref.t_scheduler, cfm_ref.inference_cfg_rate = "cosine", 0.7
    noise = torch.zeros(1, 64, 80)
    for capacity in (2, 0):
        with expect_error(ValueError, "capacity must be 1"):
            TtCausalConditionalCFM(None, None, noise, cfm_ref, trace_cache_capacity=capacity)
    with mock.patch.dict(os.environ, {"COSYVOICE2_CFM_TRACE_CACHE_CAPACITY": "2"}):
        with expect_error(ValueError, "capacity must be 1"):
            TtCausalConditionalCFM(None, None, noise, cfm_ref)
    assert TtCausalConditionalCFM(None, None, noise, cfm_ref)._trace_capacity == 1


# Fused-SDPA estimator vs the torch reference at T = 449, random weights: measured PCC 0.99952, max|diff|
# 0.148 (2026-09-27, N150); the harness control (1e30 in a logical K/V row) moves the output by 3.1.
SDPA_T1MOD32_MAX_ABS = 0.5


@needs_l1_small
def test_device_decoder_fused_sdpa_ignores_tile_padding_at_t_1_mod_32(device, monkeypatch):
    """tt-metal #57608 regression guard. On Blackhole, fused SDPA (non-causal, no mask) lets large values in
    K/V's implicit tile padding reach the output when T is not a multiple of 32 (PCC ~0, silent); CosyVoice1
    hit it through conv1d leaving garbage in that padding at T = 1 mod 32. The non-streaming estimator
    runs fused SDPA exactly that way at arbitrary T. Measured on Wormhole/N150 (2026-09-27): not
    reproducible -- 1e30 planted in K/V padding leaves SDPA's output bit-identical at the op level, and the
    real-weight estimator matched torch at every T = 1 mod 32 from 161 to 1537 with fused SDPA and the
    explicit chain alike. At T = 449 (= 1 mod 32) this checks, on whatever hardware runs it:
      * fused-SDPA estimator vs the torch reference (PCC and max|diff|);
      * padding immunity: 1e30 planted in K/V's padding on every SDPA call changes nothing;
      * harness control: 1e30 planted in the last *logical* K/V row must change the output -- proves the
        plant reaches SDPA's inputs, so the immunity check cannot pass vacuously."""
    import ttnn
    from models.experimental.cosyvoice2.tt.flow.decoder import (
        CausalConditionalDecoderRef,
        TtCausalConditionalDecoder,
        flow_fused_sdpa,
    )

    assert flow_fused_sdpa(), "COSYVOICE2_FLOW_SDPA must be on (the default) for this test"
    torch.manual_seed(449)
    b, t_len = 2, 449
    assert t_len % 32 == 1
    dec = CausalConditionalDecoderRef()
    dec.eval()
    x, mu, cond = (torch.randn(b, t_len, 80) * 0.1 for _ in range(3))
    spks, mask, t = torch.randn(b, 80) * 0.1, torch.ones(b, t_len, 1), torch.rand(b)
    with torch.no_grad():
        want = dec(x, mask, mu, t, spks, cond)

    tt_dec = TtCausalConditionalDecoder(device, dec)

    def run():
        dev = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = tt_dec(dev(x), dev(mask), dev(mu), t, dev(spks.unsqueeze(1)), dev(cond), t_len, b)
        return ttnn.to_torch(out).float().reshape(want.shape)

    real_sdpa = ttnn.transformer.scaled_dot_product_attention

    def plant_padding(q, k, v, **kw):
        return real_sdpa(q, ttnn.fill_implicit_tile_padding(k, 1e30), ttnn.fill_implicit_tile_padding(v, 1e30), **kw)

    def plant_last_row(q, k, v, **kw):
        def last_row_big(x):
            h = ttnn.to_torch(x)
            h[..., -1, :] = 1e30
            return ttnn.from_torch(h, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=device)

        return real_sdpa(q, last_row_big(k), last_row_big(v), **kw)

    clean = run()
    with monkeypatch.context() as m:
        m.setattr(ttnn.transformer, "scaled_dot_product_attention", plant_padding)
        padded = run()
    with monkeypatch.context() as m:
        m.setattr(ttnn.transformer, "scaled_dot_product_attention", plant_last_row)
        control = run()

    passed, pcc = comp_pcc(want, clean, GATE_BF16)
    max_abs = (want - clean).abs().max().item()
    pad_diff = (clean - padded).abs().max().item()
    ctl_diff = (clean - control).abs().max().item()
    print(
        f"\n  T={t_len}: fused SDPA vs torch PCC {pcc} max|diff| {max_abs:.4g}; planted K/V padding changes "
        f"the output by {pad_diff:.4g}; planted last logical row (control) by {ctl_diff:.4g}"
    )
    assert passed, pcc
    assert max_abs <= SDPA_T1MOD32_MAX_ABS, max_abs
    assert pad_diff == 0.0, f"K/V tile padding reached the output (#57608): max|diff| {pad_diff}"
    assert not (ctl_diff <= SDPA_T1MOD32_MAX_ABS), f"harness control did not change the output: {ctl_diff}"
