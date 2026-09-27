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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import CausalConv1dRef

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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import CausalBlock1DRef

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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import sinusoidal_pos_emb_torch

    emb = sinusoidal_pos_emb_torch(torch.zeros(3), dim=320)
    assert emb.shape == (3, 320)
    assert torch.allclose(emb[:, :160], torch.zeros(3, 160), atol=1e-6)
    assert torch.allclose(emb[:, 160:], torch.ones(3, 160), atol=1e-6)


def test_decoder_torch_reference_shape_and_range():
    from models.demos.audio.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef

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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import CausalConv1dRef, TtCausalConv1d

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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import CausalConditionalDecoderRef, TtCausalConditionalDecoder

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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import (
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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import (
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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import (
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
    from models.demos.audio.cosyvoice2.tt.flow.decoder import (
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
    import signal
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
    # On timeout, interrupt the child rather than kill it: `subprocess.run(timeout=...)` would SIGKILL a
    # process that holds the device, and a SIGKILL mid-op can wedge the card. SIGINT lets the child's pytest
    # unwind its fixtures and close the device; SIGKILL only if it is still alive after the grace period.
    timeout_s, grace_s = 1800, 120
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            out, err = proc.communicate(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        pytest.fail(f"tracker subprocess timed out after {timeout_s}s\n{out[-6000:]}{err[-3000:]}")
    summary = [l for l in out.splitlines() if " passed" in l or " failed" in l or " error" in l][-1:]
    print(f"\n  tracker subprocess: {summary}")
    assert proc.returncode == 0, out[-6000:] + err[-3000:]
