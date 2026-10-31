# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text to speech, end to end: real text + a preset voice -> a real 24 kHz waveform, against the reference.

THE CHAIN UNDER TEST IS THE ONE THE DEMO RUNS. Both call `pipeline.run_text_to_speech`, so there
is one copy of the wiring and a green test cannot coexist with a broken demo.

THE INPUT is the model's own speech-request layout,
`[BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT] <text> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]`, with the
`casual_male` voice embedding (the checkpoint's `voice_embedding/casual_male.pt`) substituted into the N
placeholders -- 32 distinct texts, one per row.

THE HORIZON is the model's own stop rule: each row runs until its semantic head emits
`end_audio` (id read off the reference), and the batch runs until every row has. The safety cap
(`common.resolve_max_frames`) only bounds a run that never stops, and the test ASSERTS it did not
bind. Both the TT run and the free-running HF reference use the identical rule. The WHOLE output is
compared -- every frame of every row.

NOTHING IS EVER SPLICED INTO THE TT SIDE. It runs completely free, exactly as the demo does. The
reference is TEACHER-FORCED onto the TT trajectory (`fed_codes`, `fed_hidden`): at every frame it
answers "given exactly the context this pipeline produced, what does torch compute?", and each
stage is scored on the input it actually consumed -- prefill hidden, every frame's backbone hidden,
semantic logits, flow-sampler output, the discrete codes, and the final waveform. A free-running
TT-vs-free-running-HF comparison measures how fast two chaotic trajectories separate, not the port:
the acoustic codes are a round onto 21 levels, and a perturbation smaller than ONE matmul's rounding
on this device (1.2e-3 relative, measured against float64) already scrambles later frames. The
free-running HF golden is still run -- it is the audio the quality scores are calibrated against.

A RENDERED SIGNAL IS SCORED, not only correlated: Whisper WER against the requested text, and a
UTMOS22 naturalness estimate, over each row's full output cut at its own end frame. The thresholds
are read off the HF golden scored the same way (`WER_MARGIN`, `MOS_MARGIN`).
"""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.voxtral_4b_tts_2603.reference import golden
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

# The HF golden is a 4 B model on CPU run to the model's own stop rule over the WHOLE output, twice
# (free + teacher-forced): ~16 s per frame at B=32 on this host, ~50 min on a first run. Both arms
# are memoised on disk (`common.cached_golden`), so a rerun against an unchanged pipeline pays only
# the device pass and the scoring.
pytestmark = pytest.mark.timeout(7200)

PCC_TARGET = 0.99

# The one test a consumer that takes a SINGLE node should run as the correctness gate -- optimize
# re-runs it after every change and reverts whatever fails it. It is the discrete-code test, not
# the waveform PCC: the waveform golden is teacher-forced onto THIS pipeline's own codes, so it
# moves with the pipeline and cannot fail for the stages that produce those codes.
E2E_CORRECTNESS_GATE = "test_discrete_codes_equal_the_teacher_forced_reference"

# The TT output may be at most this much worse than the HF golden on the same 32 prompts.
WER_MARGIN = 0.05  # absolute corpus word error rate
MOS_MARGIN = 0.20  # mean UTMOS22, on its 1-5 scale
CLIP_MOS_FLOOR = 2.0  # no single clip below this where the HF golden's same clip is above it

# A reference decision is DECIDABLE when its own margin clears this many standard deviations of
# the error the device's arithmetic puts on it; codes inside that band are ties and are counted,
# not waved through.
# semantic argmax: TIE_SIGMA x the measured RMS logit deviation. acoustic codes: TIE_SIGMA x that
# element's own spread under the device's measured matmul error, over NOISE_DRAWS torch draws.
TIE_SIGMA = 6.0
NOISE_DRAWS = 4

# THE TIE BANDS ARE FIXED NUMBERS, not read off the run being judged. They used to scale with the TT
# stage's own measured error, so a worse acoustic stage widened its own tolerance: the pre-fix branch
# had a held-out x_final error of 1.1e-1 (26x this build's) and its band then covered every code, so
# the test passed with 70% of acoustic codes wrong. The values below were measured on the
# accuracy-fixed build and are identical on a p300c chip and a p150 (held-out x_final RMS deviation
# 4.195e-3 at a single-matmul floor of 4.906e-4 -> noise-model scale 1.648e-3; semantic logit RMS
# deviation 1.53-1.56e-3).
ACOUSTIC_NOISE_SCALE = 1.648e-3  # relative Linear-output noise the per-element spread is drawn at
ACOUSTIC_BASELINE_RMS = 4.195e-3  # the stage's held-out x_final RMS error, the band's floor
SEMANTIC_LOGIT_RMS = 1.6e-3  # RMS semantic-logit deviation the semantic tie band is built on
# RMS frame-0 semantic-logit deviation, TT vs the FREE-running reference (which runs its own prefill), so it
# includes the prefill hidden state's deviation, not only the semantic head's: 6.09e-2 measured on a p150.
FRAME0_LOGIT_RMS = 6.1e-2
# Absolute limits on the stage itself, so a degraded stage fails here instead of only widening ties.
MAX_HELDOUT_RMS = 2.0 * ACOUSTIC_BASELINE_RMS  # held-out x_final RMS deviation, TT vs reference
MIN_ACOUSTIC_AGREEMENT = 0.98  # fraction of live acoustic codes equal to the teacher-forced reference
MAX_SEMANTIC_LOGIT_RMS = 2.0 * SEMANTIC_LOGIT_RMS


def sigma_upper_bound(sample_std, n: int, alpha: float = 0.05):
    """The (1 - alpha) upper confidence bound on a Gaussian sigma estimated from `n` draws.

    A standard deviation from 4 draws is frequently far below the true one (measured on this run:
    0.058 from 4 draws, 0.102 from 32, at the same element), and a band built on an underestimate
    fails codes the reference cannot decide. `s * sqrt((n - 1) / chi2.ppf(alpha, n - 1))`.
    """
    from scipy.stats import chi2

    return sample_std * float(((n - 1) / chi2.ppf(alpha, n - 1)) ** 0.5)


def measure_matmul_floor(device) -> float:
    """Relative RMS error of ONE device matmul, measured here, against float64.

    The configuration the acoustic modules run: a float32 activation against a weight that is exactly
    bfloat16-representable (this checkpoint's acoustic weights are), HiFi4 + fp32 DEST
    accumulation. M=32 x K=3072 x N=3072 -- the width of the acoustic transformer.
    """
    import ttnn

    gen = torch.Generator().manual_seed(0)
    a = torch.randn(32, 3072, generator=gen)
    w = (torch.randn(3072, 3072, generator=gen) / 3072**0.5).to(torch.bfloat16).float()
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    up = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    wt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.linear(ttnn.from_torch(a, **up), wt, compute_kernel_config=cfg)).double()
    ref = a.double() @ w.double()
    return float((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


@pytest.fixture(scope="module")
def evidence(device, hf_model):
    """Run the TT pipeline once, then BOTH reference arms, and hand all of it to every assertion."""
    common.use_all_cpu_threads()
    max_frames, provenance = common.resolve_max_frames(hf_model)
    input_ids, audio_mask, voice_embedding, texts = pipeline.speech_inputs(batch=common.DEFAULT_BATCH)
    pipe = pipeline.build_pipeline(
        device,
        model=hf_model,
        kv_capacity=pipeline.tts_kv_capacity(input_ids.shape[-1], max_frames),
    )
    batch = pipe.batch
    assert input_ids.shape[0] == batch
    voice = pipe.stage_voice(audio_mask, voice_embedding, input_ids=input_ids)
    x0 = pipe.noise(max_frames, batch=batch)
    cfg_alpha = torch.full((batch,), pipeline.DEFAULT_CFG_ALPHA)

    print(f"\nbatch driven (read from the pipeline): {batch}")
    print(f"prompt tokens: {input_ids.shape[1]}  (voice block {int(audio_mask[0].sum())} + text + controls)")
    print(f"safety cap: {max_frames}  <- {provenance}")
    print(f"stop rule: semantic code == end_audio (id {pipe.stop_token_id}) on every row, read off the reference")

    tt = pipe.run_text_to_speech(
        input_ids=input_ids, x0=x0, cfg_alpha=cfg_alpha, max_frames=max_frames, collect=True, voice=voice
    )

    base = dict(
        task="tts",
        ids=input_ids,
        voice=voice_embedding,
        x0=x0,
        cfg=cfg_alpha,
        max_frames=max_frames,
        B=batch,
        chain=golden.CHAIN_VERSION,
    )
    shared = {}

    def prefill():
        # Computed at most once, and only if an arm is not already cached on disk.
        if "state" not in shared:
            shared["state"] = golden.reference_prefill(hf_model, input_ids, audio_mask, voice_embedding)
        return shared["state"]

    voiced = dict(audio_mask=audio_mask, voice_embedding=voice_embedding)
    free = common.cached_golden(
        common.golden_key(arm="free", **base),
        lambda: golden.hf_reference_text_to_speech(
            hf_model, input_ids, x0, cfg_alpha, max_frames, prefill=prefill(), **voiced
        ),
    )
    tt_hidden = torch.stack([d["llm_hidden"] for d in tt["diagnostics"]], dim=-1)
    aligned = common.cached_golden(
        common.golden_key(arm="aligned", codes=tt["codes"], hidden=tt_hidden, **base),
        lambda: golden.hf_reference_text_to_speech(
            hf_model,
            input_ids,
            x0,
            cfg_alpha,
            max_frames,
            fed_codes=tt["codes"],
            fed_hidden=tt_hidden,
            prefill=prefill(),
            **voiced,
        ),
    )
    dump = os.environ.get("VOXTRAL_DUMP_EVIDENCE")
    if dump:
        # Opt-in, for offline analysis of a failure: the TT run's own outputs and the reference
        # teacher-forced on them, so nothing has to be re-run on the device to study it.
        torch.save(
            {"tt": tt, "hf": aligned, "free": free, "input_ids": input_ids, "x0": x0, "cfg_alpha": cfg_alpha}, dump
        )
    return {
        "pipe": pipe,
        "tt": tt,
        "hf": aligned,
        "free": free,
        "batch": batch,
        "texts": texts,
        "input_ids": input_ids,
        "x0": x0,
        "cfg_alpha": cfg_alpha,
        "max_frames": max_frames,
        # Every constant the discretization rule needs, READ OFF THE REFERENCE.
        "levels": int(hf_model.acoustic_transformer.acoustic_embeddings_levels),
        "n_special": golden.n_special_tokens(),
        "empty_id": common.audio_empty_token_id(hf_model),
        "stop_token_id": pipe.stop_token_id,
        "semantic_size": int(hf_model.acoustic_transformer.model_args.semantic_codebook_size),
    }


def test_golden_is_the_references_own_arithmetic(hf_model):
    """The golden replaces one `torch.randn` and nothing else -- proven, not asserted by comment."""
    common.use_all_cpu_threads()
    ids, _ = common.build_batch_inputs(batch=4)
    with torch.no_grad():
        h = hf_model.model(input_ids=ids).last_hidden_state[:, -1]
    ok, from_module, from_helper, x0 = golden.reference_frame_matches_module(
        hf_model, h, torch.full((4,), pipeline.DEFAULT_CFG_ALPHA), seed=0
    )
    print(f"golden faithfulness: module codes == helper codes -> {ok}  (x0 {tuple(x0.shape)})")
    assert ok, (
        "the golden's Euler loop is NOT the reference's arithmetic:\n"
        f"  module {from_module[0, :8].tolist()}\n  helper {from_helper[0, :8].tolist()}"
    )


def test_run_ended_on_the_models_stop_rule(evidence):
    """TERMINATION: the safety cap is a backstop, never the horizon.

    A cap that binds silently shrinks the comparison to a prefix of the real output, and everything
    that degrades after it passes unmeasured. Both sides must have stopped on `end_audio`.
    """
    tt, free, batch = evidence["tt"], evidence["free"], evidence["batch"]
    print(f"\nTT stop: {tt['stop_reason']}  ({tt['frames_decoded']} frames)")
    print(f"HF stop: {free['stop_reason']}  ({free['frames_decoded']} frames)")
    print(f"TT per-row end frames: {tt['end_frame']}")
    print(f"HF per-row end frames: {free['end_frame']}")
    cap = evidence["max_frames"]
    assert tt["stop_reason"].startswith(
        "every row emitted end_audio"
    ), f"the TT run ended on the safety cap ({cap} frames), not on the model's stop rule"
    assert free["stop_reason"].startswith(
        "every row emitted end_audio"
    ), f"the HF golden ended on the safety cap ({cap} frames), not on the model's stop rule"
    assert all(e >= 0 for e in tt["end_frame"]) and len(tt["end_frame"]) == batch


def test_shapes_and_real_task_output(evidence):
    """The output is real audio, not a smoke-test tensor."""
    tt, batch = evidence["tt"], evidence["batch"]
    frames = tt["frames_decoded"]
    assert tuple(tt["codes"].shape) == (batch, 37, frames)
    assert tuple(tt["waveform"].shape) == (batch, 1, frames * 1920)
    assert tt["sampling_rate"] == 24000
    print(
        f"\naudio: {tt['waveform'].shape[-1] / tt['sampling_rate']:.2f} s at {tt['sampling_rate']} Hz ({frames} frames)"
    )
    wav = tt["waveform"]
    assert torch.isfinite(wav).all(), "the waveform contains non-finite samples"
    assert wav.abs().max() <= 1.5, f"waveform out of audio range: max |x| = {float(wav.abs().max())}"
    assert float(wav.reshape(batch, -1).std(dim=1).min()) > 1e-4, "at least one waveform is constant"


def test_batch_is_32_independent_samples(evidence):
    """A pipeline that shape-supports B but emits 32 identical outputs is WRONG."""
    tt, batch = evidence["tt"], evidence["batch"]
    assert batch == common.DEFAULT_BATCH == 32
    rows = {tuple(r.tolist()) for r in evidence["input_ids"]}
    waves = {tt["waveform"][i].numpy().tobytes() for i in range(batch)}
    codes = {tt["codes"][i].numpy().tobytes() for i in range(batch)}
    print(f"\ndistinct inputs {len(rows)}/{batch}; code streams {len(codes)}/{batch}; waveforms {len(waves)}/{batch}")
    assert len(rows) == batch, "the 32 inputs are not pairwise distinct"
    assert len(waves) == batch, f"only {len(waves)} of {batch} waveforms are distinct"


def test_per_stage_pcc(hf_model, evidence):
    """EVERY joint of the chain, against the reference driven by this pipeline's own trajectory.

    The acoustic output `x_final` is checked POOLED (one PCC over every row and frame) plus PER FRAME
    against that frame's own sensitivity. A worst-of-2656 PCC is not a well-posed gate there: some
    frames are ill-conditioned (CFG 3.0 over 7 Euler steps), and the reference's own x_final moves by
    RMS ~0.13 at such a frame under one device matmul's rounding, while a typical frame's TT error is
    ~0.0013 (measured on a p150, where the worst frame then reads PCC 0.976).
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    frames = tt["frames_decoded"]
    diag_tt, diag_hf = tt["diagnostics"], hf["diagnostics"]
    assert len(diag_tt) == frames and len(diag_hf) == frames

    def ratio(a, b):
        return float(a.norm() / b.norm())

    prefill = min(common.pcc(tt["prefill_hidden"][i], hf["prefill_hidden"][i]) for i in range(batch))
    print(
        f"\nstage PCC  prefill hidden  (min over {batch})          = {prefill:.6f}  "
        f"|tt|/|ref|={ratio(tt['prefill_hidden'], hf['prefill_hidden']):.5f}"
    )
    per_frame = [
        min(common.pcc(diag_tt[t]["llm_hidden"][i], hf["llm_hiddens"][t][i]) for i in range(batch))
        for t in range(frames)
    ]
    hidden = min(per_frame)
    tt_h = torch.stack([d["llm_hidden"] for d in diag_tt])
    hf_h = torch.stack(hf["llm_hiddens"][:frames])
    print(
        f"stage PCC  decode hidden   (min over {batch} x {frames}) = {hidden:.6f}  |tt|/|ref|={ratio(tt_h, hf_h):.5f}"
    )
    print(f"           worst frame {int(torch.tensor(per_frame).argmin())}; last-frame min {per_frame[-1]:.6f}")

    semantic = min(
        common.pcc(diag_tt[t]["semantic_logits"][i], diag_hf[t]["semantic_logits_raw"][i])
        for t in range(frames)
        for i in range(batch)
    )
    print(f"stage PCC  semantic logits (min over {batch} x {frames}) = {semantic:.6f}")
    x_tt = torch.stack([diag_tt[t]["x_final"].clamp(-1, 1) for t in range(frames)], dim=-1)  # [B, 36, F]
    x_hf = torch.stack([diag_hf[t]["x_final"] for t in range(frames)], dim=-1)
    x_final = common.pcc(x_tt, x_hf)
    frame_rms = (x_tt - x_hf).pow(2).mean(dim=1).sqrt()  # [B, F]
    print(
        f"stage PCC  x_final         (pooled over {batch} x {frames}) = {x_final:.6f}; per-frame RMS error "
        f"median {float(frame_rms.median()):.2e}, max {float(frame_rms.max()):.2e}"
    )
    # A frame above the baseline band is checked against ITS OWN spread: the reference's x_final
    # under the device's measured matmul noise (the same per-element model the codes test uses).
    tt_hidden = torch.stack([d["llm_hidden"] for d in diag_tt], dim=-1)
    over = []
    for f in sorted(set(torch.nonzero(frame_rms > TIE_SIGMA * ACOUSTIC_BASELINE_RMS)[:, 1].tolist())):
        rows = sorted(set(torch.nonzero(frame_rms[:, f] > TIE_SIGMA * ACOUSTIC_BASELINE_RMS)[:, 0].tolist()))
        h, x0f, cfg = tt_hidden[rows, :, f], evidence["x0"][f][rows], evidence["cfg_alpha"][rows]
        spread_rows = common.cached_golden(
            common.golden_key(
                arm="spread-rowrms", hidden=h, x0=x0f, cfg=cfg, eps=round(ACOUSTIC_NOISE_SCALE, 5), draws=NOISE_DRAWS
            ),
            lambda: golden.acoustic_spread_under_matmul_noise(
                hf_model, h, x0f, cfg, ACOUSTIC_NOISE_SCALE, draws=NOISE_DRAWS
            ),
        )
        spread_rms = sigma_upper_bound(spread_rows, NOISE_DRAWS).pow(2).mean(dim=-1).sqrt()
        for k, b in enumerate(rows):
            limit = TIE_SIGMA * max(float(spread_rms[k]), ACOUSTIC_BASELINE_RMS)
            err = float(frame_rms[b, f])
            print(
                f"           x_final f{f} row{b}: RMS error {err:.4f}, own spread {float(spread_rms[k]):.4f}, limit {limit:.4f}"
            )
            if err > limit:
                over.append((f, b, err, limit))
    assert prefill >= PCC_TARGET, f"the text stack is below target at {prefill:.6f}"
    assert hidden >= PCC_TARGET, (
        f"the decode step drifts: frame-wise hidden PCC {hidden:.6f}. The reference is fed THIS "
        f"pipeline's own codes, so a drop here is the KV cache, positions or the audio-token embedding"
    )
    assert semantic >= PCC_TARGET, f"the semantic head is at {semantic:.6f}"
    assert x_final >= PCC_TARGET, f"the acoustic flow sampler is at pooled PCC {x_final:.6f}"
    assert not over, f"x_final frames off by more than their own spread allows: {over}"


def test_discretization_is_the_references_own_rule(evidence):
    """The on-device discretization IS the reference's rule, checked EXACTLY on the pipeline's own values.

    `round(((clamp(x, -1, 1) + 1) / 2) * (levels - 1)) + n_special`, `empty_audio` on a stopped row;
    and the semantic code is the reference's masked argmax. Applied in torch to THIS pipeline's
    x_final / logits, it must reproduce the pipeline's codes bit for bit.
    """
    tt = evidence["tt"]
    frames, levels = tt["frames_decoded"], evidence["levels"]
    n_special, stop_id = evidence["n_special"], evidence["stop_token_id"]
    scale = 0.5 * (levels - 1)

    x_tt = torch.stack([tt["diagnostics"][t]["x_final"] for t in range(frames)], dim=-1)
    rule = ((x_tt.clamp(-1, 1) + 1) * scale).round().long() + n_special
    stopped = (tt["codes"][:, 0, :] == stop_id).unsqueeze(1).expand_as(rule)
    rule = torch.where(stopped, torch.full_like(rule, evidence["empty_id"] + n_special), rule)
    mismatch = int((tt["codes"][:, 1:, :] != rule).sum())
    print(f"\ndiscretization rule on the pipeline's x_final reproduces {rule.numel() - mismatch}/{rule.numel()} codes")

    lo_tt = torch.stack([tt["diagnostics"][t]["semantic_logits"] for t in range(frames)], dim=-1)
    masked = lo_tt.clone()
    masked[:, evidence["empty_id"], :] = -float("inf")
    masked[:, n_special + evidence["semantic_size"] :, :] = -float("inf")
    sem_mismatch = int((tt["codes"][:, 0, :] != masked.argmax(dim=1)).sum())
    print(
        f"masked argmax on the pipeline's logits reproduces {masked[:, 0].numel() - sem_mismatch}/{masked[:, 0].numel()}"
    )
    assert mismatch == 0, f"{mismatch} acoustic codes are not the reference's rule on the pipeline's own x_final"
    assert sem_mismatch == 0, f"{sem_mismatch} semantic codes are not the reference's masked argmax"


def calibrate_stage_noise(pipe, hf_model, floor_eps):
    """The relative error scale at which the torch sampler's noise model matches THIS STAGE on device.

    Matmul rounding is not the stage's only error source (the activation x activation attention
    matmuls, the elementwise exp/rsqrt/silu), so the measured single-matmul floor under-predicts the
    stage. The scale is therefore calibrated on a HELD-OUT input -- the stage's own
    `acoustic_trace_inputs()`, the reference backbone's hidden state for other prompts, not this run -- by
    running the TT stage and the torch reference on it, and scaling the noise model until its
    x_final spread matches the TT deviation. Never below the measured matmul floor.
    """
    inputs = pipe.acoustic_trace_inputs()
    buf = pipe.acoustic_trace_setup(inputs)
    probe = {}
    pipe.acoustic.decode_frame(buf["llm_hidden"], buf["x0"], buf["cfg_alpha"], probe=probe)
    import ttnn

    x_tt = ttnn.to_torch(probe["x_final"]).float().clamp(-1, 1)
    with torch.no_grad():
        _, diag = golden.acoustic_frame(hf_model, inputs["llm_hidden"], inputs["x0"], inputs["cfg_alpha"])
    tt_rms = float((x_tt - diag["x_final"]).pow(2).mean().sqrt())
    spread = golden.acoustic_spread_under_matmul_noise(
        hf_model, inputs["llm_hidden"], inputs["x0"], inputs["cfg_alpha"], floor_eps, draws=NOISE_DRAWS
    )
    model_rms = float(spread.pow(2).mean().sqrt())
    return max(floor_eps, floor_eps * tt_rms / model_rms), tt_rms, model_rms


def test_discrete_codes_equal_the_teacher_forced_reference(device, hf_model, evidence):
    """EXACT AGREEMENT of the discrete output under teacher forcing, wherever the reference decides.

    The reference consumed THIS pipeline's hidden and codes at every frame, so the two ran the same
    trajectory and any disagreement is the pipeline's -- unless the reference's own decision is not
    a decision at the precision this hardware has. That is measured, not assumed:

    * semantic code: a tie when the reference's top-1 minus top-2 logit is under `TIE_SIGMA` RMS
      logit deviations;
    * acoustic code: the flow sampler is ill-conditioned at some elements (CFG alpha=3 over 7 Euler
      steps), so the band is PER ELEMENT: the spread of the reference's own x_final when every Linear
      carries noise at `ACOUSTIC_NOISE_SCALE`, floored at `ACOUSTIC_BASELINE_RMS`. Both are FIXED
      (measured on the accuracy-fixed build), so a worse stage cannot widen its own band. A mismatch
      is a tie only if the reference value sits within `TIE_SIGMA` of that band from the rounding edge.

    Every other mismatch fails. On top of that, three absolute limits: the stage's held-out x_final
    error (`calibrate_stage_noise`) is at most `MAX_HELDOUT_RMS`, at least `MIN_ACOUSTIC_AGREEMENT` of
    the live acoustic codes equal the reference's, and the semantic logits' RMS deviation is at most
    `MAX_SEMANTIC_LOGIT_RMS`.
    """
    tt, hf = evidence["tt"], evidence["hf"]
    frames, levels = tt["frames_decoded"], evidence["levels"]
    scale = 0.5 * (levels - 1)
    live = (tt["codes"][:, 0, :] != evidence["stop_token_id"]).unsqueeze(1)

    x_hf = torch.stack([hf["diagnostics"][t]["x_final"] for t in range(frames)], -1)
    s_hf = (x_hf + 1) * scale
    room = (s_hf - s_hf.floor() - 0.5).abs()
    differ = (tt["codes"][:, 1:, :] != hf["codes"][:, 1:, :]) & live.expand_as(room)
    agree = float((tt["codes"][:, 1:, :] == hf["codes"][:, 1:, :]).float().mean())

    floor_eps = measure_matmul_floor(device)
    _, held_out_rms, cal_model = calibrate_stage_noise(evidence["pipe"], hf_model, floor_eps)
    rel_eps, cal_tt = ACOUSTIC_NOISE_SCALE, ACOUSTIC_BASELINE_RMS
    print(
        f"\nnoise model (FIXED): scale {rel_eps:.3e}, baseline {cal_tt:.3e}. Measured here: one device matmul "
        f"{floor_eps:.3e} relative; held-out TT x_final RMS dev {held_out_rms:.3e} (limit {MAX_HELDOUT_RMS:.3e}) "
        f"vs model {cal_model:.3e} at the floor"
    )
    tt_hidden = torch.stack([d["llm_hidden"] for d in tt["diagnostics"]], dim=-1)
    ties = decided_wrong = 0
    for f in sorted(set(torch.nonzero(differ)[:, 2].tolist())):
        here = differ[..., f]
        # Only the rows that hold a mismatch: every row is an independent sample, so the draw on
        # a subset is the same arithmetic and costs a fraction of the whole batch.
        rows = sorted(set(torch.nonzero(here)[:, 0].tolist()))
        h, x0f, cfg = tt_hidden[rows, :, f], evidence["x0"][f][rows], evidence["cfg_alpha"][rows]
        spread_rows = common.cached_golden(
            common.golden_key(arm="spread-rowrms", hidden=h, x0=x0f, cfg=cfg, eps=round(rel_eps, 5), draws=NOISE_DRAWS),
            lambda: golden.acoustic_spread_under_matmul_noise(hf_model, h, x0f, cfg, rel_eps, draws=NOISE_DRAWS),
        )
        spread = torch.full_like(room[..., f], float("inf"))
        spread[rows] = sigma_upper_bound(spread_rows, NOISE_DRAWS)
        # Two measured error terms, the larger wins: this element's SENSITIVITY (the noise model)
        # and the stage's BASELINE error on the held-out input, which the Linear-only noise model
        # does not reproduce (device sin/cos in the time embedding, the activation x activation
        # attention matmuls, the elementwise ops).
        band = TIE_SIGMA * torch.clamp(spread, min=cal_tt) * scale
        tie = here & (room[..., f] <= band)
        ties += int(tie.sum())
        decided_wrong += int((here & ~tie).sum())
        for b, c in torch.nonzero(here & ~tie).tolist():
            print(
                f"  DECIDABLE mismatch f{f} row{b} cb{c}: ref {float(s_hf[b, c, f]):.4f} room {float(room[b, c, f]):.4f} "
                f"band {float(band[b, c]):.4f}"
            )
    n_live = int(live.expand_as(room).sum())
    agree_live = 1.0 - int(differ.sum()) / max(n_live, 1)
    print(
        f"\nacoustic codes vs teacher-forced reference: agreement {agree:.6f} ({agree_live:.6f} of {n_live} live codes); "
        f"{int(differ.sum())} differ -> {ties} ties (band {TIE_SIGMA} x max(per-element spread at {rel_eps:.3e}, 95% upper bound "
        f"from {NOISE_DRAWS} draws; held-out baseline {cal_tt:.3e})), {decided_wrong} decidable mismatches; codes "
        f"inside the baseline band alone: {int(((room <= TIE_SIGMA * cal_tt * scale) & live.expand_as(room)).sum())}/{n_live}"
    )

    lo_tt = torch.stack([tt["diagnostics"][t]["semantic_logits"] for t in range(frames)], -1)
    lo_hf = torch.stack([hf["diagnostics"][t]["semantic_logits"] for t in range(frames)], -1)
    finite = torch.isfinite(lo_hf)
    measured_ldev = float((lo_tt[finite] - lo_hf[finite]).pow(2).mean().sqrt())
    ldev = SEMANTIC_LOGIT_RMS
    top2 = lo_hf.topk(2, dim=1).values
    sem_decidable = (top2[:, 0] - top2[:, 1]) > TIE_SIGMA * ldev
    sem_differ = tt["codes"][:, 0, :] != hf["codes"][:, 0, :]
    sem_wrong = sem_differ & sem_decidable
    x_tt = torch.stack([tt["diagnostics"][t]["x_final"] for t in range(frames)], -1).clamp(-1, 1)
    live_x = live.expand_as(x_hf)
    pcc_x = common.pcc(x_tt[live_x], x_hf[live_x])
    pcc_lo = common.pcc(lo_tt[finite], lo_hf[finite])
    print(f"x_final vs teacher-forced reference PCC: {pcc_x:.6f}")
    print(f"semantic logits vs teacher-forced reference PCC: {pcc_lo:.6f}")
    print(
        f"semantic codes vs teacher-forced reference: agreement {float((~sem_differ).float().mean()):.6f}; "
        f"{int(sem_differ.sum())} differ -> {int((sem_differ & ~sem_decidable).sum())} ties "
        f"(band {TIE_SIGMA} x RMS {ldev:.3e}, measured {measured_ldev:.3e}), {int(sem_wrong.sum())} decidable mismatches"
    )
    assert (
        held_out_rms <= MAX_HELDOUT_RMS
    ), f"the acoustic stage's held-out x_final RMS error {held_out_rms:.3e} exceeds {MAX_HELDOUT_RMS:.3e}"
    assert (
        agree_live >= MIN_ACOUSTIC_AGREEMENT
    ), f"only {agree_live:.4f} of live acoustic codes match the reference (need {MIN_ACOUSTIC_AGREEMENT})"
    assert (
        measured_ldev <= MAX_SEMANTIC_LOGIT_RMS
    ), f"semantic logit RMS deviation {measured_ldev:.3e} exceeds {MAX_SEMANTIC_LOGIT_RMS:.3e}"
    assert decided_wrong == 0, f"{decided_wrong} DECIDABLE acoustic codes differ from the reference"
    assert int(sem_wrong.sum()) == 0, f"{int(sem_wrong.sum())} DECIDABLE semantic codes differ from the reference"


def test_signal_quality_wer_and_mos(evidence):
    """SCORE the rendered signal: intelligibility (WER) and naturalness (MOS), against the HF golden.

    Both sides are cut at their OWN rows' end frames and scored identically. The TT output must be
    no worse than the free-running HF golden by the stated margins.
    """
    from models.demos.voxtral_4b_tts_2603.reference import quality

    tt, free, texts = evidence["tt"], evidence["free"], evidence["texts"]
    rate = tt["sampling_rate"]
    tt_scores = quality.score(pipeline.trim_to_end(tt), rate, texts)
    hf_scores = quality.score(pipeline.trim_to_end(free), rate, texts)
    for i in range(evidence["batch"]):
        print(
            f"[{i:02d}] TT WER={tt_scores['wer'][i]:.3f} MOS={tt_scores['mos'][i]:.2f} | "
            f"HF WER={hf_scores['wer'][i]:.3f} MOS={hf_scores['mos'][i]:.2f} | {tt_scores['transcripts'][i]!r}"
        )
    tt_wer, hf_wer = tt_scores["corpus_wer"], hf_scores["corpus_wer"]
    tt_mos = sum(tt_scores["mos"]) / len(tt_scores["mos"])
    hf_mos = sum(hf_scores["mos"]) / len(hf_scores["mos"])
    print(f"corpus WER: TT={tt_wer:.4f} HF={hf_wer:.4f} (margin {WER_MARGIN})")
    print(f"mean MOS:   TT={tt_mos:.3f} HF={hf_mos:.3f} (margin {MOS_MARGIN})")
    assert tt_wer <= hf_wer + WER_MARGIN, f"TT wer {tt_wer:.4f} is worse than the HF golden's {hf_wer:.4f}"
    assert tt_mos >= hf_mos - MOS_MARGIN, f"TT mos {tt_mos:.3f} is worse than the HF golden's {hf_mos:.3f}"
    # PER CLIP: one unintelligible or broken row moves the corpus scores by only ~0.03 WER / ~0.08 MOS.
    # Naturalness is the per-clip signal (dead or garbled audio scores ~1-2; this build's weakest real
    # clips ~2.7); WER per clip is not used, because Whisper sometimes collapses a correct clip to one word.
    dead = [i for i in range(evidence["batch"]) if tt_scores["mos"][i] < CLIP_MOS_FLOOR <= hf_scores["mos"][i]]
    assert not dead, f"clips {dead} score MOS below {CLIP_MOS_FLOOR} where the HF golden's do not"


def test_free_running_divergence_is_reported(evidence):
    """TT-free-run vs HF-free-run, as a DIAGNOSTIC -- plus the one part of it that is decidable.

    Frame 0 takes no feedback, so its semantic code is a clean discrete comparison.
    """
    tt, free, batch = evidence["tt"], evidence["free"], evidence["batch"]
    steps = min(tt["codes"].shape[-1], free["codes"].shape[-1])
    agree = float((tt["codes"][..., :steps] == free["codes"][..., :steps]).float().mean())
    print(f"\nfree-running reference (DIAGNOSTIC, not a gate metric): code agreement={agree:.6f}")
    # Frame 0's semantic code is decidable wherever the reference's own top-2 margin clears the
    # measured logit deviation; a closer call is a tie, counted, and cannot fail the check.
    frame0 = tt["codes"][:, 0, 0] == free["codes"][:, 0, 0]
    lo_tt = tt["diagnostics"][0]["semantic_logits"].float()
    lo_hf = free["diagnostics"][0]["semantic_logits"].float()
    # Masked entries (-inf / -1e30 on either side) carry no deviation.
    real = torch.isfinite(lo_hf) & torch.isfinite(lo_tt) & (lo_hf > -1e20) & (lo_tt > -1e20)
    measured = float((lo_tt - lo_hf)[real].pow(2).mean().sqrt())
    # The FIXED logit band: a band read off this run's own deviation would widen exactly when the logits are
    # wrong. Its size is the PREFILL-inclusive deviation (FRAME0_LOGIT_RMS), because the free reference runs
    # its own prefill; the semantic head's own band (SEMANTIC_LOGIT_RMS) is ~40x narrower than that.
    ldev = FRAME0_LOGIT_RMS
    top2 = lo_hf.topk(2, dim=1).values
    decidable = (top2[:, 0] - top2[:, 1]) > TIE_SIGMA * ldev
    wrong = ~frame0 & decidable
    print(
        f"frame-0 semantic code (no feedback in it): {int(frame0.sum())}/{batch} exact; "
        f"{int((~frame0 & ~decidable).sum())} ties (band {TIE_SIGMA} x RMS {ldev:.3e}), "
        f"{int(wrong.sum())} decidable mismatches; logit RMS deviation {measured:.3e} (limit {2 * ldev:.3e})"
    )
    assert measured <= 2 * ldev, f"frame-0 semantic logits deviate by RMS {measured:.3e}: the prefill drifted"
    assert not bool(wrong.any()), "a DECIDABLE frame-0 semantic code disagrees, before any feedback exists"


def test_gate3_e2e_pcc(evidence):
    """GATE 3: the FINAL output -- the whole waveform, every frame -- against the HF golden, all 32 rows.

    The reference is torch's own chain teacher-forced onto this pipeline's trajectory, rendering the
    codes this pipeline emitted; together with the per-stage and discrete checks above it covers
    every stage from the prompt to the audio samples.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    assert torch.equal(tt["codes"], hf["fed_codes"]), "the reference was not put on the TT trajectory"
    assert tt["waveform"].shape == hf["waveform"].shape
    per_sample = [common.pcc(tt["waveform"][i], hf["waveform"][i]) for i in range(batch)]
    achieved_pcc = min(per_sample)
    worst = int(torch.tensor(per_sample).argmin())
    print(
        f"\nper-sample waveform PCC over {tt['frames_decoded']} frames: min={achieved_pcc:.6f} "
        f"mean={sum(per_sample) / batch:.6f} max={max(per_sample):.6f} (worst sample {worst})"
    )
    print(f"e2e PCC={achieved_pcc}")
    assert (
        achieved_pcc >= PCC_TARGET
    ), f"Gate 3 FAILED: waveform PCC {achieved_pcc:.6f} < {PCC_TARGET} on sample {worst}"
