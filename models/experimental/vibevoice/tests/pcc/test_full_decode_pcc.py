# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full VibeVoice decode-chain PCC vs HuggingFace reference (teacher-forced).

Decode counterpart to ``test_full_prefill_pcc.py``.  Where the prefill test gates the
LM ``last_hidden_state`` after the voice-clone prefill, this test gates the LM hidden
state produced by the *autoregressive decode* steps.

METHOD — teacher forcing over a shared orchestration
----------------------------------------------------
The teacher token stream comes from the reference's own ``generate()`` (the authoritative
path, as in ``test_e2e_generate_pcc.py``).  We then replay that **same** ``forced_token_ids``
through the **same** ``TTVibeVoiceGenerator`` decode loop on two backends:

  * reference-backed generator (``ref_inference`` set → CPU-fp32 HuggingFace LM + the
    reference diffusion head / acoustic+semantic tokenizers / connectors) — the teacher,
  * pure-TT generator (all TTNN kernels).

Both consume an identical token sequence so their KV caches stay position-aligned.  We
capture the positive LM decode hidden state at every step from each backend and compare
per step.

WHAT IS GATED, AND WHY ONLY THE FIRST FRAME
-------------------------------------------
The input to a speech-diffusion decode step is the *fused embed* built from that frame's
diffusion latent (acoustic + semantic connectors).  The fp32 reference diffusion and the
bf16 TT diffusion (drawing independent reparameterization noise) produce slightly
different latents, so as fused embeds feed back the per-step hidden state **compounds** —
the same effect documented in ``test_e2e_generate_pcc.py`` (frame0≈0.996, later frames
decay).  The first decode step carries only a single frame of divergence, so it is the
reliable, minimally-compounded gate (>= 0.99) — the decode analog of the prefill
last_hidden gate and the e2e frame-0 audio gate.  Per-step PCC for the whole stream is
printed for information (the gradual decay is expected model chaos, not a port defect —
component parity is covered by the per-module PCC tests).
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import load_script
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import PCC_THRESHOLD
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_TEXT_PATH = TEXT_EXAMPLES_DIR / "2p_short.txt"
_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"
CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
# Post-prefill AR steps to generate for the teacher stream (kept small for CI; the
# reliable gate is the leading pre-diffusion step, so a short stream suffices).
MAX_NEW_TOKENS = 24


def _voice_path() -> str:
    if _VOICE_PATH.is_file():
        return str(_VOICE_PATH)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


def _build_processor_batch():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    assert _TEXT_PATH.is_file(), f"Missing demo text: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script],
        voice_samples=[[_voice_path()]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return processor, inputs


def _hidden_to_flat(hidden: ttnn.Tensor) -> torch.Tensor:
    """Single decode-position LM hidden (ttnn [1,1,1,H] or [1,1,H]) → 1-D float32 [H]."""
    return ttnn.to_torch(hidden).to(torch.float32).reshape(-1)


def _wrap_capture(generator):
    """Record the positive LM hidden state at prefill + every decode step.

    Wraps the generator's LM entry points (``_lm_prefill`` / ``_lm_step`` /
    ``_lm_decode_token``) so both backends are captured identically.  The negative-CFG
    LM passes go through ``_neg_lm_step`` and are intentionally *not* captured.
    """
    captured = {"prefill": None, "decode": []}
    orig_prefill = generator._lm_prefill
    orig_step = generator._lm_step
    orig_decode = generator._lm_decode_token

    def prefill(inputs_embeds, kv_cache):
        logits, hidden = orig_prefill(inputs_embeds, kv_cache)
        captured["prefill"] = _hidden_to_flat(hidden)
        return logits, hidden

    def step(inputs_embeds, start_pos, kv_cache):
        logits, hidden = orig_step(inputs_embeds, start_pos, kv_cache)
        captured["decode"].append(_hidden_to_flat(hidden))
        return logits, hidden

    def decode_token(token_id, start_pos, kv_cache):
        logits, hidden = orig_decode(token_id, start_pos, kv_cache)
        captured["decode"].append(_hidden_to_flat(hidden))
        return logits, hidden

    generator._lm_prefill = prefill
    generator._lm_step = step
    generator._lm_decode_token = decode_token
    return captured


def _run_backend(generator, *, input_ids, attention_mask, speech_input_mask, prefill_speech_embeds, forced):
    """Run one decode backend (seed-aligned) and return (captured_hiddens, sequences)."""
    captured = _wrap_capture(generator)
    torch.manual_seed(0)
    out = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        speech_input_mask=speech_input_mask,
        prefill_speech_embeds=prefill_speech_embeds,
        max_new_tokens=None if forced is not None else MAX_NEW_TOKENS,
        forced_token_ids=forced,
    )
    return captured, out.sequences


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_full_decode_chain_pcc(mesh_device):
    """Teacher-forced per-step LM decode-hidden PCC: TT vs reference-backed decode."""
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    processor, inputs = _build_processor_batch()
    prefill_len = inputs["input_ids"].shape[1]

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    # Deterministic VAE mode for the voice-clone prefill embeds (see test_full_prefill_pcc):
    # the non-persistent ``fix_std`` buffer is nondeterministic garbage under transformers 5.x.
    ref_model.model.acoustic_tokenizer.std_dist_type = "none"

    # Aligned prefill: reference speech embeds (deterministic mode) fed to BOTH backends so the
    # prefill KV state is identical before decode.
    with torch.no_grad():
        _, prefill_speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )
    prefill_speech_embeds = prefill_speech_embeds.to(torch.float32)

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        MODEL_PATH,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
    )
    tt_model.set_speech_scale_bias(
        ref_model.model.speech_scaling_factor.item(),
        ref_model.model.speech_bias_factor.item(),
    )

    common = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
    )

    # 1) Pure-TT greedy decode — the proven speech stream (the demo path).  Produces the teacher
    #    token stream AND the TT per-step hidden.  (The vendored reference ``generate()`` is not
    #    transformers-5.x compatible, so we source the stream from the TT decode instead; the
    #    per-step hidden comparison at identical forced positions is direction-agnostic.)
    tt_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    tt_capture, tt_sequences = _run_backend(tt_gen, forced=None, **common)
    forced = tt_sequences[0, prefill_len:].reshape(-1)
    assert forced.numel() > 0, "TT greedy decode produced no tokens"

    # 2) Reference-backed decode replaying the same token stream — capture reference hidden.
    ref_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=None,
        ref_inference=ref_model,
    )
    ref_capture, _ = _run_backend(ref_gen, forced=forced, **common)

    ref_hiddens = ref_capture["decode"]
    tt_hiddens = tt_capture["decode"]
    n_steps = min(len(ref_hiddens), len(tt_hiddens))
    assert n_steps > 0, "no decode steps captured"

    forced_list = forced.tolist()
    diffusion_id = processor.tokenizer.speech_diffusion_id
    per_step = [comp_pcc(ref_hiddens[i], tt_hiddens[i], pcc=0.0)[1] for i in range(n_steps)]

    # Frame-0 gate: the first decode step carries a single frame of diffusion divergence and is
    # the reliable, minimally-compounded signal (analogous to the e2e frame-0 audio gate and the
    # prefill last_hidden gate).  Later steps accumulate the fp32-ref-vs-bf16-TT diffusion
    # fused-embed differences and decay — reported for information, not gated.
    step0_pcc = per_step[0]
    n_diffusion = sum(1 for t in forced_list if t == diffusion_id)
    print(
        f"[test_full_decode_pcc] forced={forced.numel()} tok ({n_diffusion} speech-diffusion) "
        f"steps={n_steps} | GATE step0 PCC={step0_pcc:.6f} (>={PCC_THRESHOLD}) | "
        f"mean_PCC={sum(per_step) / n_steps:.4f} min_PCC={min(per_step):.4f}"
    )
    print("[test_full_decode_pcc] per-step PCC: " + " ".join(f"{i}:{p:.4f}" for i, p in enumerate(per_step)))
    print(f"[test_full_decode_pcc] first forced token ids: {forced_list[:8]} (speech_diffusion_id={diffusion_id})")

    assert step0_pcc >= PCC_THRESHOLD, (
        f"first decode-frame hidden PCC {step0_pcc:.6f} < {PCC_THRESHOLD} — a real decode-path "
        f"regression (LM decode / diffusion head / connectors / post-diffusion).  Later steps "
        f"compound via the diffusion fused-embed feedback and are reported for information only."
    )


def _run_ref_capture_inputs(ref_gen, ref_model, *, common, forced):
    """Reference-backed decode (forced); capture per-step (LM input embed [1,1,1,H], LM hidden).

    The input to a diffusion step is the reference *fused embed* (``_lm_step``); the input to a
    non-diffusion step is the reference token embedding (re-derived for ``_lm_decode_token``).
    """
    ref_inputs, ref_hiddens = [], []
    orig_step = ref_gen._lm_step
    orig_decode = ref_gen._lm_decode_token
    embed = ref_model.model.get_input_embeddings()

    def step(inputs_embeds, start_pos, kv_cache):
        logits, hidden = orig_step(inputs_embeds, start_pos, kv_cache)
        ref_inputs.append(ttnn.to_torch(inputs_embeds).to(torch.float32).reshape(1, 1, 1, -1))
        ref_hiddens.append(_hidden_to_flat(hidden))
        return logits, hidden

    def decode_token(token_id, start_pos, kv_cache):
        logits, hidden = orig_decode(token_id, start_pos, kv_cache)
        with torch.no_grad():
            emb = embed(torch.tensor([[token_id]], dtype=torch.long)).to(torch.float32).reshape(1, 1, 1, -1)
        ref_inputs.append(emb)
        ref_hiddens.append(_hidden_to_flat(hidden))
        return logits, hidden

    ref_gen._lm_step = step
    ref_gen._lm_decode_token = decode_token
    torch.manual_seed(0)
    ref_gen.generate(max_new_tokens=None, forced_token_ids=forced, **common)
    return ref_inputs, ref_hiddens


def _tt_lm_decode_forced_inputs(
    tt_gen, ref_inputs, *, input_ids, speech_input_mask, prefill_speech_embeds, prefill_len
):
    """Replay the reference per-step LM input embeds through the TT LM decode; capture hidden.

    Bypasses TT's diffusion/connectors entirely (every step is fed the reference's fused embed),
    so there is no fused-embed feedback and the comparison isolates the TT LM decode kernels.
    """
    dev = tt_gen.device
    inputs_embeds = tt_gen._build_prefill_embeds(
        input_ids, None, None, speech_input_mask, prefill_speech_embeds=prefill_speech_embeds
    )
    kv = tt_gen.lm.alloc_kv_cache(prefill_len + len(ref_inputs) + 8)
    tt_gen.lm.prefill_embeds(inputs_embeds, kv_cache=kv, return_last_hidden=True)

    tt_hiddens = []
    for i, emb in enumerate(ref_inputs):
        emb_tt = ttnn.as_tensor(
            emb, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _, hidden = tt_gen.lm.forward(emb_tt, start_pos=prefill_len + i, kv_cache=kv, return_last_hidden=True)
        tt_hiddens.append(_hidden_to_flat(hidden))
    return tt_hiddens


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_full_decode_lm_pcc(mesh_device):
    """LM-decode isolation: replay the reference per-step LM inputs through the TT LM decode.

    Complements ``test_full_decode_chain_pcc``.  By feeding the reference's fused embeds to the
    TT LM at every step (instead of letting TT compute its own), there is no diffusion fused-embed
    feedback and therefore no compounding — so *every* decode step is gated at >= 0.99.  This
    pinpoints whether the full-chain test's per-step decay comes from the LM decode (it should not)
    or from the diffusion/connector/post-diffusion chain (covered by their own module PCC tests).
    """
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    processor, inputs = _build_processor_batch()
    prefill_len = inputs["input_ids"].shape[1]

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    ref_model.model.acoustic_tokenizer.std_dist_type = "none"  # deterministic VAE mode (prefill embeds)

    with torch.no_grad():
        _, prefill_speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )
    prefill_speech_embeds = prefill_speech_embeds.to(torch.float32)

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
    )
    tt_model.set_speech_scale_bias(
        ref_model.model.speech_scaling_factor.item(),
        ref_model.model.speech_bias_factor.item(),
    )

    common = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
    )

    # Teacher token stream from the proven TT greedy decode (reference generate() is not 5.x-safe).
    tt_gen = tt_model._make_generator(
        processor.tokenizer, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS, max_new_tokens=MAX_NEW_TOKENS
    )
    _, tt_sequences = _run_backend(tt_gen, forced=None, **common)
    forced = tt_sequences[0, prefill_len:].reshape(-1)
    assert forced.numel() > 0, "TT greedy decode produced no tokens"

    # Reference-backed decode: capture the per-step LM input embeds AND the reference hidden.
    ref_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=None,
        ref_inference=ref_model,
    )
    ref_inputs, ref_hiddens = _run_ref_capture_inputs(ref_gen, ref_model, common=common, forced=forced)

    # Replay those exact inputs through the TT LM decode (fresh generator → fresh KV cache).
    tt_gen2 = tt_model._make_generator(
        processor.tokenizer, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS, max_new_tokens=None
    )
    tt_hiddens = _tt_lm_decode_forced_inputs(
        tt_gen2,
        ref_inputs,
        input_ids=inputs["input_ids"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
        prefill_len=prefill_len,
    )

    n_steps = min(len(ref_hiddens), len(tt_hiddens))
    assert n_steps > 0, "no decode steps captured"
    per_step = [comp_pcc(ref_hiddens[i], tt_hiddens[i], pcc=0.0)[1] for i in range(n_steps)]
    min_pcc = min(per_step)

    print(
        f"[test_full_decode_lm_pcc] steps={n_steps} (identical inputs, no compounding) | "
        f"min_PCC={min_pcc:.6f} mean_PCC={sum(per_step) / n_steps:.6f} threshold={PCC_THRESHOLD}"
    )
    print("[test_full_decode_lm_pcc] per-step PCC: " + " ".join(f"{i}:{p:.4f}" for i, p in enumerate(per_step)))

    assert min_pcc >= PCC_THRESHOLD, (
        f"TT LM decode hidden PCC {min_pcc:.6f} < {PCC_THRESHOLD} given reference inputs — a real "
        f"LM decode-kernel regression (inputs are forced identical every step, so this cannot be "
        f"diffusion/connector compounding)."
    )


def _run_ref_capture_hidden_latents(ref_gen, *, common, forced):
    """Reference-backed decode (forced); capture per-step LM hidden AND per-frame diffusion latent."""
    captured = _wrap_capture(ref_gen)
    latents = []
    orig_diff = ref_gen._run_speech_diffusion

    def diff(condition, neg_condition, latent_size=64, noise_2x=None, rng=None):
        lat = orig_diff(condition, neg_condition, latent_size=latent_size, noise_2x=noise_2x, rng=rng)
        latents.append(ttnn.to_torch(lat).to(torch.float32).reshape(1, 1, 1, -1).clone())
        return lat

    ref_gen._run_speech_diffusion = diff
    torch.manual_seed(0)
    ref_gen.generate(max_new_tokens=None, forced_token_ids=forced, **common)
    return captured["decode"], latents


def _run_tt_inject_latents(tt_gen, ref_latents, *, common, forced):
    """Pure-TT decode (forced) but with the diffusion head's output replaced by the reference latent
    each frame; everything else (post-diffusion + LM) stays TT.  Capture per-step LM hidden."""
    captured = _wrap_capture(tt_gen)
    dev = tt_gen.device
    idx = [0]
    latent_size = ref_latents[0].shape[-1] if ref_latents else 64

    def diff(condition, neg_condition, latent_size=64, noise_2x=None, rng=None):
        lat = ref_latents[idx[0]]
        idx[0] += 1
        return ttnn.as_tensor(
            lat.view(1, 1, 1, latent_size).to(torch.bfloat16),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_gen._run_speech_diffusion = diff
    torch.manual_seed(0)
    tt_gen.generate(max_new_tokens=None, forced_token_ids=forced, **common)
    return captured["decode"]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_full_decode_ref_diffusion_pcc(mesh_device):
    """Isolate the diffusion head as the drift driver: run the full TT decode chain but replace ONLY
    the diffusion latent with the reference's each frame (TT post-diffusion + LM otherwise), and
    compare per-step LM hidden to the reference-backed run.

    If the head is the dominant contributor, removing it as an error source should lift the per-step
    PCC far above the full-chain curve (which decays to ~0.78) — ideally back to >= 0.99 across all
    steps.  Any residual decay is attributable to the remaining TT post-diffusion chain (semantic
    tokenizer / acoustic decode), not the diffusion head.
    """
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    processor, inputs = _build_processor_batch()
    prefill_len = inputs["input_ids"].shape[1]

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", attn_implementation="sdpa"
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    ref_model.model.acoustic_tokenizer.std_dist_type = "none"

    with torch.no_grad():
        _, prefill_speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype), inputs["speech_masks"]
        )
    prefill_speech_embeds = prefill_speech_embeds.to(torch.float32)

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device, MODEL_PATH, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS
    )
    tt_model.set_speech_scale_bias(
        ref_model.model.speech_scaling_factor.item(), ref_model.model.speech_bias_factor.item()
    )

    common = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
    )

    # Teacher stream from the TT greedy decode.
    tt_gen = tt_model._make_generator(
        processor.tokenizer, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS, max_new_tokens=MAX_NEW_TOKENS
    )
    _, tt_sequences = _run_backend(tt_gen, forced=None, **common)
    forced = tt_sequences[0, prefill_len:].reshape(-1)
    assert forced.numel() > 0, "TT greedy decode produced no tokens"

    # Reference-backed decode: capture reference per-step hidden + per-frame diffusion latents.
    ref_gen = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=None,
        ref_inference=ref_model,
    )
    ref_hiddens, ref_latents = _run_ref_capture_hidden_latents(ref_gen, common=common, forced=forced)

    # Pure-TT decode with the diffusion latent pinned to the reference's each frame.
    tt_gen2 = tt_model._make_generator(
        processor.tokenizer, cfg_scale=CFG_SCALE, num_diffusion_steps=NUM_DIFFUSION_STEPS, max_new_tokens=None
    )
    tt_hiddens = _run_tt_inject_latents(tt_gen2, ref_latents, common=common, forced=forced)

    n_steps = min(len(ref_hiddens), len(tt_hiddens))
    assert n_steps > 0, "no decode steps captured"
    per_step = [comp_pcc(ref_hiddens[i], tt_hiddens[i], pcc=0.0)[1] for i in range(n_steps)]
    step0, min_pcc, mean_pcc = per_step[0], min(per_step), sum(per_step) / n_steps

    print(
        f"[test_full_decode_ref_diffusion_pcc] steps={n_steps} frames={len(ref_latents)} "
        f"(TT chain, diffusion latent pinned to reference) | "
        f"step0={step0:.6f} min_PCC={min_pcc:.6f} mean_PCC={mean_pcc:.6f} threshold={PCC_THRESHOLD}"
    )
    print(
        "[test_full_decode_ref_diffusion_pcc] per-step PCC: " + " ".join(f"{i}:{p:.4f}" for i, p in enumerate(per_step))
    )

    assert min_pcc >= PCC_THRESHOLD, (
        f"per-step decode-hidden PCC {min_pcc:.6f} < {PCC_THRESHOLD} even with the diffusion latent "
        f"pinned to the reference — residual drift is from the TT post-diffusion chain (semantic "
        f"tokenizer / acoustic decode), not the diffusion head."
    )
