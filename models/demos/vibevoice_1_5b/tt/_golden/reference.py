# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference (HF golden) helpers for microsoft/VibeVoice-1.5B TTS.

Source A. This module (a) loads the reference model + processor and (b) provides
`hf_reference_tts`, a FAITHFUL reimplementation of
`VibeVoiceForConditionalGenerationInference.generate()` used ONLY to compute the
golden output the TT pipeline is compared against.

Why the loop is reimplemented rather than calling `model.generate()`: the vibevoice
package vendors a `generate()` built against an older transformers, and its
GenerationMixin plumbing (`_prepare_generation_config`, cache prep, …) is broken
against the installed transformers 5.x. The loop below drives the SAME reference
submodules (`language_model`, `lm_head`, `prediction_head`, `acoustic_tokenizer`,
`semantic_tokenizer`, `acoustic_connector`, `semantic_connector`, `noise_scheduler`)
with the SAME algorithm: greedy (`do_sample=False`) under the speech-token
constraint, `cfg_scale=1.0` (so `sample_speech_tokens`'s CFG combine
`half_eps = uncond + cfg*(cond-uncond)` reduces to `cond_eps` and the whole
negative branch is a no-op and is omitted), the LM run in full-recompute causal
mode with no KV cache (matching the graduated `qwen2_model` stub), and both HF and
TT capped to the SAME small horizon (N diffusion frames, S ddpm steps).
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import torch

MODEL_ID = "microsoft/VibeVoice-1.5B"


def _install_qwen2_tokenizer_shim():
    """transformers 5.x moved Qwen2TokenizerFast; the vibevoice package imports it
    from the old path. Alias the old module name to the current class."""
    name = "transformers.models.qwen2.tokenization_qwen2_fast"
    if name in sys.modules:
        return
    import transformers as _tf

    shim = types.ModuleType(name)
    shim.Qwen2TokenizerFast = _tf.Qwen2TokenizerFast
    sys.modules[name] = shim


def load_reference_model(model_id: str = MODEL_ID):
    """Load VibeVoiceForConditionalGenerationInference (has the speech generate chain)
    with real weights, tolerant of transformers version skew. eval() mode, CPU."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers.models.auto.auto_factory import _LazyAutoMapping
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig

    _install_qwen2_tokenizer_shim()

    orig = _LazyAutoMapping.register
    _LazyAutoMapping.register = lambda self, k, v, exist_ok=False: orig(self, k, v, exist_ok=True)
    try:
        import vibevoice.modular.modeling_vibevoice  # noqa: F401 (self-registers configs)
        import vibevoice.modular.modeling_vibevoice_inference as mvi
    finally:
        _LazyAutoMapping.register = orig

    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

    CONFIG_MAPPING.register("vibevoice", VibeVoiceConfig, exist_ok=True)
    cls = mvi.VibeVoiceForConditionalGenerationInference
    orig_tie = cls.tie_weights
    cls.tie_weights = lambda self, *a, **k: orig_tie(self)

    config = AutoConfig.from_pretrained(model_id)
    with torch.device("cpu"):
        model = cls(config)
    idx = hf_hub_download(model_id, "model.safetensors.index.json")
    weight_map = json.load(open(idx))["weight_map"]
    state_dict = {}
    for shard in sorted(set(weight_map.values())):
        state_dict.update(load_file(hf_hub_download(model_id, shard)))
    model.load_state_dict(state_dict, strict=False)
    # tie_word_embeddings=True: lm_head shares embed_tokens weight (not in checkpoint).
    model.lm_head.weight = model.model.get_input_embeddings().weight
    model.eval()
    return model


def build_processor(model_id: str = MODEL_ID):
    _install_qwen2_tokenizer_shim()
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    return VibeVoiceProcessor.from_pretrained(model_id)


def default_voice_sample(seconds: float = 2.0, sr: int = 24000) -> np.ndarray:
    """Deterministic voiced-harmonic reference voice (no external audio file needed)."""
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.5 * np.sin(2 * np.pi * 130 * t) + 0.2 * np.sin(2 * np.pi * 320 * t)
    return sig.astype(np.float32)


def make_inputs(processor, text: str, voice: np.ndarray):
    return processor(text=text, voice_samples=[voice], return_tensors="pt", padding=True)


def _constrained_argmax(logits_row: torch.Tensor, valid_ids) -> int:
    masked = torch.full_like(logits_row, float("-inf"))
    masked[valid_ids] = logits_row[valid_ids]
    return int(masked.argmax())


def make_noises(N_frames: int, dim: int, seed: int = 1234):
    """The DDPM initial noise for each diffusion frame. Shared verbatim with the TT
    side so the (deterministic) diffusion sampling is comparable numerically."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, dim, generator=g, dtype=torch.float32) for _ in range(N_frames)]


@torch.no_grad()
def hf_reference_tts(model, processor, inputs, N: int = 6, S: int = 5, noises=None):
    """Faithful golden TTS chain. Returns dict of golden tensors + the token schedule
    + the per-frame initial noise (so the TT pipeline reuses identical noise)."""
    tok = processor.tokenizer
    diff_id = tok.speech_diffusion_id
    start_id, end_id, eos = tok.speech_start_id, tok.speech_end_id, tok.eos_token_id
    valid = [start_id, end_id, diff_id, eos]

    dt = model.dtype
    m = model.model
    embed = m.get_input_embeddings()
    scaling = m.speech_scaling_factor
    bias = m.speech_bias_factor
    acoustic_vae = model.config.acoustic_vae_dim
    if noises is None:
        noises = make_noises(N + 2, acoustic_vae)

    input_ids = inputs["input_ids"]
    embeds = embed(input_ids)  # [1,L,1536]
    # Voice-sample acoustic encoding. The reference _process_speech_inputs samples the
    # acoustic latents with Gaussian noise (std_dist_type='gaussian'); the graduated TT
    # acoustic tokenizer uses the deterministic mean (noise unreproducible across TT/torch
    # RNG, and per-component PCC accepts mean-only). Use the MEAN here too so the golden
    # and the TT pipeline agree at this joint.
    enc_out = m.acoustic_tokenizer.encode(inputs["speech_tensors"].to(dt).unsqueeze(1))
    acoustic_latents = enc_out.mean  # [1,T',64] deterministic
    features = (acoustic_latents + bias) * scaling
    speech_embeds = m.acoustic_connector(features)[inputs["speech_masks"].cpu()]  # [n_valid,1536]
    embeds[inputs["speech_input_mask"]] = speech_embeds.to(embeds.dtype)
    prefill_embeds = embeds.clone()

    g = {
        "tokens": [],
        "hidden_last": [],
        "latents": [],
        "audio": [],
        "semantic": [],
        "feedback": [],
        "prefill_embeds": prefill_embeds,
        "speech_embeds": speech_embeds.float().clone(),
    }
    audio_chunks = []
    diff_count = 0
    step = 0
    while diff_count < N and step < N + 6:
        hidden = m.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state  # [1,L',1536]
        logits = model.lm_head(hidden[:, -1:, :]).float()  # [1,1,V]
        ntok = _constrained_argmax(logits[0, -1], valid)
        g["tokens"].append(ntok)
        g["hidden_last"].append(hidden[:, -1, :].float().clone())
        if ntok == diff_id:
            cond = hidden[:, -1, :].to(dt)  # [1,1536]
            m.noise_scheduler.set_timesteps(S)
            speech = noises[diff_count].to(dt)
            for ti in m.noise_scheduler.timesteps:
                eps = m.prediction_head(speech, ti.repeat(1).to(dt), condition=cond)
                speech = m.noise_scheduler.step(eps, ti, speech).prev_sample
            latent = speech  # [1,64]
            g["latents"].append(latent.float().clone())
            scaled = (latent / scaling - bias).unsqueeze(1)  # [1,1,64]
            audio = m.acoustic_tokenizer.decode(scaled.to(dt), use_cache=False)  # [1,1,T]
            audio_chunks.append(audio.float().clone())
            g["audio"].append(audio.float().clone())
            semantic = m.semantic_tokenizer.encode(audio.to(dt), use_cache=False).mean  # [1,1,128]
            g["semantic"].append(semantic.float().clone())
            a_emb = m.acoustic_connector(latent.unsqueeze(1))  # [1,1,1536]
            s_emb = m.semantic_connector(semantic)
            feedback = a_emb + s_emb
            g["feedback"].append(feedback.float().clone())
            embeds = torch.cat([embeds, feedback.to(embeds.dtype)], dim=1)
            diff_count += 1
        else:
            embeds = torch.cat([embeds, embed(torch.tensor([[ntok]])).to(embeds.dtype)], dim=1)
            if ntok in (end_id, eos):
                break
        step += 1

    g["waveform"] = torch.cat(audio_chunks, dim=-1).float() if audio_chunks else None
    g["noises"] = noises
    g["N"] = N
    g["S"] = S
    return g
