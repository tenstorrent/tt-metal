# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The HF golden for Call 1 (text-to-speech), and its cache.

ALL HuggingFace calls on the reference side live in this file. `tt/pipeline.py` never calls an
HF submodule; it only reads `config` attributes and extracts weights at build time.

IT LIVES WITH THE TEST, NOT IN `tt/`. This module is pure torch: it is the measuring stick the
TT pipeline is held against, not a part of it. `tt/` is the TTNN pipeline package -- everything
in it should be device code -- and a torch reimplementation of the model sitting there
misdescribes the port to every reader and every tool that reads `tt/` to decide what the
pipeline does. The only consumers are `tests/e2e/test_e2e_tts.py` and the demo's optional
`--check` path, both of which are test/verification surfaces.

WHAT THE GOLDEN IS. `VoxtralTtsForConditionalGeneration` registers no custom `generate()` --
its own `forward()` IS the autoregressive chain (prefill -> flow -> embed_frame feedback ->
codec), and `PreTrainedModel.generate()` is unusable here (no LM head, no
`prepare_inputs_for_generation`). So `forward`'s composition is transcribed here, call for
call, over the model's own submodules:

    hidden = backbone(inputs_embeds)[:, -1:, :]
    loop:   codes = flow(hidden[:, 0]);  stop if codes[0, 0] == config.end_audio_id
            emb   = embed_frame(codes)
            _, step = backbone.prefill_then_step(inputs_embeds, emb);  hidden = step[:, -1:]
            inputs_embeds = cat(inputs_embeds, emb)
    wav = codec(frames)

THE ONE DEPARTURE, AND WHY. `forward` lets Block 2 draw its ODE initial condition from a
Gaussian, so two runs of the reference itself do not agree. `VoxtralFlowMatching.forward` takes
an explicit `x_0` for exactly this reason (the reference: "so PCC tests stay deterministic"), and
it is passed here. Pinning an INPUT is not injecting a reference tensor at a joint: the TT
pipeline is handed the identical bank and every other value on both sides is computed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from models.demos.voxtral_tts_full.tt.pipeline import HF_MODEL_ID, _ref_module

_CACHE_DIR = Path(__file__).resolve().parents[2] / "_captured" / "e2e"
_VOICE_DIR = str(Path(HF_MODEL_ID) / "assets" / "voice_embedding")


def _digest(ids, voice, max_frames, x0_bank, stop_id) -> str:
    h = hashlib.sha1()
    h.update(torch.as_tensor(ids).to(torch.int64).numpy().tobytes())
    h.update(str(voice).encode())
    h.update(str(int(max_frames)).encode())
    h.update(str(int(stop_id)).encode())
    h.update(torch.as_tensor(x0_bank).float().numpy().tobytes())
    h.update(b"v2-explicit-x0")
    return h.hexdigest()[:16]


@torch.no_grad()
def hf_reference_tts(hf, ids, voice, max_frames, x0_bank, stop_id=None):
    """`VoxtralTtsForConditionalGeneration.forward`, transcribed with a pinned `x_0`."""
    bref = _ref_module("voxtral_backbone_ref")
    pref = _ref_module("voxtral_pipeline_ref")

    stop_id = int(getattr(hf.config, "end_audio_id") if stop_id is None else stop_id)
    wb = hf.backbone._as_dict()
    preset = pref.load_voice(voice, voice_dir=_VOICE_DIR)
    inputs_embeds = pref.build_inputs_embeds(torch.as_tensor(ids).reshape(-1), preset, wb)

    prefill_hidden = hf.backbone(inputs_embeds)
    hidden = prefill_hidden[:, -1:, :]

    frames, step_hidden = [], []
    for t in range(int(max_frames)):
        codes = hf.flow(hidden[:, 0], x_0=x0_bank[t : t + 1])  # [1, 37]
        if int(codes[0, 0]) == stop_id:
            break
        frames.append(codes)
        emb = bref.embed_frame(wb, codes[0]).reshape(1, 1, -1)
        _, step = hf.backbone.prefill_then_step(inputs_embeds, emb)
        hidden = step[:, -1:, :]
        step_hidden.append(hidden[:, 0].clone())
        inputs_embeds = torch.cat([inputs_embeds, emb], dim=1)

    if not frames:
        return {"waveform": None, "frames": None, "prefill_hidden": prefill_hidden, "step_hidden": [], "n_frames": 0}

    all_frames = torch.cat(frames, dim=0)  # [T, 37]
    return {
        "waveform": hf.codec(all_frames),
        "frames": all_frames,
        "prefill_hidden": prefill_hidden,
        "step_hidden": step_hidden,
        "n_frames": len(frames),
    }


def golden(hf_loader, ids, voice, max_frames, x0_bank, stop_id, refresh=False):
    """The golden for these exact inputs, from cache when possible.

    The CPU reference re-prefills a 200-position, 26-layer, 3.4B stack once per frame, so it is
    minutes of work; the cache key covers everything that changes the answer.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"golden_{_digest(ids, voice, max_frames, x0_bank, stop_id)}.pt"
    if path.exists() and not refresh:
        return torch.load(path, map_location="cpu", weights_only=False), str(path)

    hf = hf_loader()
    out = hf_reference_tts(hf, ids, voice, max_frames, x0_bank, stop_id)
    torch.save(out, path)
    meta = {
        "voice": voice,
        "max_frames": int(max_frames),
        "stop_id": int(stop_id),
        "n_frames": out["n_frames"],
        "prompt_len": int(torch.as_tensor(ids).numel()),
    }
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return out, str(path)
