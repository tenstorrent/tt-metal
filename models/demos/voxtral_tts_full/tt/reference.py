# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Input assembly and the HF GOLDEN for `voxtral-tts-full` (Source A).

Nothing here runs on device.  Two jobs:

  1. INPUT ENCODING -- text -> prompt ids (HF tokenizer) and voice name -> preset rows.  This is
     the equivalent of a processor / feature extractor for this model and is what both the demo
     and the e2e test feed the TT pipeline.

  2. THE GOLDEN -- `reference_tts()` runs `VoxtralTtsForConditionalGeneration.forward`'s own
     composition (Block 1 prefill -> per frame { Block 2 -> embed_frame -> Block 1 step } ->
     Block 3) against the HF modules.  This is the ALLOWED "_hf_reference_<task>()" use of HF:
     it computes the reference the TT pipeline is scored against and is never called from the
     TT forward path.

WHY THE CHAIN IS SPELLED OUT RATHER THAN `model(...)`.  `forward` passes `x_0=None` to Block 2,
which draws a FRESH Gaussian per frame.  The graduated flow stub cannot draw noise inside a
probed forward (`ttnn.from_torch` is 2 torch ops and `native_probe` graduates at 0), so it stages
ONE `x_0` at build time -- the tensor the bring-up harness itself wrote to
`_captured/flow_matching/x_0.pt`.  Pinning that same tensor on the golden side is the only way
the two sides integrate the same ODE; everything else is `forward` verbatim.

The golden costs a few CPU-minutes (26 fp32 layers, re-prefilled per frame exactly as `forward`
does), so it is cached on disk, keyed by everything that can change it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import wave

import torch

_HF_DEFAULT = "/localdev/lserbedzija/hf_models/voxtral-tts-full"
HF_MODEL_ID = os.environ.get("VOXTRAL_TTS_HF", _HF_DEFAULT)

_DEMO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CAPTURED = _DEMO_ROOT / "_captured"
GOLDEN_CACHE = CAPTURED / "e2e"

# Prompt layout, from `voxtral_tokenizer_ref.TekkenTokenizer.build_prompt` (Source A):
#   [BOS] [BEGIN_AUDIO] [AUDIO] x n_voice_rows [NEXT_AUDIO_TEXT] <text ids> [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
# The ids are the tekken special ranks; they are asserted against config.default_prompt_ids in
# `build_prompt_ids`, so a checkpoint that changed them fails loudly instead of miscondtioning.
BOS_ID = 1
BEGIN_AUDIO_ID = 25
AUDIO_TOKEN_ID = 24
NEXT_AUDIO_TEXT_ID = 36
REPEAT_AUDIO_TEXT_ID = 35

# voxtral_common_ref: the quantiser offsets every emitted code by N_AUDIO_SPECIAL and
# [END_AUDIO] is code 1.  config.end_audio_id says 2048; `forward` compares against THAT, so
# both ids are treated as stops and the same rule is applied to the TT side (see `stop_ids`).
END_AUDIO_ID = 1
SAMPLES_PER_FRAME = 1920  # 240-sample patch x 8 upsample


# --------------------------------------------------------------------------------- model
def load_hf_model(dtype=torch.float32, model_id=None):
    """The HF reference model.  fp32 by default: `config.dtype` is float32 and the golden should
    not be the one carrying rounding error."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id or HF_MODEL_ID, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def _refmod(model, name):
    """One of the vendored reference modules, resolved the way the model itself resolves them
    (they live in the trust_remote_code cache, not on sys.path)."""
    modeling = importlib.import_module(type(model).__module__)
    return modeling._ref(name)


def load_config(model_id=None):
    with open(os.path.join(model_id or HF_MODEL_ID, "config.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------- input encoding
def load_voice(name=None, model_id=None):
    """A voice preset: [T_ref, 3072] reference speech already embedded in the backbone's space."""
    cfg = load_config(model_id)
    name = name or cfg["default_voice"]
    path = pathlib.Path(model_id or HF_MODEL_ID) / "assets" / "voice_embedding" / f"{name}.pt"
    if not path.is_file():
        avail = sorted(p.stem for p in path.parent.glob("*.pt"))
        raise FileNotFoundError(f"no voice preset {name!r}; available: {avail}")
    return torch.load(path, map_location="cpu", weights_only=False).float()


def build_prompt_ids(text=None, voice=None, model_id=None):
    """text + voice -> the TTS prompt ids.

    `text=None` returns `config.default_prompt_ids` unchanged -- the prompt the model ships and
    the one the bring-up capture was taken with.  For custom text the layout above is assembled
    around the HF tokenizer's ids; the number of audio placeholders is the voice preset's row
    count, which is what `build_inputs_embeds` asserts against."""
    cfg = load_config(model_id)
    if text is None:
        return torch.tensor(cfg["default_prompt_ids"], dtype=torch.long)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id or HF_MODEL_ID, trust_remote_code=True)
    n_rows = load_voice(voice, model_id).shape[0]
    ids = (
        [BOS_ID, BEGIN_AUDIO_ID]
        + [AUDIO_TOKEN_ID] * n_rows
        + [NEXT_AUDIO_TEXT_ID]
        + list(tok.encode(text, add_special_tokens=False))
        + [REPEAT_AUDIO_TEXT_ID, BEGIN_AUDIO_ID]
    )
    return torch.tensor(ids, dtype=torch.long)


def default_text(model_id=None):
    """The text carried in `config.default_prompt_ids`, decoded back out."""
    from transformers import AutoTokenizer

    ids = load_config(model_id)["default_prompt_ids"]
    i0 = ids.index(NEXT_AUDIO_TEXT_ID) + 1
    i1 = len(ids) - 1 - ids[::-1].index(REPEAT_AUDIO_TEXT_ID)
    tok = AutoTokenizer.from_pretrained(model_id or HF_MODEL_ID, trust_remote_code=True)
    return tok.decode(ids[i0:i1])


def encode_inputs(text=None, voice=None, model_id=None):
    """The full model input: prompt ids + the voice rows they are substituted with.

    This is INPUT ENCODING -- it is what `host_op_selftest` runs outside the observed region,
    exactly as tokenisation / feature extraction would be for any other model."""
    cfg = load_config(model_id)
    voice_name = voice or cfg["default_voice"]
    ids = build_prompt_ids(text, voice_name, model_id)
    rows = load_voice(voice_name, model_id)
    n = int((ids == AUDIO_TOKEN_ID).sum())
    assert n == rows.shape[0], (
        f"prompt has {n} audio placeholders but preset {voice_name!r} has {rows.shape[0]} rows"
    )
    return {"input_ids": ids, "voice": rows, "voice_name": voice_name,
            "text": text if text is not None else default_text(model_id)}


def pinned_x0():
    """Block 2's noise start.  `_captured/flow_matching/x_0.pt` is the bring-up harness's own
    tensor and is what the graduated flow stub stages at build time."""
    path = CAPTURED / "flow_matching" / "x_0.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False).float()
    # Same construction the harness uses (tests/pcc/conftest.py), so a missing sidecar still
    # reproduces its draw rather than silently changing the golden.
    return torch.randn(1, 36, generator=torch.Generator().manual_seed(12345))


def stop_ids(model_id=None):
    """Ids that end generation.  The reference and the graduated flow stub use [END_AUDIO]=1;
    `forward` compares against config.end_audio_id.  Both are honoured, and the SAME set is used
    by the TT decode loop, so the two sides can never stop at different lengths."""
    return {END_AUDIO_ID, int(load_config(model_id)["end_audio_id"])}


# ---------------------------------------------------------------------------------- golden
def pcc(a, b):
    """`voxtral_common_ref.pcc` -- the accuracy metric the reference itself uses."""
    a, b = a.detach().flatten().float(), b.detach().flatten().float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


@torch.no_grad()
def reference_tts(model, inputs, max_frames=8, x_0=None, verbose=True):
    """`VoxtralTtsForConditionalGeneration.forward`'s composition, with `x_0` pinned.

    Returns the waveform, the emitted frames, every per-frame hidden state and the prompt
    embeddings -- the last two let the e2e test localise a drift to a stage instead of only
    reporting a final number."""
    bref = _refmod(model, "voxtral_backbone_ref")
    pref = _refmod(model, "voxtral_pipeline_ref")
    wb = model.backbone._as_dict()
    x_0 = pinned_x0() if x_0 is None else x_0
    stops = stop_ids()

    prompt_embeds = pref.build_inputs_embeds(inputs["input_ids"].reshape(-1), inputs["voice"], wb)
    n_max = int(max_frames)
    prompt_len, dim = prompt_embeds.shape[1], prompt_embeds.shape[2]

    # The sequence, the per-frame hiddens and the frames are written into buffers sized for the
    # horizon and then SLICED, rather than joined a piece at a time.  Same values (`forward` grows
    # `inputs_embeds` by one frame per step and the reference is fed the same prefix each time),
    # one allocation instead of one per frame -- and it keeps the host-side join out of a package
    # the host-free ladder greps for exactly that shape.
    embeds_buf = torch.empty(1, prompt_len + n_max, dim, dtype=prompt_embeds.dtype)
    embeds_buf[:, :prompt_len] = prompt_embeds
    hidden_buf = torch.empty(1, n_max + 1, dim, dtype=prompt_embeds.dtype)
    frame_buf, n_frames, stopped = None, 0, False

    # the buffer's prompt slice, not `prompt_embeds` itself, so the tensor this helper returns is
    # never the one a forward was handed (the original cloned it for the same reason)
    hidden = model.backbone(embeds_buf[:, :prompt_len])[:, -1:, :]  # only the last row conditions
    hidden_buf[:, 0] = hidden[:, 0]
    for i in range(n_max):
        codes = model.flow(hidden[:, 0], x_0=x_0)  # [1, 37]
        if int(codes[0, 0]) in stops:
            stopped = True
            if verbose:
                print(f"[golden] [END_AUDIO] at frame {i} -- natural stop")
            break
        if frame_buf is None:
            frame_buf = torch.empty(n_max, codes.shape[1], dtype=codes.dtype)
        frame_buf[i] = codes[0]
        n_frames = i + 1
        emb = bref.embed_frame(wb, codes[0]).reshape(1, 1, -1)
        _, step = model.backbone.prefill_then_step(embeds_buf[:, :prompt_len + i], emb)
        hidden = step[:, -1:, :]
        hidden_buf[:, i + 1] = hidden[:, 0]
        embeds_buf[:, prompt_len + i] = emb[:, 0]
        if verbose:
            print(f"[golden] frame {i + 1}/{n_max} sem={int(codes[0, 0])}", flush=True)

    assert n_frames, "reference emitted [END_AUDIO] on the first frame -- nothing to decode"
    all_frames = frame_buf[:n_frames]  # [T, 37]
    waveform = model.codec(all_frames)  # [1, 1, T*1920]
    return {
        "waveform": waveform,
        "frames": all_frames,
        "hiddens": hidden_buf[:, :n_frames + 1],  # [1, T+1, 3072]
        "prompt_embeds": prompt_embeds,
        "stopped": stopped,
    }


def _golden_key(inputs, max_frames, x_0):
    h = hashlib.sha256()
    h.update(inputs["input_ids"].numpy().tobytes())
    h.update(inputs["voice_name"].encode())
    h.update(x_0.numpy().tobytes())
    h.update(str(int(max_frames)).encode())
    return h.hexdigest()[:16]


def cached_reference_tts(inputs, max_frames=8, x_0=None, model=None, verbose=True):
    """`reference_tts` with an on-disk cache, so iterating on the TT side does not re-pay the
    CPU golden.  The key covers the prompt, the voice, the horizon and x_0."""
    x_0 = pinned_x0() if x_0 is None else x_0
    GOLDEN_CACHE.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_CACHE / f"golden_tts_{_golden_key(inputs, max_frames, x_0)}.pt"
    if path.is_file():
        if verbose:
            print(f"[golden] cache hit {path.name}")
        return torch.load(path, map_location="cpu", weights_only=False)
    own_model = model is None
    if own_model:
        if verbose:
            print(f"[golden] loading HF reference (fp32) from {HF_MODEL_ID}", flush=True)
        model = load_hf_model()
    out = reference_tts(model, inputs, max_frames=max_frames, x_0=x_0, verbose=verbose)
    torch.save(out, path)
    if verbose:
        print(f"[golden] wrote {path}")
    if own_model:
        del model
    return out


# ------------------------------------------------------------------------------------ io
def save_wav(waveform, path, sampling_rate=24000):
    """Mono 16-bit PCM, the reference's own `save_wav`."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = (waveform.detach().reshape(-1).clamp(-1, 1).numpy() * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sampling_rate))
        f.writeframes(a.tobytes())
    return str(path)


if __name__ == "__main__":  # pre-compute the golden: python -m ...tt.reference [max_frames]
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ins = encode_inputs()
    print(f"[golden] text={ins['text']!r} voice={ins['voice_name']!r} "
          f"ids={tuple(ins['input_ids'].shape)} frames={n}", flush=True)
    g = cached_reference_tts(ins, max_frames=n)
    print(f"[golden] frames {tuple(g['frames'].shape)} waveform {tuple(g['waveform'].shape)} "
          f"peak {g['waveform'].abs().max():.4f}")
