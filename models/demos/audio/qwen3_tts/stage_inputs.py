# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A vLLM-Omni Qwen3-TTS request's `additional_information` -> the talker's prompt.

vLLM-Omni serves the talker from placeholder token ids and carries what the prompt is made
of beside them, in `additional_information`: every value list-wrapped, `[value]`. This reads
the Base (voice clone) fields and builds the prompt with this directory's own builders:

    task_type          ["Base"]; the other releases are refused here
    text, language     ["..."], ["Auto" or a language name]
    ref_text           ["..."], the clip's transcript, required in context
    ref_audio          [[samples, sample_rate]], the clip, decoded by the server
    voice_clone_prompt [{"ref_code": [T, 16], "ref_spk_embedding": [hidden], "icl_mode": bool}],
                       a clip encoded earlier, used instead of `ref_audio` when present
    x_vector_only_mode [bool], clone from the voice alone
    non_streaming_mode [bool], default False for Base: the text streams a token per frame
    instruct           ["..."], x-vector cloning only
    _qwen3_tts_ref_audio_cache_key  ["..."], the server's name for the clip; once a request
                       carrying the clip and this key completes, later requests for that
                       clip carry the key alone (see `ReferenceCache`)

The server sizes the prompt before the model sees it, from an estimate, and the model fills
exactly that many positions: short prompts pad with `tts_pad`, long ones are cut. That is
vLLM-Omni's own talker behaviour, kept so positions line up with the scheduler's.
"""

from collections import OrderedDict

import torch

from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    CloneReference,
    build_clone_reference,
    build_streaming_clone_prefill,
    build_streaming_x_vector_prefill,
    build_voice_clone_prefill,
    build_x_vector_prefill,
)


def first(info, key, default=None):
    """One field of `additional_information`, unwrapped from its list."""
    value = info.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def voice_clone_prompt(info):
    """The encoded clip, if the request carries one: a dict, list-wrapped once or twice."""
    prompt = first(info, "voice_clone_prompt")
    if isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], dict):
        prompt = prompt[0]
    return prompt if isinstance(prompt, dict) else None


def clone_mode(info):
    """Whether this request clones in context (codes and transcript) rather than from the voice
    alone. An encoded `voice_clone_prompt` carries its own answer, which wins."""
    in_context = not bool(first(info, "x_vector_only_mode", False))
    prompt = voice_clone_prompt(info)
    if prompt is not None and isinstance(first(prompt, "icl_mode"), bool):
        in_context = first(prompt, "icl_mode")
    return in_context


def reference_codes(value):
    """`ref_code` as `[frames, 16]`, returned in this directory's `[16, frames]` layout.

    Values inside a `voice_clone_prompt` are the item's own, not list-wrapped like the top
    level: a nested list is the frames themselves. A leading batch of one is dropped.
    """
    codes = None if value is None else torch.as_tensor(value, dtype=torch.long)
    if codes is not None and codes.dim() == 3:
        codes = codes[0]
    groups = checkpoint.talker_config()["code_predictor_config"]["num_code_groups"]
    if codes is None or codes.dim() != 2 or codes.shape[1] != groups or not codes.numel():
        shape = None if codes is None else tuple(codes.shape)
        raise ValueError(f"voice_clone_prompt ref_code must be [frames, {groups}], got {shape}")
    return codes.t()


def clone_reference(info, device=None, encoders=None):
    """The request's `CloneReference`, from an encoded `voice_clone_prompt` or from `ref_audio`.

    Encoding `ref_audio` runs the speaker encoder, and in context the codec encoder, on
    `device`, with `encoders`' weights when given (see `build_clone_reference`).
    """
    in_context = clone_mode(info)
    prompt = voice_clone_prompt(info)
    text = first(info, "ref_text")
    if prompt is not None and prompt.get("ref_spk_embedding") is not None:
        codes = reference_codes(prompt.get("ref_code")) if in_context else None
        voice = torch.as_tensor(prompt["ref_spk_embedding"], dtype=torch.float32).reshape(1, -1)
        return CloneReference(codes, voice, first(prompt, "ref_text", text))

    clip = first(info, "ref_audio")
    if not isinstance(clip, (list, tuple)) or len(clip) != 2:
        raise ValueError(
            "a Base request needs ref_audio as [samples, sample_rate] or a voice_clone_prompt; "
            f"got ref_audio {type(clip).__name__}"
        )
    if device is None:
        raise ValueError("encoding ref_audio needs the device the encoders run on")
    samples, rate = clip
    audio = torch.as_tensor(samples, dtype=torch.float32).reshape(-1)
    expected = checkpoint.codec_config()["input_sample_rate"]
    if int(rate) != expected:
        import librosa

        audio = torch.from_numpy(librosa.resample(y=audio.numpy(), orig_sr=int(rate), target_sr=expected)).float()
    return build_clone_reference(device, audio, text, x_vector_only=not in_context, encoders=encoders)


REF_AUDIO_CACHE_KEY = "_qwen3_tts_ref_audio_cache_key"


class ReferenceCache:
    """Encoded clips by the key vLLM-Omni's speech server names each one with.

    The server sends a clip's audio until a request carrying it completes, then only its key,
    and expects the model to have kept what it encoded: vLLM-Omni's own talker keeps the same
    cache. The voice and the codes are kept, per cloning mode; the transcript comes from each
    request.
    """

    def __init__(self, max_entries=1024):
        self.max_entries = max_entries
        self._entries = OrderedDict()

    def reference(self, info, device=None, encoders=None):
        """The request's `CloneReference`, encoded from its clip or recalled by its key."""
        key = first(info, REF_AUDIO_CACHE_KEY)
        slot = (key, clone_mode(info))
        carries_clip = voice_clone_prompt(info) is not None or first(info, "ref_audio") is not None
        if not key or carries_clip:
            reference = clone_reference(info, device, encoders)
            if key:
                self._entries[slot] = (reference.codes, reference.speaker_embedding)
                self._entries.move_to_end(slot)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            return reference
        if slot not in self._entries:
            raise ValueError(f"a Base request names clip {key!r} without its ref_audio, and no such clip was encoded")
        self._entries.move_to_end(slot)
        codes, voice = self._entries[slot]
        return CloneReference(codes, voice, first(info, "ref_text"))


def build_stage_prefill(info, reference, prompt_len, tables):
    """The prompt `[1, prompt_len, hidden]` and the text still to feed (or `None`).

    Raises:
        ValueError: the request is not a Base clone, or asks for something Base refuses.
    """
    # vLLM-Omni's default when the field is absent, so an unlabelled request is not taken for Base.
    task_type = first(info, "task_type", "CustomVoice")
    if task_type != "Base":
        raise ValueError(f"only Base (voice clone) requests are supported, got task_type {task_type!r}")
    text = first(info, "text", "")
    language = first(info, "language", "Auto")
    instruct = first(info, "instruct")
    streaming = not bool(first(info, "non_streaming_mode", False))

    if clone_mode(info):
        if str(instruct or "").strip():
            raise ValueError("in-context cloning does not accept an instruction; use x_vector_only_mode")
        if streaming:
            embeddings, feed = build_streaming_clone_prefill(text, reference, language, tables)
        else:
            embeddings, feed = build_voice_clone_prefill(text, reference, language, tables)[0], None
    elif streaming:
        embeddings, feed = build_streaming_x_vector_prefill(text, reference, language, tables, instruct)
    else:
        embeddings, feed = build_x_vector_prefill(text, reference, language, tables, instruct)[0], None

    return fit_prompt(embeddings, prompt_len, tables), feed


def fit_prompt(embeddings, prompt_len, tables):
    """Pad with `tts_pad` or cut to the length the scheduler gave the prompt."""
    length = embeddings.shape[1]
    if length < prompt_len:
        return torch.cat([embeddings, tables.tts_pad.expand(1, prompt_len - length, -1)], dim=1)
    return embeddings[:, :prompt_len]
