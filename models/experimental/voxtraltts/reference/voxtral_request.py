# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from .voxtral_config import DEFAULT_VOXTRAL_MODEL


def load_mistral_tokenizer(model_name_or_path: str = DEFAULT_VOXTRAL_MODEL) -> Any:
    """Load the tekken tokenizer used by Voxtral TTS."""

    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise ImportError(
            "Voxtral TTS requires mistral-common for tekken tokenization. "
            "Install vllm-omni >= 0.18.0 or `pip install mistral-common>=1.10.0`."
        ) from exc

    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        return MistralTokenizer.from_file(str(model_path / "tekken.json"))
    return MistralTokenizer.from_hf_hub(model_name_or_path)


def get_instruct_tokenizer(tokenizer: Any) -> Any:
    """Return the speech-capable instruct tokenizer across mistral-common versions."""

    instruct_tokenizer = getattr(tokenizer, "instruct_tokenizer", None) or getattr(tokenizer, "instruct", None)
    if instruct_tokenizer is None:
        raise AttributeError("MistralTokenizer does not expose `instruct_tokenizer` or `instruct`.")
    return instruct_tokenizer


def compose_speech_request(
    text: str,
    model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
    voice: str | None = "casual_male",
    ref_audio: str | None = None,
) -> dict[str, Any]:
    """Build the vLLM-Omni input dictionary for Voxtral TTS."""

    try:
        from mistral_common.protocol.speech.request import SpeechRequest
    except ImportError as exc:
        raise ImportError(
            "Voxtral TTS request construction requires mistral-common with SpeechRequest support."
        ) from exc

    if voice is None and ref_audio is None:
        raise ValueError("Either voice or ref_audio must be provided.")

    instruct_tokenizer = get_instruct_tokenizer(load_mistral_tokenizer(model_name_or_path))
    inputs: dict[str, Any] = {}

    if voice is not None:
        tokenized = instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice))
        inputs["additional_information"] = {"voice": [voice]}
        inputs["prompt_token_ids"] = tokenized.tokens
        return inputs

    with open(ref_audio, "rb") as f:
        ref_audio_bytes = f.read()
    tokenized = instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, ref_audio=ref_audio_bytes))
    audio = tokenized.audios[0]
    inputs["multi_modal_data"] = {"audio": [(audio.audio_array, audio.sampling_rate)]}
    inputs["prompt_token_ids"] = tokenized.tokens
    return inputs
