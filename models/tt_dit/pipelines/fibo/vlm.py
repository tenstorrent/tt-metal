# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import torch
import transformers
from loguru import logger

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlCheckpoint
from models.tt_dit.encoders.transformer import GenerationOutput
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager

# Sampling settings and stop sequences of the upstream ``briaai/FIBO-VLM-prompt-to-JSON`` pipeline.
_TOP_P = 0.9
_TEMPERATURE = 0.2
_STOP_SEQUENCES = ("<|im_end|>", "<|end_of_text|>")

# The fields of the generated JSON that FIBO takes, and the numeric scores it maps to levels.
_FIELDS = (
    "short_description",
    "objects",
    "background_setting",
    "lighting",
    "aesthetics",
    "photographic_characteristics",
    "style_medium",
    "text_render",
    "context",
    "artistic_style",
)
_SCORES = (
    ("pickascore", "preference_score", (0.78, 0.82, 0.87, 0.91)),
    ("aesthetic_score", "aesthetic_score", (5.5, 6, 7, 7.6)),
)
_LEVELS = ("very low", "low", "medium", "high", "very high")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VlmOutput:
    """What the model generated, with the token counts of the prompt and of the generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Vlm:
    """FIBO-vlm, which writes the structured JSON prompt FIBO takes from a natural-language one.

    Follows the upstream ``briaai/FIBO-VLM-prompt-to-JSON`` pipeline: the prompt goes in as a
    ``<generate>`` message, and the sampled JSON is reduced to FIBO's fields with empty values
    dropped.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        prompt_length: int,
        cache_length: int,
    ) -> None:
        """Loads the tokenizer, the generation config and the encoder.

        ``prompt_length`` is what every prompt is padded to for the prefill, so that it runs on one
        set of compiled kernels. ``cache_length`` sizes the KV cache, bounding prompt and generated
        tokens together.
        """
        self._prompt_length = prompt_length
        self._cache_length = cache_length
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
        generation_config = transformers.GenerationConfig.from_pretrained(checkpoint_name)
        self._eos_tokens = generation_config.eos_token_id
        self._top_k = generation_config.top_k
        self._encoder = Qwen3VlCheckpoint(checkpoint_name).build(
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def generate(self, prompt: str, *, seed: int, traced: bool, max_length: int | None = None) -> str:
        """Generates FIBO's structured JSON prompt from a natural-language one.

        ``max_length`` bounds prompt and generated tokens together, up to the cache length, which is
        the default. Output that is not the expected JSON, as when it is cut off at ``max_length``,
        is returned as it is.
        """
        text = self.generate_raw(prompt, seed=seed, traced=traced, max_length=max_length).text
        caption = clean(text)
        return json.dumps(caption, separators=(",", ":")) if caption is not None else text.strip()

    def generate_raw(self, prompt: str, *, seed: int, traced: bool, max_length: int | None = None) -> VlmOutput:
        """Generates what the model emits, before it is reduced to the fields FIBO takes.

        The token counts are the model's own, which the reduced text no longer accounts for.
        """
        if max_length is None:
            max_length = self._cache_length
        tokens = self._tokenize(prompt)

        torch.manual_seed(seed)
        output = self._generate(tokens, max_length=max_length, traced=traced)
        generated = output.tokens[0, tokens.shape[1] :]
        logger.info(f"VLM generated {generated.shape[0]} tokens")

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        for stop in _STOP_SEQUENCES:
            text = text.split(stop, 1)[0]

        return VlmOutput(text=text, prompt_tokens=int(tokens.shape[1]), completion_tokens=int(generated.shape[0]))

    def warm_up(self, *, traced: bool) -> None:
        """Compiles the prefill and the decode step, tracing the latter, on two generated tokens."""
        tokens = self._tokenize("a dog")
        self._generate(tokens, max_length=tokens.shape[1] + 2, traced=traced)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenizes the chat message around ``prompt``.

        The prompt is truncated to fit ``prompt_length``, as the text encoders truncate theirs.
        """
        prompt = prompt.strip()
        tokens = self._chat_tokens(prompt)

        if tokens.shape[1] > self._prompt_length:
            # Drop the excess from the prompt's own tokens and re-tokenize the message, since a cut
            # can re-merge at the boundary; the loop ends because each pass drops at least one.
            ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            while tokens.shape[1] > self._prompt_length:
                ids = ids[: len(ids) - max(tokens.shape[1] - self._prompt_length, 1)]
                tokens = self._chat_tokens(self._tokenizer.decode(ids))
            logger.warning(f"prompt truncated to {self._prompt_length} tokens")

        return tokens

    def _chat_tokens(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": f"<generate>\n{prompt}"}]
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return encoded["input_ids"]

    def _generate(self, tokens: torch.Tensor, *, max_length: int, traced: bool) -> GenerationOutput:
        # Sampling runs one small torch op per token on the host, which more threads only slow down.
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._encoder.generate(
                tokens,
                mask=None,
                max_length=max_length,
                cache_length=self._cache_length,
                prefill_length=self._prompt_length,
                eos_tokens=self._eos_tokens,
                top_k=self._top_k,
                top_p=_TOP_P,
                temperature=_TEMPERATURE,
                traced=traced,
            )
        finally:
            torch.set_num_threads(num_threads)


def clean(text: str) -> dict[str, Any] | None:
    """Reduces generated output to the fields FIBO takes, or None when it is not in the expected format."""
    try:
        record = json.loads(text)
        caption = _drop_empty({field: record[field] for field in _FIELDS if field in record})

        scores = {name: _level(record[key], thresholds) for key, name, thresholds in _SCORES if key in record}
        if scores:
            caption.setdefault("aesthetics", {}).update(scores)

    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"VLM output is not in the expected format: {e}")
        return None

    return caption


def _level(value: float, thresholds: tuple[float, ...]) -> str:
    return _LEVELS[sum(value >= t for t in thresholds)]


def _drop_empty(value: Any) -> Any:
    """Drops ``None``, empty strings, containers and NaNs, from the leaves up."""
    if isinstance(value, dict):
        items = {k: _drop_empty(v) for k, v in value.items()}
        return {k: v for k, v in items.items() if not _is_empty(v)}
    if isinstance(value, list):
        items = [_drop_empty(v) for v in value]
        return [v for v in items if not _is_empty(v)]
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value in ("", {}, []) or (isinstance(value, float) and math.isnan(value))
