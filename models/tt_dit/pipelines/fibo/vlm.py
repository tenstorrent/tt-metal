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
from PIL import Image

import ttnn
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import (
    Qwen3VlCheckpoint,
    Qwen3VlVisionCheckpoint,
    mrope_position_ids,
)
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import pad_patches_for_sp, vision_cu_seqlens
from models.tt_dit.encoders.transformer import GenerationOutput
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor

# Sampling settings and image size bounds of the upstream ``briaai/FIBO-VLM-prompt-to-JSON``
# pipeline, and those of ``briaai/FIBO-edit-prompt-to-JSON`` for edit mode, which first shrinks an
# image to fit a 1024 x 1024 box. The bounds make an image 196 to 784 tokens, and 4 to 961 in edit
# mode. They do not affect the image the DiT gets.
_TOP_P = 0.9
_TEMPERATURE = 0.2
_EDIT_TEMPERATURE = 0.4
_TEXTURE_END = " End of texture answer."
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 1024 * 28 * 28
_EDIT_MIN_PIXELS = 4 * 28 * 28
_EDIT_MAX_PIXELS = 1280 * 28 * 28
_EDIT_IMAGE_BOX = (1024, 1024)

# The value of ``mm_token_type_ids`` that marks the image rows of a prompt.
_IMAGE_TOKEN_TYPE = 1


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
    "edit_instruction",  # Written only by FIBO-edit-vlm
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Inputs:
    """Holds a tokenized chat message and, with an image, its pixel patches and patch grid."""

    tokens: torch.Tensor
    token_types: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None


class Vlm:
    """Runs FIBO-vlm, which writes FIBO's structured JSON prompt from a prompt, an image, or both.

    Follows the upstream ``briaai/FIBO-VLM-prompt-to-JSON`` pipeline, which picks the task from the
    inputs: a prompt alone goes in as a ``<generate>`` message, an image alone as ``<inspire>``, and
    an image with a prompt as ``<refine>``, the prompt being the editing instructions. The sampled
    JSON is reduced to FIBO's fields with empty values dropped.

    With ``edit``, the checkpoint must be FIBO-edit-vlm.
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
        edit: bool,
    ) -> None:
        """Loads the processor, the generation config, the encoder and the vision tower.

        ``prompt_length`` is what every prompt is padded to for the prefill, so that it runs on one
        set of compiled kernels; it includes the tokens of an image. ``cache_length`` sizes the KV
        cache, bounding prompt and generated tokens together.
        """
        self._device = device
        self._edit = edit
        self._prompt_length = prompt_length
        self._cache_length = cache_length
        min_pixels, max_pixels = (_EDIT_MIN_PIXELS, _EDIT_MAX_PIXELS) if edit else (_MIN_PIXELS, _MAX_PIXELS)
        self._processor = transformers.AutoProcessor.from_pretrained(
            checkpoint_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self._tokenizer = self._processor.tokenizer

        # Images are padded to the largest patch count, in whole tiles, so the vision tower runs the
        # same programs for every size. Only attention differs: full for an image that fills the
        # padding, windowed otherwise. `warm_up` runs the largest and the smallest image to cover both.
        max_patches = max_pixels // self._processor.image_processor.patch_size**2
        self._max_patches = -(-max_patches // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self._warm_up_image_sides = (math.isqrt(max_pixels), math.isqrt(min_pixels))

        generation_config = transformers.GenerationConfig.from_pretrained(checkpoint_name)
        self._eos_tokens = generation_config.eos_token_id
        self._top_k = generation_config.top_k
        self._encoder = Qwen3VlCheckpoint(checkpoint_name).build(
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self._vision_tower = Qwen3VlVisionCheckpoint(checkpoint_name).build(
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def generate(
        self,
        prompt: str | None = None,
        *,
        image: Image.Image | None = None,
        seed: int,
        traced: bool,
        max_length: int | None = None,
    ) -> str:
        """Generates FIBO's structured JSON prompt from a natural-language one, an image, or both.

        With an image, a prompt that is ``None`` or blank asks for the image to be described, and any
        other is taken as instructions for how to change it.

        ``max_length`` bounds prompt and generated tokens together, up to the cache length, which is
        the default. Output that is not the expected JSON, as when it is cut off at ``max_length``,
        is returned as it is. The text is reduced, so the token counts of the model's output are
        those of `generate_raw`.

        With ``edit``, both an image and instructions are required.
        """
        if self._edit and (image is None or prompt is None or not prompt.strip()):
            msg = "editing requires an image and instructions"
            raise ValueError(msg)

        text = self.generate_raw(prompt, image=image, seed=seed, traced=traced, max_length=max_length).text
        if self._edit:
            # FIBO-edit-vlm ends texture descriptions with this.
            text = text.replace(_TEXTURE_END, "")
        caption = clean(text)
        return json.dumps(caption, separators=(",", ":")) if caption is not None else text.strip()

    def generate_raw(
        self,
        prompt: str | None = None,
        *,
        image: Image.Image | None = None,
        seed: int,
        traced: bool,
        max_length: int | None = None,
    ) -> VlmOutput:
        """Generates what the model emits, before it is reduced to the fields FIBO takes."""
        if max_length is None:
            max_length = self._cache_length
        inputs = self._tokenize(prompt, image)
        num_prompt_tokens = inputs.tokens.shape[1]

        torch.manual_seed(seed)
        output = self._generate(inputs, max_length=max_length, traced=traced)
        generated = output.tokens[0, num_prompt_tokens:]
        logger.info(f"VLM generated {generated.shape[0]} tokens")

        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return VlmOutput(text=text, prompt_tokens=int(num_prompt_tokens), completion_tokens=int(generated.shape[0]))

    def warm_up(self, *, traced: bool) -> None:
        """Compiles the prefill and the decode step, tracing the latter, on two generated tokens.

        An untraced call also compiles the image path.
        """
        if not traced:
            for side in self._warm_up_image_sides:
                inputs = self._tokenize(None, Image.new("RGB", (side, side)))
                self._generate(inputs, max_length=inputs.tokens.shape[1] + 2, traced=False)

        inputs = self._tokenize("a dog", None)
        self._generate(inputs, max_length=inputs.tokens.shape[1] + 2, traced=traced)

    def _tokenize(self, prompt: str | None, image: Image.Image | None) -> _Inputs:
        """Tokenizes the chat message of the task that ``prompt`` and ``image`` make.

        The prompt is truncated to fit ``prompt_length``, as the text encoders truncate theirs.
        """
        prompt = prompt.strip() if prompt is not None else ""
        if self._edit and image is not None:
            image = image.copy()
            image.thumbnail(_EDIT_IMAGE_BOX)
        inputs = self._chat_inputs(prompt, image)

        if inputs.tokens.shape[1] > self._prompt_length:
            # Cut the prompt, not the message, whose closing tokens the model needs to answer, and
            # rebuild the message around it; repeat, as the seam can tokenize differently.

            ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            while inputs.tokens.shape[1] > self._prompt_length and ids:
                excess = inputs.tokens.shape[1] - self._prompt_length
                ids = ids[: max(len(ids) - excess, 0)]
                inputs = self._chat_inputs(self._tokenizer.decode(ids), image)

            logger.warning(f"prompt truncated to {self._prompt_length} tokens")

        return inputs

    def _chat_inputs(self, prompt: str, image: Image.Image | None) -> _Inputs:
        """Tokenizes the chat message, expanding the image into its tokens and patches."""
        if image is None:
            content = f"<generate>\n{prompt}"
        elif not prompt:
            content = [{"type": "image"}, {"type": "text", "text": "<inspire>"}]
        elif self._edit:
            content = [{"type": "image"}, {"type": "text", "text": f"<edit>\nEditing instructions:\n{prompt}"}]
        else:
            content = [{"type": "image"}, {"type": "text", "text": f"<refine>\nEditing instructions:\n{prompt}"}]

        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        encoded = self._processor(
            text=[text],
            images=[image] if image is not None else None,
            return_tensors="pt",
        )
        return _Inputs(
            tokens=encoded["input_ids"],
            token_types=encoded["mm_token_type_ids"],
            pixel_values=encoded.get("pixel_values"),
            image_grid_thw=encoded.get("image_grid_thw"),
        )

    def _generate(self, inputs: _Inputs, *, max_length: int, traced: bool) -> GenerationOutput:
        return self._encoder.generate(
            inputs.tokens,
            mask=None,
            max_length=max_length,
            cache_length=self._cache_length,
            prefill_length=self._prompt_length,
            eos_tokens=self._eos_tokens,
            top_k=self._top_k,
            top_p=_TOP_P,
            temperature=_EDIT_TEMPERATURE if self._edit else _TEMPERATURE,
            traced=traced,
            **self._vision_args(inputs),
        )

    def _vision_args(self, inputs: _Inputs) -> dict[str, Any]:
        """Run the vision tower on the image of ``inputs``, if any."""
        if inputs.pixel_values is None or inputs.image_grid_thw is None:
            return {}

        vision_embeds, deepstack_embeds = self._encode_image(inputs.pixel_values, inputs.image_grid_thw)
        return {
            "positions": mrope_position_ids(
                inputs.token_types,
                image_grid_thw=inputs.image_grid_thw,
                spatial_merge_size=self._vision_tower.spatial_merge_size,
            ),
            "vision_embeds": vision_embeds,
            "vision_mask": inputs.token_types == _IMAGE_TOKEN_TYPE,
            "deepstack_embeds": deepstack_embeds,
        }

    def _encode_image(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """Run the vision tower, returning its merged tokens and one feature per deepstack layer."""
        tower = self._vision_tower

        patches, pos_embeds, (cos, sin), cu_seqlens, _ = pad_patches_for_sp(
            pixel_values,
            tower.prepare_pos_embeds(grid_thw),
            tower.prepare_rope(grid_thw),
            vision_cu_seqlens(grid_thw),
            sp_factor=1,
            length=self._max_patches,
        )

        return tower.forward(
            tensor.from_torch(patches, device=self._device),
            pos_embeds=tensor.from_torch(pos_embeds, device=self._device),
            rope=(tensor.from_torch(cos, device=self._device), tensor.from_torch(sin, device=self._device)),
            cu_seqlens=cu_seqlens,
        )


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
