# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM multimodal processing of DeepSeek-V4.1-Flash image input (imported by the frontend process: no ttnn).

The checkpoint ships no HF processor, so the processor is written here on top of the checkpoint's image processor (``tt/image_processor.py``):

  * the chat template puts one ``<｜deepseek_image｜>`` per image into the prompt text (vllm_config/chat_template.jinja; ``encoding/encoding.py`` format);
  * ``_call_hf_processor`` tokenizes the text and patchifies every image (resize plan of the checkpoint: ``patches`` [n_vit_h * n_vit_w, 3 * 14 * 14] bf16 and ``image_grid_thw`` =
    (1, n_vit_h, n_vit_w) per image: the field names the TT plugin forwards to ``prefill_forward``);
  * vLLM expands each placeholder token to the image's whole span ``[START] + ([IMAGE] * w + [NEWLINE]) * h + [END]`` (``h * (w + 1) + 2`` tokens, every one with the image token
    id: ``_get_prompt_updates``); the model tells the positions apart from the grid.
"""

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
from models.demos.blackhole.deepseek_v41_flash.tt.vision_args import vision_args_of


class DSV41ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def vision_args(self):
        if not hasattr(self, "_vargs"):
            self._vargs = vision_args_of(self.get_hf_config())
        return self._vargs

    def get_hf_processor(self, **kwargs):
        va = self.vision_args()
        return SimpleNamespace(image_token=VP.IMAGE_PLACEHOLDER, image_token_id=va.image_token_id)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}  # bounded by --limit-mm-per-prompt and the prompt length

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int] | None:
        return {"image": self.vision_args().max_tokens}

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=1344, height=1344)  # well above the 1024-token cap: the processor shrinks it to the cap

    def num_image_tokens(self, width, height):
        return VP.image_span_len(width, height, self.vision_args())


class DSV41DummyInputsBuilder(BaseDummyInputsBuilder[DSV41ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return VP.IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class DSV41MultiModalProcessor(BaseMultiModalProcessor[DSV41ProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        tok = self.info.get_tokenizer()
        out = BatchFeature(
            {
                "input_ids": torch.tensor(
                    [tok.encode(prompt, add_special_tokens=tok_kwargs.get("add_special_tokens", False))]
                )
            }
        )
        images = mm_data.get("images", [])
        if images:
            va = self.info.vision_args()
            pix, grids = [], []
            for im in images:
                patches, n_h, n_w, _, _ = VP.patchify_pil(im, va)
                pix.append(patches.reshape(n_h * n_w, -1))
                grids.append((1, n_h, n_w))
            out["pixel_values"] = torch.cat(pix).to(torch.bfloat16)
            out["image_grid_thw"] = torch.tensor(grids, dtype=torch.int64)
        return out

    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        return False  # the placeholder expansion is done by the prompt updates below

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]):
        grid = hf_inputs.get("image_grid_thw", torch.empty(0, 3, dtype=torch.int64))
        sizes = grid.prod(-1) if grid.numel() else torch.empty(0, dtype=torch.int64)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_id = self.info.vision_args().image_token_id

        def replacement(item_idx: int):
            size = mm_items.get_items("image", ImageProcessorItems).get_image_size(item_idx)
            return [image_id] * self.info.num_image_tokens(size.width, size.height)

        return [PromptReplacement(modality="image", target=[image_id], replacement=replacement)]
