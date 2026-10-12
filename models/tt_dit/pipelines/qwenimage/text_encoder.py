# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
import transformers
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.encoders.mrope import mrope_position_ids
from models.tt_dit.encoders.qwen25vl import Qwen25VlCheckpoint, Qwen25VlEncoder
from models.tt_dit.encoders.qwen25vl.vision_qwen25vl import Qwen25VlVisionModel
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.utils import tensor
from models.tt_dit.utils.mesh import reshape_for_factor
from models.tt_dit.utils.padding import torch_pad
from models.tt_dit.utils.tracing import Tracer

PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
PROMPT_DROP_IDX = 34

EDIT_PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
EDIT_PROMPT_DROP_IDX = 64


@dataclass(frozen=True)
class PromptFormat:
    """How a pipeline wraps its prompt for the text encoder.

    ``template`` takes the prompt, and the first ``drop_idx`` tokens of the encoding are dropped.
    With ``image_prompt``, each input image is announced before the prompt by ``image_prompt``,
    formatted with the image's 1-based number.
    """

    template: str
    drop_idx: int
    image_prompt: str | None = None


TEXT_TO_IMAGE_FORMAT = PromptFormat(PROMPT_TEMPLATE, PROMPT_DROP_IDX)
# Qwen-Image-Edit: one image, placed right before the prompt.
EDIT_FORMAT = PromptFormat(
    EDIT_PROMPT_TEMPLATE,
    EDIT_PROMPT_DROP_IDX,
    image_prompt="<|vision_start|><|image_pad|><|vision_end|>",
)
# Qwen-Image-Edit-2509 and 2511: numbered images.
EDIT_PLUS_FORMAT = PromptFormat(
    EDIT_PROMPT_TEMPLATE,
    EDIT_PROMPT_DROP_IDX,
    image_prompt="Picture {}: <|vision_start|><|image_pad|><|vision_end|>",
)


@dataclass(frozen=True)
class _ImageInputs:
    images: list[Image.Image]
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    # The vision tower's output on the device, None on the PyTorch path.
    embeds: ttnn.Tensor | None


class TextEncoder:
    """QwenImage's Qwen2.5-VL text encoder wrapper with PyTorch fallback.

    Each prompt is encoded on its own, padded to the smallest of ``sequence_length_buckets`` it
    fits, counted after the template prefix is dropped. Each CFG pass is padded to the longest
    length among its prompts, and both passes to a common length when they are batched together.

    With a ``prompt_format`` that has an ``image_prompt``, the encoder also takes images, which run
    through the vision tower once per call and are shared by all prompts. Image prompts are not
    traced, since the vision tower can't be.
    """

    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        sequence_length_buckets: Sequence[int],
        use_torch: bool,
        prompt_format: PromptFormat = TEXT_TO_IMAGE_FORMAT,
        offload: bool = False,
    ) -> None:
        """Build the encoder, and the vision tower if ``prompt_format`` takes images.

        With ``offload``, the device weights are loaded for each ``encode_cfg`` and freed after it,
        which leaves their memory to the transformer. Encoding is then never traced.
        """
        self._device = device
        self._parallel_config = parallel_config
        self._format = prompt_format
        self._offload = offload and not use_torch
        self._tokenizer = transformers.Qwen2Tokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._processor: transformers.Qwen2VLProcessor | None = None
        if prompt_format.image_prompt is not None:
            self._processor = transformers.Qwen2VLProcessor.from_pretrained(checkpoint_name, subfolder="processor")

        for bucket in sequence_length_buckets:
            if bucket <= 0 or bucket % 32 != 0:
                msg = f"bucket {bucket} must be a positive multiple of 32"
                raise ValueError(msg)
        self._sequence_length_buckets = sorted(set(sequence_length_buckets))

        self._torch_encoder: transformers.Qwen2_5_VLForConditionalGeneration | None = None
        self._encoder: Qwen25VlEncoder | None = None
        self._vision: Qwen25VlVisionModel | None = None
        # One tracer per encoder input length, created on first use.
        self._tracers: dict[int, Tracer] = {}

        if use_torch:
            self._torch_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_name, subfolder="text_encoder"
            )
            self._torch_encoder.eval()
        else:
            logger.info("loading text encoder weights to device...")
            self._checkpoint = Qwen25VlCheckpoint(checkpoint_name, subfolder="text_encoder")
            with self._reshape():
                self._encoder = self._checkpoint.build(
                    device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
                )
                if self._processor is not None:
                    self._vision = self._checkpoint.build_vision(
                        device=device, parallel_config=parallel_config, ccl_manager=ccl_manager
                    )
            ttnn.synchronize_device(device)
            if self._offload:
                self._deallocate_weights()

    @torch.no_grad()
    def encode_cfg(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        *,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        batch_passes: bool,
        traced: bool,
        images: Sequence[Image.Image] = (),
        on_event: PipelineEventCallback = null_callback,
    ) -> list[torch.Tensor]:
        """Returns the prompt embeddings of each CFG pass, the negative pass first.

        With ``batch_passes`` the passes are padded to a common length and batched into one entry.
        Padded positions are zero. ``images`` go into every prompt, in the size the processor
        should see them in.
        """
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"
        if images and self._processor is None:
            msg = "this prompt format takes no images"
            raise ValueError(msg)

        if not cfg_enabled:
            cfg_passes = [prompts]
        elif batch_passes:
            cfg_passes = [[*negative_prompts, *prompts]]
        else:
            cfg_passes = [negative_prompts, prompts]

        on_event(SectionStart("qwen_encoding"))
        outputs = []
        with self._reshape():
            if self._offload:
                traced = False
                self._load_weights()
            image_inputs = self._encode_images(images) if images else None
            for cfg_pass in cfg_passes:
                if image_inputs is not None:
                    embeds = [self._encode_image_prompt(prompt, image_inputs) for prompt in cfg_pass]
                else:
                    embeds = [self._encode_prompt(prompt, traced=traced) for prompt in cfg_pass]
                length = max(e.shape[1] for e in embeds)
                embeds = torch.cat([torch_pad(e, length - e.shape[1], dim=1) for e in embeds])
                outputs.append(embeds.repeat_interleave(num_images_per_prompt, dim=0))
            if image_inputs is not None and image_inputs.embeds is not None:
                ttnn.deallocate(image_inputs.embeds)
            if self._offload:
                self._deallocate_weights()
        on_event(SectionEnd("qwen_encoding"))

        return outputs

    def _load_weights(self) -> None:
        assert self._encoder is not None
        self._checkpoint.load_weights(self._encoder, device=self._device, parallel_config=self._parallel_config)
        if self._vision is not None:
            self._checkpoint.load_vision_weights(
                self._vision, device=self._device, parallel_config=self._parallel_config
            )

    def _deallocate_weights(self) -> None:
        assert self._encoder is not None
        self._encoder.deallocate_weights()
        if self._vision is not None:
            self._vision.deallocate_weights()

    def _encode_images(self, images: Sequence[Image.Image]) -> _ImageInputs:
        assert self._processor is not None
        inputs = self._processor.image_processor(images=list(images), return_tensors="pt")
        pixel_values, grid_thw = inputs.pixel_values, inputs.image_grid_thw

        embeds = None
        if self._vision is not None:
            device = self._device
            vision_inputs = self._vision.prepare_inputs(pixel_values, grid_thw)
            embeds = self._vision.forward(
                tensor.from_torch(vision_inputs.patches, device=device),
                rope=(
                    tensor.from_torch(vision_inputs.cos, device=device),
                    tensor.from_torch(vision_inputs.sin, device=device),
                ),
                reverse_index=tensor.from_torch(
                    vision_inputs.reverse_index.unsqueeze(0),
                    device=device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                cu_seqlens=vision_inputs.cu_seqlens,
                cu_window_seqlens=vision_inputs.cu_window_seqlens,
            )
        return _ImageInputs(images=list(images), pixel_values=pixel_values, grid_thw=grid_thw, embeds=embeds)

    def prompt_length(self, prompt: str, *, images: Sequence[Image.Image] = ()) -> int:
        """The token count of ``prompt`` with ``images`` past the template prefix, before padding."""
        if images:
            tokens = self._image_prompt_inputs(prompt, images).input_ids[0]
        else:
            tokens = self._tokenizer(self._format.template.format(prompt)).input_ids
        return len(tokens) - self._format.drop_idx

    def _image_prompt_inputs(self, prompt: str, images: Sequence[Image.Image]) -> transformers.BatchFeature:
        assert self._processor is not None
        assert self._format.image_prompt is not None
        announcement = "".join(self._format.image_prompt.format(i + 1) for i in range(len(images)))
        return self._processor(
            text=[self._format.template.format(announcement + prompt)], images=list(images), return_tensors="pt"
        )

    def _encode_image_prompt(self, prompt: str, image_inputs: _ImageInputs) -> torch.Tensor:
        """Like ``_encode_prompt``, for a prompt that shows ``image_inputs``; not traced."""
        drop_idx = self._format.drop_idx
        inputs = self._image_prompt_inputs(prompt, image_inputs.images)
        tokens, type_ids = inputs.input_ids, inputs.mm_token_type_ids
        token_count = tokens.shape[1]

        if self._torch_encoder is not None:
            output = self._torch_encoder.forward(
                input_ids=tokens,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                # Without it, transformers 5 gives the image tokens sequential positions instead of
                # the multimodal ones.
                mm_token_type_ids=type_ids,
                output_hidden_states=True,
            )
            embeds = output.hidden_states[-1].to("cpu")
            length = self._bucket(token_count - drop_idx) + drop_idx
            return torch_pad(embeds, length - token_count, dim=1)[:, drop_idx:]

        assert self._encoder is not None
        assert self._vision is not None
        assert image_inputs.embeds is not None
        positions = mrope_position_ids(
            type_ids, image_grid_thw=image_inputs.grid_thw, spatial_merge_size=self._vision.spatial_merge_size
        )

        # The padding at the end is masked, and the encoder is causal, so it doesn't change the
        # prompt's states.
        length = self._bucket(token_count - drop_idx) + drop_idx
        padding = length - token_count
        pad_token_id = self._tokenizer.pad_token_id
        tokens = torch.nn.functional.pad(tokens, (0, padding), value=pad_token_id)
        mask = torch_pad(torch.ones([1, token_count], dtype=torch.long), padding, dim=1)
        vision_mask = torch_pad((type_ids == 1).int(), padding, dim=1)
        positions = torch_pad(positions, padding, dim=2)

        device = self._device
        tt_embeds = self._encoder.forward(
            tensor.from_torch(tokens, device=device, dtype=ttnn.uint32),
            mask=tensor.from_torch(mask, device=device),
            positions=tensor.from_torch(positions.float(), device=device, dtype=ttnn.float32),
            vision_embeds=image_inputs.embeds,
            vision_mask=tensor.from_torch(vision_mask, device=device),
            skip_final_linear=True,
        )
        embeds = tensor.to_torch(tt_embeds)
        embeds[mask == 0] = 0.0
        return embeds[:, drop_idx:]

    def _encode_prompt(self, prompt: str, *, traced: bool) -> torch.Tensor:
        """Returns the last hidden state of ``prompt`` past the template prefix, at its bucket length."""
        drop_idx = self._format.drop_idx
        max_length = self._sequence_length_buckets[-1] + drop_idx
        tokens, mask = self._tokenize(self._format.template.format(prompt), sequence_length=max_length)

        length = self._bucket(int(mask.sum()) - drop_idx) + drop_idx
        tokens, mask = tokens[:, :length], mask[:, :length]

        if self._torch_encoder is not None:
            output = self._torch_encoder.forward(
                tokens.to(device=self._torch_encoder.device),
                attention_mask=mask.to(device=self._torch_encoder.device),
                output_hidden_states=True,
            )
            embeds = output.hidden_states[-1].to("cpu")
        else:
            embeds = self._encode_tokens(tokens, mask, traced=traced)

        embeds[mask == 0] = 0.0
        return embeds[:, drop_idx:]

    def _encode_tokens(self, tokens: torch.Tensor, mask: torch.Tensor, *, traced: bool) -> torch.Tensor:
        assert self._encoder is not None

        forward = self._encoder.forward
        if traced:
            length = tokens.shape[1]
            if length not in self._tracers:
                self._tracers[length] = Tracer(self._encoder.forward, device=self._device, clone_prep_inputs=False)
            forward = self._tracers[length]

        tt_tokens = tensor.from_torch(tokens, device=self._device, dtype=ttnn.uint32)
        tt_mask = tensor.from_torch(mask, device=self._device)
        tt_embeds = forward(tt_tokens, mask=tt_mask, skip_final_linear=True)

        # The tracer reuses its output tensor on every call, so read back before the next one.
        return tensor.to_torch(tt_embeds)

    def _reshape(self) -> AbstractContextManager[None]:
        """Reshapes the device so that the tensor parallel axis spans exactly the tensor parallel factor."""
        return reshape_for_factor(self._device, self._parallel_config.tensor_parallel)

    def _bucket(self, token_count: int) -> int:
        buckets = [bucket for bucket in self._sequence_length_buckets if bucket >= token_count]
        if not buckets:
            msg = f"the prompt's {token_count} tokens exceed the largest bucket {self._sequence_length_buckets[-1]}"
            raise ValueError(msg)
        return buckets[0]

    def _tokenize(self, prompt: str, *, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )

        if len(self._tokenizer(prompt).input_ids) > sequence_length:
            logger.warning("input text was truncated")

        return tokenized.input_ids, tokenized.attention_mask
