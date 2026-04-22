# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .embeddings import build_inputs_embeds
from .hf_utils import HFLoadSpec, load_processor_and_model
from .rope import Qwen2RopeHelper
from .vision import vision_tower_forward


@dataclass
class DotsOCRInputs:
    """
    Canonical inputs for our modular pipeline.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


class DotsOCRReference:
    """
    Thin wrapper around HF Dots OCR model that exposes modular entry points.

    This is used as the correctness oracle for TTNN modules via PCC.
    """

    def __init__(self, spec: HFLoadSpec):
        self.processor, self.model = load_processor_and_model(spec)
        self.config = self.model.config

        # The upstream Dots vision tower defaults to `bf16=True` in its forward() signature and
        # will cast `pixel_values` to bf16 unconditionally. On CPU/fp32 runs this can crash in
        # Conv2d due to dtype mismatch (bf16 input vs fp32 bias). For reference correctness runs,
        # keep vision in fp32 by forcing `bf16=False` when the model is loaded in fp32.
        if spec.dtype == torch.float32:
            vt = getattr(self.model, "vision_tower", None) or getattr(self.model, "visual", None)
            if vt is not None and hasattr(vt, "forward"):
                try:
                    orig_forward = vt.forward

                    def _forward_force_fp32(self_vt, hidden_states, grid_thw, bf16=True):
                        return orig_forward(hidden_states, grid_thw, bf16=False)

                    vt.forward = types.MethodType(_forward_force_fp32, vt)  # type: ignore[assignment]
                except Exception:
                    pass

    @property
    def tokenizer(self):
        # Expose a `.tokenizer` for API parity with other multimodal reference wrappers.
        return getattr(self.processor, "tokenizer", None)

    @property
    def image_token_id(self) -> int:
        return int(getattr(self.config, "image_token_id"))

    def preprocess_image_and_prompt(self, image, prompt: str) -> DotsOCRInputs:
        """
        Uses the HF processor to create token ids + pixel_values + image_grid_thw.

        Multimodal checkpoints (Dots, Qwen2-VL) accept ``images=``; plain CausalLM
        tokenizers only accept ``text=`` — branch accordingly for text-only tests.
        """

        def _run_processor(_prompt: str):
            try:
                return self.processor(images=image, text=_prompt, return_tensors="pt")
            except TypeError:
                if image is not None:
                    raise
                return self.processor(text=_prompt, return_tensors="pt")

        # Many multimodal LMs require a chat template to behave correctly (esp. OCR-like transcription).
        # Prefer the *processor* chat template when available (it knows how to insert image placeholders),
        # fall back to the tokenizer's template, then raw prompt.
        prompt_to_use = prompt
        if image is not None:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                apply_p = getattr(self.processor, "apply_chat_template", None)
                if callable(apply_p):
                    prompt_to_use = apply_p(messages, tokenize=False, add_generation_prompt=True)
                else:
                    tok = getattr(self.processor, "tokenizer", None)
                    apply_t = getattr(tok, "apply_chat_template", None) if tok is not None else None
                    if callable(apply_t):
                        prompt_to_use = apply_t(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_to_use = prompt

        inputs = _run_processor(prompt_to_use)
        # If an image is provided, the model expects `input_ids` to contain `config.image_token_id`
        # positions where vision embeddings are scattered. Some processor/tokenizer versions only
        # insert those tokens if the prompt contains the *exact* special token string for that id.
        if image is not None and "input_ids" in inputs:
            image_token_id = int(getattr(self.model.config, "image_token_id"))
            img_mask_sum = int((inputs["input_ids"] == image_token_id).sum().item())
            if img_mask_sum == 0:
                tok = None
                try:
                    tok = self.processor.tokenizer.convert_ids_to_tokens(image_token_id)  # type: ignore[attr-defined]
                except Exception:
                    tok = None
                if tok:
                    inputs = _run_processor(f"{tok}\n{prompt}")
                # If still no image tokens, fail fast with a clear diagnostic instead of letting
                # remote-code assert deep in `prepare_inputs_embeds`.
                img_mask_sum2 = int((inputs["input_ids"] == image_token_id).sum().item())
                if img_mask_sum2 == 0:
                    raise ValueError(
                        f"DotsOCR processor produced 0 image tokens for image_token_id={image_token_id}. "
                        f"Tried prefix token={tok!r}. Please include the model's image placeholder token in the prompt."
                    )
        return DotsOCRInputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            pixel_values=(
                inputs.get("pixel_values").to(dtype=next(self.model.parameters()).dtype)
                if inputs.get("pixel_values") is not None
                else None
            ),
            image_grid_thw=inputs.get("image_grid_thw"),
        )

    def decode_generated_suffix(self, full_sequences: torch.Tensor, prompt_input_ids: torch.Tensor) -> str:
        """
        Decode only tokens **after** the prompt. ``model.generate`` returns the full sequence
        (prompt + continuation); decoding the whole tensor repeats the user instruction in the string.
        """
        prompt_len = prompt_input_ids.shape[1]
        new_tokens = full_sequences[:, prompt_len:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    @torch.no_grad()
    def forward(self, inputs: DotsOCRInputs, max_new_tokens: int = 32, **generate_kwargs: Any) -> Dict[str, Any]:
        """
        End-to-end reference forward using HF ``generate()``.

        Pass optional ``transformers`` generation kwargs (e.g. ``repetition_penalty``, ``do_sample``)
        as keyword arguments; they are merged after ``max_new_tokens`` and must not replace tensor inputs.
        """
        gen_kwargs: Dict[str, Any] = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
        }
        # Plain CausalLM.generate() does not accept vision kwargs; multimodal checkpoints do.
        if inputs.pixel_values is not None:
            # Ensure vision inputs match model dtype (Conv2d requires input/bias dtypes to match).
            try:
                model_dtype = next(self.model.parameters()).dtype
            except Exception:
                model_dtype = inputs.pixel_values.dtype
            pv = inputs.pixel_values.to(dtype=model_dtype)
            print(f"[debug] model_param_dtype={model_dtype} pixel_values_dtype={pv.dtype}")
            gen_kwargs["pixel_values"] = pv
        if inputs.image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = inputs.image_grid_thw
        _reserved = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        for key, value in generate_kwargs.items():
            if key in _reserved or value is None:
                continue
            gen_kwargs[key] = value
        gen = self.model.generate(**gen_kwargs)
        return {"generated_ids": gen}

    @torch.no_grad()
    def vision_forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Vision token rows from `DotsVisionTransformer` (`model.vision_tower`).

        Args use processor keys: `pixel_values`, `image_grid_thw` (passed as `grid_thw` to the tower).
        """
        return vision_tower_forward(self.model, pixel_values, image_grid_thw)

    @torch.no_grad()
    def build_inputs_embeds(self, inputs: DotsOCRInputs) -> torch.Tensor:
        """
        Fused `inputs_embeds` [B, S, D] via HF `prepare_inputs_embeds` (vision + text).
        """
        return build_inputs_embeds(
            self.model,
            inputs.input_ids,
            inputs.pixel_values,
            inputs.image_grid_thw,
        )

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Get raw logits from the model for PCC comparison (text-only).

        This is used to validate that TTNN prefill produces the same logits
        as the HF reference model.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def get_rope_helper(self) -> Qwen2RopeHelper:
        """Get RoPE helper configured from model parameters."""
        config = self.model.config
        head_dim = getattr(
            config, "head_dim", getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
        )
        theta = getattr(config, "rope_theta", 10000.0)
        max_seq_len = getattr(config, "max_position_embeddings", 8192)

        return Qwen2RopeHelper(head_dim=head_dim, max_seq_len=max_seq_len, theta=theta)
