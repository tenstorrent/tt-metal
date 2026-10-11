# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in image/video adapter; retains the TP4 B16 asynchronous decoder.

Register as TTQwen38ForConditionalGeneration. This is a hardware-unqualified
feature branch, separate from the GPQA-qualified text-only architecture.
"""

import torch
from transformers import AutoConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM
from models.demos.qwen38_27b_qb2.tt.multimodal import build_plan, gather_media, item_identity
from models.demos.qwen38_27b_qb2.tt.vision import Qwen38VisionEncoder


class Qwen38ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": 4, "video": 1}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=Qwen38ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen38ForConditionalGeneration(Qwen38ForCausalLM, SupportsMultiModal):
    decode_input_update_contract = 1
    model_capabilities = {
        **Qwen38ForCausalLM.model_capabilities,
        "supports_video_inputs": True,
        "supports_multimodal_chunked_prefill": True,
    }

    def __init__(self, generator, batch_size, context, **kwargs):
        super().__init__(generator, batch_size, context, **kwargs)
        self.mm_config = AutoConfig.from_pretrained(generator.model.snapshot, local_files_only=True)
        self.vision_encoder = Qwen38VisionEncoder(generator.model, self.mm_config)
        self._mm_plans = {}
        self._mm_deltas = torch.zeros(batch_size, dtype=torch.int64)

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality in ("image", "video"):
            return f"<|vision_start|><|{modality}_pad|><|vision_end|>"
        raise ValueError("Only image and video inputs are supported")

    def _prepare_multimodal_rows(self, tokens, starts, ends, slots, kwargs):
        count = len(ends)
        names = ("mm_request_ids", "mm_prompt_token_ids", "mm_item_spans")
        if any(name not in kwargs or len(kwargs[name]) != count for name in names):
            raise ValueError("Multimodal adapter requires row-aligned request IDs, complete prompts and item spans")
        for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if name in kwargs and len(kwargs[name]) != count:
                raise ValueError(f"{name} must follow scheduled prefill row order")
        proposed = dict(self._mm_plans)
        plans = []
        for row, (start, end, slot) in enumerate(zip(starts, ends, slots)):
            request_id = kwargs["mm_request_ids"][row]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("Each multimodal row needs a nonempty request ID")
            prompt = torch.as_tensor(kwargs["mm_prompt_token_ids"][row], dtype=torch.int64).reshape(-1)
            if not 0 <= start < end <= len(prompt) <= self.context:
                raise ValueError("Scheduled prefill bounds exceed the full multimodal prompt")
            if not torch.equal(prompt[start:end], tokens[row, start:end].to(dtype=torch.int64)):
                raise ValueError("Scheduled token chunk differs from the complete multimodal prompt")
            identity = item_identity(kwargs["mm_item_spans"][row])
            if any(offset + length > len(prompt) for _, _, offset, length in identity):
                raise ValueError("Multimodal span exceeds its prompt")
            has_placeholders = bool(
                ((prompt == self.mm_config.image_token_id) | (prompt == self.mm_config.video_token_id)).any()
            )
            if not identity:
                if has_placeholders:
                    raise ValueError("Visual placeholders arrived without media identity")
                if any(
                    kwargs.get(name, [None] * count)[row] is not None
                    for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
                ):
                    raise ValueError("Visual payload arrived without media identity")
                # A fresh text request must discard the prior occupant's image
                # state. Continuation cannot silently turn a visual request text.
                if start and slot in proposed and proposed[slot].request_id == request_id:
                    raise ValueError("Multimodal continuation lost its media identity")
                proposed.pop(slot, None)
                plans.append(None)
                continue
            existing = proposed.get(slot)
            if start:
                if existing is None or not existing.matches(request_id, identity, prompt):
                    raise ValueError("No matching multimodal plan for this request/slot continuation")
                plans.append(existing)
                continue
            # New request or resumed-from-zero request always re-encodes. A
            # None media payload may be reused only by its exact live plan.
            image = gather_media(
                kwargs.get("pixel_values", [None] * count)[row], kwargs.get("image_grid_thw", [None] * count)[row]
            )
            video = gather_media(
                kwargs.get("pixel_values_videos", [None] * count)[row],
                kwargs.get("video_grid_thw", [None] * count)[row],
            )
            if image is None and video is None:
                raise ValueError("Multimodal request has no pixel payload")
            self.generator._release_traces()
            plan = build_plan(request_id, identity, prompt, image, video, self.vision_encoder, config=self.mm_config)
            proposed[slot] = plan
            plans.append(plan)
        return plans, proposed

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None, **kwargs):
        self._cache(kv_cache)
        tokens = torch.as_tensor(tokens)
        ends = torch.as_tensor(prompt_lens).reshape(-1).tolist()
        starts = [0] * len(ends) if start_pos is None else torch.as_tensor(start_pos).reshape(-1).tolist()
        slots = list(range(len(ends))) if empty_slots is None else list(empty_slots)
        if len(starts) != len(ends) or len(slots) != len(ends) or len(set(slots)) != len(slots):
            raise ValueError("Prefill rows require distinct slots and matching positions")
        if tokens.ndim != 2 or tokens.shape[0] != len(ends) or any(not 0 <= slot < self.batch_size for slot in slots):
            raise ValueError("Invalid multimodal prefill rows or slots")
        if not any(name in kwargs for name in ("mm_request_ids", "mm_prompt_token_ids", "mm_item_spans")):
            # vLLM warmup may call the adapter directly with text-only dummy
            # tokens. It has no request metadata. Real visual inputs still
            # require the plugin's complete-prompt/identity contract.
            def contains_pixels(value):
                if isinstance(value, torch.Tensor):
                    return value.numel() > 0
                return (
                    any(contains_pixels(item) for item in value)
                    if isinstance(value, (list, tuple))
                    else value is not None
                )

            if contains_pixels(kwargs.get("pixel_values")) or contains_pixels(kwargs.get("pixel_values_videos")):
                raise ValueError("Visual prefill requires complete prompt and request metadata")
            if any(start and slot in self._mm_plans for start, slot in zip(starts, slots)):
                raise ValueError("Multimodal continuation lost its request metadata")
            kwargs = {
                **kwargs,
                "mm_request_ids": [f"direct-text-{slot}" for slot in slots],
                "mm_prompt_token_ids": [tokens[row, :end].tolist() for row, end in enumerate(ends)],
                "mm_item_spans": [[] for _ in ends],
            }
        plans, proposed = self._prepare_multimodal_rows(tokens, starts, ends, slots, kwargs)
        output, _ = super().prefill_forward(
            tokens,
            page_table,
            kv_cache,
            prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            **({"multimodal_plans": plans} if any(plan is not None for plan in plans) else {}),
            **kwargs,
        )
        self._mm_plans = proposed
        deltas = torch.tensor([plan.rope_delta if plan is not None else 0 for plan in plans], dtype=torch.int64)
        self._mm_deltas[slots] = deltas
        return output, deltas

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        rope_deltas_all_users=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        **kwargs,
    ):
        self._cache(kv_cache)
        commands = (reload_inputs, reload_page_table, reload_sampling_params, reset_sampling_state)
        if any(type(flag) is not bool for flag in commands):
            raise ValueError("Decode reload commands must be booleans")
        if reload_inputs and reload_page_table:
            raise ValueError("Input reload already includes the page table")
        if reset_sampling_state and not (reload_inputs and reload_sampling_params):
            raise ValueError("Sampling reset requires input and parameter reload")
        if not self._decode_bound and not reload_inputs:
            raise ValueError("First decode after prefill requires authoritative input reload")
        proposed = dict(self._mm_plans)
        deltas = self._mm_deltas.clone()
        if slot_remap is not None:
            order = torch.as_tensor(slot_remap).reshape(-1).tolist()
            if sorted(order) != list(range(self.batch_size)) or not reload_inputs:
                raise ValueError("Slot remap requires a complete permutation and input reload")
            proposed = {row: proposed[old] for row, old in enumerate(order) if old in proposed}
            deltas = deltas[order]
        if rope_deltas_all_users is not None:
            supplied = torch.as_tensor(rope_deltas_all_users, dtype=torch.int64).reshape(-1)
            if not 1 <= len(supplied) <= self.batch_size:
                raise ValueError("M-RoPE delta snapshot exceeds decode rows")
            deltas = torch.zeros_like(deltas)
            deltas[: len(supplied)] = supplied
            if not reload_inputs and not torch.equal(deltas, self._mm_deltas):
                raise ValueError("Changed rotary deltas require input reload")
        positions = torch.as_tensor(start_pos).reshape(-1)
        device_sampling = sampling_params is not None
        if not device_sampling or reload_sampling_params:
            device_sampling = self._sampling(
                sampling_params,
                reset=reset_sampling_state,
                output_positions=(positions + 1).tolist() if reset_sampling_state else None,
            )
        if (
            self._last_device_sampling is not None
            and self._last_device_sampling != device_sampling
            and not reload_inputs
        ):
            raise ValueError("Sampling backend transition requires input reload")
        if slot_remap is not None:
            self.generator.remap_recurrent_slots(slot_remap)
        active = (positions >= 0).nonzero().reshape(-1).tolist() if reload_inputs else None
        table = self._table(page_table) if reload_inputs or reload_page_table else self.generator.page_table
        result = self.generator.decode_forward(
            tokens=tokens if reload_inputs else None,
            start_pos=positions if reload_inputs else None,
            page_table=table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=False,
            active_slots=active,
            host_sampling=not device_sampling,
            rope_deltas=deltas if reload_inputs else None,
        )
        self._mm_plans, self._mm_deltas = proposed, deltas
        self._decode_bound = True
        self._last_device_sampling = device_sampling
        if not device_sampling:
            return result.reshape(self.batch_size, 1, -1)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(result), is_tokens=True)
        return result
