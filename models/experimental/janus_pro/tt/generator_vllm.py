# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
vLLM generator bridge for Janus-Pro-7B (image+text -> text understanding).

This is the tt-metal side of the vLLM integration: the class vLLM's TT plugin
instantiates for the HF architecture ``JanusForConditionalGeneration`` (the
transformers-native ``deepseek-community/Janus-Pro-7B`` repo). It mirrors the
non-hybrid multimodal bridges in
``models/tt_transformers/tt/generator_vllm.py`` (``Mistral3...``, ``Mllama...``):
Janus's text decoder is a plain LLaMA-style full-attention stack with a single
global RoPE, so this inherits from :class:`Generator` (via
:class:`JanusMultimodalGenerator`) rather than
:class:`HybridAttentionForCausalLM`.

What this bridge reuses vs. adds:
  * Vision tower + host-side ``masked_scatter`` fusion already live in
    :class:`TtJanusProModel` / :class:`JanusMultimodalGenerator`
    (``janus_pro_e2e_model.py``). This class only adapts vLLM's calling
    convention (``tokens`` / ``prompt_lens`` / ``page_table`` / ``kv_cache`` +
    multimodal kwargs) onto that existing multimodal ``prefill_forward``.
  * KV cache allocation reuses the tt_transformers paged allocator.
  * decode is inherited unchanged (image tokens only exist during prefill).

STATUS / PRECONDITION (tt-inference-server "add a new model" Step 0):
    Upstream vLLM still has no native Janus support (tracking issues
    vllm-project/vllm #12479, #12492, #12538 were closed unimplemented). The
    Tenstorrent vLLM fork provides the missing piece as
    ``vllm.model_executor.models.janus_pro`` (``JanusProMultiModalProcessor`` /
    ``JanusProProcessingInfo`` / ``JanusProDummyInputsBuilder``). The
    registration below imports those classes; if the fork is not on a build
    that includes ``janus_pro``, registration is skipped so the module stays
    importable for the TT model-registry path.
"""

from __future__ import annotations

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.experimental.janus_pro.tt.janus_pro_e2e_model import JanusMultimodalGenerator, TtJanusProModel
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.generator import create_submeshes
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache

# Prefer the Tenstorrent-fork ``janus_pro`` processor; fall back gracefully if
# this vLLM build does not include it yet. See the module docstring.
try:
    from vllm.model_executor.models.janus_pro import (  # type: ignore
        JanusDummyInputsBuilder,
        JanusMultiModalProcessor,
        JanusProcessingInfo,
    )

    _JANUS_VLLM_PROCESSOR = (JanusMultiModalProcessor, JanusProcessingInfo, JanusDummyInputsBuilder)
except Exception:  # noqa: BLE001 - absence is expected until the vLLM fork ships janus_pro
    _JANUS_VLLM_PROCESSOR = None


class JanusForConditionalGeneration(JanusMultimodalGenerator, SupportsMultiModal):
    """vLLM bridge for Janus-Pro. Named after the HF architecture so vLLM's TT
    model registry maps ``JanusForConditionalGeneration`` -> this class."""

    # Class-level capabilities. Host greedy decode / no on-device sampler exercised
    # in bring-up (see demo/vision_demo.py), and the vision tower makes prefix
    # caching unsafe, so both are declared unsupported conservatively.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Image placeholder token id, resolved by TtJanusProModel from the HF config.
        self.IMAGE_TOKEN_ID = self.model[0].image_token_id
        # Room to generate after the prompt; matches Mllama/Mistral3 bring-up bound.
        self.max_gen_len = self.model_args[0].max_seq_len - 1

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=4096,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        # Janus uses model-specific program configs (see model_config.py); the
        # DecodersPrecision knobs the text-only bridges expose are not wired here.
        assert optimizations is None, "Custom optimizations are not supported for Janus-Pro"

        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None
        for submesh in submesh_devices:
            model_args_i = ModelArgs(
                submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                cache_hf=True,
            )
            hf_model_id = getattr(hf_config, "_name_or_path", "") or getattr(hf_config, "name_or_path", "")
            assert model_args_i.model_name.replace("-", "") in hf_model_id.replace("-", ""), (
                f"The model specified in vLLM ({hf_model_id}) does not match the model name "
                f"({model_args_i.model_name}) with model weights ({model_args_i.CKPT_DIR})."
            )
            # Janus load_state_dict() rebuilds from the reference model; load once and share.
            if state_dict is None:
                state_dict = model_args_i.load_state_dict()
            model_i = TtJanusProModel(
                args=model_args_i,
                dtype=ttnn.bfloat8_b,  # 7B decoder must fit in DRAM; vision tower stays bf16 internally
                mesh_device=submesh,
                state_dict=state_dict,
                weight_cache_path=model_args_i.weight_cache_path(ttnn.bfloat8_b),
                use_paged_kv_cache=True,
                vision_dtype=ttnn.bfloat16,
            )
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].model_cache_path

    def _extract_vision_images(self, kwargs):
        """Map vLLM's multimodal kwargs onto the per-image pixel list the Janus
        fusion path expects: a flat list of ``[1, 3, H, W]`` tensors, one per
        image in prompt order (each is encoded then coalesced onto its
        ``image_token`` placeholder block; see ``JanusMultimodalGenerator``).

        The exact kwarg names/shapes depend on the vLLM Janus processor (Step 0),
        so accept the two conventions the other TT bridges use: ``pixel_values``
        (a list, one entry per user) or ``images`` (objects exposing
        ``.pixel_values``).
        """
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is None:
            data = kwargs.get("images", None)
            if data:
                pixel_values = [getattr(im, "pixel_values", None) for im in data]
        if not pixel_values:
            return None

        images = []
        for user_pv in pixel_values:
            if user_pv is None:
                continue
            t = user_pv if torch.is_tensor(user_pv) else torch.as_tensor(user_pv)
            if t.dim() == 3:  # [3, H, W] -> [1, 3, H, W]
                t = t.unsqueeze(0)
            images += [t[i : i + 1] for i in range(t.shape[0])]
        return images or None

    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)

        # vLLM right-pads each user's row with 0s; repad with the real pad token
        # so the trailing positions embed cleanly (matches Mistral3/Qwen bridges).
        pad_token_id = self.model_args[0].tokenizer.pad_token_id or 0
        for i in range(tokens.shape[0]):
            tokens[i][prompt_lens[i] :] = pad_token_id

        vision_images = self._extract_vision_images(kwargs)
        total_lens = [int(prompt_lens[i]) + self.max_gen_len for i in range(tokens.shape[0])]

        # super() == JanusMultimodalGenerator.prefill_forward, which threads
        # pixel_values through TtJanusProModel.prepare_inputs_prefill (vision +
        # masked_scatter fusion) and otherwise runs the base text prefill.
        return super().prefill_forward(
            vision_images,  # vision_images (per-image [1,3,H,W] list, or None for text-only)
            [None] * tokens.shape[0],  # vision_masks (unused by Janus)
            tokens,
            None,  # xattn_caches (unused)
            total_lens,
            prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
        )

    def decode_forward(self, *args, **kwargs):
        # Image tokens only exist in prefill; decode is the plain text path.
        return super(JanusMultimodalGenerator, self).decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)


if _JANUS_VLLM_PROCESSOR is not None:
    JanusForConditionalGeneration = MULTIMODAL_REGISTRY.register_processor(
        _JANUS_VLLM_PROCESSOR[0],
        info=_JANUS_VLLM_PROCESSOR[1],
        dummy_inputs=_JANUS_VLLM_PROCESSOR[2],
    )(JanusForConditionalGeneration)
else:
    logger.warning(
        "vLLM has no Janus-Pro MultiModalProcessor (expected "
        "vllm.model_executor.models.janus_pro); JanusForConditionalGeneration "
        "is defined but not registered with MULTIMODAL_REGISTRY. Serving "
        "Janus-Pro over vLLM requires a vLLM build that includes janus_pro.py."
    )
