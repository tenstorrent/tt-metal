# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace compatibility shims for Qwen3-Omni-MoE generation.

`transformers` GenerationMixin.prepare_inputs_for_generation gained a
``next_sequence_length`` parameter (2nd positional). Qwen3OmniMoeTalkerForConditionalGeneration
still passes ``past_key_values`` as the 2nd positional argument, so ``next_sequence_length``
is bound incorrectly and the same name can also appear in ``**kwargs``, causing::

    TypeError: prepare_inputs_for_generation() got multiple values for argument 'next_sequence_length'

This module patches the talker to call the mixin with explicit keyword arguments.

**Audio vs text:** Thinker text uses greedy or sampling from ``generate`` as configured.
The talker honors ``talker_do_sample``, but Hugging Face still calls ``code_predictor.generate``
with hard-coded ``do_sample=True`` on each decode step. That keeps **speech-code** decoding
stochastic while the outer talker loop is greedy, which often sounds like wrong or random words
in TTS even when thinker **text** is correct. We set ``talker._symbiote_code_predictor_do_sample``
from ``talker_do_sample`` for the duration of ``Qwen3OmniMoeForConditionalGeneration.generate``.

**Code predictor on TTNN:** ``Qwen3OmniMoeTalkerCodePredictorAttention`` lives under
``talker.code_predictor``, a separate ``PreTrainedModel``. A single
``register_module_replacement_dict(model.talker, ...)`` pass can fail to replace or
device-init that subtree depending on module traversal order. We call
``ensure_qwen3_omni_code_predictor_attention_ttnn`` from the composite ``generate`` wrapper
and immediately before ``code_predictor.generate`` in the patched talker prepare path so
predictor self-attention runs through TTNN when the rest of the model uses a mesh device.
"""

import functools

import torch


def _infer_mesh_device_from_symbiote_model(root: torch.nn.Module):
    """First TTNNModule under ``root`` with a non-None ``device`` (mesh), or None."""
    from models.experimental.tt_symbiote.core.module import TTNNModule

    for _name, mod in root.named_modules():
        if isinstance(mod, TTNNModule) and getattr(mod, "device", None) is not None:
            return mod.device
    return None


def _code_predictor_subtree_has_torch_attention(code_predictor) -> bool:
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerCodePredictorAttention,
    )

    for _n, mod in code_predictor.named_modules():
        if isinstance(mod, Qwen3OmniMoeTalkerCodePredictorAttention):
            return True
    return False


def ensure_qwen3_omni_code_predictor_attention_ttnn(model_or_talker, mesh_device=None) -> None:
    """Register ``TTNNQwen3Attention`` for code-predictor layers and set mesh device on that subtree.

    Pass the composite ``Qwen3OmniMoeForConditionalGeneration`` or the talker submodule.
    If ``mesh_device`` is omitted, it is inferred from existing TTNN modules (talker or full model).

    Idempotent: safe to call from ``generate`` and from ``prepare_inputs_for_generation``.
    """
    talker = getattr(model_or_talker, "talker", None)
    if talker is None:
        talker = model_or_talker
    code_predictor = getattr(talker, "code_predictor", None)
    if code_predictor is None:
        return
    if getattr(code_predictor, "_tt_symbiote_attn_ttnn_ensured", False):
        return

    device = mesh_device
    if device is None:
        device = _infer_mesh_device_from_symbiote_model(model_or_talker)
    if device is None and talker is not model_or_talker:
        device = _infer_mesh_device_from_symbiote_model(talker)
    if device is None:
        return

    if not _code_predictor_subtree_has_torch_attention(code_predictor):
        code_predictor._tt_symbiote_attn_ttnn_ensured = True
        return

    from models.experimental.tt_symbiote.modules.attention import TTNNQwen3Attention
    from models.experimental.tt_symbiote.utils.device_management import set_device
    from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerCodePredictorAttention,
    )

    register_module_replacement_dict(
        code_predictor,
        {Qwen3OmniMoeTalkerCodePredictorAttention: TTNNQwen3Attention},
        model_config=None,
    )
    # Subtree only: talker/code_predictor forward is already wrapped by a full-model set_device pass.
    set_device(code_predictor, device, register_forward_hook=False)
    code_predictor._tt_symbiote_attn_ttnn_ensured = True


def _patch_talkers_code_predictor_class_device_dtype() -> None:
    """Ensure ``talker.code_predictor`` exposes ``.device`` / ``.dtype`` for HF ``GenerationMixin``.

    ``code_predictor.generate()`` calls ``self.device`` (e.g. when building default ``input_ids``). The nested
    ``Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration`` is not the same module as ``talker``; tests
    that only patched ``thinker``/``talker``/``code2wav`` left ``code_predictor`` without the symbiote placeholders,
    which can surface as ``AttributeError: ... has no attribute 'device'``.
    """
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as omni_mod

    cls = omni_mod.Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
    if getattr(cls, "_tt_symbiote_device_patched", False):
        return
    _cpu = torch.device("cpu")
    _dtype = torch.bfloat16
    dev_attr = getattr(cls, "device", None)
    if dev_attr is None or isinstance(dev_attr, property):
        cls.device = property(lambda self, d=_cpu: d)
    dtype_attr = getattr(cls, "dtype", None)
    if dtype_attr is None or isinstance(dtype_attr, property):
        cls.dtype = property(lambda self, d=_dtype: d)
    cls._tt_symbiote_device_patched = True


def apply_qwen3_omni_talker_prepare_inputs_fix() -> None:
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as omni_mod

    _patch_talkers_code_predictor_class_device_dtype()

    talker_cls = omni_mod.Qwen3OmniMoeTalkerForConditionalGeneration
    if not getattr(talker_cls, "_tt_symbiote_prepare_inputs_patched", False):

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            is_first_iteration=False,
            **kwargs,
        ):
            hidden_states = kwargs.pop("hidden_states", None)
            next_sequence_length = kwargs.pop("next_sequence_length", None)
            inputs = super(talker_cls, self).prepare_inputs_for_generation(
                input_ids,
                next_sequence_length=next_sequence_length,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                is_first_iteration=is_first_iteration,
                **kwargs,
            )

            inputs["position_ids"] = None

            if not is_first_iteration and kwargs.get("use_cache", True):
                input_ids = input_ids[:, -1:]
                generation_step = kwargs.get("generation_step")
                trailing_text_hidden = kwargs.get("trailing_text_hidden")
                tts_pad_embed = kwargs.get("tts_pad_embed")
                last_id_hidden = self.get_input_embeddings()(input_ids)

                cp_do_sample = getattr(self, "_symbiote_code_predictor_do_sample", None)
                if cp_do_sample is None:
                    cp_do_sample = True

                past_hidden = hidden_states[0][-1][:, -1:].to(device=last_id_hidden.device, dtype=last_id_hidden.dtype)
                _cp_kw = dict(
                    inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                    max_new_tokens=self.config.num_code_groups - 1,
                    do_sample=cp_do_sample,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                if cp_do_sample:
                    _cp_kw["top_k"] = 50
                    _cp_kw["top_p"] = 0.8
                ensure_qwen3_omni_code_predictor_attention_ttnn(self)
                predictor_result = self.code_predictor.generate(**_cp_kw)
                residual_codes = torch.cat((input_ids, predictor_result.sequences.to(input_ids.device)), dim=-1)

                mid_residual_hiddens = [hid[0].to(last_id_hidden.device) for hid in predictor_result.hidden_states[1:]]
                last_residual_hidden = self.code_predictor.get_input_embeddings()[-1](
                    predictor_result.sequences[..., -1:]
                ).to(last_id_hidden.device)
                codec_hiddens = torch.cat(
                    [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden],
                    dim=1,
                )
                inputs_embeds = codec_hiddens.sum(1, keepdim=True)

                if generation_step < trailing_text_hidden.shape[1]:
                    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1).to(
                        inputs_embeds.device
                    )
                else:
                    inputs_embeds = inputs_embeds + tts_pad_embed.to(inputs_embeds.device)
                inputs["inputs_embeds"] = inputs_embeds
                inputs["residual_codes"] = residual_codes
            return inputs

        talker_cls.prepare_inputs_for_generation = prepare_inputs_for_generation
        talker_cls._tt_symbiote_prepare_inputs_patched = True

    composite_cls = omni_mod.Qwen3OmniMoeForConditionalGeneration
    if getattr(composite_cls, "_tt_symbiote_generate_wrapped", False):
        return

    _orig_generate = composite_cls.generate

    @functools.wraps(_orig_generate)
    def generate_with_code_predictor_sample(self, *args, **kwargs):
        talker_do_sample = kwargs.get("talker_do_sample", True)
        talker = getattr(self, "talker", None)
        ret_audio = kwargs.get("return_audio")
        if ret_audio is None:
            ret_audio = getattr(self, "has_talker", False)
        if talker is not None and ret_audio:
            talker._symbiote_code_predictor_do_sample = bool(talker_do_sample)
            try:
                ensure_qwen3_omni_code_predictor_attention_ttnn(self)
                return _orig_generate(self, *args, **kwargs)
            finally:
                if hasattr(talker, "_symbiote_code_predictor_do_sample"):
                    delattr(talker, "_symbiote_code_predictor_do_sample")
        return _orig_generate(self, *args, **kwargs)

    composite_cls.generate = generate_with_code_predictor_sample
    composite_cls._tt_symbiote_generate_wrapped = True
