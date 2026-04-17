# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-MoE generation fixes: patch talker prepare_inputs (next_sequence_length vs past_key_values), code_predictor device/dtype, and align code_predictor do_sample with talker_do_sample during generate."""

import functools

import torch


def _patch_talkers_code_predictor_class_device_dtype() -> None:
    """Add device/dtype properties on code_predictor class for HF GenerationMixin (nested model lacks parent placeholders)."""
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
                return _orig_generate(self, *args, **kwargs)
            finally:
                if hasattr(talker, "_symbiote_code_predictor_do_sample"):
                    delattr(talker, "_symbiote_code_predictor_do_sample")
        return _orig_generate(self, *args, **kwargs)

    composite_cls.generate = generate_with_code_predictor_sample
    composite_cls._tt_symbiote_generate_wrapped = True
