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
"""

import torch


def apply_qwen3_omni_talker_prepare_inputs_fix() -> None:
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as omni_mod

    cls = omni_mod.Qwen3OmniMoeTalkerForConditionalGeneration
    if getattr(cls, "_tt_symbiote_prepare_inputs_patched", False):
        return

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
        inputs = super(cls, self).prepare_inputs_for_generation(
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

            past_hidden = hidden_states[0][-1][:, -1:].to(last_id_hidden.device)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=0.8,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
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

    cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    cls._tt_symbiote_prepare_inputs_patched = True
