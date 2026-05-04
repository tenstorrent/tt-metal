# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-MoE generation fixes: patch talker prepare_inputs (next_sequence_length vs past_key_values), code_predictor device/dtype, and align code_predictor do_sample with talker_do_sample during generate.

Talker cached decode expects ``hidden_states``, ``generation_step``, ``trailing_text_hidden``, and ``tts_pad_embed``. Pop them **before** ``super().prepare_inputs_for_generation`` so GenerationMixin cannot strip them from ``kwargs``. When ``hidden_states`` is omitted (e.g. transformers>=4.57), fall back to last ``inputs_embeds``; see inline comments.
"""

import functools

import torch


def _plain_torch_tensor(x):
    """Unwrap TT/tensor subclasses so ``shape`` / slicing match HF."""
    t = x
    for _ in range(8):
        if isinstance(t, torch.Tensor):
            return t
        next_t = getattr(t, "tensor", None)
        if next_t is None:
            next_t = getattr(t, "_tensor", None)
        if next_t is None and hasattr(t, "data"):
            next_t = t.data
        if next_t is None or next_t is t:
            break
        t = next_t
    raise TypeError(f"Expected a torch.Tensor or wrapper with .tensor; got {type(x).__name__}")


def _last_token_embeddings_b1h(embeds, *, device, dtype):
    """Last sequence position as ``[batch, 1, hidden]`` (code predictor prefill needs ``seq_len >= 2`` after concat)."""
    t = _plain_torch_tensor(embeds)
    if t.dim() == 3:
        if t.shape[1] == 0:
            raise ValueError(f"inputs_embeds has empty sequence dim: shape {tuple(t.shape)}")
        out = t[:, -1:, :].contiguous()
    elif t.dim() == 2:
        if t.shape[0] == 0:
            raise ValueError(f"inputs_embeds has empty sequence dim: shape {tuple(t.shape)}")
        # HF sometimes passes ``[seq_len, hidden_size]`` without a batch dimension.
        out = t[-1:, :].unsqueeze(0).contiguous()
    else:
        raise ValueError(f"Talker inputs_embeds fallback expects rank 2 or 3, got shape {tuple(t.shape)}")
    return out.to(device=device, dtype=dtype)


def _hidden_states_last_b1h(hidden_states, *, device, dtype):
    """Thinker last-layer hidden state for last token: ``[batch, 1, hidden]``."""
    # hidden_states: tuple of per-layer outputs; [-1] last layer; typically [B, S, H].
    layer_h = _plain_torch_tensor(hidden_states[0][-1])
    if layer_h.dim() == 3:
        if layer_h.shape[1] == 0:
            return None
        out = layer_h[:, -1:, :].contiguous()
    elif layer_h.dim() == 2:
        if layer_h.shape[0] == 0:
            return None
        out = layer_h[-1:, :].unsqueeze(0).contiguous()
    else:
        raise ValueError(f"Unexpected thinker hidden rank {layer_h.dim()}, shape {tuple(layer_h.shape)}")
    return out.to(device=device, dtype=dtype)


def _codec_prefill_embeds(past_b1h, lh_b1h):
    """``[B,2,H]`` for code predictor prefill; duplicate current token if past is missing (thin cache)."""
    p = _plain_torch_tensor(past_b1h)
    l = _plain_torch_tensor(lh_b1h)
    if p.dim() == 2:
        p = p.unsqueeze(1)
    if l.dim() == 2:
        l = l.unsqueeze(1)
    if p.shape[1] == 0:
        p = l.clone()
    if l.shape[1] == 0:
        raise ValueError("Talker decode: last token embedding has empty sequence dimension.")
    if p.shape[1] > 1:
        p = p[:, -1:, :]
    if l.shape[1] > 1:
        l = l[:, -1:, :]
    combined = torch.cat((p, l), dim=1)
    if combined.shape[1] < 2:
        combined = torch.cat((l, l), dim=1)
    return combined


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
            generation_step = kwargs.pop("generation_step", None)
            trailing_text_hidden = kwargs.pop("trailing_text_hidden", None)
            tts_pad_embed = kwargs.pop("tts_pad_embed", None)
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
                last_id_hidden = self.get_input_embeddings()(input_ids)

                cp_do_sample = getattr(self, "_symbiote_code_predictor_do_sample", None)
                if cp_do_sample is None:
                    cp_do_sample = True

                if generation_step is None:
                    if cache_position is not None and isinstance(cache_position, torch.Tensor):
                        generation_step = int(cache_position.reshape(-1)[-1].item())
                    else:
                        generation_step = 0

                device, dtype = last_id_hidden.device, last_id_hidden.dtype
                if hidden_states is not None:
                    past_hidden = _hidden_states_last_b1h(hidden_states, device=device, dtype=dtype)
                    if past_hidden is None:
                        embeds_src = inputs.get("inputs_embeds") or inputs_embeds
                        if embeds_src is None:
                            raise ValueError("Thinker hidden_states had empty sequence; need inputs_embeds fallback.")
                        past_hidden = _last_token_embeddings_b1h(embeds_src, device=device, dtype=dtype)
                else:
                    embeds_src = inputs.get("inputs_embeds")
                    if embeds_src is None:
                        embeds_src = inputs_embeds
                    if embeds_src is None:
                        raise ValueError(
                            "Talker cached decode needs ``hidden_states`` from prior outputs or "
                            "``inputs_embeds`` / ``inputs['inputs_embeds']`` after prepare_inputs; got none."
                        )
                    past_hidden = _last_token_embeddings_b1h(embeds_src, device=device, dtype=dtype)

                codec_prefill = _codec_prefill_embeds(past_hidden, last_id_hidden)
                _cp_kw = dict(
                    inputs_embeds=codec_prefill,
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

                if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1]:
                    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1).to(
                        inputs_embeds.device
                    )
                elif tts_pad_embed is not None:
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
