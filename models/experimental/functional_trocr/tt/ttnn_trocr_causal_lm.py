# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import copy

from models.experimental.functional_trocr.tt.ttnn_trocr_decoder import trocr_decoder
import tt_lib as ttl


def trocr_causal_lm(
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=None,
    config=None,
    device=None,
    parameters=None,
):
    config = copy.deepcopy(config)
    config.is_decoder = True
    config.is_encoder_decoder = False

    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    return_dict = return_dict if return_dict is not None else config.use_return_dict
    ttl.device.DumpDeviceProfiler(device)
    outputs = trocr_decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        config=config,
        device=device,
        parameters=parameters.model.decoder,
    )
    logits = outputs[0]
    logits = ttnn.linear(logits, parameters.output_projection.weight)

    return (
        logits,
        outputs[1:],
    )
