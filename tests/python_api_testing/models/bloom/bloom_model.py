from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.baddbmm
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp
import python_api_testing.models.bloom.bloom_block as bloom_block

from fused_ops.linear import Linear as TtLinear

from fused_ops.layernorm import Layernorm as TtLayernorm

from fused_ops.softmax import softmax as TtSoftmax
from transformers import BloomForCausalLM

from typing import Optional, Tuple, Union

def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)



class TtBloomModel(torch.nn.Module):
    def __init__(self, device, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = n_head
        self.n_layer = num_hidden_layers

        state_dict = hugging_bloom_reference_model.state_dict()
        # Embedding + LN Embedding
        self.word_embeddings = torch.nn.Embedding(vocab_size, embed_dim)

        word_embeddings_layernorm_bias = bloom_utils.tt_load_layer_weights("transformer.word_embeddings_layernorm.bias", state_dict)
        word_embeddings_layernorm_weight = bloom_utils.tt_load_layer_weights("transformer.word_embeddings_layernorm.weight", state_dict)

        self.word_embeddings_layernorm = TtLayernorm(word_embeddings_layernorm_weight, word_embeddings_layernorm_bias, layer_norm_epsilon, hidden_size, hidden_size, device, 1)

        # Transformer blocks
        self.h = torch.nn.ModuleList([
        bloom_block.TtBloomBlock(device, "transformer.h", 0, hugging_bloom_reference_model, hidden_size, self.num_heads, layer_norm_epsilon),
        bloom_block.TtBloomBlock(device, "transformer.h", 1, hugging_bloom_reference_model, hidden_size, self.num_heads, layer_norm_epsilon)]
        )

        ln_f_bias = bloom_utils.tt_load_layer_weights("transformer.ln_f.bias", state_dict)
        ln_f_weight = bloom_utils.tt_load_layer_weights("transformer.ln_f.weight", state_dict)

        # Final Layer Norm

        self.ln_f = TtLayernorm(ln_f_weight, ln_f_bias, layer_norm_epsilon, hidden_size, hidden_size, device, 1)


        # Initialize weights and apply final processing

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        device,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N

        #head_mask = self.get_head_mask(head_mask, self.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        tt_inputs_embeds = bloom_utils.torch2tt_tensor(inputs_embeds, device)

        tt_hidden_states = self.word_embeddings_layernorm(tt_inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length

        hidden_states = bloom_utils.tt2torch_tensor(tt_hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        past_key_values_length = 0
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)


        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        print(hidden_states.shape)
        print(causal_mask.shape)
        print(alibi.shape)

        outputs = self.h[0](device, hidden_states=hidden_states,attention_mask=causal_mask,head_mask=None,use_cache=False, alibi=alibi)

        #outputs = self.h[1](device, hidden_states=outputs,attention_mask=causal_mask,head_mask=None,use_cache=False, alibi=alibi)

        hidden_states = outputs

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        print('HIDDEN--------')
        print(hidden_states.shape())


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



class BloomModel(torch.nn.Module):

    def __init__(self, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = n_head
        self.n_layer = num_hidden_layers

        state_dict = hugging_bloom_reference_model.state_dict()
        # Embedding + LN Embedding
        self.word_embeddings = torch.nn.Embedding(vocab_size, embed_dim)


        self.word_embeddings_layernorm = torch.nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)

        self.word_embeddings_layernorm.bias=bloom_utils.pt_load_layer_weights("transformer.word_embeddings_layernorm.bias", state_dict)
        self.word_embeddings_layernorm.weight = bloom_utils.pt_load_layer_weights("transformer.word_embeddings_layernorm.weight", state_dict)

        # Transformer blocks
        self.h = torch.nn.ModuleList([
        hugging_bloom_reference_model.transformer.h[0],
        hugging_bloom_reference_model.transformer.h[1]
        ])

        # Final Layer Norm
        self.ln_f = torch.nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)

        self.ln_f.bias = bloom_utils.pt_load_layer_weights("transformer.ln_f.bias", state_dict)
        self.ln_f.weight = bloom_utils.pt_load_layer_weights("transformer.ln_f.weight", state_dict)

        # Initialize weights and apply final processing

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N

        #head_mask = self.get_head_mask(head_mask, self.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)


        past_key_values_length = 0
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)


        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        outputs = self.h[0](hidden_states=hidden_states,attention_mask=causal_mask,head_mask=None,use_cache=False, alibi=alibi)
        #outputs = self.h[1](hidden_states=outputs,attention_mask=causal_mask,head_mask=None,use_cache=False, alibi=alibi)

        hidden_states = outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        print('HIDDEN--------')
        print(hidden_states.shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def run_bloom_model_inference(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    print(hugging_bloom_reference_model.state_dict())


    hidden_size = hugging_bloom_reference_model.config.hidden_size # 1024
    n_head = hugging_bloom_reference_model.config.n_head
    vocab_size = hugging_bloom_reference_model.config.vocab_size
    layer_norm_epsilon = hugging_bloom_reference_model.config.layer_norm_epsilon
    num_hidden_layers = hugging_bloom_reference_model.config.num_hidden_layers


    tt_bloom_model = TtBloomModel(device, hugging_bloom_reference_model, hidden_size, n_head, vocab_size, hidden_size, layer_norm_epsilon, num_hidden_layers)
    pt_bloom_model = BloomModel(hugging_bloom_reference_model, hidden_size, n_head, vocab_size, hidden_size, layer_norm_epsilon, num_hidden_layers)
    #pt_bloom_model = BloomModel(hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)

    # Prepare input
    torch.manual_seed(0)

    input_ids = torch.randint(0, 100, (1, 64))

    pt_out = pt_bloom_model.forward(input_ids)

    print("PT---------------")
    print(pt_out[0].shape)
    #print(pt_out[1].shape)

    print("PT finished")

    tt_out = tt_bloom_model.forward(device, input_ids)

    print("TT finished")

    print("TT---------------")
    print(tt_out[0].shape())

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out[0])
    tt_out_converted = tt_out_converted.squeeze(0)
    tt_out_converted = tt_out_converted.squeeze(0)

    print(tt_out_converted.shape)

    print(comp_allclose(pt_out[0], tt_out_converted))
    #print(comp_pcc(pt_out, tt_out_converted))

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_model_inference(device)
    ttm.device.CloseDevice(device)
