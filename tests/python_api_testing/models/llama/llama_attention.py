import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from loguru import logger
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from libs import tt_lib as ttl
from typing import List, Optional, Tuple, Union

from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TTsoftmax
from python_api_testing.models.llama.llama_utils import *
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def shape_tt(states, batch_size, seq_len, n_heads, head_dim):
    tt_out = ttl.tensor.reshape(states, batch_size, seq_len, n_heads, head_dim)
    tt_out = ttl.tensor.transpose_hc(tt_out)

    return tt_out


def shape_pt(tensor: torch.Tensor, seq_len: int, bsz: int):
    num_heads = 32
    head_dim = 512
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def test_lamma_shape(device):

    batch_size = 1
    n_heads = 32
    seq_len = 128
    head_dim = 512

    torch.manual_seed(0)
    test_input = (torch.rand(1, 32, 128, 512) * 2) - 1
    test_input = test_input.to(torch.float16)

    pt_out = shape_pt(test_input, seq_len, batch_size)

    test = torch2tt_tensor(test_input, device)
    tt_out = shape_tt(test, batch_size, seq_len, n_heads, head_dim)
    tt_out = tt2torch_tensor(tt_out)

    if np.allclose(pt_out.detach().numpy(), tt_out, 1e-4, 0.17):
        logger.info("llama_shape test Passed!")
    else:
        logger.warning("llama_shape test Failed!")


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TtLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        device,
        state_dict,
        layer_num,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.device = device

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        print('a =====================================')
        self.state_dict = state_dict
        self.q_weights = tilize_to_list(pad_weight(self.state_dict[f"model.layers.{layer_num}.self_attn.q_proj.weight"]))
        self.k_weights = tilize_to_list(pad_weight(self.state_dict[f"model.layers.{layer_num}.self_attn.k_proj.weight"]))
        self.v_weights = tilize_to_list(pad_weight(self.state_dict[f"model.layers.{layer_num}.self_attn.v_proj.weight"]))
        self.o_weights = tilize_to_list(pad_weight(self.state_dict[f"model.layers.{layer_num}.self_attn.o_proj.weight"]))
        print('b =====================================')
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q_proj = TtLinear(self.hidden_size, self.num_heads * self.head_dim, weight=self.q_weights, bias=None, device=self.device)
        self.k_proj = TtLinear(self.hidden_size, self.num_heads * self.head_dim, weight=self.k_weights, bias=None, device=self.device)
        self.v_proj = TtLinear(self.hidden_size, self.num_heads * self.head_dim, weight=self.v_weights, bias=None, device=self.device)
        self.o_proj = TtLinear(self.num_heads * self.head_dim, self.hidden_size, weight=self.o_weights, bias=None, device=self.device)
        print('c =====================================')
        # self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        print('d =====================================')

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # return shape_tt(tensor, bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        # self,
        # hidden_states: torch.Tensor,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz = hidden_states.shape()[0]
        q_len = hidden_states.shape()[2]

        # ======================================================
        # query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.q_proj(hidden_states)
        query_states = shape_tt(query, bsz, q_len, self.num_heads, self.head_dim)
        print('1 =====================================')

        # key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states)
        key_states = shape_tt(key, bsz, q_len, self.num_heads, self.head_dim)
        print('2 =====================================')
        # value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states)
        value_states = shape_tt(value, bsz, q_len, self.num_heads, self.head_dim)
        print('3 =====================================')
        # return all states to pytorch =============================
        query_states = tt2torch_tensor(query_states)
        key_states = tt2torch_tensor(key_states)
        value_states = tt2torch_tensor(value_states)
        hidden_states = tt2torch_tensor(hidden_states)
        # return all states to pytorch =============================

        print('4 =====================================')
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # offset = 0
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        print('5 =====================================')
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        print('6 =====================================')
        # TT implementation for:
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        key_states_tt = torch2tt_tensor(key_states, self.device)
        query_states_tt = torch2tt_tensor(query_states, self.device)
        print('7 =====================================')
        key_states_tt_trans = ttl.tensor.transpose(key_states_tt)
        mul = ttl.tensor.bmm(query_states_tt, key_states_tt_trans)
        # create constant tensor
        const_tensor_tt = tt_const_tensor(math.sqrt(self.head_dim), mul.shape(), self.device)
        # divison
        recip = ttl.tensor.recip(const_tensor_tt)
        div = ttl.tensor.mul(mul, recip)
        print('8 =====================================')
        # return to PyTorch
        attn_weights = tt2torch_tensor(div)
        sys.exit(0)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # TT implementation not finished!
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # TT implementation for:
        # PyTorch: upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_weights = torch2tt_tensor(attn_weights, self.device)
        value_states = torch2tt_tensor(value_states, self.device)

        attn_weights = TTsoftmax(attn_weights)
        attn_output = ttl.tensor.bmm(attn_weights, value_states)

        # return to PyTorch
        attn_output = tt2torch_tensor(attn_output)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # return to TT tensor
        attn_output = torch2tt_tensor(attn_output, self.device)
        attn_output = ttl.tensor.transpose_hc(attn_output)
        attn_output = ttl.tensor.reshape(attn_output, bsz, 1, q_len, self.hidden_size)

        # TT call for PyTorch: attn_output = self.o_proj(attn_output)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.model.layers[layer_num].self_attn

        # Disable dropout
        self.attention.eval()

    def forward(self, x):
        result = self.attention(x)[0]
        return result


def run_LlamaAttention_inference():

    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf") # , torch_dtype=torch.float32
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ========================================================================
    torch.manual_seed(0)
    # batch size is equal to 32
    attention_input = (torch.rand(32, 32, 4096) * 2) - 1
    layer_num = 0
    print("Start pass!")
    # PyTorch output =======================================================================
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    # pytorch_out = pytorch_LlamaAttention_model(attention_input)
    # sys.exit(0)
    # TT hardware execution =================================================================
    tt_attention_input = attention_input.unsqueeze(1)
    tt_attention_input = torch2tt_tensor(tt_attention_input, device)
    print("First pass!")
    # get TT Attention module
    tt_LlamaAttention_model = TtLlamaAttention(
        device,
        state_dict,
        layer_num,
        configuration.hidden_size,
        configuration.num_attention_heads,
        2048
        # configuration.max_position_embeddings
    )
    print("Second pass!")
    tt_out, attn_weights, past_key_value = tt_LlamaAttention_model(tt_attention_input)
    print("Third  pass!")
    tt_out = tt2torch_tensor(tt_out).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_out))
    print(comp_pcc(pytorch_out, tt_out))

    pcc_test = comp_pcc(pytorch_out, tt_out1, 0.98)

    assert pcc_test, "PCC value is lower than 0.98"


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    # test_lamma_shape(device)
    run_LlamaAttention_inference()
    ttl.device.CloseDevice(device)
