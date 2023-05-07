from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")



from typing import Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from libs.tt_lib.fused_ops.linear import Linear as TtLinear

from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TtAttentionBlock(nn.Module):
    """

    Parameters:
        channels (`int`): The number of channels in the input and output.
        num_head_channels (`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """
    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        state_dict=None,
        base_address="",
        device=None,
        host=None
    ):
        super().__init__()
        self.device = device
        self.host = host
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.rescale_output_factor = rescale_output_factor
        self._attention_op = None

        # self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)
        # self.group_norm.weight = nn.Parameter(state_dict[f"{base_address}.group_norm.weight"])
        # self.group_norm.bias = nn.Parameter(state_dict[f"{base_address}.group_norm.bias"])

        self.group_weight = state_dict[f"{base_address}.group_norm.weight"]
        self.group_bias = state_dict[f"{base_address}.group_norm.bias"]


        self.group_norm = fallback_ops.GroupNorm(num_channels=channels,
                                                num_groups=norm_num_groups,
                                                eps=eps,
                                                affine=True,
                                                weights=self.group_weight
                                                biases=self.group_bias)

        # q_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.query.weight"]))
        # q_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.query.bias"]))
        # self.query = TtLinear(channels, channels, q_weights, q_bias, device)

        q_weights = state_dict[f"{base_address}.query.weight"]
        q_bias = state_dict[f"{base_address}.query.bias"]
        self.query = make_linear(in_features=channels,
                                out_features=channels,
                                weights=q_weights,
                                bias=q_bias,
                                device=self.device)

        # k_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.key.weight"]))
        # k_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.key.bias"]))
        # self.key = TtLinear(channels, channels, k_weights, k_bias, device)

        k_weights = state_dict[f"{base_address}.key.weight"]
        k_bias = state_dict[f"{base_address}.key.bias"]
        self.key = make_linear(in_features=channels,
                                out_features=channels,
                                weights=k_weights,
                                bias=k_bias,
                                device=self.device)

        # v_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.value.weight"]))
        # v_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.value.bias"]))
        # self.value = TtLinear(channels, channels, v_weights, v_bias, device)
        v_weights = state_dict[f"{base_address}.value.weight"]
        v_bias = state_dict[f"{base_address}.value.bias"]
        self.value = make_linear(in_features=channels,
                                out_features=channels,
                                weights=v_weights,
                                bias=v_bias,
                                device=self.device)

        # proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_attn.weight"]))
        # proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_attn.bias"]))
        # self.proj_attn = TtLinear(channels, channels, proj_weights, proj_bias, device)
        proj_weights = state_dict[f"{base_address}.proj_attn.weight"]
        proj_bias = state_dict[f"{base_address}.proj_attn.bias"]
        self.proj_attn = make_linear(in_features=channels,
                                    out_features=channels,
                                    weights=proj_weights,
                                    bias=proj_bias,
                                    device=self.device)


        self._attention_op = None


    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape()
        head_size = self.num_heads
        # tensor = ttl.tensor.reshape(tensor, batch_size, seq_len, head_size, dim // head_size)
        tensor = fallback_ops.reshape(tensor, batch_size, seq_len, head_size, dim // head_size)
        tensor = ttl.tensor.permute(tensor, 0, 2, 1, 3)
        tensor = fallback_ops.reshape(tensor, 1, batch_size * head_size, seq_len, dim // head_size)
        # tensor = tt_to_torch_tensor(tensor, self.host)
        # tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        # tensor = torch_to_tt_tensor(tensor, self.device)

        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape()
        head_size = self.num_heads
        # tensor = ttl.tensor.reshape(tensor, batch_size // head_size, head_size, seq_len, dim)
        tensor = fallback_ops.reshape(tensor, batch_size // head_size, head_size, seq_len, dim)
        # tensor = tt_to_torch_tensor(tensor, self.host)
        tensor = ttl.tensor.permute(tensor, 0, 2, 1, 3)
        tensor = fallback_ops.reshape(tensor, 1, batch_size // head_size, seq_len, dim * head_size)
        # tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        # tensor = torch_to_tt_tensor(tensor, self.device)
        return tensor

    def _baddbmm(self, query_proj, key_proj, value_proj, scale):
        input1 = torch.empty(1, query_proj.shape()[0], query_proj.shape()[1], key_proj.shape()[1])
        input1 = torch_to_tt_tensor(input1, self.device)
        key_proj_T = ttl.tensor.transpose(key_proj)
        beta = 0
        alpha = scale
        _bmm = ttl.tensor.bmm(key_proj_T, query_proj)
        _scale = torch.full(scale, _bmm.shape())
        _scale = torch_to_tt_tensor(_scale, self.device)
        return ttl.tensor.mul(_scale, _bmm)
        # _bmm = ttl.tensor.add(input1, _bmm)
        # since beta=0


    def _attention_score(self, query_proj, key_proj, value_proj, scale):

        attention_scores = self._baddbmm(query_proj, key_proj, value_proj, scale)
        attention_probs = TtSoftmax(attention_scores)
        return ttl.tensor.bmm(attention_probs, value_proj)



    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape()

        # norm
        # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
        hidden_states = self.group_norm(hidden_states)
        # hidden_states = torch_to_tt_tensor(hidden_states, self.device)

        hidden_states = fallback_ops.reshape(hidden_states, 1, batch, channel, height * width)
        # hidden_states = ttl.tensor.reshape(hidden_states, 1, batch, channel, height * width)
        hidden_states = ttl.tensor.transpose_hc(hidden_states)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        attention_scores = self._attention_score(query_proj, key_proj, value_proj, scale)


        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = ttl.tensor.transpose(hidden_states)
        # hidden_states = ttl.tensor.reshape(hidden_states, batch, channel, height, width)
        hidden_states = fallback_ops.reshape(hidden_states, batch, channel, height, width)

        hidden_states = ttl.tensor.add(hidden_states, residual)

        recip = torch.full(1/self.rescale_output_factor, residual.shape())
        recip = torch_to_tt_tensor(recip, self.device)

        return ttl.tensor.mul(hidden_states, recip)


# in_channels :  512
# temb channels:  None
# eps:  1e-06
# resnet groups 32
# dropout 0.0
# time_embedding_norm default
# output scale factor:  1
# pre norm True
# attn_num_head_channels,  None

def test_run_attention_inference(device):
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()
    test = "test1"

    if test == "test1":
        in_channels = 512
        eps = 1e-06
        resnet_groups = 32
        input_shape  = [1, 512, 64, 64]
        input = torch.randn(input_shape)

    torch_out = attention(input)

    # vae_encoder = pipe.vae.encoder
    # attention = vae_encoder.mid_block.attentions[0]

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_attention = TtAttentionBlock(channels=in_channels,
                                num_head_channels=None,
                                norm_num_groups=resnet_groups,
                                eps=eps,
                                state_dict=state_dict,
                                device=device,
                                host=host,)
    #############
    tt_out = tt_attention(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))
