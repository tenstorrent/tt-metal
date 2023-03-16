from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from utility_functions import pad_weight, tilize_to_list, print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from libs.tt_lib.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.silu import SiLU as TtSiLU


from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc


class TtAdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings, device=None, host=None, state_dict=None, base_address="down_blocks.0.attentions.0.transformer_blocks.0"):
        super().__init__()
        self.device = device
        self.host = host

        self.torch_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.torch_emb.weight = nn.Parameter(state_dict[f"{base_address}.torch_emb.weight"])
        self.silu = TtSiLU
        weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.linear.weight"]))
        bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.linear.bias"]))
        self.linear = TtLinear(embedding_dim, embedding_dim * 2, weights, bias, self.device)
        self.torch_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        t = self.torch_emb(timestep)
        t = torch_to_tt_tensor(t, self.device)
        t = self.silu(t)
        emb = self.linear(t)
        # emb = self.linear(self.silu(self.emb(timestep)))
        emb = tt_to_torch_tensor(emb, self.host)
        scale, shift = torch.chunk(emb, 2)
        scale = torch_to_tt_tensor(scale)
        shift = torch_to_tt_tensor(shift)


        x = tt_to_torch_tensor(x)
        x = self.torch_norm(x)
        x = torch_to_tt_tensor(x)

        scale1 = ttl.tensor.fill_rm(*scale.shape(), 0, 0, scale,1, 1)

        x = ttl.tensor.mul(x, scale1)
        x = ttl.tensor.add(x, shift)
        return x
        # x = self.norm(x) * (1 + scale) + shift
        # return x



def run_adalayernorm_inference(device):
    assert False, "This is not implemented in Stable Diffusion"
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    ada_layernorm = unet.down_blocks[0].attentions[0].transformer_blocks[0].norm1

    base_address = "down_blocks.0.attentions.0.transformer_blocks.0"
    #TODO: incomplete test



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_adalayernorm_inference(device)
    ttl.device.CloseDevice(device)
