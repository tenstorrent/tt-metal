from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor, torch_to_tt_tensor_rm
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from python_api_testing.models.stable_diffusion.mini_ops import Linear

from python_api_testing.models.stable_diffusion.utils import make_linear

class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=1280,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-5,
        non_linearity="silu",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        state_dict=None,
        base_address= None,
        host = None,
        device = None
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True # this is part of the original code
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.device = device
        self.host = host

        if groups_out is None:
            groups_out = groups

        print(base_address)
        norm1_weights = state_dict[f"{base_address}.norm1.weight"]
        norm1_bias = state_dict[f"{base_address}.norm1.bias"]
        self.norm1 = fallback_ops.GroupNorm(norm1_weights, norm1_bias, num_groups=groups, num_channels=self.in_channels, eps=eps, affine=True)


        conv1_weights = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = state_dict[f"{base_address}.conv1.bias"]
        self.conv1 = fallback_ops.Conv2d(conv1_weights, conv1_bias, self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            print('time_emb_proj_out_channels',time_emb_proj_out_channels)
            print('temb_channels',temb_channels)
            time_emb_proj_weights = state_dict[f"{base_address}.time_emb_proj.weight"]
            time_emb_proj_bias = state_dict[f"{base_address}.time_emb_proj.bias"]
            self.time_emb_proj = make_linear(in_features=temb_channels,
                                            out_features=time_emb_proj_out_channels,
                                            weights=time_emb_proj_weights,
                                            bias=time_emb_proj_bias,
                                            device=self.device)
            # self.time_emb_proj = Linear(temb_channels, time_emb_proj_out_channels, time_emb_proj_weights, time_emb_proj_bias)
        else:
            self.time_emb_proj = None

        norm2_weights = state_dict[f"{base_address}.norm2.weight"]
        norm2_bias = state_dict[f"{base_address}.norm2.bias"]


        self.norm2 = fallback_ops.GroupNorm(norm2_weights, norm2_bias, num_groups=groups, num_channels=self.in_channels, eps=eps, affine=True)

        conv2_weights = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = state_dict[f"{base_address}.conv2.bias"]

        self.conv2 = fallback_ops.Conv2d(conv2_weights, conv2_bias, self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


        if non_linearity == "swish":
            self.nonlinearity = TtSiLU
        elif non_linearity == "mish":
            assert False, "Mish is not implemented!"
            # self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = TtSiLU

        self.upsample = self.downsample = None
        if self.up:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            # else:
            #     self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            # else:
            #     self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")


        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            conv_shortcut_weights = state_dict[f"{base_address}.conv_shortcut.weight"]
            conv_shortcut_bias = state_dict[f"{base_address}.conv_shortcut.bias"]
            self.conv_shortcut = fallback_ops.Conv2d(conv_shortcut_weights, conv_shortcut_bias, self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def  forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            assert False, "we do not support upsample in resnet"
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            # if hidden_states.shape[0] >= 64:
            #     input_tensor = input_tensor.contiguous()
            #     hidden_states = hidden_states.contiguous()
            # input_tensor = self.upsample(input_tensor)
            # hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            assert False, "we do not support downsample in resnet"
            # input_tensor = self.downsample(input_tensor)
            # hidden_states = self.downsample(hidden_states)

        # hidden_states = tt_to_torch_tensor(hidden_states, self.host)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            # assert False, "not tested since we dont have tests for it yet"
            # print('temb size', temb.shape)
            # temb = torch_to_tt_tensor(temb, device) # to refactor
            temb = self.nonlinearity(temb)

            temb = self.time_emb_proj(temb)
            temb = fallback_ops.reshape(temb, temb.shape()[2], temb.shape()[3], 1, 1)
            # [:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = ttl.tensor.bcast(hidden_states, temb, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)
            # hidden_states = ttl.tensor.add(hidden_states, temb)

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            assert False, "this is support but not tested!"
            temb = tt_to_torch_tensor(temb, self.host)
            scale, shift = torch.chunk(temb, 2, dim=1)
            temb = torch_to_tt_tensor(temb, self.device)
            shift = torch_to_tt_tensor(shift, self.device)
            scale = torch_to_tt_tensor(scale, self.device)

            ones = torch.ones(scale.shape)
            ones = torch_to_tt_tensor(ones, self.device)

            scale = ttl.tensor.add(ones, scale)
            hidden_states = ttl.tensor.mul(hidden_states, scale)
            hidden_states = ttl.tensor.add(hidden_states, shift)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)


        # create a tensor of size output_scale_factor

        output_sc_recip = 1 / self.output_scale_factor
        output_sc_recip = fallback_ops.full(input_tensor.shape(), output_sc_recip)

        output_tensor = ttl.tensor.add(input_tensor, hidden_states)
        output_tensor = ttl.tensor.mul(output_tensor, output_sc_recip)

        return output_tensor



class TorchResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="silu",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        base_address = None,
        state_dict = None
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm1.weights = nn.Parameter(state_dict[f"{base_address}.norm1.weight"])
        self.norm1.bias = nn.Parameter(state_dict[f"{base_address}.norm1.bias"])

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1.weights = nn.Parameter(state_dict[f"{base_address}.conv1.weight"])
        self.conv1.bias = nn.Parameter(state_dict[f"{base_address}.conv1.bias"])

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
            self.time_emb_proj.weights = nn.Parameter(state_dict[f"{base_address}.time_emb_proj.weight"])
            self.time_emb_proj.bias = nn.Parameter(state_dict[f"{base_address}.time_emb_proj.bias"])
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.norm2.weights = nn.Parameter(state_dict[f"{base_address}.norm2.weight"])
        self.norm2.bias = nn.Parameter(state_dict[f"{base_address}.norm2.bias"])
        # self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        #elif non_linearity == "mish":
            #self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        # if self.up:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        #     else:
        #         self.upsample = Upsample2D(in_channels, use_conv=False)
        # elif self.down:
        #     if kernel == "fir":
        #         fir_kernel = (1, 3, 3, 1)
        #         self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
        #     elif kernel == "sde_vp":
        #         self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
        #     else:
        #         self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.conv_shortcut.weights = nn.Parameter(state_dict[f"{base_address}.conv_shortcut.weight"])
            self.conv_shortcut.bias = nn.Parameter(state_dict[f"{base_address}.conv_shortcut.bias"])

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            print('unet temb shape:',temb.shape)
            print('unet time emb linear weight shape:',self.time_emb_proj.weight.shape)
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            print('unet hidden state shape:', hidden_states.shape)
            print('unet temb shape:', temb.shape)
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            print('shortcut triggered!')
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


def test_run_resnet_inference():
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test1"
    if test == "test1":
        unet_upblock = pipe.unet.up_blocks[2]
        resnet = unet_upblock.resnets[2]
        base_address="up_blocks.2.resnets.2"
        in_channels = resnet.conv1.in_channels
        out_channels = resnet.conv2.in_channels
        temb_channels = 512
        eps = 1e-05
        resnet_groups = 32
        torch_resnet = TorchResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=1280,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="silu",
            time_embedding_norm="default",
            kernel=None,
            output_scale_factor=1.0,
            use_in_shortcut=None,
            up=False,
            down=False,
            base_address=base_address,
            state_dict = state_dict)

        input_shape  = [1, in_channels, 32, 32]
        input = torch.randn(input_shape, dtype=torch.float32)

        temb_shape  = [out_channels, out_channels]
        temb = torch.randn(temb_shape, dtype=torch.float32)

    if test == "test2":
        pass

    unet_out = resnet(input, None)
    torch_resnet_out = torch_resnet(input, None)

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_resnet = TtResnetBlock2D(in_channels=in_channels,
                            out_channels=out_channels,
                            temb_channels=temb_channels,
                            groups=resnet_groups,
                            state_dict=state_dict,
                            base_address=base_address,
                            host=host,
                            device=device)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)

    tt_out = tt_resnet(tt_input, None)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print('torch resnet out:', torch_resnet_out[0,0,0,:12])

    print('tt out:', tt_out[0,0,0,:12])

    print('unet out:', unet_out[0,0,0,:12])

    print('unet vs torch')
    print(comp_allclose_and_pcc(unet_out, torch_resnet_out))

    print('unet vs tt')
    print(comp_allclose_and_pcc(unet_out, tt_out))

    print('torch vs tt')
    print(comp_allclose_and_pcc(torch_resnet_out, tt_out))

test_run_resnet_inference()

# if __name__ == "__main__":
#     # Initialize the device
#     device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
#     ttl.device.InitializeDevice(device)
#     host = ttl.device.GetHost()
#     run_resnet_inference(host, device)
#     ttl.device.CloseDevice(device)
