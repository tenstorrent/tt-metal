# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import tt_lib
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear


class TtFeedForward(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.device = device
        self.w1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w1.weight"], self.device, put_on_device=False
        )
        self.w1 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w1_weights,
            device=self.device,
        )
        self.w2_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w2.weight"], self.device, put_on_device=False
        )
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
            device=self.device,
        )
        self.w3_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w3.weight"], self.device, put_on_device=False
        )
        self.w3 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w3_weights,
            device=self.device,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        silu_out = tt_lib.tensor.silu(self.w1(x))
        x = tt_lib.tensor.mul(silu_out, self.w3(x))
        return self.w2(x)


class TtFeedForwardSiluFolded(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.device = device
        self.w1_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w1.weight"], self.device, put_on_device=False
        )
        self.w1_weights = self.w1_weights.pad_to_tile(0).to(tt_lib.tensor.Layout.TILE).to(self.device)
        self.w1_weights = tt_lib.tensor.transpose(self.w1_weights, 2, 3)

        self.w2_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w2.weight"], self.device, put_on_device=False
        )
        self.w2 = TtLinear(
            args.hidden_dim,
            args.dim,
            self.w2_weights,
            device=self.device,
        )

        self.w3_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}w3.weight"], self.device, put_on_device=False
        )
        self.w3_weights = self.w3_weights.pad_to_tile(0).to(tt_lib.tensor.Layout.TILE).to(self.device)
        self.w3 = TtLinear(
            args.dim,
            args.hidden_dim,
            self.w3_weights,
            device=self.device,
        )

    def w1_silu(self, activation):
        interleaved_mem_config = tt_lib.tensor.MemoryConfig(
            memory_layout=tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=tt_lib.tensor.BufferType.DRAM,
        )

        in1 = self.w1_weights
        in0 = activation

        in1_t = in1
        in0_t = in0.cpu().pad_to_tile(0).to(tt_lib.tensor.Layout.TILE).to(self.device)

        program_config = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=tt_lib.tensor.FusibleActivation.SILU,
            mcast_in0=True,
        )
        output_t = tt_lib.operations.primary.matmul_1d(
            in0_t,
            in1_t,
            program_config=program_config,
            output_mem_config=interleaved_mem_config,
            output_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        return output_t

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        silu_out = self.w1_silu(x)
        w3_tiled = (
            self.w3(x)
            .cpu()
            .to(tt_lib.tensor.Layout.ROW_MAJOR)
            .pad_to_tile(0)
            .to(tt_lib.tensor.Layout.TILE)
            .to(self.device)
        )
        interim = tt_lib.tensor.mul(silu_out, w3_tiled)
        result = (
            self.w2(interim)
            .cpu()
            .to(tt_lib.tensor.Layout.ROW_MAJOR)
            .unpad_from_tile((1, 1, 11, 4096))
            .to(tt_lib.tensor.Layout.ROW_MAJOR)
            .to(self.device)
        )
        return result
