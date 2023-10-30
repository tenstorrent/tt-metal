# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import tt_lib
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.mistral.mistral_helper_funcs import Linear as TtLinear


class TtClassicFeedForward(nn.Module):
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
        x = tt_lib.tensor.mul(self.w1(x), self.w3(x))
        x = tt_lib.tensor.silu(x)
        return self.w2(x)


class TtFusedFeedForward(nn.Module):
    """
    S = matmul( concat(w1,w3,dim=1), X )
    T0 = silu( S[:,0,:,:] )
    T1 = S[:,1,:,:]
    U = mul( T0, T1 )
    V = matmul( U, W2 )
    """

    def __init__(
        self,
        args: TtModelArgs,
        base_address=None,
        device=None,
        state_dict=None,
    ):
        super().__init__()
        self.device = device
        self.fused_w1w3_weights = torch_to_tt_tensor_rm(
            torch.transpose(
                torch.cat(
                    [state_dict[f"{base_address}w1.weight"], state_dict[f"{base_address}w3.weight"]],
                ),
                -1,
                -2,
            ),
            self.device,
            put_on_device=False,
        )
        self.output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
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

    def fused_matmul(self, activation):
        return tt_lib.tensor.matmul(activation, self.fused_w1w3_weights, self.output_mem_config)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = self.fused_matmul(x)
        # x = tt_lib.tensor.transpose(x,1,3)
        x0, x1 = tt_lib.tensor.split_last_dim_two_chunks_tiled(x)
        x = tt_lib.tensor.mul(x0, x1)
        x = tt_lib.tensor.silu(x)
        # x = tt_lib.tensor.transpose(x,1,3)
        return self.w2(x)


# shim to allow disambiguate between both options
class TtFeedForward(nn.Module):
    def __init__(self, args: TtModelArgs, base_address=None, device=None, state_dict=None):
        super().__init__()
        if args.FALLBACK_FUSED_ATTENTION:
            self._impl = TtClassicFeedForward(args, base_address, device, state_dict)
        else:
            self._impl = TtFusedFeedForward(args, base_address, device, state_dict)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        return self._impl.forward(x)
