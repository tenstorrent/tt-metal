# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import torch
import torch.nn as nn
from models.experimental.mistral.tt.model_config import TtModelArgs
from models.experimental.mistral.tt.mistral_decoder import TtTransformerBlock
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.mistral_helper_funcs import (
    Linear as TtLinear,
    format_tensor,
    unpad_from_zero,
    get_freqs_cis,
)
from models.utility_functions import torch_to_tt_tensor_rm, torch2tt_tensor
from typing import Optional


class TtTransformer(nn.Module):
    def __init__(
        self,
        args=None,
        devices=None,
        state_dict=None,
        base_address=None,
        model_config=None,
        tt_cos_cached=None,
        tt_sin_cached=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.devices = devices
        self.num_devices = len(devices)
        self.base_address = base_address
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    devices=self.devices,
                    state_dict=state_dict,
                    base_address=f"layers.{i}.",
                    model_config=model_config,
                    tt_cos_cached=tt_cos_cached,
                    tt_sin_cached=tt_sin_cached,
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = TtRMSNorm(
            device=devices[0],
            base_address=f"norm.",
            state_dict=state_dict,
            model_config=model_config,
        )
        self.state_dict = state_dict

        self.output_weight = torch2tt_tensor(self.state_dict["output.weight"], tt_device=devices[0])

        self.output = TtLinear(
            args.dim,
            args.vocab_size,
            self.output_weight,
        )

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: Optional[tt_lib.tensor.Tensor],
    ):
        for layer in self.layers:
            xs = layer(xs, start_pos, current_pos, attn_masks)

        output = self.output(self.norm(xs))
        xs.deallocate()
        return output
