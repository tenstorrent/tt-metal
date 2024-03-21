# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon7b.tt.falcon_attention import TtFalconAttention
from models.demos.falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.falcon7b.tt.model_utils import get_weights_cached


class TtFalconDecoderLayer(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.devices = devices
        self.num_devices = len(devices)
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.mlp = TtFalconMLP(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        layer_name = f"{base_url}.{layer_num}"

        layernorm_weights_str = f"{layer_name}.input_layernorm.weight"
        layernorm_bias_str = f"{layer_name}.input_layernorm.bias"

        self.layernorm_gamma = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_weights_str,
            weight_config_str="INPUT_LAYERNORM_WEIGHTS",
            weights_to_cache=(self.state_dict[layernorm_weights_str] if self.state_dict else None),
            padzero=True,
        )
        self.layernorm_beta = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_bias_str,
            weight_config_str="INPUT_LAYERNORM_BIAS",
            weights_to_cache=(self.state_dict[layernorm_bias_str] if self.state_dict else None),
            padzero=True,
        )

        self.layernorm_eps = config.layer_norm_epsilon

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        layernorm_output = []
        for i in range(self.num_devices):
            layernorm_output.append(
                tt_lib.tensor.layernorm(
                    hidden_states[i],
                    self.layernorm_eps,
                    output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
                )
            )
        for i in range(self.num_devices):
            layernorm_output[i] = tt_lib.tensor.bcast(
                layernorm_output[i],
                self.layernorm_gamma[i],
                tt_lib.tensor.BcastOpMath.MUL,
                tt_lib.tensor.BcastOpDim.H,
                output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
            )
        for i in range(self.num_devices):
            layernorm_output[i] = tt_lib.tensor.bcast(
                layernorm_output[i],
                self.layernorm_beta[i],
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.H,
                output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
            )

        residual = hidden_states

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=layernorm_output,
            alibi=alibi,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output, layer_present = attn_outputs[0], attn_outputs[1]

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(layernorm_output)

        output = []
        for i in range(self.num_devices):
            output.append(
                tt_lib.tensor.add(
                    mlp_output[i],
                    attention_output[i],
                    output_mem_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
                )
            )
            mlp_output[i].deallocate()
            attention_output[i].deallocate()

        # dropout_add
        # For inference, this is just add
        for i in range(self.num_devices):
            output[i] = tt_lib.tensor.add(
                output[i],
                residual[i],
                output_mem_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
            )
            residual[i].deallocate()

        if use_cache:
            outputs = (output, layer_present)
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs
