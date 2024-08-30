# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
from models.demos.falcon7b_common.tt.falcon_attention import TtFalconAttentionDecode, TtFalconAttentionPrefill
from models.demos.falcon7b_common.tt.falcon_mlp import TtFalconMLPDecode, TtFalconMLPPrefill
from models.demos.falcon7b_common.tt.model_utils import get_weights_cached, layernorm
from torch import nn


class TtFalconDecoderLayer(nn.Module):
    def __init__(
        self,
        mesh_device,
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
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.weights_dict = {}

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn_prefill = TtFalconAttentionPrefill(
            mesh_device=mesh_device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.self_attn_decode = TtFalconAttentionDecode(
            mesh_device=mesh_device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.mlp_prefill = TtFalconMLPPrefill(
            mesh_device=mesh_device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.mlp_decode = TtFalconMLPDecode(
            mesh_device=mesh_device,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        layer_name = f"{base_url}.{layer_num}"

        layernorm_weights_str = f"{layer_name}.input_layernorm.weight"
        layernorm_bias_str = f"{layer_name}.input_layernorm.bias"

        self.layernorm_gamma = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            layernorm_weights_str,
            weight_config_str="INPUT_LAYERNORM_WEIGHTS",
            weights_to_cache=(self.state_dict[layernorm_weights_str] if self.state_dict else None),
        )
        self.layernorm_beta = get_weights_cached(
            mesh_device,
            model_config,
            tt_cache_path,
            layernorm_bias_str,
            weight_config_str="INPUT_LAYERNORM_BIAS",
            weights_to_cache=(self.state_dict[layernorm_bias_str] if self.state_dict else None),
        )

        self.layernorm_eps = config.layer_norm_epsilon

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        layernorm_output = layernorm(
            hidden_states,
            self.layernorm_eps,
            self.layernorm_gamma,
            self.layernorm_beta,
            self.model_config,
        )

        residual = hidden_states

        # Attention and MLP execution
        # mlp will deallocate layernorm_output
        if llm_mode == "prefill":
            attn_outputs = self.self_attn_prefill(
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
            mlp_output = self.mlp_prefill(layernorm_output)

        elif llm_mode == "decode":
            attn_outputs = self.self_attn_decode(
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
            mlp_output = self.mlp_decode(layernorm_output)

        else:
            raise ValueError(f"Unknown llm_mode: {llm_mode}")

        output = ttnn.add(
            mlp_output,
            attention_output,
            memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
        )
        mlp_output.deallocate()
        attention_output.deallocate()

        # dropout_add
        # For inference, this is just add
        output = ttnn.add(
            output,
            residual,
            memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
        )
        residual.deallocate()

        if use_cache:
            outputs = (output, layer_present)
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs
