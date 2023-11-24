# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import math
import torch

from typing import Optional
import tt_lib
from tt_lib.utils import pad_weight
from models.utility_functions import torch2tt_tensor


def mha(qkv_weight, qkv_bias, hidden_dim, num_heads, device, model_config):
    assert isinstance(num_heads, int) and num_heads > 0

    # Used to scale down the input to the softmax
    freciprocal_of_sqrt_hidden_dim = 1 / math.sqrt(hidden_dim // num_heads)

    def op1_qkv_fused(activation, qkv_weight, qkv_bias):
        qkv = tt_lib.tensor.bert_large_fused_qkv_matmul(
            activation,
            qkv_weight,
            qkv_bias,
            output_mem_config=model_config["OP1_FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP1_FUSED_QKV_MM_OUTPUT_DTYPE"],
        )
        return qkv

    def op2to6_create_qkv_heads(qkv):
        (
            q_heads,
            kt_heads,
            v_heads,
        ) = tt_lib.operations.primary.transformers.split_fused_qkv_and_split_heads(
            qkv,
            tt_lib.tensor.CoreCoord(12, 9),
            output_mem_config=model_config["OP2TO6_SPLIT_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        return q_heads, kt_heads, v_heads

    def op7_bmm(Q_heads, K_T_heads):
        qkt = tt_lib.tensor.bert_large_pre_softmax_bmm(
            Q_heads,
            K_T_heads,
            output_mem_config=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP7_PRE_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )
        return qkt

    def op8_scale_mask_softmax(qkt, attention_mask):
        # Attention scores computation

        # Input and output tensors of this fused op is: [9, 1, 6144, 384] instead of [9, 16, 384, 384]
        # No-op reshapes are handled within pre-softmax (op 7) and post-softmax bmms (op 9)
        if attention_mask is not None:
            attention_scores = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                qkt, freciprocal_of_sqrt_hidden_dim, attention_mask
            )
        else:
            # No pass in mha sub-graph or full bert encoder uses this anymore
            assert False, "Must provide attention_mask to scale_mask_softmax in mha sub-graph!"

        return attention_scores

    def op9_bmm(attention_scores, V_heads):
        weighted_activation = tt_lib.tensor.bert_large_post_softmax_bmm(
            attention_scores,
            V_heads,
            output_mem_config=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_MEMCFG"],
            output_dtype=model_config["OP9_POST_SOFTMAX_BMM_OUTPUT_DTYPE"],
        )

        return weighted_activation

    def op10_unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            retval = tt_lib.tensor.nlp_concat_heads(
                x,
                output_mem_config=model_config["OP10_CONCATENATE_ATTENTION_HEADS_OUTPUT_MEMCFG"],
            )
            return retval

    def mha_(activation, attention_mask):
        qkv = op1_qkv_fused(activation, qkv_weight, qkv_bias)
        # activation.deallocate()

        Q_heads, K_T_heads, V_heads = op2to6_create_qkv_heads(qkv)
        qkv.deallocate()

        qkt = op7_bmm(Q_heads, K_T_heads)
        Q_heads.deallocate()
        K_T_heads.deallocate()

        attention_scores = op8_scale_mask_softmax(qkt, attention_mask)
        # Should be a no-op deallocate since it was moved?
        # qkt.deallocate()
        weighted_activation = op9_bmm(attention_scores, V_heads)
        attention_scores.deallocate()
        V_heads.deallocate()

        res = op10_unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]
        weighted_activation.deallocate()

        return res

    return mha_


class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device, model_config, tt_cache_path):
        super().__init__()

        layer_name = f"bert.encoder.layer.{encoder_idx}.attention.self"

        if tt_cache_path is not None:
            qkv_weight = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path / f"{layer_name}.qkv.weight_{model_config['OP1_FUSED_QKV_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, model_config["OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG"])
            qkv_bias = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"{layer_name}.qkv.bias_{model_config['OP1_FUSED_QKV_MM_BIAS_DTYPE'].name}.bin")
            ).to(device, model_config["OP1_FUSED_QKV_MM_BIAS_MEMCFG"])
        else:
            qw = pad_weight(state_dict[f"{layer_name}.query.weight"])
            qb = pad_weight(state_dict[f"{layer_name}.query.bias"])
            kw = pad_weight(state_dict[f"{layer_name}.key.weight"])
            kb = pad_weight(state_dict[f"{layer_name}.key.bias"])
            vw = pad_weight(state_dict[f"{layer_name}.value.weight"])
            vb = pad_weight(state_dict[f"{layer_name}.value.bias"])

            # Weights pre-transposed on host​. No on-the fly transpose of W​
            qw = torch.transpose(qw, -1, -2)
            kw = torch.transpose(kw, -1, -2)
            vw = torch.transpose(vw, -1, -2)

            qkv_weight = torch.cat((qw, kw, vw), -1)
            qkv_bias = torch.cat((qb, kb, vb), -1)

            qkv_weight = torch2tt_tensor(
                qkv_weight,
                device,
                tt_layout=tt_lib.tensor.Layout.TILE,
                tt_memory_config=model_config["OP1_FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                tt_dtype=model_config["OP1_FUSED_QKV_MM_WEIGHTS_DTYPE"],
            )

            qkv_bias = torch2tt_tensor(
                qkv_bias,
                device,
                tt_layout=tt_lib.tensor.Layout.TILE,
                tt_memory_config=model_config["OP1_FUSED_QKV_MM_BIAS_MEMCFG"],
                tt_dtype=model_config["OP1_FUSED_QKV_MM_BIAS_DTYPE"],
            )

        # Hidden dim
        hidden_dim = qkv_weight.shape()[-1] // 3

        self.mha = mha(
            qkv_weight,
            qkv_bias,
            hidden_dim,
            config.num_attention_heads,
            device,
            model_config,
        )

    def forward(
        self, activation: tt_lib.tensor.Tensor, attention_mask: Optional[tt_lib.tensor.Tensor] = None
    ) -> tt_lib.tensor.Tensor:
        result = self.mha(activation, attention_mask)
        return result
