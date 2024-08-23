# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from loguru import logger

import ttnn

from models.demos.metal_BERT_large_11.tt.embeddings import TtEmbeddings
from models.demos.metal_BERT_large_11.tt.bert_encoder import TtBertEncoder

from tt_lib.utils import pad_activation, pad_weight


class TtBertBatchDram:
    def __init__(self, config, hugging_face_reference_model, device, model_config, tt_cache_path):
        self.device = device
        self.model_config = model_config

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        self.hidden_states_list = []
        self.tt_attention_mask_list = []

        self.embeddings = TtEmbeddings(
            hugging_face_reference_model,
            device,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask

        self.encoders = [
            TtBertEncoder(config, encoder_idx, state_dict, device, model_config, tt_cache_path)
            for encoder_idx in range(config.num_hidden_layers)
        ]

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        if tt_cache_path is not None:
            weight = ttnn.load_tensor(
                str(tt_cache_path / f"qa_outputs.weight_{self.model_config['QA_LINEAR_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, self.model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
            bias = ttnn.load_tensor(
                str(tt_cache_path / f"qa_outputs.bias_{self.model_config['QA_LINEAR_BIAS_DTYPE'].name}.bin")
            ).to(device, self.model_config["QA_LINEAR_BIAS_MEMCFG"])
        else:
            weight = pad_weight(torch.transpose(state_dict["qa_outputs.weight"], -2, -1))
            weight = (
                ttnn.Tensor(
                    weight.reshape(-1).tolist(),
                    weight.shape,
                    model_config["QA_LINEAR_WEIGHTS_DTYPE"],
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device, model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
            )
            bias = pad_weight(state_dict["qa_outputs.bias"])
            bias = (
                ttnn.Tensor(
                    bias.reshape(-1).tolist(),
                    bias.shape,
                    model_config["QA_LINEAR_BIAS_DTYPE"],
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device, model_config["QA_LINEAR_BIAS_MEMCFG"])
            )

        # QA linear
        # TODO: Replace with custom op with fused bias?
        def qa_linear_(activation):
            output = ttnn.matmul(activation, weight, memory_config=model_config["QA_LINEAR_OUTPUT_MEMCFG"])
            output_plus_bias = ttnn.bcast(
                output,
                bias,
                ttnn.BcastOpMath.ADD,
                ttnn.BcastOpDim.H,
                memory_config=model_config["QA_LINEAR_OUTPUT_MEMCFG"],
            )
            return output_plus_bias

        self.qa_linear = qa_linear_

    def model_embedding(self, input_ids, token_type_ids=None, position_ids=None):
        tt_embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        if "OP1_FUSED_QKV_MM_INPUT_SHARDED_MEMCFG" in self.model_config:
            tt_embeddings = ttnn.interleaved_to_sharded(
                tt_embeddings,
                self.model_config["GRID_SIZE"],
                self.model_config["SHARD_SIZE"],
                self.model_config["OP1_FUSED_QKV_MM_INPUT_SHARDED_MEMCFG"].memory_layout,
                self.model_config["SHARD_ORIENTATION"],
                self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
            )
        elif tt_embeddings.get_dtype() != self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"]:
            logger.warning("Perf warning: On host conversion of dtype after embeddings")
            embeddings = tt_embeddings.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            tt_embeddings = (
                ttnn.Tensor(
                    embeddings,
                    # output of embeddings dtype should be same as op1
                    self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
                ).to(ttnn.TILE_LAYOUT)
                # output config of embeddings should be same as op1_input
                .to(self.device, self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"])
            )

        return tt_embeddings

    def model_attention_mask(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
            extended_attention_mask = torch.clamp(
                extended_attention_mask, -100000
            )  # Limit neg value that goes into exp
            if self.model_config["OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG"].is_sharded():
                extended_attention_mask = extended_attention_mask.reshape(extended_attention_mask.shape[0], 1, -1, 32)
                tt_attention_mask = ttnn.Tensor(
                    extended_attention_mask,
                    self.model_config["OP4_SOFTMAX_ATTENTION_MASK_DTYPE"],
                )
            else:
                extended_attention_mask = pad_activation(extended_attention_mask)
                tt_attention_mask = ttnn.Tensor(
                    extended_attention_mask,
                    self.model_config["OP4_SOFTMAX_ATTENTION_MASK_DTYPE"],
                ).to(ttnn.TILE_LAYOUT)
        else:
            tt_attention_mask = attention_mask
        return tt_attention_mask

    def __call__(self, tt_embeddings, tt_attention_mask=None):
        # profiler.start("_run_encoders")
        hidden_states = tt_embeddings
        attention_mask = tt_attention_mask

        for encoder in self.encoders:
            # profiler.start("__one_encoder")
            hidden_states = encoder(hidden_states, attention_mask)
            if self.model_config["MOVE_ENCODER_OUTPUT_BOOL"]:
                hidden_states = ttnn.move(hidden_states)
            # profiler.end("__one_encoder")
        if hidden_states.memory_config().is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(
                hidden_states,
                self.model_config["QA_LINEAR_OUTPUT_MEMCFG"],
                self.model_config["QA_LINEAR_WEIGHTS_DTYPE"],
            )
        # profiler.end("_run_encoders")

        # profiler.start("_qa_linear")
        res = self.qa_linear(hidden_states)
        # profiler.end("_qa_linear")

        return res
