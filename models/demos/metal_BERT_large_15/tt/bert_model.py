# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from loguru import logger

import tt_lib

from models.demos.metal_BERT_large_15.tt.embeddings import TtEmbeddings
from models.demos.metal_BERT_large_15.tt.bert_encoder import TtBertEncoder

from tt_lib.utils import pad_activation, pad_weight


class TtBertBatchDram(torch.nn.Module):
    def __init__(self, config, hugging_face_reference_model, device, model_config, tt_cache_path):
        super().__init__()
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

        self.encoders = torch.nn.ModuleList(
            [
                TtBertEncoder(config, encoder_idx, state_dict, device, model_config, tt_cache_path)
                for encoder_idx in range(config.num_hidden_layers)
            ]
        )

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        if tt_cache_path is not None:
            weight = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"qa_outputs.weight_{self.model_config['QA_LINEAR_WEIGHTS_DTYPE'].name}.bin")
            ).to(device, self.model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
            bias = tt_lib.tensor.load_tensor(
                str(tt_cache_path / f"qa_outputs.bias_{self.model_config['QA_LINEAR_BIAS_DTYPE'].name}.bin")
            ).to(device, self.model_config["QA_LINEAR_BIAS_MEMCFG"])
        else:
            weight = pad_weight(torch.transpose(state_dict["qa_outputs.weight"], -2, -1))
            weight = (
                tt_lib.tensor.Tensor(
                    weight.reshape(-1).tolist(),
                    weight.shape,
                    model_config["QA_LINEAR_WEIGHTS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["QA_LINEAR_WEIGHTS_MEMCFG"])
            )
            bias = pad_weight(state_dict["qa_outputs.bias"])
            bias = (
                tt_lib.tensor.Tensor(
                    bias.reshape(-1).tolist(),
                    bias.shape,
                    model_config["QA_LINEAR_BIAS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["QA_LINEAR_BIAS_MEMCFG"])
            )

        # QA linear
        # TODO: Replace with custom op with fused bias?
        def qa_linear_(activation):
            output = tt_lib.tensor.matmul(activation, weight, model_config["QA_LINEAR_OUTPUT_MEMCFG"])
            output_plus_bias = tt_lib.tensor.bcast(
                output,
                bias,
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.H,
                model_config["QA_LINEAR_OUTPUT_MEMCFG"],
            )
            return output_plus_bias

        self.qa_linear = qa_linear_

    def model_embedding(self, input_ids, token_type_ids=None, position_ids=None):
        tt_embeddings = self.embeddings(input_ids, token_type_ids, position_ids)
        embeddings_shape = tt_embeddings.shape()
        if tt_embeddings.dtype() != self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"]:
            logger.warning("Perf warning: On host conversion of dtype after embeddings")
            embeddings = tt_embeddings.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
            tt_embeddings = (
                tt_lib.tensor.Tensor(
                    embeddings.reshape(-1).tolist(),
                    (
                        embeddings_shape[0],
                        1,
                        embeddings_shape[-2],
                        embeddings_shape[-1],
                    ),
                    # output of embeddings dtype should be same as op1
                    self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                ).to(tt_lib.tensor.Layout.TILE)
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
            extended_attention_mask = pad_activation(extended_attention_mask)
            tt_attention_mask = (
                tt_lib.tensor.Tensor(
                    extended_attention_mask.reshape(-1).tolist(),
                    extended_attention_mask.shape,
                    self.model_config["OP8_SOFTMAX_ATTENTION_MASK_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(self.device, self.model_config["OP8_SOFTMAX_ATTENTION_MASK_MEMCFG"])
            )
        else:
            tt_attention_mask = attention_mask
        return tt_attention_mask

    def forward(self, tt_embeddings, tt_attention_mask=None):
        # profiler.start("_run_encoders")
        hidden_states = tt_embeddings
        attention_mask = tt_attention_mask

        for encoder in self.encoders:
            # profiler.start("__one_encoder")
            hidden_states = encoder(hidden_states, attention_mask)
            if self.model_config["MOVE_ENCODER_OUTPUT_BOOL"]:
                hidden_states = tt_lib.tensor.move(hidden_states)
            # profiler.end("__one_encoder")

        # profiler.end("_run_encoders")

        # profiler.start("_qa_linear")
        res = self.qa_linear(hidden_states)
        # profiler.end("_qa_linear")

        return res


# class TtBertBatchDram(torch.nn.Module):
#     def __init__(self, config, hugging_face_reference_model, device, model_config):
#         super().__init__()
#         self.device = device
#         self.model_config = model_config

#         # NOTE: Once we make embeddings run on device, pass in state dict
#         # instead of model itself
#         state_dict = hugging_face_reference_model.state_dict()

#         self.hidden_states_list = []
#         self.tt_attention_mask_list = []
#         self.get_extended_attention_mask = hugging_face_reference_model.get_extended_attention_mask

#         # Weights and bias tensors for QA Linear (at the end of model)
#         # (pad tensor so that last two dimensions are divisible by 32 and convert to TT tensor)
#         weight = pad_weight(torch.transpose(state_dict["qa_outputs.weight"], -2, -1))
#         weight = torch2tt_tensor(
#             weight,
#             device,
#             tt_layout=tt_lib.tensor.Layout.TILE,
#             tt_memory_config=model_config["QA_LINEAR_WEIGHTS_MEMCFG"],
#             tt_dtype=model_config["QA_LINEAR_WEIGHTS_DTYPE"],
#         )
#         bias = pad_weight(state_dict["qa_outputs.bias"])
#         bias = torch2tt_tensor(
#             bias,
#             device,
#             tt_layout=tt_lib.tensor.Layout.TILE,
#             tt_memory_config=model_config["QA_LINEAR_BIAS_MEMCFG"],
#             tt_dtype=model_config["QA_LINEAR_BIAS_DTYPE"],
#         )

#         # QA Linear implementation
#         def qa_linear_(activation):
#             output = tt_lib.tensor.matmul(activation, weight, model_config["QA_LINEAR_OUTPUT_MEMCFG"])
#             output_plus_bias = tt_lib.tensor.bcast(
#                 output,
#                 bias,
#                 tt_lib.tensor.BcastOpMath.ADD,
#                 tt_lib.tensor.BcastOpDim.H,
#                 model_config["QA_LINEAR_OUTPUT_MEMCFG"],
#             )
#             return output_plus_bias

#         # TT Model definition (Embeddings -> Encoder -> Encoder -> ... -> Encoder -> QA Linear)
#         self.embeddings = PytorchEmbeddings(
#             hugging_face_reference_model
#         )  # So far on CPU until we add embeddings support on device
#         self.encoders = torch.nn.ModuleList(
#             [
#                 TtBertEncoder(config, encoder_idx, state_dict, device, model_config)
#                 for encoder_idx in range(config.num_hidden_layers)
#             ]
#         )
#         self.qa_linear = qa_linear_

#     def model_preprocessing(self, input_ids, attention_mask=None, token_type_ids=None):
#         embeddings = self.embeddings(input_ids, token_type_ids)

#         # pad tensor so that last two dimensions are divisible by 32 and convert to TT tensor
#         pad_embeddings = pad_activation(embeddings)
#         tt_embeddings = torch2tt_tensor(
#             pad_embeddings,
#             self.device,
#             tt_layout=tt_lib.tensor.Layout.TILE,
#             tt_memory_config=self.model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"],
#             tt_dtype=self.model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
#         )

#         if attention_mask is not None:
#             extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)
#             # Limit neg value that goes into exp
#             extended_attention_mask = torch.clamp(extended_attention_mask, -100000)

#             # pad tensor so that last two dimensions are divisible by 32 and convert to TT tensor
#             extended_attention_mask = pad_activation(extended_attention_mask)
#             tt_attention_mask = torch2tt_tensor(
#                 extended_attention_mask,
#                 self.device,
#                 tt_layout=tt_lib.tensor.Layout.TILE,
#                 tt_memory_config=self.model_config["OP8_SOFTMAX_ATTENTION_MASK_MEMCFG"],
#                 tt_dtype=self.model_config["OP8_SOFTMAX_ATTENTION_MASK_DTYPE"],
#             )
#         else:
#             tt_attention_mask = attention_mask

#         return tt_embeddings, tt_attention_mask

#     def forward(
#         self,
#         tt_embeddings: tt_lib.tensor.Tensor,
#         tt_attention_mask: Optional[tt_lib.tensor.Tensor] = None,
#     ) -> tt_lib.tensor.Tensor:

#         hidden_states = tt_embeddings
#         attention_mask = tt_attention_mask

#         for encoder in self.encoders:
#             hidden_states = encoder(hidden_states, attention_mask)
#             if self.model_config["MOVE_ENCODER_OUTPUT_BOOL"]:
#                 hidden_states = tt_lib.tensor.move(hidden_states)

#         res = self.qa_linear(hidden_states)

#         return res
