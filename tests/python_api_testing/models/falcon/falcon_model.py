import torch
import pytest
from torch import nn
from abc import abstractmethod

import tt_lib

from tests.python_api_testing.models.falcon.falcon_decoder import TtFalconDecoderLayer
from tt_models.utility_functions import (
    torch2tt_tensor,
    pad_by_zero,
)


class TtFalconModelShared(torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        # So far on CPU until we add embeddings support on device
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(
            state_dict[f"{base_url}.word_embeddings.weight"]
        )

        # stack all decoders
        self.layers = torch.nn.ModuleList(
            [
                TtFalconDecoderLayer(
                    device=device,
                    state_dict=state_dict,
                    base_url=f"{base_url}.h",
                    layer_num=layer_num,
                    config=config,
                    max_position_embeddings=max_position_embeddings,
                    model_config=model_config,
                    tt_cache_path=tt_cache_path,
                )
                for layer_num in range(num_layers)
            ]
        )

        layer_name = f"{base_url}"

        embeddings_weights_str = f"{layer_name}.word_embeddings.weight"
        layernorm_weights_str = f"{layer_name}.ln_f.weight"
        layernorm_bias_str = f"{layer_name}.ln_f.bias"
        if tt_cache_path is not None:
            # self.embeddings_weight = tt_lib.tensor.load_tensor(
            #     str(tt_cache_path
            #     / f"{embeddings_weights_str}_{self.model_config['WORD_EMBEDDING_WEIGHTS_DTYPE'].name}.bin")
            # ).to(device, self.model_config["WORD_EMBEDDING_WEIGHTS_MEMCFG"])
            self.layernorm_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layernorm_weights_str}_{self.model_config['LN_F_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["LN_F_WEIGHTS_MEMCFG"])
            self.layernorm_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layernorm_bias_str}_{self.model_config['LN_F_BIAS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["LN_F_BIAS_MEMCFG"])
        else:
            # self.embeddings_weight = torch2tt_tensor(
            #     self.state_dict[embeddings_weights_str],
            #     device,
            #     tt_lib.tensor.Layout.ROW_MAJOR,
            #     self.model_config["WORD_EMBEDDING_WEIGHTS_MEMCFG"],
            #     self.model_config['WORD_EMBEDDING_WEIGHTS_DTYPE']
            # )
            self.layernorm_gamma = pad_by_zero(
                self.state_dict[layernorm_weights_str],
                device,
                tt_memory_config=self.model_config["LN_F_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["LN_F_WEIGHTS_DTYPE"],
            )[0]
            self.layernorm_beta = pad_by_zero(
                self.state_dict[layernorm_bias_str],
                device,
                tt_memory_config=self.model_config["LN_F_BIAS_MEMCFG"],
                tt_dtype=self.model_config["LN_F_BIAS_DTYPE"],
            )[0]
        self.layernorm_eps = config.layer_norm_epsilon

    def model_preprocessing(self, input_ids):
        # input_ids: torch.Tensor with shape [batch, seq_len]
        assert input_ids.dim() == 2
        embeddings = self.embeddings(input_ids)
        tt_embeddings = torch2tt_tensor(
            embeddings,
            self.device,
            tt_memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
            tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
        )

        batch, seq_len = input_ids.shape
        q_len, kv_seq_len = seq_len, seq_len
        tt_attention_mask = (
            torch.ones(batch, self.config.n_head, q_len, kv_seq_len) * -100000
        ).triu(diagonal=1)
        tt_attention_mask = torch2tt_tensor(
            tt_attention_mask,
            self.device,
            tt_memory_config=self.model_config["ATTN_MASK_MEMCFG"],
            tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
        )

        return tt_embeddings, tt_attention_mask

    @abstractmethod
    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        layer_output = input_embeddings
        for idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=layer_output, alibi=None, attention_mask=attention_mask
            )[0]

        # apply final norm layer
        layer_output = tt_lib.tensor.layernorm(
            layer_output,
            self.layernorm_eps,  # These don't fit: self.layernorm_gamma, self.layernorm_beta
            output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["LN_F_OUTPUT_DTYPE"], # Not currently supported
        )
        layer_output = tt_lib.tensor.bcast(
            layer_output,
            self.layernorm_gamma,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["LN_F_OUTPUT_DTYPE"], # Not currently supported
        )
        layer_output = tt_lib.tensor.bcast(
            layer_output,
            self.layernorm_beta,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            # output_dtype=self.model_config["LN_F_OUTPUT_DTYPE"], # Not currently supported
        )

        return layer_output


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=base_url,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = super().forward(input_embeddings, attention_mask)
        return hidden_states
