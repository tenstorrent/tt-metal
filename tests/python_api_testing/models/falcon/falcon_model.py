import torch
import pytest
from torch import nn
from abc import abstractmethod
from typing import Optional, Tuple

import tt_lib

from tests.python_api_testing.models.falcon.falcon_decoder import TtFalconDecoderLayer
from models.utility_functions import (
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
        llm_mode,
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
        self.llm_mode = llm_mode
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
                    llm_mode=llm_mode,
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

    def model_preprocessing(self, input_ids, kv_cache_len):
        # input_ids: torch.Tensor with shape [batch, seq_len]

        assert input_ids.dim() == 2
        batch, seq_len = input_ids.shape

        embeddings = self.embeddings(input_ids)

        # Generate input and attention_mask ---------------------------------------------
        # TODO: Generate attention_mask on device
        if self.llm_mode == "prefill":
            q_len, kv_len = seq_len, seq_len
            assert batch == 1, "For prefill, batch must be 1!"
            assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
            assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

            tt_embeddings = torch2tt_tensor(
                embeddings.unsqueeze(1),
                self.device,
                tt_memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
            )

            attention_mask_bool = torch.ones(batch, 1, q_len, kv_len, dtype=bool).triu(
                diagonal=1
            )
            tt_attention_mask = torch2tt_tensor(
                (attention_mask_bool * -100000).expand(-1, self.config.n_head, -1, -1),
                self.device,
                tt_memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
            )

        elif self.llm_mode == "decode":
            q_len, kv_len = seq_len, kv_cache_len + 1
            assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
            assert q_len == 1, "For decode, q_len must be 1!"

            tt_embeddings = torch2tt_tensor(
                embeddings.unsqueeze(1).transpose(0, 2),
                self.device,
                tt_memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
            )

            attention_mask_bool = torch.zeros(batch, 1, q_len, kv_len, dtype=bool)
            attention_mask_bool[:, :, :, -1] = True

            kv_len_padded = (kv_len + 31) // 32 * 32
            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
                ),
                dim=-1,
            )
            tt_attention_mask = torch2tt_tensor(
                (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                    -1, self.config.n_head, -1, -1
                ),
                self.device,
                tt_memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
            )

        else:
            raise NotImplementedError(
                f"Llm mode {self.llm_mode} is not supported! Must be one of prefill or decode."
            )

        return tt_embeddings, tt_attention_mask

    @abstractmethod
    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
    ) -> tt_lib.tensor.Tensor:
        layer_output = input_embeddings
        for idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=layer_output,
                alibi=None,
                attention_mask=attention_mask,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
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
        llm_mode,
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
            llm_mode=llm_mode,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = super().forward(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
        )
        return hidden_states
