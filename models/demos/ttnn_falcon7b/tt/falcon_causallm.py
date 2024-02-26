# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_model import TtFalconModelShared


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        device,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        parameters,
    ):
        super().__init__(
            device=device,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            parameters=parameters.transformer,
        )
        self.model_config = model_config
        self.lm_head_weights = parameters.lm_head.weight

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        lm_logits = ttnn.matmul(
            hidden_states,
            self.lm_head_weights,
            memory_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
            use_1d_systolic_array=True,
        )

        return lm_logits, presents
