# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon7b.tt.falcon_model import TtFalconModelShared
from models.demos.falcon7b.tt.model_utils import get_weights_cached


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            devices=devices,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )
        self.num_devices = len(devices)
        self.model_config = model_config

        lm_head_str = f"lm_head.weight"

        self.lm_head_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            lm_head_str,
            weight_config_str="LM_HEAD_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(self.state_dict[f"lm_head.weight"], -2, -1) if self.state_dict else None),
        )

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

        lm_logits = []
        for i in range(self.num_devices):
            lm_logits.append(
                tt_lib.tensor.falcon_lm_head_matmul(
                    hidden_states[i],
                    self.lm_head_weights[i],
                    bias=None,
                    output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                )
            )

        return lm_logits, presents
