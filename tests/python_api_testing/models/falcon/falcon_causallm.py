import torch
import pytest
from torch import nn

import tt_lib

from tests.python_api_testing.models.falcon.falcon_model import TtFalconModelShared
from models.helper_funcs import Linear as TTLinear
from models.utility_functions import torch2tt_tensor


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
    ):
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
        )

        self.lm_head_weights = tt_lib.tensor.transpose(
            torch2tt_tensor(
                self.state_dict[f"lm_head.weight"],
                self.device,
                tt_dtype=tt_lib.tensor.DataType.BFLOAT8_B,
            )
        )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = super().forward(input_embeddings, attention_mask)
        lm_logits = tt_lib.tensor.falcon_lm_head_matmul(hidden_states, self.lm_head_weights)

        return lm_logits
