import torch
import pytest
from torch import nn

import tt_lib

from tests.python_api_testing.models.falcon.falcon_model import TtFalconModelShared
from tt_models.helper_funcs import Linear as TTLinear
from tt_models.utility_functions import torch2tt_tensor


class TtFalconCausalLM(TtFalconModelShared):
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
        assert base_url == "", "base_url should be empty at the root of the model!"

        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=f"transformer",
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )
        self.model_config = model_config

        lm_head_str = f"lm_head.weight"
        if tt_cache_path is not None:
            self.lm_head_weights = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{lm_head_str}_{self.model_config['LM_HEAD_MM_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"])
        else:
            self.lm_head_weights = torch2tt_tensor(
                torch.transpose(self.state_dict[f"lm_head.weight"], -2, -1),
                self.device,
                tt_memory_config=self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
            )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = super().forward(input_embeddings, attention_mask)
        lm_logits = tt_lib.tensor.falcon_lm_head_matmul(
            hidden_states,
            self.lm_head_weights,
            output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
        )

        return lm_logits
