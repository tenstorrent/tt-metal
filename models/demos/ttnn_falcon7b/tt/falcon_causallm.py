# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_model import TtFalconModelShared
from models.utility_functions import is_wormhole_b0


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

        if is_wormhole_b0():
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
            )
            self.core_grid = ttnn.CoreGrid(y=7, x=8)
        else:
            self.compute_kernel_config = None
            self.core_grid = ttnn.CoreGrid(y=9, x=12)

    def __call__(
        self,
        input_embeddings: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().__call__(
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
            core_grid=self.core_grid,
        )

        return lm_logits, presents
