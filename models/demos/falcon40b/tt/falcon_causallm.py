# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon40b.tt.falcon_model import TtFalconModelShared
from models.utility_functions import torch2tt_tensor


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
        use_global_cos_sin_cache,
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
            use_global_cos_sin_cache=use_global_cos_sin_cache,
        )
        self.model_config = model_config

        lm_head_str = f"lm_head.weight"
        num_devices = len(devices)
        self.lm_head_weights = []
        for i in range(num_devices):
            lm_head_path = (
                tt_cache_path
                / f"{lm_head_str}_{i}_{num_devices}_{self.model_config['LM_HEAD_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (lm_head_path).exists():
                self.lm_head_weights.append(
                    tt_lib.tensor.load_tensor(str(lm_head_path)).to(
                        devices[i], self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                lm_head_weights_host = torch2tt_tensor(
                    torch.transpose(torch.chunk(self.state_dict[f"lm_head.weight"], num_devices)[i], -2, -1),
                    None,
                    tt_memory_config=self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
                )
                self.lm_head_weights.append(
                    lm_head_weights_host.to(devices[i], self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(lm_head_path),
                    lm_head_weights_host,
                )

    def __call__(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        llm_mode: str,
        attention_mask: tt_lib.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> tt_lib.tensor.Tensor:
        hidden_states, presents = super().__call__(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        lm_logits = []
        for i in range(len(hidden_states)):
            lm_logits.append(
                tt_lib.operations.primary.matmul_1d(
                    hidden_states[i],
                    self.lm_head_weights[i],
                    program_config=self.model_config["LM_HEAD_MM_PROGCFG"],
                    output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            hidden_states[i].deallocate(True)

        return lm_logits, presents
