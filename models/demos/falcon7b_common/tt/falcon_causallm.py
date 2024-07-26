# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
from models.demos.falcon7b_common.tt.falcon_lm_head import falcon_lm_head_matmul_2d
from models.demos.falcon7b_common.tt.falcon_model import TtFalconModelShared
from models.demos.falcon7b_common.tt.model_utils import get_falcon_default_core_grid, get_weights_cached
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    torch_tensors_to_tt_tensors,
)


def falcon_lm_head_matmul(
    input_tensor_a,
    input_tensor_b,
    core_grid,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    output_dtype=None,
):
    seq_len = input_tensor_a.get_legacy_shape()[2]
    if seq_len > 512:
        # TODO: Review if this path is used? If not, we can delete
        return ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=output_mem_config, dtype=output_dtype)

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    elif is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    else:
        compute_kernel_config = None

    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_mem_config,
        dtype=output_dtype,
        core_grid=core_grid,
        compute_kernel_config=compute_kernel_config,
    )


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
        seq_len,
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
        self.seq_len = seq_len

        lm_head_weight = None
        if self.state_dict:
            lm_head_weight = self.state_dict["lm_head.weight"]
            lm_head_weight = torch.transpose(lm_head_weight, -2, -1)

        if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.seq_len > 512:
            # Optimization for lm_head matmul
            self.num_slices = 4 if self.seq_len <= 1024 else 8
            if lm_head_weight is not None:
                PADDING = torch.zeros([64, lm_head_weight.shape[1] // self.num_slices])
                lm_head_weights = torch.chunk(lm_head_weight, self.num_slices, dim=-1)
                lm_head_weights_padded = [torch.cat([weight, PADDING], 0) for weight in lm_head_weights]
            # Cache sliced weights for lm_head with different seq_len
            self.lm_head_sliced_weights = [
                get_weights_cached(
                    devices,
                    model_config,
                    tt_cache_path,
                    f"lm_head.weight_slice_{i}_of_{self.num_slices}",
                    weight_config_str="LM_HEAD_MM_WEIGHTS",
                    weights_to_cache=lm_head_weights_padded[i] if lm_head_weight is not None else None,
                )
                for i in range(self.num_slices)
            ]
            # Generate padding for lm_head > 512
            padding = torch.zeros([1, 1, seq_len, 64])

            tt_paddings = torch_tensors_to_tt_tensors(
                [padding.detach().clone() for _ in range(self.num_devices)],
                ttnn.experimental.tensor.Layout.TILE,
                self.model_config["LM_HEAD_MM_INPUT_DTYPE"],
                self.model_config["LM_HEAD_MM_INPUT_MEMCFG"],
                self.devices,
            )
            self.lm_head_padding = tt_paddings

        self.lm_head_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            f"lm_head.weight",
            weight_config_str="LM_HEAD_MM_WEIGHTS",
            weights_to_cache=lm_head_weight,
        )

    def forward(
        self,
        input_ids: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        attention_mask: ttnn.experimental.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.experimental.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
        device_perf_run: bool = False,
    ) -> ttnn.experimental.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
            device_perf_run=device_perf_run,
        )

        if llm_mode == "prefill":
            if self.model_config["PREFILL_OPTIMIZED_MODE"] and hidden_states[0].get_legacy_shape()[-2] > 512:
                lm_logits = [
                    falcon_lm_head_matmul_2d(
                        hidden_states[device_id],
                        [weights[device_id] for weights in self.lm_head_sliced_weights],
                        self.num_slices,
                        lm_head_padding=self.lm_head_padding[device_id],
                        out_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                        out_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                    )
                    for device_id in range(self.num_devices)
                ]
            else:
                lm_logits = [
                    ttnn.matmul(
                        hidden_states[device_id],
                        self.lm_head_weights[device_id],
                        memory_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                        dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                        core_grid=get_falcon_default_core_grid(hidden_states[device_id].device()),
                        compute_kernel_config=self.model_config["LM_HEAD_KERNEL_CONFIG"],
                    )
                    for device_id in range(self.num_devices)
                ]
        else:
            lm_logits = [
                falcon_lm_head_matmul(
                    hidden_states[device_id],
                    self.lm_head_weights[device_id],
                    output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                    core_grid=get_falcon_default_core_grid(hidden_states[device_id].device()),
                )
                for device_id in range(self.num_devices)
            ]

        return lm_logits, presents
