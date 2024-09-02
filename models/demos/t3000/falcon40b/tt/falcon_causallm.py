# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, Tuple

import ttnn
from ttnn import ShardTensorToMesh
from models.demos.t3000.falcon40b.tt.falcon_model import TtFalconModelShared

from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul, determine_tensor_deallocation


class TtFalconCausalLM(TtFalconModelShared):
    def __init__(
        self,
        mesh_device,
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
            mesh_device=mesh_device,
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
        self.mesh_device = mesh_device
        lm_head_str = f"lm_head.weight"

        lm_head_path = tt_cache_path / f"{lm_head_str}_{self.model_config['LM_HEAD_MM_WEIGHTS_DTYPE'].name}"

        self.lm_head_weights = ttnn.as_tensor(
            tensor=self.state_dict[f"lm_head.weight"],
            dtype=self.model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
            cache_file_name=lm_head_path,
            preprocess=lambda x: torch.transpose(x.reshape(1, 1, *x.shape), -2, -1),
        )
        self.perf_e2e_test_tile_tensor = ttnn.from_torch(
            torch.zeros((1, 1, 32, 32)), device=mesh_device.get_devices()[0]
        )

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        if llm_mode == "prefill":
            return self.fwd_prefill_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
        elif llm_mode == "decode":
            return self.fwd_decode_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
        else:
            assert False

    def fwd_prefill_causallm(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        need_low_l1_workaround = hidden_states.shape[2] > 1024

        # Workaround for non deterministic output/hang; issue: 7066
        overwrite_subblock_h = 1
        overwrite_subblock_w = 1 if hidden_states.shape[2] < 512 else 4

        should_deallocate_ln_tensors = determine_tensor_deallocation(
            self.model_config["layernorm_params"]["slice_size"], hidden_states.get_legacy_shape()[2]
        )
        # LM Head
        lm_logits = falcon_prefill_matmul(
            hidden_states,
            self.lm_head_weights,
            (
                self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"]
                if need_low_l1_workaround
                else self.model_config["COMPUTE_KERNEL_CONFIG"]
            ),  # FP16 accumulation format leads to lower PCC! But can't fit 2k S otherwise atm
            output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
            overwrite_per_core_k=1,  # TODO: can we increase this?
            overwrite_subblock_w=overwrite_subblock_w,
            overwrite_subblock_h=overwrite_subblock_h,
        )

        if should_deallocate_ln_tensors:
            hidden_states.deallocate(True)

        return lm_logits, presents

    def fwd_decode_causallm(
        self,
        input_ids: ttnn.Tensor,
        llm_mode: str,
        attention_mask: ttnn.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states, presents = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )

        # LM Head
        lm_logits = ttnn.matmul(
            hidden_states,
            self.lm_head_weights,
            program_config=self.model_config["LM_HEAD_MM_PROGCFG"],
            memory_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
            dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        hidden_states.deallocate(True)

        return lm_logits, presents
