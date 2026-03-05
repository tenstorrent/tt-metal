# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import create_pipeline_configuration_from_num_procs
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input


class ModelPipeline:
    def __init__(
        self,
        cache_path: Path,
        use_real_weights: bool,
        lm_head_fp32_dest_acc_en: bool = True,
        lm_head_persistent_mode: bool = True,
        dense_layer_id_override: int | None = None,
        moe_layer_id_override: int | None = None,
    ):
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline (weights={}, lm_head_fp32={}, lm_head_persistent_mode={})",
            "real" if use_real_weights else "synthetic",
            lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode,
        )
        if not is_slow_dispatch():
            raise RuntimeError(
                "DeepSeek V3 B1 pod pipeline requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
            )
        self.mesh_device = self._open_mesh_device()
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64):
            raise RuntimeError(f"Pod pipeline requires 4, 16, or 64 distributed processes; got {num_procs}")
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)

        provider: WeightProvider = CacheWeightProvider(cache_path) if use_real_weights else SyntheticWeightProvider()
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
        )
        if config.num_stages != num_procs:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes")

        logger.info("Building pipeline")
        self.pipeline = config.build_pipeline(self.mesh_device)

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._model: DeepSeekV3 | None = None
        self._page_size_datums: int = 0
        if self.pipeline.my_mesh_id == 0:
            self._model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
            )
            self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        logger.info("Created ModelPipeline for mesh id {}.", self.pipeline.my_mesh_id)

    @property
    def is_host(self) -> bool:
        return self.pipeline.my_mesh_id == 0

    def _open_mesh_device():
        fabric_router_config = ttnn._ttnn.fabric.FabricRouterConfig()
        fabric_router_config.max_packet_payload_size_bytes = 7168
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_2D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            fabric_router_config,
        )
        return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None],
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Run full inference: prefill the prompt then decode until EOS or max_new_tokens.

        Calls on_token(token_id) for each generated token (including the first
        one sampled after prefill). Returns the list of all generated token IDs.

        Must only be called on mesh_id == 0 (the host process).
        """
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("run_inference() must only be called on mesh id 0")
        assert self._model is not None

        prompt_tensors = [
            to_padded_input(
                torch.tensor([[tid]], dtype=torch.int32),
                batch_size=1,
                page_size_datums=self._page_size_datums,
            )
            for tid in prompt_token_ids
        ]
        last_output = self._model.prefill(prompt_tensors)
        next_token_id = int(ttnn.to_torch(last_output).to(torch.int32)[0, 0].item())
        generated = [next_token_id]
        on_token(next_token_id)
        logger.info("Prefill done ({} prompt tokens); first token: {}", len(prompt_token_ids), next_token_id)

        for step in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token_id == eos_token_id:
                logger.info("EOS token {} at decode step {}", eos_token_id, step)
                break
            output = self._model.decode_step(
                torch.tensor([[next_token_id]], dtype=torch.int32),
            )
            next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
            generated.append(next_token_id)
            on_token(next_token_id)

        logger.info("Generation complete ({} tokens generated)", len(generated))
        return generated
