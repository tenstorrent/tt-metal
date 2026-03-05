# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        mesh_device: ttnn.MeshDevice,
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
        self.mesh_device = mesh_device
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64):
            raise RuntimeError(f"Pod pipeline requires 4, 16, or 64 distributed processes; got {num_procs}")
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)

        # Each host loads/creates only the weights for its stage via the provider.
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

        logger.info(f"Building pipeline")
        self.pipeline = config.build_pipeline(self.mesh_device)

        logger.info(f"Setting up and running pipeline")
        self.pipeline.setup_and_run()

        if self.pipeline.my_mesh_id == 0:
            # Initialize host-side model interface for mesh id 0 (first stage)
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
            )
        logger.info(f"Created ModelPipeline for mesh id {self.pipeline.my_mesh_id}.")

    def prefill_forward(self, tokens: list[int]) -> int:
        """Prefill 1 user's prompt tokens and return the next token id."""
        # Host-side model interface is only invoked on mesh id 0
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        prompt_token_tensors = [
            to_padded_input(
                torch.tensor([[tid]], dtype=torch.int32),
                batch_size=1,
                page_size_datums=page_size_datums,
            )
            for tid in tokens
        ]
        last_output = self.model.prefill(prompt_token_tensors)
        next_token_id = int(ttnn.to_torch(last_output).to(torch.int32)[0, 0].item())
        logger.debug(f"Done prefilling with {len(tokens)} tokens.")
        return next_token_id

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id."""
        # Host-side model interface is only invoked on mesh id 0
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("decode_forward() should only be called on mesh id 0")
        output = self.model.decode_step(
            torch.tensor([[input_token]], dtype=torch.int32),
        )
        next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
        return next_token_id

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None],
        eos_token_id: int | None = None,
    ) -> None:
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("run_inference() must only be called on mesh id 0")
        next_token_id = self.prefill_forward(prompt_token_ids)
        on_token(next_token_id)
        for i in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token_id == eos_token_id:
                logger.debug("EOS token {} at decode step {}", eos_token_id, i)
                break
            next_token_id = self.decode_forward(next_token_id)
            logger.debug("Decoded token {} at decode step {}", next_token_id, i)
            on_token(next_token_id)
        logger.debug("Generation complete ({} tokens generated)", max_new_tokens)

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
