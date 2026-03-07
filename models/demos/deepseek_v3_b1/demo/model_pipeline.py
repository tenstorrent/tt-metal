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
from models.demos.deepseek_v3_b1.demo.stage import TOKEN_PAGE_SIZE_BYTES
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)

TOKEN_ID_BYTES = 4
TOKEN_PAGE_SIZE_DATUMS = TOKEN_PAGE_SIZE_BYTES // TOKEN_ID_BYTES


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

        if self.pipeline.is_output_rank:
            self._d2h_output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_DATUMS, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        logger.info(f"Created ModelPipeline for mesh id {self.pipeline.my_mesh_id}.")

    def _make_token_tensor(self, token_id: int) -> ttnn.Tensor:
        """Build a PCIe-aligned token tensor for write_token."""
        buf = torch.zeros(1, TOKEN_PAGE_SIZE_DATUMS, dtype=torch.int32)
        buf[0, 0] = token_id
        return ttnn.from_torch(buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    @property
    def is_input_rank(self) -> bool:
        return bool(self.pipeline.is_input_rank)

    def run_inference(
        self,
        prompt_token_ids: list[int] | None,
        max_new_tokens: int | None,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run full inference across all ranks.

        Only the input rank needs real prompt_token_ids/max_new_tokens. The input rank
        sends (num_prefill, max_new_tokens) to the output rank so other ranks may pass None.
        Both input and output ranks independently break early when EOS is produced after prefill.

        - Input rank: writes tokens (prefill then decode) and receives output
          token ids from the output rank via recv_token.
        - Output rank: reads D2H output and sends the token id back to the
          input rank via send_token.
        - Other ranks: no-op (pipeline kernels handle forwarding).
        """
        if self.pipeline.is_input_rank:
            assert prompt_token_ids is not None, "Input rank requires prompt_token_ids"
            assert max_new_tokens is not None, "Input rank requires max_new_tokens"

            num_prefill = len(prompt_token_ids)
            ttnn.send_token(num_prefill, ttnn.Rank(self.pipeline.output_rank))
            ttnn.send_token(max_new_tokens, ttnn.Rank(self.pipeline.output_rank))
            return self._run_input_rank(
                prompt_token_ids,
                num_prefill=num_prefill,
                max_new_tokens=max_new_tokens,
                on_token=on_token,
                eos_token_id=eos_token_id,
                return_generated_tokens=return_generated_tokens,
            )
        elif self.pipeline.is_output_rank:
            num_prefill = int(ttnn.recv_token(ttnn.Rank(self.pipeline.input_rank)))
            max_new_tokens = int(ttnn.recv_token(ttnn.Rank(self.pipeline.input_rank)))
            self._run_output_rank(
                num_prefill=num_prefill,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        return None

    def _run_input_rank(
        self,
        prompt_token_ids: list[int],
        num_prefill: int,
        max_new_tokens: int,
        on_token: Callable[[int], None] | None,
        eos_token_id: int | None,
        return_generated_tokens: bool,
    ) -> list[int] | None:
        generated: list[int] = []

        for i in range(num_prefill + max_new_tokens - 1):
            if i < num_prefill:
                token_id = prompt_token_ids[i]
            elif generated:
                token_id = generated[-1]
            else:
                token_id = 0

            self.pipeline.write_token(self._make_token_tensor(token_id))
            out_token = self.pipeline.recv_output_token()
            logger.debug(f"Received token {out_token} at iteration {i}")

            if i >= num_prefill - 1:
                generated.append(out_token)
                if on_token is not None:
                    on_token(out_token)
                if eos_token_id is not None and out_token == eos_token_id:
                    logger.debug("EOS token {} at iteration {}", eos_token_id, i)
                    break

        logger.debug("Generation complete ({} tokens generated)", len(generated))
        if return_generated_tokens:
            return generated
        return None

    def _run_output_rank(
        self,
        num_prefill: int,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> None:
        for i in range(num_prefill + max_new_tokens - 1):
            out_token = self.pipeline.read_and_send_output_token(self._d2h_output_tensor)
            if i >= num_prefill - 1 and eos_token_id is not None and out_token == eos_token_id:
                break

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
