# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import create_pipeline_configuration_from_num_procs
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    StateDictWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.model import (
    TOKEN_ID_BYTES,
    DecodeResult,
    DeepSeekV3,
    TokenType,
    page_size_bytes,
    to_spec_input,
)


class ModelPipeline:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        weights_mode: Literal["synthetic", "real", "state_dict"] = "real",
        cache_path: Path | None = None,
        model_path: Path | None = None,
        lm_head_fp32_dest_acc_en: bool = True,
        lm_head_persistent_mode: bool = True,
        dense_layer_id_override: int | None = None,
        moe_layer_id_override: int | None = None,
        io_socket_descriptor_prefix: str | None = None,
    ):
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline (weights={}, lm_head_fp32={}, lm_head_persistent_mode={})",
            weights_mode,
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
        if weights_mode == "real":
            if cache_path is None:
                raise ValueError("weights_mode='real' requires cache_path")
            provider: WeightProvider = CacheWeightProvider(cache_path)
        elif weights_mode == "state_dict":
            if model_path is None:
                raise ValueError("weights_mode='state_dict' requires model_path")
            provider = StateDictWeightProvider(model_path)
        elif weights_mode == "synthetic":
            provider = SyntheticWeightProvider()
        else:
            raise ValueError(f"Unknown weights_mode: {weights_mode!r}")
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            enable_mtp=True,
        )
        if config.num_stages != num_procs:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes")

        logger.info("Building pipeline")
        self.pipeline = config.build_pipeline(self.mesh_device)

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        self.model: DeepSeekV3 | None = None
        if self.pipeline.my_mesh_id == 0:
            # Initialize host-side model interface for mesh id 0 (first stage)
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
            )

            # if io_socket_descriptor_prefix is not None:
            #     self.pipeline.export_host_socket_descriptors(io_socket_descriptor_prefix)

        logger.info(f"Created ModelPipeline for mesh id {self.pipeline.my_mesh_id}.")

    def prefill_forward(self, tokens: list[int], slot_id: int) -> list[DecodeResult]:
        """Prefill prompt tokens and return the DecodeResults from the last prompt token's outputs."""
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        assert self.model is not None
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        prompt_token_tensors = [
            to_spec_input(
                tid, user_id=slot_id, position_id=i, page_size_datums=self._page_size_datums, token_type=TokenType.BASE
            )
            for i, tid in enumerate(tokens)
        ]
        results = self.model.prefill(prompt_token_tensors, slot_id)
        logger.debug(f"Done prefilling with {len(tokens)} tokens.")
        return results

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id (legacy non-speculative path)."""
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("decode_forward() should only be called on mesh id 0")
        assert self.model is not None

        output = self.model.decode_step(
            to_spec_input(input_token, user_id=0, position_id=self.position_id, page_size_datums=self._page_size_datums)
        )
        self.position_id += 1

        next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
        return next_token_id

    def _write_spec_pair(self, token_0: int, pos_0: int, token_1: int, pos_1: int, user_id: int = 0) -> None:
        """Write two tokens (base + speculation) into the pipeline."""
        assert self.model is not None
        self.model.write_input(token_0, user_id, pos_0, token_type=TokenType.BASE)
        self.model.write_input(token_1, user_id, pos_1, token_type=TokenType.SPEC)

    def run_inference(
        self,
        prompt_token_ids: list[list[int]],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run speculative-decode inference: prefill then decode with multi-token prediction.

        The pipeline produces structured 2-token output pages (base + speculation).
        Each output page carries position IDs that the host uses directly for
        write-back, so no manual position tracking is needed.

        The host drives the ACCEPT / REJECT / STALE / CONTINUE state machine:
          - ACCEPT:   emit confirmed token, then read CONTINUE.
          - CONTINUE: emit bonus token, write tokens at device-supplied positions.
          - REJECT:   emit base output, write tokens at device-supplied positions.
          - STALE:    discard and re-read.
        """
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("run_inference() should only be called on mesh id 0")
        assert self.model is not None
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"

        generated_tokens: list[list[int]] = [[] for _ in range(2)]
        verified_spec_tokens: list[int] = []
        unverified_spec_tokens: list[int] = []

        def is_eos(token_id: int) -> bool:
            """Returns True if a token is the EOS token"""
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int, slot_id: int) -> None:
            """Emit a token to the caller"""
            if on_token is not None:
                on_token(token_id)
            generated_tokens[slot_id].append(token_id)

        for slot_id in range(2):
            prefill_results = self.prefill_forward(prompt_token_ids[slot_id], slot_id)
            pending: deque[DecodeResult] = deque(prefill_results)
            self.position_id = self.model._position[slot_id]
            # --- Speculative decode state machine --------------------------------
            while len(generated_tokens[slot_id]) < max_new_tokens:
                result = pending.popleft() if pending else self.model.read_result()

                print("Got MD from Device: ")
                print(f"Token 0 Pos: {result.token_0_pos}, Token 1 Pos: {result.token_1_pos}")
                print(f"Token 0 Type: {result.token_0_type}, Token 1 Type: {result.token_1_type}")
                print(f"Token 0: {result.token_0}, Token 1: {result.token_1}")
                print(f"Slot ID: {result.slot_id}")

                emit(result.token_1, slot_id)

                if is_eos(result.token_1) or len(generated_tokens[slot_id]) >= max_new_tokens:
                    break
                self.model.write_input(result.token_1, result.slot_id, self.position_id, token_type=TokenType.BASE)
                self.position_id += 1

                logger.debug("Generation complete ({} tokens generated)", len(generated_tokens[slot_id]))

        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
