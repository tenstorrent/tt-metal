# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import create_pipeline_configuration_from_num_procs
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    StateDictWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
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
        num_slots: int = 64,
        bspm_dir: Path | None = None,
        bspm_variant: str = "B",
        bspm_budget: float = 3.5,
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
            if model_path is None:
                raise ValueError("weights_mode='real' requires model_path")
            provider: WeightProvider = CacheWeightProvider(
                cache_path,
                model_path,
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
            )
        elif weights_mode == "state_dict":
            if model_path is None:
                raise ValueError("weights_mode='state_dict' requires model_path")
            provider = StateDictWeightProvider(
                model_path,
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
            )
        elif weights_mode == "synthetic":
            provider = SyntheticWeightProvider(
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
            )
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
            num_slots=num_slots,
        )
        if config.num_stages != num_procs:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes")

        logger.info("Building pipeline")
        # Propagate an identical, explicit pipeline_config and stages_metadata
        # to every rank. Without this, each rank independently calls
        # generate_blitz_decode_pipeline() and the submesh-aware code paths
        # added in #42002 can produce inconsistent entry/exit MeshCoordinates
        # across ranks, causing the inter-mesh MeshSocket handshake in
        # SpecEmbeddingPipelineBlock.__init__'s h2d_host_io to time out.
        pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
        stages_metadata = {i: StageMetadata(rank=i, mesh_id=i) for i in range(num_procs)}
        self.pipeline = config.build_pipeline(
            self.mesh_device,
            stages_metadata=stages_metadata,
            pipeline_config=pipeline_config,
        )

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        self.model: DeepSeekV3 | None = None
        if self.pipeline.my_stage_idx == 0:
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
                pipeline_depth=config.num_stages,
            )

            if io_socket_descriptor_prefix is not None:
                self.pipeline.export_host_socket_descriptors(io_socket_descriptor_prefix)

        logger.info(f"Created ModelPipeline for stage {self.pipeline.my_stage_idx}.")

    def prefill_forward(self, tokens: list[int]) -> list[DecodeResult]:
        """Prefill prompt tokens and return the DecodeResults from the last prompt token's outputs."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        assert self.model is not None
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        prompt_token_tensors = [
            to_spec_input(
                tokens[i],
                tokens[i + 1] if i < len(tokens) - 1 else -1,
                user_id=0,
                position_id=i,
                page_size_datums=self._page_size_datums,
                token_type=TokenType.BASE,
            )
            for i in range(len(tokens))
        ]
        results = self.model.prefill(prompt_token_tensors)
        logger.debug(f"Done prefilling with {len(tokens)} tokens.")
        return results

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id."""
        if self.pipeline.my_stage_idx != 0:
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
        self.model.write_input(token_0, -1, user_id, pos_0, token_type=TokenType.BASE)
        self.model.write_input(token_1, -1, user_id, pos_1, token_type=TokenType.SPEC)

    def run_inference(
        self,
        prompt_token_ids: list[int],
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
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("run_inference() should only be called on stage 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"

        generated_tokens: list[int] = []
        verified_spec_tokens: list[int] = []
        unverified_spec_tokens: list[int] = []

        def is_eos(token_id: int) -> bool:
            """Returns True if a token is the EOS token"""
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            """Emit a token to the caller"""
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)

        # --- Prefill --------------------------------------------------------
        prefill_results = self.prefill_forward(prompt_token_ids)

        # Seed the state machine with both pages from the last prefill write,
        # then read from the pipeline for all subsequent results.
        pending: deque[DecodeResult] = deque(prefill_results)
        base_accept = 0
        spec_accept = 0
        base_reject = 0
        spec_reject = 0
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
        # --- Speculative decode state machine --------------------------------
        iteration = 0
        start_time = time.time()
        num_emits = 0
        num_writes = 0
        num_reads = 0
        signal_to_exit = False
        while len(generated_tokens) < max_new_tokens or signal_to_exit:
            iteration += 1
            # logger.debug(
            #     f"\n\nIteration {iteration}: Base Accept: {base_accept}, Base Reject: {base_reject}, Spec Accept: {spec_accept}, Spec Reject: {spec_reject}, Base Accept Rate: {base_accept / (base_accept + base_reject + 1e-5)}, Spec Accept Rate: {spec_accept / (spec_accept + spec_reject + 1e-5)}"
            # )

            if pending:
                result = pending.popleft()
            else:
                result = self.model.read_result()
                num_reads += 1

            # logger.debug("Got MD from Device: ")
            # logger.debug(f"Token 0 Pos: {result.token_0_pos}, Token 1 Pos: {result.token_1_pos}")
            # logger.debug(f"Token 0 Type: {result.token_0_type}, Token 1 Type: {result.token_1_type}")
            # logger.debug(
            #     f"Token 0: {tokenizer.decode([result.token_0], skip_special_tokens=False)}, Token 1: {tokenizer.decode([result.token_1], skip_special_tokens=False)}"
            # )
            # logger.debug(f"Slot ID: {result.slot_id}")

            if not unverified_spec_tokens and not verified_spec_tokens:
                unverified_spec_tokens.append(result.token_1)
                emit(result.token_0)
                num_emits += 1
                # logger.debug("Prefill done")
            else:
                if result.token_0_type == TokenType.BASE:
                    # On acceptance, we check that the base token matches the first token of the last unverified spec token
                    if result.token_0 == unverified_spec_tokens[-1]:
                        verified_spec_tokens.append(unverified_spec_tokens.pop())
                        emit(result.token_0)
                        base_accept += 1
                        # logger.debug("Base Accept")
                        num_emits += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens
                        continue
                    # On rejection, we discard the last unverified spec token and populate the new spec token
                    else:
                        unverified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)
                        emit(result.token_0)
                        base_reject += 1
                        # logger.debug("Base Reject")
                        num_emits += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens

                if result.token_0_type == TokenType.SPEC:
                    # If we have a verified spec token it means we have an acceptance case, remove it and emit the token
                    if verified_spec_tokens:
                        verified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)

                        if signal_to_exit:
                            break

                        emit(result.token_0)
                        spec_accept += 1
                        num_emits += 1
                        # logger.debug("Spec Accept")
                    else:
                        # logger.debug("Spec Reject")
                        if signal_to_exit:
                            break
                        spec_reject += 1
                        continue

            if is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens:
                break

            self._write_spec_pair(
                result.token_0,
                result.token_0_pos,
                result.token_1,
                result.token_1_pos,
            )
            num_writes += 2

        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        end_time = time.time()
        logger.debug(f"Time taken: {end_time - start_time} seconds")
        logger.debug(f"Tokens per second: {num_emits / (end_time - start_time)}")
        logger.debug(
            f"Base Accept: {base_accept}, Base Reject: {base_reject}, Spec Accept: {spec_accept}, Spec Reject: {spec_reject}, Base Accept Rate: {base_accept / (base_accept + base_reject + 1e-5)}, Spec Accept Rate: {spec_accept / (spec_accept + spec_reject + 1e-5)}"
        )
        logger.debug("Generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def dump_kv_cache(self, out_dir) -> None:
        self.pipeline.dump_kv_cache(out_dir)

    def dump_per_token_outputs(self, out_dir) -> None:
        self.pipeline.dump_per_token_outputs(out_dir)

    def run_inference_with_capture(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None,
        capture_dir: Path,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run inference with per-token snapshot of decoder outputs. ALL ranks must call this.

        Driver rank (stage 0) runs the speculative-decode loop with embedded
        ``barrier+snapshot+barrier`` at each iteration; non-driver ranks run a parallel
        snapshot-only loop. Coordination via ``ttnn.distributed_context_barrier()`` and a sentinel
        file in ``capture_dir`` (must be on shared filesystem).
        """
        is_driver = self.pipeline.my_stage_idx == 0
        capture_dir = Path(capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        sentinel = capture_dir / ".keep_capturing"

        if is_driver:
            sentinel.touch()

        ttnn.distributed_context_barrier()

        if is_driver:
            result = self._driver_loop_with_capture(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                sentinel=sentinel,
            )
            return result if return_generated_tokens else None
        else:
            self._capture_only_loop(sentinel)
            return None

    def _capture_only_loop(self, sentinel: Path) -> None:
        iter_idx = 0
        stage = self.pipeline.my_stage_idx
        while True:
            ttnn.distributed_context_barrier()  # barrier1
            if not sentinel.exists():
                logger.info(f"[stage={stage}] capture done at iter={iter_idx}")
                break
            self.pipeline.snapshot_outputs(iter_idx)
            ttnn.distributed_context_barrier()  # barrier2
            iter_idx += 1

    def _driver_loop_with_capture(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None,
        sentinel: Path,
    ) -> list[int]:
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("_driver_loop_with_capture should only run on stage 0")
        assert max_new_tokens >= 1

        generated_tokens: list[int] = []
        verified_spec_tokens: list[int] = []
        unverified_spec_tokens: list[int] = []

        def is_eos(token_id: int) -> bool:
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            generated_tokens.append(token_id)

        prefill_results = self.prefill_forward(prompt_token_ids)
        pending: deque[DecodeResult] = deque(prefill_results)

        iter_idx = 0
        signal_to_exit = False
        num_writes = 0
        num_reads = 0

        def do_capture(idx: int) -> None:
            ttnn.distributed_context_barrier()
            self.pipeline.snapshot_outputs(idx)
            ttnn.distributed_context_barrier()

        try:
            while len(generated_tokens) < max_new_tokens or signal_to_exit:
                if pending:
                    result = pending.popleft()
                else:
                    result = self.model.read_result()
                    num_reads += 1

                do_capture(iter_idx)
                iter_idx += 1

                if not unverified_spec_tokens and not verified_spec_tokens:
                    unverified_spec_tokens.append(result.token_1)
                    emit(result.token_0)
                else:
                    if result.token_0_type == TokenType.BASE:
                        if result.token_0 == unverified_spec_tokens[-1]:
                            verified_spec_tokens.append(unverified_spec_tokens.pop())
                            emit(result.token_0)
                            signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens
                            continue
                        else:
                            unverified_spec_tokens.pop()
                            unverified_spec_tokens.append(result.token_1)
                            emit(result.token_0)
                            signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens

                    if result.token_0_type == TokenType.SPEC:
                        if verified_spec_tokens:
                            verified_spec_tokens.pop()
                            unverified_spec_tokens.append(result.token_1)
                            if signal_to_exit:
                                break
                            emit(result.token_0)
                        else:
                            if signal_to_exit:
                                break
                            continue

                if is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens:
                    break

                self._write_spec_pair(
                    result.token_0,
                    result.token_0_pos,
                    result.token_1,
                    result.token_1_pos,
                )
                num_writes += 2

            while num_reads < num_writes:
                self.model.read_result()
                num_reads += 1
        finally:
            if sentinel.exists():
                sentinel.unlink()
            ttnn.distributed_context_barrier()

        logger.debug("Capture-mode generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens

    def terminate(self) -> None:
        self.pipeline.terminate()
