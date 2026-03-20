# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import safe_open

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
        self._cache_path = cache_path
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

        logger.info("Building pipeline")
        self.pipeline = config.build_pipeline(self.mesh_device)

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        my_rank = self.pipeline.my_mesh_id
        prev_rank = my_rank - 1 if my_rank > 0 else None
        next_rank = my_rank + 1 if my_rank < num_procs - 1 else None
        has_logits = self.pipeline.get_logits_tensor() is not None
        is_last = my_rank == num_procs - 1
        self.model = DeepSeekV3(
            write_fn=self.pipeline.write_token,
            read_fn=self.pipeline.read_output,
            batch_size=1,
            prev_rank=prev_rank,
            next_rank=next_rank,
            outputs_tokens=has_logits or is_last,
        )
        logger.info(f"Created ModelPipeline for mesh id {my_rank} (prev={prev_rank}, next={next_rank}).")

    def prefill_forward(
        self,
        tokens: list[int] | None,
        trace_dir: str | Path | None = None,
        trace_start_layer: int = 3,
        logits_file: str | Path | None = None,
        save_logits_path: str | Path | None = None,
        save_outputs_dir: str | Path | None = None,
        pcc_threshold: float = 0.90,
    ) -> dict | None:
        """Prefill with real pipeline token flow and optional per-stage validation.

        Tokens flow through all stages via MPI send/recv. When trace_dir is
        provided, each stage compares its output against reference after each
        token (embedding/passthrough stages are skipped).
        """
        assert self.model is not None

        prompt_token_tensors = None
        num_iterations = None
        if self.pipeline.my_mesh_id == 0:
            assert tokens is not None
            print(f"Prefilling with {len(tokens)} tokens...")
            num_iterations = len(tokens)
            prompt_token_tensors = [
                to_padded_input(
                    torch.tensor([[tid]], dtype=torch.int32),
                    batch_size=1,
                    page_size_datums=self._page_size_datums,
                )
                for tid in tokens
            ]
            ttnn.send_token(num_iterations, ttnn.Rank(1))
        else:
            num_iterations = ttnn.recv_token(ttnn.Rank(self.pipeline.my_mesh_id - 1))
            print(f"Stage {self.pipeline.my_mesh_id} received num_iterations: {num_iterations}")
            num_procs = int(ttnn.distributed_context_get_size())
            if self.pipeline.my_mesh_id < num_procs - 1:
                ttnn.send_token(num_iterations, ttnn.Rank(self.pipeline.my_mesh_id + 1))

        has_logits = self.pipeline.get_logits_tensor() is not None
        is_first_stage = self.pipeline.my_mesh_id == 0
        num_procs = int(ttnn.distributed_context_get_size())
        is_last_stage = self.pipeline.my_mesh_id == num_procs - 1
        is_decoder_stage = not is_first_stage and not has_logits and not is_last_stage
        ref_hs = None
        ref_lg = None
        li = None

        if trace_dir is not None:
            trace_file = str(Path(trace_dir) / "hidden_states.safetensors")
            if is_first_stage:
                if tokens is not None:
                    emb_path = Path(self._cache_path) / "embedding" / "embedding.tensorbin"
                    logger.info(f"Stage 0: looking for embedding at {emb_path} (exists={emb_path.exists()})")
                    if emb_path.exists():
                        logger.info("Stage 0: loading embedding tensorbin from host...")
                        emb_tt = ttnn.load_tensor(str(emb_path))
                        logger.info(f"Stage 0: loaded ttnn tensor, extracting first shard...")
                        emb_first = ttnn.get_device_tensors(emb_tt)[0]
                        emb_host = ttnn.to_torch(emb_first).float()
                        logger.info(f"Stage 0: torch tensor shape={list(emb_host.shape)}, dtype={emb_host.dtype}")
                        while emb_host.dim() > 2:
                            emb_host = emb_host[0]
                        logger.info(f"Stage 0: squeezed to shape={list(emb_host.shape)}")
                        ref_hs = torch.stack([emb_host[tid] for tid in tokens])
                        logger.info(f"Stage 0: built ref_hs shape={list(ref_hs.shape)} for tokens {tokens}")
                        li = "embedding"
                        logger.info(
                            f"Stage 0: embedding validation ready "
                            f"(weight shape={list(emb_host.shape)}, {len(tokens)} tokens)"
                        )
                    else:
                        logger.info(f"Stage 0: embedding weight file not found at {emb_path}, skipping validation")
                else:
                    logger.info(f"Stage {self.pipeline.my_mesh_id}: skipping validation (embedding, no tokens)")
            elif has_logits:
                li = 60
                if logits_file is not None:
                    with safe_open(str(logits_file), framework="pt") as f:
                        ref_lg = f.get_tensor("logits")
                    logger.info(
                        f"Stage {self.pipeline.my_mesh_id}: LM head validation (ref logits {list(ref_lg.shape)})"
                    )
            elif is_decoder_stage:
                li = self.pipeline.my_mesh_id - 1
                with safe_open(trace_file, framework="pt") as f:
                    ref_hs = f.get_tensor(f"decoder_output_layer_{li}")
                logger.info(f"Stage {self.pipeline.my_mesh_id}: decoder validation for layer {li}")
            else:
                logger.info(f"Stage {self.pipeline.my_mesh_id}: skipping validation (embedding/passthrough)")

        self.model.prefill(
            prompt_token_tensors,
            num_iterations=num_iterations,
            ref_hidden_states=ref_hs,
            ref_logits=ref_lg,
            layer_idx=li,
            logits_tensor_fn=self.pipeline.get_logits_tensor if has_logits else None,
            mesh_device=self.mesh_device if has_logits else None,
            save_outputs_dir=save_outputs_dir if trace_dir is not None else None,
            pcc_threshold=pcc_threshold,
        )
        return None

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id."""
        # Host-side model interface is only invoked on mesh id 0
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("decode_forward() should only be called on mesh id 0")
        assert self.model is not None
        output = self.model.decode_step(
            torch.tensor([[input_token]], dtype=torch.int32),
        )
        next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
        return next_token_id

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        return_generated_tokens: bool = False,
        trace_dir: str | Path | None = None,
        trace_start_layer: int = 3,
        logits_file: str | Path | None = None,
        save_logits_path: str | Path | None = None,
        save_outputs_dir: str | Path | None = None,
        pcc_threshold: float = 0.90,
    ) -> list[int] | None:
        """Run full inference: prefill the prompt then decode until EOS or max_new_tokens.
        Calls on_token(token_id) for each generated token (including the first
        one sampled after prefill). Optionally returns the list of all generated token IDs.

        When trace_dir is provided, runs trace validation instead of normal prefill.
        """
        result = self.prefill_forward(
            prompt_token_ids,
            trace_dir=trace_dir,
            trace_start_layer=trace_start_layer,
            logits_file=logits_file,
            save_logits_path=save_logits_path,
            save_outputs_dir=save_outputs_dir,
            pcc_threshold=pcc_threshold,
        )
        if trace_dir is not None:
            return result
        return

        if on_token is not None:
            on_token(next_token_id)
        if return_generated_tokens:
            generated_tokens = [next_token_id]

        # Generation loop: feed y[t], get output, sample y[t+1].
        num_decode_steps = 0
        for i in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token_id == eos_token_id:
                logger.debug("EOS token {} at decode step {}", eos_token_id, i)
                break
            next_token_id = self.decode_forward(next_token_id)
            num_decode_steps += 1
            if on_token is not None:
                on_token(next_token_id)
            if return_generated_tokens:
                generated_tokens.append(next_token_id)
            logger.debug("Decode step {} output token: {}", i + 1, next_token_id)

        logger.debug("Generation complete ({} tokens generated)", 1 + num_decode_steps)
        if return_generated_tokens:
            return generated_tokens

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
