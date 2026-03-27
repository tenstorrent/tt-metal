# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input


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

        flash_mla_program_config = FlashMLADecode.ProgramConfig
        assert (
            self.mesh_device.shape[0] == flash_mla_program_config.sp_dim
            and self.mesh_device.shape[1] == flash_mla_program_config.tp_dim
        )

        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs == 1:
            return
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
        self.model: DeepSeekV3 | None = None
        if self.pipeline.my_mesh_id == 0:
            # Initialize host-side model interface for mesh id 0 (first stage)
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
            )
        logger.info(f"Created ModelPipeline for mesh id {self.pipeline.my_mesh_id}.")

    def get_kv_cache_metadata(
        self, layer_id: int, pos_id: int, slot_id: int, ttnn_kv_cache_tensor: ttnn.Tensor = None
    ) -> dict:
        """
        Get KV cache metadata.
        """
        # kv_cache_tensor should come from the pipeline, but for testing pass in a tensor

        if ttnn_kv_cache_tensor is None:
            raise ValueError("ttnn_kv_cache_tensor is required")

        # Row 0: SP0.TP0, SP0.TP1
        fabric_id_sp0_tp0 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
        fabric_id_sp0_tp1 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, 1))
        # Row 1: SP1.TP0, SP1.TP1
        fabric_id_sp1_tp0 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1, 0))
        fabric_id_sp1_tp1 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(1, 1))
        # Row 2: SP2.TP0, SP2.TP1
        fabric_id_sp2_tp0 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(2, 0))
        fabric_id_sp2_tp1 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(2, 1))
        # Row 3: SP3.TP0, SP3.TP1
        fabric_id_sp3_tp0 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(3, 0))
        fabric_id_sp3_tp1 = self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(3, 1))

        sp_idx_to_fabric_ids = {
            0: [fabric_id_sp0_tp0, fabric_id_sp0_tp1],
            1: [fabric_id_sp1_tp0, fabric_id_sp1_tp1],
            2: [fabric_id_sp2_tp0, fabric_id_sp2_tp1],
            3: [fabric_id_sp3_tp0, fabric_id_sp3_tp1],
        }

        tokens_per_kv_chunk = ttnn_kv_cache_tensor.get_tile().tile_shape[0]
        k_tile_size = ttnn_kv_cache_tensor.get_tile().get_tile_size(ttnn_kv_cache_tensor.dtype)
        k_base_addr = ttnn_kv_cache_tensor.buffer_address()

        kv_cache_shape = ttnn_kv_cache_tensor.shape
        max_seq_len = kv_cache_shape[2]
        max_kv_cache_slots = kv_cache_shape[0]

        kv_cache_dim = kv_cache_shape[3]
        print(f"kv_cache_shape: {kv_cache_shape}")
        print(f"kv_cache_seq_len: {max_seq_len}")
        print(f"kv_cache_dim: {kv_cache_dim}")
        flash_mla_program_config = FlashMLADecode.ProgramConfig(k_chunk_size=128)
        flash_mla_optimal_grid = flash_mla_program_config.grid

        assert max_seq_len == flash_mla_program_config.max_seq_len, "KV cache sequence length must match max seq len"
        assert (
            max_kv_cache_slots == flash_mla_program_config.max_kv_cache_slots
        ), "KV cache slots must match max kv cache slots"

        # a bit confusing, the k_chunk_size is the number of tokens for a block of compute
        # for migration purposes, we call a kv_chunk the 32x576 unit to transfer

        tokens_per_device = max_seq_len // flash_mla_program_config.k_chunk_size
        kv_cache_slot_size_device = tokens_per_device * k_tile_size // tokens_per_kv_chunk
        print(f"kv_cache_slot_size_device: {kv_cache_slot_size_device}")
        print(f"tokens_per_device: {tokens_per_device}")
        print(f"k_tile_size: {k_tile_size}")
        print(f"tokens_per_kv_chunk: {tokens_per_kv_chunk}")
        return {
            "noc_addr": self._position,
            "fabric_node_ids": self._output_buffer,
            "kv_chunk_size": k_tile_size,
            "num_tokens_in_chunk": tokens_per_kv_chunk,
        }

    def prefill_forward(self, tokens: list[int]) -> int:
        """Prefill 1 user's prompt tokens and return the next token id."""
        # Host-side model interface is only invoked on mesh id 0
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        assert self.model is not None
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        prompt_token_tensors = [
            to_padded_input(
                torch.tensor([[tid]], dtype=torch.int32),
                batch_size=1,
                page_size_datums=self._page_size_datums,
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
    ) -> list[int] | None:
        """Run full inference: prefill the prompt then decode until EOS or max_new_tokens.
        Calls on_token(token_id) for each generated token (including the first
        one sampled after prefill). Optionally returns the list of all generated token IDs.
        """
        if self.pipeline.my_mesh_id != 0:
            raise RuntimeError("run_inference() should only be called on mesh id 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"

        # Prefill: send prompt tokens; discard outputs for i < S-1; use last output to sample y0.
        next_token_id = self.prefill_forward(prompt_token_ids)
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
