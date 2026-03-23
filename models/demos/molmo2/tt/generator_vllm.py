# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for Molmo2-8B model.

This module provides the vLLM-compatible wrapper class for Molmo2-8B,
enabling integration with tt-inference-server via the vLLM plugin.
"""

from typing import List, Mapping, Optional, Tuple, Union

import torch
from loguru import logger
from PIL.Image import Image
from tqdm import tqdm
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.multimodal.profiling import BaseDummyInputsBuilder

import ttnn
from models.demos.molmo2.demo.demo import (
    Molmo2Generator,
    create_model,
    load_model_weights,
    load_processor,
    preprocess_image_molmo2,
)
from models.demos.molmo2.tt.model_config import Molmo2ModelArgs


def allocate_molmo2_kv_cache(
    kv_cache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    num_layers: int,
    mesh_device: ttnn.MeshDevice,
    tt_cache_path: str,
) -> List[List[ttnn.Tensor]]:
    """
    Allocate vLLM-style KV cache for Molmo2 text model.

    Args:
        kv_cache_shape: Shape of each KV cache tensor (num_blocks, num_kv_heads, block_size, head_size)
        dtype: Data type for KV cache
        num_layers: Number of transformer layers
        mesh_device: TT mesh device
        tt_cache_path: Path for caching TT tensors

    Returns:
        List of [K cache, V cache] pairs for each layer
    """
    kv_cache = []
    cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)

    for layer_num in tqdm(range(num_layers), desc="Allocating TT KV caches for Molmo2"):
        kv_tt_i = [
            ttnn.as_tensor(
                cache_kv,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,  # Molmo2 uses bfloat16 for KV cache
                cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
            )
            for kv in ["k", "v"]
        ]
        kv_cache.append(kv_tt_i)

    return [kv_cache]  # Wrap in list for data parallel compatibility


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder):
    """
    Dummy inputs builder for Molmo2 multimodal processor registration.

    Note: We don't do profiling in vLLM for TT devices, so these methods
    raise NotImplementedError.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        raise NotImplementedError("Molmo2 dummy text generation not implemented")

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        raise NotImplementedError("Molmo2 dummy multimodal data generation not implemented")


class Molmo2ForConditionalGeneration(SupportsMultiModal):
    """
    vLLM-compatible wrapper for Molmo2-8B vision-language model.

    This class provides the interface expected by vLLM's TT plugin for:
    - Model initialization via `initialize_vllm_model`
    - KV cache allocation via `allocate_kv_cache`
    - Prefill and decode forward passes
    """

    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,  # Vision models typically don't support prefix caching
    }

    # Molmo2-specific constants
    MOLMO2_IMAGE_TOKEN_ID = 151938  # <im_patch> token

    def __init__(
        self,
        model: "Molmo2Model",
        model_args: Molmo2ModelArgs,
        mesh_device: ttnn.MeshDevice,
        tokenizer,
        generator: Molmo2Generator,
    ):
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.tokenizer = tokenizer
        self.generator = generator
        self.max_gen_len = model_args.max_seq_len - 1

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: Optional[str] = None,
    ):
        """
        Initialize Molmo2-8B for vLLM inference.

        Args:
            hf_config: HuggingFace model config
            mesh_device: TT mesh device
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            tt_data_parallel: Data parallel factor
            optimizations: Optimization mode (not used for Molmo2)

        Returns:
            Initialized Molmo2ForConditionalGeneration instance
        """
        logger.info(f"Initializing Molmo2-8B for vLLM with max_batch_size={max_batch_size}, max_seq_len={max_seq_len}")

        # Load tokenizer
        tokenizer = load_processor()

        # Load model weights
        state_dict = load_model_weights()

        # Create model
        model = create_model(mesh_device, state_dict, num_layers=None)

        # Create model args
        model_args = Molmo2ModelArgs(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Create generator
        generator = Molmo2Generator(
            mesh_device=mesh_device,
            model=model,
            tokenizer=tokenizer,
            num_layers=36,
            batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        logger.info("Molmo2-8B initialized successfully for vLLM")

        return cls(
            model=model,
            model_args=model_args,
            mesh_device=mesh_device,
            tokenizer=tokenizer,
            generator=generator,
        )

    @property
    def cache_path(self):
        """Path for TT tensor caching."""
        return self.model_args.get_text_args().model_cache_path

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        images: Union[List[Image], List[List[Image]]],
        page_table: torch.Tensor,
        kv_cache,
        prompt_lens,
        cross_page_table: Optional[torch.Tensor] = None,
    ):
        """
        Run prefill forward pass for Molmo2.

        Args:
            tokens: Input token IDs [batch, seq_len]
            images: List of PIL images (one per batch item)
            page_table: Page table for paged attention
            kv_cache: KV cache tensors
            prompt_lens: Length of each prompt in the batch
            cross_page_table: Cross-attention page table (not used for Molmo2)

        Returns:
            Logits tensor
        """
        batch_size = tokens.shape[0]

        for user_id in range(batch_size):
            image = images[user_id]
            if isinstance(image, list):
                assert len(image) == 1, "Only one image per prompt is supported"
                image = image[0]

            if image is not None:
                # Preprocess image
                image_inputs = preprocess_image_molmo2(image)

                # Run prefill with image
                logits, _ = self.generator.run_prefill(
                    input_ids=tokens[user_id : user_id + 1, : prompt_lens[user_id]],
                    pixel_values=image_inputs["pixel_values"],
                    pooled_patches_idx=image_inputs["image_token_pooling"].unsqueeze(0),
                    use_trace=True,
                    use_vision_trace=True,
                )
            else:
                # Run prefill without image (text-only)
                logits, _ = self.generator.run_prefill(
                    input_ids=tokens[user_id : user_id + 1, : prompt_lens[user_id]],
                    pixel_values=None,
                    pooled_patches_idx=None,
                    use_trace=True,
                    use_vision_trace=False,
                )

        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache,
        prompt_lens,
    ):
        """
        Run decode forward pass for Molmo2.

        Args:
            tokens: Current token IDs [batch, 1]
            page_table: Page table for paged attention
            kv_cache: KV cache tensors
            prompt_lens: Cumulative sequence lengths

        Returns:
            Logits tensor
        """
        # Convert tokens to device
        token_id_ttnn = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Run decode step
        logits_ttnn, _ = self.generator.run_decode_step(
            token_id_ttnn=token_id_ttnn,
            use_trace=True,
            is_first=False,
        )

        # Convert logits to torch
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)

        return logits

    def allocate_kv_cache(
        self,
        kv_cache_shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_layers: int,
    ) -> List[List[ttnn.Tensor]]:
        """
        Allocate KV cache for Molmo2 text model.

        Args:
            kv_cache_shape: Shape of KV cache tensors
            dtype: Data type for KV cache
            num_layers: Number of transformer layers

        Returns:
            List of KV cache tensor pairs per layer
        """
        return allocate_molmo2_kv_cache(
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers,
            mesh_device=self.mesh_device,
            tt_cache_path=self.cache_path,
        )
