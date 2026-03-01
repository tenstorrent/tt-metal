# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B Demo for Tenstorrent Hardware.

This demo showcases multimodal visual question answering using Molmo2-8B
running on Tenstorrent devices (N150/N300/T3K).

Features:
- Vision-language multimodal inference
- KV cache for efficient autoregressive generation
- Optional tracing for improved performance
- Proper warm-up and timing (TTFT, decode throughput)

Usage:
    # Run with default image and prompt
    python -m models.demos.molmo2.demo.demo

    # Run with custom image
    python -m models.demos.molmo2.demo.demo --image path/to/image.jpg

    # Run with tracing enabled
    python -m models.demos.molmo2.demo.demo --use-trace
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import einops
import numpy as np
import torch
import torchvision.transforms
from loguru import logger
from PIL import Image

import ttnn

# Default paths
DEMO_DIR = Path(__file__).parent
DEFAULT_IMAGE = DEMO_DIR / "dog.jpg"
MODEL_ID = "allenai/Molmo2-8B"

# Molmo2 image tokens
IMAGE_PATCH_TOKEN = "<im_patch>"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
IM_COL_TOKEN = "<im_col>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
IMAGE_PROMPT = "<|image|>"

# Molmo2 normalization constants (from HuggingFace Molmo2ImageProcessor)
# These differ from standard ImageNet normalization
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]


def load_processor():
    """Load the Molmo2 tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    return tokenizer


def load_model_weights():
    """Load all model weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info(f"Loading model weights from {MODEL_ID}")
    state_dict = load_state_dict_from_safetensors(MODEL_ID)
    logger.info(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


# =============================================================================
# Molmo2 Image Preprocessing (standalone, no video dependencies)
# =============================================================================


def resize_image(image: np.ndarray, target_size: list, resample=None) -> np.ndarray:
    """Resize image using PIL to match HuggingFace preprocessing."""
    # Convert numpy to PIL
    pil_image = Image.fromarray(image.astype(np.uint8))
    # Resize using PIL BILINEAR (same as HuggingFace)
    resized_pil = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    # Convert back to numpy and normalize to [0, 1]
    resized = np.array(resized_pil, dtype=np.float32) / 255.0
    return resized


def normalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Normalize image with ImageNet mean/std."""
    image = image - np.array(mean, dtype=np.float32)[None, None, :]
    image = image / np.array(std, dtype=np.float32)[None, None, :]
    return image


def select_tiling(h: int, w: int, patch_size: int, max_crops: int) -> tuple:
    """Select optimal tiling for image crops."""
    tilings = []
    for i in range(1, max_crops + 1):
        for j in range(1, max_crops + 1):
            if i * j <= max_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))

    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size
    original_size = np.array([h, w], dtype=np.float32)

    with np.errstate(divide="ignore"):
        required_scale = np.min(candidate_resolutions.astype(np.float32) / original_size, axis=-1, keepdims=True)

    if np.all(required_scale < 1):
        ix = np.argmax(required_scale)
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)

    return tuple(candidate_tilings[ix])


def build_overlapping_crops(
    image: np.ndarray,
    max_crops: int,
    overlap_margins: list,
    base_size: int,
    patch_size: int,
) -> tuple:
    """Build overlapping crops from image."""
    left_margin, right_margin = overlap_margins
    total_margin = patch_size * (left_margin + right_margin)
    crop_patches = base_size // patch_size
    crop_window_patches = crop_patches - (left_margin + right_margin)
    crop_window_size = crop_window_patches * patch_size

    h, w = image.shape[:2]
    tiling = select_tiling(h - total_margin, w - total_margin, crop_window_size, max_crops)

    # Resize to fit tiling
    target_h = tiling[0] * crop_window_size + total_margin
    target_w = tiling[1] * crop_window_size + total_margin
    src = resize_image(image, [target_h, target_w], torchvision.transforms.InterpolationMode.BILINEAR)
    src = normalize_image(src, IMAGENET_MEAN, IMAGENET_STD)

    # Extract crops
    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, base_size, base_size, 3], dtype=src.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patches, crop_patches], dtype=np.int32)

    on_crop = 0
    for i in range(tiling[0]):
        y0 = i * crop_window_size
        for j in range(tiling[1]):
            x0 = j * crop_window_size
            crop_arr[on_crop] = src[y0 : y0 + base_size, x0 : x0 + base_size]

            patch_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
            patch_idx = patch_idx + on_crop * crop_patches * crop_patches

            # Mask overlap regions
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0] - 1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -right_margin:] = -1

            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # Reorder patch indices
    patch_idx_arr = patch_idx_arr.reshape(tiling[0], tiling[1], crop_patches, crop_patches)
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3]).reshape(-1)
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(src.shape[0] // patch_size, src.shape[1] // patch_size)

    return crop_arr, patch_idx_arr, tiling


def arange_for_pooling(idx_arr: np.ndarray, pool_h: int, pool_w: int) -> np.ndarray:
    """Arrange indices for pooling."""
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]], constant_values=-1)
    return einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def preprocess_image_molmo2_simple(
    image_path: str,
    base_size: int = 378,
    patch_size: int = 14,
    pooling_size: list = None,
) -> dict:
    """
    Simple image preprocessing for Molmo2 - just resize to base_size.

    This is a simplified version that skips multi-crop processing.
    Just resizes the image to 378x378 and computes pooling indices.

    Returns:
        Dict with pixel_values, image_token_pooling, image_grids, and image_num_crops
    """
    if pooling_size is None:
        pooling_size = [2, 2]

    pool_h, pool_w = pooling_size
    crop_patches = base_size // patch_size  # 27

    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Resize to base_size
    resized = resize_image(image_np, [base_size, base_size], torchvision.transforms.InterpolationMode.BILINEAR)
    resized = normalize_image(resized, IMAGENET_MEAN, IMAGENET_STD)

    # Create pooling indices for 2x2 pooling
    # Original patch grid: 27x27 = 729 patches
    # After 2x2 pooling: 14x14 = 196 output positions (with some padding)
    resize_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape(-1, pool_h * pool_w)

    # Convert to tensors
    # pixel_values: [1, C, H, W] for single image
    pixel_values = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()

    image_token_pooling = torch.from_numpy(resize_idx).long()
    # For simple single-image mode, high-res grid is same as low-res (no extra crops)
    image_grids = torch.tensor([[resized_h, resized_w, 0, 0]])
    image_num_crops = torch.tensor([1])

    return {
        "pixel_values": pixel_values,  # [1, 3, H, W] for ViT
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
    }


def preprocess_image_molmo2(
    image_path: str,
    base_size: int = 378,
    patch_size: int = 14,
    max_crops: int = 8,
    overlap_margins: list = None,
    pooling_size: list = None,
    use_simple: bool = True,  # Default to simple mode for now
) -> dict:
    """
    Preprocess image for Molmo2.

    Args:
        use_simple: If True, use simple resize-only preprocessing (faster, works with current implementation).
                   If False, use full multi-crop preprocessing (requires batch processing in ViT).

    Returns:
        Dict with pixel_values, image_token_pooling, image_grids, and image_num_crops
    """
    if use_simple:
        return preprocess_image_molmo2_simple(image_path, base_size, patch_size, pooling_size)

    # Full multi-crop preprocessing (for reference)
    if overlap_margins is None:
        overlap_margins = [4, 4]
    if pooling_size is None:
        pooling_size = [2, 2]

    pool_h, pool_w = pooling_size
    crop_patches = base_size // patch_size

    # Load and convert image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Build overlapping crops
    crop_arr, patch_idx_arr, tiling = build_overlapping_crops(
        image_np, max_crops, overlap_margins, base_size, patch_size
    )

    # Pooling indices for high-res crops
    pooling_idx = arange_for_pooling(patch_idx_arr, pool_h, pool_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape(-1, pool_h * pool_w)

    # Build low-res (global) image
    resized = resize_image(image_np, [base_size, base_size], torchvision.transforms.InterpolationMode.BILINEAR)
    resized = normalize_image(resized, IMAGENET_MEAN, IMAGENET_STD)
    resized = np.expand_dims(resized, 0)

    resize_idx = np.arange(crop_patches * crop_patches).reshape(crop_patches, crop_patches)
    resize_idx = arange_for_pooling(resize_idx, pool_h, pool_w)
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape(-1, pool_h * pool_w)

    # Combine: global image first, then crops
    # all_crops shape: [n_crops, H, W, C] (H, W = base_size)
    all_crops = np.concatenate([resized, crop_arr], axis=0)

    # Adjust pooling indices (global image patches come first)
    pooling_idx = np.where(pooling_idx >= 0, pooling_idx + crop_patches * crop_patches, -1)
    pooling_idx = np.concatenate([resize_idx, pooling_idx], axis=0)

    # Convert crops to [n_crops, C, H, W] format for ViT
    # all_crops is [n_crops, H, W, C] -> [n_crops, C, H, W]
    pixel_values = torch.from_numpy(all_crops).permute(0, 3, 1, 2).float()

    image_token_pooling = torch.from_numpy(pooling_idx).long()
    image_grids = torch.tensor([[resized_h, resized_w, h, w]])
    image_num_crops = torch.tensor([all_crops.shape[0]])

    return {
        "pixel_values": pixel_values,  # [n_crops, 3, H, W] for ViT
        "image_token_pooling": image_token_pooling,
        "image_grids": image_grids,
        "image_num_crops": image_num_crops,
    }


def get_image_tokens(image_grid: torch.Tensor, use_col_tokens: bool = True) -> str:
    """Generate image token string from grid dimensions."""
    resized_h, resized_w, h, w = image_grid.tolist()

    # Low-res tokens (always present)
    per_row_low = IMAGE_PATCH_TOKEN * resized_w
    if use_col_tokens:
        per_row_low += IM_COL_TOKEN
    low_res_tokens = LOW_RES_IMAGE_START_TOKEN + (per_row_low * resized_h) + IM_END_TOKEN

    # High-res tokens (only if present)
    if h > 0 and w > 0:
        per_row = IMAGE_PATCH_TOKEN * w
        if use_col_tokens:
            per_row += IM_COL_TOKEN
        high_res_tokens = IM_START_TOKEN + (per_row * h) + IM_END_TOKEN
        return low_res_tokens + high_res_tokens
    else:
        return low_res_tokens


def create_model(mesh_device, state_dict, num_layers: Optional[int] = None):
    """
    Create the Molmo2 TTNN model.

    Args:
        mesh_device: TTNN device or mesh device
        state_dict: Model state dict
        num_layers: Optional number of text layers (default: 36)

    Returns:
        Molmo2Model instance
    """
    from models.demos.molmo2.tt.molmo2_model import Molmo2Model

    logger.info("Creating Molmo2 TTNN model")

    text_num_layers = num_layers if num_layers is not None else 36

    model = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        # Vision config
        vit_num_layers=25,
        vit_hidden_dim=1152,
        vit_intermediate_dim=4304,
        vit_num_heads=16,
        vit_head_dim=72,
        patch_size=14,
        image_size=378,
        feature_layers=(18, 24),
        # Adapter config
        adapter_hidden_dim=1152,
        adapter_intermediate_dim=12288,
        adapter_num_heads=16,
        adapter_head_dim=72,
        # Text config
        text_num_layers=text_num_layers,
        text_hidden_dim=4096,
        text_intermediate_dim=12288,
        text_num_heads=32,
        text_num_kv_heads=8,
        text_head_dim=128,
        vocab_size=152064,
        max_seq_len=8192,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        dtype=ttnn.bfloat8_b,
    )

    logger.info("Model created successfully")
    return model


class Molmo2Generator:
    """
    Molmo2 generator with separate tracing for prefill and decode.

    Tracing captures the computation graph and replays it for improved performance.
    - Prefill trace: processes the full input sequence
    - Decode trace: processes one token at a time with KV cache

    Timing follows simple_text_demo.py pattern:
    - compile_prefill: First prefill run (warm-up)
    - inference_prefill: Actual prefill (TTFT)
    - compile_decode: First decode run (warm-up)
    - inference_decode: Subsequent decode iterations
    """

    def __init__(
        self,
        mesh_device,
        model,
        tokenizer,
        num_layers: int,
        batch_size: int = 1,
        max_seq_len: int = 2048,
    ):
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Separate trace state for prefill and decode
        self.prefill_traces = {}  # {seq_len: (trace_id, trace_inputs, trace_output)}
        self.decode_trace_id = None
        self.decode_trace_tensors = None
        self.decode_trace_output = None

        # KV cache (initialized on first run)
        self.kv_caches = None
        self.current_pos = None
        self.decode_position = 0  # Track position on host for trace updates

        # Mesh mapper
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

    def init_kv_cache(self):
        """Initialize KV cache for generation."""
        from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache

        if self.kv_caches is None:
            self.kv_caches = init_kv_cache(
                mesh_device=self.mesh_device,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                num_kv_heads=8,
                max_seq_len=self.max_seq_len,
                head_dim=128,
                dtype=ttnn.bfloat16,  # Use bfloat16 to match RoPE output dtype
            )
            self.current_pos = init_decode_position(
                mesh_device=self.mesh_device,
                batch_size=self.batch_size,
                initial_pos=0,
            )

    def reset_kv_cache(self, start_pos: int = 0):
        """Reset KV cache position for new generation."""
        # Reset host-side position tracker
        self.decode_position = start_pos

        if self.current_pos is not None:
            pos_tensor = torch.full((self.batch_size,), start_pos, dtype=torch.int32)
            pos_ttnn = ttnn.from_torch(
                pos_tensor,
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(pos_ttnn, self.current_pos)
            ttnn.deallocate(pos_ttnn)

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        Prepare text model inputs by processing vision and fusing embeddings.

        This must be done BEFORE trace capture since it involves host-device transfers.

        Args:
            input_ids: Input token IDs
            pixel_values: Preprocessed image tensor
            pooled_patches_idx: Indices for vision pooling

        Returns:
            Tuple of (hidden_states_ttnn, hidden_states_torch)
        """
        # Process vision and prepare fused embeddings (this has host-device transfers)
        if pixel_values is not None and pooled_patches_idx is not None:
            visual_embeddings = self.model.embed_image(pixel_values, pooled_patches_idx)
            hidden_states = self.model.prepare_inputs_for_multimodal(input_ids, visual_embeddings)
        else:
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)
            hidden_states = (
                ttnn.to_torch(hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0]
                .squeeze(0)
                .squeeze(0)
            )

        # Keep torch version for later use
        hidden_states_torch = hidden_states.unsqueeze(0).unsqueeze(0).clone()

        # Convert to TTNN tensor
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return hidden_states_ttnn, hidden_states_torch

    def _allocate_prefill_trace_tensors(
        self,
        seq_len: int,
        hidden_dim: int = 4096,
    ) -> dict:
        """Pre-allocate all tensors needed for traced prefill."""
        # Allocate hidden states input tensor
        hidden_states_shape = [1, 1, seq_len, hidden_dim]
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape(hidden_states_shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-compute rotation matrices (these will be used during trace)
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

        # Allocate rot_mats tensors (we'll copy into these)
        trace_cos = ttnn.allocate_tensor_on_device(
            rot_mats[0].shape,
            rot_mats[0].dtype,
            rot_mats[0].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_sin = ttnn.allocate_tensor_on_device(
            rot_mats[1].shape,
            rot_mats[1].dtype,
            rot_mats[1].layout,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Copy initial values
        ttnn.copy(rot_mats[0], trace_cos)
        ttnn.copy(rot_mats[1], trace_sin)

        # Clean up temporary rot_mats
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        return {
            "hidden_states": trace_hidden_states,
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
        }

    def _capture_prefill_trace(
        self,
        trace_tensors: dict,
    ) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for text model prefill phase."""
        logger.info("Capturing text model prefill trace...")
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        logits_trace, _ = self.model.text_model.forward(
            hidden_states=trace_tensors["hidden_states"],
            start_pos=0,
            attn_mask=None,
            kv_caches=self.kv_caches,  # Pass KV cache to fill during prefill
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Text model prefill trace captured")

        return trace_id, logits_trace

    def _execute_prefill_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """Execute captured prefill trace with new inputs."""
        # Copy new hidden states to trace input location
        new_hidden = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        ttnn.copy(new_hidden, trace_tensors["hidden_states"])
        ttnn.deallocate(new_hidden)

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _allocate_decode_trace_tensors(self, hidden_dim: int = 4096) -> dict:
        """Allocate tensors needed for traced decode."""
        # Allocate hidden states tensor [1, 1, 1, hidden_dim]
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Allocate position index tensor for traced embedding lookup
        # Allocate position index tensor for embedding lookup
        # This will be updated before each trace execution
        trace_rot_idxs = self.model.text_model.rotary_setup.allocate_decode_rot_idxs(initial_pos=0)

        return {
            "hidden_states": trace_hidden_states,
            "rot_idxs": trace_rot_idxs,
        }

    def _capture_decode_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for decode phase (single token generation).

        The embedding lookup for rot_mats is included in the trace so that
        on execution, we only need to update the rot_idxs tensor.
        """
        logger.info("Capturing decode trace...")

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Embedding lookup for rot_mats - this is part of the traced operations
        # On trace execution, updating rot_idxs will cause this to produce new rot_mats
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(trace_tensors["rot_idxs"])

        logits_trace = self.model.text_model.forward_decode(
            hidden_states=trace_tensors["hidden_states"],
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
            rot_mats=rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Decode trace captured")

        return trace_id, logits_trace

    def _execute_decode_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states: ttnn.Tensor,
        position: int,
    ) -> ttnn.Tensor:
        """Execute captured decode trace with new inputs.

        Uses host tensors and copy_host_to_device_tensor to avoid allocations
        which would be unsafe with an active trace.

        The embedding lookup for rot_mats is part of the trace, so we only
        need to update the rot_idxs tensor here.
        """
        # Copy new hidden states to trace input
        ttnn.copy(hidden_states, trace_tensors["hidden_states"])

        # Update current_pos tensor using HOST tensor pattern (no device allocation)
        # Create host tensor with device=None
        new_pos_host = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            dtype=ttnn.int32,
            device=None,  # HOST tensor - no device allocation
            mesh_mapper=self.mesh_mapper,
        )
        # Copy from host to pre-allocated device tensor
        ttnn.copy_host_to_device_tensor(new_pos_host, self.current_pos)

        # Update rot_idxs tensor (position index for embedding lookup in the trace)
        # The embedding lookup is part of the traced operations, so updating
        # rot_idxs will cause the trace to produce correct rot_mats
        batch = self.batch_size
        pad_size = ((batch + 31) // 32) * 32 - batch
        position_idxs = torch.full((1, batch + pad_size), position, dtype=torch.int32)
        rot_idxs_host = ttnn.from_torch(
            position_idxs,
            dtype=ttnn.uint32,
            device=None,  # HOST tensor - no device allocation
            mesh_mapper=self.mesh_mapper,
        )
        ttnn.copy_host_to_device_tensor(rot_idxs_host, trace_tensors["rot_idxs"])

        # Execute trace - rot_mats embedding lookup happens inside the trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def warmup_prefill(
        self,
        hidden_states_ttnn: ttnn.Tensor,
        trace_tensors: dict,
        use_trace: bool,
    ):
        """Run prefill warm-up (compile) pass."""
        logger.info("Running prefill warm-up (compile)...")
        start = time.perf_counter()

        if use_trace:
            # Copy hidden states to trace tensor
            ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

            # Run forward to compile - MUST pass kv_caches to compile fill_cache ops
            # This matches the tt_transformers pattern where compilation happens with kv_cache
            rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
            logits, _ = self.model.text_model.forward(
                hidden_states=trace_tensors["hidden_states"],
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Pass KV cache to compile fill_cache
                rot_mats=rot_mats,
            )
        else:
            logits, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Also pass KV cache for non-traced warmup
            )

        compile_time = (time.perf_counter() - start) * 1000
        logger.info(f"Prefill compile completed in {compile_time:.2f}ms")
        return compile_time

    def warmup_decode(
        self,
        hidden_states: ttnn.Tensor,
        trace_tensors: dict,
        use_trace: bool,
    ):
        """Run decode warm-up (compile) pass."""
        logger.info("Running decode warm-up (compile)...")
        start = time.perf_counter()

        if use_trace:
            # Copy hidden states to trace tensor
            ttnn.copy(hidden_states, trace_tensors["hidden_states"])

            # Update rot_idxs for current position (matches trace capture pattern)
            batch = self.batch_size
            pad_size = ((batch + 31) // 32) * 32 - batch
            position_idxs = torch.full((1, batch + pad_size), self.decode_position, dtype=torch.int32)
            rot_idxs_host = ttnn.from_torch(
                position_idxs,
                dtype=ttnn.uint32,
                device=None,  # HOST tensor
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy_host_to_device_tensor(rot_idxs_host, trace_tensors["rot_idxs"])

            # Run forward to compile with embedding lookup (same as trace capture)
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(trace_tensors["rot_idxs"])
            logits = self.model.text_model.forward_decode(
                hidden_states=trace_tensors["hidden_states"],
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mats=rot_mats,
            )
        else:
            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
            )

        compile_time = (time.perf_counter() - start) * 1000
        logger.info(f"Decode compile completed in {compile_time:.2f}ms")
        return compile_time

    def run_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        use_trace: bool = False,
    ) -> Tuple[ttnn.Tensor, dict]:
        """
        Run prefill phase (process prompt + image).

        Returns:
            Tuple of (logits, timing_dict)
        """
        # Initialize KV cache if needed
        self.init_kv_cache()

        seq_len = input_ids.shape[1]
        timing = {}

        # Prepare inputs (vision processing) - outside trace
        logger.info("Preparing inputs (vision processing)...")
        vision_start = time.perf_counter()
        hidden_states_ttnn, hidden_states_torch = self._prepare_text_inputs(input_ids, pixel_values, pooled_patches_idx)
        timing["vision_ms"] = (time.perf_counter() - vision_start) * 1000
        logger.info(f"Vision processing completed in {timing['vision_ms']:.2f}ms")

        if use_trace:
            # Allocate trace tensors if needed
            if seq_len not in self.prefill_traces:
                logger.info("Allocating prefill trace tensors...")
                trace_tensors = self._allocate_prefill_trace_tensors(seq_len, hidden_dim=4096)

                # Warm-up (compile)
                timing["compile_prefill_ms"] = self.warmup_prefill(hidden_states_ttnn, trace_tensors, use_trace=True)

                # Capture trace
                trace_id, trace_output = self._capture_prefill_trace(trace_tensors)
                self.prefill_traces[seq_len] = (trace_id, trace_tensors, trace_output)

            trace_id, trace_tensors, trace_output = self.prefill_traces[seq_len]

            # Execute trace (actual TTFT measurement)
            ttft_start = time.perf_counter()
            logits = self._execute_prefill_trace(trace_id, trace_tensors, trace_output, hidden_states_torch)
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

            ttnn.deallocate(hidden_states_ttnn)
        else:
            # Warm-up (compile)
            timing["compile_prefill_ms"] = self.warmup_prefill(hidden_states_ttnn, None, use_trace=False)

            # Actual prefill (TTFT) - pass KV cache to fill during forward
            ttft_start = time.perf_counter()
            logits, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Pass pre-allocated cache to fill
            )
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

        logger.info(f"TTFT: {timing['ttft_ms']:.2f}ms")

        # Update position for decode
        self.reset_kv_cache(seq_len)

        return logits, timing

    def run_decode_step(
        self,
        token_id: int,
        use_trace: bool = False,
        is_first: bool = False,
    ) -> Tuple[ttnn.Tensor, float]:
        """
        Run single decode step.

        Args:
            token_id: Token ID to decode
            use_trace: Whether to use tracing
            is_first: Whether this is the first decode step (for warm-up)

        Returns:
            Tuple of (logits, decode_time_ms)
        """
        # Get current position
        current_pos_value = self.decode_position

        # Create token tensor and get embeddings
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        input_ids_ttnn = ttnn.from_torch(
            token_tensor,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        hidden_states = self.model.text_model.embed_tokens(input_ids_ttnn)

        if use_trace:
            if self.decode_trace_id is None:
                # Allocate trace tensors
                logger.info("Allocating decode trace tensors...")
                self.decode_trace_tensors = self._allocate_decode_trace_tensors(hidden_dim=4096)

                # Warm-up (compile)
                compile_time = self.warmup_decode(hidden_states, self.decode_trace_tensors, use_trace=True)

                # Capture trace
                trace_id, trace_output = self._capture_decode_trace(self.decode_trace_tensors)
                self.decode_trace_id = trace_id
                self.decode_trace_output = trace_output

            # Execute trace (actual decode)
            start_time = time.perf_counter()
            logits = self._execute_decode_trace(
                self.decode_trace_id,
                self.decode_trace_tensors,
                self.decode_trace_output,
                hidden_states,
                current_pos_value,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000
        else:
            if is_first:
                # Warm-up (compile)
                compile_time = self.warmup_decode(hidden_states, None, use_trace=False)

            # Actual decode
            start_time = time.perf_counter()
            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000

        ttnn.deallocate(hidden_states)

        # Increment position
        self.decode_position += 1

        # Update device position tensor
        new_pos_ttnn = ttnn.from_torch(
            torch.tensor([self.decode_position], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.mesh_device,
            mesh_mapper=self.mesh_mapper,
        )
        ttnn.copy(new_pos_ttnn, self.current_pos)
        ttnn.deallocate(new_pos_ttnn)

        return logits, decode_time

    def run_inference(
        self,
        image_inputs: dict,
        prompt: str,
        max_new_tokens: int = 100,
        use_trace: bool = False,
        use_decode_trace: bool = False,
    ) -> Tuple[str, dict]:
        """
        Run full inference with autoregressive generation.

        Args:
            image_inputs: Dict from preprocess_image_molmo2
            prompt: Text prompt (should include <|image|> token)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode (disabled by default
                              to avoid memory corruption from tensor allocation during trace)

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        # Build prompt with image tokens
        image_grid = image_inputs["image_grids"][0]
        image_tokens_str = get_image_tokens(image_grid)
        full_prompt = prompt.replace(IMAGE_PROMPT, image_tokens_str)

        # Tokenize input
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")

        # Get pooling indices
        pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)

        # Get pixel values (need to reshape for our model)
        # pixel_values shape: [n_crops, n_patches, patch_pixels]
        # We need: [B, C, H, W] for vision encoder
        pixel_values = image_inputs["pixel_values"]

        # Run prefill
        logits, prefill_timing = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
        )

        # Get first prediction from prefill
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()
        if logits_torch.dim() == 2:
            next_token_logits = logits_torch[-1, :]
        else:
            next_token_logits = logits_torch

        next_token = torch.argmax(next_token_logits).item()
        generated_tokens = [next_token]

        # Autoregressive generation
        decode_times = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            # Check for EOS
            if next_token == eos_token_id:
                break

            # Run decode step (use_decode_trace is separate from prefill tracing)
            logits, decode_time = self.run_decode_step(
                next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            # Get next token
            logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()
            if logits_torch.dim() >= 2:
                next_token_logits = logits_torch[-1, :]
            else:
                next_token_logits = logits_torch

            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens, last decode: {decode_time:.2f}ms")

        # Decode generated tokens
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Calculate metrics
        total_decode_time = sum(decode_times) if decode_times else 0.0
        avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
        tokens_per_sec = len(decode_times) / (total_decode_time / 1000) if total_decode_time > 0 else 0

        perf_metrics = {
            "vision_ms": prefill_timing.get("vision_ms", 0),
            "compile_prefill_ms": prefill_timing.get("compile_prefill_ms", 0),
            "ttft_ms": prefill_timing.get("ttft_ms", 0),
            "avg_decode_ms": avg_decode_time,
            "total_decode_ms": total_decode_time,
            "input_tokens": input_ids.shape[1],
            "generated_tokens": len(generated_tokens),
            "num_generated_tokens": len(generated_tokens),  # Alias for compatibility
            "tokens_per_sec": tokens_per_sec,
            "decode_throughput": tokens_per_sec,  # Alias for compatibility
            "output_text": output_text,
        }

        logger.info(f"Input tokens: {input_ids.shape[1]}")
        logger.info(f"Generated {len(generated_tokens)} tokens")
        logger.info(f"Output: '{output_text[:100]}...' " if len(output_text) > 100 else f"Output: '{output_text}'")

        return output_text, perf_metrics


def run_demo(
    image_path: Optional[str] = None,
    prompt: str = "<|image|> Describe this image in detail.",
    max_new_tokens: int = 100,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    use_trace: bool = False,
    use_decode_trace: bool = False,
):
    """
    Run the Molmo2 demo.

    Args:
        image_path: Path to input image (uses default if None)
        prompt: Text prompt for the model (must include <|image|>)
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        use_trace: Whether to use tracing for performance
    """
    if image_path is None:
        image_path = str(DEFAULT_IMAGE)

    logger.info("=" * 60)
    logger.info("Molmo2-8B Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess image using Molmo2 processor
    logger.info("Preprocessing image...")
    image_inputs = preprocess_image_molmo2(image_path)
    logger.info(f"Image preprocessed: {image_inputs['image_num_crops'].item()} crops")

    # Load weights
    state_dict = load_model_weights()

    # Open multi-device mesh for T3K (8 devices) to enable bfloat16 weight sharding
    # This prevents numerical overflow during decode by using higher precision weights
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        # Create model
        model = create_model(device, state_dict, num_layers)
        text_num_layers = num_layers if num_layers is not None else 36

        # Create generator
        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=2048,
        )

        # Run inference
        logger.info("\n" + "=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info("=" * 60)

        response, perf_metrics = generator.run_inference(
            image_inputs=image_inputs,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Performance Metrics:")
        logger.info(f"  Vision processing: {perf_metrics['vision_ms']:.2f}ms")
        logger.info(f"  Prefill compile: {perf_metrics['compile_prefill_ms']:.2f}ms")
        logger.info(f"  TTFT (Time to First Token): {perf_metrics['ttft_ms']:.2f}ms")
        logger.info(f"  Avg decode time: {perf_metrics['avg_decode_ms']:.2f}ms")
        logger.info(f"  Total decode time: {perf_metrics['total_decode_ms']:.2f}ms")
        logger.info(f"  Input tokens: {perf_metrics['input_tokens']}")
        logger.info(f"  Generated tokens: {perf_metrics['generated_tokens']}")
        logger.info(f"  Decode throughput: {perf_metrics['tokens_per_sec']:.2f} tok/s")
        logger.info("=" * 60)
        logger.info(f"Output: {perf_metrics['output_text']}")
        logger.info("=" * 60)

        return perf_metrics

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")


def main():
    parser = argparse.ArgumentParser(description="Molmo2-8B Demo")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (uses default dog.jpg if not specified)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|image|> Describe this image in detail.",
        help="Text prompt for the model (must include <|image|> token)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="TTNN device ID",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of text layers (default: 36, use fewer for faster testing)",
    )
    parser.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable tracing for prefill (improved compilation)",
    )
    parser.add_argument(
        "--use-decode-trace",
        action="store_true",
        help="Enable tracing for decode (experimental - may cause memory corruption)",
    )

    args = parser.parse_args()

    run_demo(
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        device_id=args.device,
        num_layers=args.num_layers,
        use_trace=args.use_trace,
        use_decode_trace=args.use_decode_trace,
    )


if __name__ == "__main__":
    main()
