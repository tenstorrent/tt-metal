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

    # Video (recommended flags — matches eval harness; paged KV + traces):
    python -m models.demos.molmo2.demo.demo \\
        --video 'https://.../clip.mp4' \\
        --prompt $'<|video|>\\nYour question...' \\
        --max-tokens 16 \\
        --paged-attention --use-decode-trace

    # Video defaults: decode trace ON, vision trace ON, prefill trace ON (prefill trace is
    # auto-disabled when HF multimodal token_type_ids are used — see run_prefill warning).
    # Opt out: --no-use-vision-trace, --no-use-trace, --no-use-decode-trace

    # Video: decode trace is ON by default (~30+ tok/s). Debug without trace:
    python -m models.demos.molmo2.demo.demo --video URL --prompt "..." --no-use-decode-trace
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger

import ttnn
from models.demos.molmo2.tt.model_loader import create_model, load_model_weights, load_processor
from models.demos.molmo2.tt.trace_capture_utils import trace_capture_run_begin, trace_capture_run_end


@dataclass
class PenaltyArgs:
    """Arguments for TTPenalties initialization."""

    max_batch_size: int = 32
    vocab_size: int = 152064
    padded_vocab_size: int = 152064


# Import HF processor wrapper (uses correct configs from HuggingFace)
from models.demos.molmo2.tt.hf_processor import preprocess_image, preprocess_video
from models.demos.molmo2.tt.prefill_attention_mask import build_molmo2_prefill_attention_bias

# Import shared utilities from tt module
from models.demos.molmo2.tt.utils import (
    IMAGE_PROMPT,
    PREFILL_SEQ_BUCKETS,
    VIDEO_PROMPT,
    get_image_tokens,
    pad_input_ids,
    pad_seq_2d_right,
)

# Default paths
DEMO_DIR = Path(__file__).parent
DEFAULT_IMAGE = DEMO_DIR / "dog.jpg"

# Video defaults (HF processor uses these internally, but we expose them for CLI)
VIDEO_MAX_FRAMES = 384  # HF default
VIDEO_MAX_FPS = 2.0  # HF default

# Re-export MODEL_ID for backwards compatibility
MODEL_ID = "allenai/Molmo2-8B"


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
        use_paged_attention: bool = False,
        block_size: int = 64,
        num_blocks: int = 512,
        repetition_penalty: float = 1.0,
    ):
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.repetition_penalty = repetition_penalty

        # Paged attention config
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.page_table = None  # ttnn tensor for current request's page table
        self.page_table_torch = None  # torch tensor for page table management

        # Simple CPU-based repetition penalty tracking
        self.generated_token_ids = set()  # Track generated tokens for penalty
        if repetition_penalty != 1.0:
            logger.info(f"Repetition penalty enabled: {repetition_penalty}")

        # Chunked prefill config - process long sequences in chunks to avoid OOM
        # 8192 is safe for ~12GB DRAM per device (attention matrix fits in memory)
        self.max_prefill_chunk_size = 8192

        # Separate trace state for prefill and decode
        self.prefill_traces = {}  # {seq_len: (trace_id, trace_inputs, trace_output)}
        self.decode_trace_id = None
        self.decode_trace_tensors = None
        self.decode_trace_output = None

        # Decode trace state tracking (TTTransformers pattern)
        # Track previous page_table to detect changes that require full input re-copy
        self.prev_page_table = None
        # Flag to force full trace input re-copy on next decode (set by reset_state)
        self.decode_trace_needs_reset = True

        # Vision trace state (ViT encoder) -- single-image / non-DP path
        self.vision_trace_id = None
        self.vision_trace_tensors = None
        self.vision_trace_outputs = None  # [feature_layer_18, feature_layer_24]

        # DP=8 ViT trace state (video path)
        self.dp_vit_trace_id = None
        self.dp_vit_trace_input = None  # sharded [num_devices,1,fpd*729,588]
        self.dp_vit_trace_output = None  # sharded [num_devices,1,fpd*729,2304]
        self.dp_vit_pos_tiled = None  # replicated [1,1,fpd*729,vit_hidden_dim]
        self.dp_vit_frames_per_device = None
        self.dp_vit_num_devices = None

        # Pool chunk trace state (TP=8, one trace per fixed chunk shape)
        # Uses chunk-relative indexing: feat_buf = [chunk_frames*patches_per_frame, pool_dim]
        self.dp_pool_trace_id = None
        self.dp_pool_trace_tensors = None  # {image_features_2d, idx, valid_mask}
        self.dp_pool_trace_output = None
        self.dp_pool_chunk_frames = None
        self.dp_pool_n_out = None
        self.dp_pool_k_pool = None
        self.dp_pool_patches_per_frame = None  # e.g. 729 for 27×27 patches

        # Eager vision path: track whether the first (compile) call has been done
        self._vision_eager_compiled = False

        # KV cache (initialized on first run)
        self.kv_caches = None
        self.current_pos = None
        self.decode_position = 0  # Track position on host for trace updates

        # Mesh mapper
        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

    def init_kv_cache(self):
        """Initialize KV cache, position tensors, and RoPE index tensors."""
        from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache, init_paged_kv_cache

        if self.kv_caches is None:
            if self.use_paged_attention:
                # Paged KV cache: [batch_size, num_kv_heads, max_seq_len, head_dim]
                # page_table maps virtual positions to physical blocks within max_seq_len
                self.kv_caches = init_paged_kv_cache(
                    mesh_device=self.mesh_device,
                    num_layers=self.num_layers,
                    num_blocks=self.num_blocks,
                    num_kv_heads=8,
                    block_size=self.block_size,
                    head_dim=128,
                    dtype=ttnn.bfloat16,
                    batch_size=self.batch_size,
                    max_seq_len=self.max_seq_len,
                )
                logger.info(f"Initialized paged KV cache: batch={self.batch_size}, max_seq_len={self.max_seq_len}")
            else:
                # Non-paged KV cache: [batch, num_kv_heads, max_seq_len, head_dim]
                self.kv_caches = init_kv_cache(
                    mesh_device=self.mesh_device,
                    num_layers=self.num_layers,
                    batch_size=self.batch_size,
                    num_kv_heads=8,
                    max_seq_len=self.max_seq_len,
                    head_dim=128,
                    dtype=ttnn.bfloat16,
                )
            self.current_pos = init_decode_position(
                mesh_device=self.mesh_device,
                batch_size=self.batch_size,
                initial_pos=0,
            )
            self.rot_mat_idxs = self.model.text_model.rotary_setup.allocate_decode_rot_idxs(initial_pos=0)

    def init_page_table(self, seq_len: int):
        """
        Initialize page table for a new request.

        For paged attention, creates a page table that maps sequence positions
        to physical block indices. Each block holds block_size tokens.

        Args:
            seq_len: Initial sequence length (prompt length)
        """
        if not self.use_paged_attention:
            return

        # Calculate number of blocks needed for this sequence
        num_blocks_needed = (seq_len + self.block_size - 1) // self.block_size
        # Add extra blocks for generation (up to max_seq_len)
        max_blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
        num_blocks_needed = min(num_blocks_needed + 128, max_blocks_per_seq)  # Extra for generation

        # Create page table: [batch_size, max_blocks_per_seq]
        # Each entry maps to a physical block index (0 to num_blocks-1)
        # For demo, we use sequential allocation starting from 0
        self.page_table_torch = torch.arange(num_blocks_needed, dtype=torch.int32).unsqueeze(0)

        # Convert to ttnn tensor
        self.page_table = ttnn.from_torch(
            self.page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        logger.info(f"Initialized page table: {num_blocks_needed} blocks for seq_len={seq_len}")

    def reset_kv_cache(self, start_pos: Union[int, List[int]] = 0):
        """Reset KV cache position and RoPE indices for new generation.

        Args:
            start_pos: Starting position(s). Can be int (same for all batch) or
                       List[int] (per-batch positions for parallel processing).
        """
        # Track position on host (for simple case, use first position)
        if isinstance(start_pos, int):
            self.decode_position = start_pos
            pos_values = [start_pos] * self.batch_size
        else:
            self.decode_position = start_pos[0]  # Use first for tracking
            pos_values = list(start_pos)
            # Pad to batch_size if fewer positions provided (for batched inference with < batch_size prompts)
            if len(pos_values) < self.batch_size:
                pos_values = pos_values + [0] * (self.batch_size - len(pos_values))
            assert (
                len(pos_values) == self.batch_size
            ), f"start_pos length {len(pos_values)} != batch_size {self.batch_size}"

        if self.current_pos is not None:
            pos_tensor = torch.tensor(pos_values, dtype=torch.int32)
            pos_ttnn = ttnn.from_torch(
                pos_tensor,
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(pos_ttnn, self.current_pos)
            ttnn.deallocate(pos_ttnn)

        if self.rot_mat_idxs is not None:
            batch = self.batch_size
            pad_size = ((batch + 31) // 32) * 32 - batch
            # For RoPE indices, use the first position value for padding
            rot_values = pos_values + [pos_values[0]] * pad_size
            rot_idxs_tensor = torch.tensor([rot_values], dtype=torch.int32)
            rot_idxs_ttnn = ttnn.from_torch(
                rot_idxs_tensor,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            ttnn.copy(rot_idxs_ttnn, self.rot_mat_idxs)
            ttnn.deallocate(rot_idxs_ttnn)

    def reset_state(self):
        """Reset all state between requests to prevent memory leaks.

        Call this between sequential video/image requests to:
        1. Reset KV cache position to 0
        2. Clear KV cache content to prevent stale data leakage
        3. Reset VisionAttention SDPA counters
        4. Mark decode trace for full input re-copy (TTTransformers pattern)
        5. Synchronize device to flush pending operations
        6. Run garbage collection to free Python objects
        """
        import gc

        from models.demos.molmo2.tt.vision_attention import VisionAttention

        # Reset KV cache position
        self.reset_kv_cache(0)

        # Clear KV cache content to prevent stale data leakage between requests
        # This is critical for correct behavior when reusing traces across multiple inferences
        if self.kv_caches is not None:
            for layer_idx, (k_cache, v_cache) in enumerate(self.kv_caches):
                # Get cache shape and dtype for creating zeros
                k_shape = list(k_cache.shape)
                k_dtype = k_cache.dtype

                # Create zeros tensor matching the cache configuration
                zeros_torch = torch.zeros(k_shape, dtype=torch.bfloat16)

                # Create zeros on device with matching dtype and copy to cache
                zeros_k = ttnn.from_torch(
                    zeros_torch,
                    dtype=k_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                zeros_v = ttnn.from_torch(
                    zeros_torch,
                    dtype=k_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )

                # Copy zeros to cache
                ttnn.copy(zeros_k, k_cache)
                ttnn.copy(zeros_v, v_cache)

                # Deallocate temporary zeros
                ttnn.deallocate(zeros_k)
                ttnn.deallocate(zeros_v)

            logger.debug(f"Cleared KV cache content for {len(self.kv_caches)} layers")

        # Reset page table tracking - next decode will re-copy all inputs
        self.prev_page_table = None
        self.decode_trace_needs_reset = True

        # Reset page table if exists
        if hasattr(self, "page_table") and self.page_table is not None:
            # Re-initialize page table for next request
            pass  # Page table will be re-initialized in run_prefill

        # Reset VisionAttention counters to prevent unbounded growth
        VisionAttention.reset_counters()

        # Synchronize device to ensure all pending operations complete
        ttnn.synchronize_device(self.mesh_device)

        # Force garbage collection to free Python objects
        gc.collect()

        logger.debug("State reset complete")

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        use_data_parallel: bool = False,
        frames_per_device: int = 8,
    ) -> ttnn.Tensor:
        """
        Prepare text model inputs by processing vision and fusing embeddings on device.

        No CPU roundtrip: vision runs on TTNN, fusion is done via selector matmul.
        Must be called BEFORE trace capture since it computes new tensors.

        Args:
            input_ids: Input token IDs
            pixel_values: Preprocessed image tensor
            pooled_patches_idx: Indices for vision pooling

        Returns:
            Fused hidden states [1, 1, seq_len, hidden_dim] on device
        """
        if pixel_values is not None and pooled_patches_idx is not None:
            # Track request number for debugging SDPA issues
            from models.demos.molmo2.tt.vision_attention import VisionAttention

            VisionAttention._current_request += 1
            request_num = VisionAttention._current_request
            sdpa_calls_before = VisionAttention._sdpa_call_count

            logger.info(f"_prepare_text_inputs (DEMO): Starting vision+text fusion (REQUEST #{request_num})")
            logger.info(f"  SDPA calls so far: {sdpa_calls_before}")

            # pixel_values can be:
            # 1. Pre-unfolded from vLLM: [num_crops, num_patches, 588] - 3D with last dim == 588
            # 2. Raw image: [C, H, W] or [B, C, H, W]
            # Only add batch dim for raw image format, not pre-unfolded
            patch_features = 14 * 14 * 3  # 588
            if pixel_values.dim() == 3 and pixel_values.shape[-1] != patch_features:
                # Raw image [C, H, W] -> [1, C, H, W]
                pixel_values = pixel_values.unsqueeze(0)
            visual_embeddings_ttnn, valid_token = self.model.embed_image(
                pixel_values,
                pooled_patches_idx,
                use_data_parallel=use_data_parallel,
                frames_per_device=frames_per_device,
            )

            sdpa_calls_after = VisionAttention._sdpa_call_count
            logger.info(f"_prepare_text_inputs (DEMO): embed_image completed (REQUEST #{request_num})")
            logger.info(
                f"  SDPA calls this request: {sdpa_calls_after - sdpa_calls_before} (total: {sdpa_calls_after})"
            )

            # Debug: dump visual embedding statistics for comparison
            is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
            if is_mesh:
                ve_torch = ttnn.to_torch(
                    visual_embeddings_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
            else:
                ve_torch = ttnn.to_torch(visual_embeddings_ttnn)
            logger.info(f"  Visual embeddings shape: {ve_torch.shape}")
            logger.info(
                f"  Visual embeddings stats: mean={ve_torch.mean().item():.6f}, std={ve_torch.std().item():.6f}, min={ve_torch.min().item():.6f}, max={ve_torch.max().item():.6f}"
            )

            fused_ttnn = self.model.prepare_inputs_for_multimodal(input_ids, visual_embeddings_ttnn, valid_token)
            ttnn.deallocate(visual_embeddings_ttnn)
        else:
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )
            fused_ttnn = self.model.text_model.embed_tokens(input_ids_ttnn)
            ttnn.deallocate(input_ids_ttnn)

        return fused_ttnn

    def _prepare_vision_inputs_for_trace(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> dict:
        """
        Prepare vision inputs for traced execution.

        Converts all inputs to TTNN tensors so the forward can be fully traced.

        Args:
            pixel_values: Raw pixel values [B, C, H, W]
            pooled_patches_idx: Patch indices [B, N_out, K_pool]

        Returns:
            Dict with TTNN tensors and metadata for traced forward
        """
        batch_size = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # 1. Patch embedding on TTNN (unfold on CPU, linear+pos_embed on device)
        vit = self.model.vision_backbone.image_vit
        patch_features = vit.patch_size * vit.patch_size * 3  # 14*14*3 = 588

        # Detect input format:
        # - Pre-unfolded from vLLM: [num_crops, num_patches, 588] - 3D with last dim == 588
        # - Raw image format: [B, C, H, W] - 4D or 3D [C, H, W]
        if pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features:
            # Pre-unfolded patch format from vLLM [num_crops, num_patches, 588]
            embedded_ttnn = vit.patch_embed_from_patches_ttnn(pixel_values)
        else:
            # Raw image format [B, C, H, W] or [C, H, W]
            if pixel_values.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                pixel_values = pixel_values.unsqueeze(0)
            embedded_ttnn = vit.patch_embed_ttnn(pixel_values)  # [1, 1, B*N, hidden_dim] on device

        # 2. Prepare indices for TTNN gather
        # Identify valid indices (>= 0) and clip negative to 0
        valid = pooled_patches_idx >= 0  # [B, N_out, K_pool]
        valid_token = torch.any(valid, dim=-1)  # [B, N_out]
        clipped_idx = torch.clip(pooled_patches_idx, min=0)

        # Flatten indices for embedding lookup: [B, N_out, K_pool] -> [1, B*N_out*K_pool]
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)

        # Create valid mask: [1, 1, B*N_out*K_pool, 1]
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        # 3. Convert remaining tensors to TTNN (embedded_ttnn already on device)

        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten().float(),  # Must convert bool to float before bfloat16
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return {
            "embedded": embedded_ttnn,
            "idx": idx_ttnn,
            "valid_mask": valid_mask_ttnn,
            "valid_token": valid_token_ttnn,
            "valid_token_torch": valid_token,  # Keep torch version for final filtering
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
        }

    # =========================================================================
    # DP=8 ViT TRACE (video path)
    # =========================================================================

    def _warmup_vit_trace(
        self,
        frames_per_device: int = 8,
        num_devices: int = 8,
    ) -> None:
        """
        Compile and capture the ViT DP=8 trace for one pass of ``frames_per_device``
        frames per device.

        Must be called with the model already loaded on the mesh device.
        After this call ``dp_vit_trace_id`` / ``dp_vit_trace_input`` /
        ``dp_vit_trace_output`` / ``dp_vit_pos_tiled`` are all populated.
        """
        if not self.is_mesh_device:
            logger.warning("DP ViT trace requires a MeshDevice; skipping.")
            return

        vit = self.model.vision_backbone.image_vit
        num_patches_per_frame = (vit.image_size // vit.patch_size) ** 2  # 729
        patch_features = vit.patch_size * vit.patch_size * 3  # 588
        pool_dim = vit.hidden_dim * 2  # 2304 (concat of 2 feature layers)

        shard_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)

        # --- pre-compute positional embedding tiled for frames_per_device frames ---
        pos_tiles = [vit.positional_embedding] * frames_per_device
        pos_tiled = ttnn.concat(pos_tiles, dim=2)
        self.dp_vit_pos_tiled = pos_tiled  # persistent device tensor (replicated)

        # --- warmup (compile) pass ---
        logger.info(f"  [ViT trace] Compiling DP=8 ViT: {frames_per_device} frames/device × {num_devices} devices")
        dummy_patches = torch.zeros(
            num_devices, 1, frames_per_device * num_patches_per_frame, patch_features, dtype=torch.bfloat16
        )
        patches_compile = ttnn.from_torch(
            dummy_patches,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_mapper,
        )
        compile_out = self.model._vit_pass_forward_ttnn(patches_compile, pos_tiled)
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(compile_out)
        ttnn.deallocate(patches_compile)
        logger.info("  [ViT trace] Compile pass done")

        # --- allocate stable trace input buffer (sharded) ---
        trace_input = ttnn.from_torch(
            torch.zeros(
                num_devices, 1, frames_per_device * num_patches_per_frame, patch_features, dtype=torch.bfloat16
            ),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_mapper,
        )

        # --- capture trace ---
        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            trace_output = self.model._vit_pass_forward_ttnn(trace_input, pos_tiled, trace_capture=True)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        self.dp_vit_trace_id = trace_id
        self.dp_vit_trace_input = trace_input
        self.dp_vit_trace_output = trace_output
        self.dp_vit_frames_per_device = frames_per_device
        self.dp_vit_num_devices = num_devices

        # Multi-CQ pipelining: CQ0 for ops, CQ1 for input transfers
        # Initialize event for synchronization between CQs
        self.dp_vit_op_event = ttnn.record_event(self.mesh_device, 0)
        logger.info("  [ViT trace] Captured (multi-CQ pipelining enabled)")
        # Execute the ViT trace once with the dummy data already in trace_input so Metal
        # transitions it out of "active" state before pool trace buffers are allocated.
        # Without this, Metal warns "Allocating device buffers is unsafe due to the
        # existence of an active trace" and those pool buffers get corrupted on first
        # ViT trace replay (which causes pool trace execute_trace to hang).
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        logger.info("  [ViT trace] Dummy execute done (trace no longer active)")

    def _execute_vit_trace_pass(
        self,
        all_patches_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Execute the pre-captured ViT DP=8 trace for one pass.

        Uses multi-CQ pipelining:
        - CQ1: Input transfer (async)
        - CQ0: ViT trace execution

        Args:
            all_patches_torch: CPU tensor [num_devices, 1, fpd*729, 588].

        Returns:
            vit_features_ttnn: Device tensor [1, 1, num_devices*fpd*729, pool_dim]
                               (replicated across mesh, stays on device).
        """
        CQ_OPS = 0
        CQ_INPUT_WRITE = 1

        shard_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        pool_dim = self.model.vision_backbone.image_vit.hidden_dim * 2  # 2304

        # Create host ttnn tensor (no device transfer yet)
        host_patches_ttnn = ttnn.from_torch(
            all_patches_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=shard_mapper,
        )

        # CQ1: Async input transfer - wait for previous ops to complete first
        ttnn.wait_for_event(CQ_INPUT_WRITE, self.dp_vit_op_event)
        ttnn.copy_host_to_device_tensor(host_patches_ttnn, self.dp_vit_trace_input, cq_id=CQ_INPUT_WRITE)
        write_event = ttnn.record_event(self.mesh_device, CQ_INPUT_WRITE)

        # CQ0: Execute trace - wait for input transfer to complete
        ttnn.wait_for_event(CQ_OPS, write_event)
        ttnn.execute_trace(self.mesh_device, self.dp_vit_trace_id, cq_id=CQ_OPS, blocking=False)
        self.dp_vit_op_event = ttnn.record_event(self.mesh_device, CQ_OPS)

        ttnn.synchronize_device(self.mesh_device)

        # All-gather to replicate sharded output across all devices
        # Input: [num_devices shards of 1, 1, fpd*729, pool_dim]
        # Output: [1, 1, num_devices*fpd*729, pool_dim] replicated
        gathered = ttnn.all_gather(
            self.dp_vit_trace_output,
            dim=2,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return gathered

    # =========================================================================
    # Pool chunk TRACE (TP=8, video path)
    # =========================================================================

    def _warmup_pool_chunk_trace(
        self,
        chunk_frames: int,
        n_out: int,
        k_pool: int,
    ) -> None:
        """
        Compile and capture a single-chunk pooling trace for the given shape.

        Uses CHUNK-RELATIVE indexing: each chunk's feature buffer only holds
        that chunk's frames (chunk_frames * patches_per_frame rows), not the
        full video.  Global pool indices are converted to chunk-relative before
        each trace execution.  This keeps feat_2d_buf at ~53 MB instead of the
        previous ~268 MB (which caused OOM on 12 GB Wormhole devices).

        The trace input tensors are:
          - ``image_features_2d``: [chunk_frames*patches_per_frame, pool_dim]
            ROW_MAJOR (updated once per chunk).
          - ``idx``: [1, chunk_frames*n_out*k_pool] uint32 CHUNK-RELATIVE
            (updated per chunk).
          - ``valid_mask``: [1, 1, chunk_frames*n_out*k_pool, 1] bfloat16
            (updated per chunk).

        After capture, ``dp_pool_trace_id`` / ``dp_pool_trace_tensors`` /
        ``dp_pool_trace_output`` / ``dp_pool_patches_per_frame`` are set.
        """
        pool_dim = self.model.vision_backbone.image_vit.hidden_dim * 2  # 2304
        patches_per_frame = (
            self.model.vision_backbone.image_vit.image_size // self.model.vision_backbone.image_vit.patch_size
        ) ** 2  # 729
        chunk_total_patches = chunk_frames * patches_per_frame  # 16 * 729 = 11664
        replicate = self.mesh_mapper  # ReplicateTensorToMesh or None
        idx_len = chunk_frames * n_out * k_pool

        logger.info(
            f"  [Pool trace] Compiling chunk pooling: chunk_frames={chunk_frames} "
            f"n_out={n_out} k_pool={k_pool} chunk_patches={chunk_total_patches} "
            f"(chunk-relative indexing, ~{chunk_total_patches * pool_dim * 2 / 1e6:.0f} MB feat buf)"
        )

        # --- allocate stable trace input buffers (chunk-sized, not video-sized) ---
        feat_2d_buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([chunk_total_patches, pool_dim]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        idx_buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, idx_len]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        valid_mask_buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, idx_len, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialise with dummy valid data so compile sees realistic shapes
        dummy_feat = ttnn.from_torch(
            torch.zeros(chunk_total_patches, pool_dim, dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        ttnn.copy(dummy_feat, feat_2d_buf)
        ttnn.deallocate(dummy_feat)

        dummy_idx = ttnn.from_torch(
            torch.zeros(1, idx_len, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        ttnn.copy(dummy_idx, idx_buf)
        ttnn.deallocate(dummy_idx)

        dummy_mask = ttnn.from_torch(
            torch.ones(1, 1, idx_len, 1, dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        ttnn.copy(dummy_mask, valid_mask_buf)
        ttnn.deallocate(dummy_mask)

        # --- warmup (compile) pass ---
        compile_out = self.model.vision_backbone.pool_chunk_from_features_ttnn(
            feat_2d_buf, idx_buf, valid_mask_buf, chunk_frames, n_out, k_pool
        )
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(compile_out)
        logger.info("  [Pool trace] Compile pass done")

        # --- capture trace ---
        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            trace_output = self.model.vision_backbone.pool_chunk_from_features_ttnn(
                feat_2d_buf, idx_buf, valid_mask_buf, chunk_frames, n_out, k_pool
            )
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        self.dp_pool_trace_id = trace_id
        self.dp_pool_trace_tensors = {
            "image_features_2d": feat_2d_buf,
            "idx": idx_buf,
            "valid_mask": valid_mask_buf,
        }
        self.dp_pool_trace_output = trace_output
        self.dp_pool_chunk_frames = chunk_frames
        self.dp_pool_n_out = n_out
        self.dp_pool_k_pool = k_pool
        self.dp_pool_patches_per_frame = patches_per_frame
        logger.info(
            f"  [Pool trace] Captured: feat_buf={chunk_total_patches}×{pool_dim} "
            f"({chunk_total_patches * pool_dim * 2 / 1e6:.0f} MB, chunk-relative)"
        )
        ttnn.synchronize_device(self.mesh_device)

    def _embed_image_data_parallel_traced(
        self,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        frames_per_device: int = 8,
        num_devices: int = 8,
        max_frames_per_pool_chunk: int = 16,
    ) -> Tuple[ttnn.Tensor, torch.Tensor]:
        """
        DP=8 video embed with pre-captured ViT and pool traces.

        Replaces the eager ``_embed_image_data_parallel`` when both
        ``dp_vit_trace_id`` and ``dp_pool_trace_id`` are set.

        The ViT trace is replayed once per pass (same trace, new data).
        The pool trace is replayed for all chunks (including partial ones padded to
        the trace shape); padded output is sliced after replay.
        """
        from models.demos.molmo2.tt.vision_backbone import MAX_FRAMES_FOR_SINGLE_POOL

        vit = self.model.vision_backbone.image_vit
        num_patches_per_frame = (vit.image_size // vit.patch_size) ** 2  # 729
        patch_features = vit.patch_size * vit.patch_size * 3  # 588
        pool_dim = vit.hidden_dim * 2  # 2304

        total_frames = pooled_patches_idx.shape[0]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        frames_per_pass = frames_per_device * num_devices
        num_passes = (total_frames + frames_per_pass - 1) // frames_per_pass

        shard_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        replicate_mapper = self.mesh_mapper
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)

        logger.info(
            f"embed_image_data_parallel_traced: {total_frames} frames, "
            f"{frames_per_device} fpd, {num_devices} devices, {num_passes} passes"
        )

        # ---- Stage 1: ViT passes (traced, features stay on device) ----
        # Detect input format: HF processor gives [n_frames, 729, 588] (pre-unfolded)
        is_patches_format = pixel_values.dim() == 3 and pixel_values.shape[-1] == patch_features

        # Pre-unfold ALL frames upfront if not already unfolded (avoid per-pass CPU work)
        if is_patches_format:
            # Already unfolded: [total_frames, 729, 588]
            all_patches_cpu = pixel_values.float()
            logger.info(f"  Using pre-unfolded patches format: {all_patches_cpu.shape}")
        else:
            # Raw images [total_frames, C, H, W] - unfold ALL upfront
            logger.info(f"  Unfolding all {total_frames} frames upfront...")
            patches_list = []
            for frame_idx in range(total_frames):
                frame = pixel_values[frame_idx : frame_idx + 1]  # [1, C, H, W]
                x = frame.unfold(2, vit.patch_size, vit.patch_size)
                x = x.unfold(3, vit.patch_size, vit.patch_size)
                x = x.permute(0, 2, 3, 4, 5, 1).reshape(num_patches_per_frame, patch_features)
                patches_list.append(x)
            all_patches_cpu = torch.stack(patches_list, dim=0).float()  # [total_frames, 729, 588]
            logger.info(f"  Pre-unfolded all frames: {all_patches_cpu.shape}")

        all_vit_features: List[ttnn.Tensor] = []

        for pass_idx in range(num_passes):
            pass_start = pass_idx * frames_per_pass
            pass_end = min(pass_start + frames_per_pass, total_frames)
            actual_frames_this_pass = pass_end - pass_start

            # Slice pre-unfolded patches (cheap view operation)
            pass_patches = all_patches_cpu[pass_start:pass_end]  # [actual, 729, 588]

            # Pad to full frames_per_pass for uniform sharding
            if actual_frames_this_pass < frames_per_pass:
                pad_frames = frames_per_pass - actual_frames_this_pass
                padding = torch.zeros((pad_frames, num_patches_per_frame, patch_features), dtype=pass_patches.dtype)
                pass_patches = torch.cat([pass_patches, padding], dim=0)

            # Reshape for sharding: [frames_per_pass, 729, 588] → [num_devices, 1, fpd*729, 588]
            pass_patches = pass_patches.reshape(num_devices, frames_per_device, num_patches_per_frame, patch_features)
            all_patches = pass_patches.reshape(
                num_devices, 1, frames_per_device * num_patches_per_frame, patch_features
            )

            logger.debug(f"  Pass {pass_idx+1}/{num_passes}: frames {pass_start}-{pass_end}")
            vit_features_ttnn = self._execute_vit_trace_pass(all_patches)

            # Trim padding on device
            actual_patches = actual_frames_this_pass * num_patches_per_frame
            total_patches_this_pass = frames_per_pass * num_patches_per_frame
            if actual_patches < total_patches_this_pass:
                vit_features_ttnn = ttnn.slice(
                    vit_features_ttnn,
                    [0, 0, 0, 0],
                    [1, 1, actual_patches, pool_dim],
                )
            all_vit_features.append(vit_features_ttnn)

        # Concatenate on device (no CPU round-trip!)
        if len(all_vit_features) == 1:
            combined_vit_features_ttnn = all_vit_features[0]
        else:
            combined_vit_features_ttnn = ttnn.concat(all_vit_features, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in all_vit_features:
                ttnn.deallocate(t)
        logger.info(f"  Combined ViT features (on device): {combined_vit_features_ttnn.shape}")

        # ---- Stage 2: Pooling (traced for all chunks; partial last chunk padded to trace shape) ----
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)
        batch_size = total_frames

        if batch_size <= MAX_FRAMES_FOR_SINGLE_POOL and self.dp_pool_trace_id is None:
            # Single-pass eager path: only when no pool trace is captured
            # Features already on device from Stage 1 - no CPU round-trip!
            clipped_idx = torch.clip(pooled_patches_idx, min=0)
            flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
            valid_mask = valid.reshape(1, 1, -1, 1).float()

            # Reshape to 2D feature table (already on device)
            image_features_2d = ttnn.reshape(combined_vit_features_ttnn, [-1, pool_dim])
            image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

            idx_ttnn = ttnn.from_torch(
                flat_idx,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )
            valid_mask_ttnn = ttnn.from_torch(
                valid_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper,
            )

            visual_embeddings = self.model.vision_backbone.pool_and_project_from_features_ttnn(
                image_features_2d=image_features_2d,
                pooled_patches_idx_ttnn=idx_ttnn,
                valid_mask_ttnn=valid_mask_ttnn,
                n_out=n_out,
                k_pool=k_pool,
                batch_size=batch_size,
            )
            ttnn.deallocate(image_features_2d)
            ttnn.deallocate(idx_ttnn)
            ttnn.deallocate(valid_mask_ttnn)
        else:
            # Chunked pooling: traced for all chunks with CHUNK-RELATIVE indexing.
            # Each chunk uploads only its own frames' features (53 MB) instead of the
            # full video feature table (268 MB), avoiding OOM on 12 GB Wormhole devices.

            # Transfer to CPU for chunk slicing (required for memory efficiency)
            combined_vit_features = ttnn.to_torch(combined_vit_features_ttnn, mesh_composer=mesh_composer)
            ttnn.deallocate(combined_vit_features_ttnn)
            # Pre-compute 2D CPU feature table for cheap chunk slicing
            logger.info(
                f"  [pool] flattening ViT features to 2D for chunk upload "
                f"(patches={combined_vit_features.shape[2]}, dim={pool_dim})..."
            )
            combined_vit_features_2d = combined_vit_features.reshape(-1, pool_dim)
            patches_per_frame = self.dp_pool_patches_per_frame if self.dp_pool_trace_id is not None else 729

            all_device_chunks: List[ttnn.Tensor] = []
            for chunk_start in range(0, batch_size, max_frames_per_pool_chunk):
                chunk_end = min(chunk_start + max_frames_per_pool_chunk, batch_size)
                chunk_frames = chunk_end - chunk_start

                chunk_idx = pooled_patches_idx[chunk_start:chunk_end]
                chunk_valid = valid[chunk_start:chunk_end]
                # Clip to 0 (invalid positions become 0, masked out by valid_mask later)
                flat_chunk_idx = torch.clip(chunk_idx, min=0).reshape(1, -1).to(torch.int32)
                flat_chunk_valid = chunk_valid.reshape(1, 1, -1, 1).float()

                is_trace_output = False
                is_partial_padded = False

                if (
                    self.dp_pool_trace_id is not None
                    and n_out == self.dp_pool_n_out
                    and k_pool == self.dp_pool_k_pool
                    and chunk_frames <= self.dp_pool_chunk_frames
                ):
                    trace_feat_buf = self.dp_pool_trace_tensors["image_features_2d"]
                    chunk_total_patches = self.dp_pool_chunk_frames * patches_per_frame  # e.g. 16*729=11664
                    pad_frames = self.dp_pool_chunk_frames - chunk_frames

                    # --- 1. Upload chunk features to trace feat_buf (chunk-relative) ---
                    chunk_start_feat = chunk_start * patches_per_frame
                    chunk_end_feat = chunk_end * patches_per_frame  # may exceed total if partial
                    chunk_end_feat = min(chunk_end_feat, combined_vit_features_2d.shape[0])
                    chunk_feats_cpu = combined_vit_features_2d[
                        chunk_start_feat:chunk_end_feat, :
                    ]  # [actual_patches, pool_dim]
                    actual_chunk_patches = chunk_feats_cpu.shape[0]
                    logger.debug(
                        f"  [chunk {chunk_start}] trace path: frames={chunk_frames}, "
                        f"feat_rows={actual_chunk_patches}/{chunk_total_patches}"
                    )

                    feat_tmp = ttnn.from_torch(
                        chunk_feats_cpu,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    logger.info(f"  [chunk {chunk_start}] pool trace: feat host→device upload done")
                    if actual_chunk_patches < chunk_total_patches:
                        # Pad partial chunk features to full trace size (on-device, no CPU roundtrip)
                        feat_padded = ttnn.pad(
                            feat_tmp,
                            padding=[(0, chunk_total_patches - actual_chunk_patches), (0, 0)],
                            value=0.0,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        ttnn.copy(feat_padded, trace_feat_buf)
                        ttnn.deallocate(feat_padded)
                    else:
                        ttnn.copy(feat_tmp, trace_feat_buf)
                    ttnn.deallocate(feat_tmp)
                    # Ensure feat copy is visible before idx/mask uploads + trace replay (mesh CQ ordering).
                    ttnn.synchronize_device(self.mesh_device)

                    # --- 2. Convert global indices to chunk-relative ---
                    # Valid indices are in [chunk_start_feat, chunk_end_feat); after shift they're in [0, chunk_total_patches)
                    # Invalid (clipped-to-0) indices become negative after shift → re-clip to 0
                    flat_chunk_idx_relative = torch.clip(flat_chunk_idx - chunk_start_feat, min=0).to(torch.int32)

                    # --- 3. Pad partial chunk idx/mask to full trace shape ---
                    if pad_frames > 0:
                        flat_chunk_idx_relative = torch.nn.functional.pad(
                            flat_chunk_idx_relative, (0, pad_frames * n_out * k_pool), value=0
                        )
                        flat_chunk_valid = torch.nn.functional.pad(
                            flat_chunk_valid, (0, 0, 0, pad_frames * n_out * k_pool), value=0.0
                        )
                        is_partial_padded = True

                    # --- 4. Upload idx/mask and execute pool trace ---
                    idx_tmp = ttnn.from_torch(
                        flat_chunk_idx_relative,
                        device=self.mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    mask_tmp = ttnn.from_torch(
                        flat_chunk_valid,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    ttnn.copy(idx_tmp, self.dp_pool_trace_tensors["idx"])
                    ttnn.copy(mask_tmp, self.dp_pool_trace_tensors["valid_mask"])
                    ttnn.deallocate(idx_tmp)
                    ttnn.deallocate(mask_tmp)
                    logger.info(f"  [chunk {chunk_start}] pool trace: executing (blocking=True)...")
                    # blocking=True: avoids rare CQ / fabric hangs with blocking=False on mesh + sync
                    ttnn.execute_trace(self.mesh_device, self.dp_pool_trace_id, cq_id=0, blocking=True)
                    ttnn.synchronize_device(self.mesh_device)
                    logger.info(f"  [chunk {chunk_start}] pool trace: execute+sync done, reading output to CPU")
                    pooled_chunk = self.dp_pool_trace_output
                    is_trace_output = True
                else:
                    # Eager path: no trace captured, or n_out/k_pool mismatch
                    # Upload full video features on first chunk; reuse on subsequent chunks
                    if chunk_start == 0:
                        combined_features_ttnn = ttnn.from_torch(
                            combined_vit_features,
                            device=self.mesh_device,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=replicate_mapper,
                        )
                        image_features_2d = ttnn.reshape(combined_features_ttnn, [-1, pool_dim])
                        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)
                        ttnn.deallocate(combined_features_ttnn)
                    idx_tmp = ttnn.from_torch(
                        flat_chunk_idx,
                        device=self.mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    mask_tmp = ttnn.from_torch(
                        flat_chunk_valid,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=replicate_mapper,
                    )
                    pooled_chunk = self.model.vision_backbone.pool_chunk_from_features_ttnn(
                        image_features_2d, idx_tmp, mask_tmp, chunk_frames, n_out, k_pool
                    )
                    ttnn.deallocate(idx_tmp)
                    ttnn.deallocate(mask_tmp)

                # Keep chunk on device (trace output is overwritten on next execute_trace so always slice).
                # Use ROW_MAJOR to avoid tile-alignment issues when chunk_frames*n_out is not a multiple of 32.
                chunk_tokens = chunk_frames * n_out
                if is_trace_output:
                    logger.info(f"  [chunk {chunk_start}] pool trace: slicing [{chunk_tokens} tokens] to device buffer")
                    pooled_slice = ttnn.slice(
                        self.dp_pool_trace_output,
                        (0, 0, 0, 0),
                        (1, 1, chunk_tokens, pool_dim),
                    )
                    pooled_device = ttnn.to_layout(pooled_slice, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(pooled_slice)
                else:
                    # Eager path: pool_chunk_from_features_ttnn returns [1,1,chunk_tokens,pool_dim]
                    pooled_device = ttnn.to_layout(pooled_chunk, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(pooled_chunk)
                all_device_chunks.append(pooled_device)

            # Eager path: deallocate the full-video device feature table (only exists in eager path)
            if self.dp_pool_trace_id is None:
                ttnn.deallocate(image_features_2d)

            # Concat on device → to TILE → projector (no CPU roundtrip)
            combined_rm = ttnn.concat(all_device_chunks, dim=2)
            for t in all_device_chunks:
                ttnn.deallocate(t)
            combined_ttnn = ttnn.to_layout(combined_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(combined_rm)
            visual_embeddings = self.model.vision_backbone.image_projector(combined_ttnn)
            ttnn.deallocate(combined_ttnn)

        logger.info(f"embed_image_data_parallel_traced: Complete, output shape: {visual_embeddings.shape}")
        return visual_embeddings, valid_token

    # =========================================================================
    # UPFRONT WARMUP (all traces together before inference)
    # =========================================================================

    def warmup_video_traces(
        self,
        frames_per_device: int = 8,
        num_devices: int = 8,
        prefill_buckets: Optional[List[int]] = None,
        max_frames_per_pool_chunk: int = 16,
        pool_n_out: int = 81,
        pool_k_pool: int = 9,
        max_vit_frames: int = 80,
        use_prefill_trace: bool = True,
        use_decode_trace: bool = True,
    ) -> None:
        """
        Capture ALL traces upfront in the correct order:

          1. DP=8 ViT trace
          2. TP=8 pool-chunk trace
          3. Prefill traces (per bucket)
          4. Decode trace

        Call this after model creation and before ``run_video_inference``.

        If ``prefill_buckets`` is omitted, uses ``PREFILL_SEQ_BUCKETS`` from ``tt.utils``
        (1024 … 65536), keeping only buckets ``<= max_seq_len``.
        """
        if prefill_buckets is None:
            prefill_buckets = [b for b in PREFILL_SEQ_BUCKETS if b <= self.max_seq_len]

        logger.info("=" * 60)
        logger.info("WARMUP (video): capturing all traces upfront")
        logger.info(f"  ViT: {frames_per_device} fpd × {num_devices} devices")
        logger.info(f"  Pool chunk: {max_frames_per_pool_chunk} frames, n_out={pool_n_out}, k_pool={pool_k_pool}")
        logger.info(f"  Prefill buckets: {prefill_buckets}")
        logger.info("=" * 60)

        self.init_kv_cache()

        # Prefill traces + Decode trace
        if use_prefill_trace or use_decode_trace:
            logger.info("Capturing prefill/decode traces...")
            self.warmup_all_buckets(
                bucket_sizes=prefill_buckets,
                use_decode_trace=use_decode_trace,
                use_prefill_trace=use_prefill_trace,
            )
            logger.info("WARMUP complete: Prefill + Decode traces ready")
        else:
            logger.info("WARMUP complete (no traces captured)")
        logger.info("=" * 60)

    def _run_dp_vision_traced(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        frames_per_device: int,
        num_devices: int,
        max_frames_per_pool_chunk: int = 16,
    ) -> "ttnn.Tensor":
        """
        Run DP=8 vision processing with pre-captured traces and fuse into text embeddings.

        Returns hidden_states [1, 1, seq_len, hidden_dim] on device.
        """
        visual_embeddings_ttnn, valid_token = self._embed_image_data_parallel_traced(
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            frames_per_device=frames_per_device,
            num_devices=num_devices,
            max_frames_per_pool_chunk=max_frames_per_pool_chunk,
        )

        # Fuse visual embeddings with text embeddings using the existing traced fusion path
        hidden_states_ttnn = self._prepare_text_inputs_traced(
            input_ids=input_ids,
            visual_embeddings_ttnn=visual_embeddings_ttnn,
            valid_token_torch=valid_token,
        )
        return hidden_states_ttnn

    def _allocate_vision_trace_tensors(
        self,
        batch_size: int,
        n_out: int,
        k_pool: int,
        num_patches: int,
    ) -> dict:
        """Return slices into vision trace buffers pre-allocated on VisionBackbone at model init."""
        return self.model.vision_backbone.get_vision_trace_tensors(
            batch_size=batch_size,
            n_out=n_out,
            k_pool=k_pool,
            num_patches=num_patches,
        )

    def _capture_vision_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """Capture vision trace for ViT + pooling + projection."""
        logger.info("Capturing vision trace...")

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            visual_embeddings = self.model.vision_backbone.forward_ttnn(
                images_embedded=trace_tensors["embedded"],
                pooled_patches_idx_ttnn=trace_tensors["idx"],
                valid_mask_ttnn=trace_tensors["valid_mask"],
                valid_token_ttnn=trace_tensors["valid_token"],
                n_out=trace_tensors["n_out"],
                k_pool=trace_tensors["k_pool"],
                batch_size=trace_tensors["batch_size"],
                trace_capture=True,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        logger.info("Vision trace captured")

        return trace_id, visual_embeddings

    def _execute_vision_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        vision_inputs: dict,
    ) -> ttnn.Tensor:
        """Execute vision trace with new inputs."""
        # Copy new inputs to trace tensors
        ttnn.copy(vision_inputs["embedded"], trace_tensors["embedded"])
        ttnn.copy(vision_inputs["idx"], trace_tensors["idx"])
        ttnn.copy(vision_inputs["valid_mask"], trace_tensors["valid_mask"])
        ttnn.copy(vision_inputs["valid_token"], trace_tensors["valid_token"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _prepare_text_inputs_traced(
        self,
        input_ids: torch.Tensor,
        visual_embeddings_ttnn: ttnn.Tensor,
        valid_token_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Prepare text inputs using traced vision output.

        Fuses visual embeddings with text embeddings entirely on device.
        Uses matmul with selector matrix to avoid CPU roundtrip.
        Returns fused hidden states [1, 1, seq_len, hidden_dim] on device.
        """
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4096
        image_patch_id = 151938  # Molmo2 image patch token ID

        # 1. Get text embeddings on device
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        text_embeddings_ttnn = self.model.text_model.embed_tokens(input_ids_ttnn)
        ttnn.deallocate(input_ids_ttnn)

        # 2. Filter visual embeddings by valid tokens (on device using ttnn.embedding as gather)
        # valid_token_torch is [n_out] boolean, visual_embeddings_ttnn is [1, 1, n_out, hidden_dim]
        valid_indices = valid_token_torch.flatten().nonzero(as_tuple=True)[0].to(torch.int32)
        num_valid = len(valid_indices)

        if num_valid > 0:
            # Use ttnn.embedding as gather to select valid visual embeddings
            valid_indices_ttnn = ttnn.from_torch(
                valid_indices.unsqueeze(0),  # [1, num_valid]
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

            # Reshape visual embeddings for gather: [1, 1, n_out, hidden_dim] -> [n_out, hidden_dim]
            visual_for_gather = ttnn.reshape(visual_embeddings_ttnn, [1, -1, hidden_dim])

            # Gather valid embeddings: [1, num_valid, hidden_dim]
            valid_visual_ttnn = ttnn.embedding(valid_indices_ttnn, visual_for_gather)
            ttnn.deallocate(valid_indices_ttnn)

            # Reshape to 4D for matmul: [1, num_valid, hidden_dim] -> [1, 1, num_valid, hidden_dim]
            valid_visual_ttnn = ttnn.reshape(valid_visual_ttnn, [1, 1, num_valid, hidden_dim])

            # 3. Scatter visual embeddings into text positions via ttnn.embedding (no CPU loop, no matmul).
            #
            # Build a [seq_len] index vector (CPU, vectorized):
            #   visual_index[text_pos] = i  (0..num_valid-1) for the i-th image-patch position
            #   visual_index[text_pos] = num_valid  for non-image positions → zero row
            # Then extend the valid_visual table with one zero row and gather with ttnn.embedding.
            # Upload cost: [seq_len] int32 ≈ 4 KB vs old [seq_len × num_valid] bfloat16 ≈ 1.3 MB.
            image_positions = (input_ids[0] == image_patch_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == num_valid:
                # --- 3a. Build scatter index (vectorized, no Python loop) ---
                visual_index = torch.full((seq_len,), num_valid, dtype=torch.int32)
                visual_index[image_positions] = torch.arange(num_valid, dtype=torch.int32)

                # --- 3b. Extend embedding table with zero row on device ---
                # valid_visual_ttnn: [1, 1, num_valid, hidden_dim] → [num_valid, hidden_dim] ROW_MAJOR
                valid_visual_2d = ttnn.reshape(valid_visual_ttnn, [num_valid, hidden_dim])
                valid_visual_2d_rm = ttnn.to_layout(valid_visual_2d, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(valid_visual_2d)
                ttnn.deallocate(valid_visual_ttnn)

                zero_row_ttnn = ttnn.from_torch(
                    torch.zeros(1, hidden_dim, dtype=torch.bfloat16),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                # [num_valid+1, hidden_dim] ROW_MAJOR — the extra row is the "background" zero
                valid_visual_ext = ttnn.concat([valid_visual_2d_rm, zero_row_ttnn], dim=0)
                ttnn.deallocate(valid_visual_2d_rm)
                ttnn.deallocate(zero_row_ttnn)

                # --- 3c. Upload scatter index and gather (single embedding op) ---
                visual_index_ttnn = ttnn.from_torch(
                    visual_index.unsqueeze(0),  # [1, seq_len]
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                # [1, seq_len, hidden_dim]: non-image positions get the zero row
                visual_scattered = ttnn.embedding(
                    visual_index_ttnn,
                    valid_visual_ext,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(visual_index_ttnn)
                ttnn.deallocate(valid_visual_ext)

                # 4. Fuse: text_embeddings + scattered visual (on device)
                visual_scattered_4d = ttnn.reshape(visual_scattered, [1, 1, seq_len, hidden_dim])
                ttnn.deallocate(visual_scattered)
                fused_ttnn = ttnn.add(text_embeddings_ttnn, visual_scattered_4d)
                ttnn.deallocate(text_embeddings_ttnn)
                ttnn.deallocate(visual_scattered_4d)
            else:
                logger.warning(
                    f"Position mismatch: {len(image_positions)} placeholders vs {num_valid} visual tokens. "
                    "Falling back to text-only."
                )
                ttnn.deallocate(valid_visual_ttnn)
                fused_ttnn = text_embeddings_ttnn
        else:
            # No visual tokens - just use text embeddings
            fused_ttnn = text_embeddings_ttnn

        return fused_ttnn

    def _allocate_prefill_trace_tensors(
        self,
        seq_len: int,
        hidden_dim: int = 4096,
        max_num_blocks: int = 64,
    ) -> dict:
        """Pre-allocate all tensors needed for traced prefill.

        Args:
            seq_len: Sequence length for prefill
            hidden_dim: Hidden dimension size
            max_num_blocks: Maximum number of blocks per sequence for page_table
        """
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

        # Allocate page_table trace tensor for paged attention
        # Shape: [batch_size, max_num_blocks]
        trace_page_table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.batch_size, max_num_blocks]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "hidden_states": trace_hidden_states,
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
            "page_table": trace_page_table,
        }

    def _capture_prefill_trace(
        self,
        trace_tensors: dict,
    ) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for text model prefill phase."""
        logger.info("Capturing text model prefill trace...")
        rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]

        # Only pass page_table if paged attention is enabled
        page_table_for_trace = trace_tensors.get("page_table") if self.use_paged_attention else None

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            logits_trace, _ = self.model.text_model.forward(
                hidden_states=trace_tensors["hidden_states"],
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,  # Pass KV cache to fill during prefill
                rot_mats=rot_mats,
                page_table=page_table_for_trace,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        if self.use_paged_attention:
            logger.info("Text model prefill trace captured with paged attention")
        else:
            logger.info("Text model prefill trace captured (non-paged attention)")

        return trace_id, logits_trace

    def _execute_prefill_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states_ttnn: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Execute captured prefill trace with new inputs."""
        # Device-to-device copy: no host roundtrip
        ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

        # Copy page_table to trace tensor if provided (paged attention)
        if page_table is not None and "page_table" in trace_tensors:
            # Pad page_table to match trace tensor shape if needed
            trace_page_table_shape = list(trace_tensors["page_table"].shape)
            page_table_shape = list(page_table.shape)
            if page_table_shape[-1] < trace_page_table_shape[-1]:
                # Need to pad the page_table to match trace tensor size
                pad_size = trace_page_table_shape[-1] - page_table_shape[-1]
                page_table_torch = ttnn.to_torch(
                    page_table, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
                page_table_padded = torch.nn.functional.pad(page_table_torch, (0, pad_size), value=0)
                page_table_tt = ttnn.from_torch(
                    page_table_padded.unsqueeze(0) if page_table_padded.dim() == 1 else page_table_padded,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                ttnn.copy(page_table_tt, trace_tensors["page_table"])
                ttnn.deallocate(page_table_tt)
            else:
                ttnn.copy(page_table, trace_tensors["page_table"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        # trace_output is the logits buffer; callers using ttnn.to_torch should copy to CPU
        # (e.g. .detach().cpu().clone()) before another execute_trace overwrites it.
        return trace_output

    # =========================================================================
    # UNIFIED VISION + PREFILL TRACE (on-device fusion)
    # =========================================================================

    def _allocate_unified_trace_tensors(
        self,
        seq_len: int,
        num_visual_tokens: int,
        num_patches: int = 729,
        vit_hidden_dim: int = 1152,
        hidden_dim: int = 4096,
        n_out: int = 169,
        k_pool: int = 4,
        batch_size: int = 1,
    ) -> dict:
        """Allocate all tensors needed for unified vision + prefill trace.

        This includes input_ids so embed_tokens can be called inside the trace.
        """
        # Vision inputs
        trace_embedded = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * num_patches, vit_hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_idx = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, batch_size * n_out * k_pool]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_valid_mask = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, batch_size * n_out * k_pool, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        trace_valid_token = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch_size * n_out]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Input IDs for embed_tokens (called INSIDE trace)
        trace_input_ids = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, seq_len]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Selector matrix for matmul-based fusion [1, 1, seq_len, num_visual_tokens]
        # Each row has a 1 at the column corresponding to which visual token goes there
        trace_selector_matrix = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, seq_len, num_visual_tokens]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Rotation matrices for prefill
        rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)
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
        ttnn.copy(rot_mats[0], trace_cos)
        ttnn.copy(rot_mats[1], trace_sin)
        ttnn.deallocate(rot_mats[0])
        ttnn.deallocate(rot_mats[1])

        result = {
            # Vision inputs
            "embedded": trace_embedded,
            "idx": trace_idx,
            "valid_mask": trace_valid_mask,
            "valid_token": trace_valid_token,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
            # Text inputs (for embed_tokens inside trace)
            "input_ids": trace_input_ids,
            "selector_matrix": trace_selector_matrix,
            "num_visual_tokens": num_visual_tokens,
            # Prefill inputs
            "cos": trace_cos,
            "sin": trace_sin,
            "seq_len": seq_len,
        }

        # Add page_table for paged attention if enabled
        if self.use_paged_attention:
            max_num_blocks = 64  # Match decode trace allocation
            trace_page_table = ttnn.allocate_tensor_on_device(
                ttnn.Shape([self.batch_size, max_num_blocks]),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            result["page_table"] = trace_page_table

        return result

    def _capture_unified_trace(self, trace_tensors: dict) -> Tuple[int, ttnn.Tensor]:
        """
        Capture unified trace for Vision + embed_tokens + Fusion + Text Prefill.

        This eliminates the CPU roundtrip between vision and prefill by keeping
        everything on device:
        1. Vision backbone (ViT + pooling + projection)
        2. Text embeddings via embed_tokens (INSIDE trace)
        3. Fusion via matmul with selector matrix
        4. Text model forward (transformer layers + lm_head)
        """
        logger.info("Capturing unified vision + prefill trace...")

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            # Step 1: Vision backbone (ViT + pooling + projection)
            visual_embeddings = self.model.vision_backbone.forward_ttnn(
                images_embedded=trace_tensors["embedded"],
                pooled_patches_idx_ttnn=trace_tensors["idx"],
                valid_mask_ttnn=trace_tensors["valid_mask"],
                valid_token_ttnn=trace_tensors["valid_token"],
                n_out=trace_tensors["n_out"],
                k_pool=trace_tensors["k_pool"],
                batch_size=trace_tensors["batch_size"],
                trace_capture=True,
            )
            # visual_embeddings shape: [1, 1, num_visual_tokens, 4096]

            # Step 2: Text embeddings (INSIDE trace - this is key for performance)
            text_embeddings = self.model.text_model.embed_tokens(trace_tensors["input_ids"])
            # text_embeddings shape: [1, 1, seq_len, 4096]

            # Step 3: On-device fusion using matmul with selector matrix
            # selector_matrix: [1, 1, seq_len, num_visual_tokens] - one-hot rows indicating placement
            # visual_part = selector_matrix @ visual_embeddings => [1, 1, seq_len, 4096]
            # fused = text_embeddings + visual_part
            visual_part = ttnn.matmul(
                trace_tensors["selector_matrix"],
                visual_embeddings,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # visual_part: [1, 1, seq_len, 4096]

            # Add visual part to text embeddings
            fused_embed = ttnn.add(
                text_embeddings,
                visual_part,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Step 4: Text model prefill
            rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]

            # Only pass page_table if paged attention is enabled
            page_table_for_trace = trace_tensors.get("page_table") if self.use_paged_attention else None

            logits, _ = self.model.text_model.forward(
                hidden_states=fused_embed,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,
                rot_mats=rot_mats,
                page_table=page_table_for_trace,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)

        if self.use_paged_attention:
            logger.info("Unified vision + prefill trace captured with paged attention")
        else:
            logger.info("Unified vision + prefill trace captured (non-paged attention)")

        return trace_id, logits

    def _prepare_unified_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> dict:
        """
        Prepare all inputs for the unified trace.

        Note: embed_tokens is called INSIDE the trace, not here.
        This just prepares input_ids and other tensors for copying to trace tensors.
        """
        batch_size = self.batch_size
        seq_len = input_ids.shape[1]

        # Prepare vision inputs -- patch embedding on TTNN (no CPU matmul)
        vit = self.model.vision_backbone.image_vit
        embedded_ttnn = vit.patch_embed_ttnn(pixel_values)  # [1, 1, B*N, hidden_dim] on device

        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)
        clipped_idx = torch.clip(pooled_patches_idx, min=0)
        flat_idx = clipped_idx.reshape(1, -1).to(torch.int32)
        valid_mask = valid.reshape(1, 1, -1, 1).float()

        # Get number of valid visual tokens
        num_visual_tokens = valid_token.flatten().sum().item()

        # Find image token positions in input_ids (CPU - fast)
        image_patch_id = self.model.image_patch_id
        image_positions = (input_ids[0] == image_patch_id).nonzero(as_tuple=True)[0]

        # Verify counts match
        assert (
            len(image_positions) == num_visual_tokens
        ), f"Mismatch: {len(image_positions)} placeholders vs {num_visual_tokens} visual tokens"

        # Create selector matrix for matmul-based fusion: [seq_len, num_visual_tokens]
        # Row i has 1 at column j if position i should receive visual embedding j
        selector_matrix = torch.zeros(seq_len, num_visual_tokens, dtype=torch.float32)
        for j, pos in enumerate(image_positions):
            selector_matrix[pos, j] = 1.0
        # Reshape to [1, 1, seq_len, num_visual_tokens] for TTNN
        selector_matrix = selector_matrix.unsqueeze(0).unsqueeze(0)

        # Convert input_ids to TTNN (embed_tokens called inside trace)
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Convert other tensors to TTNN
        # embedded_ttnn already on device from patch_embed_ttnn above
        idx_ttnn = ttnn.from_torch(
            flat_idx,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        valid_mask_ttnn = ttnn.from_torch(
            valid_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        valid_token_ttnn = ttnn.from_torch(
            valid_token.flatten().float(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )
        selector_ttnn = ttnn.from_torch(
            selector_matrix,  # [1, 1, seq_len, num_visual_tokens]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        return {
            # Vision inputs
            "embedded": embedded_ttnn,
            "idx": idx_ttnn,
            "valid_mask": valid_mask_ttnn,
            "valid_token": valid_token_ttnn,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
            # Text inputs (embed_tokens called inside trace)
            "input_ids": input_ids_ttnn,
            "selector_matrix": selector_ttnn,
            "num_visual_tokens": num_visual_tokens,
            # Metadata
            "seq_len": seq_len,
        }

    def _execute_unified_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        inputs: dict,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Execute unified trace with new inputs."""
        # Copy vision inputs to trace tensors
        ttnn.copy(inputs["embedded"], trace_tensors["embedded"])
        ttnn.copy(inputs["idx"], trace_tensors["idx"])
        ttnn.copy(inputs["valid_mask"], trace_tensors["valid_mask"])
        ttnn.copy(inputs["valid_token"], trace_tensors["valid_token"])

        # Copy text/fusion inputs
        ttnn.copy(inputs["input_ids"], trace_tensors["input_ids"])
        ttnn.copy(inputs["selector_matrix"], trace_tensors["selector_matrix"])

        # Copy page_table if paged attention is enabled
        if page_table is not None and "page_table" in trace_tensors:
            # Pad page_table to match trace tensor shape if needed
            trace_page_table_shape = list(trace_tensors["page_table"].shape)
            page_table_shape = list(page_table.shape)
            if page_table_shape[-1] < trace_page_table_shape[-1]:
                # Need to pad the page_table to match trace tensor size
                pad_size = trace_page_table_shape[-1] - page_table_shape[-1]
                page_table_torch = ttnn.to_torch(
                    page_table, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
                page_table_padded = torch.nn.functional.pad(page_table_torch, (0, pad_size), value=0)
                page_table_tt = ttnn.from_torch(
                    page_table_padded.unsqueeze(0) if page_table_padded.dim() == 1 else page_table_padded,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                ttnn.copy(page_table_tt, trace_tensors["page_table"])
                ttnn.deallocate(page_table_tt)
            else:
                ttnn.copy(page_table, trace_tensors["page_table"])

        # Execute trace
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_output

    def _run_unified_prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
        timing: dict,
        original_seq_len: Optional[int] = None,
    ) -> Tuple[ttnn.Tensor, dict]:
        """
        Run unified Vision + Fusion + Prefill in single trace.

        This eliminates the CPU roundtrip between vision and prefill,
        keeping everything on device using scatter_add for fusion.
        """
        seq_len = input_ids.shape[1]
        if original_seq_len is None:
            original_seq_len = seq_len
        logger.info("Running unified Vision + Prefill trace...")

        # Prepare all inputs (vision + text embeddings + fusion indices)
        prep_start = time.perf_counter()
        inputs = self._prepare_unified_inputs(input_ids, pixel_values, pooled_patches_idx)
        timing["prep_ms"] = (time.perf_counter() - prep_start) * 1000
        logger.info(f"Input preparation: {timing['prep_ms']:.2f}ms")

        # Check if we have a cached trace for this configuration
        trace_key = (seq_len, inputs["num_visual_tokens"], inputs["n_out"], inputs["k_pool"])

        if not hasattr(self, "unified_traces"):
            self.unified_traces = {}

        if trace_key not in self.unified_traces:
            # First run: warmup + capture trace
            logger.info("Running unified warmup (compile)...")
            warmup_start = time.perf_counter()

            # Run full pipeline once for compilation (same ops as trace)
            # Step 1: Vision
            visual_embeddings = self.model.vision_backbone.forward_ttnn(
                images_embedded=inputs["embedded"],
                pooled_patches_idx_ttnn=inputs["idx"],
                valid_mask_ttnn=inputs["valid_mask"],
                valid_token_ttnn=inputs["valid_token"],
                n_out=inputs["n_out"],
                k_pool=inputs["k_pool"],
                batch_size=inputs["batch_size"],
            )

            # Step 2: Text embeddings (compile embed_tokens)
            text_embeddings = self.model.text_model.embed_tokens(inputs["input_ids"])

            # Step 3: Fusion via matmul with selector matrix
            visual_part = ttnn.matmul(
                inputs["selector_matrix"],
                visual_embeddings,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            fused_embed = ttnn.add(
                text_embeddings,
                visual_part,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Step 4: Text prefill
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

            # Initialize page table for paged attention warmup
            warmup_page_table = None
            if self.use_paged_attention:
                self.init_page_table(seq_len)
                warmup_page_table = self.page_table

            logits, _ = self.model.text_model.forward(
                hidden_states=fused_embed,
                start_pos=0,
                attn_mask=None,
                kv_caches=self.kv_caches,
                rot_mats=rot_mats,
                page_table=warmup_page_table,
            )
            ttnn.synchronize_device(self.mesh_device)
            timing["compile_ms"] = (time.perf_counter() - warmup_start) * 1000
            logger.info(f"Unified compile completed in {timing['compile_ms']:.2f}ms")

            # Deallocate warmup outputs
            ttnn.deallocate(visual_embeddings)
            ttnn.deallocate(text_embeddings)
            ttnn.deallocate(visual_part)
            ttnn.deallocate(fused_embed)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])
            ttnn.deallocate(logits)

            # Allocate trace tensors
            logger.info("Allocating unified trace tensors...")
            trace_tensors = self._allocate_unified_trace_tensors(
                seq_len=seq_len,
                num_visual_tokens=inputs["num_visual_tokens"],
                n_out=inputs["n_out"],
                k_pool=inputs["k_pool"],
                batch_size=inputs["batch_size"],
            )

            # Copy initial data to trace tensors
            ttnn.copy(inputs["embedded"], trace_tensors["embedded"])
            ttnn.copy(inputs["idx"], trace_tensors["idx"])
            ttnn.copy(inputs["valid_mask"], trace_tensors["valid_mask"])
            ttnn.copy(inputs["valid_token"], trace_tensors["valid_token"])
            ttnn.copy(inputs["input_ids"], trace_tensors["input_ids"])
            ttnn.copy(inputs["selector_matrix"], trace_tensors["selector_matrix"])

            # Synchronize before trace capture to ensure all copies are complete
            ttnn.synchronize_device(self.mesh_device)

            # Capture trace
            trace_id, trace_output = self._capture_unified_trace(trace_tensors)
            self.unified_traces[trace_key] = (trace_id, trace_tensors, trace_output)

        trace_id, trace_tensors, trace_output = self.unified_traces[trace_key]

        # Initialize page table for paged attention if enabled
        effective_page_table = None
        if self.use_paged_attention:
            self.init_page_table(seq_len)
            effective_page_table = self.page_table

        # Execute trace (actual timing measurement)
        ttft_start = time.perf_counter()
        logits = self._execute_unified_trace(
            trace_id, trace_tensors, trace_output, inputs, page_table=effective_page_table
        )
        ttnn.synchronize_device(self.mesh_device)
        timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000
        timing["vision_ms"] = 0  # Included in ttft_ms for unified trace

        logger.info(f"Unified TTFT: {timing['ttft_ms']:.2f}ms")

        # Cleanup temporary input tensors (trace tensors are reused)
        ttnn.deallocate(inputs["embedded"])
        ttnn.deallocate(inputs["idx"])
        ttnn.deallocate(inputs["valid_mask"])
        ttnn.deallocate(inputs["valid_token"])
        ttnn.deallocate(inputs["input_ids"])
        ttnn.deallocate(inputs["selector_matrix"])

        # Update position for decode using ORIGINAL seq_len (not padded)
        self.reset_kv_cache(original_seq_len)

        return logits, timing

    def _allocate_decode_trace_tensors(self, hidden_dim: int = 4096, max_num_blocks: int = 64) -> dict:
        """Allocate tensors needed for traced decode.

        Args:
            hidden_dim: Hidden dimension size
            max_num_blocks: Maximum number of blocks per sequence for page_table
        """
        trace_hidden_states = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.batch_size, hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Allocate page_table trace tensor for paged attention
        # Shape: [batch_size, max_num_blocks]
        trace_page_table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.batch_size, max_num_blocks]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "hidden_states": trace_hidden_states,
            "page_table": trace_page_table,
        }

    def _copy_decode_trace_inputs(
        self,
        trace_tensors: dict,
        hidden_states: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> None:
        """Copy runtime decode inputs into persistent trace input buffers."""
        ttnn.copy(hidden_states, trace_tensors["hidden_states"])

        if page_table is not None and "page_table" in trace_tensors:
            trace_page_table_shape = list(trace_tensors["page_table"].shape)
            page_table_shape = list(page_table.shape)
            if page_table_shape[-1] < trace_page_table_shape[-1]:
                pad_size = trace_page_table_shape[-1] - page_table_shape[-1]
                page_table_torch = ttnn.to_torch(
                    page_table, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                )[0]
                page_table_padded = torch.nn.functional.pad(page_table_torch, (0, pad_size), value=0)
                page_table_tt = ttnn.from_torch(
                    page_table_padded.unsqueeze(0) if page_table_padded.dim() == 1 else page_table_padded,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                ttnn.copy(page_table_tt, trace_tensors["page_table"])
                ttnn.deallocate(page_table_tt)
            else:
                ttnn.copy(page_table, trace_tensors["page_table"])

    def _capture_decode_trace(
        self,
        trace_tensors: dict,
        hidden_states: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> Tuple[int, ttnn.Tensor]:
        """Capture trace for decode phase (single token generation).

        Match tt_transformers ``Generator._capture_decode_trace_text``: copy inputs into
        trace buffers, then record a single ``forward_decode`` (rot_mats path). Do not run
        an extra eager forward through trace tensors before capture.

        The RoPE embedding lookup reads from self.rot_mat_idxs (managed via
        ttnn.plus_one outside the trace). KV cache position reads from
        self.current_pos (also managed via ttnn.plus_one outside the trace).

        When paged attention is enabled, the trace includes paged attention operations
        with page_table as an input. Before capture (and each replay), the actual
        page_table values are copied to the trace_tensors["page_table"] tensor.
        """
        logger.info("Capturing decode trace...")
        self._copy_decode_trace_inputs(trace_tensors, hidden_states, page_table)

        page_table_for_trace = trace_tensors.get("page_table") if self.use_paged_attention else None

        # Compile warmup: run decode path once to trigger JIT compilation BEFORE trace capture.
        # forward_decode deallocates its input hidden_states; never run compile on
        # trace_tensors["hidden_states"] — use a scratch copy.
        logger.info("Compile warmup: decode forward (rot_mats path, scratch buffer)...")
        rot_mats_warmup = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)
        compile_hidden = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(trace_tensors["hidden_states"].shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(trace_tensors["hidden_states"], compile_hidden)
        compile_logits = self.model.text_model.forward_decode(
            hidden_states=compile_hidden,
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
            rot_mats=rot_mats_warmup,
            page_table=page_table_for_trace,
        )
        ttnn.deallocate(compile_hidden)
        ttnn.deallocate(compile_logits)

        tok = trace_capture_run_begin()
        try:
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

            # RoPE embedding lookup INSIDE trace capture so it uses updated rot_mat_idxs
            # on each trace replay. This is critical for correct position encoding.
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_decode_traced(self.rot_mat_idxs)

            logits_trace = self.model.text_model.forward_decode(
                hidden_states=trace_tensors["hidden_states"],
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mats=rot_mats,
                page_table=page_table_for_trace,
            )

            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        finally:
            trace_capture_run_end(tok)
        # Drain the command queue after capture. The allocator may still emit a one-time
        # hint: "Allocating device buffers is unsafe due to the existence of an active trace"
        # when the next step allocates before the first execute_trace — that is expected and OK.
        ttnn.synchronize_device(self.mesh_device)
        if self.use_paged_attention:
            logger.info("Decode trace captured with paged attention support")
        else:
            logger.info("Decode trace captured (non-paged attention)")

        return trace_id, logits_trace

    def _execute_decode_trace(
        self,
        trace_id: int,
        trace_tensors: dict,
        trace_output: ttnn.Tensor,
        hidden_states: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Execute captured decode trace with new inputs.

        TTTransformers pattern: Track state changes and force full synchronization
        when switching between requests. This prevents stale state from causing
        incorrect behavior across multiple sequential inferences.

        Position tensors (current_pos, rot_mat_idxs) are kept on device and
        incremented via ttnn.plus_one in run_decode_step after trace execution.
        The trace reads their current values for RoPE and KV cache updates.

        For paged attention, the page_table values are copied to the trace
        tensor before execution so the trace uses the correct block mappings.

        Use blocking=True so mesh trace replay finishes before we read logits
        (non-blocking + immediate to_torch/argmax can sample stale/corrupt output).
        """
        # TTTransformers pattern: detect if inputs need full re-synchronization
        # This is critical for correct behavior across multiple sequential requests
        reset_inputs = self.decode_trace_needs_reset

        # Check if page_table has changed (TTTransformers tracks prev_page_table)
        if page_table is not None:
            if self.prev_page_table is None:
                reset_inputs = True
            else:
                # Compare page tables - trigger reset if different
                try:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                    curr_pt = ttnn.to_torch(page_table, mesh_composer=mesh_composer)[0]
                    prev_pt = ttnn.to_torch(self.prev_page_table, mesh_composer=mesh_composer)[0]
                    if not torch.equal(curr_pt, prev_pt):
                        reset_inputs = True
                except Exception:
                    # If comparison fails, force reset to be safe
                    reset_inputs = True

        if reset_inputs:
            # Full synchronization: ensure device state is consistent before copying
            ttnn.synchronize_device(self.mesh_device)
            logger.debug("Decode trace: full input reset (new request or page_table changed)")

            # Update page_table tracking
            if page_table is not None:
                self.prev_page_table = page_table
            self.decode_trace_needs_reset = False

        # Copy inputs to trace tensors
        self._copy_decode_trace_inputs(trace_tensors, hidden_states, page_table)

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        return trace_output

    def warmup_all_buckets(
        self,
        bucket_sizes: List[int] = None,
        use_decode_trace: bool = True,
        use_prefill_trace: bool = True,
    ):
        """
        Warmup prefill traces for all bucket sizes and decode trace.

        This ensures any input size will have a matching trace ready.

        Args:
            bucket_sizes: List of bucket sizes to warmup (default: [128, 1024, 2048, 4096])
            use_decode_trace: Whether to also warmup decode trace
            use_prefill_trace: Whether to capture prefill traces (disable for debugging)
        """
        if bucket_sizes is None:
            # Cover: text-only (128), image (256-1024), video (1024-4096)
            # Note: 8k/16k buckets cause OOM during warmup - add when memory optimized
            bucket_sizes = [128, 256, 512, 1024, 2048, 4096]

        hidden_dim = 4096
        logger.info(f"Warming up prefill traces for buckets: {bucket_sizes}")

        # Calculate max_num_blocks for paged attention
        max_num_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size

        # Initialize KV cache BEFORE capturing traces so KV ops are traced
        if self.kv_caches is None:
            self.init_kv_cache()

        # Initialize a page table for warmup (sequential block mapping)
        # Shape: [batch_size, max_num_blocks] - replicate same mapping for all batches
        warmup_page_table_torch = (
            torch.arange(max_num_blocks, dtype=torch.int32).unsqueeze(0).expand(self.batch_size, -1).contiguous()
        )
        warmup_page_table = ttnn.from_torch(
            warmup_page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Phase 1: Allocate all trace tensors BEFORE capturing any traces so we do not
        # allocate during trace recording (allocator would complain during capture).
        all_trace_tensors = {}
        valid_buckets = []
        for seq_len in bucket_sizes:
            if seq_len > self.max_seq_len:
                logger.info(f"  Skipping bucket {seq_len} (exceeds max_seq_len {self.max_seq_len})")
                continue
            valid_buckets.append(seq_len)
            logger.info(f"  Allocating trace tensors for bucket {seq_len}...")
            trace_tensors = self._allocate_prefill_trace_tensors(seq_len, hidden_dim, max_num_blocks=max_num_blocks)
            if "page_table" in trace_tensors:
                ttnn.copy(warmup_page_table, trace_tensors["page_table"])
            all_trace_tensors[seq_len] = trace_tensors

        # Phase 2: Warmup and capture traces for each bucket
        for seq_len in valid_buckets:
            logger.info(f"  Warming up bucket {seq_len}...")
            trace_tensors = all_trace_tensors[seq_len]

            # Create dummy hidden states
            dummy_hidden = torch.zeros(1, seq_len, hidden_dim, dtype=torch.bfloat16)
            hidden_states_ttnn = ttnn.from_torch(
                dummy_hidden.unsqueeze(0),  # [1, 1, seq_len, hidden_dim]
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

            # Reset KV cache position before each bucket warmup
            self.reset_kv_cache(0)

            # Warmup (compile)
            self.warmup_prefill(hidden_states_ttnn, trace_tensors, use_trace=use_prefill_trace, attn_mask=None)

            if use_prefill_trace:
                # Capture trace
                trace_id, trace_output = self._capture_prefill_trace(trace_tensors)
                self.prefill_traces[seq_len] = (trace_id, trace_tensors, trace_output)
                logger.info(f"  Bucket {seq_len} trace captured")
            else:
                logger.info(f"  Bucket {seq_len} compile complete (no trace)")

            ttnn.deallocate(hidden_states_ttnn)

        # Warmup decode trace
        if use_decode_trace:
            logger.info("Warming up decode trace...")

            # Reset position for decode warmup
            self.reset_kv_cache(0)

            # Allocate decode trace tensors
            self.decode_trace_tensors = self._allocate_decode_trace_tensors(
                hidden_dim=hidden_dim, max_num_blocks=max_num_blocks
            )

            # Initialize page_table trace tensor with sequential block mapping
            if "page_table" in self.decode_trace_tensors:
                decode_page_table_torch = (
                    torch.arange(max_num_blocks, dtype=torch.int32)
                    .unsqueeze(0)
                    .expand(self.batch_size, -1)
                    .contiguous()
                )
                decode_page_table = ttnn.from_torch(
                    decode_page_table_torch,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self.mesh_mapper,
                )
                ttnn.copy(decode_page_table, self.decode_trace_tensors["page_table"])
                ttnn.deallocate(decode_page_table)

            # Create dummy hidden states for decode (one token per batch element)
            dummy_decode = torch.zeros(1, self.batch_size, hidden_dim, dtype=torch.bfloat16)
            decode_hidden = ttnn.from_torch(
                dummy_decode.unsqueeze(0),  # [1, 1, batch_size, hidden_dim]
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

            # Compile decode without trace (matches tt_transformers; avoids eager forward
            # through trace tensors before capture). Only pass page_table when using paged
            # KV — non-paged cache + paged_update_cache + wide page_table hits TT_FATAL.
            pt = warmup_page_table if self.use_paged_attention else None
            self.warmup_decode(decode_hidden, page_table=pt)
            ttnn.deallocate(decode_hidden)
            decode_hidden = ttnn.from_torch(
                dummy_decode.unsqueeze(0),  # [1, 1, batch_size, hidden_dim]
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.mesh_mapper,
            )

            trace_id, trace_output = self._capture_decode_trace(self.decode_trace_tensors, decode_hidden, page_table=pt)
            self.decode_trace_id = trace_id
            self.decode_trace_output = trace_output

            ttnn.deallocate(decode_hidden)
            logger.info("Decode trace captured")

        # Cleanup warmup page table (must be after decode warmup which uses it for paged attention)
        ttnn.deallocate(warmup_page_table)

        logger.info(
            f"Warmup complete: {len(self.prefill_traces)} prefill buckets, decode={'yes' if use_decode_trace else 'no'}"
        )

    def _build_mm_prefill_attn_mask(
        self,
        token_type_ids: Optional[torch.Tensor],
        hf_attention_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Optional[ttnn.Tensor]:
        """HF-style multimodal prefill additive mask; ``None`` if not applicable."""
        if token_type_ids is None or seq_len <= 1:
            return None
        if token_type_ids.shape[1] != seq_len:
            raise ValueError(f"token_type_ids length {token_type_ids.shape[1]} != hidden seq_len {seq_len}")
        if hf_attention_mask is not None and hf_attention_mask.shape[1] != seq_len:
            raise ValueError(f"hf_attention_mask length {hf_attention_mask.shape[1]} != seq_len {seq_len}")
        bias = build_molmo2_prefill_attention_bias(token_type_ids, attention_mask=hf_attention_mask).to(torch.bfloat16)
        is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
        mm = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None
        return ttnn.from_torch(
            bias,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )

    def warmup_prefill(
        self,
        hidden_states_ttnn: ttnn.Tensor,
        trace_tensors: dict,
        use_trace: bool,
        page_table: Optional[ttnn.Tensor] = None,
        attn_mask: Optional[ttnn.Tensor] = None,
    ):
        """Run prefill warm-up (compile) pass."""
        logger.info("Running prefill warm-up (compile)...")
        start = time.perf_counter()

        if use_trace:
            # Copy hidden states to trace tensor
            ttnn.copy(hidden_states_ttnn, trace_tensors["hidden_states"])

            # Run forward to compile - MUST pass kv_caches to compile fill_cache ops
            # Also pass page_table to compile paged attention ops
            rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
            effective_page_table = trace_tensors.get("page_table")  # Get page_table from trace_tensors if available
            logits, _ = self.model.text_model.forward(
                hidden_states=trace_tensors["hidden_states"],
                start_pos=0,
                attn_mask=attn_mask,
                kv_caches=self.kv_caches,  # Pass KV cache to compile fill_cache
                rot_mats=rot_mats,
                page_table=effective_page_table,  # Compile paged attention ops
            )
        else:
            # Non-traced path: use page_table argument if provided, else try trace_tensors
            if page_table is not None:
                effective_page_table = page_table
            elif trace_tensors is not None and self.use_paged_attention:
                effective_page_table = trace_tensors.get("page_table")
            else:
                effective_page_table = None

            # Get rot_mats from trace_tensors if available, else compute them
            if trace_tensors is not None:
                rot_mats = [trace_tensors["cos"], trace_tensors["sin"]]
            else:
                seq_len = hidden_states_ttnn.shape[-2]
                rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, start_pos=0)

            logits, _ = self.model.text_model.forward(
                hidden_states=hidden_states_ttnn,
                start_pos=0,
                attn_mask=attn_mask,
                kv_caches=self.kv_caches,  # Also pass KV cache for non-traced warmup
                rot_mats=rot_mats,
                page_table=effective_page_table,  # Pass page_table for paged attention
            )

        compile_time = (time.perf_counter() - start) * 1000
        logger.info(f"Prefill compile completed in {compile_time:.2f}ms")
        return compile_time

    def warmup_decode(
        self,
        hidden_states: ttnn.Tensor,
        page_table: Optional[ttnn.Tensor] = None,
    ):
        """Run decode warm-up (compile) pass without trace (rot_mat_idxs path)."""
        logger.info("Running decode warm-up (compile)...")
        start = time.perf_counter()

        self.model.text_model.forward_decode(
            hidden_states=hidden_states,
            kv_caches=self.kv_caches,
            current_pos=self.current_pos,
            rot_mat_idxs=self.rot_mat_idxs,
            page_table=page_table,
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
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
        use_dp_vision_trace: bool = False,
        page_table: Optional[ttnn.Tensor] = None,
        user_id: int = 0,
        use_data_parallel: bool = False,
        frames_per_device: int = 8,
        token_type_ids: Optional[torch.Tensor] = None,
        hf_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, dict]:
        """
        Run prefill phase (process prompt + image).

        Args:
            input_ids: Token IDs
            pixel_values: Image tensor
            pooled_patches_idx: Patch pooling indices
            use_trace: Whether to trace text model prefill
            use_vision_trace: Whether to trace vision backbone (ViT + pooling)
            use_unified_trace: Whether to use unified Vision+Prefill trace (eliminates CPU roundtrip)
            use_dp_vision_trace: Use pre-captured DP=8 ViT + pool traces (video path)
            page_table: Optional page table for paged attention (vLLM)
            user_id: Batch index for multi-user batching (determines KV cache slot)
            token_type_ids: Optional HF ``[B, S]`` multimodal token types (non-zero = image/mm)
            hf_attention_mask: Optional HF padding mask ``[B, S]`` (1 = valid)

        Returns:
            Tuple of (logits, timing_dict)
        """
        # Initialize KV cache if needed
        self.init_kv_cache()

        # Pad input_ids to next bucket size for trace reuse
        original_seq_len = input_ids.shape[1]
        input_ids, seq_len, _ = pad_input_ids(input_ids, pad_token_id=0)
        if seq_len != original_seq_len:
            logger.info(f"Padded input_ids from {original_seq_len} to {seq_len} for trace reuse")

        token_type_ids = pad_seq_2d_right(
            token_type_ids,
            original_len=original_seq_len,
            padded_len=seq_len,
            pad_value=0,
        )
        hf_attention_mask = pad_seq_2d_right(
            hf_attention_mask,
            original_len=original_seq_len,
            padded_len=seq_len,
            pad_value=0,
        )

        prefill_attn_mask_ttnn = self._build_mm_prefill_attn_mask(token_type_ids, hf_attention_mask, seq_len)

        if token_type_ids is not None and (use_trace or use_unified_trace):
            logger.warning(
                "Disabling prefill/unified trace: multimodal attention mask (token_type_ids) is incompatible with traced SDPA."
            )
            use_trace = False
            use_unified_trace = False

        # Initialize page table for paged attention (demo mode)
        # If page_table is provided externally (vLLM), use that instead
        effective_page_table = page_table
        if effective_page_table is None and self.use_paged_attention:
            self.init_page_table(seq_len)
            effective_page_table = self.page_table

        timing = {}

        # Unified trace path: Vision + embed_tokens + Fusion + Prefill in single trace
        # This eliminates CPU roundtrip between vision and text prefill
        if use_unified_trace and pixel_values is not None and prefill_attn_mask_ttnn is None:
            logits, timing = self._run_unified_prefill(
                input_ids, pixel_values, pooled_patches_idx, timing, original_seq_len
            )
            # Store original_seq_len in timing for correct logits indexing
            timing["original_seq_len"] = original_seq_len
            return logits, timing

        # Start end-to-end TTFT timer (vision + fusion + prefill)
        e2e_ttft_start = None

        if use_dp_vision_trace and self.dp_vit_trace_id is not None and pixel_values is not None:
            # DP=8 ViT + pool trace path (video)
            e2e_ttft_start = time.perf_counter()
            logger.info("Using pre-captured DP=8 ViT + pool traces for vision processing...")
            vision_start = time.perf_counter()
            hidden_states_ttnn = self._run_dp_vision_traced(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pooled_patches_idx=pooled_patches_idx,
                frames_per_device=self.dp_vit_frames_per_device,
                num_devices=self.dp_vit_num_devices,
            )
            timing["vision_ms"] = (time.perf_counter() - vision_start) * 1000
            logger.info(f"Vision (DP traced) completed in {timing['vision_ms']:.2f}ms")
        elif use_vision_trace and pixel_values is not None:
            # Vision tracing path - uses forward_ttnn with TTNN-native gather
            logger.info("Preparing vision inputs for tracing...")
            vision_prep_start = time.perf_counter()
            vision_inputs = self._prepare_vision_inputs_for_trace(pixel_values, pooled_patches_idx)
            timing["vision_prep_ms"] = (time.perf_counter() - vision_prep_start) * 1000

            # Check if we need to capture a new trace
            if self.vision_trace_id is None:
                # First run: warmup + capture trace
                logger.info("Running vision warmup (compile)...")
                warmup_start = time.perf_counter()
                warmup_output = self.model.vision_backbone.forward_ttnn(
                    images_embedded=vision_inputs["embedded"],
                    pooled_patches_idx_ttnn=vision_inputs["idx"],
                    valid_mask_ttnn=vision_inputs["valid_mask"],
                    valid_token_ttnn=vision_inputs["valid_token"],
                    n_out=vision_inputs["n_out"],
                    k_pool=vision_inputs["k_pool"],
                    batch_size=vision_inputs["batch_size"],
                )
                ttnn.synchronize_device(self.mesh_device)
                timing["vision_compile_ms"] = (time.perf_counter() - warmup_start) * 1000
                logger.info(f"Vision compile completed in {timing['vision_compile_ms']:.2f}ms")
                ttnn.deallocate(warmup_output)

                # Vision trace I/O buffers are pre-allocated on VisionBackbone at model init; slice to this run
                logger.info("Binding vision trace tensor views...")
                num_patches = vision_inputs["embedded"].shape[2] // vision_inputs["batch_size"]
                self.vision_trace_tensors = self._allocate_vision_trace_tensors(
                    batch_size=vision_inputs["batch_size"],
                    n_out=vision_inputs["n_out"],
                    k_pool=vision_inputs["k_pool"],
                    num_patches=num_patches,
                )

                # Copy initial data to trace tensors
                ttnn.copy(vision_inputs["embedded"], self.vision_trace_tensors["embedded"])
                ttnn.copy(vision_inputs["idx"], self.vision_trace_tensors["idx"])
                ttnn.copy(vision_inputs["valid_mask"], self.vision_trace_tensors["valid_mask"])
                ttnn.copy(vision_inputs["valid_token"], self.vision_trace_tensors["valid_token"])

                # Capture trace
                self.vision_trace_id, self.vision_trace_outputs = self._capture_vision_trace(self.vision_trace_tensors)

            # Execute vision trace - START of end-to-end TTFT measurement
            e2e_ttft_start = time.perf_counter()
            vision_trace_start = time.perf_counter()
            # Copy new inputs to trace tensors
            ttnn.copy(vision_inputs["embedded"], self.vision_trace_tensors["embedded"])
            ttnn.copy(vision_inputs["idx"], self.vision_trace_tensors["idx"])
            ttnn.copy(vision_inputs["valid_mask"], self.vision_trace_tensors["valid_mask"])
            ttnn.copy(vision_inputs["valid_token"], self.vision_trace_tensors["valid_token"])
            # Execute trace
            ttnn.execute_trace(self.mesh_device, self.vision_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            timing["vision_trace_ms"] = (time.perf_counter() - vision_trace_start) * 1000
            logger.info(f"Vision trace executed in {timing['vision_trace_ms']:.2f}ms")

            # Get visual embeddings from trace output
            visual_embeddings_ttnn = self.vision_trace_outputs

            # Fuse visual embeddings with text embeddings
            logger.info("Fusing visual and text embeddings...")
            fuse_start = time.perf_counter()
            hidden_states_ttnn = self._prepare_text_inputs_traced(
                input_ids, visual_embeddings_ttnn, vision_inputs["valid_token_torch"]
            )
            timing["fuse_ms"] = (time.perf_counter() - fuse_start) * 1000

            # Cleanup temporary vision input tensors (trace tensors are reused)
            ttnn.deallocate(vision_inputs["embedded"])
            ttnn.deallocate(vision_inputs["idx"])
            ttnn.deallocate(vision_inputs["valid_mask"])
            ttnn.deallocate(vision_inputs["valid_token"])

            # Total vision time
            timing["vision_ms"] = (
                timing.get("vision_prep_ms", 0) + timing.get("vision_trace_ms", 0) + timing.get("fuse_ms", 0)
            )
            logger.info(f"Vision processing completed in {timing['vision_ms']:.2f}ms (traced)")
        else:
            # Original path - no vision tracing
            # Separate compile pass from timed inference, mirroring the non-traced prefill path.
            if not self._vision_eager_compiled and pixel_values is not None:
                logger.info("Running eager vision compile pass (ViT + pool)...")
                compile_vision_start = time.perf_counter()
                warmup_hs = self._prepare_text_inputs(
                    input_ids,
                    pixel_values,
                    pooled_patches_idx,
                    use_data_parallel=use_data_parallel,
                    frames_per_device=frames_per_device,
                )
                ttnn.synchronize_device(self.mesh_device)
                timing["compile_vision_ms"] = (time.perf_counter() - compile_vision_start) * 1000
                ttnn.deallocate(warmup_hs)
                self._vision_eager_compiled = True
                logger.info(f"Eager vision compile completed in {timing['compile_vision_ms']:.2f}ms")

            # START of end-to-end TTFT measurement (compile-free)
            e2e_ttft_start = time.perf_counter()
            logger.info("Preparing inputs (vision processing)...")
            vision_start = time.perf_counter()
            hidden_states_ttnn = self._prepare_text_inputs(
                input_ids,
                pixel_values,
                pooled_patches_idx,
                use_data_parallel=use_data_parallel,
                frames_per_device=frames_per_device,
            )
            timing["vision_ms"] = (time.perf_counter() - vision_start) * 1000
            logger.info(f"Vision processing completed in {timing['vision_ms']:.2f}ms")

        if use_trace:
            # Prefill traces must be captured upfront (warmup_all_buckets / warmup_video_traces).
            # No lazy allocation or capture during inference — avoids compile in the TTFT path.
            if seq_len not in self.prefill_traces:
                raise RuntimeError(
                    f"Prefill trace for seq_len={seq_len} not found. "
                    "Call warmup_all_buckets(...) or warmup_video_traces(...) before run_prefill with use_trace=True."
                )

            trace_id, trace_tensors, trace_output = self.prefill_traces[seq_len]

            # Execute trace (actual TTFT measurement)
            ttft_start = time.perf_counter()
            logits = self._execute_prefill_trace(
                trace_id, trace_tensors, trace_output, hidden_states_ttnn, page_table=effective_page_table
            )
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

            ttnn.deallocate(hidden_states_ttnn)
        else:
            # Warm-up (compile) - use smaller chunk for warmup if chunked prefill
            compile_seq_len = min(seq_len, self.max_prefill_chunk_size)
            if compile_seq_len < seq_len and prefill_attn_mask_ttnn is None:
                # Create a smaller hidden_states for compilation
                warmup_hidden_states = ttnn.slice(
                    hidden_states_ttnn,
                    (0, 0, 0, 0),
                    (1, 1, compile_seq_len, 4096),
                )
                timing["compile_prefill_ms"] = self.warmup_prefill(
                    warmup_hidden_states, None, use_trace=False, page_table=effective_page_table, attn_mask=None
                )
                ttnn.deallocate(warmup_hidden_states)
            else:
                timing["compile_prefill_ms"] = self.warmup_prefill(
                    hidden_states_ttnn,
                    None,
                    use_trace=False,
                    page_table=effective_page_table,
                    attn_mask=prefill_attn_mask_ttnn,
                )

            # Actual prefill (TTFT) - use chunked prefill for long sequences
            ttft_start = time.perf_counter()

            if seq_len > self.max_prefill_chunk_size and self.use_paged_attention and prefill_attn_mask_ttnn is None:
                # Chunked prefill for long sequences
                logger.info(f"Using chunked prefill: seq_len={seq_len}, chunk_size={self.max_prefill_chunk_size}")
                logits, last_chunk_size = self._run_chunked_prefill(
                    hidden_states_ttnn,
                    seq_len,
                    effective_page_table,
                    user_id,
                )
                # For chunked prefill, logits are only for the last chunk
                # Store the index within the last chunk for correct logits access
                timing["chunked_prefill"] = True
                timing["last_chunk_size"] = last_chunk_size
                # Compute the position within the last chunk
                # original_seq_len may be less than seq_len (padded)
                last_chunk_start = ((original_seq_len - 1) // self.max_prefill_chunk_size) * self.max_prefill_chunk_size
                timing["last_token_idx_in_chunk"] = original_seq_len - 1 - last_chunk_start
            else:
                # Standard prefill for short sequences (or long with multimodal mask — chunking unsupported)
                if seq_len > self.max_prefill_chunk_size and prefill_attn_mask_ttnn is not None:
                    logger.warning(
                        "Long sequence with multimodal attention mask: using full prefill (chunked path has no mask support)"
                    )
                logits, _ = self.model.text_model.forward(
                    hidden_states=hidden_states_ttnn,
                    start_pos=0,
                    attn_mask=prefill_attn_mask_ttnn,
                    kv_caches=self.kv_caches,  # Pass pre-allocated cache to fill
                    page_table=effective_page_table,
                    user_id=user_id,
                )
                if prefill_attn_mask_ttnn is not None:
                    ttnn.deallocate(prefill_attn_mask_ttnn)
            ttnn.synchronize_device(self.mesh_device)
            timing["ttft_ms"] = (time.perf_counter() - ttft_start) * 1000

        # Calculate end-to-end TTFT (vision start to prefill end)
        if e2e_ttft_start is not None:
            timing["e2e_ttft_ms"] = (time.perf_counter() - e2e_ttft_start) * 1000
            logger.info(f"End-to-end TTFT (vision + fusion + prefill): {timing['e2e_ttft_ms']:.2f}ms")

        logger.info(f"Prefill-only TTFT: {timing['ttft_ms']:.2f}ms")

        # Update position for decode using ORIGINAL seq_len (not padded)
        # The decode step should continue from the actual end of the prompt
        self.reset_kv_cache(original_seq_len)

        # Store original_seq_len in timing so caller can index logits correctly
        timing["original_seq_len"] = original_seq_len

        return logits, timing

    def _run_chunked_prefill(
        self,
        hidden_states_ttnn: ttnn.Tensor,
        seq_len: int,
        page_table: ttnn.Tensor,
        user_id: int = 0,
    ) -> Tuple[ttnn.Tensor, int]:
        """
        Run prefill in chunks to avoid OOM for long sequences.

        For sequences longer than max_prefill_chunk_size, processes in chunks
        using chunked_scaled_dot_product_attention which reads previous KV from cache.

        Args:
            hidden_states_ttnn: Full hidden states [1, 1, seq_len, hidden_dim]
            seq_len: Total sequence length
            page_table: Page table for paged attention
            user_id: User ID for KV cache slot

        Returns:
            Tuple of (logits from the last chunk, size of the last chunk)
        """
        chunk_size = self.max_prefill_chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        logger.info(f"Chunked prefill: {num_chunks} chunks of {chunk_size} tokens")

        logits = None

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            actual_chunk_size = chunk_end - chunk_start

            logger.info(
                f"CHUNKED PREFILL: chunk {chunk_idx + 1}/{num_chunks}: positions {chunk_start}-{chunk_end}, size={actual_chunk_size}"
            )

            # Slice hidden states for this chunk
            chunk_hidden = ttnn.slice(
                hidden_states_ttnn,
                (0, 0, chunk_start, 0),
                (1, 1, chunk_end, 4096),
            )

            # Get rotation matrices for this chunk's positions
            rot_mats = self.model.text_model.rotary_setup.get_rot_mats_prefill(actual_chunk_size, start_pos=chunk_start)

            # Compute chunk page table (pages for this chunk)
            blocks_per_chunk = (chunk_size + self.block_size - 1) // self.block_size
            chunk_start_block = chunk_start // self.block_size
            chunk_end_block = (chunk_end + self.block_size - 1) // self.block_size

            # Get page table for this chunk
            chunk_page_table_torch = self.page_table_torch[:, chunk_start_block:chunk_end_block]
            chunk_page_table = ttnn.from_torch(
                chunk_page_table_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # Forward pass with chunked attention
            chunk_logits, _ = self.model.text_model.forward(
                hidden_states=chunk_hidden,
                start_pos=chunk_start,
                attn_mask=None,
                kv_caches=self.kv_caches,
                rot_mats=rot_mats,
                page_table=page_table,  # Full page table for reading previous KV
                user_id=user_id,
                chunk_page_table=chunk_page_table,  # Chunk page table for writing new KV
                chunk_start_idx=chunk_start,  # Enables chunked attention
            )

            ttnn.deallocate(chunk_hidden)
            ttnn.deallocate(chunk_page_table)

            # Keep logits from the last chunk
            if chunk_idx == num_chunks - 1:
                logits = chunk_logits
                last_chunk_size = actual_chunk_size
            else:
                ttnn.deallocate(chunk_logits)

        return logits, last_chunk_size

    def run_decode_step(
        self,
        token_id_ttnn: ttnn.Tensor,
        use_trace: bool = False,
        is_first: bool = False,
        page_table: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, float]:
        """
        Run single decode step with CPU-free forward pass.

        After input prep, the entire forward pass (embed -> transformer -> logits ->
        argmax -> position increment) runs on device with no CPU roundtrips.

        Args:
            token_id_ttnn: Token ID on device [1, batch_size] uint32
            use_trace: Whether to use tracing
            is_first: Whether this is the first decode step (for warm-up)
            page_table: Optional page table for paged attention (vLLM)

        Returns:
            Tuple of (tt_next_token on device [1, batch_size] uint32, decode_time_ms)
        """
        # Use demo's page_table if paged attention is enabled and no external page_table provided
        effective_page_table = page_table
        if effective_page_table is None and self.use_paged_attention:
            effective_page_table = self.page_table

        hidden_states = self.model.text_model.embed_tokens(token_id_ttnn)

        if use_trace:
            if self.decode_trace_id is None:
                raise RuntimeError(
                    "Decode trace not captured (decode_trace_id is None). "
                    "Call warmup_all_buckets(..., use_decode_trace=True) or warmup_video_traces(...) "
                    "before run_decode_step with use_trace=True."
                )

            start_time = time.perf_counter()
            logits = self._execute_decode_trace(
                self.decode_trace_id,
                self.decode_trace_tensors,
                self.decode_trace_output,
                hidden_states,
                page_table=effective_page_table,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000
        else:
            if is_first:
                compile_time = self.warmup_decode(hidden_states, page_table=effective_page_table)

            start_time = time.perf_counter()
            logits = self.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=self.kv_caches,
                current_pos=self.current_pos,
                rot_mat_idxs=self.rot_mat_idxs,
                page_table=effective_page_table,
            )
            ttnn.synchronize_device(self.mesh_device)
            decode_time = (time.perf_counter() - start_time) * 1000

        ttnn.deallocate(hidden_states)

        # Greedy token selection
        if self.repetition_penalty == 1.0:
            # Device-side argmax when no repetition penalty (faster, no CPU roundtrip)
            # logits shape: [1, 1, 1, vocab_size] or similar
            next_token_ttnn = ttnn.argmax(logits, dim=-1, keepdim=False)
            mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            next_token_cpu = ttnn.to_torch(next_token_ttnn, mesh_composer=mesh_composer)[0]
            ttnn.deallocate(next_token_ttnn)
            next_token = int(next_token_cpu.flatten()[0].item())

            # For debug logging, we still need logits on CPU
            if is_first or self.decode_position % 1000 == 0 or self.decode_position < 5:
                logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0]
                if logits_torch.dim() == 4:
                    logits_vec = logits_torch[0, 0, 0, :].clone()
                elif logits_torch.dim() == 3:
                    logits_vec = logits_torch[0, 0, :].clone()
                else:
                    logits_vec = logits_torch[0].flatten()
        else:
            # CPU argmax when repetition penalty is used (need to modify logits)
            mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0]
            if logits_torch.dim() == 4:
                logits_vec = logits_torch[0, 0, 0, :].clone()
            elif logits_torch.dim() == 3:
                logits_vec = logits_torch[0, 0, :].clone()
            else:
                logits_vec = logits_torch[0].flatten()

            num_penalized = len(self.generated_token_ids)
            for token_id in self.generated_token_ids:
                if logits_vec[token_id] > 0:
                    logits_vec[token_id] /= self.repetition_penalty
                else:
                    logits_vec[token_id] *= self.repetition_penalty

            next_token = logits_vec.argmax().item()
            self.generated_token_ids.add(next_token)
            if len(self.generated_token_ids) <= 5 or len(self.generated_token_ids) % 10 == 0:
                logger.debug(f"Repetition penalty: penalized {num_penalized} tokens, selected token {next_token}")

        token_batch = torch.tensor([[next_token] * self.batch_size], dtype=torch.long)
        tt_next_token = ttnn.from_torch(
            token_batch,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # DEBUG: Log decode position and logits for first few decodes
        if is_first or self.decode_position % 1000 == 0 or self.decode_position < 5:
            logger.info(f"  DECODE: position={self.decode_position}, is_first={is_first}")
            top5_vals, top5_ids = logits_vec.topk(5)
            logger.info(
                f"    DECODE logits stats: mean={logits_vec.mean():.2f}, std={logits_vec.std():.2f}, min={logits_vec.min():.2f}, max={logits_vec.max():.2f}"
            )
            logger.info(
                f"    DECODE top 5: {[(int(tid), float(tv)) for tid, tv in zip(top5_ids.tolist(), top5_vals.tolist())]}"
            )

        # On-device position increment (no CPU tensor creation)
        ttnn.plus_one(self.current_pos)
        ttnn.plus_one(self.rot_mat_idxs)
        self.decode_position += 1

        return tt_next_token, decode_time

    def _read_token_from_device(self, tt_token: ttnn.Tensor) -> int:
        """Read first token value from device (tiny transfer for EOS check).

        For batch > 1, returns only the first token since all batch items
        process the same prompt and should generate the same tokens.
        """
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        tokens_torch = ttnn.to_torch(tt_token, mesh_composer=mesh_composer)[0]
        # Return first batch item for EOS check (all batch items have same prompt)
        if tokens_torch.numel() > 1:
            return tokens_torch[0].item()
        return tokens_torch.item()

    def _read_all_tokens_from_device(self, tt_token: ttnn.Tensor) -> List[int]:
        """Read all token values from device for parallel batch processing.

        Returns a list of tokens, one per batch item.
        """
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        tokens_torch = ttnn.to_torch(tt_token, mesh_composer=mesh_composer)[0]
        if tokens_torch.dim() == 0:
            return [tokens_torch.item()]
        return tokens_torch.flatten().tolist()

    def run_inference(
        self,
        image_inputs: dict,
        prompt: str = None,  # Deprecated for images: HF processor includes input_ids
        max_new_tokens: int = 100,
        use_trace: bool = False,
        use_decode_trace: bool = False,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
    ) -> Tuple[str, dict]:
        """
        Run full inference with autoregressive generation.

        Args:
            image_inputs: Dict from preprocess_image (HF processor) or None for text-only
                - input_ids: [1, seq_len] with visual tokens already inserted
                - pixel_values: [n_crops, 3, H, W]
                - image_token_pooling: [n_tokens, k_pool]
            prompt: Text prompt (only needed for text-only, HF processor already tokenized images)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode
            use_vision_trace: Whether to use tracing for vision backbone
            use_unified_trace: Whether to use unified Vision+Prefill trace

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        # Check if using HF processor format (has input_ids already)
        has_hf_input_ids = image_inputs is not None and "input_ids" in image_inputs
        hf_token_type_ids = image_inputs.get("token_type_ids") if has_hf_input_ids else None
        hf_attention_mask_mm = image_inputs.get("attention_mask") if has_hf_input_ids else None

        if has_hf_input_ids:
            # HF processor format: input_ids already tokenized
            input_ids = image_inputs["input_ids"]
            # Check if this is text-only (no pixel_values) or image (has pixel_values)
            if "pixel_values" in image_inputs:
                pixel_values = image_inputs["pixel_values"]
                pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)
                logger.debug(f"Using HF processor input_ids (image): {input_ids.shape}")
            else:
                # Text-only from HF processor
                pixel_values = None
                pooled_patches_idx = None
                logger.debug(f"Using HF processor input_ids (text-only): {input_ids.shape}")
        elif prompt is not None and IMAGE_PROMPT not in prompt:
            # Text-only: no image processing needed (legacy path)
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
            pixel_values = None
            pooled_patches_idx = None
        else:
            # Legacy format: manual token building (fallback)
            image_grid = image_inputs["image_grids"][0]
            image_tokens_str = get_image_tokens(image_grid)
            content_with_images = prompt.replace(IMAGE_PROMPT, image_tokens_str)
            messages = [{"role": "user", "content": content_with_images}]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
            pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)
            pixel_values = image_inputs["pixel_values"]
            logger.debug(f"Using legacy format input_ids: {input_ids.shape}")

        # Run prefill
        logits, prefill_timing = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
            token_type_ids=hf_token_type_ids,
            hf_attention_mask=hf_attention_mask_mm,
        )

        # Get first prediction from prefill (one-time CPU argmax is acceptable)
        # Use original_seq_len from timing to index correctly (accounting for padding)
        original_seq_len = prefill_timing.get("original_seq_len", input_ids.shape[1])
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()

        # DEBUG: Log prefill diagnostics
        logger.info(f"=== PREFILL DIAGNOSTICS ===")
        logger.info(f"  input_ids shape: {input_ids.shape}, original_seq_len: {original_seq_len}")
        logger.info(f"  input_ids (first 20): {input_ids[0, :20].tolist()}")
        logger.info(f"  input_ids (last 20): {input_ids[0, -20:].tolist()}")
        logger.info(f"  logits_torch shape: {logits_torch.shape}")
        logger.info(
            f"  logits_torch stats: mean={logits_torch.mean().item():.4f}, std={logits_torch.std().item():.4f}, min={logits_torch.min().item():.4f}, max={logits_torch.max().item():.4f}"
        )

        if logits_torch.dim() == 2:
            next_token_logits = logits_torch[original_seq_len - 1, :]
            logger.info(f"  Indexing logits at position {original_seq_len - 1}")
        else:
            next_token_logits = logits_torch
            logger.info(f"  Using 1D logits directly")

        logger.info(
            f"  next_token_logits stats: mean={next_token_logits.mean().item():.4f}, std={next_token_logits.std().item():.4f}, min={next_token_logits.min().item():.4f}, max={next_token_logits.max().item():.4f}"
        )

        # Get top 5 predictions
        top5_values, top5_indices = torch.topk(next_token_logits, 5)
        logger.info(f"  Top 5 predictions:")
        for i, (val, idx) in enumerate(zip(top5_values.tolist(), top5_indices.tolist())):
            decoded = self.tokenizer.decode([idx])
            logger.info(f"    {i+1}. token={idx}, logit={val:.2f}, decoded='{decoded}'")

        next_token = torch.argmax(next_token_logits).item()
        logger.info(f"  Selected first token: {next_token} -> '{self.tokenizer.decode([next_token])}'")
        logger.info(f"=== END PREFILL DIAGNOSTICS ===")
        generated_tokens = [next_token]

        # Put first token on device for CPU-free decode loop
        # Replicate to batch_size for batch processing (all batch items get same token)
        token_batch = torch.tensor([[next_token] * self.batch_size], dtype=torch.long)
        tt_next_token = ttnn.from_torch(
            token_batch,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Reset repetition penalty tracking
        self.generated_token_ids = set()
        if self.repetition_penalty != 1.0:
            # Add first generated token to tracking
            self.generated_token_ids.add(next_token)

        # Autoregressive generation -- forward pass is fully on device
        decode_times = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break

            # CPU-free decode: embed -> forward -> argmax -> plus_one all on device
            tt_next_token, decode_time = self.run_decode_step(
                tt_next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            # Read back single token int for EOS check and logging
            next_token = self._read_token_from_device(tt_next_token)
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
            "vision_trace_ms": prefill_timing.get("vision_trace_ms", 0),
            "compile_vision_ms": prefill_timing.get("compile_vision_ms", 0),
            "compile_prefill_ms": prefill_timing.get("compile_prefill_ms", 0),
            "ttft_ms": prefill_timing.get("ttft_ms", 0),
            "e2e_ttft_ms": prefill_timing.get("e2e_ttft_ms", 0),  # End-to-end TTFT (vision + fusion + prefill)
            # Unified trace specific metrics
            "prep_ms": prefill_timing.get("prep_ms", 0),
            "compile_ms": prefill_timing.get("compile_ms", 0),
            # Decode metrics
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

    def run_video_inference(
        self,
        video_inputs: dict,
        prompt: str = None,  # Deprecated: HF processor includes input_ids
        max_new_tokens: int = 200,
        use_trace: bool = False,
        use_decode_trace: bool = False,
        use_vision_trace: bool = False,
        use_unified_trace: bool = False,
        use_dp_vision_trace: bool = True,
        use_data_parallel: bool = False,
        frames_per_device: int = 8,
    ) -> Tuple[str, dict]:
        """
        Run full inference on a video input with autoregressive generation.

        Args:
            video_inputs: Dict from preprocess_video (HF processor)
                - input_ids: [1, seq_len] with visual tokens already inserted
                - pixel_values: [n_frames, 3, H, W]
                - image_token_pooling: [n_tokens, k_pool]
                - n_frames, pooled_h, pooled_w, k_pool, timestamps
            prompt: Deprecated (HF processor already includes tokenized input)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode
            use_vision_trace: Whether to use tracing for vision backbone
            use_unified_trace: Whether to use unified Vision+Prefill trace
            use_dp_vision_trace: Use pre-captured DP=8 ViT + pool traces (default True; pass False for eager vision)

        Returns:
            Tuple of (output_text, perf_metrics)
        """
        n_frames = video_inputs["n_frames"]
        timestamps = video_inputs["timestamps"]
        pooled_h = video_inputs["pooled_h"]
        pooled_w = video_inputs["pooled_w"]

        # Use input_ids from HF processor (already tokenized with visual tokens)
        input_ids = video_inputs["input_ids"]
        logger.debug(f"Using HF processor input_ids: {input_ids.shape}")

        # pooled_patches_idx shape: [n_tokens, k_pool] from HF processor
        # Reshape to [n_frames, n_out, k_pool] for compatibility with run_prefill
        n_tokens = video_inputs["n_tokens"]
        k_pool = video_inputs["k_pool"]
        n_out = n_tokens // n_frames  # tokens per frame (81 for video with 3x3 pooling)

        pooled_patches_idx = video_inputs["image_token_pooling"]  # [n_tokens, k_pool]
        if pooled_patches_idx.dim() == 2 and pooled_patches_idx.shape[0] == n_tokens:
            # Reshape from [n_tokens, k_pool] to [n_frames, n_out, k_pool]
            pooled_patches_idx = pooled_patches_idx.reshape(n_frames, n_out, k_pool)
            logger.debug(f"Reshaped pooled_patches_idx: {pooled_patches_idx.shape}")

        pixel_values = video_inputs["pixel_values"]  # [n_frames, 3, H, W]

        num_visual_tokens = n_frames * pooled_h * pooled_w
        logger.info(
            f"Video: {n_frames} frames, {num_visual_tokens} visual tokens, {input_ids.shape[1]} total input tokens"
        )

        # Vision timing starts here
        vision_start = time.perf_counter()

        # DP vision path only when requested and warmup captured ViT+pool traces (no silent auto-enable).
        _use_dp_vision_trace = bool(use_dp_vision_trace and self.dp_vit_trace_id is not None)

        # Run prefill (vision + text model)
        logits, prefill_timing = self.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=use_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
            use_dp_vision_trace=_use_dp_vision_trace,
            use_data_parallel=use_data_parallel,
            frames_per_device=frames_per_device,
            token_type_ids=video_inputs.get("token_type_ids"),
            hf_attention_mask=video_inputs.get("attention_mask"),
        )

        vision_total_ms = (time.perf_counter() - vision_start) * 1000

        # First token from prefill logits (one-time CPU argmax)
        # Use original_seq_len from timing to index correctly (accounting for padding)
        original_seq_len = prefill_timing.get("original_seq_len", input_ids.shape[1])
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()

        # DEBUG: Log prefill diagnostics for video
        logger.info(f"=== VIDEO PREFILL DIAGNOSTICS ===")
        logger.info(f"  input_ids shape: {input_ids.shape}, original_seq_len: {original_seq_len}")
        logger.info(f"  input_ids (first 20): {input_ids[0, :20].tolist()}")
        logger.info(f"  input_ids (last 20): {input_ids[0, -20:].tolist()}")
        logger.info(f"  logits_torch shape: {logits_torch.shape}")
        logger.info(
            f"  logits_torch stats: mean={logits_torch.mean().item():.4f}, std={logits_torch.std().item():.4f}, min={logits_torch.min().item():.4f}, max={logits_torch.max().item():.4f}"
        )

        if logits_torch.dim() == 2:
            # For chunked prefill, use the index within the last chunk
            if prefill_timing.get("chunked_prefill", False):
                logits_idx = prefill_timing["last_token_idx_in_chunk"]
                logger.info(f"  Using chunked prefill logits_idx: {logits_idx}")
            else:
                logits_idx = original_seq_len - 1
                logger.info(f"  Using standard logits_idx: {logits_idx}")
            next_token_logits = logits_torch[logits_idx, :]
        else:
            next_token_logits = logits_torch
            logger.info(f"  Using 1D logits directly")

        logger.info(
            f"  next_token_logits stats: mean={next_token_logits.mean().item():.4f}, std={next_token_logits.std().item():.4f}, min={next_token_logits.min().item():.4f}, max={next_token_logits.max().item():.4f}"
        )

        # Get top 5 predictions
        top5_values, top5_indices = torch.topk(next_token_logits, 5)
        logger.info(f"  Top 5 predictions:")
        for i, (val, idx) in enumerate(zip(top5_values.tolist(), top5_indices.tolist())):
            decoded = self.tokenizer.decode([idx])
            logger.info(f"    {i+1}. token={idx}, logit={val:.2f}, decoded='{decoded}'")

        next_token = torch.argmax(next_token_logits).item()
        logger.info(f"  Selected first token: {next_token} -> '{self.tokenizer.decode([next_token])}'")
        logger.info(f"=== END VIDEO PREFILL DIAGNOSTICS ===")
        generated_tokens = [next_token]

        # Put first token on device for CPU-free decode loop
        # Replicate to batch_size for batch processing (all batch items get same token)
        token_batch = torch.tensor([[next_token] * self.batch_size], dtype=torch.long)
        tt_next_token = ttnn.from_torch(
            token_batch,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Reset repetition penalty tracking
        self.generated_token_ids = set()
        if self.repetition_penalty != 1.0:
            # Add first generated token to tracking
            self.generated_token_ids.add(next_token)

        # Autoregressive generation
        decode_times = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break

            tt_next_token, decode_time = self.run_decode_step(
                tt_next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            next_token = self._read_token_from_device(tt_next_token)
            generated_tokens.append(next_token)

            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i + 1} tokens, last decode: {decode_time:.2f}ms")

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        total_decode_time = sum(decode_times) if decode_times else 0.0
        avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
        tokens_per_sec = len(decode_times) / (total_decode_time / 1000) if total_decode_time > 0 else 0
        vision_ms = prefill_timing.get("vision_ms", vision_total_ms)
        frames_per_sec = n_frames / (vision_ms / 1000) if vision_ms > 0 else 0

        perf_metrics = {
            "n_frames": n_frames,
            "frames_per_sec": frames_per_sec,
            "vision_ms": vision_ms,
            "vision_trace_ms": prefill_timing.get("vision_trace_ms", 0),
            "compile_vision_ms": prefill_timing.get("compile_vision_ms", 0),
            "compile_prefill_ms": prefill_timing.get("compile_prefill_ms", 0),
            "ttft_ms": prefill_timing.get("ttft_ms", 0),
            "e2e_ttft_ms": prefill_timing.get("e2e_ttft_ms", 0),
            "prep_ms": prefill_timing.get("prep_ms", 0),
            "compile_ms": prefill_timing.get("compile_ms", 0),
            "avg_decode_ms": avg_decode_time,
            "total_decode_ms": total_decode_time,
            "input_tokens": input_ids.shape[1],
            "generated_tokens": len(generated_tokens),
            "tokens_per_sec": tokens_per_sec,
            "output_text": output_text,
        }

        logger.info(f"Video: {n_frames} frames processed")
        logger.info(f"Vision throughput: {frames_per_sec:.2f} frames/sec")
        logger.info(f"TTFT: {prefill_timing.get('ttft_ms', 0):.2f}ms")
        logger.info(f"Decode: {tokens_per_sec:.2f} tok/s ({len(generated_tokens)} tokens)")
        logger.info(f"Output: '{output_text[:100]}...' " if len(output_text) > 100 else f"Output: '{output_text}'")

        return output_text, perf_metrics

    def run_batched_inference(
        self,
        image_inputs_list: List[dict],
        prompts: List[str],
        max_new_tokens: int = 100,
        use_trace: bool = False,
        use_decode_trace: bool = False,
        use_vision_trace: bool = False,
    ) -> Tuple[List[str], dict]:
        """
        Run batched inference with multiple different prompts in parallel.

        Unlike run_inference which replicates the same token across batch slots,
        this method processes different prompts simultaneously, following the
        tt_transformers/qwen3_vl pattern.

        Args:
            image_inputs_list: List of image input dicts (one per batch item)
            prompts: List of text prompts (one per batch item)
            max_new_tokens: Maximum tokens to generate
            use_trace: Whether to use tracing for prefill
            use_decode_trace: Whether to use tracing for decode
            use_vision_trace: Whether to use tracing for vision backbone

        Returns:
            Tuple of (list of output texts, perf_metrics dict)
        """
        batch_size = len(prompts)
        assert batch_size <= self.batch_size, f"Number of prompts ({batch_size}) exceeds batch_size ({self.batch_size})"
        assert len(image_inputs_list) == batch_size, "image_inputs_list must match prompts length"

        logger.info(f"Running batched inference with {batch_size} prompts")

        # Step 1: Prefill each prompt sequentially (different image sizes)
        # Track per-user prefill timing and ending positions
        prefill_timings = []
        decoding_positions = []
        first_tokens = []

        for user_id in range(batch_size):
            logger.info(f"Prefilling user {user_id}/{batch_size}")
            image_inputs = image_inputs_list[user_id]
            prompt = prompts[user_id]

            # Check if using HF processor format (has input_ids already)
            has_hf_input_ids = image_inputs is not None and "input_ids" in image_inputs
            hf_token_type_ids = image_inputs.get("token_type_ids") if has_hf_input_ids else None
            hf_attention_mask_mm = image_inputs.get("attention_mask") if has_hf_input_ids else None

            if has_hf_input_ids:
                # HF processor format: input_ids already tokenized with visual tokens
                input_ids = image_inputs["input_ids"]
                pixel_values = image_inputs["pixel_values"]
                pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)
            elif IMAGE_PROMPT not in prompt:
                # Text-only
                messages = [{"role": "user", "content": prompt}]
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
                pixel_values = None
                pooled_patches_idx = None
            else:
                # Legacy format
                image_grid = image_inputs["image_grids"][0]
                image_tokens_str = get_image_tokens(image_grid)
                content_with_images = prompt.replace(IMAGE_PROMPT, image_tokens_str)
                messages = [{"role": "user", "content": content_with_images}]
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
                pooled_patches_idx = image_inputs["image_token_pooling"].unsqueeze(0)
                pixel_values = image_inputs["pixel_values"]

            # Run prefill for this user (user_id determines KV cache slot)
            logits, prefill_timing = self.run_prefill(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pooled_patches_idx=pooled_patches_idx,
                use_trace=use_trace,
                use_vision_trace=use_vision_trace,
                use_unified_trace=False,  # Batched mode doesn't support unified trace
                user_id=user_id,
                token_type_ids=hf_token_type_ids,
                hf_attention_mask=hf_attention_mask_mm,
            )

            # Get first token from prefill logits
            original_seq_len = prefill_timing.get("original_seq_len", input_ids.shape[1])
            mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            logits_torch = ttnn.to_torch(logits, mesh_composer=mesh_composer)[0].squeeze()
            if logits_torch.dim() == 2:
                next_token_logits = logits_torch[original_seq_len - 1, :]
            else:
                next_token_logits = logits_torch

            first_token = torch.argmax(next_token_logits).item()
            first_tokens.append(first_token)
            decoding_positions.append(original_seq_len)
            prefill_timings.append(prefill_timing)

            logger.debug(f"User {user_id}: prefilled {original_seq_len} tokens, first decode token: {first_token}")

        # Step 2: Set up per-user decode positions
        self.reset_kv_cache(start_pos=decoding_positions)

        # Step 3: Create batched token tensor (different tokens per user!)
        # Pad to self.batch_size if fewer prompts
        tokens_list = first_tokens + [first_tokens[0]] * (self.batch_size - batch_size)
        token_batch = torch.tensor([tokens_list], dtype=torch.long)
        tt_next_token = ttnn.from_torch(
            token_batch,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_mapper,
        )

        # Step 4: Per-user output tracking
        all_outputs = [[first_tokens[i]] for i in range(batch_size)]
        user_done = [False] * batch_size
        eos_token_id = self.tokenizer.eos_token_id

        # Step 5: Batched decode loop
        decode_times = []
        logger.info(f"Starting batched decode for {batch_size} users...")

        for i in range(max_new_tokens - 1):
            # Check if all users are done
            if all(user_done):
                break

            # Run decode step (processes all batch items simultaneously)
            tt_next_token, decode_time = self.run_decode_step(
                tt_next_token,
                use_trace=use_decode_trace,
                is_first=(i == 0),
            )
            decode_times.append(decode_time)

            # Read all tokens from device
            all_tokens = self._read_all_tokens_from_device(tt_next_token)

            # Track outputs per user
            for user_id in range(batch_size):
                if not user_done[user_id]:
                    token = all_tokens[user_id]
                    all_outputs[user_id].append(token)
                    if token == eos_token_id:
                        user_done[user_id] = True
                        logger.debug(f"User {user_id} finished at iteration {i}")

            # Log progress
            if (i + 1) % 10 == 0:
                active_users = sum(1 for d in user_done if not d)
                logger.debug(f"Generated {i + 1} tokens, {active_users} users still active")

        # Step 6: Decode outputs for each user
        output_texts = []
        for user_id in range(batch_size):
            text = self.tokenizer.decode(all_outputs[user_id], skip_special_tokens=True)
            output_texts.append(text)
            logger.info(f"User {user_id}: '{text[:50]}...'" if len(text) > 50 else f"User {user_id}: '{text}'")

        # Calculate metrics
        total_decode_time = sum(decode_times) if decode_times else 0.0
        avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0.0
        total_tokens = sum(len(out) - 1 for out in all_outputs)  # -1 for first token
        tokens_per_sec = total_tokens / (total_decode_time / 1000) if total_decode_time > 0 else 0

        perf_metrics = {
            "batch_size": batch_size,
            "avg_decode_ms": avg_decode_time,
            "total_decode_ms": total_decode_time,
            "total_generated_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_sec_per_user": tokens_per_sec / batch_size if batch_size > 0 else 0,
            "per_user_tokens": [len(out) for out in all_outputs],
        }

        logger.info(f"Batched decode: {tokens_per_sec:.2f} tok/s total ({tokens_per_sec/batch_size:.2f} tok/s/user)")

        return output_texts, perf_metrics


def run_video_demo(
    video_path: str,
    prompt: str = "<|video|> Describe what happens in this video.",
    max_new_tokens: int = 200,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    max_seq_len: int = 65536,  # 64k for ~320-frame video support
    max_frames: int = VIDEO_MAX_FRAMES,
    max_fps: float = VIDEO_MAX_FPS,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
    use_dp_vision_trace: bool = False,
    use_paged_attention: bool = False,
    batch_size: int = 1,
    num_devices: int = 8,
    use_data_parallel: bool = False,
    frames_per_device: int = 8,
    repetition_penalty: float = 1.0,
    use_async_ccl: bool = False,
):
    """
    Run the Molmo2 demo with video input.

    Args:
        video_path: Path or URL to video file (.mp4, .webm)
        prompt: Text prompt (must include <|video|>)
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        max_seq_len: Maximum sequence length for KV cache (default: 16384 for video)
        max_frames: Maximum frames to sample from video (default: 8)
        max_fps: Maximum frames per second to sample (default: 2.0)
        use_trace: Whether to use tracing for prefill
        use_decode_trace: Whether to use tracing for decode
        use_vision_trace: Whether to use tracing for vision backbone
        use_unified_trace: Whether to use unified Vision+Prefill trace
        use_paged_attention: Whether to use paged attention (for vLLM compatibility)
    """
    logger.info("=" * 60)
    logger.info("Molmo2-8B Video Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess video using HF processor (correct pooling_size [3,3], max_fps=2.0)
    logger.info(f"Preprocessing video: {video_path}")
    video_extraction_start = time.perf_counter()
    video_inputs = preprocess_video(
        video_path,
        prompt,
        num_frames=max_frames,  # HF processor uses num_frames param
    )
    video_extraction_ms = (time.perf_counter() - video_extraction_start) * 1000
    n_frames = video_inputs["n_frames"]
    k_pool = video_inputs["k_pool"]
    logger.info(f"Extracted {n_frames} frames in {video_extraction_ms:.2f}ms (k_pool={k_pool})")
    logger.info(f"Frame extraction: {n_frames / (video_extraction_ms / 1000):.2f} frames/sec (CPU)")

    # Load weights
    state_dict = load_model_weights()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        model = create_model(
            device,
            state_dict,
            num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_async_ccl=use_async_ccl,
        )
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_paged_attention=use_paged_attention,
            repetition_penalty=repetition_penalty,
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"Prompt: {prompt}")
        logger.info("=" * 60)

        # Upfront warmup: capture ALL traces before inference
        if use_dp_vision_trace or use_decode_trace or use_trace:
            from models.demos.molmo2.tt.vision_backbone import MAX_VIT_FRAMES_FOR_POOL

            _n_out = video_inputs["n_tokens"] // video_inputs["n_frames"]
            _k_pool = video_inputs["k_pool"]
            # Use fixed MAX_VIT_FRAMES_FOR_POOL (80) so the pool trace buffer is always
            # the same size and can be reused across different videos. The actual video
            # features are zero-padded on-device to fill the buffer.
            generator.warmup_video_traces(
                frames_per_device=frames_per_device,
                num_devices=num_devices,
                prefill_buckets=[b for b in PREFILL_SEQ_BUCKETS if b <= generator.max_seq_len],
                pool_n_out=_n_out,
                pool_k_pool=_k_pool,
                max_vit_frames=MAX_VIT_FRAMES_FOR_POOL,
                use_prefill_trace=use_trace,
                use_decode_trace=use_decode_trace,
            )

        response, perf_metrics = generator.run_video_inference(
            video_inputs=video_inputs,
            # prompt not needed - HF processor already tokenized with visual tokens
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
            use_dp_vision_trace=use_dp_vision_trace,
            use_data_parallel=use_data_parallel,
            frames_per_device=frames_per_device,
        )

        vision_ms = perf_metrics["vision_ms"]
        ttft_ms = perf_metrics["ttft_ms"]
        total_decode_ms = perf_metrics["total_decode_ms"]
        compile_vision_ms = perf_metrics.get("compile_vision_ms", 0)
        compile_prefill_ms = perf_metrics.get("compile_prefill_ms", 0)
        total_inference_ms = video_extraction_ms + vision_ms + ttft_ms + total_decode_ms

        logger.info("\n" + "=" * 60)
        logger.info("Video Performance Metrics:")
        logger.info(f"  Frames processed:    {perf_metrics['n_frames']}")
        logger.info(
            f"  Frame extraction:    {video_extraction_ms:.2f}ms ({n_frames / (video_extraction_ms/1000):.2f} frames/sec CPU)"
        )
        if compile_vision_ms > 0:
            logger.info(f"  Vision compile:      {compile_vision_ms:.2f}ms  [excluded from total]")
        logger.info(f"  Vision (TTNN):       {vision_ms:.2f}ms ({perf_metrics['frames_per_sec']:.2f} frames/sec)")
        if compile_prefill_ms > 0:
            logger.info(f"  Prefill compile:     {compile_prefill_ms:.2f}ms  [excluded from total]")
        logger.info(f"  TTFT (prefill):      {ttft_ms:.2f}ms")
        if perf_metrics.get("e2e_ttft_ms", 0) > 0:
            logger.info(f"  E2E TTFT:            {perf_metrics['e2e_ttft_ms']:.2f}ms")
        logger.info(f"  Input tokens:        {perf_metrics['input_tokens']}")
        logger.info(f"  Generated tokens:    {perf_metrics['generated_tokens']}")
        logger.info(
            f"  Decode:              {perf_metrics['avg_decode_ms']:.2f}ms/token ({perf_metrics['tokens_per_sec']:.2f} tok/s)"
        )
        logger.info(f"  Total decode:        {total_decode_ms:.2f}ms")
        logger.info("-" * 60)
        logger.info(
            f"  TOTAL (preproc+vision+prefill+decode): {total_inference_ms:.0f}ms ({total_inference_ms/1000:.2f}s)"
        )
        logger.info(
            f"    = preproc {video_extraction_ms:.0f}ms + vision {vision_ms:.0f}ms + prefill {ttft_ms:.0f}ms + decode {total_decode_ms:.0f}ms"
        )
        logger.info("=" * 60)
        logger.info(f"Response: {response}")
        logger.info("=" * 60)

        return perf_metrics

    finally:
        ttnn.close_mesh_device(device)
        logger.info("Device closed")


def run_demo(
    image_path: Optional[str] = None,
    prompt: str = "<|image|> Describe this image in detail.",
    max_new_tokens: int = 100,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    max_seq_len: int = 2048,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    use_unified_trace: bool = False,
    use_paged_attention: bool = False,
    batch_size: int = 1,
    num_devices: int = 8,
    repetition_penalty: float = 1.0,
    use_async_ccl: bool = False,
):
    """
    Run the Molmo2 demo.

    Args:
        image_path: Path to input image (uses default if None)
        prompt: Text prompt for the model (must include <|image|>)
        max_new_tokens: Maximum tokens to generate
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        max_seq_len: Maximum sequence length for KV cache (default: 2048 for image)
        use_trace: Whether to use tracing for text prefill
        use_decode_trace: Whether to use tracing for decode
        use_vision_trace: Whether to use tracing for vision backbone
        use_unified_trace: Whether to use unified Vision+Prefill trace (eliminates CPU roundtrip)
    """
    # Validate incompatible options
    # Paged attention is incompatible with UNIFIED traces because paged KV cache
    # writes (paged_fill_cache) cannot be captured during trace capture.
    # Regular prefill traces (--use-trace) work because they don't include the unified vision+prefill path.
    if use_paged_attention and use_unified_trace:
        logger.warning(
            "WARNING: --paged-attention and --use-unified-trace are incompatible. "
            "Paged attention writes during prefill cannot be captured in traces. Disabling unified trace."
        )
        use_unified_trace = False

    # Note: Both prefill trace (--use-trace) and decode trace (--use-decode-trace) ARE compatible
    # with paged attention. The page_table is allocated as a trace input tensor and gets updated
    # via ttnn.copy() before each trace execution.

    # Determine if this is a text-only or image prompt
    is_text_only = IMAGE_PROMPT not in prompt

    # Only default to an image if prompt expects one but none provided
    if not is_text_only and image_path is None:
        image_path = str(DEFAULT_IMAGE)
        logger.info(f"No image provided, using default: {image_path}")

    logger.info("=" * 60)
    logger.info("Molmo2-8B Demo")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess based on modality
    if is_text_only:
        # Text-only: use HF processor for tokenization
        from models.demos.molmo2.tt.hf_processor import preprocess_text

        logger.info("Text-only prompt (no <|image|> token)")
        image_inputs = preprocess_text(prompt)
        image_inputs["input_ids"] = image_inputs["input_ids"]  # Already correct shape
        logger.info(f"Text tokenized: {image_inputs['input_ids'].shape[1]} tokens")
    else:
        # Image: preprocess with HF processor
        logger.info(f"Preprocessing image: {image_path}")
        image_inputs = preprocess_image(image_path, prompt)
        logger.info(f"Image preprocessed: {image_inputs['n_crops']} crops, k_pool={image_inputs['k_pool']}")

    # Load weights
    state_dict = load_model_weights()

    # Open multi-device mesh for T3K (8 devices) to enable bfloat16 weight sharding
    # This prevents numerical overflow during decode by using higher precision weights
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        # Create model
        model = create_model(
            device,
            state_dict,
            num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_async_ccl=use_async_ccl,
        )
        text_num_layers = num_layers if num_layers is not None else 36

        # Create generator
        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_paged_attention=use_paged_attention,
            repetition_penalty=repetition_penalty,
        )

        # Warmup: compile ops and capture traces for all bucket sizes
        logger.info("\n" + "=" * 60)
        logger.info("WARMUP: Compiling ops and capturing traces for all buckets...")
        logger.info("=" * 60)

        warmup_start = time.perf_counter()
        generator.warmup_all_buckets(
            bucket_sizes=[128, 256, 512, 1024, 2048, 4096],
            use_decode_trace=use_decode_trace,
            use_prefill_trace=use_trace,  # Only capture prefill traces if --use-trace is passed
        )
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        logger.info(f"Total warmup time: {warmup_time:.2f}ms")

        # Actual inference run (post-warmup)
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE (post-warmup):")
        logger.info(f"Prompt: {prompt}")
        logger.info("=" * 60)

        response, perf_metrics = generator.run_inference(
            image_inputs=image_inputs,
            # prompt not needed - HF processor already tokenized with visual tokens
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
            use_vision_trace=use_vision_trace,
            use_unified_trace=use_unified_trace,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Performance Metrics:")
        # Check if unified trace was actually used (indicated by prep_ms and compile_ms being set)
        unified_trace_used = use_unified_trace and perf_metrics.get("prep_ms", 0) > 0
        if unified_trace_used:
            logger.info("  [Unified Vision+Prefill Trace]")
            logger.info(f"    - Input preparation: {perf_metrics['prep_ms']:.2f}ms")
            if perf_metrics.get("compile_ms", 0) > 0:
                logger.info(f"    - Unified compile: {perf_metrics['compile_ms']:.2f}ms")
            logger.info(f"  TTFT (Vision+Prefill): {perf_metrics['ttft_ms']:.2f}ms")
        else:
            logger.info(f"  Vision processing: {perf_metrics['vision_ms']:.2f}ms")
            if perf_metrics.get("vision_trace_ms", 0) > 0:
                logger.info(f"    - Vision trace execution: {perf_metrics['vision_trace_ms']:.2f}ms")
            if perf_metrics.get("compile_vision_ms", 0) > 0:
                logger.info(f"    - Vision compile: {perf_metrics['compile_vision_ms']:.2f}ms")
            if perf_metrics.get("compile_prefill_ms", 0) > 0:
                logger.info(f"  Prefill compile: {perf_metrics['compile_prefill_ms']:.2f}ms")
            logger.info(f"  Prefill-only TTFT: {perf_metrics['ttft_ms']:.2f}ms")
            if perf_metrics.get("e2e_ttft_ms", 0) > 0:
                logger.info(f"  ** End-to-End TTFT (Vision+Fusion+Prefill): {perf_metrics['e2e_ttft_ms']:.2f}ms **")
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


def run_batched_demo(
    input_file: str,
    max_new_tokens: int = 100,
    device_id: int = 0,
    num_layers: Optional[int] = None,
    max_seq_len: int = 4096,
    use_trace: bool = False,
    use_decode_trace: bool = False,
    use_vision_trace: bool = False,
    batch_size: int = 4,
    num_devices: int = 8,
    use_async_ccl: bool = False,
):
    """
    Run batched demo with multiple different prompts processed in parallel.

    Args:
        input_file: Path to JSON file with prompts and images
        max_new_tokens: Maximum tokens to generate per prompt
        device_id: TTNN device ID
        num_layers: Number of text layers (default: 36)
        max_seq_len: Maximum sequence length for KV cache
        use_trace: Whether to use tracing for prefill
        use_decode_trace: Whether to use tracing for decode
        use_vision_trace: Whether to use tracing for vision backbone
        batch_size: Maximum batch size for parallel processing
    """
    import json

    logger.info("=" * 60)
    logger.info("Molmo2-8B Batched Demo (Parallel Processing)")
    logger.info("=" * 60)

    # Load prompts from JSON file
    input_path = Path(input_file)
    with open(input_file, "r") as f:
        prompts_data = json.load(f)

    # Handle both formats:
    # 1. Flat list: [{"image": "...", "prompt": "..."}, ...]
    # 2. Nested: {"prompts": [{"image": "...", "prompt": "..."}, ...]}
    if isinstance(prompts_data, dict) and "prompts" in prompts_data:
        prompts_data = prompts_data["prompts"]
    if not isinstance(prompts_data, list):
        prompts_data = [prompts_data]

    num_prompts = len(prompts_data)
    logger.info(f"Loaded {num_prompts} prompts from {input_file}")

    # Limit to batch_size
    if num_prompts > batch_size:
        logger.warning(f"Truncating {num_prompts} prompts to batch_size={batch_size}")
        prompts_data = prompts_data[:batch_size]
        num_prompts = batch_size

    # Load tokenizer
    tokenizer = load_processor()

    # Preprocess images using HF processor
    logger.info("Preprocessing images...")
    image_inputs_list = []
    prompts = []
    for i, item in enumerate(prompts_data):
        image_path = item.get("image", str(DEFAULT_IMAGE))
        # Resolve relative paths relative to input file directory
        if not Path(image_path).is_absolute():
            image_path = str(input_path.parent / image_path)
        prompt = item.get("prompt", f"{IMAGE_PROMPT} Describe this image.")
        # Ensure prompt has image token
        if IMAGE_PROMPT not in prompt:
            prompt = f"{IMAGE_PROMPT} {prompt}"
        prompts.append(prompt)
        # Use HF processor for correct preprocessing
        image_inputs = preprocess_image(image_path, prompt)
        image_inputs_list.append(image_inputs)
        logger.info(
            f"  Prompt {i}: image={Path(image_path).name}, k_pool={image_inputs['k_pool']}, prompt={prompt[:50]}..."
        )

    # Load weights
    state_dict = load_model_weights()

    # Open device
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, num_devices)
    logger.info(f"Opening TTNN mesh device with shape {mesh_shape}")
    device = ttnn.open_mesh_device(mesh_shape)
    logger.info(f"Opened mesh device with {device.get_num_devices()} devices")

    try:
        model = create_model(
            device,
            state_dict,
            num_layers,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_async_ccl=use_async_ccl,
        )
        text_num_layers = num_layers if num_layers is not None else 36

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_paged_attention=False,  # Batched mode doesn't use paged attention
        )

        logger.info("\n" + "=" * 60)
        for i, prompt in enumerate(prompts):
            logger.info(f"Prompt {i}: {prompt[:80]}...")
        logger.info("=" * 60)

        output_texts, perf_metrics = generator.run_batched_inference(
            image_inputs_list=image_inputs_list,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode_trace,
            use_vision_trace=use_vision_trace,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Batched Inference Results:")
        logger.info(f"  Batch size:             {perf_metrics['batch_size']}")
        logger.info(f"  Total tokens generated: {perf_metrics['total_generated_tokens']}")
        logger.info(f"  Total decode time:      {perf_metrics['total_decode_ms']:.2f}ms")
        logger.info(f"  Avg decode time:        {perf_metrics['avg_decode_ms']:.2f}ms/iteration")
        logger.info(f"  Throughput:             {perf_metrics['tokens_per_sec']:.2f} tok/s")
        logger.info(f"  Per-user throughput:    {perf_metrics['tokens_per_sec_per_user']:.2f} tok/s/user")
        logger.info("=" * 60)

        for i, text in enumerate(output_texts):
            logger.info(f"\n[User {i}] Output:")
            logger.info(f"  {text[:200]}..." if len(text) > 200 else f"  {text}")

        logger.info("=" * 60)

        return output_texts, perf_metrics

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
        "--video",
        type=str,
        default=None,
        help="Path or URL to input video (.mp4, .webm). Video mode always uses paged "
        "attention internally; pass --paged-attention to acknowledge / quiet the info log. "
        "Use with --use-decode-trace for the throughput decode path (also default for video).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for the model (must include <|image|> or <|video|> token)",
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
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for KV cache (default: 2048 for image, 16384 for video)",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=VIDEO_MAX_FRAMES,
        help=f"Maximum video frames to sample (default: {VIDEO_MAX_FRAMES})",
    )
    parser.add_argument(
        "--max-video-fps",
        type=float,
        default=VIDEO_MAX_FPS,
        help=f"Maximum frames per second for video sampling (default: {VIDEO_MAX_FPS})",
    )
    parser.add_argument(
        "--use-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefill (text) trace: video defaults to ON (may be auto-disabled when "
        "multimodal token_type_ids are present). Image/text/batched default OFF unless set.",
    )
    parser.add_argument(
        "--use-decode-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode trace: video defaults to ON (~30+ tok/s). Use --no-use-decode-trace for debug / HF-like greedy without trace. Image/text default OFF unless --use-decode-trace.",
    )
    parser.add_argument(
        "--use-vision-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Vision backbone (ViT + pooling) trace: video defaults to ON. "
        "Image/text/batched default OFF unless set.",
    )
    parser.add_argument(
        "--use-unified-trace",
        action="store_true",
        help="Enable unified Vision+Prefill trace (eliminates CPU roundtrip, best TTFT)",
    )
    parser.add_argument(
        "--use-dp-vision-trace",
        action="store_true",
        default=False,
        help="Enable DP=8 ViT + pool chunk traces for video (requires warmup_video_traces upfront)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help='Path to JSON file with multiple prompts for batched inference (format: [{"image": "path", "prompt": "text"}, ...])',
    )
    parser.add_argument(
        "--paged-attention",
        action="store_true",
        help="Video: optional — paged attention is always on for video; set this for explicit "
        "CLI parity with harness / to skip the auto-enable info line.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=8,
        help="Number of TT devices to use (default: 8 for T3K)",
    )
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        help="Use data parallelism for video frames (shard frames across devices)",
    )
    parser.add_argument(
        "--frames-per-device",
        type=int,
        default=8,
        help="Frames per device per pass when using data parallel (default: 8)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=">1.0 down-weights repeated tokens (HF greedy uses 1.0). Values like 1.1 can garble short multiple-choice answers.",
    )
    parser.add_argument(
        "--use-async-ccl",
        action="store_true",
        default=False,
        help="Use async CCL operations for tensor parallelism (reduces trace hangs with DP>1)",
    )

    args = parser.parse_args()

    # Decode trace: video defaults ON (throughput); image/text/batched default OFF unless --use-decode-trace.
    # Use --no-use-decode-trace on video for debug when trace logits look wrong.
    _dt = args.use_decode_trace
    decode_trace_video = True if _dt is None else _dt
    decode_trace_image_text = False if _dt is None else _dt

    # Prefill + vision trace: video defaults OFF for now (memory pressure during warmup)
    _ut = args.use_trace
    _vt = args.use_vision_trace
    trace_prefill_video = False if _ut is None else _ut
    trace_vision_video = False if _vt is None else _vt
    trace_prefill_image_text = False if _ut is None else _ut
    trace_vision_image_text = False if _vt is None else _vt

    if args.input_file is not None:
        # Batched inference with multiple prompts
        max_seq_len = args.max_seq_len if args.max_seq_len is not None else 4096
        run_batched_demo(
            input_file=args.input_file,
            max_new_tokens=args.max_tokens,
            device_id=args.device,
            num_layers=args.num_layers,
            max_seq_len=max_seq_len,
            use_trace=trace_prefill_image_text,
            use_decode_trace=decode_trace_image_text,
            use_vision_trace=trace_vision_image_text,
            batch_size=args.batch_size,
            num_devices=args.num_devices,
            use_async_ccl=args.use_async_ccl,
        )
    elif args.video is not None:
        if args.prompt is not None:
            # Auto-prepend <|video|> if user forgot it
            prompt = args.prompt if VIDEO_PROMPT in args.prompt else f"{VIDEO_PROMPT} {args.prompt}"
        else:
            prompt = f"{VIDEO_PROMPT} Describe what happens in this video."
        max_seq_len = args.max_seq_len if args.max_seq_len is not None else 65536

        # VIDEO: Force always-on features for optimal performance
        # - paged_attention: Required for chunked prefill with long sequences
        # - decode_trace: Required for good decode throughput (30+ tok/s)
        # - data_parallel: Always use DP=8 (handled in embed_image routing)
        use_paged_attention_video = True
        use_decode_trace_video = decode_trace_video
        if not args.paged_attention:
            logger.info("Video: Auto-enabling paged attention (required for long sequences)")
        if use_decode_trace_video:
            logger.info("Video: decode trace ON (default; ~30+ tok/s). For debug without trace: --no-use-decode-trace")
        else:
            logger.info("Video: decode trace OFF (debug / correct-greedy path; slower decode)")
        if trace_prefill_video:
            logger.info(
                "Video: prefill trace ON by default (auto-disabled if multimodal token_type_ids; see run_prefill). "
                "Opt out: --no-use-trace"
            )
        else:
            logger.info("Video: prefill trace OFF (--no-use-trace)")
        if trace_vision_video:
            logger.info("Video: vision trace ON by default. Opt out: --no-use-vision-trace")
        else:
            logger.info("Video: vision trace OFF (--no-use-vision-trace)")

        run_video_demo(
            video_path=args.video,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            device_id=args.device,
            num_layers=args.num_layers,
            max_seq_len=max_seq_len,
            use_async_ccl=args.use_async_ccl,
            max_frames=args.max_video_frames,
            max_fps=args.max_video_fps,
            use_trace=trace_prefill_video,
            use_decode_trace=use_decode_trace_video,
            use_vision_trace=trace_vision_video,
            use_unified_trace=args.use_unified_trace,
            use_dp_vision_trace=args.use_dp_vision_trace,
            use_paged_attention=use_paged_attention_video,
            batch_size=args.batch_size,
            num_devices=args.num_devices,
            use_data_parallel=True,  # Always use DP=8 for video
            frames_per_device=args.frames_per_device,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        # Image or text-only inference
        if args.image is not None:
            # Image provided: auto-prepend <|image|> if user forgot it
            if args.prompt is not None:
                prompt = args.prompt if IMAGE_PROMPT in args.prompt else f"{IMAGE_PROMPT} {args.prompt}"
            else:
                prompt = f"{IMAGE_PROMPT} Describe this image in detail."
        else:
            # No image provided: text-only mode (don't add <|image|> token)
            if args.prompt is not None:
                prompt = args.prompt
            else:
                # Default to a simple text prompt
                prompt = "Hello! How can I help you today?"
        max_seq_len = args.max_seq_len if args.max_seq_len is not None else 2048
        run_demo(
            image_path=args.image,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            device_id=args.device,
            num_layers=args.num_layers,
            max_seq_len=max_seq_len,
            use_trace=trace_prefill_image_text,
            use_decode_trace=decode_trace_image_text,
            use_vision_trace=trace_vision_image_text,
            use_unified_trace=args.use_unified_trace,
            use_paged_attention=args.paged_attention,
            batch_size=args.batch_size,
            num_devices=args.num_devices,
            repetition_penalty=args.repetition_penalty,
            use_async_ccl=args.use_async_ccl,
        )


if __name__ == "__main__":
    main()
