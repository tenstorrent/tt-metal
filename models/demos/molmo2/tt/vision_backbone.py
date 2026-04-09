# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Backbone for Molmo2.

Combines the Vision Transformer encoder with the image pooling and projector
to produce visual embeddings ready for the language model.

Pipeline:
    1. ViT encoder: images -> multi-scale hidden states (layers 18, 24)
    2. Concat features on hidden dim: [B*T, N, 1152] x 2 -> [B, T*N, 2304]
    3. Gather features using pooled_patches_idx
    4. Cross-attention pooling: [B*N_out, K_pool, 2304] -> [B, N_out, 1152]
    5. SwiGLU projection: [B, N_out, 1152] -> [B, N_out, 4096]
    6. Filter by valid_token mask: [valid_tokens, 4096]
"""

from typing import Dict, Tuple

import torch

import ttnn

# Threshold for single-pass vs chunked pooling. 32 frames = 23328 patches fits in DRAM;
# 53+ frames OOMs during cross-attention all_reduce in image_pooling.
MAX_FRAMES_FOR_SINGLE_POOL = 32

# Fixed upper bound for the pool trace feature buffer (image_features_2d rows).
# Using 80 frames = 58320 patches × 2304 × 2 bytes ≈ 268 MB per device.
# This must be a compile-time constant so the same pool trace can be reused across videos.
MAX_VIT_FRAMES_FOR_POOL = 80

# Upper bounds for buffers created once in VisionBackbone.__init__ (pooling mask + vision trace I/O).
DEFAULT_PATCHES_PER_FRAME = (378 // 14) ** 2  # 729
DEFAULT_VISION_TRACE_MAX_FRAMES = MAX_FRAMES_FOR_SINGLE_POOL
DEFAULT_VISION_TRACE_MAX_N_OUT = 256  # covers 81 (video) and 169 (multi-crop) layouts
DEFAULT_VISION_TRACE_MAX_K_POOL = 16


def _masked_mean_query_hf(
    query_sum: ttnn.Tensor,
    valid_mask_ttnn: ttnn.Tensor,
    batch_size: int,
    n_out: int,
    k_pool: int,
    mesh_device,
    mesh_mapper,
    is_mesh_device: bool,
) -> ttnn.Tensor:
    """
    Query = sum(gathered) / count(valid) per output slot (matches HF Molmo2VisionBackbone when pooling_attention_mask is true).
    valid_mask_ttnn: [1, 1, B*N_out*K_pool, 1] with 1.0 = valid, 0 = pad.
    """
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh_device else None
    vm = ttnn.to_torch(valid_mask_ttnn, mesh_composer=mesh_composer)
    if is_mesh_device:
        vm = vm[0]
    vm = vm.reshape(batch_size * n_out, k_pool).float()
    denom = vm.sum(dim=-1, keepdim=True).clamp(min=1.0)
    denom_4d = denom.reshape(1, batch_size * n_out, 1, 1).float()
    denom_ttnn = ttnn.from_torch(
        denom_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    out = ttnn.div(query_sum, denom_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(denom_ttnn)
    return out


def _pooling_attn_mask_ttnn(
    valid_bool: torch.Tensor,
    batch_size: int,
    n_out: int,
    k_pool: int,
    mesh_device,
    mesh_mapper,
) -> ttnn.Tensor:
    """Additive mask for image_pooling_2d: 0 = attend, -inf = ignore (HF-aligned). valid_bool: [B, N_out, K_pool]."""
    vm = valid_bool.reshape(batch_size * n_out, k_pool).float()
    mask = torch.where(vm[:, None, None, :] > 0.5, 0.0, float("-inf"))
    return ttnn.from_torch(
        mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _masked_mean_query_hf_device(
    query_sum: ttnn.Tensor,
    valid_mask_ttnn: ttnn.Tensor,
    batch_size: int,
    n_out: int,
    k_pool: int,
) -> ttnn.Tensor:
    """
    Same math as _masked_mean_query_hf but fully on device (no host reads).
    Required for ttnn trace capture (reads / to_torch are not allowed during capture).
    """
    vm = ttnn.reshape(valid_mask_ttnn, [1, batch_size * n_out, k_pool, 1])
    denom_sum = ttnn.sum(vm, dim=2, keepdim=True)
    denom = ttnn.clamp(denom_sum, min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(denom_sum)
    out = ttnn.div(query_sum, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(denom)
    return out


def preprocess_pooling_indices_device(
    pooled_patches_idx: torch.Tensor,
    mesh_device,
    mesh_mapper,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, torch.Tensor]:
    """
    Preprocess pooling indices for transfer to device.

    CRITICAL: Do NOT use bfloat16 for indices - it only has 7 mantissa bits
    and cannot represent integers above ~256 accurately. For large video
    indices like 21869, bfloat16 would corrupt them.

    Args:
        pooled_patches_idx: Raw indices [batch, n_out, k_pool] with -1 for invalid
        mesh_device: TTNN mesh device
        mesh_mapper: Mesh mapper for replication

    Returns:
        Tuple of:
          - idx_ttnn: Clipped indices [1, batch*n_out*k_pool] as uint32 for embedding
          - valid_mask_ttnn: Valid mask [1, 1, batch*n_out*k_pool, 1] as bfloat16
          - valid_token: [batch, n_out] bool tensor on CPU for final filtering
    """
    batch_size = pooled_patches_idx.shape[0]
    n_out = pooled_patches_idx.shape[1]
    k_pool = pooled_patches_idx.shape[2]
    total_elements = batch_size * n_out * k_pool

    # CPU-side preprocessing: compute valid mask and clip indices
    # This avoids bfloat16 precision loss for large indices
    idx_flat = pooled_patches_idx.reshape(-1)
    valid_mask_cpu = (idx_flat >= 0).float()  # [total_elements]
    clipped_idx_cpu = torch.clamp(idx_flat, min=0).to(torch.int32)  # [total_elements]

    # Reshape for TTNN
    clipped_idx_cpu = clipped_idx_cpu.reshape(1, -1)  # [1, total_elements]
    valid_mask_cpu = valid_mask_cpu.reshape(1, 1, -1, 1)  # [1, 1, total_elements, 1]

    # Transfer to device as uint32 (no precision loss)
    idx_ttnn = ttnn.from_torch(
        clipped_idx_cpu,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # Transfer valid mask as bfloat16 (0/1 values are fine in bfloat16)
    valid_mask_ttnn = ttnn.from_torch(
        valid_mask_cpu,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # CPU: compute valid_token for final filtering (small tensor, fast)
    valid = pooled_patches_idx >= 0
    valid_token = torch.any(valid, dim=-1)  # [batch, n_out]

    return idx_ttnn, valid_mask_ttnn, valid_token


def preprocess_pooling_indices_chunk_device(
    chunk_idx: torch.Tensor,
    mesh_device,
    mesh_mapper,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Preprocess pooling indices for a single chunk.

    CRITICAL: Do NOT use bfloat16 for indices - it only has 7 mantissa bits
    and cannot represent integers above ~256 accurately. For index 11663,
    bfloat16 would corrupt it to a much smaller value.

    Args:
        chunk_idx: Raw indices [chunk_frames, n_out, k_pool] with -1 for invalid
        mesh_device: TTNN mesh device
        mesh_mapper: Mesh mapper for replication

    Returns:
        Tuple of:
          - idx_ttnn: Clipped indices [1, chunk_frames*n_out*k_pool] as uint32
          - valid_mask_ttnn: Valid mask [1, 1, chunk_frames*n_out*k_pool, 1] as bfloat16
    """
    total_elements = chunk_idx.numel()

    # CPU-side preprocessing: compute valid mask and clip indices
    # This avoids bfloat16 precision loss for large indices
    idx_flat = chunk_idx.reshape(-1)
    valid_mask_cpu = (idx_flat >= 0).float()  # [total_elements]
    clipped_idx_cpu = torch.clamp(idx_flat, min=0).to(torch.int32)  # [total_elements]

    # Reshape for TTNN
    clipped_idx_cpu = clipped_idx_cpu.reshape(1, -1)  # [1, total_elements]
    valid_mask_cpu = valid_mask_cpu.reshape(1, 1, -1, 1)  # [1, 1, total_elements, 1]

    # Transfer to device as uint32 (no precision loss)
    idx_ttnn = ttnn.from_torch(
        clipped_idx_cpu,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # Transfer valid mask as bfloat16 (0/1 values are fine in bfloat16)
    valid_mask_ttnn = ttnn.from_torch(
        valid_mask_cpu,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    return idx_ttnn, valid_mask_ttnn


from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.image_pooling import ImagePooling
from models.demos.molmo2.tt.image_projector import ImageProjector
from models.demos.molmo2.tt.vision_transformer import VisionTransformer


class VisionBackbone(LightweightModule):
    """
    Complete vision backbone for Molmo2.

    Processes images through ViT, pools features using cross-attention,
    and projects to the language model hidden dimension.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        # ViT config
        vit_num_layers: int = 25,
        vit_hidden_dim: int = 1152,
        vit_intermediate_dim: int = 4304,
        vit_num_heads: int = 16,
        vit_head_dim: int = 72,
        patch_size: int = 14,
        image_size: int = 378,
        # Feature layers (0-indexed) - match HF order: [-3, -9] -> [24, 18]
        feature_layers: Tuple[int, int] = (24, 18),
        # Adapter config
        adapter_hidden_dim: int = 1152,
        adapter_intermediate_dim: int = 12288,
        adapter_num_heads: int = 16,
        adapter_head_dim: int = 72,
        # Output config
        output_dim: int = 4096,
        # Common config
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        dtype=ttnn.bfloat16,  # Changed from bfloat8_b for better vision precision
        use_tensor_parallel: bool = False,  # ViT uses DP (data parallel) for frames, not TP
        # Pre-allocated bounds (see _init_pooling_aux_buffers / _init_vision_trace_io_buffers)
        max_pooling_slots: int = DEFAULT_VISION_TRACE_MAX_FRAMES * DEFAULT_VISION_TRACE_MAX_N_OUT,
        max_k_pool: int = DEFAULT_VISION_TRACE_MAX_K_POOL,
        vision_trace_max_frames: int = DEFAULT_VISION_TRACE_MAX_FRAMES,
        patches_per_frame: int = DEFAULT_PATCHES_PER_FRAME,
        vision_trace_max_n_out: int = DEFAULT_VISION_TRACE_MAX_N_OUT,
        vision_trace_max_k_pool: int = DEFAULT_VISION_TRACE_MAX_K_POOL,
    ):
        """
        Initialize VisionBackbone.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing all weights
            vit_num_layers: Number of ViT layers to use (25)
            vit_hidden_dim: ViT hidden dimension (1152)
            vit_intermediate_dim: ViT MLP intermediate dimension (4304)
            vit_num_heads: ViT attention heads (16)
            vit_head_dim: ViT head dimension (72)
            patch_size: Image patch size (14)
            image_size: Expected image size (378)
            feature_layers: ViT layers to extract features from (24, 18) - HF order
            adapter_hidden_dim: Adapter hidden dimension (1152)
            adapter_intermediate_dim: Projector intermediate dimension (12288)
            adapter_num_heads: Pooling attention heads (16)
            adapter_head_dim: Pooling head dimension (72)
            output_dim: Output dimension for language model (4096)
            layer_norm_eps: Epsilon for LayerNorm
            weight_cache_path: Path to cache weights
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.feature_layers = feature_layers
        self.vit_hidden_dim = vit_hidden_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.output_dim = output_dim

        # Pool input dimension is concat of feature layers
        pool_input_dim = vit_hidden_dim * len(feature_layers)  # 2304

        # Vision Transformer encoder (TP=8 shards weights across devices)
        self.image_vit = VisionTransformer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            num_layers=vit_num_layers,
            hidden_dim=vit_hidden_dim,
            intermediate_dim=vit_intermediate_dim,
            num_heads=vit_num_heads,
            head_dim=vit_head_dim,
            patch_size=patch_size,
            image_size=image_size,
            layer_norm_eps=layer_norm_eps,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_vit",
            dtype=dtype,
            use_tensor_parallel=use_tensor_parallel,
        )

        # Image pooling (cross-attention)
        self.image_pooling_2d = ImagePooling(
            mesh_device=mesh_device,
            state_dict=state_dict,
            input_dim=pool_input_dim,
            hidden_dim=adapter_hidden_dim,
            num_heads=adapter_num_heads,
            head_dim=adapter_head_dim,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_pooling_2d",
            dtype=dtype,
        )

        # Image projector (SwiGLU)
        self.image_projector = ImageProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            input_dim=adapter_hidden_dim,
            intermediate_dim=adapter_intermediate_dim,
            output_dim=output_dim,
            weight_cache_path=weight_cache_path,
            state_dict_prefix="model.vision_backbone.image_projector",
            dtype=dtype,
        )

        self._max_pooling_slots = max_pooling_slots
        self._max_k_pool = max_k_pool
        self._vision_trace_max_frames = vision_trace_max_frames
        self._patches_per_frame = patches_per_frame
        self._vision_trace_max_n_out = vision_trace_max_n_out
        self._vision_trace_max_k_pool = vision_trace_max_k_pool
        self._vision_trace_max_tokens = vision_trace_max_frames * patches_per_frame
        self._vision_trace_max_idx_len = vision_trace_max_frames * vision_trace_max_n_out * vision_trace_max_k_pool
        self._vision_trace_max_valid_slots = vision_trace_max_frames * vision_trace_max_n_out

        self._init_pooling_aux_buffers()
        self._init_vision_trace_io_buffers()

    def _init_pooling_aux_buffers(self) -> None:
        """Pre-create zeros / -inf tensors for pooling attention mask (no per-forward alloc)."""
        shape_4d = (self._max_pooling_slots, 1, 1, self._max_k_pool)
        z = torch.zeros(shape_4d, dtype=torch.bfloat16)
        ni = torch.full(shape_4d, float("-inf"), dtype=torch.bfloat16)
        is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None
        self._pooling_attn_zeros_buf = ttnn.from_torch(
            z,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self._pooling_attn_neg_inf_buf = ttnn.from_torch(
            ni,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _init_vision_trace_io_buffers(self) -> None:
        """Pre-allocate max-sized vision trace input buffers (demo / vLLM execute_trace copies into these)."""
        self._vision_trace_embedded_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self._vision_trace_max_tokens, self.vit_hidden_dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vision_trace_idx_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self._vision_trace_max_idx_len]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vision_trace_mask_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self._vision_trace_max_idx_len, 1]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vision_trace_valid_token_full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self._vision_trace_max_valid_slots]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_vision_trace_tensors(
        self,
        batch_size: int,
        n_out: int,
        k_pool: int,
        num_patches: int,
    ) -> Dict:
        """
        Return views into pre-allocated vision trace buffers for the given logical shape.

        All buffers are allocated in __init__; this only slices to the active prefix.
        """
        num_tokens = batch_size * num_patches
        idx_len = batch_size * n_out * k_pool
        valid_slots = batch_size * n_out
        if num_tokens > self._vision_trace_max_tokens:
            raise ValueError(f"Vision trace: num_tokens={num_tokens} exceeds init max {self._vision_trace_max_tokens}")
        if idx_len > self._vision_trace_max_idx_len:
            raise ValueError(f"Vision trace: idx_len={idx_len} exceeds init max {self._vision_trace_max_idx_len}")
        if valid_slots > self._vision_trace_max_valid_slots:
            raise ValueError(
                f"Vision trace: batch_size*n_out={valid_slots} exceeds init max {self._vision_trace_max_valid_slots}"
            )
        if batch_size > self._vision_trace_max_frames:
            raise ValueError(f"Vision trace: batch_size={batch_size} exceeds init max {self._vision_trace_max_frames}")
        if n_out > self._vision_trace_max_n_out or k_pool > self._vision_trace_max_k_pool:
            raise ValueError(
                f"Vision trace: n_out={n_out}, k_pool={k_pool} exceed init "
                f"max_n_out={self._vision_trace_max_n_out}, max_k_pool={self._vision_trace_max_k_pool}"
            )

        embedded = self._vision_trace_embedded_full[:, :, :num_tokens, :]
        idx = self._vision_trace_idx_full[:, :idx_len]
        valid_mask = self._vision_trace_mask_full[:, :, :idx_len, :]
        valid_token = self._vision_trace_valid_token_full[:valid_slots]
        return {
            "embedded": embedded,
            "idx": idx,
            "valid_mask": valid_mask,
            "valid_token": valid_token,
            "n_out": n_out,
            "k_pool": k_pool,
            "batch_size": batch_size,
        }

    def _pooling_attn_mask_from_valid_ttnn(
        self,
        valid_mask_ttnn: ttnn.Tensor,
        batch_size: int,
        n_out: int,
        k_pool: int,
    ) -> ttnn.Tensor:
        """
        Additive attention mask for image_pooling_2d from device valid_mask only.
        Shape [B*N_out, 1, 1, K_pool]; uses pre-allocated zeros / -inf from __init__.
        """
        slots = batch_size * n_out
        if slots > self._max_pooling_slots or k_pool > self._max_k_pool:
            raise ValueError(
                f"Pooling mask needs slots={slots}, k_pool={k_pool}; init bounds are "
                f"max_pooling_slots={self._max_pooling_slots}, max_k_pool={self._max_k_pool}"
            )
        vm = ttnn.reshape(valid_mask_ttnn, [1, slots, k_pool, 1])
        vm = ttnn.reshape(vm, [slots, k_pool, 1])
        vm4 = ttnn.reshape(vm, [slots, 1, 1, k_pool])
        zeros = self._pooling_attn_zeros_buf[0:slots, 0:1, 0:1, 0:k_pool]
        neg_inf = self._pooling_attn_neg_inf_buf[0:slots, 0:1, 0:1, 0:k_pool]
        return ttnn.where(vm4, zeros, neg_inf, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Class-level request counter for debugging
    _encode_image_request_count = 0

    def encode_image(
        self,
        images_embedded: ttnn.Tensor,
        num_crops: int = 1,
        trace_capture: bool = False,
    ) -> ttnn.Tensor:
        """
        Encode images through ViT and extract multi-scale features.

        Args:
            images_embedded: Embedded image patches [B*T, N, hidden_dim]
                             after patch embedding and positional embedding
            num_crops: Number of crops per image (T)
            trace_capture: If True, skip host reads for per-layer debug stats (for ttnn trace capture).

        Returns:
            Concatenated multi-scale features [B, T*N, pool_input_dim]
        """
        from loguru import logger

        # Track request count for debugging
        VisionBackbone._encode_image_request_count += 1
        request_num = VisionBackbone._encode_image_request_count
        logger.info(f"encode_image REQUEST #{request_num}: Starting ViT forward...")
        logger.info(f"  images_embedded shape: {list(images_embedded.shape)}")

        # Run through ViT and collect all hidden states
        hidden_states = self.image_vit.forward(
            images_embedded,
            return_all_hidden_states=True,
        )

        logger.info(
            f"encode_image REQUEST #{request_num}: ViT forward complete, got {len(hidden_states)} hidden states"
        )

        # Extract features from specified layers and concat
        features = []
        used_indices = set(self.feature_layers)
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])
            # Debug: per-layer stats (skip host reads during trace capture — forbidden)
            if trace_capture:
                logger.info(f"  Layer {layer_idx}: shape={list(hidden_states[layer_idx].shape)}")
            else:
                try:
                    layer_torch = (
                        ttnn.to_torch(hidden_states[layer_idx], mesh_composer=mesh_composer)[0]
                        if is_mesh_device
                        else ttnn.to_torch(hidden_states[layer_idx])
                    )
                    logger.info(
                        f"  Layer {layer_idx}: shape={list(layer_torch.shape)}, mean={layer_torch.mean():.4f}, std={layer_torch.std():.4f}, min={layer_torch.min():.4f}, max={layer_torch.max():.4f}"
                    )
                except Exception:
                    logger.info(f"  Using layer {layer_idx}, shape: {list(hidden_states[layer_idx].shape)}")

        # CRITICAL: Deallocate unused hidden states to prevent memory leak
        # For video with 8 frames, each hidden state is ~27MB, 25 layers = ~670MB total
        # Only using 2 layers means 23 layers (~620MB) would be leaked per request!
        deallocated_count = 0
        for i, hs in enumerate(hidden_states):
            if i not in used_indices:
                ttnn.deallocate(hs)
                deallocated_count += 1
        logger.info(f"encode_image REQUEST #{request_num}: Deallocated {deallocated_count} unused hidden states")

        # Concatenate on hidden dimension
        # Each feature is [1, 1, B*T*N, hidden_dim]
        image_features = ttnn.concat(features, dim=-1)

        # NOTE: Do NOT deallocate features here - ttnn.concat may return a view that references inputs
        # The deallocation of unused hidden states above is sufficient for memory management
        logger.info(f"encode_image REQUEST #{request_num}: Concat complete (keeping feature tensors)")

        logger.info(f"encode_image REQUEST #{request_num}: Complete, output shape: {list(image_features.shape)}")

        return image_features

    def encode_image_from_pixels(
        self,
        pixel_values: torch.Tensor,
        num_crops: int = 1,
    ) -> ttnn.Tensor:
        """
        Encode images from raw pixel values through ViT.

        Args:
            pixel_values: Raw pixel values [B, C, H, W] as torch tensor
            num_crops: Number of crops per image (T)

        Returns:
            Concatenated multi-scale features [B, T*N, pool_input_dim]
        """
        # Run through ViT with patch embedding
        hidden_states = self.image_vit.forward_with_patch_embed(
            pixel_values,
            return_all_hidden_states=True,
        )

        # Extract features from specified layers and concat
        features = []
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])

        # Concatenate on hidden dimension
        image_features = ttnn.concat(features, dim=-1)

        return image_features

    def forward(
        self,
        images_embedded,
        pooled_patches_idx: torch.Tensor,
        num_crops: int = 1,
        use_attention_mask: bool = True,
    ) -> ttnn.Tensor:
        """
        Full forward pass through vision backbone.

        Args:
            images_embedded: Either:
                - Embedded image patches [1, 1, B*T*N, hidden_dim] (ttnn.Tensor)
                - Raw pixel values [B, C, H, W] (torch.Tensor)
            pooled_patches_idx: Indices for gathering patch features (on CPU)
                                Shape: [B, N_out, K_pool]
            num_crops: Number of crops per image (T)
            use_attention_mask: Whether to use attention mask in pooling

        Returns:
            Visual embeddings for language model [valid_tokens, output_dim]
        """
        batch_size = pooled_patches_idx.shape[0]

        # 1. Encode image through ViT
        # Check if input is raw pixels (torch.Tensor) or already embedded (ttnn.Tensor)
        if isinstance(images_embedded, torch.Tensor):
            image_features = self.encode_image_from_pixels(images_embedded, num_crops)
        else:
            image_features = self.encode_image(images_embedded, num_crops)

        # 2. Convert to torch for gathering (CPU operation)
        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        if is_mesh_device:
            image_features_torch = ttnn.to_torch(
                image_features, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[0]
        else:
            image_features_torch = ttnn.to_torch(image_features)

        image_features_torch = image_features_torch.squeeze(0).squeeze(0)  # [B*T*N, pool_dim]

        # 3. Gather features using pooled_patches_idx (CPU)
        # pooled_patches_idx: [B, N_out, K_pool] - indices into flattened features
        pool_dim = image_features_torch.shape[-1]
        n_out = pooled_patches_idx.shape[1]
        k_pool = pooled_patches_idx.shape[2]

        # Identify valid indices (>= 0)
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, dim=-1)  # [B, N_out]

        # Gather features
        # Flatten features: [B*T*N, pool_dim] -> use as lookup table
        batch_idx = torch.arange(batch_size, dtype=torch.long)
        batch_idx = batch_idx.view(batch_size, 1, 1).expand(-1, n_out, k_pool)

        # Clip negative indices to 0 for gathering (will be masked later)
        clipped_idx = torch.clip(pooled_patches_idx, min=0)

        # Reshape features for batched indexing: [B, T*N, pool_dim]
        total_patches = image_features_torch.shape[0] // batch_size
        features_batched = image_features_torch.reshape(batch_size, total_patches, pool_dim)

        # Gather: [B, N_out, K_pool, pool_dim]
        to_pool = features_batched[batch_idx, clipped_idx]

        # Mask invalid positions
        to_pool = to_pool * valid.unsqueeze(-1).float()

        # Reshape for attention: [B*N_out, K_pool, pool_dim]
        to_pool = to_pool.reshape(-1, k_pool, pool_dim)

        # 4. Compute query (mean of valid features)
        if use_attention_mask:
            valid_flat = valid.reshape(-1, k_pool).float()
            denom = valid_flat.sum(-1, keepdim=True)
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            query = to_pool.sum(-2, keepdim=True) / denom.unsqueeze(-1)
            attn_mask = valid.reshape(-1, 1, 1, k_pool).float()
            # Convert mask to additive form for SDPA
            attn_mask = torch.where(attn_mask == 0, float("-inf"), 0.0)
        else:
            query = to_pool.mean(-2, keepdim=True)
            attn_mask = None

        # 5. Convert back to TTNN for pooling
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        query_ttnn = ttnn.from_torch(
            query.unsqueeze(0),  # [1, B*N_out, 1, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        to_pool_ttnn = ttnn.from_torch(
            to_pool.unsqueeze(0),  # [1, B*N_out, K_pool, pool_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        if attn_mask is not None:
            attn_mask_ttnn = ttnn.from_torch(
                attn_mask,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
        else:
            attn_mask_ttnn = None

        # 6. Cross-attention pooling
        pooled_features_raw = self.image_pooling_2d(
            query=query_ttnn,
            key_value=to_pool_ttnn,
            attn_mask=attn_mask_ttnn,
        )

        ttnn.deallocate(query_ttnn)
        ttnn.deallocate(to_pool_ttnn)
        if attn_mask_ttnn is not None:
            ttnn.deallocate(attn_mask_ttnn)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features_raw, [1, 1, batch_size * n_out, -1])
        # NOTE: Do NOT deallocate pooled_features_raw - reshape may return a view

        # 7. Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)
        ttnn.deallocate(pooled_features)

        # 8. Filter by valid tokens (return only valid embeddings)
        # Convert to torch for filtering
        if is_mesh_device:
            visual_embeddings_torch = ttnn.to_torch(
                visual_embeddings, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[0]
        else:
            visual_embeddings_torch = ttnn.to_torch(visual_embeddings)

        # CRITICAL: Deallocate TTNN tensor after converting to torch
        ttnn.deallocate(visual_embeddings)

        visual_embeddings_torch = visual_embeddings_torch.squeeze(0).squeeze(0)  # [B*N_out, output_dim]

        # Apply valid token mask
        valid_embeddings = visual_embeddings_torch[valid_token.flatten()]

        return valid_embeddings

    def forward_encode_only(
        self,
        images_embedded: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Encode images only (without pooling/projection).

        Useful for testing the ViT encoder separately.

        Args:
            images_embedded: Embedded image patches [1, 1, N, hidden_dim]

        Returns:
            Concatenated multi-scale features [1, 1, N, pool_input_dim]
        """
        return self.encode_image(images_embedded)

    def forward_ttnn(
        self,
        images_embedded: ttnn.Tensor,
        pooled_patches_idx_ttnn: ttnn.Tensor,
        valid_mask_ttnn: ttnn.Tensor,
        valid_token_ttnn: ttnn.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int = 1,
        trace_capture: bool = False,
    ) -> ttnn.Tensor:
        """
        Full forward pass using TTNN ops (traceable).

        This version keeps everything in TTNN to enable tracing.

        Args:
            images_embedded: Embedded image patches [1, 1, B*T*N, hidden_dim]
            pooled_patches_idx_ttnn: Flattened indices [1, B*N_out*K_pool] (clipped to >= 0)
            valid_mask_ttnn: Valid mask [1, 1, B*N_out*K_pool, 1] for masking gathered features
            valid_token_ttnn: Valid token mask [B*N_out] for final filtering
            n_out: Number of output positions
            k_pool: Pooling kernel size
            batch_size: Batch size (default 1)

        Returns:
            Visual embeddings [1, 1, B*N_out, output_dim]

        Args:
            trace_capture: If True, skip all host tensor reads (required inside ttnn.begin_trace_capture).
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh_device else None

        def _stats(t, name):
            try:
                x = ttnn.to_torch(t, mesh_composer=mesh_composer)[0] if is_mesh_device else ttnn.to_torch(t)
                return f"{name}: shape={list(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}"
            except Exception:
                return f"{name}: stats unavailable"

        # 1. Encode image through ViT
        image_features = self.encode_image(images_embedded, trace_capture=trace_capture)
        if not trace_capture:
            logger.debug(_stats(image_features, "ViT features"))
        # image_features: [1, 1, B*T*N, pool_dim]

        # Squeeze to 2D for embedding lookup: [B*T*N, pool_dim]
        image_features_2d = ttnn.reshape(image_features, [-1, image_features.shape[-1]])

        # Convert to ROW_MAJOR for embedding lookup (embedding table must be ROW_MAJOR)
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

        # 2. Gather features using ttnn.embedding
        # pooled_patches_idx_ttnn: [1, B*N_out*K_pool] contains indices into B*T*N
        gathered = ttnn.embedding(
            pooled_patches_idx_ttnn,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # gathered: [1, B*N_out*K_pool, pool_dim]

        # Reshape to [1, 1, B*N_out*K_pool, pool_dim] for masking
        pool_dim = image_features.shape[-1]
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])

        # 3. Apply valid mask (zero out invalid positions)
        # valid_mask_ttnn: [1, 1, B*N_out*K_pool, 1]
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if not trace_capture:
            logger.debug(_stats(gathered, "gathered (after mask)"))

        # Reshape to [1, B*N_out, K_pool, pool_dim]
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])

        # 4. Query = masked mean (HF: sum / count(valid) per slot)
        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, B*N_out, 1, pool_dim]
        query = _masked_mean_query_hf_device(
            query_sum,
            valid_mask_ttnn,
            batch_size,
            n_out,
            k_pool,
        )
        ttnn.deallocate(query_sum)
        if not trace_capture:
            logger.debug(_stats(query, "query (masked mean)"))
            logger.debug(_stats(to_pool, "to_pool (key/value)"))

        # 5. Pooling KV mask (HF: -inf on invalid keys) — device-only, pre-alloc zeros/neg_inf from __init__
        attn_mask_ttnn = self._pooling_attn_mask_from_valid_ttnn(
            valid_mask_ttnn,
            batch_size,
            n_out,
            k_pool,
        )

        pooled_features = self.image_pooling_2d(
            query=query,
            key_value=to_pool,
            attn_mask=attn_mask_ttnn,
        )
        ttnn.deallocate(attn_mask_ttnn)
        if not trace_capture:
            logger.debug(_stats(pooled_features, "pooled_features (after pooling, before projection)"))

        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)
        ttnn.deallocate(gathered)
        ttnn.deallocate(image_features)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # 6. Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features, trace_capture=trace_capture)
        ttnn.deallocate(pooled_features)

        return visual_embeddings

    def encode_image_only_ttnn(self, images_embedded: ttnn.Tensor) -> ttnn.Tensor:
        """
        Run only ViT encoding without pooling/projection.

        Use this for chunked video processing where pooling needs cross-frame
        attention across ALL frames. Call this in chunks, concatenate the
        outputs, then call pool_and_project_ttnn on the combined features.

        Args:
            images_embedded: Embedded patches [1, 1, B*N, hidden_dim]

        Returns:
            image_features: ViT output [1, 1, B*N, pool_dim] (concatenated multi-scale features)
        """
        return self.encode_image(images_embedded)

    def pool_and_project_ttnn(
        self,
        image_features: ttnn.Tensor,
        pooled_patches_idx_ttnn: ttnn.Tensor,
        valid_mask_ttnn: ttnn.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int,
    ) -> ttnn.Tensor:
        """
        Run gather, pooling, and projection on pre-computed ViT features.

        Use this after encode_image_only_ttnn when doing chunked processing.
        The image_features should contain ALL frames concatenated so that
        pooling can attend across all frames via global indices.

        Args:
            image_features: ViT output [1, 1, total_patches, pool_dim] from ALL frames
            pooled_patches_idx_ttnn: Flattened GLOBAL indices [1, B*N_out*K_pool]
            valid_mask_ttnn: Valid mask [1, 1, B*N_out*K_pool, 1]
            n_out: Number of output positions per frame
            k_pool: Pooling kernel size
            batch_size: Total number of frames

        Returns:
            visual_embeddings: [1, 1, B*N_out, output_dim]
        """
        from loguru import logger

        logger.debug(f"pool_and_project_ttnn (SINGLE-PASS): batch_size={batch_size}, n_out={n_out}, k_pool={k_pool}")

        # Squeeze to 2D for embedding lookup: [total_patches, pool_dim]
        image_features_2d = ttnn.reshape(image_features, [-1, image_features.shape[-1]])

        # Convert to ROW_MAJOR for embedding lookup
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

        # Gather features using ttnn.embedding with GLOBAL indices
        gathered = ttnn.embedding(
            pooled_patches_idx_ttnn,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Deallocate 2D embedding table now that gather is complete
        ttnn.deallocate(image_features_2d)
        # gathered: [1, B*N_out*K_pool, pool_dim]

        # Reshape to [1, 1, B*N_out*K_pool, pool_dim] for masking
        pool_dim = image_features.shape[-1]
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])

        # Apply valid mask (zero out invalid positions)
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape to [1, B*N_out, K_pool, pool_dim]
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])

        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)  # [1, B*N_out, 1, pool_dim]
        query = _masked_mean_query_hf_device(query_sum, valid_mask_ttnn, batch_size, n_out, k_pool)
        ttnn.deallocate(query_sum)

        attn_mask_ttnn = self._pooling_attn_mask_from_valid_ttnn(valid_mask_ttnn, batch_size, n_out, k_pool)

        pooled_features = self.image_pooling_2d(
            query=query,
            key_value=to_pool,
            attn_mask=attn_mask_ttnn,
        )
        ttnn.deallocate(attn_mask_ttnn)
        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)
        ttnn.deallocate(gathered)

        # Reshape: [1, B*N_out, 1, hidden_dim] -> [1, 1, B*N_out, hidden_dim]
        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])

        # Project to language model dimension
        visual_embeddings = self.image_projector(pooled_features)
        ttnn.deallocate(pooled_features)

        return visual_embeddings

    def pool_and_project_from_features_ttnn(
        self,
        image_features_2d: "ttnn.Tensor",
        pooled_patches_idx_ttnn: "ttnn.Tensor",
        valid_mask_ttnn: "ttnn.Tensor",
        n_out: int,
        k_pool: int,
        batch_size: int,
        trace_capture: bool = False,
    ) -> "ttnn.Tensor":
        """
        Gather, pool, and project starting from a pre-computed 2D feature table.

        This is the fully-traceable pooling+projection path used by the DP=8 video
        pipeline after ViT features have been gathered from all devices and the full
        feature table has been uploaded to device.

        Args:
            image_features_2d: ViT output already reshaped to [total_patches, pool_dim]
                                ROW_MAJOR (vocabulary for ttnn.embedding lookup).
            pooled_patches_idx_ttnn: Flattened GLOBAL indices [1, batch*n_out*k_pool] uint32.
            valid_mask_ttnn: Valid mask [1, 1, batch*n_out*k_pool, 1] bfloat16.
            n_out: Number of output positions per frame.
            k_pool: Pooling kernel size.
            batch_size: Number of frames in this pool operation.
            trace_capture: Skip host reads when inside ttnn trace capture.

        Returns:
            visual_embeddings: [1, 1, batch*n_out, output_dim]
        """
        pool_dim = image_features_2d.shape[-1]

        gathered = ttnn.embedding(
            pooled_patches_idx_ttnn,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered = ttnn.reshape(gathered, [1, 1, batch_size * n_out * k_pool, pool_dim])
        gathered = ttnn.mul(gathered, valid_mask_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        to_pool = ttnn.reshape(gathered, [1, batch_size * n_out, k_pool, pool_dim])
        ttnn.deallocate(gathered)

        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
        query = _masked_mean_query_hf_device(query_sum, valid_mask_ttnn, batch_size, n_out, k_pool)
        ttnn.deallocate(query_sum)

        attn_mask_ttnn = self._pooling_attn_mask_from_valid_ttnn(valid_mask_ttnn, batch_size, n_out, k_pool)
        pooled_features = self.image_pooling_2d(query=query, key_value=to_pool, attn_mask=attn_mask_ttnn)
        ttnn.deallocate(attn_mask_ttnn)
        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)

        pooled_features = ttnn.reshape(pooled_features, [1, 1, batch_size * n_out, -1])
        visual_embeddings = self.image_projector(pooled_features, trace_capture=trace_capture)
        ttnn.deallocate(pooled_features)

        return visual_embeddings

    def pool_chunk_from_features_ttnn(
        self,
        image_features_2d: "ttnn.Tensor",
        idx_chunk: "ttnn.Tensor",
        valid_mask_chunk: "ttnn.Tensor",
        chunk_frames: int,
        n_out: int,
        k_pool: int,
    ) -> "ttnn.Tensor":
        """
        One chunk of gather+pool WITHOUT projection (traceable).

        Used in the chunked pooling trace loop: each chunk is traced once and
        replayed per chunk; the projection is applied after all chunks are
        concatenated.

        Args:
            image_features_2d: Full ViT feature table [total_patches, pool_dim] ROW_MAJOR.
            idx_chunk: GLOBAL indices for this chunk [1, chunk_frames*n_out*k_pool] uint32.
            valid_mask_chunk: Valid mask [1, 1, chunk_frames*n_out*k_pool, 1] bfloat16.
            chunk_frames: Number of output frames in this chunk.
            n_out: Number of output positions per frame.
            k_pool: Pooling kernel size.

        Returns:
            pooled_chunk: [1, 1, chunk_frames*n_out, pool_dim]
        """
        pool_dim = image_features_2d.shape[-1]

        gathered = ttnn.embedding(
            idx_chunk,
            image_features_2d,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered = ttnn.reshape(gathered, [1, 1, chunk_frames * n_out * k_pool, pool_dim])
        gathered = ttnn.mul(gathered, valid_mask_chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        to_pool = ttnn.reshape(gathered, [1, chunk_frames * n_out, k_pool, pool_dim])
        ttnn.deallocate(gathered)

        query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
        query = _masked_mean_query_hf_device(query_sum, valid_mask_chunk, chunk_frames, n_out, k_pool)
        ttnn.deallocate(query_sum)

        attn_mask = self._pooling_attn_mask_from_valid_ttnn(valid_mask_chunk, chunk_frames, n_out, k_pool)
        pooled = self.image_pooling_2d(query=query, key_value=to_pool, attn_mask=attn_mask)
        ttnn.deallocate(attn_mask)
        ttnn.deallocate(query)
        ttnn.deallocate(to_pool)

        pooled = ttnn.reshape(pooled, [1, 1, chunk_frames * n_out, -1])
        return pooled

    def pool_and_project_chunked_ttnn(
        self,
        image_features: ttnn.Tensor,
        pooled_patches_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        n_out: int,
        k_pool: int,
        batch_size: int,
        max_frames_per_pool_chunk: int = 16,
    ) -> ttnn.Tensor:
        """
        Chunked pooling for large videos - processes output frames in chunks
        while keeping all ViT features available for cross-frame attention.

        This reduces peak memory by processing pooling in batches, but still
        allows each output to reference patches from ANY frame via global indices.

        Args:
            image_features: ViT output [1, 1, total_patches, pool_dim] from ALL frames
            pooled_patches_idx: GLOBAL indices [batch_size, n_out, k_pool] (torch, CPU)
            valid_mask: Valid mask [batch_size, n_out, k_pool] (torch, CPU)
            n_out: Number of output positions per frame
            k_pool: Pooling kernel size
            batch_size: Total number of frames
            max_frames_per_pool_chunk: Max frames to pool at once

        Returns:
            visual_embeddings: [1, 1, B*N_out, output_dim]
        """
        from loguru import logger

        is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

        logger.debug(f"pool_and_project_chunked: {batch_size} frames in chunks of {max_frames_per_pool_chunk}")

        # Convert ViT features to 2D embedding table (stays on device throughout)
        pool_dim = image_features.shape[-1]
        image_features_2d = ttnn.reshape(image_features, [-1, pool_dim])
        image_features_2d = ttnn.to_layout(image_features_2d, ttnn.ROW_MAJOR_LAYOUT)

        # Deallocate original 4D tensor now
        ttnn.deallocate(image_features)

        # Process pooling in chunks of frames
        all_embeddings = []

        for chunk_start in range(0, batch_size, max_frames_per_pool_chunk):
            chunk_end = min(chunk_start + max_frames_per_pool_chunk, batch_size)
            chunk_frames = chunk_end - chunk_start

            # Extract chunk of indices (indices are still GLOBAL)
            chunk_idx = pooled_patches_idx[chunk_start:chunk_end]  # [chunk_frames, n_out, k_pool]

            # Preprocess indices (CPU-side to avoid bfloat16 precision loss)
            idx_ttnn, valid_ttnn = preprocess_pooling_indices_chunk_device(chunk_idx, self.mesh_device, mesh_mapper)

            # Gather using GLOBAL indices from full feature table
            gathered = ttnn.embedding(
                idx_ttnn,
                image_features_2d,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(idx_ttnn)

            # Reshape and mask
            gathered = ttnn.reshape(gathered, [1, 1, chunk_frames * n_out * k_pool, pool_dim])
            gathered = ttnn.mul(gathered, valid_ttnn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Reshape for pooling
            to_pool = ttnn.reshape(gathered, [1, chunk_frames * n_out, k_pool, pool_dim])
            ttnn.deallocate(gathered)

            # Use device-only ops for masked mean and attention mask (reuse valid_ttnn)
            query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
            query = _masked_mean_query_hf_device(query_sum, valid_ttnn, chunk_frames, n_out, k_pool)
            ttnn.deallocate(query_sum)

            attn_mask_ttnn = self._pooling_attn_mask_from_valid_ttnn(valid_ttnn, chunk_frames, n_out, k_pool)
            ttnn.deallocate(valid_ttnn)
            pooled = self.image_pooling_2d(query=query, key_value=to_pool, attn_mask=attn_mask_ttnn)
            ttnn.deallocate(attn_mask_ttnn)

            ttnn.deallocate(query)
            ttnn.deallocate(to_pool)

            # Reshape pooled features for later concatenation
            pooled = ttnn.reshape(pooled, [1, 1, chunk_frames * n_out, -1])

            # Keep on device - no CPU roundtrip
            logger.debug(f"  Chunk {chunk_start//max_frames_per_pool_chunk} pooled (on device): shape={pooled.shape}")
            all_embeddings.append(pooled)
            # Note: Don't deallocate pooled here - needed for concat

        # Deallocate embedding table
        ttnn.deallocate(image_features_2d)

        # Concatenate pooled features on device (no CPU roundtrip)
        if len(all_embeddings) == 1:
            combined_ttnn = all_embeddings[0]
        else:
            combined_ttnn = ttnn.concat(all_embeddings, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Deallocate individual chunks after concat
            for emb in all_embeddings:
                ttnn.deallocate(emb)

        logger.debug(f"pool_and_project_chunked: Combined pooled shape: {combined_ttnn.shape}")

        visual_embeddings = self.image_projector(combined_ttnn)
        ttnn.deallocate(combined_ttnn)

        return visual_embeddings
