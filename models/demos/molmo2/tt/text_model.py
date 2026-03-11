# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Text Model (Language Model Decoder) for Molmo2.

Implements the full decoder-only transformer:
- Token embedding (wte) + visual token embedding (new_embedding)
- 36 decoder blocks with GQA, QK-norm, and SwiGLU
- Final RMSNorm (ln_f)
- Language model head (lm_head)

Supports:
- Autoregressive generation with KV cache
- Mixed text/vision token sequences
"""

from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.text_block import TextBlock
from models.demos.molmo2.tt.text_rmsnorm import TextRMSNorm
from models.demos.molmo2.tt.text_rotary_emb import TextRotaryEmbedding
from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup


class TextModel(LightweightModule):
    """
    Full decoder-only language model for Molmo2.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        num_layers: int = 36,
        hidden_dim: int = 4096,
        intermediate_dim: int = 12288,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 152064,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextModel.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            num_layers: Number of decoder layers (36)
            hidden_dim: Hidden dimension (4096)
            intermediate_dim: MLP intermediate dimension (11008)
            num_heads: Number of query heads (32)
            num_kv_heads: Number of KV heads (8)
            head_dim: Dimension per head (128)
            vocab_size: Vocabulary size (152064)
            max_seq_len: Maximum sequence length (8192)
            rope_theta: RoPE theta (1,000,000)
            rms_norm_eps: Epsilon for RMSNorm
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dtype = dtype

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Token embedding - pre-concatenate wte + new_embedding for trace compatibility
        # This avoids ttnn.concat inside embed_tokens which fails during trace capture
        wte = state_dict[f"{state_dict_prefix}.wte.embedding"]
        new_embedding = state_dict[f"{state_dict_prefix}.wte.new_embedding"]

        # Concatenate on CPU: [vocab_size, hidden_dim] + [new_vocab_size, hidden_dim]
        # Result: [full_vocab_size, hidden_dim]
        full_embedding = torch.cat([wte, new_embedding], dim=0)

        # Store as single tensor for trace-compatible embed_tokens
        self.full_embedding = ttnn.as_tensor(
            full_embedding.unsqueeze(0).unsqueeze(0),  # [1, 1, full_vocab_size, hidden_dim]
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wte.full_embedding"),
        )

        # Rotary position embeddings (TTNN-native for prefill)
        self.rotary_setup = TextRotarySetup(
            mesh_device=mesh_device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            batch_size=1,  # Single batch for now
            datatype=ttnn.bfloat16,
        )
        self.transformation_mats = self.rotary_setup.get_transformation_mats()

        # Rotary embeddings for decode mode (PyTorch-based, simpler)
        self.rotary_emb = TextRotaryEmbedding(
            mesh_device=mesh_device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            dtype=ttnn.bfloat16,
        )

        # Decoder blocks
        self.blocks = []
        for layer_num in range(num_layers):
            block = TextBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_num=layer_num,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                weight_cache_path=weight_cache_path,
                state_dict_prefix=f"{state_dict_prefix}.blocks",
                dtype=dtype,
            )
            self.blocks.append(block)

        # Final layer normalization
        self.ln_f = TextRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=f"{state_dict_prefix}.ln_f",
        )

        # Language model head
        # Note: lm_head is at top level, not under model.transformer
        lm_head = state_dict["lm_head.weight"]
        lm_head_t = torch.transpose(lm_head, -2, -1).unsqueeze(0).unsqueeze(0)

        self.lm_head = ttnn.as_tensor(
            lm_head_t,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("lm_head.weight") if weight_cache_path else None,
        )

        # Compute kernel config for lm_head
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def embed_tokens(
        self,
        input_ids: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Embed input token IDs.

        Args:
            input_ids: Token IDs of shape [1, seq_len]

        Returns:
            Embeddings of shape [1, 1, seq_len, hidden_dim] in TILE layout

        Note:
            Uses pre-concatenated embedding table (wte + new_embedding) to handle
            both regular tokens and special tokens (image_patch_id, etc.)
            The concatenation is done once during __init__ to enable tracing.
        """
        # Use pre-concatenated embedding table (no concat during trace)
        embeddings = ttnn.embedding(input_ids, self.full_embedding)
        embeddings = ttnn.reshape(embeddings, [1, 1, -1, self.hidden_dim])
        # Convert to TILE layout for rest of model
        embeddings = ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)
        return embeddings

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int = 0,
        attn_mask: Optional[ttnn.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        output_hidden_states: bool = False,
        rot_mats: Optional[List[ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through text model (without embedding).

        Args:
            hidden_states: Input embeddings of shape [1, 1, seq_len, hidden_dim]
            start_pos: Starting position for KV cache
            attn_mask: Optional attention mask
            kv_caches: Optional list of (k_cache, v_cache) per layer
            output_hidden_states: Whether to return all hidden states
            rot_mats: Optional pre-computed rotation matrices [cos, sin] for tracing

        Returns:
            Tuple of (logits, new_kv_caches)
        """
        seq_len = hidden_states.shape[-2]

        # Get RoPE rotation matrices (prefill mode)
        # Use provided rot_mats if available (for tracing), otherwise compute
        if rot_mats is None:
            rot_mats = self.rotary_setup.get_rot_mats_prefill(seq_len, start_pos)

        # Process through decoder blocks
        all_hidden_states = [] if output_hidden_states else None
        new_kv_caches = []

        x = hidden_states
        for layer_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)

            kv_cache = kv_caches[layer_idx] if kv_caches else None
            x, new_kv_cache = block(x, rot_mats, self.transformation_mats, attn_mask, start_pos, kv_cache)
            new_kv_caches.append(new_kv_cache)

        # Final normalization
        x = self.ln_f(x)

        if output_hidden_states:
            all_hidden_states.append(x)

        # Language model head
        logits = ttnn.linear(
            x,
            self.lm_head,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return logits, new_kv_caches

    def forward_with_embedding(
        self,
        input_ids: ttnn.Tensor,
        start_pos: int = 0,
        attn_mask: Optional[ttnn.Tensor] = None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
    ) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass with embedding lookup.

        Args:
            input_ids: Token IDs of shape [1, seq_len]
            start_pos: Starting position for KV cache
            attn_mask: Optional attention mask
            kv_caches: Optional list of (k_cache, v_cache) per layer

        Returns:
            Tuple of (logits, new_kv_caches)
        """
        hidden_states = self.embed_tokens(input_ids)
        return self.forward(hidden_states, start_pos, attn_mask, kv_caches)

    def forward_decode(
        self,
        hidden_states: ttnn.Tensor,
        kv_caches: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
        current_pos: ttnn.Tensor,
        rot_mats: Optional[List[ttnn.Tensor]] = None,
        rot_mat_idxs: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Decode-mode forward pass (single token at a time).

        Uses paged_update_cache and scaled_dot_product_attention_decode
        for efficient autoregressive generation.

        Args:
            hidden_states: Input embeddings of shape [1, 1, 1, hidden_dim]
            kv_caches: List of (k_cache, v_cache) per layer
            current_pos: Current decode position tensor [batch]
            rot_mats: Optional pre-computed [cos, sin] rotation matrices (for tracing)
            rot_mat_idxs: Optional device tensor [1, padded_batch] for RoPE lookup

        Returns:
            Logits tensor
        """
        if rot_mats is None:
            assert rot_mat_idxs is not None, "Either rot_mats or rot_mat_idxs must be provided"
            rot_mats = self.rotary_setup.get_rot_mats_decode_traced(rot_mat_idxs)

        # Get decode transformation matrix
        transformation_mat = self.transformation_mats["decode"]

        # Process through decoder blocks
        x = hidden_states
        for layer_idx, block in enumerate(self.blocks):
            kv_cache = kv_caches[layer_idx]
            x = block.forward_decode(x, rot_mats, transformation_mat, kv_cache, current_pos)

        # Final normalization
        x = self.ln_f(x)

        # Language model head
        logits = ttnn.linear(
            x,
            self.lm_head,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return logits


def init_kv_cache(
    mesh_device,
    num_layers: int,
    batch_size: int = 1,
    num_kv_heads: int = 8,
    max_seq_len: int = 8192,
    head_dim: int = 128,
    dtype=ttnn.bfloat8_b,
) -> List[Tuple[ttnn.Tensor, ttnn.Tensor]]:
    """
    Initialize pre-allocated KV cache for all layers.

    With tensor parallelism, each device gets a subset of KV heads.

    Args:
        mesh_device: TTNN mesh device or single device
        num_layers: Number of decoder layers
        batch_size: Batch size
        num_kv_heads: Number of KV heads (total across all devices)
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head
        dtype: Data type for cache

    Returns:
        List of (k_cache, v_cache) tuples per layer
    """
    from loguru import logger

    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"

    if is_mesh_device:
        num_devices = mesh_device.get_num_devices()
        num_kv_heads_per_device = num_kv_heads // num_devices
        # Shard KV cache along heads dimension (dim=1)
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        logger.info(
            f"Initializing KV cache with tensor parallelism: {num_layers} layers, "
            f"{num_kv_heads_per_device} kv_heads per device, max_seq_len={max_seq_len}"
        )
    else:
        num_kv_heads_per_device = num_kv_heads
        mesh_mapper = None
        logger.info(f"Initializing KV cache: {num_layers} layers, max_seq_len={max_seq_len}")

    kv_caches = []
    for layer_idx in range(num_layers):
        # Pre-allocate K cache: [batch, num_kv_heads, max_seq_len, head_dim]
        # With tensor parallelism, this gets sharded to [batch, num_kv_heads_per_device, max_seq_len, head_dim]
        k_cache = ttnn.as_tensor(
            torch.zeros((batch_size, num_kv_heads, max_seq_len, head_dim)),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Pre-allocate V cache: [batch, num_kv_heads, max_seq_len, head_dim]
        v_cache = ttnn.as_tensor(
            torch.zeros((batch_size, num_kv_heads, max_seq_len, head_dim)),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        kv_caches.append((k_cache, v_cache))

    logger.info(f"KV cache initialized: {len(kv_caches)} layers")
    return kv_caches


def init_decode_position(
    mesh_device,
    batch_size: int = 1,
    initial_pos: int = 0,
) -> ttnn.Tensor:
    """
    Initialize decode position tensor.

    Args:
        mesh_device: TTNN mesh device
        batch_size: Batch size
        initial_pos: Initial position value

    Returns:
        Position tensor of shape [batch_size] on device
    """
    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

    pos_tensor = torch.full((batch_size,), initial_pos, dtype=torch.int32)
    return ttnn.from_torch(
        pos_tensor,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )
