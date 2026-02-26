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

        # Token embedding
        wte = state_dict[f"{state_dict_prefix}.wte.embedding"]
        self.wte = ttnn.as_tensor(
            wte.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wte.embedding"),
        )

        # Visual token embedding (for image tokens)
        new_embedding = state_dict[f"{state_dict_prefix}.wte.new_embedding"]
        self.new_embedding = ttnn.as_tensor(
            new_embedding.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wte.new_embedding"),
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
        """
        # Use TTNN embedding lookup
        embeddings = ttnn.embedding(input_ids, self.wte)
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
    ) -> ttnn.Tensor:
        """
        Decode-mode forward pass (single token at a time).

        Uses paged_update_cache and scaled_dot_product_attention_decode
        for efficient autoregressive generation.

        Args:
            hidden_states: Input embeddings of shape [1, 1, 1, hidden_dim]
            kv_caches: List of (k_cache, v_cache) per layer
            current_pos: Current decode position tensor [batch]

        Returns:
            Logits tensor
        """
        # Get RoPE embeddings for current position (PyTorch-based for decode)
        pos_torch = ttnn.to_torch(current_pos)[0].item()
        cos, sin = self.rotary_emb.get_cos_sin(seq_len=1, start_pos=pos_torch)

        # Process through decoder blocks
        x = hidden_states
        for layer_idx, block in enumerate(self.blocks):
            kv_cache = kv_caches[layer_idx]
            x = block.forward_decode(x, cos, sin, kv_cache, current_pos)

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

    Args:
        mesh_device: TTNN mesh device or single device
        num_layers: Number of decoder layers
        batch_size: Batch size
        num_kv_heads: Number of KV heads
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head
        dtype: Data type for cache

    Returns:
        List of (k_cache, v_cache) tuples per layer
    """
    from loguru import logger

    logger.info(f"Initializing KV cache: {num_layers} layers, max_seq_len={max_seq_len}")

    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

    kv_caches = []
    for layer_idx in range(num_layers):
        # Pre-allocate K cache: [batch, num_kv_heads, max_seq_len, head_dim]
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
