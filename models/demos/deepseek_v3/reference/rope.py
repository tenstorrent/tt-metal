# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek Rotary Position Embeddings (RoPE) modules."""

import math

import torch
from torch import nn

import ttnn


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# YARN helper functions
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Inverse dim formula to find dim based on number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dim range bounds based on rotations"""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """
    YARN (Yet Another RoPE Extension) rotary embeddings with length extrapolation.

    Pseudo-code:
    ```
    def forward(x, seq_len):
        # Input: x is just for device/dtype reference
        # Output: (cos, sin) tensors of shape [seq_len, rope_head_dim=64]

        # 1. Create position indices
        t = arange(seq_len)  # [seq_len]

        # 2. Compute inverse frequencies with YARN scaling
        # - Interpolate between original and scaled frequencies based on beta values
        # - Apply ramp mask to smoothly transition between frequency regions
        dim = qk_rope_head_dim = 64
        freq_extra = 1.0 / (base ** (arange(0, dim, 2) / dim))
        freq_inter = 1.0 / (scaling_factor * base ** (arange(0, dim, 2) / dim))

        # Find correction range based on rotation thresholds
        low, high = yarn_find_correction_range(beta_fast=32, beta_slow=1, ...)
        mask = yarn_linear_ramp_mask(low, high, dim//2)
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask  # [32]

        # 3. Compute frequencies
        freqs = outer(t, inv_freq)  # [seq_len, 32]

        # 4. Apply mscale correction
        mscale_factor = yarn_get_mscale(scaling_factor=40, mscale=1.0)

        # 5. Create cos and sin embeddings
        emb = cat([freqs, freqs], dim=-1)  # [seq_len, 64]
        cos = cos(emb) * mscale_factor  # [seq_len, 64]
        sin = sin(emb) * mscale_factor  # [seq_len, 64]

        return cos, sin
    ```

    Shape Examples:
    - Prefill: seq_len=100 -> cos/sin: [100, 64]
    - Decode:  seq_len=101 -> cos/sin: [101, 64] (with KV cache)

    Notes:
    - qk_rope_head_dim = 64 (from config)
    - scaling_factor = 40, beta_fast = 32, beta_slow = 1
    - YARN allows for better length extrapolation beyond training length
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Pseudo-code:
    ```
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # Input shapes:
        # q, k: [batch, num_heads, seq_len, rope_dim=64]
        # cos, sin: [max_seq_len, rope_dim=64]
        # position_ids: [batch, seq_len]
        # Output: rotated q, k with same shapes

        # 1. Gather cos/sin values for current positions
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, 64]
        sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, 64]

        # 2. Reshape q and k to separate pairs of dimensions
        # [batch, heads, seq, 64] -> [batch, heads, seq, 32, 2]
        q = q.view(b, h, s, d//2, 2).transpose(-1, -2)  # -> [b, h, s, 2, 32]
        k = k.view(b, h, s, d//2, 2).transpose(-1, -2)  # -> [b, h, s, 2, 32]

        # 3. Apply rotation
        # rotate_half swaps and negates: [x1, x2] -> [-x2, x1]
        q_rot = rotate_half(q)  # [-q2, q1]
        k_rot = rotate_half(k)  # [-k2, k1]

        # 4. Apply RoPE formula: x * cos + rotate_half(x) * sin
        q_embed = q * cos + q_rot * sin
        k_embed = k * cos + k_rot * sin

        return q_embed, k_embed
    ```

    Shape Examples:
    - Prefill: q/k [1, 128, 100, 64] -> [1, 128, 100, 64]
    - Decode:  q/k [1, 128, 1, 64] -> [1, 128, 1, 64]

    Notes:
    - Only applies to rope dimensions (last 64 dims of q/k)
    - Implements complex number rotation in real number pairs
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV3YarnRotaryEmbeddingTTNN(nn.Module):
    """
    TTNN implementation of YARN (Yet Another RoPE Extension) rotary embeddings with length extrapolation.

    Pseudo-code:
    ```
    def forward(seq_len, is_decode_mode):
        # Input: seq_len to generate embeddings for
        # Output: (cos, sin) tensors
        #   Prefill: [1, 1, seq_len, qk_rope_head_dim=64]
        #   Decode: [1, 1, BATCH_SIZE, qk_rope_head_dim=64] (with position broadcasting)

        # For decode mode, we typically use ttnn.embedding lookup
        # For prefill, we compute or lookup precomputed values

        if is_decode_mode:
            # Decode: lookup based on position indices
            # position_ids shape: [1, 1, BATCH_SIZE, 1]
            cos = ttnn.embedding(position_ids, cos_cached)  # [1, 1, BATCH_SIZE, 64]
            sin = ttnn.embedding(position_ids, sin_cached)  # [1, 1, BATCH_SIZE, 64]
        else:
            # Prefill: use full precomputed cos/sin for sequence
            cos = cos_cached[:seq_len]  # [1, 1, seq_len, 64]
            sin = sin_cached[:seq_len]  # [1, 1, seq_len, 64]

        return cos, sin
    ```

    Shape Examples:
    - Prefill: seq_len=SEQ_LEN -> cos/sin: [1, 1, SEQ_LEN, 64]
    - Decode:  position_ids [1, 1, BATCH_SIZE, 1] -> cos/sin: [1, 1, BATCH_SIZE, 64]

    Notes:
    - qk_rope_head_dim = 64 (from config)
    - scaling_factor = 40, beta_fast = 32, beta_slow = 1
    - YARN allows for better length extrapolation beyond training length
    - mscale factor is pre-applied to cos/sin values
    - For TTNN, we precompute and store cos/sin tables on device
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Precomputed cos/sin tables will be loaded as ttnn tensors
        self.cos_cached = None  # Will be set during initialization
        self.sin_cached = None  # Will be set during initialization

    def compute_cos_sin_cache(self, seq_len, device, dtype=ttnn.bfloat16):
        """
        Compute cos/sin cache for YARN RoPE
        This would typically be done once during model initialization
        """
        # This is a simplified version - actual implementation would port
        # the YARN frequency computation logic

        # For now, return placeholder
        # Shape: [1, 1, seq_len, rope_head_dim]
        self.cos_cached = None  # Would be computed YARN cos values
        self.sin_cached = None  # Would be computed YARN sin values

    def forward(self, position_ids=None, seq_len=None, memory_config=None):
        """
        Args:
            position_ids: For decode mode - TTNN tensor [1, 1, BATCH_SIZE, 1] with positions
            seq_len: For prefill mode - sequence length to return embeddings for
            memory_config: Optional memory configuration

        Returns:
            cos, sin: TTNN tensors
                Prefill: [1, 1, seq_len, 64]
                Decode: [1, 1, BATCH_SIZE, 64]
        """
        if position_ids is not None:
            # Decode mode - use embedding lookup
            cos = ttnn.embedding(position_ids, self.cos_cached, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
            sin = ttnn.embedding(position_ids, self.sin_cached, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
            # Output shape: [1, 1, BATCH_SIZE, 64]
        else:
            # Prefill mode - slice precomputed values
            # In practice, might use ttnn.slice or precomputed chunks
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
            # Output shape: [1, 1, seq_len, 64]

        return cos, sin


def apply_rotary_pos_emb_ttnn(q, k, cos, sin, is_decode_mode=False):
    """TTNN implementation of Rotary Position Embedding application.

    Pseudo-code:
    ```
    def apply_rotary_pos_emb_ttnn(q, k, cos, sin, is_decode_mode):
        # Input shapes:
        # Prefill:
        #   q, k: [1, num_heads, seq_len, rope_dim=64]
        #   cos, sin: [1, 1, seq_len, rope_dim=64]
        # Decode:
        #   q, k: [1, batch_size, num_heads, rope_dim=64]
        #   cos, sin: [1, 1, batch_size, rope_dim=64]
        # Output: rotated q, k with same shapes

        # Use fused RoPE operation for efficiency
        if is_decode_mode:
            # Decode mode expects [batch, 1, heads, dim]
            q_rot = ttnn.experimental.rotary_embedding_llama(
                q, cos, sin, transformation_matrix, is_decode_mode=True
            )
            k_rot = ttnn.experimental.rotary_embedding_llama(
                k, cos, sin, transformation_matrix, is_decode_mode=True
            )
        else:
            # Prefill mode expects [1, heads, seq, dim]
            q_rot = ttnn.experimental.rotary_embedding_llama(
                q, cos, sin, transformation_matrix, is_decode_mode=False
            )
            k_rot = ttnn.experimental.rotary_embedding_llama(
                k, cos, sin, transformation_matrix, is_decode_mode=False
            )

        return q_rot, k_rot
    ```

    Shape Examples:
    - Prefill: q/k [1, 128, SEQ_LEN, 64] -> [1, 128, SEQ_LEN, 64]
    - Decode:  q/k [1, BATCH_SIZE, 128, 64] -> [1, BATCH_SIZE, 128, 64]

    Notes:
    - Uses fused RoPE kernel for efficiency
    - transformation_matrix pre-computed for rotate_half operation
    - Handles both prefill and decode tensor layouts
    """
    # TTNN has a fused RoPE operation
    # Transformation matrix for rotate_half would be precomputed
    transformation_matrix = None  # Would be precomputed

    q_rot = ttnn.experimental.rotary_embedding_llama(q, cos, sin, transformation_matrix, is_decode_mode=is_decode_mode)

    k_rot = ttnn.experimental.rotary_embedding_llama(k, cos, sin, transformation_matrix, is_decode_mode=is_decode_mode)

    return q_rot, k_rot


def create_transformation_matrix_ttnn(head_dim, device):
    """
    Create transformation matrix for rotate_half operation in RoPE
    This matrix when multiplied performs: [-x2, x1] for consecutive pairs

    Args:
        head_dim: Dimension of attention heads (64 for rope dimensions)
        device: TTNN device

    Returns:
        TTNN tensor of shape [1, 1, head_dim, head_dim]
    """
    # Create transformation matrix that performs rotate_half
    # when applied as matrix multiplication
    # This is precomputed once and reused

    # Placeholder - actual implementation would create the matrix
    # that swaps and negates alternating elements
    transformation_matrix = None

    return transformation_matrix
