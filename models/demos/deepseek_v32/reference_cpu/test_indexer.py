# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-compatible test script for DeepSeek V3.2 Indexer layer.

This script:
1. Creates random dummy inputs for the Indexer
2. Initializes the Indexer with random weights
3. Runs forward pass on CPU
4. Saves outputs (cache states, top-k indices, index scores)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CPU-compatible utilities
from indexer_cpu_utils import act_quant_cpu, fp8_index_cpu, rotate_activation_cpu
from loguru import logger

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

logger.info(f"Random seed set to {SEED}")


@dataclass
class ModelArgs:
    """Model arguments for Indexer from config_671B_v3.2.json"""

    max_batch_size: int = 8
    max_seq_len: int = 16384
    dim: int = 7168
    index_n_heads: int = 64
    index_head_dim: int = 128
    qk_rope_head_dim: int = 64
    index_topk: int = 2048
    q_lora_rank: int = 1536
    # Quantization scale format. None -> linear scale; "ue8m0" -> power-of-two
    # rounded scale. Matches config_671B_v3.2.json ("scale_fmt": "ue8m0").
    scale_fmt: Optional[str] = "ue8m0"
    # RoPE parameters
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1


# ===== Helper functions =====


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args: Model arguments containing positional embedding parameters

    Returns:
        torch.Tensor: Precomputed complex exponential values [seq_len, rope_head_dim//2]
    """
    import math

    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """Computes the correction dimension for rotary positional embedding."""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """Computes the range of correction dimensions."""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        """Computes a linear ramp function for smoothing."""
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Compute base frequencies
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Apply correction for extended sequence lengths (YaRN)
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices and compute outer product
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)  # [seq_len, rope_head_dim//2]

    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """
    Applies rotary positional embeddings.

    Args:
        x: Input tensor
        freqs_cis: Precomputed complex exponential values
        interleaved: Whether to use interleaved format

    Returns:
        Tensor with rotary embeddings applied
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


class Linear(nn.Module):
    """
    Simple linear layer matching reference implementation.

    Note: Weights are allocated as empty tensors and must be initialized
    via initialize_weights() before use. The reference indexer constructs all
    of its projections bias-free (bias=False), so bias defaults to off here.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Allocate empty tensors (matches reference implementation)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LayerNorm(nn.Module):
    """
    LayerNorm matching reference implementation (model.py:LayerNorm).

    Note: γ (weight) is initialized to 1 and β (bias) to 0. Computation is done
    in float32 and cast back to the input dtype, with eps=1e-6 to match the
    reference.
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        # Initialize γ to 1, β to 0 (standard LayerNorm initialization), float32
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (self.weight.shape[0],), self.weight, self.bias, self.eps).type_as(x)


class IndexerCPU(nn.Module):
    """
    CPU-compatible version of the Indexer layer.

    Modified from reference implementation to use CPU-compatible operations.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.scale_fmt = args.scale_fmt

        # Projections (bias-free, matching the reference indexer)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim**-0.5

        # Buffers (using bfloat16 instead of FP8 for CPU compatibility)
        self.register_buffer(
            "k_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // 128, dtype=torch.float32),
            persistent=False,
        )

        logger.info(f"Initialized IndexerCPU with {self.n_heads} heads, head_dim={self.head_dim}")

    def initialize_weights(self, checkpoint_path: Optional[str] = None):
        """
        Initialize model weights either from checkpoint or randomly.

        Args:
            checkpoint_path: Path to checkpoint file. If None, initialize randomly.
        """
        if checkpoint_path is not None:
            logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
            # TODO: Implement checkpoint loading
            # This would use something like:
            # state_dict = torch.load(checkpoint_path)
            # self.load_state_dict(state_dict, strict=False)
            raise NotImplementedError("Checkpoint loading not yet implemented")
        else:
            logger.info("Initializing weights randomly for testing")
            # Initialize Linear layers with normal distribution
            for name, module in self.named_modules():
                if isinstance(module, Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0.0, std=0.02)
                    logger.debug(f"  Initialized {name}: weight {module.weight.shape}")
                elif isinstance(module, LayerNorm):
                    # LayerNorm already initialized correctly (ones for weight, zeros for bias)
                    logger.debug(f"  {name}: LayerNorm already initialized")

            logger.info(f"✓ Random weight initialization complete")

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of Indexer layer.

        Args:
            x: Input features [B, L, dim]
            qr: Query representation [B, L, q_lora_rank]
            start_pos: Starting position in cache
            freqs_cis: Rotary embedding frequencies
            mask: Optional attention mask

        Returns:
            topk_indices: Top-K token indices [B, L, n_heads, topk]
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        logger.info(f"Forward pass: bsz={bsz}, seqlen={seqlen}, start_pos={start_pos}")

        # Query projection
        q = self.wq_b(qr)  # [B, L, n_heads * head_dim]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # [B, L, H, D]

        # Split RoPE and non-RoPE parts
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # Apply RoPE (non-interleaved)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)

        # Concatenate back
        q = torch.cat([q_pe, q_nope], dim=-1)  # [B, L, H, D]

        # Key projection
        k = self.wk(x)  # [B, L, D]
        k = self.k_norm(k)  # [B, L, D]

        # Split and apply RoPE to key
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)  # [B, L, D]

        # Hadamard transform
        logger.info("Applying Hadamard transform to q and k")
        q = rotate_activation_cpu(q)
        k = rotate_activation_cpu(k)

        # FP8 quantization (CPU version using bfloat16)
        logger.info("Quantizing q and k")
        q_fp8, q_scale = act_quant_cpu(q, block_size=128, scale_fmt=self.scale_fmt)
        k_fp8, k_scale = act_quant_cpu(k, block_size=128, scale_fmt=self.scale_fmt)

        # Update cache
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

        logger.info(f"Updated cache: k_cache[{bsz}, {start_pos}:{end_pos}]")

        # Compute weights
        weights = self.weights_proj(x.float())  # [B, L, H]
        weights = weights * (self.n_heads**-0.5)  # Scale by 1/sqrt(n_heads)
        weights = weights.unsqueeze(-1)  # [B, L, H, 1]
        weights = weights * q_scale * self.softmax_scale  # [B, L, H, 1]

        # Compute index scores using FP8 index operation
        logger.info("Computing index scores")
        index_score = fp8_index_cpu(
            q_fp8,  # [B, L, H, D]
            weights.squeeze(-1),  # [B, L, H]
            self.k_cache[:bsz, :end_pos],  # [B, C, D]
            self.k_scale_cache[:bsz, :end_pos].squeeze(-1),  # [B, C]
        )

        logger.info(f"Index scores shape: {index_score.shape}")

        # index_score is summed over heads: [B, L, C] where C == end_pos
        assert index_score.shape == (
            bsz,
            seqlen,
            end_pos,
        ), f"Expected index_score shape {(bsz, seqlen, end_pos)}, got {tuple(index_score.shape)}"

        # Apply mask if provided ([seqlen, end_pos] broadcasts over batch)
        if mask is not None:
            logger.info("Applying mask to index scores")
            index_score = index_score + mask

        # Top-K selection
        topk_k = min(self.index_topk, end_pos)
        logger.info(f"Selecting top-{topk_k} indices")
        topk_indices = index_score.topk(topk_k, dim=-1)[1]  # [B, L, topk]

        assert topk_indices.shape == (
            bsz,
            seqlen,
            topk_k,
        ), f"Expected topk_indices shape {(bsz, seqlen, topk_k)}, got {tuple(topk_indices.shape)}"

        logger.info(f"Top-K indices shape: {topk_indices.shape}")

        return topk_indices, index_score


# ===== Main test script =====


def create_random_inputs(args: ModelArgs, batch_size: int, seq_len: int):
    """
    Create random inputs for testing.

    Args:
        args: Model arguments
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dictionary with input tensors
    """
    logger.info(f"Creating random inputs: batch_size={batch_size}, seq_len={seq_len}")

    # Main input features
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Query representation (LoRA compressed)
    qr = torch.randn(batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16)

    # Starting position
    start_pos = 0

    # Rotary embedding frequencies (precomputed properly)
    # Full precomputed freqs: [max_seq_len, rope_head_dim//2]
    freqs_cis_full = precompute_freqs_cis(args)
    # Extract only the needed portion for this sequence
    freqs_cis = freqs_cis_full[start_pos : start_pos + seq_len]  # [seq_len, rope_head_dim//2]
    # Reshape to expected format: [1, seq_len, 1, rope_head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rope_head_dim//2]

    # Mask (optional, set to None for now)
    mask = None

    inputs = {
        "x": x,
        "qr": qr,
        "start_pos": start_pos,
        "freqs_cis": freqs_cis,
        "mask": mask,
    }

    logger.info("✓ Random inputs created")
    return inputs


def save_outputs(indexer: IndexerCPU, topk_indices: torch.Tensor, index_scores: torch.Tensor, test_params: dict):
    """
    Save Indexer outputs to files.

    Args:
        indexer: Indexer instance (for accessing cache)
        topk_indices: Top-K indices tensor
        index_scores: Index scores tensor
        test_params: Test parameters dictionary
    """
    output_dir = Path("test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving outputs to {output_dir}")

    # Save tensors
    torch.save(topk_indices, output_dir / "topk_indices.pt")
    torch.save(index_scores, output_dir / "index_scores.pt")
    torch.save(indexer.k_cache, output_dir / "k_cache.pt")
    torch.save(indexer.k_scale_cache, output_dir / "k_scale_cache.pt")

    # Save metadata
    metadata = {
        **test_params,
        "topk_indices_shape": list(topk_indices.shape),
        "index_scores_shape": list(index_scores.shape),
        "k_cache_shape": list(indexer.k_cache.shape),
        "k_scale_cache_shape": list(indexer.k_scale_cache.shape),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✓ Outputs saved successfully")
    logger.info(f"  - topk_indices: {topk_indices.shape}")
    logger.info(f"  - index_scores: {index_scores.shape}")
    logger.info(f"  - k_cache: {indexer.k_cache.shape}")
    logger.info(f"  - k_scale_cache: {indexer.k_scale_cache.shape}")


def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("DeepSeek V3.2 Indexer CPU Test")
    logger.info("=" * 80)

    # Test parameters
    batch_size = 2
    seq_len = 8

    # Create model arguments
    args = ModelArgs()
    test_params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dim": args.dim,
        "index_n_heads": args.index_n_heads,
        "index_head_dim": args.index_head_dim,
        "q_lora_rank": args.q_lora_rank,
        "index_topk": args.index_topk,
        "seed": SEED,
    }

    logger.info(f"Test parameters: {test_params}")

    # Create random inputs
    inputs = create_random_inputs(args, batch_size, seq_len)

    # Create Indexer model
    logger.info("Creating IndexerCPU model...")
    indexer = IndexerCPU(args)

    # Initialize weights (random for testing, could load from checkpoint)
    checkpoint_path = None  # Set to path for loading pretrained weights
    indexer.initialize_weights(checkpoint_path)

    # Run forward pass
    logger.info("Running forward pass...")
    try:
        topk_indices, index_scores = indexer.forward(
            x=inputs["x"],
            qr=inputs["qr"],
            start_pos=inputs["start_pos"],
            freqs_cis=inputs["freqs_cis"],
            mask=inputs["mask"],
        )

        logger.info("✓ Forward pass completed successfully")

        # Save outputs
        save_outputs(indexer, topk_indices, index_scores, test_params)

        # Verify outputs
        logger.info("=" * 80)
        logger.info("Output verification:")
        logger.info(f"  topk_indices - min: {topk_indices.min()}, max: {topk_indices.max()}")
        logger.info(f"  index_scores - min: {index_scores.min():.6f}, max: {index_scores.max():.6f}")
        logger.info(f"  index_scores - mean: {index_scores.mean():.6f}, std: {index_scores.std():.6f}")

        # Check for NaN/Inf
        if torch.isnan(index_scores).any():
            logger.error("❌ NaN values detected in index_scores!")
        elif torch.isinf(index_scores).any():
            logger.error("❌ Inf values detected in index_scores!")
        else:
            logger.info("✓ No NaN/Inf values detected")

        logger.info("=" * 80)
        logger.info("✓ Test completed successfully")

    except Exception as e:
        logger.error(f"❌ Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
