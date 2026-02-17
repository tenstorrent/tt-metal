#!/usr/bin/env python3
"""
Training Step Time and Memory Estimator for Tenstorrent Galaxy.

Estimates training step time (with detailed breakdown) and per-chip peak memory
for distributed training of Llama 3 8B on Tenstorrent Galaxy hardware.

Usage:
    python estimator.py

    For what-if analysis in a notebook or script:

        from config import EstimatorConfig
        from estimator import estimate_step_time, estimate_memory_per_chip
        from dataclasses import replace

        cfg = EstimatorConfig(fwd_time_s=1.5, bwd_time_s=3.0, ...)
        baseline = estimate_step_time(cfg)

        # What if we get 30% better MFU?
        improved = estimate_step_time(replace(cfg, mfu_scale=0.7))
"""

from dataclasses import replace
from config import EstimatorConfig


# =============================================================================
# Step Time Estimation
# =============================================================================


def estimate_step_time(cfg: EstimatorConfig) -> dict:
    """Estimate wall-clock time for one optimizer step with detailed breakdown.

    The model decomposes step time into:
      1. Forward compute (TP-adjusted, per micro-batch, scaled by grad accum)
      2. Forward TP CCL (all-reduce/all-gather during forward)
      3. Backward compute (TP-adjusted)
      4. Backward TP CCL
      5. DDP gradient synchronization (within Galaxy)
      6. 2-tier inter-host communication (if multi-host)
      7. Optimizer step (elementwise, runs per-chip in parallel)

    Key assumptions:
      - Compute scales linearly with batch size (hardware saturated)
      - TP CCL volume scales linearly with batch size (data ∝ batch * seq * dim)
      - DDP all-reduce is ~constant for DDP >= 2 (ring all-reduce)
      - 2-tier aggregator processes workers sequentially (current implementation)
      - No overlap between compute and communication (current implementation)

    Returns:
        Dictionary with total step time, per-component breakdown, and throughput metrics.
    """
    tp = cfg.tp
    ddp = cfg.ddp
    B = cfg.local_batch_size
    ga = cfg.grad_accum_steps

    # Per-chip fraction of total model parameters.
    # This determines optimizer time and communication volume scaling.
    # params_fraction = (1 - tp_mem_shard) + tp_mem_shard / TP
    #   If tp_mem_shard=0.85, TP=8: fraction = 0.15 + 0.85/8 = 0.256
    params_fraction = cfg.params_fraction_per_chip

    # -------------------------------------------------------------------------
    # 1. Forward pass time (one micro-batch)
    # -------------------------------------------------------------------------
    # Compute portion: split into TP-parallelizable and non-parallelizable.
    #   - Non-parallelizable (LayerNorm, softmax, embedding, loss): runs at full cost
    #   - Parallelizable (linear layers): cost divided by TP degree
    #   - MFU_SCALE < 1 models better hardware utilization → faster compute
    #   - Scales linearly with local_batch_size (saturation assumption)
    fwd_compute_per_microbatch = (
        (
            cfg.fwd_time_s * (1.0 - cfg.tp_perf_perc)  # non-parallelizable portion
            + cfg.fwd_time_s * cfg.tp_perf_perc / tp  # parallelizable / TP
        )
        * B
        * cfg.mfu_scale
    )

    # CCL portion: TP all-reduce and all-gather ops during forward pass.
    #   For Megatron-style TP, each transformer block has ~2 all-reduce ops in forward
    #   (one for attention row-parallel, one for MLP row-parallel).
    #   tp_ccl_fwd_s was measured for the FULL model (all blocks) at batch=1.
    #   Scales linearly with batch size because data volume ∝ B * S * D.
    fwd_ccl_per_microbatch = cfg.tp_ccl_fwd_s * B * cfg.tp_ccl_scale

    fwd_per_microbatch = fwd_compute_per_microbatch + fwd_ccl_per_microbatch

    # -------------------------------------------------------------------------
    # 2. Backward pass time (one micro-batch)
    # -------------------------------------------------------------------------
    # Same decomposition as forward. Typically backward ≈ 2× forward.
    # CCL ops in backward are the "reverse" of forward (broadcast ↔ all-reduce),
    # so costs may differ — that's why we measure tp_ccl_bwd separately.
    bwd_compute_per_microbatch = (
        (
            cfg.bwd_time_s * (1.0 - cfg.tp_perf_perc)
            + cfg.bwd_time_s * cfg.tp_perf_perc / tp
        )
        * B
        * cfg.mfu_scale
    )

    bwd_ccl_per_microbatch = cfg.tp_ccl_bwd_s * B * cfg.tp_ccl_scale

    bwd_per_microbatch = bwd_compute_per_microbatch + bwd_ccl_per_microbatch

    # -------------------------------------------------------------------------
    # 3. Worker compute per optimizer step (all micro-batches)
    # -------------------------------------------------------------------------
    # With gradient checkpointing at block boundaries:
    #   During backward, each block's forward is recomputed before backprop.
    #   Total = 2 × forward + 1 × backward per micro-batch.
    # Without checkpointing:
    #   Total = 1 × forward + 1 × backward per micro-batch.
    #
    # Gradient accumulation: repeat forward+backward for `grad_accum` micro-batches,
    # accumulating gradients before the optimizer step.
    checkpoint_multiplier = 2 if cfg.use_grad_checkpoint else 1

    # Total forward time across all micro-batches (including recomputation)
    total_fwd_compute = fwd_compute_per_microbatch * checkpoint_multiplier * ga
    total_fwd_ccl = fwd_ccl_per_microbatch * checkpoint_multiplier * ga

    # Total backward time across all micro-batches
    total_bwd_compute = bwd_compute_per_microbatch * ga
    total_bwd_ccl = bwd_ccl_per_microbatch * ga

    worker_compute = (
        total_fwd_compute + total_fwd_ccl + total_bwd_compute + total_bwd_ccl
    )

    # -------------------------------------------------------------------------
    # 4. DDP gradient synchronization (once per optimizer step)
    # -------------------------------------------------------------------------
    # All-reduce of gradients across DDP replicas within the Galaxy.
    # Happens ONCE after all micro-batches complete (not per micro-batch).
    #
    # Without 2-tier: DDP all-reduce runs on the worker Galaxy.
    # With 2-tier: DDP all-reduce runs on the aggregator Galaxy
    #   (workers send raw gradients without local all-reduce).
    #
    # Either way, the wall-clock contribution to step time is the same:
    # the all-reduce must complete before the optimizer can run.
    #
    # Ring all-reduce time ≈ 2*(n-1)/n * data / bandwidth ≈ constant for n >= 2.
    # DDP_CCL_SCALE = 0 models full overlap with backward compute.
    ddp_sync = cfg.ddp_ar_time_s * (1 if cfg.use_ddp else 0) * cfg.ddp_ccl_scale

    # -------------------------------------------------------------------------
    # 5. 2-tier inter-host communication (once per optimizer step)
    # -------------------------------------------------------------------------
    # In the 2-tier setup:
    #   1. Each worker sends gradients to the aggregator Galaxy
    #   2. Aggregator receives from workers SEQUENTIALLY (current implementation)
    #   3. Aggregator averages gradients, runs DDP + optimizer
    #   4. Aggregator sends updated weights back to workers SEQUENTIALLY
    #
    # two_tier_comm_per_worker_s includes both recv + send for one worker,
    # plus the per-worker elementwise add on the aggregator.
    # Total scales linearly with n_workers (sequential processing).
    #
    # NOTE: If the implementation gains parallel recv/send (e.g., via async fabric),
    # use INTER_HOST_CCL_SCALE < 1 to model the improvement.
    if cfg.use_2tier and cfg.n_workers > 0:
        two_tier_comm = (
            cfg.two_tier_comm_per_worker_s * cfg.n_workers * cfg.inter_host_ccl_scale
        )
    else:
        two_tier_comm = 0.0

    # -------------------------------------------------------------------------
    # 6. Optimizer step (once per optimizer step)
    # -------------------------------------------------------------------------
    # AdamW: elementwise operations on parameters (update m, v, then weights).
    # Runs per-chip in parallel across all 32 devices on the Galaxy.
    # Wall-clock time ∝ per-chip parameter count, NOT total across Galaxy.
    #
    # opt_time_s was measured at TP=1 (full model on one chip).
    # With TP, each chip has fewer parameters → scale by params_fraction.
    optimizer_time = cfg.opt_time_s * params_fraction * cfg.mfu_scale

    # -------------------------------------------------------------------------
    # 7. Total step time
    # -------------------------------------------------------------------------
    total_step_time = (
        worker_compute  # forward + backward (all micro-batches)
        + ddp_sync  # gradient synchronization
        + two_tier_comm  # inter-host communication (if 2-tier)
        + optimizer_time  # parameter update
    )

    # -------------------------------------------------------------------------
    # 8. Throughput metrics
    # -------------------------------------------------------------------------
    # Each worker Galaxy has `ddp` data-parallel groups, each processing
    # `local_batch_size` samples per micro-batch, for `grad_accum` micro-batches.
    n_data_galaxies = cfg.n_workers if cfg.use_2tier else 1
    effective_batch_size = B * ddp * ga * n_data_galaxies
    effective_tokens = effective_batch_size * cfg.seq_len
    tokens_per_sec = effective_tokens / total_step_time if total_step_time > 0 else 0.0

    # Total Galaxies used (workers + 1 aggregator if 2-tier)
    total_galaxies = n_data_galaxies + (1 if cfg.use_2tier else 0)
    tokens_per_sec_per_galaxy = (
        tokens_per_sec / total_galaxies if total_galaxies > 0 else 0.0
    )

    # Total devices used
    total_devices = total_galaxies * cfg.devices_per_galaxy

    return {
        # Total
        "total_step_time_s": total_step_time,
        # Per-component breakdown (summing these equals total_step_time_s)
        "fwd_compute_s": total_fwd_compute,
        "fwd_ccl_s": total_fwd_ccl,
        "bwd_compute_s": total_bwd_compute,
        "bwd_ccl_s": total_bwd_ccl,
        "ddp_sync_s": ddp_sync,
        "two_tier_comm_s": two_tier_comm,
        "optimizer_s": optimizer_time,
        # Worker compute subtotal (for convenience)
        "worker_compute_s": worker_compute,
        # Throughput
        "effective_batch_size": effective_batch_size,
        "effective_tokens_per_step": effective_tokens,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_per_galaxy": tokens_per_sec_per_galaxy,
        "tokens_per_sec_per_device": tokens_per_sec / total_devices
        if total_devices > 0
        else 0.0,
        "total_galaxies": total_galaxies,
        "total_devices": total_devices,
        # Config echo (for report)
        "tp": cfg.tp,
        "ddp": ddp,
        "local_batch_size": B,
        "grad_accum": ga,
        "n_workers": cfg.n_workers,
        "checkpoint_multiplier": checkpoint_multiplier,
    }


# =============================================================================
# Memory Estimation
# =============================================================================


def estimate_memory_per_chip(cfg: EstimatorConfig) -> dict:
    """Estimate per-chip peak DRAM usage for worker and aggregator.

    Memory layout depends on role (worker vs aggregator) and grad accumulation:

    WORKER peak memory (during backward of block i):
      - Model weights (always live)
      - Optimizer states (only if NOT 2-tier; aggregator holds them in 2-tier)
      - Gradients:
          Without grad_accum: only current block's param gradients are live.
            Gradients are consumed by optimizer (or sent to aggregator) per-block
            and freed immediately — no need to keep all blocks' gradients.
          With grad_accum: ALL parameter gradients must persist across micro-batches
            (accumulated in-place), so full model gradients are live.
      - Activation checkpoints (block inputs, if grad checkpointing enabled)
      - Live activations (1 block's forward + backward working set)
      - Embedding / output head activations
      - Misc working buffers

    AGGREGATOR peak memory (no forward/backward, just gradient accumulation + optimizer):
      - Model weights (for broadcasting updated weights)
      - Gradient recv/accumulation buffers (2× weights: running sum + incoming)
      - Optimizer states (m + v for AdamW)
      - Small working buffers

    Flash attention and fused FFN flags reduce activation memory on workers.

    Returns:
        Dictionary with per-component memory breakdown for worker and aggregator.
    """
    B = cfg.local_batch_size
    S = cfg.seq_len
    D = cfg.embedding_dim
    H = cfg.num_heads
    H_kv = cfg.num_kv_heads
    D_head = D // H
    D_ff = cfg.ffn_dim
    L = cfg.num_blocks
    V = cfg.vocab_size
    dt = cfg.dtype_bytes
    tp = cfg.tp

    params_fraction = cfg.params_fraction_per_chip
    uses_grad_accum = cfg.grad_accum_steps > 1

    # =========================================================================
    # 1. Model Weights
    # =========================================================================
    # Each chip stores a fraction of total parameters:
    #   TP-sharded params (e.g., linear weights): 1/TP per chip
    #   Replicated params (e.g., LayerNorm): full copy per chip
    # params_fraction = (1 - tp_mem_shard) + tp_mem_shard / TP
    weights_bytes = cfg.total_params * params_fraction * dt

    # =========================================================================
    # 2. Gradients — depends on whether we use gradient accumulation
    # =========================================================================
    # Per-block parameter count (for gradient sizing when not accumulating).
    # Computed from architecture: all linear projections + RMSNorm per block.
    D_kv = H_kv * D_head
    per_block_linear_params = (
        D * D  # Q projection
        + D * D_kv  # K projection
        + D * D_kv  # V projection
        + D * D  # output projection
        + D * D_ff  # gate projection (SwiGLU)
        + D * D_ff  # up projection (SwiGLU)
        + D_ff * D  # down projection (SwiGLU)
    )
    per_block_norm_params = 2 * D  # 2 × RMSNorm (attention + MLP)

    # Per-block gradients per chip:
    #   Linear params are TP-sharded → each chip holds 1/TP
    #   Norm params are replicated → each chip holds full copy
    per_block_grads_per_chip_bytes = (
        per_block_linear_params / tp + per_block_norm_params
    ) * dt

    # Full-model gradients per chip (same size as weights)
    full_model_grads_per_chip_bytes = weights_bytes

    if uses_grad_accum:
        # WITH gradient accumulation:
        #   Gradients are accumulated across micro-batches in-place.
        #   All parameter gradients must persist until the optimizer step.
        #   Peak gradient memory = full model gradients.
        grads_bytes = full_model_grads_per_chip_bytes
    else:
        # WITHOUT gradient accumulation:
        #   Only one optimizer step per backward pass.
        #   Gradients can be consumed (sent to aggregator / applied by optimizer)
        #   and freed per-block during backward.
        #   Peak gradient memory = one block's parameter gradients.
        grads_bytes = per_block_grads_per_chip_bytes

    # =========================================================================
    # 3. Optimizer States (AdamW: first moment m + second moment v)
    # =========================================================================
    # AdamW maintains 2 state tensors (m, v) per parameter, each with the same
    # element count as the parameter but potentially different dtype:
    #   BF16 (2 bytes): smaller but may need Kahan summation for accuracy
    #   FP32 (4 bytes): more accurate, doubles optimizer state memory
    #
    # In 2-tier: optimizer runs on aggregator → workers do NOT store opt states.
    # Without 2-tier: optimizer runs locally → workers store opt states.
    opt_state_dtype = cfg.opt_dtype_bytes  # 2 (BF16) or 4 (FP32)
    opt_per_chip = 2.0 * cfg.total_params * params_fraction * opt_state_dtype

    opt_states_worker = 0.0
    opt_states_aggregator = 0.0

    if cfg.use_2tier:
        # Workers don't store optimizer state — aggregator handles optimization
        opt_states_aggregator = opt_per_chip
    else:
        # Workers run the optimizer locally
        opt_states_worker = opt_per_chip

    # =========================================================================
    # 4. Activation Checkpoints (block inputs stored during forward)
    # =========================================================================
    # With gradient checkpointing at each block boundary:
    #   Store the INPUT to each transformer block: shape [B, S, D].
    #   These are NOT TP-sharded: they are the fully-reduced output of the
    #   previous block's row-parallel all-reduce.
    #
    # Without checkpointing:
    #   All per-block activations remain live from forward (handled in section 5).
    if cfg.use_grad_checkpoint:
        # L block inputs, each [B, S, D] in dtype
        checkpoint_bytes = L * B * S * D * dt
    else:
        checkpoint_bytes = 0.0

    # =========================================================================
    # 5. Per-Block Activation Memory (peak working set)
    # =========================================================================
    # These are activations that must be STORED for the backward pass of one
    # transformer block. With checkpointing, only 1 block is processed at a time.
    # Without checkpointing, all L blocks' activations are live simultaneously.
    #
    # During backward of block i (with checkpointing):
    #   1. Recompute forward of block i (from checkpoint)
    #   2. Store recomputed activations needed for backward
    #   3. Run backward through block i
    #   4. Free block i's activations (and block i's gradients if no grad accum)
    # Peak memory = activations from step 2 + working memory from step 3.

    # ----- Attention sub-layer activations -----

    # RMSNorm: saves input + normalized output for backward
    #   Both are [B, S, D], NOT TP-sharded (full hidden dim)
    attn_norm_bytes = 2 * B * S * D * dt

    # Q, K, V projections: saved for attention backward
    #   MHA: Q=[B, S, D/TP], K=[B, S, D_kv/TP], V=[B, S, D_kv/TP]
    #   where D_kv = H_kv * D_head (= D for MHA when H_kv = H)
    qkv_bytes = B * S * (D + 2 * D_kv) // tp * dt

    # Attention score matrix: [B, H/TP, S, S]
    #   This is the DOMINANT activation for long sequences.
    #   Standard attention: full S×S matrix per head → O(S²) memory.
    #   Flash attention: computed in tiles, only small tile buffers needed.
    #     Saves the logsumexp statistics instead: [B, H/TP, S] (negligible).
    if cfg.use_flash_attention:
        # Flash attention: only logsumexp stats, no full score matrix
        attn_scores_bytes = B * (H // tp) * S * dt  # logsumexp: [B, H/TP, S]
    else:
        # Standard attention: full [B, H/TP, S, S] score matrix
        attn_scores_bytes = B * (H // tp) * S * S * dt

    # Attention output before output projection all-reduce: [B, S, D/TP]
    attn_out_bytes = B * S * D // tp * dt

    total_attn_activation_bytes = (
        attn_norm_bytes + qkv_bytes + attn_scores_bytes + attn_out_bytes
    )

    # ----- MLP (SwiGLU) sub-layer activations -----

    # RMSNorm: saves input + normalized output
    mlp_norm_bytes = 2 * B * S * D * dt

    # SwiGLU intermediate activations (TP-sharded along D_ff dimension):
    #   gate  = input @ W_gate   → [B, S, D_ff/TP]   (saved for SiLU backward)
    #   up    = input @ W_up     → [B, S, D_ff/TP]   (saved for multiply backward)
    #   silu(gate) * up          → [B, S, D_ff/TP]   (saved for W_down backward)
    #
    # Fused FFN: gate, up, and silu(gate)*up are NOT materialized in memory.
    #   The fused kernel recomputes them during backward.
    #   Only the final hidden state (input to W_down) may need to be stored,
    #   OR it can also be recomputed. We model the optimistic case where
    #   only 1 intermediate tensor is stored (vs 3 without fusion).
    if cfg.use_fused_ffn:
        # Fused: only store 1 intermediate (hidden before W_down)
        mlp_intermediate_bytes = 1 * B * S * D_ff // tp * dt
    else:
        # Standard: store gate + up + silu(gate)*up = 3 tensors
        mlp_intermediate_bytes = 3 * B * S * D_ff // tp * dt

    # MLP output before all-reduce: [B, S, D/TP]
    mlp_out_bytes = B * S * D // tp * dt

    total_mlp_activation_bytes = mlp_norm_bytes + mlp_intermediate_bytes + mlp_out_bytes

    # ----- Total per-block activations -----
    per_block_activation_bytes = (
        total_attn_activation_bytes + total_mlp_activation_bytes
    )

    # ----- Scale by number of live blocks -----
    if cfg.use_grad_checkpoint:
        # With checkpointing: process 1 block at a time.
        # During backward of block i, recompute its forward from checkpoint,
        # then backprop through it. Forward activations are freed as consumed.
        live_blocks = 1
        activation_bytes = per_block_activation_bytes * live_blocks
    else:
        # Without checkpointing: all L blocks' forward activations are live
        # (stored from forward pass for use in backward).
        # Plus ~1 block of backward working memory.
        activation_bytes = per_block_activation_bytes * (L + 1)

    # =========================================================================
    # 6. Embedding and Output Head Activations
    # =========================================================================
    # Embedding lookup output: [B, S, D]
    embedding_act_bytes = B * S * D * dt

    # Output logits: [B, S, V] — can be significant with large vocab
    # This is typically the largest single activation outside the transformer blocks.
    output_logits_bytes = B * S * V * dt

    # =========================================================================
    # 7. Miscellaneous / Working Buffers
    # =========================================================================
    # Temporary buffers for operations (residual connections, masks, etc.)
    # Conservative estimate: ~2× one full hidden-dim activation.
    misc_bytes = 2 * B * S * D * dt

    # =========================================================================
    # 8. Total Worker Memory
    # =========================================================================
    # Peak occurs during backward of a transformer block.
    # Key insight: gradient memory depends on grad_accum mode:
    #   - No grad_accum: grads freed per-block → only 1 block's grads live at peak
    #   - With grad_accum: all grads persist → full model grads live at peak
    total_worker_bytes = (
        weights_bytes
        + grads_bytes  # per-block or full-model depending on grad_accum
        + opt_states_worker  # 0 if 2-tier (optimizer on aggregator)
        + checkpoint_bytes
        + activation_bytes
        + embedding_act_bytes
        + output_logits_bytes
        + misc_bytes
    )

    # =========================================================================
    # 9. Total Aggregator Memory (2-tier only)
    # =========================================================================
    # Aggregator does NOT run forward/backward — no activations needed.
    # It needs:
    #   - Model weights (to hold and broadcast updated weights)
    #   - Gradient recv/accumulation buffers: running sum + incoming tensor = 2× weights
    #   - Optimizer states (m + v for AdamW)
    #   - Small working buffers
    aggregator_grad_buffer_bytes = 2 * full_model_grads_per_chip_bytes
    total_aggregator_bytes = (
        weights_bytes
        + aggregator_grad_buffer_bytes
        + opt_states_aggregator
        + misc_bytes
    )

    device_dram_bytes = 32.0 * 1e9  # 32 GB per chip

    return {
        # Per-component breakdown
        "weights_bytes": weights_bytes,
        "grads_bytes": grads_bytes,
        "grads_mode": "full model (grad_accum)"
        if uses_grad_accum
        else "per-block (freed eagerly)",
        "per_block_grads_bytes": per_block_grads_per_chip_bytes,
        "full_model_grads_bytes": full_model_grads_per_chip_bytes,
        "opt_states_worker_bytes": opt_states_worker,
        "opt_states_aggregator_bytes": opt_states_aggregator,
        "opt_state_dtype": "FP32" if cfg.use_fp32_optimizer_state else "BF16",
        "checkpoint_bytes": checkpoint_bytes,
        "activation_bytes": activation_bytes,
        # Per-block details (for reference)
        "per_block_activation_bytes": per_block_activation_bytes,
        "attn_scores_per_block_bytes": attn_scores_bytes,
        "attn_out_per_block_bytes": attn_out_bytes,
        "mlp_intermediate_per_block_bytes": mlp_intermediate_bytes,
        "attn_norm_per_block_bytes": attn_norm_bytes,
        "qkv_per_block_bytes": qkv_bytes,
        "mlp_norm_per_block_bytes": mlp_norm_bytes,
        "mlp_out_per_block_bytes": mlp_out_bytes,
        # Other activations
        "embedding_act_bytes": embedding_act_bytes,
        "output_logits_bytes": output_logits_bytes,
        "misc_bytes": misc_bytes,
        # Totals
        "total_worker_bytes": total_worker_bytes,
        "total_aggregator_bytes": total_aggregator_bytes,
        "device_dram_bytes": device_dram_bytes,
        "worker_utilization_pct": total_worker_bytes / device_dram_bytes * 100,
        "aggregator_utilization_pct": total_aggregator_bytes / device_dram_bytes * 100,
        # How many more batch elements could fit
        "worker_headroom_bytes": device_dram_bytes - total_worker_bytes,
    }


# =============================================================================
# Formatting Utilities
# =============================================================================


def fmt_bytes(b: float) -> str:
    """Format bytes as a human-readable string."""
    if abs(b) >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if abs(b) >= 1e6:
        return f"{b / 1e6:.2f} MB"
    if abs(b) >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b:.0f} B"


def fmt_time(s: float) -> str:
    """Format seconds as a human-readable string."""
    if s == 0.0:
        return "0"
    if abs(s) >= 1.0:
        return f"{s:.4f} s"
    if abs(s) >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.3f} us"


# =============================================================================
# Report Printing
# =============================================================================


def print_step_time_report(r: dict) -> None:
    """Print a detailed, formatted step time breakdown."""
    total = r["total_step_time_s"]

    def pct(val):
        return f"({val / total * 100:5.1f}%)" if total > 0 else "(  N/A)"

    print()
    print("=" * 72)
    print("  STEP TIME ESTIMATE")
    print("=" * 72)
    print(
        f"  Config: TP={r['tp']}, DDP={r['ddp']}, "
        f"batch={r['local_batch_size']}, grad_accum={r['grad_accum']}, "
        f"workers={r['n_workers']}, "
        f"checkpoint={'ON' if r['checkpoint_multiplier'] == 2 else 'OFF'}"
    )
    print("-" * 72)

    rows = [
        ("Forward compute", r["fwd_compute_s"]),
        ("Forward TP CCL", r["fwd_ccl_s"]),
        ("Backward compute", r["bwd_compute_s"]),
        ("Backward TP CCL", r["bwd_ccl_s"]),
        ("  Worker subtotal", r["worker_compute_s"]),
        ("DDP gradient sync", r["ddp_sync_s"]),
        ("2-tier communication", r["two_tier_comm_s"]),
        ("Optimizer step", r["optimizer_s"]),
    ]

    print(f"  {'Component':<36} {'Time':>12}  {'Share':>8}")
    print(f"  {'-' * 36} {'-' * 12}  {'-' * 8}")
    for name, val in rows:
        marker = ">" if name.startswith("  ") else " "
        print(f"  {marker}{name:<35} {fmt_time(val):>12}  {pct(val)}")

    print(f"  {'-' * 36} {'-' * 12}  {'-' * 8}")
    print(f"  {'TOTAL STEP TIME':<36} {fmt_time(total):>12}")

    print()
    print(f"  {'Effective batch size':<36} {r['effective_batch_size']:>12,}")
    print(f"  {'Tokens per step':<36} {r['effective_tokens_per_step']:>12,}")
    if total > 0:
        print(f"  {'Tokens/sec (total)':<36} {r['tokens_per_sec']:>12,.0f}")
        print(f"  {'Tokens/sec/galaxy':<36} {r['tokens_per_sec_per_galaxy']:>12,.0f}")
        print(f"  {'Tokens/sec/device':<36} {r['tokens_per_sec_per_device']:>12,.0f}")
    print(f"  {'Total galaxies':<36} {r['total_galaxies']:>12}")
    print(f"  {'Total devices':<36} {r['total_devices']:>12}")
    print()


def print_memory_report(r: dict, cfg: EstimatorConfig) -> None:
    """Print a detailed, formatted per-chip memory breakdown."""
    print()
    print("=" * 72)
    print("  PER-CHIP MEMORY ESTIMATE")
    print("=" * 72)
    print(
        f"  Config: TP={cfg.tp}, DDP={cfg.ddp}, batch={cfg.local_batch_size}, "
        f"seq_len={cfg.seq_len}"
    )
    print(
        f"  Flags: grad_ckpt={'ON' if cfg.use_grad_checkpoint else 'OFF'}, "
        f"flash_attn={'ON' if cfg.use_flash_attention else 'OFF'}, "
        f"fused_ffn={'ON' if cfg.use_fused_ffn else 'OFF'}"
    )

    # --- Worker ---
    print()
    print(f"  --- Worker Memory ---")
    print(f"  {'Component':<40} {'Size':>12}")
    print(f"  {'-' * 40} {'-' * 12}")

    grad_label = f"Gradients ({r['grads_mode']})"
    opt_label = f"Optimizer states ({r['opt_state_dtype']})"

    worker_rows = [
        ("Model weights", r["weights_bytes"]),
        (grad_label, r["grads_bytes"]),
        (opt_label, r["opt_states_worker_bytes"]),
        ("Activation checkpoints", r["checkpoint_bytes"]),
        ("Live activations (peak)", r["activation_bytes"]),
        ("Embedding activations", r["embedding_act_bytes"]),
        ("Output logits", r["output_logits_bytes"]),
        ("Misc / working buffers", r["misc_bytes"]),
    ]
    for name, val in worker_rows:
        print(f"  {name:<40} {fmt_bytes(val):>12}")

    print(f"  {'-' * 40} {'-' * 12}")
    print(f"  {'TOTAL WORKER':<40} {fmt_bytes(r['total_worker_bytes']):>12}")
    print(f"  {'Device DRAM':<40} {fmt_bytes(r['device_dram_bytes']):>12}")
    print(f"  {'Utilization':<40} {r['worker_utilization_pct']:>11.1f}%")
    print(f"  {'Headroom':<40} {fmt_bytes(r['worker_headroom_bytes']):>12}")

    # Per-block details
    print()
    print(f"  --- Per-Block Activation Detail ---")
    print(f"  {'Component':<40} {'Size':>12}")
    print(f"  {'-' * 40} {'-' * 12}")
    block_rows = [
        ("Attention RMSNorm (in + out)", r["attn_norm_per_block_bytes"]),
        ("Q, K, V projections", r["qkv_per_block_bytes"]),
        ("Attention scores (S x S)", r["attn_scores_per_block_bytes"]),
        ("Attention output (pre-reduce)", r["attn_out_per_block_bytes"]),
        ("MLP RMSNorm (in + out)", r["mlp_norm_per_block_bytes"]),
        ("MLP intermediates (SwiGLU)", r["mlp_intermediate_per_block_bytes"]),
        ("MLP output (pre-reduce)", r["mlp_out_per_block_bytes"]),
    ]
    for name, val in block_rows:
        suffix = ""
        if "Attention scores" in name and cfg.use_flash_attention:
            suffix = "  (flash attn: logsumexp only)"
        if "MLP intermediates" in name and cfg.use_fused_ffn:
            suffix = "  (fused: 1 tensor vs 3)"
        print(f"  {name:<40} {fmt_bytes(val):>12}{suffix}")
    print(f"  {'-' * 40} {'-' * 12}")
    print(f"  {'Per-block total':<40} {fmt_bytes(r['per_block_activation_bytes']):>12}")

    # --- Aggregator ---
    if cfg.use_2tier:
        print()
        print(f"  --- Aggregator Memory ---")
        print(f"  {'Component':<40} {'Size':>12}")
        print(f"  {'-' * 40} {'-' * 12}")
        agg_opt_label = f"Optimizer states, {r['opt_state_dtype']} (m + v)"
        agg_rows = [
            ("Model weights", r["weights_bytes"]),
            ("Gradient buffers (2x for accum)", 2 * r["weights_bytes"]),
            (agg_opt_label, r["opt_states_aggregator_bytes"]),
            ("Misc / working buffers", r["misc_bytes"]),
        ]
        for name, val in agg_rows:
            print(f"  {name:<40} {fmt_bytes(val):>12}")
        print(f"  {'-' * 40} {'-' * 12}")
        print(
            f"  {'TOTAL AGGREGATOR':<40} {fmt_bytes(r['total_aggregator_bytes']):>12}"
        )
        print(f"  {'Utilization':<40} {r['aggregator_utilization_pct']:>11.1f}%")

    print()


def print_comparison(configs: list[tuple[str, EstimatorConfig]]) -> None:
    """Print side-by-side comparison of step time for multiple configs.

    Args:
        configs: List of (label, EstimatorConfig) tuples.
    """
    results = [(label, estimate_step_time(cfg)) for label, cfg in configs]

    print()
    print("=" * 72)
    print("  COMPARISON")
    print("=" * 72)

    # Header
    header = f"  {'Metric':<30}"
    for label, _ in results:
        header += f" {label:>18}"
    print(header)
    print(f"  {'-' * 30}" + (" " + "-" * 18) * len(results))

    metrics = [
        ("Step time", "total_step_time_s", fmt_time),
        ("  Fwd compute", "fwd_compute_s", fmt_time),
        ("  Fwd CCL", "fwd_ccl_s", fmt_time),
        ("  Bwd compute", "bwd_compute_s", fmt_time),
        ("  Bwd CCL", "bwd_ccl_s", fmt_time),
        ("  DDP sync", "ddp_sync_s", fmt_time),
        ("  2-tier comm", "two_tier_comm_s", fmt_time),
        ("  Optimizer", "optimizer_s", fmt_time),
        ("Tokens/sec", "tokens_per_sec", lambda x: f"{x:,.0f}"),
        ("Tokens/sec/gal", "tokens_per_sec_per_galaxy", lambda x: f"{x:,.0f}"),
        ("Eff batch size", "effective_batch_size", lambda x: f"{x:,}"),
    ]

    for name, key, formatter in metrics:
        row = f"  {name:<30}"
        for _, r in results:
            row += f" {formatter(r[key]):>18}"
        print(row)

    # Speedup relative to first config
    if len(results) > 1:
        base_time = results[0][1]["total_step_time_s"]
        if base_time > 0:
            row = f"  {'Speedup vs baseline':<30}"
            for _, r in results:
                speedup = (
                    base_time / r["total_step_time_s"]
                    if r["total_step_time_s"] > 0
                    else float("inf")
                )
                row += f" {speedup:>17.2f}x"
            print(row)

    print()


# =============================================================================
# Main
# =============================================================================


def main():
    cfg = EstimatorConfig()

    print()
    print("=" * 72)
    print("  Llama 3 8B Training Estimator — Tenstorrent Galaxy")
    print("=" * 72)
    print(
        f"  Model: {cfg.total_params / 1e9:.1f}B params, "
        f"{cfg.num_blocks} blocks, D={cfg.embedding_dim}, "
        f"seq={cfg.seq_len}"
    )
    print(
        f"  Scales: MFU={cfg.mfu_scale}, TP_CCL={cfg.tp_ccl_scale}, "
        f"DDP_CCL={cfg.ddp_ccl_scale}, INTER_HOST={cfg.inter_host_ccl_scale}"
    )

    # --- Step time ---
    step = estimate_step_time(cfg)
    print_step_time_report(step)

    # --- Memory ---
    mem = estimate_memory_per_chip(cfg)
    print_memory_report(mem, cfg)

    # --- Example: what-if comparison ---
    # Uncomment and customize to compare configurations:
    #
    # print_comparison([
    #     ("Baseline",    cfg),
    #     ("MFU 0.8",     replace(cfg, mfu_scale=0.8)),
    #     ("No DDP CCL",  replace(cfg, ddp_ccl_scale=0.0)),
    # ])

    # --- Memory comparison: flash attention + fused FFN ---
    print("--- Memory What-If: Flash Attention + Fused FFN ---")
    for label, override in [
        ("Standard", {}),
        ("Flash Attn", {"use_flash_attention": True}),
        ("Fused FFN", {"use_fused_ffn": True}),
        ("Flash + Fused", {"use_flash_attention": True, "use_fused_ffn": True}),
    ]:
        m = estimate_memory_per_chip(replace(cfg, **override))
        print(
            f"  {label:<20}  worker={fmt_bytes(m['total_worker_bytes']):>10}  "
            f"util={m['worker_utilization_pct']:.1f}%  "
            f"attn_scores/blk={fmt_bytes(m['attn_scores_per_block_bytes']):>10}  "
            f"mlp_inter/blk={fmt_bytes(m['mlp_intermediate_per_block_bytes']):>10}"
        )
    print()


if __name__ == "__main__":
    main()
