"""
Training Estimator Configuration for Llama 3 8B on Tenstorrent Galaxy.

All time values are in SECONDS.
Memory is computed in bytes internally, displayed in human-readable units.

How to use:
  1. Run Phase 1-4 experiments (see experiment list in estimator.py)
  2. Fill in measured values below (replace 0.0 placeholders)
  3. Run: python estimator.py
  4. For what-if analysis, use dataclasses.replace():
       cfg2 = replace(cfg, mfu_scale=0.8, tp=4)
       result = estimate_step_time(cfg2)
"""

from dataclasses import dataclass


@dataclass
class EstimatorConfig:
    # =========================================================================
    # Model Architecture (Llama 3 8B defaults)
    # =========================================================================
    total_params: float = 1679888384  # 8-block Llama model (from training log)
    embedding_dim: int = 4096  # Hidden dimension (D)
    num_heads: int = 32  # Number of attention heads (H)
    num_kv_heads: int = 8  # Number of KV heads (32=MHA, 8=GQA for original Llama3)
    num_blocks: int = 8  # Transformer blocks (L)
    vocab_size: int = 32000  # Vocabulary size (V)
    seq_len: int = 2048  # Sequence length (S)
    # ffn_dim: int = 14336            # SwiGLU intermediate dimension (D_ff)
    ffn_dim: int = 10922  # <== This is incorrect, but whatever            # SwiGLU intermediate dimension (D_ff)
    dtype_bytes: int = 2  # Weight/activation dtype in bytes (2 = BF16)

    # =========================================================================
    # Measured Single-Device Baselines
    # Experiment: TP=1, DDP=1, single chip, batch_size=1
    # =========================================================================
    fwd_time_s: float = 735.6  # Forward pass time for batch=1 [seconds]
    bwd_time_s: float = 978.3  # Backward pass time for batch=1 [seconds]
    opt_time_s: float = 608.9  # Optimizer step time (AdamW) [seconds]
    #   Should be roughly constant w.r.t. batch size

    # =========================================================================
    # TP (Tensor Parallel) Characterization
    # Measured by comparing TP=1 vs TP=N at batch=1
    # =========================================================================
    tp_perf_perc: float = 0.85  # Fraction of compute that speeds up with TP.
    #   Typically ~0.85 for transformers (linear layers
    #   dominate, but LayerNorm/softmax/embedding don't shard).
    #   Derived from: (time_tp1 - time_tpN) / time_tp1
    #   accounting for CCL overhead.

    tp_ccl_fwd_s: float = (
        0.0  # Total TP CCL time during FORWARD pass, batch=1 [seconds].
    )
    #   This is the aggregate of all all-reduce/all-gather ops
    #   across all blocks in one forward pass.
    #   Measure via profiler or by subtracting pure compute from
    #   total forward time with TP enabled.

    tp_ccl_bwd_s: float = (
        0.0  # Total TP CCL time during BACKWARD pass, batch=1 [seconds].
    )
    #   Same as above but for backward. May differ from forward
    #   because forward does all-reduce, backward may do
    #   all-gather (and vice versa for different layers).

    tp_mem_shard: float = 0.85  # Fraction of model parameters that are TP-sharded.
    #   Linear layer weights are sharded (~85% of params).
    #   LayerNorm, embeddings are typically replicated.
    #   Derived from: 1 - (replicated_params / total_params)

    # =========================================================================
    # DDP (Data Parallel) Characterization
    # Measured with DDP enabled, all-reduce time only
    # =========================================================================
    ddp_ar_time_s: float = 0.0  # Gradient all-reduce time within one Galaxy [seconds].
    #   For ring all-reduce, roughly constant for DDP >= 2
    #   (scales as 2*(n-1)/n * data / bandwidth).
    #   Measure: time between backward done and gradients synced.

    # =========================================================================
    # 2-Tier Multi-Host Characterization
    # Measured with 2-tier setup, varying n_workers
    # =========================================================================
    two_tier_comm_per_worker_s: float = 0.0
    # Time for the aggregator to process ONE worker [seconds].
    #   Includes: recv gradients from worker + elementwise add
    #   + send updated weights back to that worker.
    #   Measure: (total_aggregator_cycle - opt_time - ddp_time)
    #            / n_workers
    #   Or: run with n_workers=1 and subtract opt + ddp time.

    # =========================================================================
    # Training Configuration
    # =========================================================================
    devices_per_galaxy: int = 32  # Fixed: 32 Tenstorrent chips per Galaxy (4x8 mesh)
    tp: int = 1  # Tensor parallel degree (4, 8, or 32)
    local_batch_size: int = 1  # Micro-batch size per chip
    grad_accum_steps: int = 1  # Gradient accumulation micro-batches per optimizer step
    n_workers: int = 1  # Number of worker Galaxies in 2-tier setup
    use_grad_checkpoint: bool = True  # Gradient checkpointing at block boundaries
    use_ddp: bool = True  # Data parallelism within Galaxy
    use_2tier: bool = False  # 2-tier multi-host training

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    use_fp32_optimizer_state: bool = False
    # If True: optimizer states (m, v) use FP32 (4 bytes each).
    # If False: optimizer states use BF16 (2 bytes each).
    #   BF16 may require Kahan summation for numerical accuracy.
    #   FP32 doubles optimizer state memory but avoids precision issues.

    # =========================================================================
    # Scale Factors for What-If Analysis
    #   1.0 = current measured performance
    #   <1.0 = improvement (e.g., 0.8 = 20% faster)
    #   0.0 = fully eliminated/overlapped
    # =========================================================================
    mfu_scale: float = 1.0  # Scale raw compute time (fwd, bwd, opt).
    #   Decrease to model better MFU / optimized ops.
    #   Example: 0.7 means "what if we achieve 30% better MFU"

    tp_ccl_scale: float = 1.0  # Scale TP communication (all-reduce, all-gather).
    #   Decrease to model faster CCL or partial overlap.
    #   0.0 = fully overlapped with compute.

    ddp_ccl_scale: float = 1.0  # Scale DDP gradient all-reduce.
    #   0.0 = fully overlapped with backward compute.

    inter_host_ccl_scale: float = 1.0  # Scale 2-tier inter-host communication.
    #   Decrease to model faster fabric / parallel transfers.

    # =========================================================================
    # Feature Flags (Upcoming Optimizations)
    #   These affect MEMORY only (assume perf neutral, tune via mfu_scale).
    # =========================================================================
    use_flash_attention: bool = False
    # Flash attention: computes attention in tiles without
    # materializing the full [B, H, S, S] score matrix.
    # Saves: B * (H/TP) * S * S * dtype per block.
    # Adds: small logsumexp buffer B * (H/TP) * S * dtype.

    use_fused_ffn: bool = False  # Fused SwiGLU FFN: computes gate/up projection,
    # SiLU activation, element-wise multiply, and optionally
    # down projection in a single fused kernel.
    # Saves: up to 3 * B * S * D_ff/TP * dtype per block
    # (gate, up, silu(gate)*up intermediates not materialized).

    # =========================================================================
    # Derived Properties
    # =========================================================================
    @property
    def ddp(self) -> int:
        """DDP degree derived from Galaxy size and TP degree."""
        return self.devices_per_galaxy // self.tp

    @property
    def opt_dtype_bytes(self) -> int:
        """Optimizer state dtype size in bytes (derived from use_fp32_optimizer_state)."""
        return 4 if self.use_fp32_optimizer_state else 2

    @property
    def params_fraction_per_chip(self) -> float:
        """Fraction of total model params stored on one chip.

        TP-sharded params: each chip stores 1/TP of the parameter.
        Non-TP params (LayerNorm, embeddings): fully replicated on every chip.
        """
        return (1.0 - self.tp_mem_shard) + self.tp_mem_shard / self.tp
