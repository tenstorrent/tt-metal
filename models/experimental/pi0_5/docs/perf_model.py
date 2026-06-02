#!/usr/bin/env python3
"""
Analytical perf model for pi0.5 on a 32-chip Blackhole Galaxy.

Compares four deployment options:
  A  — pure TP=32 (no pipeline). All-reduce after every matmul.
  B  — 4 stages x 8 chips, TP=8 within stage, D2D sockets between stages.
  C  — heterogeneous pipeline: 3 vision + 1 vision-embed + 18 prefill + 6 denoise chips.
  C' — same as C but every logical chip becomes a 2-chip submesh with internal TP=2.

These are first-order estimates: peak hardware FLOPS scaled by a single utilization
fraction, no kernel-launch overhead, no MoE / fusion modelling. Use them to compare
options qualitatively, not as a substitute for a measured trace.

Run as a standalone script (no tt-metal dependencies):
    python perf_model.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Constants — workload + hardware
# ---------------------------------------------------------------------------

# SigLIP-27 per-layer FLOPs (single token, single image).
#  attn = 4 * (1152 x 1152) projections (Q, K, V, O)
#  mlp  = fc1 (1152 -> 4304) + fc2 (4304 -> 1152) = 2 * 4304 * 1152
# Multiplier 2 because each MAC is one mul + one add.
SIGLIP_LAYER_FLOPS = 2 * (4 * 1152 * 1152 * 1152 + 2 * 4304 * 1152 * 1152)


def vlm_layer_flops(seq: int) -> int:
    """Per-layer VLM (Gemma-2B) FLOPs at sequence length `seq`.

    QKV(O) at width 2048 with KV-head reduction (kv_dim=256):
        Q [seq, 2048] @ [2048, 2048]
        K, V [seq, 2048] @ [2048, 256] each
        O [seq, 2048] @ [2048, 2048]
        => seq * (2 * 2048*2048 + 2 * 256*2048) MACs.
    MLP at hidden 16384, 3 projections (gate, up, down):
        seq * 3 * 16384 * 2048 MACs.

    Returned with the leading factor of 2 (MAC -> FLOP).
    """
    return 2 * (seq * (2 * 2048 * 2048 + 2 * 256 * 2048) + seq * 3 * 16384 * 2048)  # QKVO  # MLP


def expert_layer_flops(seq: int) -> int:
    """Per-layer Action-Expert FLOPs at suffix length `seq`.

    Width 1024, kv_dim 256 (shared with VLM), MLP hidden 4096.
        attn QKVO at width 1024 with kv_dim=256.
        MLP: gate, up, down at hidden 4096.
    Cross-attention reads VLM KV, but we only count the expert-side projections
    for this layer-local FLOP estimate (KV bytes are accounted for separately).
    """
    return 2 * (seq * (2 * 1024 * 1024 + 2 * 256 * 1024) + seq * 3 * 4096 * 1024)  # QKVO  # MLP


# Sequences ------------------------------------------------------------------
VISION_SEQ_BS = 256 * 2  # 256 patches per image, base + wrist
VLM_PREFILL_SEQ_DEFAULT = 968  # upstream openpi pi05_libero prefix length
VLM_PREFILL_SEQ_FINETUNE = 544  # lerobot finetune variant
EXPERT_SEQ = 50  # suffix tokens
DENOISE_STEPS = 10  # Euler integrator steps
SIGLIP_LAYERS = 27
VLM_LAYERS = 18
EXPERT_LAYERS = 18

# Hardware (Blackhole Galaxy) -----------------------------------------------
BH_FP16_TFLOPS = 745  # advertised per chip
BH_BF8_TFLOPS = 1490  # effective bf8 throughput
BH_ETH_GBPS = 100  # per ethernet link
BH_ETH_LINKS_PER_CHIP = 12  # ~12 D2D links per chip
BH_DRAM_GBPS = 512  # per-chip DRAM
BH_L1_BYTES = 180 * 1024 * 1024
BH_UTIL_FRACTION = 0.50  # 50% of peak FLOPS realized
ETH_LINK_UTIL = 0.50  # realistic fraction of advertised link rate

# Weight bytes (matches PI0_5_GALAXY_DEPLOYMENT_PLAN.md) --------------------
SIGLIP_LAYER_BYTES = 16.2e6
VLM_LAYER_BYTES = 110e6
EXPERT_LAYER_BYTES = 6e6  # weights only, no adaLN Dense (precomputed)
EMBED_TABLE_BYTES = 527e6
SIGLIP_TOTAL_BYTES = 565e6
EXPERT_TOTAL_BYTES = 108e6
SUFFIX_MLP_BYTES = 2.5e6
SIGLIP_PROJ_BYTES = 5e6  # patch conv + pos_embed + final LN + mm_proj

# Activation / KV / scratch (per chip, rough) --------------------------------
ACTIVATION_BF8_PREFILL = 2e6  # [968, 2048] bf8 ~ 2 MB
ACTIVATION_BF8_SUFFIX = 0.05e6  # [50, 1024] bf8 ~ 50 KB
KV_PER_LAYER_BYTES = 0.5e6  # [2, 968, 256] bf8 ~ 500 KB per layer
KV_TOTAL_BYTES = 9e6  # 18 layers * 500 KB
SCRATCH_BYTES = 10e6  # 10 MB realistic scratch budget per chip


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

GiB = 1024**3


def effective_flops_per_chip(bf8: bool = True) -> float:
    """Sustained FLOPS per chip, including utilization derating."""
    peak = (BH_BF8_TFLOPS if bf8 else BH_FP16_TFLOPS) * 1e12
    return peak * BH_UTIL_FRACTION


def compute_time(layer_flops: float, num_chips: int) -> float:
    """Wall-clock time for `layer_flops` distributed evenly across `num_chips`."""
    return layer_flops / (num_chips * effective_flops_per_chip(bf8=True))


def link_bps() -> float:
    """Effective per-link bytes/sec after utilization derating."""
    # GBPS is gigabits/sec by the convention used here, but the deployment plan
    # numbers (e.g. "BH_ETH_GBPS = 100" with payloads in MB and times in ms)
    # only work out if we treat it as gigabytes/sec. We follow the plan.
    return BH_ETH_GBPS * 1e9 * ETH_LINK_UTIL


def all_reduce_time(num_chips: int, payload_bytes: float, num_links: int = 1) -> float:
    """Ring-allreduce latency on `num_chips`.

    For a ring all-reduce, each chip sends ~2 * (N-1)/N * P bytes, where P is the
    full tensor size. We additionally assume `num_links` parallel ring lanes.
    """
    if num_chips <= 1:
        return 0.0
    bytes_per_chip = 2.0 * (num_chips - 1) / num_chips * payload_bytes
    bw = link_bps() * num_links
    return bytes_per_chip / bw


def d2d_transfer_time(payload_bytes: float, num_parallel_links: int = 1) -> float:
    return payload_bytes / (link_bps() * num_parallel_links)


# ---------------------------------------------------------------------------
# 3. Per-option models
# ---------------------------------------------------------------------------


@dataclass
class OptionResult:
    name: str
    description: str
    per_chip_mem_bytes: Dict[str, float] = field(default_factory=dict)
    vision_time_s: float = 0.0
    prefill_time_s: float = 0.0
    denoise_step_time_s: float = 0.0
    inter_stage_transfer_s: float = 0.0
    kv_migration_s: float = 0.0
    all_reduce_overhead_s: float = 0.0
    end_to_end_s: float = 0.0
    stage_utilization: Dict[str, float] = field(default_factory=dict)
    throughput_inf_per_s: float = 0.0
    chips_used: int = 0
    chips_budget: int = 32
    over_budget: bool = False
    notes: List[str] = field(default_factory=list)


# ---- Option A: pure TP=32 -------------------------------------------------


def model_option_a(prefill_seq: int) -> OptionResult:
    """All 32 chips hold 1/32 of every weight. All-reduce after every matmul."""
    r = OptionResult(name="A", description="Pure TP=32 (no pipeline)")
    r.chips_used = 32

    # Per-chip memory: 1/32 of every weight, plus replicated norms/biases.
    weights_total = (
        SIGLIP_TOTAL_BYTES
        + EMBED_TABLE_BYTES
        + VLM_LAYERS * VLM_LAYER_BYTES
        + EXPERT_LAYERS * EXPERT_LAYER_BYTES
        + SUFFIX_MLP_BYTES
    )
    per_chip_weights = weights_total / 32
    replicated = 2e6  # LN, biases, suffix tail
    activation = ACTIVATION_BF8_PREFILL  # peak activation is the prefill one
    kv_local = KV_TOTAL_BYTES / 32  # KV is also sharded across heads
    scratch = SCRATCH_BYTES
    per_chip_total = per_chip_weights + replicated + activation + kv_local + scratch

    r.per_chip_mem_bytes = {
        "weights": per_chip_weights,
        "replicated": replicated,
        "activations": activation,
        "kv": kv_local,
        "scratch": scratch,
        "total": per_chip_total,
    }
    r.over_budget = per_chip_total > BH_L1_BYTES

    # Compute time -- all 32 chips active for every matmul. We model the
    # *useful* compute as if it were perfectly tensor-parallel; the all-reduce
    # cost is added separately.
    siglip_total = SIGLIP_LAYERS * SIGLIP_LAYER_FLOPS * 2  # bs=2 images
    r.vision_time_s = compute_time(siglip_total, 32)

    vlm_per_layer = vlm_layer_flops(prefill_seq)
    vlm_total = VLM_LAYERS * vlm_per_layer
    r.prefill_time_s = compute_time(vlm_total, 32)

    expert_per_layer = expert_layer_flops(EXPERT_SEQ)
    expert_total = EXPERT_LAYERS * expert_per_layer
    r.denoise_step_time_s = compute_time(expert_total, 32)

    # All-reduce overhead: one ring all-reduce per matmul output.
    # SigLIP (per-layer): 4 attn out [256, 1152] + 1 MLP out [256, 1152] (down proj)
    #   = 5 outputs per layer at 256*1152*1B = 295 KB each (bf8 activation).
    # VLM (per-layer): O at [seq, 2048] + down at [seq, 2048] = 2 outputs at 2 MB
    #   per layer (at seq=968 bf8).
    # Expert: O at [50, 1024] + down at [50, 1024] = 2 outputs at 50 KB.
    def ar_per_chip(num_chips: int, num_ar: int, payload: float) -> float:
        return num_ar * all_reduce_time(num_chips, payload)

    siglip_ar = (
        SIGLIP_LAYERS * 2 * (4 * all_reduce_time(32, 256 * 1152) + 1 * all_reduce_time(32, 256 * 1152))
    )  # 2x for bs=2 images, 5 all-reduces per layer
    vlm_ar = VLM_LAYERS * 2 * all_reduce_time(32, prefill_seq * 2048)
    expert_ar = EXPERT_LAYERS * 2 * all_reduce_time(32, EXPERT_SEQ * 1024) * DENOISE_STEPS
    r.all_reduce_overhead_s = siglip_ar + vlm_ar + expert_ar

    # No inter-stage transfer (single replica).
    r.inter_stage_transfer_s = 0.0
    r.kv_migration_s = 0.0

    e2e = r.vision_time_s + r.prefill_time_s + DENOISE_STEPS * r.denoise_step_time_s + r.all_reduce_overhead_s
    r.end_to_end_s = e2e
    r.stage_utilization = {"tp32": 1.0}  # every chip is active every cycle
    r.throughput_inf_per_s = 1.0 / e2e

    r.notes.append("Every chip active every layer; all-reduce traffic dominates at seq=968")
    return r


# ---- Option B: 4 stages x 8 chips/stage ----------------------------------


def model_option_b(prefill_seq: int) -> OptionResult:
    r = OptionResult(name="B", description="4 stages x 8 chips/stage (TP=8 within stage)")
    r.chips_used = 32

    # Per-stage memory.
    stage0_weights = SIGLIP_TOTAL_BYTES + SIGLIP_PROJ_BYTES + EMBED_TABLE_BYTES / 8 * 8
    # Note: embed table goes on stage 0 too in plan -> divide by 8 across stage.
    stage0_per_chip = (SIGLIP_TOTAL_BYTES + EMBED_TABLE_BYTES) / 8

    stage1_per_chip = (9 * VLM_LAYER_BYTES) / 8  # VLM layers 0-8
    stage2_per_chip = (9 * VLM_LAYER_BYTES) / 8  # VLM layers 9-17
    stage3_per_chip = (EXPERT_TOTAL_BYTES + SUFFIX_MLP_BYTES) / 8

    activation = ACTIVATION_BF8_PREFILL
    kv_stage12 = KV_TOTAL_BYTES / 2 / 8  # KV for 9 layers, sharded TP=8
    scratch = SCRATCH_BYTES

    def stage_mem(weight: float, kv: float = 0.0) -> Dict[str, float]:
        total = weight + activation + kv + scratch
        return {
            "weights": weight,
            "activations": activation,
            "kv": kv,
            "scratch": scratch,
            "total": total,
        }

    stage_mems = {
        "stage0_siglip+embed (8 chips)": stage_mem(stage0_per_chip),
        "stage1_vlm_0-8 (8 chips)": stage_mem(stage1_per_chip, kv_stage12),
        "stage2_vlm_9-17 (8 chips)": stage_mem(stage2_per_chip, kv_stage12),
        "stage3_expert+suffix (8 chips)": stage_mem(stage3_per_chip, KV_TOTAL_BYTES / 8),  # expert reads all VLM KV
    }
    r.per_chip_mem_bytes = {k: v["total"] for k, v in stage_mems.items()}
    r.over_budget = any(v["total"] > BH_L1_BYTES for v in stage_mems.values())

    # Compute times per stage --------------------------------------------------
    # SigLIP runs vision (bs=2). VLM stages 1/2 share prefill compute (9 layers each).
    siglip_flops = SIGLIP_LAYERS * SIGLIP_LAYER_FLOPS * 2
    siglip_time = compute_time(siglip_flops, 8)

    vlm_half = 9 * vlm_layer_flops(prefill_seq)
    vlm_half_time = compute_time(vlm_half, 8)

    expert_full = EXPERT_LAYERS * expert_layer_flops(EXPERT_SEQ)
    expert_step_time = compute_time(expert_full, 8)

    r.vision_time_s = siglip_time
    r.prefill_time_s = 2 * vlm_half_time  # serial across stages 1 -> 2
    r.denoise_step_time_s = expert_step_time

    # TP=8 all-reduce within stage (much cheaper than TP=32):
    siglip_ar = SIGLIP_LAYERS * 2 * (4 * all_reduce_time(8, 256 * 1152) + 1 * all_reduce_time(8, 256 * 1152))
    vlm_ar_per_stage = 9 * 2 * all_reduce_time(8, prefill_seq * 2048)
    vlm_ar_total = 2 * vlm_ar_per_stage  # both prefill stages
    expert_ar = EXPERT_LAYERS * 2 * all_reduce_time(8, EXPERT_SEQ * 1024) * DENOISE_STEPS
    r.all_reduce_overhead_s = siglip_ar + vlm_ar_total + expert_ar

    # Inter-stage D2D transfers (per inference):
    # stage0 -> stage1: vision tokens + lang_embed ~ 1.4 MB
    # stage1 -> stage2: VLM activation [968, 2048] bf8 ~ 2 MB
    # stage2 -> stage3: same ~ 2 MB
    # stage3 internal denoise loop: no inter-stage during denoise
    transfer = d2d_transfer_time(1.4e6) + d2d_transfer_time(2e6) + d2d_transfer_time(2e6)
    r.inter_stage_transfer_s = transfer

    # KV migration: stage2 -> stage3 needs VLM KV from both stage1 and stage2.
    # stage1 -> stage3 KV: 9 layers * 500 KB = 4.5 MB
    # stage2 -> stage3 KV: same.
    # The two streams can run in parallel across different ethernet lanes.
    r.kv_migration_s = d2d_transfer_time(4.5e6, num_parallel_links=2)

    e2e = (
        r.vision_time_s
        + r.prefill_time_s
        + r.kv_migration_s
        + DENOISE_STEPS * r.denoise_step_time_s
        + r.inter_stage_transfer_s
        + r.all_reduce_overhead_s
    )
    r.end_to_end_s = e2e

    # Stage utilization = stage_active_time / e2e.
    r.stage_utilization = {
        "stage0_siglip": siglip_time / e2e,
        "stage1_vlm_first_half": vlm_half_time / e2e,
        "stage2_vlm_second_half": vlm_half_time / e2e,
        "stage3_expert (10x denoise)": DENOISE_STEPS * expert_step_time / e2e,
    }
    # Throughput: with deep enough pipelining, the slowest stage sets the rate.
    # Stage 3 owns 10x denoise so it is dominant.
    slowest_stage = max(siglip_time, vlm_half_time, DENOISE_STEPS * expert_step_time)
    r.throughput_inf_per_s = 1.0 / slowest_stage

    r.notes.append(
        "Stage 3 is the throughput bottleneck (10x denoise); stages 0-2 idle "
        "during denoise loop -> ~70% mesh idle for single-shot inference."
    )
    return r


# ---- Option C: heterogeneous pipeline ------------------------------------


def model_option_c(prefill_seq: int) -> OptionResult:
    r = OptionResult(
        name="C",
        description=("Heterogeneous pipeline: 3 vision + 1 vision-embed + 18 prefill + 6 denoise (28 chips used)"),
    )
    r.chips_used = 28  # 4 spare

    # Per-chip memory ----------------------------------------------------------
    # Vision chip (one of 3): 9 SigLIP layers + activations + scratch.
    vision_chip_weights = 9 * SIGLIP_LAYER_BYTES
    vision_embed_chip_weights = SIGLIP_PROJ_BYTES  # patch conv + pos + LN + mm_proj
    prefill_chip_weights = VLM_LAYER_BYTES  # 1 layer per chip
    denoise_chip_weights = 3 * EXPERT_LAYER_BYTES + SUFFIX_MLP_BYTES / 6  # 3 layers / chip
    # KV: a denoise chip needs VLM KV for the 3 layers it pairs with.
    denoise_chip_kv = 3 * KV_PER_LAYER_BYTES
    prefill_chip_kv = KV_PER_LAYER_BYTES

    chips = {
        "vision (3 chips, 9 SigLIP layers each)": (
            vision_chip_weights + ACTIVATION_BF8_SUFFIX * 0 + 1.1e6 + SCRATCH_BYTES
        ),  # 1.1 MB peak activation [256, 4304]
        "vision-embed (1 chip)": vision_embed_chip_weights + 1e6 + SCRATCH_BYTES,
        "prefill (18 chips, 1 VLM layer each)": (
            prefill_chip_weights + ACTIVATION_BF8_PREFILL + prefill_chip_kv + SCRATCH_BYTES
        ),
        "denoise (6 chips, 3 expert layers each)": (
            denoise_chip_weights + ACTIVATION_BF8_SUFFIX + denoise_chip_kv + SCRATCH_BYTES
        ),
    }
    r.per_chip_mem_bytes = chips
    r.over_budget = any(v > BH_L1_BYTES for v in chips.values())

    # Vision: each chip handles 9 layers serially; 3 chips form a pipeline.
    # Per-chip per-image cost: 9 * SIGLIP_LAYER_FLOPS / effective_flops.
    siglip_per_chip_flops = 9 * SIGLIP_LAYER_FLOPS * 2  # bs=2
    siglip_stage_time = compute_time(siglip_per_chip_flops, 1)  # 1 chip per stage
    # With 3-stage pipeline, full vision time = 3 * stage time for cold start +
    # but since we only do 1 inference at a time per request:
    vision_time = 3 * siglip_stage_time

    # Prefill: 18 chips * 1 layer each in a depth-18 pipeline.
    # For a single inference, total prefill time = 18 * per-layer time + (18-1) hops.
    vlm_layer_time = compute_time(vlm_layer_flops(prefill_seq), 1)
    prefill_compute = 18 * vlm_layer_time

    # Denoise: 6 chips * 3 layers each, depth-6 pipeline, runs 10x.
    expert_per_chip = 3 * expert_layer_flops(EXPERT_SEQ)
    expert_chip_time = compute_time(expert_per_chip, 1)
    denoise_step_time = 6 * expert_chip_time  # one Euler step traverses 6 chips
    r.vision_time_s = vision_time
    r.prefill_time_s = prefill_compute
    r.denoise_step_time_s = denoise_step_time

    # No within-stage all-reduce (each chip owns full layer weights).
    r.all_reduce_overhead_s = 0.0

    # Inter-stage D2D activation transfers per inference:
    # vision[i] -> vision[i+1]: 590 KB (3 hops)
    # vision-embed -> prefill[0]: 1.4 MB
    # prefill[i] -> prefill[i+1]: 2 MB (17 hops)
    # prefill final -> denoise: 2 MB (one hop)
    # denoise[i] -> denoise[i+1]: 50 KB (5 hops) * 10 steps
    r.inter_stage_transfer_s = (
        3 * d2d_transfer_time(590e3)
        + d2d_transfer_time(1.4e6)
        + 17 * d2d_transfer_time(2e6)
        + d2d_transfer_time(2e6)
        + DENOISE_STEPS * 5 * d2d_transfer_time(50e3)
    )

    # KV migration: 18 prefill -> 6 denoise, 18 * 500 KB = 9 MB, 18 sources in parallel.
    r.kv_migration_s = d2d_transfer_time(KV_TOTAL_BYTES, num_parallel_links=18)

    e2e = (
        vision_time + prefill_compute + r.kv_migration_s + DENOISE_STEPS * denoise_step_time + r.inter_stage_transfer_s
    )
    r.end_to_end_s = e2e

    r.stage_utilization = {
        "vision (3 chips)": vision_time / e2e,
        "vision-embed (1 chip)": 1e-3,  # spends almost no time
        "prefill (18 chips)": prefill_compute / e2e,
        "denoise (6 chips, 10x)": DENOISE_STEPS * denoise_step_time / e2e,
    }
    # Throughput: the rate-limiting stage in steady state is denoise (10x cost),
    # but in pipelined mode each stage sees one inference at a time. The
    # *throughput* is bounded by max stage time.
    slowest = max(
        siglip_stage_time,
        vlm_layer_time,
        DENOISE_STEPS * denoise_step_time,
    )
    r.throughput_inf_per_s = 1.0 / slowest

    r.notes.append(
        "Denoise stage at 10x cost per inference dominates pipeline throughput " "even though it uses only 6/32 chips."
    )
    return r


# ---- Option C': 1x2 submeshes (TP=2 inside each logical chip) ----------------


def model_option_c_prime(prefill_seq: int) -> OptionResult:
    r = OptionResult(name="C'", description=("Uniform 1x2 submeshes: every logical chip becomes 2-chip TP=2 submesh"))
    # Chip counts: vision 3*2=6, vision-embed 1*2=2, prefill 18*2=36, denoise 6*2=12.
    vision = 3 * 2
    vision_embed = 1 * 2
    prefill = 18 * 2
    denoise = 6 * 2
    r.chips_used = vision + vision_embed + prefill + denoise  # 56
    r.over_budget = r.chips_used > 32

    # Per-submesh per-chip memory: halve weight footprint (TP=2 within submesh).
    vision_chip_weights = 9 * SIGLIP_LAYER_BYTES / 2
    vision_embed_chip_weights = SIGLIP_PROJ_BYTES / 2
    prefill_chip_weights = VLM_LAYER_BYTES / 2
    denoise_chip_weights = 3 * EXPERT_LAYER_BYTES / 2 + SUFFIX_MLP_BYTES / 12

    chips = {
        "vision (6 chips, TP=2)": vision_chip_weights + 1.1e6 + SCRATCH_BYTES,
        "vision-embed (2 chips, TP=2)": vision_embed_chip_weights + 1e6 + SCRATCH_BYTES,
        "prefill (36 chips, TP=2)": (
            prefill_chip_weights + ACTIVATION_BF8_PREFILL + KV_PER_LAYER_BYTES / 2 + SCRATCH_BYTES
        ),
        "denoise (12 chips, TP=2)": (
            denoise_chip_weights + ACTIVATION_BF8_SUFFIX + 3 * KV_PER_LAYER_BYTES / 2 + SCRATCH_BYTES
        ),
    }
    r.per_chip_mem_bytes = chips

    # Compute times: TP=2 halves the layer compute time relative to Option C.
    siglip_chip_flops = 9 * SIGLIP_LAYER_FLOPS * 2  # 9 layers, bs=2
    siglip_stage_time = compute_time(siglip_chip_flops, 2)  # TP=2
    vision_time = 3 * siglip_stage_time

    vlm_layer_time = compute_time(vlm_layer_flops(prefill_seq), 2)
    prefill_compute = 18 * vlm_layer_time

    expert_per_chip_flops = 3 * expert_layer_flops(EXPERT_SEQ)
    expert_chip_time = compute_time(expert_per_chip_flops, 2)
    denoise_step_time = 6 * expert_chip_time

    r.vision_time_s = vision_time
    r.prefill_time_s = prefill_compute
    r.denoise_step_time_s = denoise_step_time

    # Each "logical chip" boundary now also has within-submesh all-reduce.
    # SigLIP TP=2 AR: 27 layers * (4 attn + 1 mlp) = 135 AR * payload 256*1152 (bs=2)
    siglip_ar = SIGLIP_LAYERS * 2 * (4 + 1) * all_reduce_time(2, 256 * 1152)
    vlm_ar = VLM_LAYERS * 2 * all_reduce_time(2, prefill_seq * 2048)
    expert_ar = EXPERT_LAYERS * 2 * all_reduce_time(2, EXPERT_SEQ * 1024) * DENOISE_STEPS
    r.all_reduce_overhead_s = siglip_ar + vlm_ar + expert_ar

    # Inter-stage transfers identical to C in topology, but every hop now exits
    # one submesh and enters another (same hop count). Use same totals as C.
    r.inter_stage_transfer_s = (
        3 * d2d_transfer_time(590e3)
        + d2d_transfer_time(1.4e6)
        + 17 * d2d_transfer_time(2e6)
        + d2d_transfer_time(2e6)
        + DENOISE_STEPS * 5 * d2d_transfer_time(50e3)
    )
    r.kv_migration_s = d2d_transfer_time(KV_TOTAL_BYTES, num_parallel_links=18)

    e2e = (
        vision_time
        + prefill_compute
        + r.kv_migration_s
        + DENOISE_STEPS * denoise_step_time
        + r.inter_stage_transfer_s
        + r.all_reduce_overhead_s
    )
    r.end_to_end_s = e2e

    r.stage_utilization = {
        "vision (6 chips)": vision_time / e2e,
        "vision-embed (2 chips)": 1e-3,
        "prefill (36 chips)": prefill_compute / e2e,
        "denoise (12 chips, 10x)": DENOISE_STEPS * denoise_step_time / e2e,
    }
    slowest = max(
        siglip_stage_time,
        vlm_layer_time,
        DENOISE_STEPS * denoise_step_time,
    )
    r.throughput_inf_per_s = 1.0 / slowest

    overage = r.chips_used - 32
    if overage > 0:
        r.notes.append(
            f"EXCEEDS Galaxy budget: {r.chips_used} chips needed vs 32 available "
            f"(+{overage} chips, ~{r.chips_used / 32:.2f} Galaxies)."
        )
        # Identify which sub-component first pushes past 32.
        running = 0
        order = [
            ("vision (TP=2)", vision),
            ("vision-embed (TP=2)", vision_embed),
            ("prefill (TP=2)", prefill),
            ("denoise (TP=2)", denoise),
        ]
        for name, n in order:
            running += n
            if running > 32:
                excess_within = running - 32
                r.notes.append(
                    f"Budget exhausted partway through {name}: after that group "
                    f"total = {running} (exceeds 32 by {excess_within})."
                )
                break
    r.notes.append(
        "TP=2 halves per-layer compute and weight footprint, but doubles the "
        "chip count -> only viable on >=2 Galaxies."
    )
    return r


# ---------------------------------------------------------------------------
# 4. Output formatting
# ---------------------------------------------------------------------------


def fmt_bytes(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f} GB"
    if x >= 1e6:
        return f"{x / 1e6:.1f} MB"
    if x >= 1e3:
        return f"{x / 1e3:.1f} KB"
    return f"{x:.0f} B"


def fmt_time(x: float) -> str:
    if x >= 1.0:
        return f"{x:.3f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f} ms"
    return f"{x * 1e6:.1f} us"


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(results: List[OptionResult], prefill_seq: int) -> str:
    out = StringIO()
    out.write("=" * 88 + "\n")
    out.write(f"pi0.5 Blackhole Galaxy perf model (prefill_seq = {prefill_seq} tokens)\n")
    out.write("=" * 88 + "\n\n")

    out.write(
        "Hardware assumption: Blackhole, BF8 effective TFLOPS = "
        f"{BH_BF8_TFLOPS}, utilization = {BH_UTIL_FRACTION:.0%}, link bw = "
        f"{BH_ETH_GBPS} GB/s, link util = {ETH_LINK_UTIL:.0%}, "
        f"L1/chip = {fmt_bytes(BH_L1_BYTES)}.\n\n"
    )

    # Per-option memory tables -----------------------------------------------
    for r in results:
        out.write(f"### Option {r.name}: {r.description}\n")
        out.write(f"Chips used: {r.chips_used} / {r.chips_budget}\n")
        if r.over_budget:
            out.write("WARNING: option exceeds chip / memory budget.\n")
        out.write("Per-chip / per-stage memory:\n")
        for k, v in r.per_chip_mem_bytes.items():
            flag = "  <-- OVER 180 MB" if v > BH_L1_BYTES else ""
            out.write(f"  {k:<55s} {fmt_bytes(v):>10s}{flag}\n")
        out.write("\n")

    # Comparison table -------------------------------------------------------
    metrics = [
        ("Chips used", lambda r: f"{r.chips_used}", str),
        ("Vision compute", "vision_time_s", fmt_time),
        ("Prefill compute", "prefill_time_s", fmt_time),
        ("Denoise step", "denoise_step_time_s", fmt_time),
        ("Denoise (x10)", lambda r: DENOISE_STEPS * r.denoise_step_time_s, fmt_time),
        ("All-reduce overhead", "all_reduce_overhead_s", fmt_time),
        ("Inter-stage transfer", "inter_stage_transfer_s", fmt_time),
        ("KV migration", "kv_migration_s", fmt_time),
        ("End-to-end latency", "end_to_end_s", fmt_time),
        ("Throughput (inf/s)", "throughput_inf_per_s", lambda x: f"{x:.2f}"),
    ]
    headers = ["Metric"] + [f"Option {r.name}" for r in results]
    col_widths = [28] + [16] * len(results)

    def write_row(cells: List[str]) -> None:
        out.write("  ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + "\n")

    out.write("Comparison table\n")
    out.write("-" * 88 + "\n")
    write_row(headers)
    out.write("-" * 88 + "\n")
    for name, accessor, fmt in metrics:
        cells = [name]
        for r in results:
            if callable(accessor):
                v = accessor(r)
            else:
                v = getattr(r, accessor)
            if isinstance(v, str):
                cells.append(v)
            else:
                cells.append(fmt(v))
        write_row(cells)
    out.write("-" * 88 + "\n\n")

    # Stage utilization per option ---------------------------------------------
    out.write("Stage utilization (compute-time / end-to-end latency)\n")
    out.write("-" * 88 + "\n")
    for r in results:
        out.write(f"Option {r.name}:\n")
        for stage, frac in r.stage_utilization.items():
            out.write(f"  {stage:<40s}  {fmt_pct(frac):>8s}\n")
        out.write("\n")

    # Per-option summary paragraphs ------------------------------------------
    out.write("Option summaries\n")
    out.write("-" * 88 + "\n")

    def bottleneck(r: OptionResult) -> str:
        candidates = {
            "all-reduce": r.all_reduce_overhead_s,
            "inter-stage transfer": r.inter_stage_transfer_s,
            "vision compute": r.vision_time_s,
            "prefill compute": r.prefill_time_s,
            "denoise loop (10x)": DENOISE_STEPS * r.denoise_step_time_s,
            "kv migration": r.kv_migration_s,
        }
        return max(candidates, key=candidates.get)

    for r in results:
        bn = bottleneck(r)
        out.write(f"Option {r.name} — {r.description}\n")
        out.write(f"  end-to-end latency  : {fmt_time(r.end_to_end_s)}\n")
        out.write(f"  bottleneck          : {bn}\n")
        for note in r.notes:
            out.write(f"  note: {note}\n")
        out.write("\n")

    # Caveats ----------------------------------------------------------------
    out.write("Caveats\n")
    out.write("-" * 88 + "\n")
    out.write(
        "  * First-order analytical estimate. Peak FLOPS derated by a single\n"
        "    utilization fraction ({:.0%}); real kernels may achieve more or less.\n"
        "  * All-reduce latency grows roughly as 2*(N-1)/N * payload / link_bw and\n"
        "    is sized only for ring topology. Tree/recursive-doubling rings can shave\n"
        "    latency further if the framework picks them.\n"
        "  * Pipeline options have a long tail during the 10-step denoise loop;\n"
        "    only the denoise stage is active for ~Nx(stage_time) of that window.\n"
        "  * Memory headroom must include activations (~2 MB peak), per-layer KV\n"
        "    (~250 KB), and ~10 MB scratch in addition to weights.\n"
        "  * Numbers assume bf8 weights everywhere. Mixing in bf16 layers (e.g.\n"
        "    expert attn QKV) will inflate weight bytes proportionally.\n".format(BH_UTIL_FRACTION)
    )
    return out.getvalue()


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    prefill_seq = int(os.environ.get("PI0_PREFILL_SEQ", VLM_PREFILL_SEQ_DEFAULT))
    results = [
        model_option_a(prefill_seq),
        model_option_b(prefill_seq),
        model_option_c(prefill_seq),
        model_option_c_prime(prefill_seq),
    ]
    report = render_report(results, prefill_seq=prefill_seq)
    print(report)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "perf_model_output.txt",
    )
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
