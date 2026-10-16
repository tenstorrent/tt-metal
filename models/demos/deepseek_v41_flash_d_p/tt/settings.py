# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of deepseek_v41_flash_d_p in one table (core/model_settings.py). Environment variable =
``DS41_<NAME>``. Nothing else in tt/ or bringup/hooks.py reads the environment."""

from __future__ import annotations

from models.demos.common.bringup.core.model_settings import Setting, Settings

TABLE = {
    # ---- precision (owner rule 7: HiFi4 + fp32 accumulation for every matmul during bring-up)
    "MATMUL_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi3", "HiFi2"),
        "owner rule: HiFi4 for every matmul (mHC projection, ...); lower only by an owner perf pick",
    ),
    "FP32_DEST_ACC": Setting(True, None, "owner rule: fp32 accumulation in every matmul / reduction"),
    "ATTN_IN_MM": Setting(
        "fpu",
        ("fpu", "exact"),
        "the fp8 attention-input projection q(attn_norm) @ [wq_a | wkv] (tt/q_a.py:TtAttnInProj): fpu = one "
        "ttnn.linear (HiFi4, fp32 accumulation), whose in-tile partial sums round to ~TF32 (rel 1.8e-4: ~0.8 % fp8 "
        "flips in the window KV, DS41-16); exact = 32 masked copies in one batched matmul + SFPU fp32 adds "
        "(tt/exact_mm.py; ~8 ms vs 0.4 ms per layer at [1024, 2560] x [2560, 1792]). Owner decision 2026-10-15 "
        "(spec rule 'Accuracy is judged against the noise floor', OV-3): fpu is the default and the max-precision "
        "value; exact stays selectable for comparison only",
    ),
    "FP32_WEIGHT_ROUND": Setting(
        "rne",
        ("rne", "truncate"),
        "fp32 matmul weights the FPU reads (mHC hc_fn, tt/common.fpu_weight): rne = rounded to TF32 to nearest-even at "
        "load, so the FPU's TF32 read is exact and unbiased (host: the weight's share of the mixes error 3.5e-4, all "
        "bias -> 1.5e-5; device attn_hc vs CPU 6.5e-4 -> 6.1e-4, the rest is the in-tile rounding, DS41-16); "
        "truncate = uploaded as fp32 and truncated by the FPU (K-30, DS41-31). Load time only, no per-chunk cost",
    ),
    "ENGRAM_KV_QDQ": Setting(
        "qdq",
        ("qdq", "skip"),
        "the fp8 act quant of the engram rows before wkv (tt/engram_kv.py): qdq = tt/quant.fp8_qdq (bit-exact with the "
        "reference for any input); skip = none, the identity on real table rows (e4m3 x 2^k: 0 values changed on the "
        "layer-1 golden) except groups with amax < 1e-4 (a perf pick for the owner)",
    ),
    # ---- attention (tt/attention.py)
    "WINDOW_SDPA": Setting(
        "high_precision",
        ("high_precision", "exact", "source"),
        "the window attention: high_precision = ttnn.bringup.sparse_sdpa(high_precision=True, attention_sink=...) "
        "(bf16 q, scores, probabilities and output; rel 0.021 vs the CPU step, DS41-18); exact = tt/window_exact.py "
        "(q and the probabilities as fp32 bf16-piece sums, every in-tile dot one exact product by lane masks on the "
        "keys / values, fp32 SFPU softmax and sink, fp32 output; 32x the matmul work, 64 ms per layer on 2x2; "
        "DS41-19); source = ttnn.transformer.sparse_sdpa (bf16 running state, approximate exp), a perf pick. Owner "
        "decision 2026-10-15 (spec rule 'Accuracy is judged against the noise floor', OV-3): high_precision is the "
        "default and the max-precision value; exact stays selectable for comparison only",
    ),
    "WINDOW_SDPA_OUT": Setting(
        "fp32",
        ("fp32", "bf16"),
        "the output of the high_precision sparse_sdpa: fp32 = ttnn.bringup.sparse_sdpa(fp32_output=True), the "
        "normalized fp32 DEST value, untilized exactly (DS41-29); bf16 = rounded to bf16 in the op (the fork's previous "
        "output; DS41-18: S.swa.06 attention vs the CPU step worst row 0.052 > 0.05)",
    ),
    "WINDOW_SDPA_SCORES": Setting(
        "fp32",
        ("fp32", "bf16"),
        "the scores, softmax probabilities and q of the high_precision sparse_sdpa: fp32 = "
        "ttnn.bringup.sparse_sdpa(fp32_scores=True) with q rounded to TF32 (nearest, tt/quant.round_tf32) and kept "
        "fp32 through the head -> seq reshard; the op keeps scores / probabilities in Float32 and rounds every value "
        "its FPU reads to TF32 to nearest (the FPU truncates); attention vs the CPU step rel 0.0143, +2.1 ms per layer "
        "(28.6 vs 26.5 ms, 2x2, chunk 2048); bf16 = q and the scores / probabilities in bf16 (the fork's previous "
        "program, rel 0.0189), a perf pick. DS41-31: S.swa.11 router vs golden 0.9917 (bf16: 0.9887 < 0.99)",
    ),
    "WINDOW_PIECES": Setting(
        3,
        (3, 2),
        "exact window attention: bf16 pieces per fp32 operand (q, probabilities); 3 = every fp32 bit, 2 = 16 bits "
        "(2/3 of the matmul work)",
    ),
    "ATTN_O_SUM": Setting(
        "gather_add",
        ("gather_add", "reduce_scatter"),
        "the wo_b partial-sum exchange over axis 1 ([S/SP, H] fp32): gather_add = all_gather + SFPU ttnn.add, then "
        "the chip's hidden slice (exact fp32, K-30); reduce_scatter = one CCL whose FPU add reads fp32 at about TF32",
    ),
    # ---- routed experts (tt/experts.py)
    "EXPERTS_IMPL": Setting(
        "bfp8",
        ("bfp8", "fp4_decode"),
        "routed experts (tt/experts.py), per local expert extract -> ttnn.linear SwiGLU with the routing weight and "
        "fp8 q(h) before w2 -> insert, on: bfp8 = every expert's w1 | w3 and w2 converted to bfloat8_b ONCE at load, "
        "kept as stored [N, K] and used with transpose_b (exact: every e2m1 x e8m0 value, asserted at load; W^T would "
        "round 48 of the run's 2304 matrices, DS41-35), no per-chunk weight work (FIX.1, OV-6); fp4_decode = the "
        "packed fp4 codes kept on the device and decoded to bf16 W^T for every expert in every chunk (a per-chunk "
        "weight rebuild against agent rule 5; half the weight memory; comparison only). Layer 0, 2x2, chunk 2048, "
        "timed loop tt/perf_experts.py: bfp8 319 ms vs fp4_decode 611 ms per layer per chunk, outputs bit-identical; "
        "load 44 s vs 4 s. The fused unified_routed_expert_moe does not fit (no routing weight and fp8 q(h) between "
        "the activation and w2)",
    ),
    "EXPERTS_DECODE": Setting(
        "gather",
        ("gather", "arith"),
        "EXPERTS_IMPL=fp4_decode only: the e2m1 code -> value step of the expert weight decode: gather = "
        "ttnn.gather from the 16-entry table (4.0 ms per [2304, 5120] matrix); arith = composed float arithmetic "
        "(6.4 ms); both bit-exact",
    ),
    "EXPERTS_SUM": Setting(
        "gather_add",
        ("gather_add", "reduce_scatter"),
        "the dispatch groups' expert partials (and the shared expert's TP partials, tt/shared_expert.py) [S/SP, H] fp32 "
        "summed over axis 1: gather_add = all_gather + SFPU "
        "ttnn.add, then the chip's hidden slice (exact fp32, K-30); reduce_scatter = one CCL whose FPU add reads fp32 "
        "at about TF32",
    ),
    # ---- mHC (tt/mhc.py, tt/residual.py)
    "RESIDUAL_MIX": Setting(
        "fused",
        ("fused", "addcmul"),
        "the mHC residual post * y + comb^T x (tt/residual.py): fused = ttnn.bringup.mhc_post in one program; addcmul "
        "= per output stream multiply + 4 x addcmul; both fp32 on the SFPU",
    ),
    "HC_TP_SUM": Setting(
        "gather_add",
        ("gather_add", "all_reduce"),
        "the [S/SP, 32] mHC partial-row exchange over axis 1: gather_add = all_gather + SFPU ttnn.add (exact fp32, "
        "K-30); all_reduce = one CCL whose FPU add reads fp32 at about TF32",
    ),
}

# Component and swap tests run at these values (framework rule); the shipped defaults above are judged end to end.
MAX_PRECISION = {
    "MATMUL_FIDELITY": "HiFi4",
    "FP32_DEST_ACC": True,
    "HC_TP_SUM": "gather_add",
    "ENGRAM_KV_QDQ": "qdq",
    "FP32_WEIGHT_ROUND": "rne",
    "ATTN_IN_MM": "fpu",  # owner 2026-10-15 (OV-3): accuracy against the noise floor, not exactness
    "WINDOW_SDPA": "high_precision",  # owner 2026-10-15 (OV-3); "exact" only for comparison
    "WINDOW_SDPA_OUT": "fp32",
    "WINDOW_SDPA_SCORES": "fp32",
    "WINDOW_PIECES": 3,
    "ATTN_O_SUM": "gather_add",
    "EXPERTS_SUM": "gather_add",
}

settings = Settings("DS41_", TABLE, max_precision=MAX_PRECISION)
