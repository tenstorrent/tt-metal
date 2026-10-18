# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of deepseek_v41_flash_d_p in one table (core/model_settings.py). Environment variable =
``DS41_<NAME>``. Nothing else in tt/ or bringup/hooks.py reads the environment."""

from __future__ import annotations

from models.demos.common.bringup.core.model_settings import Setting, Settings

TABLE = {
    # ---- which model bringup/hooks.py:device_model returns
    "DEVICE_MODEL": Setting(
        "device",
        ("device", "hybrid"),
        "device = the all-device model (tt/model.py: the hidden state stays on the device from the embedding to the "
        "final norm, every block through run_block with the validated modules; M.1); hybrid = the CPU reference with "
        "the device steps swapped in, host in / host out around each step (bringup/hooks.py:HybridDeviceModel; "
        "debugging only)",
    ),
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
        "srca",
        ("srca", "rne", "truncate"),
        "fp32 matmul weights the FPU reads as matmul in1 (mHC hc_fn, tt/common.fpu_weight), rounded to nearest-even "
        "at load so what the FPU multiplies is exact and the error unbiased. srca = to the 9 explicit mantissa bits "
        "HiFi4 multiplies of in1 (srcA; in0 / srcB keeps TF32's 10), FIX.4 / DS41-42; rne = to TF32 (the device then "
        "truncates the 10th bit: mixes bias -3.46e-4); truncate = uploaded as fp32 and truncated by the FPU (K-30, "
        "DS41-31). Load time only, no per-chunk cost. Measured FIX.4 attempt 5 (golden inputs, 2x2, L0 attn / L0 ffn / "
        "L1 attn / L1 ffn / L2 attn; Sinkhorn sfpu_fp32): device mixes vs fp64 rne 3.4e-4..3.9e-4 (bias -2.8e-4.."
        "-3.4e-4), srca 7.5e-5..1.5e-4 (bias +1.9e-5..+6.6e-5); step vs the CPU step rne 6.1e-5 / 1.1e-4 / 1.4e-4 / "
        "1.5e-4 / 1.0e-4, srca 1.1e-5 / 4.4e-5 / 6.0e-5 / 6.6e-5 / 5.9e-5 (limits 3.5e-5 / 1.5e-4 / 3.9e-4 / 2.8e-4 / "
        "4.1e-4). Either rounding leaves a fixed per-column error (host alone: worst column bias tf32 9e-5..1.7e-3, "
        "srca 1.9e-4..1.8e-3 of rms) that only an exact fp32 weight (a split matmul, owner: no) removes",
    ),
    "ENGRAM_KV_QDQ": Setting(
        "qdq",
        ("qdq", "skip"),
        "the fp8 act quant of the engram rows before wkv (tt/engram_kv.py): qdq = tt/quant.fp8_qdq (bit-exact with the "
        "reference for any input); skip = none, the identity on real table rows (e4m3 x 2^k: 0 values changed on the "
        "layer-1 golden) except groups with amax < 1e-4 (a perf pick for the owner)",
    ),
    # ---- compressor (tt/compressor.py)
    "COMPRESSOR_X": Setting(
        "round_tf32",
        ("round_tf32", "truncate"),
        "the fp32 matmul inputs of the compressor (attn_norm into [wkv | wgate], the latent into wk; and attn_norm into "
        "the indexer's weights_proj, tt/indexer.py, which the plan fuses into the compressor's projection; matmul in0, which "
        "the FPU reads at TF32 by truncation, K-30): round_tf32 = rounded to TF32 to nearest even first "
        "(tt/quant.round_tf32, DS41-31: an unbiased read, same bits); truncate = the FPU's own truncation",
    ),
    # ---- attention (tt/attention.py)
    "WINDOW_SDPA": Setting(
        "high_precision",
        ("high_precision", "exact", "source"),
        "the window attention: high_precision = ttnn.bringup.sparse_sdpa(high_precision=True, attention_sink=...) "
        "at the dtypes WINDOW_SDPA_SCORES / _PROBS / _OUT pick (defaults: the reference kernel's, FIX.2); exact = tt/window_exact.py "
        "(q and the probabilities as fp32 bf16-piece sums, every in-tile dot one exact product by lane masks on the "
        "keys / values, fp32 SFPU softmax and sink, fp32 output; 32x the matmul work, 64 ms per layer on 2x2; "
        "DS41-19); source = ttnn.transformer.sparse_sdpa (bf16 running state, approximate exp), a perf pick. Owner "
        "decision 2026-10-15 (spec rule 'Accuracy is judged against the noise floor', OV-3): high_precision is the "
        "default and the max-precision value; exact stays selectable for comparison only",
    ),
    "WINDOW_SDPA_OUT": Setting(
        "bf16",
        ("bf16", "fp32"),
        "the output of the high_precision sparse_sdpa: bf16 = the normalized fp32 DEST value rounded to bf16 by the "
        "packer, the reference kernel's output dtype (inference/kernel.py sparse_attn: o bf16; FIX.2); fp32 = "
        "ttnn.bringup.sparse_sdpa(fp32_output=True), untilized exactly (DS41-29), above the reference: comparison only",
    ),
    "WINDOW_SDPA_SCORES": Setting(
        "fp32",
        ("fp32", "bf16"),
        "the scores of the high_precision sparse_sdpa: fp32 = ttnn.bringup.sparse_sdpa(fp32_scores=True), Float32 "
        "scores rounded to TF32 to nearest in DEST (the FPU reads them at TF32; the reference kernel keeps them fp32); "
        "bf16 = the scores rounded to bf16 (the fork's earlier program, below the reference: a perf pick). DS41-31",
    ),
    "WINDOW_SDPA_PROBS": Setting(
        "bf16",
        ("bf16", "fp32"),
        "WINDOW_SDPA_SCORES=fp32 only: the softmax probabilities and q. bf16 = the reference kernel's: bf16 q, "
        "ttnn.bringup.sparse_sdpa(bf16_probs=True): the running max Float32, the row-sum from the TF32 probabilities, "
        "then the probabilities rounded to bf16 (nearest) before P.V (FIX.2); fp32 = DS41-31's variant: q rounded to "
        "TF32 (tt/quant.round_tf32) and kept fp32 through the head -> seq reshard, Float32 (TF32) probabilities, the "
        "running max bf16; above the reference, comparison only (needs WINDOW_SDPA_OUT=fp32)",
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
        ("bfp8", "bfp4", "fp4_decode"),
        "routed experts (tt/experts.py), per local expert extract -> ttnn.linear SwiGLU with the routing weight and "
        "fp8 q(h) before w2 -> insert, on: bfp8 = every expert's w1 | w3 and w2 converted to bfloat8_b ONCE at load, "
        "kept as stored [N, K] and used with transpose_b (exact: every e2m1 x e8m0 value, asserted at load; W^T would "
        "round 48 of the run's 2304 matrices, DS41-35), no per-chunk weight work (FIX.1, OV-6); fp4_decode = the "
        "packed fp4 codes kept on the device and decoded to bf16 W^T for every expert in every chunk (a per-chunk "
        "weight rebuild against agent rule 5; half the weight memory; comparison only). Layer 0, 2x2, chunk 2048, "
        "timed loop tt/perf_experts.py: bfp8 319 ms vs fp4_decode 611 ms per layer per chunk, outputs bit-identical; "
        "load 44 s vs 4 s. The expert loop or the fused op: EXPERTS_KERNEL. bfp4 (owner pick 2026-10-17, P.1: "
        "memory for galaxy_4x4) = the same path with "
        "every expert's w1 | w3 and w2 converted to bfloat4_b ONCE at load ([N, K], transpose_b, groups of 16 along "
        "K inside one e8m0 block): lossy (3 magnitude bits under the group's exponent, round to nearest even; "
        "changes 16.4-16.6 % of the fp4 values per layer, worst matrix ||dW|| / ||W|| 0.118-0.124), 0.5625 B per "
        "value: 0.44 GiB per layer per chip (17.8 GiB at 40 layers on 4x4, 8.9 on 8x4; bfp8 33.6 / 16.8); "
        "stats logged at load. P.1, 2x2, chunk 53248-55296, plain profile: experts 312.7 vs 315.3 ms per layer "
        "(no speed gain, DS41-56); ladder last worst layer L03 0.9954 vs 0.9967. Component and swap tests stay at "
        "bfp8 (max precision)",
    ),
    "EXPERTS_KERNEL": Setting(
        "flat",
        ("loop", "unified", "flat"),
        "how the routed experts run on the dispatched buffer (tt/experts.py; EXPERTS_IMPL bfp8 or bfp4): loop = per "
        "local expert extract -> ttnn.linear w1 | w3 -> SFPU SwiGLU x r -> tt/quant.fp8_qdq -> ttnn.linear w2 -> "
        "insert (24 experts x ~10 ops at the whole-chunk cap; 2522 of 3025 ms device per chunk on 2x2, X.1); "
        "unified = ONE ttnn.bringup.unified_routed_expert_moe program over every local expert (counts on the "
        "device, only the routed rows computed): ClampedSiluGlu, HiFi4 + fp32 dest, high_precision, the fork's "
        "routing_weights (each dispatched row's fp32 weight from its routing pieces, one selector matmul per chunk) "
        "+ fp8_act_quant (q(r h) per 32, ue8m0, bit-exact with tt/quant.fp8_qdq, fork tests/unit) + "
        "weights_transposed (the [N, K] block-float weights as stored: bfp8 exact, DS41-35). flat (P.3, owner "
        "pick to try Marko Staletovic's flat expert; the default stays unified until the owner accepts the pick, "
        "F104) = ONE ttnn.bringup.flat_routed_expert program (spatially pipelined: the DRAM bank readers stream "
        "every local expert's weights once, x relays tilize and multicast the dispatched rows, the down cores write y at each expert's region), clamped_silu, fidelity "
        "MATMUL_FIDELITY, fp32 DEST (gate/up; down: FP32_DEST_ACC), bf16 x / h / y, the fork's routing_weights + "
        "fp8_act_quant (q(r h) per 32 in the gate/up cores' fp32 DEST, bit-exact with tt/quant.fp8_qdq on exact "
        "inputs; a 16-column-unit plan's split h tiles are quantized whole on one core) + weights_nk (the stored "
        "[N, K] matrices laid out tile-transposed: bfp8 exact, asserted at load); the dispatched row's first H values "
        "read in place (hidden=H, no x copy)",
    ),
    "EXPERTS_PRECISION": Setting(
        "full",
        ("full", "glm"),
        "the routed experts' own precision (tt/experts.py; MATMUL_FIDELITY stays the rest of the model's): full = the "
        "expert matmuls at MATMUL_FIDELITY with FP32_DEST_ACC, the flat op's x / h / y as bf16 tiles; glm (P.4, owner "
        "comparison 2026-10-18) = Marko Staletovic's flat expert as GLM-5.3 runs it (glm53_flash_d_p/tt/experts_ag.py): "
        "the expert matmuls at LoFi, the flat op's x and h tiles bfp8 (x_bf16 = h_bf16 = False), y bfp8 TILE "
        "(y_bf16 = False; combine reads bfp8 TILE), fp32 DEST in gate/up and down (down_fp32 = True, GLM_AG_DOWN_FP32's "
        "default); EXPERTS_KERNEL=unified: LoFi only (the fused op has no bfp8 tile options); loop: its w1 | w3 and "
        "w2 linears at LoFi. The routing-weight selector matmuls (no expert matmuls) stay at MATMUL_FIDELITY, and the V4.1 math (routing weight on h, fp8 q(h) before w2) runs in both. Combined with "
        "EXPERTS_IMPL=bfp4 for GLM's weight format. P.4, 2x2, chunk 53248-55296, plain profile, flat: experts 128.9 "
        "(glm + bfp4) vs 149.2 ms (full + bfp8) per chunk, device total 631.8 vs 651.9; ladder last (glm + bfp4) worst "
        "layer L03 0.99524, state min 0.99710 (bfp4 at full, P.1: 0.99535 / 0.99726); layer-0 experts vs CPU rel "
        "0.131, bias +3.4 % (DS41-64)",
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
    "HC_SINKHORN": Setting(
        "sfpu_fp32",
        ("sfpu_fp32", "fpu"),
        "the mHC mixes after the axis-1 exchange (tt/mhc.py): rsqrt scale, split, sigmoids and Sinkhorn. sfpu_fp32 = "
        "ttnn.bringup.mhc_pre_xing(deepseek=True): one program, every step lane-wise fp32 on the SFPU (accurate exp, "
        "Newton reciprocal / rsqrt). fpu = the composed rsqrt chain + ttnn.experimental.deepseek_prefill."
        "mhc_split_sinkhorn, whose FPU steps read fp32 at TF32 (K-20, DS41-40). Measured FIX.4 overseer.1 (golden "
        "inputs, 2x2, 1024 rows per chip, FP32_WEIGHT_ROUND=srca; L0 attn / L0 ffn / L1 attn / L1 ffn / L2 attn): "
        "Sinkhorn alone vs a CPU fp32 Sinkhorn on the device's mixes sfpu_fp32 rel 6.5e-8..7.6e-8, fpu 5.9e-4.."
        "1.2e-3; whole step vs the CPU step sfpu_fp32 1.07e-5 / 4.44e-5 / 6.02e-5 / 6.58e-5 / 5.88e-5 = 0.61 / 0.60 / "
        "0.31 / 0.46 / 0.29 x the noise floor (gate limit / 2; the own-floor column bias left is hc_fn's rounding, "
        "DS41-43, owner waiver), fpu 5.94e-4 / 1.19e-3 / 6.45e-4 / 7.58e-4 / 7.44e-4 = 34 / 16 / 3.3 / 5.4 / 3.7 x; "
        "time (wall, 20 synced calls) rsqrt + Sinkhorn sfpu_fp32 0.030-0.035 ms, fpu 0.448-0.455 ms; whole step "
        "0.409-0.413 vs 0.764-0.805 ms; per layer per chunk (attn_hc + ffn_hc) 0.82 vs 1.56 ms",
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
    "HC_SINKHORN": "sfpu_fp32",  # FIX.4: the official kernel computes the Sinkhorn in fp32
    "ENGRAM_KV_QDQ": "qdq",
    "FP32_WEIGHT_ROUND": "srca",  # FIX.4: what HiFi4 multiplies of in1 (DS41-42)
    "COMPRESSOR_X": "round_tf32",  # unbiased TF32 read of the fp32 matmul inputs (DS41-31)
    "ATTN_IN_MM": "fpu",  # owner 2026-10-15 (OV-3): accuracy against the noise floor, not exactness
    "WINDOW_SDPA": "high_precision",  # owner 2026-10-15 (OV-3); "exact" only for comparison
    "WINDOW_SDPA_OUT": "bf16",  # FIX.2: the reference kernel's dtypes (owner rule: never above the reference)
    "WINDOW_SDPA_SCORES": "fp32",
    "WINDOW_SDPA_PROBS": "bf16",
    "WINDOW_PIECES": 3,
    "ATTN_O_SUM": "gather_add",
    "EXPERTS_SUM": "gather_add",
    "EXPERTS_IMPL": "bfp8",  # exact fp4 values (DS41-10); bfp4 is the owner's lossy memory pick (P.1)
    "EXPERTS_PRECISION": "full",  # P.4: glm (LoFi, bfp8 x / h / y) is judged end to end only
}

settings = Settings("DS41_", TABLE, max_precision=MAX_PRECISION)
