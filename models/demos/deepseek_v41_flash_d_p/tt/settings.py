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
    # ---- precision (agent rule 7 / F123 / F137, owner: realistic from the start; every test runs the defaults)
    "MATMUL_FIDELITY": Setting(
        "HiFi2",
        ("HiFi4", "HiFi3", "HiFi2"),
        "the fidelity of the matmuls ATTN_FIDELITY and EXPERTS_PRECISION do not set (mHC hc_fn projection, router "
        "gate, engram wkv; tt/common.py:compute_config). "
        "HiFi2 (P.13, owner rule F123: GLM-5.3's production fidelity) is the default; HiFi4 = the bring-up precision "
        "(a comparison value). P.13, 2x2, chunk 53248-55296, plain profile, HiFi2 alone: device total 220.79 -> 219.52 ms a "
        "chunk (engram_kv 6.02 -> 4.87; mHC hc_fn, router: unchanged); the engram wkv matmul at bfp8 [1024, 6144] x "
        "[6144, 12800]: 2.48 ms HiFi2 (rel 0.0041 vs float64, bias -0.28 %, K-25)",
    ),
    "MATMUL_CONFIG": Setting(
        "tuned",
        ("tuned", "auto"),
        "the program config of the model's dense ttnn.linear / ttnn.matmul calls (tt/mm_configs.py, P.23, owner "
        "comparison): auto = none passed, ttnn picks it (the path before P.23); tuned = the config a rule derives "
        "from the call's M, K, N, dtypes, fp32 DEST and the compute grid (swept per call site by "
        "tests/perf/test_matmul_sweep.py), auto where the rule has no entry. Dtypes, compute kernel configs, memory "
        "configs and the op sequence are the same either way",
    ),
    "WO_A_CONFIG": Setting(
        "tuned",
        ("tuned", "auto"),
        "the program config of the batched attention output matmul (attn.wo_a, one weight per group: 2x1024x4096 . "
        "2x4096x1024 bfp8 a chip on 2x2, 8 calls a chunk; tt/mm_configs.py, P.24, owner comparison): auto = none "
        "passed, ttnn picks it (2D multicast, fuse_batch=False, in0_block_w 1: the path before P.24); tuned = the same "
        "2D-multicast family with fuse_batch=False at the widest in0 block whose CBs fit (32 tiles of K at K 4096): "
        "bit-identical to auto, 0.476 -> 0.116 ms a call (640 rows: 0.258 -> 0.068). Only with MATMUL_CONFIG=tuned "
        "(MATMUL_CONFIG=auto passes no config anywhere). Never MatmulMultiCoreReuseProgramConfig (DS41-113)",
    ),
    "FP32_ACC": Setting(
        "needed",
        ("all", "needed"),
        "fp32 accumulation (fp32 DEST) of the model's programs (tt/common.py:fp32_acc; P.13, replaces the boolean "
        "FP32_DEST_ACC): all = every program (the bring-up precision, a comparison value); needed = only the sites that pass "
        "needed=True to tt/common.py:compute_config / attn_compute_config / fp32_acc. Measured P.13 (tt/perf_precision.py, "
        "2x2, per-chip shapes, real weights, bfp8 + HiFi2, random inputs, vs float64): with a 16-bit DEST every dense "
        "projection loses its accuracy, the matmul's reloaded partials biased high (DS41-78): q_a [wq_a | wkv] K 2560 "
        "rel 0.0044 -> 0.034 (bias +2.6 %), wq_b K 1280 0.0041 -> 0.018 (+1.1 %), wo_a K 4096 0.0041 -> 0.056 (+4.5 %), "
        "wo_b K 2048 0.0044 -> 0.026 (+1.8 %), indexer wq_b 0.0041 -> 0.018, indexer weights_proj K 2560 0.0066 -> "
        "0.045, compressor [wkv | wgate] 0.0066 -> 0.045, engram wkv K 6144 0.0041 -> 0.084 (+6.8 %), router gate K "
        "5120 0.0041 -> 0.084 (row 0.12; a top-6 selection); packer L1 accumulation removes the bias but leaves 2-3.5x "
        "the fp32 error (K 1024: 0.0086, K 4096: 0.0144 vs 0.0041). It stays on (needed) at every site: these long "
        "reductions; the stock window sparse_sdpa's softmax statistics (rel vs float64 0.019-0.024 with, 0.026-0.042 "
        "without, worst row 0.064 vs 0.278 at sharp scores); the indexer's ring score (head sum, K-13); the mHC "
        "hc_fn matmul, Sinkhorn, norms and the engram gate sums (fp32 data on the SFPU, DS41-27 / DS41-40); the routed "
        "and shared experts (as GLM-5.3: fp32 DEST gate/up and down). So needed == all today: no program changes (the "
        "matmuls would save 45-55 % without it: e.g. wo_b 0.60 -> 0.28 ms, engram wkv 2.48 -> 1.39 ms a call)",
    ),
    "DENSE_WEIGHTS": Setting(
        "bfp8",
        ("bf16", "bfp8"),
        "the device format of every dense weight the checkpoint stores in fp8 (e4m3 x ue8m0 per 32 x 32; tt/common.py:"
        "dense_dtype): attention wq_a | wkv (tt/q_a.py), wq_b / wo_a / wo_b (tt/attention.py), the indexer's wq_b "
        "(tt/indexer.py), the engram wkv (tt/engram_kv.py); the shared expert (tt/shared_expert.py) is bfp8 in the flat "
        "op's layout since P.12. The compressor's wkv / wgate / wk and the indexer's weights_proj are bf16 in "
        "the checkpoint and stay bf16 (rule 7). bf16 = the dequantized values exactly; bfp8 (P.13, "
        "owner rule F123: the checkpoint's own 8-bit width; DeepSeek multiplies them with fp8 kernels) = bfloat8_b, "
        "which keeps 7 magnitude bits under one exponent per 16 values, so an fp8 value more than 3 binades under its "
        "group's largest loses bits: 10-22 % of the values change, ||dW|| / ||W|| 0.0034-0.0051 per matrix (logged at "
        "load, DENSE_BFP8; worst the engram wkv of layer 14, 21.9 % / 0.0051; [N, K] instead of W^T rounds the same "
        "except the engram wkv, 0.0035 vs 0.0045 at layer 1). P.13, 2x2, chunk 53248-55296, plain profile, bfp8 alone: "
        "device total 220.79 -> 220.90 ms (no section moves: these matmuls are compute-bound at 1024 rows a chip); "
        "memory: about half the dense attention / indexer weights (2x2: ~37 MB a layer a chip, + 74 MB an engram layer)",
    ),
    "GEMM_INPUTS": Setting(
        "bfp8",
        ("bfp8", "bf16"),
        "the device format of every dense GEMM input the reference quantizes to fp8 (tt/common.py:gemm_input). OWNER "
        "DECISION 2026-10-28 (F208, realistic precision = whatever the reference uses; FIX.7): the released code's fp8 "
        "act quant-dequant of every fp8 linear's input (reference/ds41_ref.py _q8 / _fp8_linear, lines ~620-630; "
        "docstring 'native', lines ~89-91) and its fp8 q-dq of the attention output before wo_a (line ~889) are the "
        "device's own 8-bit format: the matmul reads bfloat8_b, stored and computed in it (agent rule 17: never a "
        "rounding the device does not compute in). bfp8 (default) = the attention input projection [wq_a | wkv] "
        "(tt/q_a.py, tt/kv.py), wq_b (the softmax scale's multiply writes bfp8), wo_a (its input heads, one typecast "
        "after the head concat), wo_b (wo_a writes bfp8), the indexer's wq_b (tt/indexer.py) and the engram wkv "
        "(tt/engram_kv.py) read bfloat8_b; the routed experts' x / h were bfp8 already (EXPERTS_PRECISION=glm). The "
        "shared expert's x / h on MOE_IMPL=block (FIX.9, DS41-132): MoEDtypes.shared_activations (tt/"
        "moe_block_adapter.py:block_dtypes) - MOE_HEAD=fused: the engine's x / h tiles bfloat8_b "
        "(models/demos/moe_block/tt/se_engine3.py x_dtype / h_dtype; one typecast of x a layer for the engine and "
        "the x gather, the router keeps bf16 x); MOE_HEAD=ops: x cast for gate / up, h cast for down. Weights, "
        "fidelities (ATTN_FIDELITY HiFi2) and fp32 accumulation unchanged. bf16 = the path before FIX.7 (bf16 matmul "
        "inputs; the shared expert's as before FIX.9)",
    ),
    "ATTN_FIDELITY": Setting(
        "HiFi2",
        ("HiFi4", "HiFi2"),
        "the attention-side matmuls' fidelity (tt/common.py:attn_compute_config; P.7, owner comparison 2026-10-18 of "
        "Marko Staletovic's GLM-5.3 attn_fidelity HiFi2): the attention input projection [wq_a | wkv] (tt/q_a.py), "
        "wq_b, the window / compressed sparse_sdpa programs, wo_a, wo_b (tt/attention.py), the indexer's weights_proj / "
        "wq_b and ring score (tt/indexer.py), the compressor's [wkv | wgate] and wk (tt/compressor.py). mHC, norms, "
        "router, experts, shared expert keep MATMUL_FIDELITY; fp32 DEST per FP32_ACC. HiFi4 = the "
        "bring-up precision; HiFi2 (the default, P.7) = GLM-5.3's (HiFi2 matmuls read ~0.28 % low, K-25)",
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
    # ---- attention (tt/attention.py)
    "ATTN_O_SUM": Setting(
        "gather_add",
        ("gather_add", "reduce_scatter"),
        "the wo_b partial-sum exchange over axis 1 ([S/SP, H] fp32): gather_add = all_gather + SFPU ttnn.add, then "
        "the chip's hidden slice (exact fp32, K-30); reduce_scatter = one CCL whose FPU add reads fp32 at about TF32",
    ),
    "ATTN_GLUE": Setting(
        "on",
        ("on", "off"),
        "P.15 (owner comparison 2026-10-20, glue report items 1, 2, 6): the attention / RoPE glue folded into ops. on = "
        "(1) the rope tail's pair interleave / de-interleave (tt/rope.py: ROW_MAJOR reshapes to 8-byte pages) as an "
        "exact 0/1 [64, 64] bf16 matmul after the bf16 cast (tt/rope.PairPerm, HiFi4: the one exact fidelity), at q, "
        "the attention output, the indexer q, the window kv and the compressor; the window layers keep the table's "
        "stored (interleaved) order like the compressed ones (no table transform); the output's re-interleave dropped "
        "by permuting wo_a's input rows at load; (2) the q head split through ttnn.experimental.nlp_create_q_heads_split "
        "(attention, indexer); (6) the attention output kept in bf16 (no bf16 -> fp32 -> bf16 of the 448 nope "
        "columns), nlp_concat_heads after wo_a, and the reduce-scatter's bf16 result handed to mhc_post where it takes "
        "bf16 (the residual casts only for MHC_FUSED's stats pass, which needs fp32). Exact (bit-identical up to the "
        "summation order of permuted dot products). off = the P.14 path. Not a precision switch. Measured P.15 (2x2, "
        "chunk 53248-55296, plain profile, off -> on): attention 114.36 -> 43.01 ms a chunk, indexer 34.97 -> 25.15, kv "
        "2.11 -> 1.24, compressor 2.38 -> 1.90, attn_residual 2.93 -> 3.29 (the A12 cast moved there), device sum "
        "202.19 -> 119.82 ms; programs 9772 -> 8892 (attention 3296 -> 2648); quick A/B on: chunk wall 201.8 ms (P.14: "
        "213.7; host-bound now), worst layer L03 0.99489 (P.14: 0.99488). Window attention L0 out path vs float64: "
        "rel 0.00736 off and on (HiFi4 + the fork high_precision sdpa 0.00136 both)",
    ),
    "ROPE_IMPL": Setting(
        "on",
        ("on", "off"),
        "P.17 (owner comparison 2026-10-22, precision-debt report J3, fusion report item 1): RoPE at the 34 sites a "
        "chunk (attention q and output inverse, indexer q, window kv, compressor ckv / k). on = ONE stock "
        "ttnn.experimental.deepseek_prefill.rotary_embedding_indexed per site on bf16 heads (tt/rope.rope_stock: Meta "
        "pair order with the rotation trans-mat, so every weight stays in checkpoint order; the 64-dim tail roped in "
        "place by rotary_offset, or the q nope prepended by concat_prefix; bf16 cos / sin [rows, 64] from the load-time "
        "table sliced per start (the op refuses fp32 cos / sin); inverse = the transposed trans-mat; kv_actual_global 0 "
        "on pre-sliced rows; the op's default HiFi4 + fp32 DEST); the attention q's 512^-0.5 folded into its bf16 input "
        "cast, wq_b / indexer wq_b writing bf16, kv_norm / k_norm writing bf16. off = the P.16 path (fp32 SFPU rope "
        "chains on de-interleaved weight rows, PairPerm at HiFi4). Requires ATTN_GLUE=on. Probe ds41_rope_stock/"
        "probe_002 (2x2, [1, 16, 1024, 512] bf16, tail vs float64): rel 0.0037 (one bf16 rounding 0.0017; HiFi2 "
        "0.0052), 0.09-0.10 ms a call",
    ),
    "NORM_IMPL": Setting(
        "on",
        ("on", "off"),
        "P.18 (owner comparison 2026-10-22, precision-debt report J4): the RMS norms (attn_norm, ffn_norm, the final "
        "norm, q_norm, kv_norm, the compressor's norm and k_norm; tt/norm.py). on = ttnn.bringup.rms_norm at the fork's "
        "normal settings (HiFi4, fp32 DEST): a column-split norm = ONE rms_norm(stats_out=True) (the chip's partial "
        "sums of squares) + one all_gather of those [S/SP, 32 B] partials over TP + ONE rms_norm(partial_stats=, "
        "stats_width=H, weight=, dtype=) (fork options of CHANGELOG 6); q_norm / kv_norm read their columns of the "
        "fused [wq_a | wkv] output in place (input_window=, no slice), kv_norm / k_norm write bf16 (dtype=); the "
        "compressor norm is the fork's default program. The statistic is FPU-grade (x^2 and the row sum read fp32 at "
        "TF32, DS41-44): probe ds41_norm_split/probe_002 (2x2, bf16-valued weights, vs float64) fp32 rows: rel 6.1e-4, "
        "median row-norm ratio -3.5e-4 (fp32 out), rel 1.77e-3 (bf16 out; one bf16 rounding 1.66e-3); bf16-valued rows "
        "-1.6e-4. off = the P.17 path (fp32 SFPU compositions: mhc_pre_xing_pack + slice + exact tp_sum + scale, eps, "
        "rsqrt, two multiplies; 8-15 calls a norm)",
    ),
    "GENERIC_OP_PREPARED": Setting(
        "on",
        ("on", "off"),
        "P.1 (galaxy 8x4, owner focus: host time per chunk): on = every fabric_reduce_scatter (tt/ccl.py: the attention "
        "out sum_cols, the q_a tp_sum, compressor / indexer sums; 92 a chunk on the block MoE path) dispatches its "
        "cached descriptor as a prepared program (fork option prepared=True: ttnn.bringup.prepare_program once per "
        "descriptor, ttnn.bringup.run_prepared each call, no runtime argument re-applied); needs GLUE_FOLD=on "
        "(program_hash). off = ttnn.generic_op on the cached descriptor (the path before P.1). Same programs, same "
        "runtime arguments: device time and results unchanged",
    ),
    "GLUE_FOLD": Setting(
        "on",
        ("on", "off"),
        "P.19 (owner comparison 2026-10-22, the rest of the fusion report: items 4, 6, 7, 9 and the host overhead): "
        "on = (a) the mHC exchange as ONE fabric all_gather of the chips' fp32 partial hc rows [S/SP, 32] over TP, "
        "summed in mesh order inside ttnn.bringup.mhc_pre_xing (fork option partials=TP, exact fp32 SFPU adds; was "
        "typecast to bf16 + bf16 reduce-scatter + gather + typecast back, CCL_IMPL=fabric_bf16); (b) "
        "ttnn.bringup.mhc_post reading post / comb from the packed [S/SP, 24] mixes (fork option mixes=, no slices) "
        "and its stats pass taking the bf16 sublayer output as is (no typecast to fp32; the attention's and the MoE's "
        "bf16 reduce-scatter results); (c) fabric_reduce_scatter(program_hash=True): the cached descriptors carry a "
        "custom_program_hash (no per-call hashing of the kernel sources in generic_op); (d) the window path: no pad "
        "mask when the chunk is whole (end - start == chunk), the carry rows [start - 128, start) sliced from the "
        "previous chunk's kv rows when this chunk follows it, the server-order index row and the r2 caches' device "
        "scalars memoized per start; (e) the model's deallocations through the binding (no FastOperation wrapper). "
        "off = the P.18 path. (a) changes the rounding of the hc mixes' exchange (exact fp32 instead of bf16: closer "
        "to the reference's fp32 mixes); the rest are data moves",
    ),
    "STATIC_TABLES": Setting(
        "on",
        ("on", "off"),
        "P.21 (owner comparison 2026-10-22, fusion report item 3): the sparse-attention table of the compressed layers. "
        "on = ONE persistent table per KV source and slot (tt/compressor.AttnTable, allocated with the source's served "
        "compressed cache): [compressed history in natural order (group j at row j), max_seq / r rows | window region "
        "(128 carry + S chunk rows)] bf16 ROW_MAJOR, replicated; the compressor appends the chunk's rows with ONE "
        "ttnn.experimental.slice_write (and a host load of the served cache writes the same rows), each compressed "
        "layer slice_writes its [carry | chunk] window rows into the region (the layers of one source run in order "
        "and reuse it), and the index list is [load-time window ids + the region's base | the top-k ids as they are]: "
        "mesh_partition + concat (+ the compaction gather for starts < 128). No per-chunk gather / concat of the "
        "history, no top-k -> table-row remap; the table's shape is the same in every chunk. off = the P.19 path "
        "(the history gathered over the mesh per chunk, [window | history] concat, the fp32 integer remap). Data "
        "moves only (the same bf16 rows and the same keys per query): not a precision switch. Memory: a replicated "
        "copy of each source's compressed KV per slot (2x2 at 57,344: r2 28 MiB, r1 56 MiB a chip)",
    ),
    "BF16_ACTIVATIONS": Setting(
        "on",
        ("on", "off"),
        "P.20 (owner comparison 2026-10-22, precision-debt report J5): the activations DeepSeek's GPU code keeps in bf16 "
        "(bringup/official_taps.yaml) held in bf16 on the device. on = the four mHC residual streams (embedding, "
        "engram add, attn / ffn residual: block boundary and h_mid, tt/common.stream_dtype) bf16 (the mHC forks read "
        "them exactly into their fp32 DEST and write them with one rounding, fork option: bf16 streams), the "
        "collapses (attn_in / ffn_in) bf16, attn_norm and the final norm bf16, the attention-input projection's "
        "exchanged [wq_a | wkv] output and q_resid (q_norm) bf16, wo_a writing bf16, the compressor's normed latent "
        "bf16 (the rope and wk read it), the engram projection bf16 (engram_kv), with no casts between them. fp32 stays "
        "where the reference is fp32: the hc mixes, the Sinkhorn, every norm's statistic and the engram gates (fp32 "
        "SFPU math on the exact bf16 values), the router; every matmul keeps its fp32 accumulation (FP32_ACC). off = "
        "the P.21 path (fp32 streams, collapses, attn_norm, q_resid, latent; casts to bf16 at each consumer)",
    ),
    # ---- router (tt/router.py)
    "ROUTER_IMPL": Setting(
        "fused",
        ("fused", "ops"),
        "P.22 (owner comparison 2026-10-22: Marko Staletovic's fused MoE routing, fewer host calls): everything after "
        "the gate matmul (sqrt(softplus), the recentred bias added for the selection only, top 6, the gather of the "
        "unbiased scores, (sum + 1e-20) normalization, route scale 1.5). fused = ONE "
        "ttnn.bringup.moe_grouped_topk (the fork of Marko's fused gate; n_groups=1, score_func='sqrtsoftplus', "
        "fp32_selection=True, fp32_activation per ROUTER_SOFTPLUS): the selection keys and the normalization in fp32 on the SFPU (K-23), weights written "
        "bf16 (the reference's own dtype: model.py Gate returns weights.type_as(x); the experts read bf16 weights "
        "either way), ids uint16; the bias as a load-time [max_rows, E] fp32 tensor (the op takes a full-shape bias). "
        "ops = the P.21 chain (softplus, sqrt, add, topk, gather, sum, add, multiply, div: 9 calls after the matmul). "
        "The gate matmul is the same in both. Not a precision switch beyond the weights' bf16 output",
    ),
    "ROUTER_SOFTPLUS": Setting(
        "fp32",
        ("fp32", "bf16"),
        "ROUTER_IMPL=fused only (P.22): the fused gate's sqrt(softplus) polynomials (fork option "
        "ttnn.bringup.moe_grouped_topk fp32_activation). fp32 = the fp32 SFPU softplus (degree 8 on [0, 5] + the exp "
        "tail; what stock ttnn.softplus runs on fp32 and the reference's fp32 softplus): keys vs float64 on the device's "
        "logits max 2.5e-5 like the ops path; bf16 = the source op's program (degree 6, softplus = 0 below -5): "
        "sqrt(softplus) ~8e-4 relative off on [-3, 9], up to 0.08 absolute below, keys max error 8.2e-2, and 0-11 of "
        "2048 rows per layer select other experts than the float64 selection on the same logits (golden s4096 chunk "
        "1, 8 layers; ops path 0-2): below the reference's precision, kept for the owner's comparison only (DS41-105)",
    ),
    # ---- routed experts (tt/experts.py)
    "INDEXER_GLUE": Setting(
        "on",
        ("on", "off"),
        "P.16 (owner comparison 2026-10-20, glue report items 3, 7, 8, 9, 10): the indexer / shared-source glue folded "
        "into ops or computed once. on = (3) layer 20's candidate block max (max over 8 keys) written by "
        "ttnn.bringup.ring_indexer_score_dsa's writer as a second output (fork option block_max_out; was 8 strided "
        "slices + 7 maximum of the scores), the newest block pinned there too; (8) the re-indexer's candidate bias "
        "applied by the score op's writer (fork option block_bias: blocks whose bias is -inf written -inf; was "
        "repeat_interleave x 8 + add of [Sq, T]); (7) the KV source's gathered compressed history, the index list of "
        "a shared top-k and the tiled index keys computed once per (source, chunk) and reused by the reader layers "
        "(tt/sparse_attention.ChunkMemo, freed after their last reader); (9) the RoPE cos / sin rows (and the "
        "indexer's TP split of them, the compressed-row rows) computed once per (table, start), and the router's "
        "weights handed to the weighted experts (no dense scatter, no to_layout + gather); (10) the indexer gates "
        "written bf16 by their linear and reduce-scattered once (the RS block is the chip's query rows: no all_gather, "
        "mesh_partition or casts), the compressor partial written in the CCL's partial dtype. off = the P.15 path. "
        "Data moves and the same roundings (the bf16 partials as q_a's, P.6): not a precision switch. Measured P.16 "
        "(2x2, chunk 53248-55296, plain profile, off (P.15) -> on): indexer 25.15 -> 12.94 ms a chunk, attention 43.01 "
        "-> 40.56, router 1.82 -> 1.47, experts 17.18 -> 16.82, compressor 1.90 -> 1.79, kv 1.24 -> 1.18; device sum "
        "119.82 -> 104.28 ms; programs (4 chips) 8892 -> 7800 (indexer 660 -> 376, attention 2648 -> 2244, kv 928 -> "
        "800, compressor 1048 -> 964, router / experts 480 -> 384 each). Score op alone (layer 20's shapes): 1.68 ms, "
        "+ block max and pin 1.96, + bias 1.69. Quick A/B: chunk wall 197.0 (off) -> 171.9-175.7 ms (on); layers equal "
        "to 1e-5 (worst L03 0.99489 / 0.99490)",
    ),
    "EXPERTS_IMPL": Setting(
        "bfp4",
        ("bfp8", "bfp4"),
        "the routed experts' weight format (tt/experts.py): every expert's w1 / w3 / w2 laid out ONCE at load in the "
        "flat op's bank layout from the stored [N, K] matrices (weights_nk; no per-chunk weight work). bfp8 = "
        "bfloat8_b, exact for every e2m1 x e8m0 value along K (asserted at load; W^T would round 48 of the run's 2304 "
        "matrices, DS41-35); bfp4 (owner pick 2026-10-17, P.1: memory for galaxy_4x4; the DEFAULT since P.10, agent "
        "rule 7 / F117: the checkpoint's own fp4 width) = bfloat4_b, lossy (3 magnitude bits under the group's "
        "exponent, round to nearest even; changes 16.4-16.6 % of the fp4 values per layer, worst matrix ||dW|| / ||W|| "
        "0.118-0.124, logged at load), 0.5625 B per value: 0.44 GiB per layer per chip (17.8 GiB at 40 layers on 4x4, "
        "8.9 on 8x4; bfp8 33.6 / 16.8). P.10 on the shipped path (flat + weighted combine, bf16 sums, glm precision): "
        "the flat plan keeps its cores (48 gate/up, 32 down, 10 free cells, 9 accumulators, one pass) with mt 1 -> 2 "
        "and a 1156 -> 948 KB flat arena; synced experts call, 2x2, chunk 2048, random inputs "
        "(tt/perf_experts.py::test_weights_bfp4): 4.47 -> 3.47 ms at layers 0 and 24; vs float64 with the exact fp4 "
        "weights bfp8 rel 0.0202 / 0.0206, bfp4 0.184 / 0.205 (bias +0.037 / +0.040), bfp4 vs float64 with "
        "bfp4-rounded weights 0.0202 (the device path as exact as bfp8's; the rest is bfloat4_b's rounding of the "
        "weights, DS41-74). FIX.9 deleted the per-chunk fp4_decode value and the loop / unified kernels (EXPERTS_KERNEL)",
    ),
    "EXPERTS_XG_PINNED": Setting(
        "off",
        ("off", "on"),
        "Overseer measurement hack (owner 2026-10-27, until the new flat expert op lands): on = the column gather of "
        "the routed experts' x (weighted_combine.gather_x) writes one persistent DRAM buffer, so "
        "flat_routed_expert_with_program (whose program key bakes xg's address) hits its program cache on every "
        "warm run; off (shipped) = a new gather buffer a call (56 addresses over 5 runs: never warm, F196 census). "
        "Moot since FIX.5 with EXPERTS_PROGRAM_KEY=patch (the shipped default: the op's key holds no address); it "
        "matters only with EXPERTS_PROGRAM_KEY=addresses",
    ),
    "EXPERTS_PROGRAM_KEY": Setting(
        "patch",
        ("patch", "addresses"),
        "FIX.5 (owner order 2026-10-27): how flat_routed_expert_with_program keys its program cache. patch (shipped) = "
        "the op's patch_addresses: no buffer address in the key, every one (x, y, arena, counts, words, weights, the "
        "combine's arena CBs; the combine's runtime args when its extra_addrs change) patched on a hit, so the 40 "
        "layers share one program per pass and a new gather buffer (xg) never rebuilds it; addresses (the op before) "
        "= every baked address in the key: a new xg address a call is a miss (+20..40 programs a 40-layer chunk, ~0.5 "
        "s of host each, never warm under F196). Bit-identical outputs (tt/perf_flat_program_key.py). MOE_IMPL=flat "
        "only: the block's exchange calls the op address-keyed (it pins its buffers instead)",
    ),
    "EXPERTS_LAYOUT": Setting(
        "marko",
        ("legacy", "marko"),
        "the flat expert op's work layout (owner 2026-10-27: Marko's by default after the 2x2 A/B, device 65.74 vs 68.10 ms / chunk on our layout, experts 15.88 vs 17.85, L.last within limits): legacy = the plan before "
        "his DRISC-only / even-split defaults (the cached weights and numbers so far); marko = his defaults plus what his "
        "MoE block passes per model (rows_hint = expected rows an expert, one x helper; moe_block.layout_of). marko "
        "re-lays out the cached weights (new cache files)",
    ),
    "EXPERTS_WEIGHT_CACHE": Setting(
        "on",
        ("on", "off"),
        "FIX.4 (overseer): the routed experts' laid-out weights (the flat op's bank layout, EXPERTS_IMPL's block "
        "format) cached on disk by FlatRoutedExpert(cache_prefix=) under the run's tt_cache "
        "(<art>/tt_cache/<mesh>[-scaled]/<version>/experts-<revision>/, tt/experts.py:expert_cache_prefix: keyed by "
        "the layer's stored expert tensors, the layer and EXPERTS_IMPL; the fork's file names add the dtype and the "
        "layout's hash), with a completion marker that also keeps bfp4_stats. A hit skips the host dequant, checks "
        "and packing. off = build from the checkpoint every time (the miss path, no files). 8x4 layer 0, bfp4: see "
        "BREADCRUMBS FIX.4 for miss / hit times; 7.6 GB of disk per layer (bfp4), 13.6 GB (bfp8)",
    ),
    "EXPERTS_PRECISION": Setting(
        "glm",
        ("full", "glm"),
        "the routed experts' and the shared expert's own precision on the flat op (tt/experts.py, tt/shared_expert.py; "
        "MATMUL_FIDELITY stays the rest of the model's): full = the expert matmuls at MATMUL_FIDELITY with fp32 DEST, "
        "the flat op's x / h / y as bf16 tiles; glm (P.4, owner comparison 2026-10-18) = Marko Staletovic's flat "
        "expert as GLM-5.3 runs it (glm53_flash_d_p/tt/experts_ag.py): the expert matmuls at LoFi, x and h tiles bfp8 "
        "(x_bf16 = h_bf16 = False), y bfp8 TILE (y_bf16 = False; combine reads bfp8 TILE), fp32 DEST in gate/up and "
        "down (down_fp32 = True, GLM_AG_DOWN_FP32's default). Combined with EXPERTS_IMPL=bfp4 for GLM's weight format. "
        "P.4, 2x2, chunk 53248-55296, plain profile, flat: experts 128.9 (glm + bfp4) vs 149.2 ms (full + bfp8) per "
        "chunk (the deleted separate-combine chain); ladder last (glm + bfp4) worst layer L03 0.99524, state min 0.99710; "
        "layer-0 experts vs CPU rel 0.131, bias +3.4 % (DS41-64)",
    ),
    "COMBINE_ACC": Setting(
        "bf16",
        ("fp32", "bf16"),
        "P.9 (owner comparison 2026-10-20, option B): the routed experts' weighted combine's running sum of each "
        "home token in its accumulator core's L1 (combine_rf sparse_combine_rf acc_dtype): fp32 = 20 KB a token, so the "
        "free cells' 9 accumulators hold <= 558 tokens and a 2x2 chip's 1024 home tokens run in 2 passes (the flat expert "
        "streams every expert's weights twice a layer, DS41-69); bf16 = 10 KB a token (the packer's L1 accumulation adds "
        "each weighted row from fp32 DEST onto a bf16 sum: one bf16 rounding per added row; the weight multiply "
        "unchanged, at fp32 DEST), so the chunk runs in ONE pass (asserted, tt/experts.py). Same fixed summation "
        "order (det): calls stay bit-identical. bf16 is the default since the owner accepted P.9. P.9, 2x2, chunk 2048, synced experts call, shipped glm precision: 5.74 ms (2 passes) -> 4.29 "
        "ms (1 pass) at layers 0 and 24; full precision vs float64 (128 tokens) fp32 rel 0.0029, bf16 0.0041, bf16 vs "
        "fp32 0.0029 (bias -2.3e-4)",
    ),
    "COMBINE_PLANNER": Setting(
        "multi",
        ("one", "multi"),
        "P.11 (owner comparison 2026-10-20): the routed experts' weighted combine's per-call planner "
        "(combine_rf combine_planner_prepare, column mode: the column's experts, id offset, passes, the det order table) "
        "on one = ONE core per pass (today's program; ~1.57 ms a layer at 1024 home tokens per chip, DS41-73); multi = "
        "the column multi-core planner (Marko Staletovic's V2 multi-core structure ported to the column options: up to "
        "32 cores per pass, chunks of a home's tokens, count exchange, row slices, det order lists per chunk), its "
        "outputs bit-identical to the one-core planner's (fork tests/unit/test_column_planner_mc.py), so the model's "
        "numbers do not change. Not a precision switch",
    ),
    "FFN_NORM_DTYPE": Setting(
        "bf16",
        ("fp32", "bf16"),
        "P.8 (owner comparison 2026-10-19): the dtype ffn_norm writes (tt/norm.py, gather=True): fp32 = today's; bf16 = "
        "the normalized row rounded once to bf16 by the last multiply's packer, so the row all-gather moves half the "
        "bytes; the router (bf16 x @ bf16 gate) and the experts / shared expert read it as is (no conversion op)",
    ),
    "EXPERTS_SUM": Setting(
        "gather_add",
        ("gather_add", "reduce_scatter"),
        "the dispatch groups' expert partials (and the shared expert's TP partials, tt/shared_expert.py) [S/SP, H] fp32 "
        "summed over axis 1: gather_add = all_gather + SFPU "
        "ttnn.add, then the chip's hidden slice (exact fp32, K-30); reduce_scatter = one CCL whose FPU add reads fp32 "
        "at about TF32",
    ),
    "MOE_IMPL": Setting(
        "block",
        ("flat", "block"),
        "the MoE layer (tt/MOE_BLOCK.md; owner decisions 2026-10-27, reports/moe_block_decisions_2026-10-27.md, an A/B "
        "on the 2x2). Default block: OWNER DECISION 2026-10-28 (FIX.6), the one configuration is the block MoE with its "
        "fused head (MOE_HEAD / MOE_TAIL fused) on the torus. Evidence (supervision.md 06:20): 8x4 FABRIC_2D_TORUS_XY, "
        "one 5120-token chunk, 40 layers, device 219.5 / 233.9 / 237.5 / 249.1 / 284.4 ms at context 0 / 50k / 100k / "
        "200k / 500k vs flat on FABRIC_2D 308.1 / 314.6 / 323.3 / 338.0 / 375.7; warm host wall 615-646 ms vs 672-1005; "
        "L.s10240 PASS (top5 1.0, final hidden rel 0.192) and L.smoke Paris with block on the torus. The galaxy boxes "
        "open FABRIC_2D_TORUS_XY; flat on the torus hangs at the first MoE call (DS41-130) and block's fused head does "
        "not build on 8x4 FABRIC_2D lines (DS41-129), so FABRIC_2D is no longer a target on the Galaxy (owner: 'I don't "
        "care about fabric2d if torus works'); flat stays selectable (the FABRIC_2D boxes, comparisons). "
        "flat = today's steps: router (ROUTER_IMPL), routed experts "
        "(flat op + weighted combine), shared expert (flat op), moe_add. block = Marko Staletovic's MoE block "
        "(models/demos/moe_block, tt/moe_block_adapter.py) for all four, in the experts step: x as ffn_norm gathers it "
        "(D2: the block's x_gathered input, ffn_norm unchanged); the block's ops router (D3: partial logits over H/TP, "
        "reduce-scatter, softplus + sqrt, router_topk) with this model's gate weight, recentred selection bias, route "
        "scale 1.5, top 6; the ops head (D4); the shared expert as ttnn matmuls at LoFi with bfp8 weights (D5); the "
        "routed experts as our FlatRoutedExpert (bfp4, weights_nk, our weight cache, EXPERTS_LAYOUT: D6, marko = the "
        "block's own layout_of) in the block's "
        "token exchange (owner dispatch, flat experts + combine in one program) with the fixed combine order (D7: the "
        "exchange's deterministic planner); EXPERTS_PRECISION=glm only (D8); lines or a torus (D10 lines on the 2x2; D14 torus, the Galaxy's fabric since FIX.6). MOE_HEAD picks the "
        "head",
    ),
    "MOE_HEAD": Setting(
        "fused",
        ("ops", "fused"),
        "MOE_IMPL=block only (tt/MOE_BLOCK.md): the MoE block's head. ops = a program a stage (D4): ffn_norm gathers x, "
        "the block's ops router and ttnn shared expert, the exchange's fixed combine order (D7). fused (default: owner "
        "decision 2026-10-27, after the 2x2 A/B - device 61.97 vs 66.9 ms a chunk, MoE section 17.45 vs 22.03, clean "
        "wall 102.2 vs 160.6; flat 65.74 / 138.0; the ladder at the ops head's metrics) = Marko Staletovic's fused head "
        '(owner decision 2026-10-27 "go with fused head"): one program a chip - x gathered over TP on its ports (ffn_norm then does not gather), the '
        "shared expert engine (bfp8, LoFi as D5, clamped SiLU 10) with its row reduce-scatter, the router (logits HiFi3, "
        "fp32 sums, sqrt(softplus) with the fp32 softplus, + bias, top 6, weights / sum x 1.5), the routing exchange, "
        "the combine planner and dispatch prep; its planner has no fixed combine order, so the combine snaps each term to "
        "2^-19 (owner decision 2026-10-27: keep the snap - it supersedes D7 for the fused head; order-independent fp32 "
        "sums: bit-identical runs; DS41's routed sums peak at 5.4, under 2^24 x 2^-19 = 32)",
    ),
    "SE_PACK_PRECISE": Setting(
        "on",
        ("on", "off"),
        "MOE_HEAD=fused, GEMM_INPUTS=bfp8 (FIX.9, DS41-132): the fused head's shared expert packs its bfp8 h with "
        "bfp8_pack_precise - one rounding from fp32 DEST at the 16-value group's shared exponent, as ttnn.typecast "
        "packs its bfp8 x and as the reference casts h to fp8 once (models/demos/moe_block/tt/se_engine3.py "
        "pack_precise; MoEDtypes.shared_pack_precise). off = the packer's default: first to 7 bits at each value's own "
        "exponent, then at the shared one (ties away twice, tests/test_bfp_packer_rounding.py). Same format and "
        "speed. 8x4, layer 0, fused head's shared expert vs float64 native (tt/perf_se_activations.py): off rel 0.01704 "
        "bias +0.0074, on 0.01647 / +0.0063 (bf16 x / h: 0.01158 / -0.0085); head 462.2 vs 463.9 us a call",
    ),
    "MOE_TAIL": Setting(
        "fused",
        ("ops", "fused"),
        "MOE_IMPL=block only (tt/MOE_BLOCK.md): the MoE block's tail, the combine's owner rows back to [S/SP, H/TP]. ops "
        "= a program a stage: tilize, + the shared expert's owner rows, all_to_all_async_generic over TP. fused = Marko "
        "Staletovic's TailA2A (models/demos/moe_block/tt/tail_a2a.py; the default, owner decision 2026-10-27 after the 2x2 "
        "A/B: device 61.2 vs 61.97 ms / chunk, alternating clean walls 110.8 / 110.8 vs 120.0 / 125.2, ladder "
        "metrics identical): the same three in one program a chip, the tilize and add under the "
        "sends; the same output (bf16 TILE, then GLUE_FOLD's typecast as for ops)",
    ),
    "TAIL_A2A_PREPARED": Setting(
        "on",
        ("on", "off"),
        "P.2 (galaxy 8x4, owner focus: host time per chunk): MOE_IMPL=block, MOE_TAIL=fused only. on = the fused tail's "
        "(TailA2A, models/demos/moe_block/tt/tail_a2a.py) cached descriptors, one per (y, se, out) addresses, are each "
        "prepared once (ttnn.bringup.prepare_program, MoEOptions.tail_prepared) and dispatched with "
        "ttnn.bringup.run_prepared: no per-call hashing of the inline kernel sources, no runtime argument re-applied "
        "over 32 chips' programs (was ~1.05 ms host a call, 40 a chunk). off = ttnn.generic_op on the cached descriptor "
        "(the path before P.2). Same programs, same runtime arguments: device time and results unchanged",
    ),
    # ---- mHC (tt/mhc.py, tt/residual.py)
    "RESIDUAL_MIX": Setting(
        "fused",
        ("fused", "addcmul"),
        "the mHC residual post * y + comb^T x (tt/residual.py): fused = ttnn.bringup.mhc_post in one program; addcmul "
        "= per output stream multiply + 4 x addcmul; both fp32 on the SFPU",
    ),
    "MHC_FUSED": Setting(
        "on",
        ("on", "off"),
        "the mHC chain of the all-device model (tt/model.py; P.14, owner comparison). on = fewer passes over the fp32 "
        "streams: each hc step computes its coefficients AND collapses the streams with the pending / previous pre in "
        "ONE ttnn.bringup.mhc_pre_xing(collapse_pre=) program (attn_collapse / ffn_collapse do no work; the block "
        "boundary carries the whole [S/SP, 24] mixes as the pending pre); the residuals (ttnn.bringup.mhc_post "
        "stats_weight=) also emit the next hc step's partial row (sum of squares + the chip's hc_fn slice), so that hc "
        "step reads no streams before its exchange; the final collapse is mhc_pre_xing(coefficients_given=True); the "
        "engram add's partials come from one mhc_pre_xing_pack(engram=) pass and its add is mhc_post with an "
        "identity comb. off = the P.13 chain (matmul + pack per hc step, the multiply / addcmul collapses, the "
        "engram op chains)",
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
    "CCL_IMPL": Setting(
        "fabric_bf16",
        ("stock", "fabric", "fabric_bf16"),
        "the cross-chip collectives (tt/ccl.py, P.5): stock = ttnn.all_gather, and the exact TP partial sums (ATTN_O_SUM "
        "/ EXPERTS_SUM / HC_TP_SUM gather_add) as all_gather + SFPU ttnn.add (+ mesh_partition for the hidden-split "
        "ones); fabric = the bring-up fabric forks: every all_gather whose contract fits on ttnn.bringup."
        "fabric_all_gather, the hidden-split sums as ONE ttnn.bringup.fabric_reduce_scatter(dim=3, exact_fp32_add=True) "
        "each, the replicated ones (HC_TP_SUM: tp_sum) as fabric_reduce_scatter(dim=2, exact) + fabric_all_gather; "
        "every add one exact fp32 rounding on the SFPU (TP = 2: bit-identical to stock). The head <-> seq all_to_all "
        "stays stock (no fork). stock and fabric: the same numerics. fabric_bf16 (P.6, owner comparison: Marko Staletovic's "
        "GLM-5.3 choice, GLM_SCATTER_OP=fabric_bf16) = fabric, but the reduce-scattered TP partial sums (sum_cols: "
        "ATTN_O_SUM / EXPERTS_SUM; tp_sum: HC_TP_SUM's tile-aligned ones) typecast to bf16, reduced by "
        "fabric_reduce_scatter's default FPU add, tp_sum's row gather in bf16, then typecast back to fp32: lossy, a "
        "precision switch (exact: fabric); the norm / engram-gate sums the fork does not take stay exact fp32. "
        "The producers of the big partials (attention wo_b, moe_add, q_a; tt/ccl.py:partial_dtype) write them in bf16 "
        "directly: a separate fp32 -> bf16 typecast ([1024, 5120] 0.071 ms) costs more than the bf16 RS saves "
        "(DS41-67). P.6, 2x2, chunk 53248-55296, plain profile: device total 639.62 vs fabric 641.23 (P.5; the "
        "typecast-both-ways version 642.09); ladder last worst layer L03 0.99674 vs 0.99681, state min 0.99776",
    ),
    "TP_SUM_BLOCKS": Setting(
        "cols",
        ("cols", "rows"),
        "how tt/ccl.py:tp_sum (the replicated TP sums: q_a / wkv exchange, mHC mixes, compressor, indexer and engram "
        "partials) splits its fabric reduce-scatter. cols (K.1 contract, DS41-128): column blocks (dim 3) + a column "
        "gather; when the width does not split into 32 x TP columns, the fixed-order gather_add. The add order then "
        "depends on the column only, so a position's sum is the same whatever row of the chunk it lands on (the "
        "serving contract: layer 0's window KV rewritten by the pulled-back chunk with the same bytes). rows: the old "
        "row blocks (dim 2) + row gather: on TP > 2 each row block adds the TP partials in another ring order, so a "
        "position's bf16 sum depends on the chunk's start (1 ulp on whole tiles). Same volume either way",
    ),
    "HC_TP_SUM": Setting(
        "gather_add",
        ("gather_add", "all_reduce"),
        "the [S/SP, 32] mHC partial-row exchange over axis 1: gather_add = all_gather + SFPU ttnn.add (exact fp32, "
        "K-30); all_reduce = one CCL whose FPU add reads fp32 at about TF32",
    ),
    # ---- engram host path (2x2 session, bringup/ENGRAM_HOST_PATH.md; ENGRAM_TABLE / _GATHER_THREADS: rows bit-identical)
    "ENGRAM_TABLE": Setting(
        "resident",
        ("mmap", "random", "resident"),
        "where tt/engram.py reads the engram tables (layers 1, 14; 94.5 GiB each, 8-bit). mmap = the checkpoint "
        "mapping with the kernel's read-ahead (each random 256-byte row faults in up to read_ahead_kb); random = the "
        "same mapping with MADV_RANDOM over the table's bytes; resident = the 8-bit bytes copied into anonymous memory "
        "once at build (ENGRAM_HUGEPAGES, ENGRAM_BUILD) (~189 GiB of host RAM for both layers). QB 2x2 host (503 GB RAM, 990 EVO Plus "
        "NVMe, read_ahead_kb 128, 32 CPUs), layer 1, 5120 tokens, gather a layer a chunk: cold mmap 7.2 s, cold "
        "random 0.42 s (8 threads), resident 9-15 ms; load of one layer resident 109 s. With ENGRAM_ROWS=bfp8 resident "
        "holds the rows already as bfloat8_b (272 B a row, 104.4 GB a layer; built in 84 s at 1.24 GB/s) and a 5120-token "
        "chunk takes 7.6 ms on 8 threads from hash to TILE bytes. Default resident (owner 2026-10-28; random before). "
        "Tests: random (owner 2026-10-28: no table build at every test's start; rows bit-identical, "
        "tests/test_engram_table_modes.py)",
        test="random",
    ),
    "ENGRAM_HUGEPAGES": Setting(
        "off",
        ("off", "on"),
        "FIX.10 (DS41-133): MADV_HUGEPAGE on the resident engram rows (ENGRAM_TABLE=resident). The Galaxy host's THP "
        "defrag is 'madvise', so every 2 MB fault of a MADV_HUGEPAGE region compacts memory in the faulting thread. "
        "Both bfp8 tables, page cache full of the weight cache, the thread fix in: on = 171 s build (layer 14 alone 130 s, "
        "sys 1480 s, compact_stall 125k, 22k fallbacks to 4 KB anyway) and direct reclaim dropped 95 GB of the page "
        "cache (the weight cache the load reads next); off = 74 s, no compaction, page cache kept. on gathers faster: "
        "hash + gather + TILE bytes of a 5120-token chunk 6.3 vs 8.5 ms a layer (8 threads; ~4 ms a chunk on the host "
        "for both layers, off the critical path with ENGRAM_ASYNC). Default off (no load time)",
    ),
    "ENGRAM_BUILD": Setting(
        "background",
        ("background", "load"),
        "FIX.10: when ENGRAM_TABLE=resident builds its tables. background = on one thread started when the model's "
        "engram layers are created (before the decoder layers load), so the ~74 s build (2 x 104 GB read at ~2.7 GB/s and "
        "packed by numba, nogil) overlaps the weight load; the first chunk's begin_chunk waits for whatever is left. "
        "load = in the constructor, blocking (the path before FIX.10). Same bytes either way",
    ),
    "ENGRAM_ROWS": Setting(
        "bfp8",
        ("bf16", "bfp8"),
        "the engram rows' format from the host to engram_kv's matmul. bf16 = exact (e4m3 x 2^k; the host converts "
        "8-bit -> fp32 -> bf16, 2 B a value); bfp8 = bfloat8_b packed from the table's bytes by integer operations "
        "only (tt/engram_bfp8.py, byte-equal to ttnn's packer; 1.0625 B a value), wrapped per mesh row without a "
        "float conversion and read by the matmul as sent. bfp8 is lossy: on 1M table rows a layer it changes 10 % "
        "of the values (1 % to 0), rel 3.5e-3, and the wkv output by rel 3.5e-3, where DeepSeek's own fp8 re-quant "
        "of the rows changes none (today's device engram_kv: rel 4.1e-3 from the bfp8 weights); 2x2 ladder bf16 vs bfp8: "
        "every metric within 1.3 %. Host 8 ms pack + 7 ms tiling vs bf16's ~40 ms. Default bfp8 (owner 2026-10-28; "
        "the Galaxy 40-layer ladder + K.1 is the confirmation)",
    ),
    "ENGRAM_ASYNC": Setting(
        "on",
        ("on", "off"),
        "P.6 (owner 2026-10-28): when begin_chunk does the engram layers' host work (hash, lookup, pack, upload). on = "
        "begin_chunk allocates each engram layer's device rows on the main thread (same allocation order as off), "
        "computes the new tails (3 ids) and returns; one worker thread then fills them, layer 1 first (hash, lookup + "
        "pack, wrap into host shards, the upload: copy_host_to_device_tensor, GIL released) while the main thread "
        "dispatches embed, layer 0, ...; take_engram_rows(i) waits for layer i's job just before its engram op. The write "
        "goes into CQ 0 (the harness and tt-d-gen open one CQ; MeshDevice has no Python CQ count), which is in order, "
        "so it lands before layer i's ops, enqueued after the wait. Same bytes as off "
        "(tests/test_engram_bfp8.py, test_engram_table_modes.py; tt/perf_engram_async.py compares the device bytes). "
        "off = today's path: both layers' hash + lookup + pack + upload on the main thread before embed",
    ),
    "ENGRAM_GATHER_THREADS": Setting(
        8,
        None,
        "Python threads the engram row gather is split over (torch indexing releases the GIL). QB: cold random 1 / 8 "
        "/ 32 threads 1265 / 424 / 484 ms; mmap hot 1 / 8 threads 10 / 19 ms; resident 1 / 8 threads 11-15 / 9-13 ms "
        "(torch intra-op 16 / 8 threads). Default 8: the cold gather (owner 2026-10-28); also the bfp8 pack's threads",
    ),
    "CKV_FORMAT": Setting(
        "bfp4_rows",
        ("bf16", "bfp4_rows"),
        "the compressed KV attention table (tt/compressor.AttnTable, read by sparse_sdpa; needs STATIC_TABLES=on). OWNER "
        "DECISION 2026-10-28 (F208, FIX.8; developed on the 2x2; default bfp4_rows after the 2x2 ladder, owner: a win): "
        "the reference stores the compressed KV in 4 bits (fp4 e2m1 + e4m3 per 16, ds41_ref.py:797-799). bfp4_rows = the "
        "history as bfloat4_b row records, 320 B a row ([32 exponents | the same 32 | 256 nibble bytes]), written by the "
        "packer (ttnn.bringup.ckv_bfp4: SFPU round to nearest even, byte-equal to the host packer; the packer alone "
        "truncates bfp4 in effect) and read by the sparse_sdpa fork's kv_format ROW_BFP4 through the unpacker (no "
        "data-movement RISC byte work), the window region a separate bf16 tensor (index list not compacted). The served "
        "records stay bf16 rows holding the rounded values (contract and tt-d-gen unchanged; a host load re-encodes "
        "them exactly). 2x2 rung last (55296, chunk 2048, 8 layers): device 61.56 -> 61.79 ms per chunk (+0.4 %), "
        "sparse_sdpa 0.93-0.95 ms both; ladder err vs golden: compress_kv 0.004-0.039 -> 0.111-0.118 (limits "
        "0.19-0.21), layers within limits (worst L20 0.032 / 0.051, L24 0.029 / 0.042); table 3.3x smaller. "
        "bf16 = bf16 rows, stock sparse_sdpa",
    ),
}

# F137 (owner): every test runs these defaults; there is no max-precision device run and no max-precision path. FIX.8
# deleted the parity-only switches (NUMERICS, COMPRESSOR_X, ENGRAM_KV_QDQ, ATTN_IN_MM, WINDOW_SDPA and its _OUT /
# _SCORES / _PROBS / WINDOW_PIECES) with their code (tt/quant.py, tt/exact_mm.py, tt/window_exact.py). FIX.9 deleted the
# second expert pipeline (EXPERTS_KERNEL loop / unified, EXPERTS_IMPL fp4_decode, EXPERTS_DECODE, EXPERTS_COMBINE separate,
# SHARED_EXPERT_IMPL linear): the flat routed expert with the weighted combine and the flat shared expert are the only path.
settings = Settings("DS41_", TABLE)
