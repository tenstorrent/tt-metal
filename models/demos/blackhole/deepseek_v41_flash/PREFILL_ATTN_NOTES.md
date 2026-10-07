# Prefill attention / indexer: 4k per-op profile, ranking, fused fp4 kernel (branch ssinghal/dsv4p1-pf-attn2, base df4ffc9f838)

Scope (user): 4k ISL only (4k B=16: U=4 users/row, C=1024, chunk at s0=3072); long-context ideas (sparse_sdpa gather) are out of scope until a confirmed 4k gain.

## Profile method
`tests/test_prefill_prof.py` (eager chunk, device profiler drained per layer, 32-chip mean of DEVICE KERNEL DURATION), layers 0,2,3,20,21,24, tools/prof2_sum.py.
Layer types: L0 = sliding-window dense (no compressor), L2/L20/L24 = ratio-4 compressed + indexer (L2, L20 key owners; L24 index source aliasing L20's keys), L3/L21 = ratio-128 (shared ids, no indexer).
Raw: /mnt/tt-data/ssinghal/dsv4-logs/pf_attn2_prof4k/summary.txt (baseline), pf_attn2_p_fused (fused fp4). ms are per layer per chunk (4 users x 1024 tokens per row), kernel time.

## Per-layer totals (ms), baseline
| layer | type | kernel sum | attention-side part | MoE+CCL | mHC |
| L0 | SWA dense | 28.2 | 8.1 (sdpa 1.9, proj 2.6, other 2.7, rs 1.1) | ~12.6 | 7.4 |
| L3 / L21 | ratio-128 | 38.1 / 40.9 | 19.7 | ~11 | 7.6 |
| L2 / L24 / L20 | ratio-4 + indexer | 56.2 / 53.5 / 63.9 | 33-35 | ~13-24 | 7.6 |

## Attention-side ops, ranked, index layer (L24; L2/L20 similar), ms per layer per chunk
| rank | ms | % of layer | op group |
| 1 | 9.7 | 18% | sparse_sdpa (4 users; DRAM random gather bound) -- also 9.7 on L3/L21; ~11 on L2/L3 |
| 2 | 7.2 | 13% | indexer fp4 simulation of q (pf_tune.fp4_fast: ~170 elementwise ops, each a full DRAM pass over [4,64,1024,128]) |
| 3 | 2.6 | 5% | real projections (wqkv, wq_b, wo_a, wo_b, minimal_matmul 1.46 + rope pe 1.1) |
| 4 | 5.3 | 10% | indexer rest: wproj/wq_b linear 1.74, indexer_score_dsa 0.96, topk 0.48, permute/to_layout/where ~2 (of which rope matmul/addcmul 1.0) |
| 5 | 1.6+1.5 | 6% | sparse glue (compact/gather, kv-table concat 0.6, q pad) + "sparse other" |
| 6 | 1.1 | 2% | attention output reduce-scatter (CCL) |
| 7 | 0.9-1.8 | 2-3% | compressor (L2 1.76, L20 0.89) |
Non-attention for reference: mHC 7.6 (14%), moe reduce_scatter+all_to_all 5.6-13.6, moe combine/dispatch/experts 2-4 each, allgathers 1.5.

Ranking by (gain x ease): (1) fused fp4 kernel: -7.1 ms on each of ~19 index layers = ~135 ms of ~2.3 s kernel per chunk (~5.5-6%), pure Python/JIT, bit-identical -> chosen.
(2) rope (matmul P + addcmul) of the indexer q, 1.0 ms/index layer: could be folded into the same kernel (~1%), not done.
(3) sparse_sdpa gather 9.7 ms x 38: biggest single op but changes selection semantics / needs C++ (out of scope).

## Change: DSV41_PFA_FP4=fused (default OFF)
tt/pf_fp4.py + tt/pf_kernels/fp4_{reader,compute,writer}.cpp, fp4_sfpu.h: one generic_op per call (bf16 tile in/out). Per tile (= one 32-col scale block per row):
|x| (SFPU) -> FPU row-max reduce -> SFPU scale=2^ceil(log2(max(amax,6*2^-126)/6)) and 1/scale (exponent arithmetic) -> FPU bcast-col multiply -> SFPU e2m1 grid (thresholds, half-up, clamp 6) -> bcast-col multiply by scale.
Test tests/test_pf_fp4_fused.py: 0 differing elements vs pf_tune.fp4_fast on 4 shapes/scales incl. amax = 6*2^k, zero blocks, tiny values. Device time of the two fp4 calls in an index layer: 7.5 ms -> 0.42 ms.
Layer kernel sum (same profile, same host): L2 56.2 -> 49.1, L20 63.9 -> 56.7, L24 53.5 -> 46.7 (-7.1 ms each). L3/L21/L0 unchanged.
Accuracy: output is bit-identical to the current default (fast), so layer PCC change is exactly 0 and sparse selection is unchanged.

## Paired 40-layer results, 4k B=16 (isl4k_b16, grid env: DSV41_LAYERS=0-39 DSV41_ENGRAM_RAM=1 DSV41_SESSION=isl4k_b16 DSV41_BUILD_SLOTS=10 DSV41_BUILD_STAGGER_S=480 DSV41_SPEC=0, no MEMLOG), same commit, same host
Logs /mnt/tt-data/ssinghal/dsv4-logs/pf_spec_int_a_{off,on}_h{44,34}*.log. "replay loop" = prefill timing total_replay_loop (device replay + host, the stable number); TTFT also contains compile-run/host noise that differs run to run (off TTFT on h44: 10.97 s and 9.29 s for identical settings).
| host | flag | replay loop (s) | replay_per_chunk (s) | TTFT (s) |
| .44 | off | 8.73 / 8.73 (2 runs) | 8.21 / 8.20 | 10.97 / 9.29 |
| .44 | on  | 8.40 | 7.95 | 8.96 |
| .34 | off | 8.70 | 8.19 | 9.27 |
| .34 | on  | 8.47 | 7.94 | 11.51 (ran while other builds hammered NFS/host) |
Replay-loop gain: -3.8% (.44), -2.6% (.34): about 0.25-0.33 s of 8.7 s (the eager kernel sum predicts ~5%; the traced replay overlaps part of it). TTFT is not a reliable discriminator at this size (noise 1-2 s).
Accuracy: 16/16 users' 64-token outputs identical on vs off (h44 pair and h34 pair); kernel bit-identical to fp4_fast in the unit test.
Recommendation: safe to enable by default (bit-identical, ~3% of 4k prefill, scales with the number of index layers and with the q-head count x tokens, so it is slightly larger share at larger chunk work); next candidates: fold the indexer rope (matmul+addcmul, ~1 ms/layer) into the same kernel; mHC (7.6 ms/layer, 14%) and the MoE reduce-scatter are bigger but not attention.
Reproduce: DSV41_PFA_FP4=fused with tools/spec_lt.sh <host> <name> "<grid env>" models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session; unit: pytest tests/test_pf_fp4_fused.py.

## Extended validation of DSV41_PFA_FP4=fused (grid env, no MEMLOG, same host per pair, flag off vs on, 40 layers)
Logs dsv4-logs/pf_spec_int_v_*_{off,on}.log. Replay loop = prefill timing total_replay_loop (s); outputs = the demo's decoded outputs of every user, compared off vs on.
| cell | host | replay loop off -> on | outputs identical |
| 4k B=32 (U=8) | .34 | 16.74 -> 16.16 (-3.5%) | 32/32 |
| 4k B=4 (U=1) | .44 | 2.29 -> 2.23 (-2.6%) | 4/4 |
| 60k B=4 (C=2048, U=1, ISL 60453) | .43 | 40.36 -> 39.21 (-2.9%); TTFT 43.7 -> 40.0 s | 4/4 |
| 4k B=16, DEFAULT adaptive spec decode {1,3,5}, default unified/ring MoE | .44 | 8.72 -> 8.52 (-2.3%) | 16/16 plain + 2/2 spec outputs |
| 4k B=16 (earlier) | .44 / .34 | 8.73 -> 8.40 / 8.70 -> 8.47 | 16/16 |
Unit test (tests/test_pf_fp4_fused.py, 12 cases, device): 0 differing elements on q [U,64,C,128] for (1,2048) (= 60k B=4 and the 2048-row ratio-4 chunk at long context; the fp4 input depends only on the chunk, not on the context length), (4,1024), (8,512), (1,1024), and key shapes [U,1,C/4,128] (1,512), (4,256), (8,128), (1,32).
Decode/spec: the fused kernel is used ONLY by the prefill indexer (DSV41PrefillIndexer._fp4 in tt/prefill_sparse.py). Decode and spec verify rounds use their own fp4 code (DSV41DecodeIndexer._fp4_blocks, tt/spec_paged.py), untouched by the flag.
