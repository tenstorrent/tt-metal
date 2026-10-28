# Round 1 proposals (G2-r1). All pre-approved kinds (skill §7) unless marked.

Ranked by expected saving at full depth (from analysis.md). Each is screened at BENCH_LAYERS=2+2 with a back-to-back baseline, then the full bench and suite run on the kept ones.

| id | change | expected | risk / measurement |
|---|---|---|---|
| R1-1 | Fold rotate-half into the projections: extra q·R / k·R output columns (R is a signed permutation, so the weights are exact), RoPE = q*cos + qR*sin | about -200 ms | none numerically (an exact permutation, not a scalar fold, v1.19); test_model + suite |
| R1-2 | ttnn SDPA (scale 1.0, explicit additive mask, q_chunk = whole Sq so the chunk count is 1, v1.21), replacing scores/softmax/P@V | about -210 to -230 ms | precision: softmax no longer fp32 end to end, so a variance table and the full check_gates. Needs the F1/F9 table with v1.13 positive and negative controls at B = 2/16/32/512 for (96, 96), (32, 96) and (32, 32) before it is kept. Fallback R1-2b: MatmulMultiCoreReuseProgramConfig over the full grid for the batched matmuls, keeping fp32 scores |
| R1-3 | Cross-attn head split through nlp_create_qkv_heads / split heads without reshape + permute | about -85 ms | order-only; test_model |
| R1-4 | Linear program configs (2D / minimal_matmul) swept per role in one job, plus relu fused in the matmul (a precision change, v1.14: disclosed) | about -35 to -45 ms | sweep job; suite |
| R1-5 | Eltwise cleanup: LoFi on any remaining R path, drop the typecast pass | about -15 ms | small |
| R1-6 | Host: RoPE tables / DRoPE MLPs on device, vectorized prepare (prepare_s 523 ms, upload 45 ms) | wall time only | does not count toward device time; measured as wall |

Infrastructure for round 1: `compare_bench.py` (CONTRACT §4) does not exist in the tree yet; it is written first.
