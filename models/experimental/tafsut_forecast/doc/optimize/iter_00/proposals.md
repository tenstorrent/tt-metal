# Tafsut round 1 proposals (from round 0, job 955)

Target: paper, 1697.51 ms device (fixed record, job 1061; 1697.8 in job 942). Ranked by expected gain. All pre-approved kinds (skill §7), so G2-r1 is
automatic once G1 is approved; round-1 work waits for G1 (D1).

| id | change | expected | risk | measure |
|---|---|---|---|---|
| R1-1 | RoPE: fold batch into M for the `rot` matmul (reshape to (1,1,B*NH*S,64)), then the fused `qkv_heads_rope` op from chronos_forecast (ops/qkv_heads_rope, tt/mha_core.py:124-200, commits ec79b81bd6e, 41c6856ef36), which also removes the 6 RoPE binaries and NlpCreateHeads | -35 % (reshape), -44 % (fused) | low / medium (L1 at s_pad 128, batch-shared cos/sin) | BENCH_LAYERS=4 screen, test_attention + suite |
| R1-2 | 2D program configs per role (QKV, wo, ff up, ff down, embed), one sweep job; copy chronos tt/program_configs.py:150-290 (599000d80f5, 0b755e48966) | -25 % | low | sweep json, bench job |
| R1-3 | relu fused in the ff-up program config (comes with R1-2) | -3.4 % | none | bench |
| R1-4 | residual add in the wo / ff-down epilogue, or chronos eltwise_add / add_rms_norm (df78db0dabd, 109ab6e059e) | -2.2 % | low-medium | bench + suite |
| R1-5 | SDPA output_concat_heads | -0.8 % | low | bench |
| R1-6 | long_context chunk micro-benchmark 1088/32 vs 1152/64 (D1 ruling 1) | n/a (long_context guard) | low | micro-bench |
| R1-7 | fp32_dest_acc_en off for HiFi2 linears (8-tile subblocks); RMSNorm-gamma fold; exp_approx_mode (D0 Q5) | -3-5 % | accuracy: variance table + full check_gates (v1.19) | suite both precisions |
| R1-8 | QKV HiFi4 -> HiFi2 at the performance preset only (accuracy keeps HiFi4, D1 ruling 4) | -5.6 % perf preset | accuracy (job 923) | suite |

Projection: R1-1..R1-3 take device time from ~1698 ms to about 600 ms.
