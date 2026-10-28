# Tafsut candidates (PLAN G1 item 6)

One line per candidate, in the order tried. Status: proposed / screening / kept / dropped / blocked. Rulings from D2.
Round 0 record: paper 1697.8 ms device, accuracy (iter_00/bench.json).

| id | round | change | ruling (D2) | status | evidence |
|---|---|---|---|---|---|
| R1-1 | 1 | RoPE: batch into M for `rot`, then fused `qkv_heads_rope` (model-local generic_op, absolute kernel paths, micro-bench, fresh-input trace test); alt: fold the signed permutation R into Wq/Wk (order-only if bitwise-equal weights, else v1.19) | pre-approved | proposed | |
| R1-2 | 1 | 2D program configs per role (one sweep job) | pre-approved | proposed | |
| R1-3 | 1 | relu fused in ff-up program config | pre-approved | proposed | |
| R1-4 | 1 | residual add in wo / ff-down epilogue, or add_rms_norm | pre-approved | proposed | |
| R1-5 | 1 | SDPA output_concat_heads | pre-approved | proposed | |
| R1-6 | 1 | long_context chunk micro-bench 1088/32 vs 1152/64 | pre-approved | proposed | |
| R1-7a | 1 | fp32_dest_acc_en off, HiFi2 linears | performance preset only; at accuracy needs an X gate | proposed | |
| R1-7b | 1+ | RMSNorm-gamma fold; exp_approx_mode | v1.19: variance table, full check_gates, device check that can fail | proposed | |
| R1-8 | 1 | QKV HiFi4 -> HiFi2 at the performance preset | allowed under drift; accuracy keeps HiFi4 | proposed | |

Tracking (PLAN v1.12): the performance preset ships only at >= 5 % faster than accuracy; round 0 it is 2.6 %.
