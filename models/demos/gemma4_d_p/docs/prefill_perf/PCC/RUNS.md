# All PCC runs

Back to the [summary](README.md). Groups: A = 8k context, B = 256k context (flag-based tree), C = 256k context (perf branch
synced to the PR code, local switches). The error ratios compare against the base run at the same context and chunk
(`ref`); `n/a` means no base run exists for that context and chunk.

| Group | Label | Ctx | Chunk | Config | Min PCC @layer | Gate 0.91 | Overall PCC | Err L0-20 vs base | L0-20 worse | Err L21-59 vs base |
|---|---|---|---|---|---|---|---|---|---|---|
| A | `all_fixes_c2048` | 8k | 2048 | norm + MLP + attention, all LoFi bf16 (before fp32 attention) | 0.9148 @L39 | pass | 0.9812 | 1.024x | 17/21 | 1.000x |
| A | `all_fixes_c4096` | 8k | 4096 | norm + MLP + attention, all LoFi bf16 | 0.9088 @L39 | FAIL | 0.9796 | 1.016x | 16/21 | 0.842x |
| A | `all_fixes_c8192` | 8k | 8192 | norm + MLP + attention, all LoFi bf16 | 0.9103 @L39 | pass | 0.9800 | 1.018x | 16/21 | 1.034x |
| A | `attn_only_c4096` | 8k | 4096 | attention explicit config only (LoFi, bf16 acc) | 0.9008 @L39 | FAIL | 0.9772 | 1.067x | 21/21 | 0.891x |
| A | `attn_only_c8192` | 8k | 8192 | attention explicit config only (LoFi, bf16 acc) | 0.8992 @L39 | FAIL | 0.9766 | 1.047x | 21/21 | 1.119x |
| A | `attnfp32_c8192` | 8k | 8192 | attention explicit config, HiFi2 + fp32 acc | 0.9236 @L39 | pass | 0.9829 | 0.853x | 0/21 | 0.941x |
| A | `base_c2048` | 8k | 2048 | base (combined branch, no fixes) | 0.9077 @L39 | FAIL | 0.9805 | ref | ref | ref |
| A | `base_c4096` | 8k | 4096 | base | 0.8514 @L39 | FAIL | 0.9685 | ref | ref | ref |
| A | `base_c4096_r2` | 8k | 4096 | base, rerun 2 (determinism) | 0.8514 @L39 | FAIL | 0.9685 | 1.000x | 0/21 | 1.000x |
| A | `base_c4096_r3` | 8k | 4096 | base, rerun 3 (determinism) | 0.8514 @L39 | FAIL | 0.9685 | 1.000x | 0/21 | 1.000x |
| A | `base_c8192` | 8k | 8192 | base | 0.9183 @L39 | pass | 0.9813 | ref | ref | ref |
| A | `base_c8192_rep` | 8k | 8192 | base, rerun (determinism) | 0.9183 @L39 | pass | 0.9813 | 1.000x | 0/21 | 1.000x |
| A | `dram_only_c4096` | 8k | 4096 | base with activations in DRAM (L1 placement control) | 0.8514 @L39 | FAIL | 0.9685 | 1.000x | 0/21 | 1.000x |
| A | `fp32acc_c4096` | 8k | 4096 | MLP fp32 acc flag alone (no explicit MLP config) | 0.9205 @L39 | pass | 0.9825 | 0.880x | 0/21 | 0.767x |
| A | `mlp_only_c4096` | 8k | 4096 | MLP explicit config only (LoFi, bf16 acc) | 0.9158 @L39 | pass | 0.9818 | 0.981x | 7/21 | 0.787x |
| A | `mlp_only_c8192` | 8k | 8192 | MLP explicit config only (LoFi, bf16 acc) | 0.9203 @L39 | pass | 0.9827 | 0.966x | 0/21 | 0.966x |
| A | `norm_only_c4096` | 8k | 4096 | norm block-sharding only | 0.9168 @L39 | pass | 0.9820 | 0.993x | 2/21 | 0.773x |
| A | `norm_only_c8192` | 8k | 8192 | norm block-sharding only | 0.9030 @L39 | FAIL | 0.9792 | 1.001x | 10/21 | 1.045x |
| A | `packerl1_c2048` | 8k | 2048 | #57454 with attention HiFi2 bf16 + packer_l1 | 0.9151 @L39 | pass | 0.9815 | 1.015x | 10/21 | 0.990x |
| A | `packerl1_c4096` | 8k | 4096 | #57454 with attention HiFi2 bf16 + packer_l1 | 0.9133 @L39 | pass | 0.9804 | 1.013x | 8/21 | 0.819x |
| A | `packerl1_c8192` | 8k | 8192 | attention only, HiFi2 bf16 + packer_l1 | 0.9093 @L39 | FAIL | 0.9789 | 1.002x | 12/21 | 1.065x |
| A | `pr2cfg_c8192` | 8k | 8192 | norm + MLP (no attention change) | 0.8543 @L39 | FAIL | 0.9692 | 0.999x | 8/21 | 1.259x |
| A | `pr2full_c2048` | 8k | 2048 | #57454 as first posted: norm + MLP LoFi bf16 + attention HiFi2 fp32 | 0.9034 @L39 | FAIL | 0.9795 | 0.861x | 0/21 | 1.031x |
| A | `pr2full_c4096` | 8k | 4096 | #57454 as first posted | 0.9047 @L39 | FAIL | 0.9800 | 0.845x | 0/21 | 0.814x |
| A | `pr2full_c8192` | 8k | 8192 | #57454 as first posted | 0.9053 @L39 | FAIL | 0.9800 | 0.852x | 0/21 | 1.020x |
| B | `attn256k_c8192` | 256k | 8192 | attention only (HiFi2 fp32) | 0.9118 @L39 | pass | 0.9737 | 0.911x | 0/21 | 0.991x |
| B | `base256k_c2048` | 256k | 2048 | base | 0.9106 @L39 | pass | 0.9736 | ref | ref | ref |
| B | `base256k_c8192` | 256k | 8192 | base | 0.9108 @L39 | pass | 0.9734 | ref | ref | ref |
| B | `mlp256k_c8192` | 256k | 8192 | MLP explicit config only (LoFi bf16) | 0.9088 @L39 | FAIL | 0.9725 | 1.000x | 12/21 | 1.016x |
| B | `mlpfp32_256k_c8192` | 256k | 8192 | MLP explicit config, HiFi2 fp32 | 0.9146 @L39 | pass | 0.9746 | 0.940x | 0/21 | 0.977x |
| B | `norm256k_c8192` | 256k | 8192 | norm only | 0.9102 @L39 | pass | 0.9730 | 1.001x | 12/21 | 1.008x |
| B | `pr2_256k_c2048` | 256k | 2048 | #57454 as first posted | 0.9094 @L39 | FAIL | 0.9726 | 0.905x | 0/21 | 1.013x |
| B | `pr2_256k_c8192` | 256k | 8192 | #57454 as first posted | 0.9083 @L39 | FAIL | 0.9723 | 0.909x | 0/21 | 1.016x |
| B | `pr2mlpfp32_256k_c2048` | 256k | 2048 | #57454 + MLP HiFi2 fp32 | 0.9143 @L39 | pass | 0.9744 | 0.837x | 0/21 | 0.979x |
| B | `pr2mlpfp32_256k_c8192` | 256k | 8192 | #57454 + MLP HiFi2 fp32 | 0.9133 @L39 | pass | 0.9742 | 0.840x | 0/21 | 0.981x |
| C | `mlpexp_c1_attnlofi_mlpm1_c8192` | 256k | 8192 | M1 + attention LoFi bf16 | 0.9009 @L39 | FAIL | 0.9698 | 1.023x | 20/21 | 1.061x |
| C | `mlpexp_c3_attnhifi2bf16_mlpm1_c8192` | 256k | 8192 | M1 + attention HiFi2 bf16 | 0.9071 @L39 | FAIL | 0.9720 | 1.002x | 13/21 | 1.024x |
| C | `mlpexp_m1_lofi_fp32_c2048` | 256k | 2048 | #57454 + MLP LoFi fp32 (M1) | 0.9131 @L39 | pass | 0.9744 | 0.915x | 0/21 | 0.984x |
| C | `mlpexp_m1_lofi_fp32_c4096` | 256k | 4096 | #57454 + MLP LoFi fp32 (M1) | 0.9121 @L39 | pass | 0.9741 | n/a | n/a | n/a |
| C | `mlpexp_m1_lofi_fp32_c8192` | 256k | 8192 | #57454 + MLP LoFi fp32 (M1) | 0.9123 @L39 | pass | 0.9742 | 0.919x | 0/21 | 0.985x |
| C | `mlpexp_m3_hifi2_bf16_c8192` | 256k | 8192 | #57454 + MLP HiFi2 bf16 (fidelity restored, M3) | 0.9017 @L39 | FAIL | 0.9694 | 0.948x | 4/21 | 1.065x |
| C | `mlpexp_mlpoff_c8192` | 256k | 8192 | #57454 with MLP on its default path (norm + attention only) | 0.9118 @L39 | pass | 0.9736 | 0.909x | 0/21 | 0.993x |

## Flags and tree per run

| Label | Flags | Tree sha |
|---|---|---|
| `all_fixes_c2048` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1` | 7facef7e68e |
| `all_fixes_c4096` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1` | 7facef7e68e |
| `all_fixes_c8192` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1` | 7facef7e68e |
| `attn256k_c8192` | `GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | df0d9eaa019 |
| `attn_only_c4096` | `GEMMA4_ATTN_MM_PC=1` | 7facef7e68e |
| `attn_only_c8192` | `GEMMA4_ATTN_MM_PC=1` | 7facef7e68e |
| `attnfp32_c8192` | `GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | 7facef7e68e |
| `base256k_c2048` | `(none)` | df0d9eaa019 |
| `base256k_c8192` | `(none)` | df0d9eaa019 |
| `base_c2048` | `(none)` | 7facef7e68e |
| `base_c4096` | `(none)` | 7facef7e68e |
| `base_c4096_r2` | `(none)` | 7facef7e68e |
| `base_c4096_r3` | `(none)` | 7facef7e68e |
| `base_c8192` | `(none)` | 7facef7e68e |
| `base_c8192_rep` | `(none)` | 7facef7e68e |
| `dram_only_c4096` | `GEMMA4_ACTIVATIONS_DRAM_ONLY=1` | 7facef7e68e |
| `fp32acc_c4096` | `GEMMA4_DIAG_MLP_FP32ACC=1` | 7facef7e68e |
| `mlp256k_c8192` | `GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10` | df0d9eaa019 |
| `mlp_only_c4096` | `GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10` | 7facef7e68e |
| `mlp_only_c8192` | `GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10` | 7facef7e68e |
| `mlpexp_c1_attnlofi_mlpm1_c8192` | `GEMMA4_EXP_ATTN=lofi_bf16 GEMMA4_EXP_MLP_FIDELITY=LoFi GEMMA4_EXP_MLP_FP32=1 GEMMA4_EXP_MLP_PACKER=0` | 07759d549a6 |
| `mlpexp_c3_attnhifi2bf16_mlpm1_c8192` | `GEMMA4_EXP_ATTN=hifi2_bf16 GEMMA4_EXP_MLP_FIDELITY=LoFi GEMMA4_EXP_MLP_FP32=1 GEMMA4_EXP_MLP_PACKER=0` | 07759d549a6 |
| `mlpexp_m1_lofi_fp32_c2048` | `GEMMA4_EXP_MLP_FIDELITY=LoFi GEMMA4_EXP_MLP_FP32=1 GEMMA4_EXP_MLP_PACKER=0` | 07759d549a6 |
| `mlpexp_m1_lofi_fp32_c4096` | `GEMMA4_EXP_MLP_FIDELITY=LoFi GEMMA4_EXP_MLP_FP32=1 GEMMA4_EXP_MLP_PACKER=0` | 07759d549a6 |
| `mlpexp_m1_lofi_fp32_c8192` | `GEMMA4_EXP_MLP_FIDELITY=LoFi GEMMA4_EXP_MLP_FP32=1 GEMMA4_EXP_MLP_PACKER=0` | 07759d549a6 |
| `mlpexp_m3_hifi2_bf16_c8192` | `GEMMA4_EXP_MLP_FIDELITY=HiFi2` | 07759d549a6 |
| `mlpexp_mlpoff_c8192` | `GEMMA4_EXP_MLP_OFF=1` | 07759d549a6 |
| `mlpfp32_256k_c8192` | `GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_DIAG_MLP_FP32ACC=1` | df0d9eaa019 |
| `norm256k_c8192` | `GEMMA4_NORM_SHARD=1` | df0d9eaa019 |
| `norm_only_c4096` | `GEMMA4_NORM_SHARD=1` | 7facef7e68e |
| `norm_only_c8192` | `GEMMA4_NORM_SHARD=1` | 7facef7e68e |
| `packerl1_c2048` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_PACKER_L1=1` | df0d9eaa019 |
| `packerl1_c4096` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_PACKER_L1=1` | df0d9eaa019 |
| `packerl1_c8192` | `GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_PACKER_L1=1` | 218ffa98bc0 |
| `pr2_256k_c2048` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | df0d9eaa019 |
| `pr2_256k_c8192` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | df0d9eaa019 |
| `pr2cfg_c8192` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10` | 7facef7e68e |
| `pr2full_c2048` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | 7facef7e68e |
| `pr2full_c4096` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | 7facef7e68e |
| `pr2full_c8192` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1` | 7facef7e68e |
| `pr2mlpfp32_256k_c2048` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1 GEMMA4_DIAG_MLP_FP32ACC=1` | df0d9eaa019 |
| `pr2mlpfp32_256k_c8192` | `GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1 GEMMA4_DIAG_ATTN_FP32ACC=1 GEMMA4_DIAG_MLP_FP32ACC=1` | df0d9eaa019 |
