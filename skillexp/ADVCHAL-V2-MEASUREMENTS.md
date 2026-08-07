# advchal-v2 — complete measurement ledger

Every harness measurement recorded by every cell of stage 02b (`$advisor-challenger`), in the order
the cell ran it, reconstructed from the session transcripts
(`~/skillexp-logs/p-advchal-v2-*/02-02b-advisor-challenger.jsonl`).

All 15 cells were measured on the **same host** (`machine=a`, `qb2-120-p05t03`); the `-onA` suffix in an
arm name refers to where that cell's *incumbent* was produced by stage 02, not where 02b ran. So no
cross-cell difference below is a hardware difference.

This is the **detail tier**. For the readable account of what each cell was doing and why, read
[`ADVCHAL-V2-READ-THIS.md`](ADVCHAL-V2-READ-THIS.md).

⚠ **Scope: "complete" means every measurement *the cells* took.** It does not include my own follow-up
measurements, which are numerous and are what several later findings rest on — the E1–E8 experiments, the
E9–E27 counterfactuals, the isolated single-op legality tests, the oracle sweeps, and the four `ttnn-advise`
re-runs. Those live in [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md),
[`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) and
[`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md), with raw logs under
`~/skillexp-logs/exp-advisor-probe/`, `exp-rope-faithful/` and `exp-readvise/`. If you are checking whether a
number in this corpus was measured, check there too before concluding it was not.

Every row is one invocation of the mandated harness: **≥10 untimed warm-ups, then n=5 timed blocks,
each block the mean of ≥50 traced decode replays**; `ms` is the median of the 5 blocks and `floor`
is their max−min spread. A candidate may only ship if *every* one of its 5 blocks beats *every* one
of the control's 5 blocks (the non-overlap rule) — so a better median alone is not enough.

`vs ctl` is measured against the first control of the same layer kind, and is meaningless where a
cell has several layer kinds with different costs (marked ⚠). Read those against the per-kind control.

**149 measurements across 15 cells.**


## llama-3.2-1B — llama-3.2-1B, `exp17`

Control: **0.3731 ms/layer**, noise floor 0.146 µs (0.039 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.3731 | +0.00 % | 0.146 | 0.039 % |
| 2 | `incumbent_profile` | 0.3847 | +3.12 % | 5.254 | 1.366 % |
| 3 | `dense_2_residual_chain_64` | 0.4203 | +12.67 % | 0.605 | 0.144 % |
| 4 | `concat_output_dram` | 0.3732 | +0.04 % | 6.501 | 1.742 % |

> **Cell's own closing summary:** Full-model estimate: **5,672.576 µs before → 5,672.576 µs after, ±2.336 µs**. Measured `$shard-advise` contribution: **0.000 µs/model**. The frozen incumbent remains shipped unchanged. - Incumbent: 0.373080 ms/layer at decode batch 32. - 64-core extended chain: 0.420341 ms, 12.7% slower. - DRAM concat candidate: 0.373237 ms with overlapping repeats; rejected. - SDPA→concat chain: infeasible because paged GQA SDPA rejects sharded output. - Rotary-K DRAM candidate: incompatible with paged cache update. - Advisor agreement with shipped: 258.260 µs/layer, 72.84% of the profiled window. - Advisor-agreed boundaries: 1.320 µs/layer, reported but not credited. - Layer handoff: no DRAM round trip detected. Artifacts: [README.md](/home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_2_1b_instruct/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_2_1b_instruct/doc/advisor_challenger/final.json). The requested gate passes in strict mode with no warnings. Candidate knobs remain default-off.


## llama-3.1-8B — llama-3.1-8B, `exp17`

Control: **0.6650 ms/layer**, noise floor 0.697 µs (0.105 % of the layer).

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.6650 | +0.00 % | 0.697 | 0.105 % |
| 2 | `profile_incumbent` | 0.6767 | +1.75 % | 5.321 | 0.786 % |
| 3 | `profile_incumbent` | 0.6769 | +1.78 % | 5.586 | 0.825 % |
| 4 | `profile_incumbent` | 0.6776 | +1.89 % | 5.109 | 0.754 % |
| 5 | `dense_2_skip_attn_output_reshard` | 0.6676 | +0.38 % | 0.376 | 0.056 % |
| 6 | `dense_3_skip_mlp_output_reshard` | 0.6677 | +0.40 % | 0.374 | 0.056 % |
| 7 | `dense_geometry_64` | 0.6931 | +4.22 % | 0.608 | 0.088 % |

> **Cell's own closing summary:** Full-model estimate: **20,747.296 µs before → 20,747.296 µs after, ±22.304 µs**. Measured `$shard-advise` contribution: **0.000 µs/model**. The frozen batch-32 incumbent remains shipped unchanged. Key results: - Incumbent: 0.665046 ms/layer; 0.697 µs noise floor. - Advisor ceiling: 4.394 µs/layer, or 140.608 µs/model. - Top boundary candidate: hard error—concat-heads requires sharded input. - Attention regrid removal: +2.536 µs/layer. - MLP regrid removal: +2.635 µs/layer. - Exactly-dividing 64-core candidate: +28.077 µs/layer. - Advisor independently re-derived 76.32% of the measured window. - Strict advisor-challenger gate passes with no warnings. Artifacts are summarized in [README.md](/home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_1_8b_instruct/doc/advisor_challenger/README.md), with machine-readable results in [final.json](/home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_1_8b_instruct/doc/advisor_challenger/final.json). All losing experimental knobs remain default-off.


## phi A — phi-3.5-mini, `nofuse-noadvise-onA`

Control: **0.6570 ms/layer**, noise floor 0.430 µs (0.065 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.6570 | +0.00 % | 0.430 | 0.065 % |
| 2 | `incumbent_profile` | 0.6650 | +1.22 % | 1.388 | 0.209 % |
| 3 | `rope_l1_rect32` | **0.6072** | -7.58 % | 1.860 | 0.306 % |
| 4 | `rope_l1_rect32_confirm` | **0.6079** | -7.47 % | 0.518 | 0.085 % |

> **Cell's own closing summary:** Shipped the advisor-derived L1 RoPE chain for decode batch 32. - Full-model estimate: **18,210.88 → 16,616.75 µs** - Improvement: **1,594.13 µs/model (8.75%)** - Uncertainty band: **±13.76 µs** - Per-layer: **0.656989 → 0.607172 ms** - Fresh-process confirmation: **0.607902 ms** - Real-weight differential oracle: **PCC 0.9999988** - Batch-32 regression test: **passed**, PCC 0.9999951 - Advisor agreement: **224.287 µs/layer** - Out-of-scope layer handoff: **43.369 µs/model** - Mandatory advisor-challenger gate: **passed** Artifacts and detailed accounting are in [README.md](/home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/final.json). The shipped implementation is in [optimized_decoder.py](/home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/tt/optimized_decoder.py).


## phi B — phi-3.5-mini, `nofuse-noadvise`

Control: **0.7888 ms/layer**, noise floor 0.433 µs (0.055 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.7888 | +0.00 % | 0.433 | 0.055 % |
| 2 | `incumbent_profile` | 0.7987 | +1.25 % | 3.173 | 0.397 % |
| 3 | `rope_l1_chain` | **0.7487** | -5.09 % | 0.196 | 0.026 % |
| 4 | `rope_l1_chain_confirm` | **0.7485** | -5.12 % | 0.524 | 0.070 % |
| 5 | `incumbent` | 0.7894 | +0.07 % | 5.567 | 0.705 % |
| 6 | `incumbent` | 0.7886 | -0.03 % | 0.713 | 0.090 % |

> **Cell's own closing summary:** Shipped the advisor-derived RoPE L1-residency chain. - Full-model estimate: **22,392.6 → 21,107.7 µs**, uncertainty **±22.8 µs** - Improvement: **1,284.9 µs/model (5.74%)** - Per-layer: **0.788610 → 0.748458 ms** - All candidate repeats beat all incumbent repeats, including fresh-process confirmation - Real-weight oracle: **PCC 0.998993**, threshold 0.995 - Advisor pin: `618cd4e75d` - Decode/capture batch: 32 - Layer count: `dense: 32` The unrelated down-projection tuning was excluded from advisor attribution. Layer-handoff cost, advisor-agreed boundaries, and tracer-unreachable fused-cache time are reported without screening or attribution. Artifacts: [README.md](/home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/final.json). The required `02b-advisor-challenger.check.sh` gate passes cleanly with no warnings.


## phi FN — phi-3.5-mini, `fuse-noadvise`

Control: **0.8072 ms/layer**, noise floor 1.064 µs (0.132 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.8072 | +0.00 % | 1.064 | 0.132 % |
| 2 | `incumbent_profile` | 0.8154 | +1.03 % | 1.271 | 0.156 % |
| 3 | `incumbent_profile` | 0.8196 | +1.54 % | 1.532 | 0.187 % |
| 4 | `rope_l1_query` | **0.7934** | -1.70 % | 1.680 | 0.212 % |
| 5 | `rope_l1_key` | **0.7926** | -1.80 % | 1.013 | 0.128 % |
| 6 | `rope_l1_query_key` | **0.7681** | -4.84 % | 1.105 | 0.144 % |
| 7 | `rope_l1_query_key_confirm` | **0.7675** | -4.91 % | 1.198 | 0.156 % |
| 8 | `norm_11c` | **0.7459** | -7.59 % | 1.527 | 0.205 % |
| 9 | `norm_12c` | **0.7490** | -7.20 % | 2.810 | 0.375 % |
| 10 | `norm_24c` | **0.7485** | -7.26 % | 5.680 | 0.759 % |
| 11 | `rope_l1_query_key_norm_11c` | **0.7003** | -13.24 % | 0.830 | 0.119 % |
| 12 | `rope_l1_query_key_profile` | **0.7710** | -4.48 % | 1.665 | 0.216 % |

Environment knobs swept by this cell:

- `CHALLENGER_ADVISOR_NORM_CORES` = `11`, `\""'<swept>`
- `CHALLENGER_ADVISOR_ROPE_L1` = `\""'$candidate`, `query_key`
- `CHALLENGER_ADVISOR_SDPA_CONCAT_L1` = `1`

## phi exp17 — phi-3.5-mini, `exp17`

Control: **1.1009 ms/layer**, noise floor 1.092 µs (0.099 % of the layer).

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.1009 | +0.00 % | 1.092 | 0.099 % |
| 2 | `profile_incumbent` | 1.1080 | +0.64 % | 2.593 | 0.234 % |
| 3 | `rope_l1_tail` | **1.1007** | -0.02 % | 1.642 | 0.149 % |

> **Cell's own closing summary:** The audit found one important weakness in the earlier completion claim: the stage ships batch-32 enablement as a prerequisite, so the blanket “a shipped change needs a real-weight oracle” applies even though the advisor contribution itself is zero. The recorded real-weight test was skipped because its pinned cache path was absent. I’m checking for equivalent cached weights elsewhere and will run a real batch-32 oracle if available; otherwise this cannot honestly be marked complete yet.


## qwen FN — qwen3.6-27B, `fuse-noadvise`

Control: **19.1402 ms/layer**, noise floor 12.349 µs (0.065 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 19.1402 | +0.00 % | 12.349 | 0.065 % |
| 2 | `incumbent_full_attention` | 1.2083 | -93.69 % | 1.609 | 0.133 % |
| 3 | `packed_qkv_l1_chain` | **1.1804** | -93.83 % | 1.155 | 0.098 % |
| 4 | `packed_qkv_l1_profile` | **1.1926** | -93.77 % | 3.061 | 0.257 % |
| 5 | `packed_qkv_l1_confirm` | **1.1813** | -93.83 % | 0.911 | 0.077 % |
| 6 | `incumbent_full_profile_current` | 1.2174 | -93.64 % | 1.482 | 0.122 % |
| 7 | `rope_query_c32` | **1.2225** | -93.61 % | 0.886 | 0.072 % |
| 8 | `rope_both_c32` | **1.2332** | -93.56 % | 3.957 | 0.321 % |
| 9 | `packed_qkv_l1_shipped_confirm` | **1.1811** | -93.83 % | 1.707 | 0.145 % |

> **Cell's own closing summary:** Shipped the measured packed-QKV L1 chain in [optimized_decoder.py](/home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:417). Full-model estimate: **938,063.85 µs → 937,618.16 µs, ±618.50 µs**. - Estimated contribution: **−445.69 µs/model**, inside its uncertainty band. - Full-attention layer: **1.208257 → 1.180402 ms** (`−2.31%`). - All 5 candidate repeats beat all 5 incumbent repeats. - Fresh-process confirmation: **1.181348 ms**. - Real checkpoint differential oracle: **PCC 1.0**. - Linear attention: measured contribution **zero** across 48/64 layers because the pinned tracer cannot cross its mutable-state `ttnn.copy` boundary. *(Since fixed at the tracer — see [`BLOCKER-AUDIT`](ADVCHAL-V2-BLOCKER-AUDIT.md) §5; a pipeline abort blocks it now, not coverage.)* - Advisor independently matched **67.67%** of the existing full-attention window. Artifacts and detailed accounting are in [README.md](/home/mvasiljevic/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/advisor_challenger/README.md). The advisor-challenger gate passes, and all six optimized-decoder unit tests pass.


## qwen B — qwen3.6-27B, `nofuse-noadvise`

Control: **1.4494 ms/layer**, noise floor 0.749 µs (0.052 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.4494 | +0.00 % | 0.749 | 0.052 % |
| 2 | `incumbent_linear_attention` | 15.8498 | +993.53 % | 5.887 | 0.037 % |
| 3 | `incumbent_profile` | 1.4579 | +0.59 % | 2.512 | 0.172 % |
| 4 | `incumbent_profile_trace` | 1.4523 | +0.20 % | 0.376 | 0.026 % |
| 5 | `advisor_rope_q_l1` | 1.4544 | +0.35 % | 2.873 | 0.198 % |
| 6 | `advisor_rope_k_l1` | 1.4536 | +0.29 % | 3.857 | 0.265 % |
| 7 | `advisor_qkv_direct` | 1.4501 | +0.05 % | 0.745 | 0.051 % |
| 8 | `advisor_rope_dram` | 1.4516 | +0.15 % | 3.103 | 0.214 % |
| 9 | `incumbent_linear_attention` | 15.8526 | +993.72 % | 5.795 | 0.037 % |

> **Cell's own closing summary:** I’m reopening this as incomplete. The current artifacts pass the non-strict gate, but they do not yet prove the requested contribution measurement: candidate chains were relabeled in the reconciliation JSON instead of measured, the linear kind lacks a successful reconciliation, and the headline mixes harness-derived estimates instead of the prescribed per-kind reconciliation method. I’ll correct those rather than treating the current green gate as sufficient.


## gemma-4-12B — gemma-4-12B, `exp11`

Control: **1.2541 ms/layer**, noise floor 1.727 µs (0.138 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.2541 | +0.00 % | 1.727 | 0.138 % |
| 2 | `incumbent_full_attention` | 1.3774 | +9.82 % | 5.056 | 0.367 % |
| 3 | `sliding_b17_cos_rope_row_major` | **1.2488** | -0.42 % | 4.621 | 0.370 % |
| 4 | `sliding_b19_sin_rope_row_major` | **1.2474** | -0.54 % | 0.788 | 0.063 % |
| 5 | `sliding_b17_b19_rope_row_major` | **1.2413** | -1.02 % | 2.842 | 0.229 % |
| 6 | `full_b17_cos_rope_row_major` | 1.3662 | +8.94 % | 1.124 | 0.082 % |
| 7 | `full_b19_sin_rope_row_major` | 1.3656 | +8.89 % | 0.287 | 0.021 % |
| 8 | `full_b17_b19_rope_row_major` | 1.3558 | +8.10 % | 1.157 | 0.085 % |
| 9 | `incumbent` | 1.2415 | -1.01 % | 0.712 | 0.057 % |
| 10 | `incumbent_full_attention` | 1.3559 | +8.11 % | 3.381 | 0.249 % |
| 11 | `sliding_keep_k_l1_chain` | **1.2308** | -1.86 % | 0.802 | 0.065 % |
| 12 | `full_keep_k_l1_chain` | 1.3359 | +6.52 % | 2.053 | 0.154 % |
| 13 | `confirm_shipped_sliding` | **1.2305** | -1.89 % | 5.255 | 0.427 % |
| 14 | `confirm_shipped_full` | 1.3351 | +6.45 % | 4.177 | 0.313 % |
| 15 | `shipped_l1_interleaved_sliding` | **1.2288** | -2.02 % | 0.544 | 0.044 % |
| 16 | `shipped_l1_interleaved_full` | 1.3359 | +6.52 % | 5.253 | 0.393 % |
| 17 | `sliding_keep_v_l1` | **1.2322** | -1.75 % | 0.651 | 0.053 % |
| 18 | `sliding_mlp_direct_down` | **1.2352** | -1.51 % | 5.614 | 0.455 % |
| 19 | `sliding_k_v_mlp` | **1.2238** | -2.42 % | 5.048 | 0.412 % |
| 20 | `full_k_v_mlp` | 1.3198 | +5.23 % | 0.793 | 0.060 % |
| 21 | `profile_final_sliding` | 1.2383 | -1.26 % | 5.870 | 0.474 % |
| 22 | `sliding_q_l1_extended_sdpa` | **1.2342** | -1.59 % | 0.949 | 0.077 % |
| 23 | `full_q_l1_extended_sdpa` | 1.3428 | +7.07 % | 1.178 | 0.088 % |
| 24 | `sliding_q_k_v_mlp` | **1.2188** | -2.82 % | 1.765 | 0.145 % |
| 25 | `full_q_k_v_mlp` | 1.3080 | +4.29 % | 2.962 | 0.226 % |
| 26 | `sliding_grouped_o_l1` | **1.2413** | -1.03 % | 0.778 | 0.063 % |
| 27 | `full_grouped_o_l1` | 1.3541 | +7.97 % | 0.487 | 0.036 % |
| 28 | `full_q_k_v_mlp_o` | 1.3060 | +4.14 % | 1.372 | 0.105 % |

> **Cell's own closing summary:** The skill audit exposed two evidence weaknesses that the gate alone does not catch: generated reconciliation rows were post-annotated with sentinel “measurements” for some unexecuted chains, and the kept candidate lacks its own op-level profile. Those do not meet the contribution method. I’m treating the previous “passed” result as provisional and correcting these gaps rather than leaning on the checker.


## g26 B — gemma-4-26B, `nofuse-noadvise`

Control: **1.2597 ms/layer**, noise floor 2.814 µs (0.223 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.2597 | +0.00 % | 2.814 | 0.223 % |
| 2 | `incumbent_full_attention` | 1.2617 | +0.16 % | 0.852 | 0.068 % |
| 3 | `profile_sliding` | 1.2939 | +2.72 % | 4.648 | 0.359 % |
| 4 | `profile_full` | 1.2965 | +2.92 % | 1.983 | 0.153 % |
| 5 | `profile_sliding_trace` | 1.2666 | +0.55 % | 13.159 | 1.039 % |
| 6 | `profile_sliding_bounded` | 1.2966 | +2.93 % | 5.310 | 0.410 % |
| 7 | `profile_full_bounded` | 1.2971 | +2.97 % | 5.735 | 0.442 % |
| 8 | `sliding_attention_o_chain` | **1.2540** | -0.46 % | 1.264 | 0.101 % |
| 9 | `full_attention_qkv_chain` | 1.2906 | +2.46 % | 1.387 | 0.107 % |
| 10 | `sliding_attention_o_chain_confirm` | **1.2536** | -0.48 % | 0.602 | 0.048 % |
| 11 | `profile_candidate_sliding_o` | 1.2909 | +2.48 % | 3.246 | 0.251 % |

> **Cell's own closing summary:** Shipped the advisor-measured sliding-attention O-projection DRAM sharding. - Full-model estimate: **36,224.1 ± 36.5 µs → 36,076.2 ± 36.5 µs** - Estimated contribution: **−147.9 µs/model** - Sliding incumbent: 1.258866–1.260157 ms - Winner: 1.253223–1.254487 ms - Fresh confirmation: 1.253043–1.253644 ms - Real-weight PCC: 0.998358 prefill, 0.999499 decode; threshold 0.995 - Full-attention QKV candidate rejected: 1.290230–1.291618 ms - All batches were decode/capture/requested batch 1. - Strict advisor-challenger gate passes without warnings. Artifacts: [advisor_challenger README](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/final.json).


## g26 onA — gemma-4-26B, `nofuse-noadvise-onA`

Control: **1.8252 ms/layer**, noise floor 3.349 µs (0.183 % of the layer).

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.8252 | +0.00 % | 3.349 | 0.183 % |
| 2 | `incumbent` | 2.0132 | +10.30 % | 2.939 | 0.146 % |
| 3 | `incumbent` | 2.0139 | +10.34 % | 3.543 | 0.176 % |
| 4 | `incumbent` | 1.8240 | -0.06 % | 0.587 | 0.032 % |
| 5 | `advisor_norm88_sliding` | **1.5873** | -13.03 % | 1.766 | 0.111 % |
| 6 | `advisor_norm88_full` | **1.7767** | -2.66 % | 1.988 | 0.112 % |
| 7 | `advisor_norm88_sliding_confirm` | **1.5874** | -13.03 % | 3.467 | 0.218 % |
| 8 | `advisor_norm88_full_confirm` | **1.7777** | -2.60 % | 1.370 | 0.077 % |
| 9 | `advisor_norm88_sliding_profile` | **1.6357** | -10.38 % | 25.426 | 1.554 % |
| 10 | `advisor_norm88_full_profile` | **1.8228** | -0.13 % | 26.299 | 1.443 % |
| 11 | `advisor_norm88_sliding_shipped_default` | **1.5987** | -12.41 % | 13.301 | 0.832 % |
| 12 | `advisor_norm88_full_shipped_default` | **1.7765** | -2.66 % | 1.400 | 0.079 % |

> **Cell's own closing summary:** Full-model estimate: **54,633.6 ± 32.4 µs before and after**. `shard-advise` contribution is **0.0 µs (0.0%)** at decode batch 1. The frozen incumbent remains shipped unchanged. Key evidence: - Layer counts: 25 sliding attention + 5 full attention. - Both reconciliations close at 100%, are not degraded, and report a 0.000 µs advisor-attributable ceiling. - Sparse experts remain unreachable: 64.70% of the sliding window and 58.51% of the full-attention window. - Advisor-agreed conversion costs—7.882 µs/layer sliding and 8.826 µs/layer full—were reported but not credited. - Pinned advisor: `618cd4e75d`. - Required gate passes with advisory notices for out-of-scope norm/grid and DRAM-resident optimization opportunities. Artifacts: [advisor_challenger/README.md](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/final.json).


## g26 FN — gemma-4-26B, `fuse-noadvise`

Control: **1.3412 ms/layer**, noise floor 2.804 µs (0.209 % of the layer).

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 1.3412 | +0.00 % | 2.804 | 0.209 % |
| 2 | `incumbent` | 1.5394 | +14.78 % | 2.157 | 0.140 % |
| 3 | `advisor_norm88_sliding` | 1.3469 | +0.43 % | 1.476 | 0.110 % |
| 4 | `advisor_norm88_full` | 1.5499 | +15.57 % | 3.405 | 0.220 % |
| 5 | `advisor_concat_projection_sliding` | **1.3184** | -1.69 % | 1.217 | 0.092 % |
| 6 | `advisor_concat_projection_full` | 1.4945 | +11.44 % | 0.773 | 0.052 % |
| 7 | `advisor_concat_projection_sliding_confirm` | **1.3182** | -1.71 % | 0.864 | 0.066 % |
| 8 | `advisor_concat_projection_full_confirm` | 1.4946 | +11.44 % | 0.665 | 0.044 % |
| 9 | `shipped_profile_sliding` | 1.3518 | +0.79 % | 7.658 | 0.567 % |
| 10 | `shipped_profile_full` | 1.5283 | +13.95 % | 3.143 | 0.206 % |
| 11 | `shipped_default_sliding` | **1.3191** | -1.65 % | 0.573 | 0.043 % |
| 12 | `shipped_default_full` | 1.4947 | +11.45 % | 0.992 | 0.066 % |

> **Cell's own closing summary:** Shipped the advisor-derived concat-heads → output-projection sharded chain. Full-model estimate: **38,887.6 ± 80.9 µs before → 38,095.8 ± 80.9 µs after** - Estimated contribution: **791.7 µs/model (2.04%)** - Sliding: 1.341153 → 1.318449 ms - Full attention: 1.539374 → 1.494548 ms - Every winning repeat beat every incumbent repeat. - Fresh-process confirmations passed. - Real-weight HuggingFace oracle passed above PCC 0.995. - The 88-core norm candidate regressed both layer kinds and remains default-off. - Sharded SDPA output was rejected with the hard GQA constraint. - Final advisor-challenger gate passes cleanly. Artifacts: [advisor_challenger/README.md](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/README.md) and [final.json](/home/mvasiljevic/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/advisor_challenger/final.json).


## nm FN — north-mini, `fuse-noadvise`

Control: **0.1727 ms/layer**, noise floor 0.849 µs (0.492 % of the layer).

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.1727 | +0.00 % | 0.849 | 0.492 % |
| 2 | `incumbent` | 0.5781 | +234.77 % | 1.638 | 0.283 % |
| 3 | `incumbent` | 0.5537 | +220.63 % | 1.206 | 0.218 % |
| 4 | `advisor_moe_norm_22` | 0.5433 | +214.64 % | 1.750 | 0.322 % |
| 5 | `advisor_moe_norm_32` | 0.5187 | +200.40 % | 2.094 | 0.404 % |
| 6 | `advisor_moe_norm_64` | 0.5733 | +232.00 % | 0.922 | 0.161 % |
| 7 | `advisor_moe_norm_22` | 0.5444 | +215.26 % | 1.302 | 0.239 % |
| 8 | `advisor_moe_norm_32` | 0.5197 | +200.97 % | 1.236 | 0.238 % |
| 9 | `advisor_moe_norm_64` | 0.5751 | +233.05 % | 0.637 | 0.111 % |
| 10 | `confirm_advisor_moe_norm_32` | 0.5183 | +200.13 % | 0.685 | 0.132 % |
| 11 | `confirm_advisor_moe_norm_32` | 0.5199 | +201.09 % | 0.765 | 0.147 % |
| 12 | `advisor_moe_norm_22` | 0.5429 | +214.38 % | 0.723 | 0.133 % |
| 13 | `advisor_moe_norm_32` | 0.5183 | +200.17 % | 0.660 | 0.127 % |
| 14 | `advisor_moe_norm_64` | 0.5735 | +232.10 % | 0.522 | 0.091 % |
| 15 | `advisor_moe_norm_22` | 0.5465 | +216.51 % | 2.630 | 0.481 % |
| 16 | `advisor_moe_norm_32` | 0.5201 | +201.17 % | 0.132 | 0.025 % |
| 17 | `advisor_moe_norm_64` | 0.5753 | +233.20 % | 0.682 | 0.119 % |
| 18 | `confirm_advisor_moe_norm_32` | 0.5190 | +200.55 % | 0.730 | 0.141 % |
| 19 | `confirm_advisor_moe_norm_32` | 0.5202 | +201.25 % | 0.333 | 0.064 % |

> **Cell's own closing summary:** # Stage Review Verdict: clean-pass ## Required Work - None. ## Other Concerns - None material. ## Hard-Check Gaps - No standalone gate script exists in the advisor-challenger skill. I independently checked the relevant contracts and arithmetic from the artifacts. ## Anomaly Ledger - Observed anomaly: One-core QKV-head work and one-core MoE RMSNorm could previously be read as having the same advisor attribution. Evidence: The README now states that `nlp_create_qkv_heads_decode` agrees with the advisor and is handed to `$optimize`, while the advisor changes MoE RMSNorm from one core to 22-core width sharding. Affected path: Advisor-contribution accounting. Control or comparison: Advisor reports retain QKV heads on one core; reconciliation identifies RMSNorm as `material_ops_on_le_2_cores`, with the measured 22/32/64 sweep. Likely subsystem: Documentation/accounting only. Investigation performed: Compared README claims with reconciliation JSON, advisor reports/IR, decision JSON, measurement records, and implementation diff. Resolution: fixed. - Observed anomaly: Full-attention MoE layer-4 weights were unavailable locally. Evidence: The oracle uses official layer-1 tensors remapped to 


## nm B — north-mini, `nofuse-noadvise`

Control: **0.6138 ms/layer**, noise floor 0.936 µs (0.152 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.6138 | +0.00 % | 0.936 | 0.152 % |
| 2 | `incumbent_dense_full_forced_rope` | 0.2033 | -66.88 % | 0.168 | 0.083 % |
| 3 | `incumbent_full_no_rope_moe` | 0.6122 | -0.27 % | 1.263 | 0.206 % |
| 4 | `profile_dense` | 0.2156 | -64.87 % | 5.807 | 2.693 % |
| 5 | `dense_advised_down_ds` | **0.2074** | -66.21 % | 1.489 | 0.718 % |
| 6 | `sliding_advice_aggregate_noop` | **0.6138** | -0.01 % | 0.581 | 0.095 % |
| 7 | `47:advisor_dense_chain_exact` | **0.2341** | -61.87 % | 0.416 | 0.178 % |
| 8 | `postreset_default_verification` | **0.2030** | -66.93 % | 1.458 | 0.718 % |

> **Cell's own closing summary:** The audit found one remaining measurement gap: the full DRAM-sharded chain stalled, so I’m isolating it with the already-measurable 32-core down projection after a device-health smoke. Then I’ll regenerate the reconciliations and final zero-contribution report from the complete evidence set.


## nm onA — north-mini, `nofuse-noadvise-onA`

Control: **0.2918 ms/layer**, noise floor 1.841 µs (0.631 % of the layer). ⚠ multiple layer kinds — `vs ctl` spans kinds, compare within a kind only

| # | measurement | ms/layer | vs ctl | floor µs | floor as % of layer |
|---|---|---|---|---|---|
| 1 | `incumbent` | 0.2918 | +0.00 % | 1.841 | 0.631 % |
| 2 | `incumbent_sliding_attention_sparse_moe` | 0.8465 | +190.06 % | 14.526 | 1.716 % |
| 3 | `incumbent_full_attention_sparse_moe` | 0.8445 | +189.35 % | 0.847 | 0.100 % |
| 4 | `profile_dense_full_attention` | 0.3002 | +2.87 % | 5.800 | 1.932 % |
| 5 | `profile_dense_full_attention` | 0.3001 | +2.83 % | 6.338 | 2.112 % |

> **Cell's own closing summary:** Measured `$shard-advise` contribution: **0 µs**. The incumbent decoder remains unchanged. Full-model decode estimate at batch 1: - Before: **39,940.817 µs** - After: **39,940.817 µs** - Uncertainty: **±534.941 µs** All three layer kinds were `not_measurable`; their complete advisor-attributable ceilings were below their frozen-control noise floors, so the method prohibited candidate screening. Artifacts: - [README.md](/home/mvasiljevic/tt-metal/models/autoports/coherelabs_north_mini_code_1_0/doc/advisor_challenger/README.md) - [final.json](/home/mvasiljevic/tt-metal/models/autoports/coherelabs_north_mini_code_1_0/doc/advisor_challenger/final.json) - [incumbent.json](/home/mvasiljevic/tt-metal/models/autoports/coherelabs_north_mini_code_1_0/doc/advisor_challenger/incumbent.json) Validation completed: - Advisor pin: `618cd4e75d` - Reconciliation self-tests: passed - Requested gate: passed - Strict gate: passed - Python compilation and `git diff --check`: passed I also corrected reconciliation/gate handling so `not_measurable` cells are recorded as terminal decisions without contradictory screening warnings.

