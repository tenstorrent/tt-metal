# Ornith-1.5-9B fused decoder

Stage 2 for `ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53` implements both layer kinds in [fused_decoder.py](../../tt/fused_decoder.py). `FusedDecoder` is standalone; its forward path never calls `FunctionalDecoder`. Independent [stage review](stage_review_round1.md) returned **clean-pass** in round 1. Tests default to the fused class and assert its identity. The explicit `--baseline` option is used only for matched controls.

## Results

Serial Blackhole P300c chip 0, batch 1, 1024-token prefill followed by one decode token. Both versions use the same real weights, BF16 activations/weights, HiFi4 matmuls with FP32 accumulation, and FP32 recurrent state. Median of seven warmed prefill samples and seven groups of 50 nonblocking traced decode replays, synchronized once per group. Setup, input refresh, cache restoration and readback are outside timing; watcher/profiler/allocation tracking are disabled for headline timing.

| Layer kind | Functional prefill ms | Fused prefill ms | Speedup | Functional traced decode ms | Fused traced decode ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Linear attention (layer 0) | 13.6156 | 5.8670 | 2.32x | 1.69801 | 1.20754 | 1.41x |
| Full attention (layer 3) | 8.6822 | 6.5334 | 1.33x | 1.89786 | 1.12971 | 1.68x |

Sources: `baseline_{linear,full}_s1024.json`, `default_{linear,full}_s1024.json`, [performance_summary.json](performance_summary.json). Matched length sweeps also cover 128,1023,1025,8192; [candidate_measurements.csv](candidate_measurements.csv) includes the complete candidate history. The final defaults reproduce the fastest correct selected candidates (1.2078 / 1.12978 ms) within timing noise, and beat the functional and interleaved fused controls. Tiny final geometry gains are not claimed as statistically significant.

| Matched HF PCC at S=1024 | Functional prefill | Fused prefill | Functional decode | Fused decode |
|---|---:|---:|---:|---:|
| Linear | 0.99992341 | 0.99992043 | 0.99988717 | 0.99984705 |
| Full | 0.99984235 | 0.99984348 | 0.99989212 | 0.99989337 |

The unchanged acceptance bar is **PCC >=0.995**, including residual-subtracted contribution checks. Fusing convolution/normalization and using the chunk recurrence at decode changes rounding. The direct functional/fused boundary comparison reaches minima 0.99984127 linear and 0.99988186 full; 128 successive distinct decode inputs reach 0.99955308 and 0.99996328. Those matched trajectories also compare final recurrent/KV state. Final HF short/medium/batch whole-output minima are 0.99943870 and 0.99963588 versus prior functional battery minima 0.99968177 and 0.99966115; the fused battery additionally covers lengths 2,3,4. All remain well above the unchanged bar. See `*_paired.json`, `*_trajectory128.json`, `*_final.json` and the long-context evidence below; no quality gate was relaxed.

## Selected graph and configuration

- Pack shared-input QKV/Z/B/A or Q/gate/K/V projections; reorder full-attention Q/gate columns during setup. Use dedicated prefill head splitting and concat, and adapted decode concat.
- Replace primitive temporal convolution with `qkv_causal_conv1d_silu`, channel chunk 128. This kernel needs `packer_l1_acc=False`; matmuls retain their original HiFi4/FP32 policy.
- Feed raw flat Q/K/V into `chunk_gated_delta_rule`, chunk 32, 16 key heads, head-major output. This fuses Q/K normalization, scale and GQA expansion. Both prefill and single-token decode use this path. Inactive tail rows have beta=0 and decay=0, preserving state; history uses only logical input rows.
- Keep GDN output head-major through norm and concatenate heads once. Merge SiLU/sigmoid/softplus into applicable binary inputs.
- Use dedicated partial 64-dimensional RoPE. Decode reads one packed row from a setup-prepared row-major cos/sin table, eliminating two full-table untilizes per token.
- Prefill MLP uses tile-interleaved gate/up weights with `minimal_matmul(fuse_swiglu=True)`; decode uses packed ordinary gate/up matmul and binary SiLU. Both default and adapted minimal-matmul decode variants were slower.
- Four decode projection roles use DRAM-sharded BF16 weights: four readers/bank, 64-core activation/output layout target, `in0_block_w=1`, `per_core_M=1`, `per_core_N` equal to output shard width. A same-precision reader/core/block sweep includes adapted configurations after L1 failures.

[pattern_audit.md](pattern_audit.md) assesses every graph-fusing pattern. [work_log.md](work_log.md) records source snapshots, before/after measurements, initial API failures, adapted retries and rejected options. No optimized-decoder or later-stage implementation is included.

## Runtime and context contract

`FusedDecoder` preserves the functional API: `from_state_dict`, `allocate_state`, `prefill_forward`, `decode_forward`, `prefill_batch`, `decode_batch`. Inputs/outputs are device BF16 `[1,1,S,4096]`. Logical lengths need not divide 32 or 1024. Internal chunking/padding trims outputs and preserves logical state. Non-aligned full-attention continuation decodes to the next aligned boundary before chunked prefill.

The [functional public/trace contract](../functional_decoder/README.md) remains applicable: independent request state; persistent device INT32 positions and page tables; caller-validated bounds; preparation of every physical execution shape before live trace capture; stable buffer addresses and state restoration after warmup/capture. Full-attention BF16 caches remain `[physical_blocks,4,32,256]`, two tensors totaling 1 GiB at context 262144. State allocation requires a multiple of 32, but logical requests do not. Linear state remains FP32 `[1,32,128,128]` plus BF16 three-row convolution history. Ragged batch 32 is a serial per-row API, not dense fused batching.

DRAM sharding adds persistent decode weight copies (438304768 bytes linear; 419430400 bytes full) and full attention adds a 67108864-byte packed RoPE lookup. Prefill retains its interleaved weights; decode traffic does not read both copies. The context contract records this overhead and revalidates the full **262144-token** capability without reduction.

Measured forwards contain no Torch tensor operations or host upload/readback. A runtime guard covers both paths. Remaining device conversions serve actual operator boundaries: tiled three-row convolution history to the causal-convolution row-major input; tiny decode RoPE lookup tilization; full-attention Q/K/V and head-concat sharding; and four interleaved-specialized-op to DRAM-sharded-matmul boundaries. Removing these without adapting consumers is invalid; adapted head-concat and alternative projection layouts were measured. Native reshapes replace unconditional layout round trips. Carrying a broader sharded residual layout is an untried hand-off after the stop rule.

## Evidence and profiler conclusions

| Gate | Artifacts and coverage |
|---|---|
| Fused real-weight correctness | `linear_final.json`, `full_final.json`: lengths 1,2,3,4,31,32,33,1023,1024,1025,2051,17; continuation, page permutations, physical KV comparisons, batch 32 |
| Determinism and trace lifetime | Same final JSON/logs: three bit-equal replays, changed inputs/positions/pages across retained trace requests, prefill before/after capture; exact output-tensor acknowledgment only |
| Allocation tracking | Final and long runs use `TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; `trace_alloc_tracking/positive_control.json` detects a deliberate unrelated live allocation |
| Stress and functional parity | `*_paired.json`: boundary matrix plus eight successive decode tokens; `*_trajectory128.json`: 128 distinct successive tokens with final state comparison |
| Context 262144 | `linear_long.json`, `full_long.json`: complete TTNN prefill at 262139,262143,262144; final-position traced decode. First/middle/terminal HF windows include all preceding history; long PCC is sampled, not every row |
| Watcher | `*_watcher.json`, `watcher_audit.json`, `watcher/*/generated/watcher/watcher.log`: entire short/medium/continuation/batch battery, interval 10, noinline, no checks disabled |
| Delivered pytest | `synthetic_pytest.log`: two layer-kind tests exercise FusedDecoder using deterministic real-weight-statistics fixtures; real-weight gates above are separate |
| Final profiler | `*_profile_final.json`, `tracy_final/{linear,full}/{prefill,decode}_perf_report.{csv,txt}`; advice enabled, separate from watcher |

Final profiler kernel sums (prefill/decode) are 5.408780/1.144266 ms linear and 6.042186/1.093674 ms full. Counts are 27/48 and 33/54 operations. Decode matmul rows verify BF16/HiFi4, DRAM-sharded inputs and block 1. Early full profiling identified 394.394 us of whole-RoPE-table untilize; final profiles remove it. The DRAM-sharded advice led to the selected four-reader implementation. Experimental operation names remain unclassified. The rendered DRAM matmul core counts (8 versus raw 80) and derived FLOP percentages above 100% are reporting limitations; use raw durations and the independently calculated bandwidth roofline, not those derived FLOP fields. These are not device errors.

Same-run decode reconciliation, in milliseconds: linear 1.144266 kernels + 0.041831 gaps + 0.086464 host/instrumentation = 1.272561 single-replay wall; full 1.093674 + 0.046840 + 0.084196 = 1.224710. These instrumented single-replay walls include per-call synchronization/guards and are not substituted for the matched 50-replay amortized headline timings.

Stop Rule 2: at the report's Blackhole peak of 512 GB/s, active padded BF16 projection reads plus recurrent read/write (linear) or 1025-token KV reads (full) give lower bounds 0.864256/0.827400 ms. The final unprofiled warmed trace achieves 71.57%/73.24% of these rooflines. The last two kept same-policy layout refinements each gained <3%. Duplicate prefill weights are excluded from per-token traffic. The instrumented single-call regime reaches only 67.91%/67.56%; the stop-rule claim is specifically for the requested warmed traced-decode regime.

## Reproduction and limits

From the repository root, use the active `python_env`, `HF_HOME=/home/ttuser/ornith-9b/hf`, and the installed model-bringup `scripts/environment.py` exports. Retain topology discovery; use one mesh on physical chip 0. Serialize device jobs and run watcher separately from profiler.

```bash
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_fused_evidence.sh
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_fused_contracts.sh
python models/autoports/ornith_ai_ornith_1_5_9b/tests/summarize_fused.py
```

The scripts specify per-command timeouts, exact runner flags and artifact destinations. Driver logs record completion. Candidate commands/configurations are in `run_fusion_candidates.sh`, `run_dram_candidates.sh`, individual JSON/log pairs and the work log. Historical candidates retain source hashes/snapshots; only final artifacts are tied to delivered source SHA256 `8f094843786f8f388b241449c0322a684fc398244500274f34ae3862fb07cc10`.

Large raw artifacts remain locally; lossless `.gz` archives and `archive_manifest.json` preserve hashes for committed logs and postprocessed CSVs. Raw Tracy captures/device event streams are local-only with manifest hashes. This is layer-level validation, not full-model text quality or serving readiness. Long-context agreement is sampled with complete history. The 128-token stress compares matched functional/fused trajectories; no model-level accumulation or generation claim is made. Untried post-stop candidates are explicit in the work log.
