# Full-model performance and terminal decisions

Final unprofiled confirmation after the NoC repair is `readiness_confirmed.json`
with clean process/launcher exit 0: S128/G128, B1, TP4, warmed TTFT 97.244 ms
and 39.055 t/s/user (25.605 ms/token); AIME S203/G100 teacher forcing
99.557 ms and 38.760 t/s/user. The earlier `readiness_final.json` measured
88.561 ms / 39.079 t/s/user; the reproduced result above is the headline,
including the higher observed TTFT. Both use separate model/sampling traces. Token-out
has no steady token/position/RoPE/page-table uploads or full-logit readback;
teacher forcing intentionally uploads the reference next token. First-token
prefill and sampling, request reset and parameter updates are included in TTFT.
Weight/tokenizer/RoPE-table setup and initial cold trace capture are excluded
by reporting the second identical request.

## Candidate evidence

The reduced wrapper retains real layers0/3, real embeddings/final norm/head,
TP4 geometry, CCL, cache types and split traces. S128/G128 measurements:

| Candidate | Tokens/s | Artifact |
| --- | ---: | --- |
| Interleaved head, split greedy k1/p0/T1 | 400.296 | split_baseline.json |
| Same head, common force argmax k1/p0/T1 | 226.638 | argmax_baseline.json |
| DRAM head block10 | 415.688 | dram_probe_block10.json |
| DRAM head block10 + sharded final norm | 433.093 | dram_norm_probe.json |
| Selected terminal repeated with interleaved logit oracle | 434.091 | dram_compare.json |

The DRAM block20 family was adapted after its3456-byte L1 overlap
(exact addresses in `dram_probe.log`); block10 is legal. Selected greedy output
matches the baseline for128 tokens and final-logit PCC is .9999766. This reduced
output is a pipeline probe, not meaningful text; full-model HF gates validate the
selected terminal with all layers. The split sampler is substantially faster than
the force-argmax alternative. There is no sampling mode that silently switches
greedy to non-greedy for the comparison.

## Reduced device profiling

No all-layer Tracy/device-profiler run was performed. `profile_split` failed
postprocessing after dropped markers and is excluded. `profile_split_drained`
, `profile_selected`, and post-repair `profile_final` drain buffers between setup/warmup and each signposted
prefill, model, sampling and token-out window. Advice-enabled tables are per-device;
cross-device timestamp gaps are not summed. Compressed raw ops CSVs preserve the
input to `tt-perf-report`; bulky raw captures are archived by exact path/hash in
`profile_archive.json`. `profile_summary.json` uses **microseconds**, including
separate device time and op gaps.

The final post-repair device0 model window is 1.8225 ms, sampler 0.4897 ms
and token-out 2.3105 ms (`tracy/profile_final`).
Sampling is28 device ops; local top-k is about0.31ms, candidate gathers about0.013ms,
and native seeded sampling plus tie handling makes up the rest. Sampling is below
2% of unprofiled full-stack token-out time, so it does not dominate decode.

Baseline final norm95us becomes6us on40 cores. The baseline interleaved
62080-column head costs996us and is classified `SLOW`; the selected eight
DRAM-sharded chunks cost about 120–124 us each (tail 75 us) and are classified `BOTH` or
`FLOP`. Actual rows show BF8/HiFi2 head and BFP4/LoFi decoder matmuls.

## Advice disposition and boundaries

- **LM-head input in L1 / small output subblocks:** implemented as a coherent
  sharded-norm, L1-input, bank-sharded-weight DRAM family with native program
  packing. The first L1 error was adapted, then correctness and latency compared.
  The selected program is compute/bandwidth-bound in the report.
- **Higher fidelity for non-FLOP-bound matmuls:** this conditional advice came
  from the baseline `SLOW` head. The selected head is now `BOTH`/`FLOP`, so that
  condition no longer applies. It uses BF16 activations, BF8 weights, HiFi2 with
  FP32 accumulation, and passes HF top5/top100 gates. Decoder policy is unchanged.
- **Tracing suggested for large gaps:** both measured decode graphs are already
  traced. The model-entry embedding/gather and sampling RNG gaps remain named
  terminal costs; they are not hidden host token feedback. Separate trace overhead
  is small in the combined token-out window.
- **Embedding gather:** a single BF16 hidden-column all-gather at model entry is
  about 14 us device time in the selected report. It does not recur between layers.
  A vocabulary lookup/reduce topology would change the entry ownership contract
  to target less than0.1% of full decode, so no performance win is claimed for it.
- **Inter-layer conversions:** none added. Layout conversions shown inside each
  decoder are the inherited selected implementation, with the Stage5 ledger intact.
- **Prefill timing anomaly:** selected profiled prefill is17.16ms versus6.38ms
  baseline, driven by14.0ms versus3.1ms of host gaps. That capture overlapped the
  HF CPU-control job. Unprofiled reduced TTFT is6.95ms versus7.28ms, and final
  full-stack TTFT is measured independently without profiler/watcher/tracker.
  The final capture, with the CPU control paused, measures 8.917 ms prefill
  (3.099 ms device work and 5.819 ms host gaps). This reduces the contention
  anomaly but remains above the earlier 6.374 ms baseline capture; the selected
  terminal adds eager prefill dispatches. No profiled prefill speedup is claimed.
  Headline full-stack TTFT is the independently reproduced 97.244 ms.

## Stack accounting

The previous Stage5 best layer measurements were 0.421990 ms linear and
0.306304 ms full attention. Their primary inherited comparison is
`48 * 0.421990 + 16 * 0.306304 = 25.156384 ms`, versus the final full-model
25.605 ms/token. Those isolated layer measurements include per-layer trace replay
and synchronization overhead, so their weighted sum is a comparison estimate,
not a pure kernel lower bound or a claim that only 0.449 ms is terminal work.
The full stack amortizes those isolated-call costs. See the Stage5 README and
its primary selected measurement artifacts for the source values.

`stack_accounting.json` extrapolates the two representative device0 layer windows.
Kernel-only durations give a **22.890 ms stack floor** for48 linear +16 full layers.
Including the measured instrumented gaps estimates 25.436 ms for the stack and
27.010 ms with entry, terminal and sampling. The unprofiled full-stack 25.605 ms
lies above the kernel-only floor; the larger profiled extrapolation includes
instrumentation and independently measured per-layer gaps. It is an estimate,
not a claim that the complete stack was profiled. Final norm/head, embedding/RoPE,
sampling, dispatch gaps and output-token delivery account for the remaining work.
No full-logit gather or host argmax is included in the selected token-out path.

## Debug instrumentation and native repair

The allocation-tracked fullB32/256-token suite is correctness evidence only.
The tracker calls `gc.collect()` before replay (`ttnn/ttnn/unsafe_allocation_tracker.py:77`),
which becomes expensive with all-layer state. `tracked_generation_host_stack.log`
locates a sampled host stack in `gc_collect_main` while outputs continue to advance.
The tracker is disabled for final latency; this is not an alternate model path.

Watcher found a real over-sized single-packet read in the selected BF8 DRAM head.
`AUTOTRIAGE_dram_head.md` proves its17408-byte row exceeded the16384-byte packet
limit. The native helper now selects packet splitting by actual row size, keeping
the same weights/layout/tags. `native_wide_watcher.log` and `watcher_fixed.json`
pass focused and original regressions. `readiness_confirmed` and `profile_final`
both exit 0; their final post-fix measurements supersede the earlier selected
measurements. No allocator tracker, watcher or CPU HF computation was active
during these performance windows.
