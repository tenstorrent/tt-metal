# Optimized Multichip Decoder

Model: `microsoft/Phi-3.5-mini-instruct`

Scope: one dense Phi-3.5 decoder layer on the completed 1x8 TTNN multichip path. No full-model or vLLM work was started.

Status: optimized multichip decoder state is complete for this decoder-layer goal. The final watcher run uses `TT_METAL_WATCHER=10` on the real 1x8 multichip path and is clean.

## Final Decoder Defaults

The optimized path in `tt/multichip_decoder.py` now defaults to:

- 1x8 ring mesh with tensor parallel local QKV/O/gate-up/down weights.
- Inter-layer residual contract: replicated BF16 residual tensor of shape `[1, 1, T, 3072]` on every device.
- Width-sharded L1 activations only inside a layer for local decode matmuls.
- No gather, reshard, or all-reduce between decoder layers.
- In-layer collectives only after row-parallel O projection and MLP down projection.
- Decode O/down DRAM-sharded local matmuls use `in0_block_w=2`.
- Decode in-layer all-reduce uses BF8 CCL payloads with BF16 casts around the collectives.
- Decode matmuls/SDPA use LoFi math fidelity by default.
- Prefill CCL payloads and prefill matmuls remain BF16/HiFi2 to keep prefill PCC and avoid prefill gap regressions.

Useful override env vars remain:

- `PHI35_MULTICHIP_LOCAL_MATMUL_MIN_IN0_BLOCK_W`
- `PHI35_MULTICHIP_CCL_DTYPE`
- `PHI35_MULTICHIP_PREFILL_CCL_DTYPE`
- `PHI35_MULTICHIP_MATMUL_FIDELITY`
- `PHI35_MULTICHIP_DECODE_MATMUL_FIDELITY`
- `PHI35_MULTICHIP_CCL=sync_all_reduce|async_all_reduce`

## Before/After Summary

All measured paths below are the 1x8 multichip decoder path, not a single-chip or replicated fallback.

| Metric | Baseline | Final default |
| --- | ---: | ---: |
| Prefill PCC vs single-chip layer | 0.999991928 | 0.999991789 |
| Decode PCC vs single-chip layer | 0.999993522 | 0.999975694 |
| Host-timed warmed traced decode E2E | 580.685 us | 559.258 us |
| tt-perf decode device time | 570.031 us | 543.090 us |
| tt-perf decode op-to-op gap | 575.837 us | 93.426 us |
| tt-perf decode total | 1145.868 us | 636.516 us |
| tt-perf prefill device time | 798.305 us | 798.644 us |
| tt-perf prefill op-to-op gap | 3719.245 us | 4059.331 us |
| tt-perf prefill total | 4517.550 us | 4857.975 us |

The prefill device time is flat after the final split. Prefill total time remains dominated by untraced op-to-op gaps and was not improved by the decode-focused changes.

## Accepted Changes

1. DRAM-sharded decode O/down matmul blocking:
   - Baseline advice identified `in0_block_w=1` on O and down local matmuls.
   - `in0_block_w=2` required aligning decode local hidden/intermediate shard width and explicit O/down output memory configs.
   - Host traced decode improved from 580.685 us to 571.521 us in the corrected standalone trial.

2. BF8 decode CCL payloads:
   - Decode RS/AG payloads changed from BF16 to BF8 with typecasts before and after each in-layer all-reduce.
   - Final decode RS/AG times were about 61/33 us and 60/34 us, down from baseline about 66/37 us and 67/36 us.
   - Prefill BF8 CCL was rejected; it reduced prefill PCC margin and worsened profile gap behavior.

3. Decode LoFi fidelity:
   - Decode matmuls and SDPA use LoFi by default.
   - Prefill stays HiFi2. An all-path LoFi trial lowered prefill PCC to 0.999955840, so fidelity was split by phase.

## Rejected Options

| Option | Result |
| --- | --- |
| Explicit async all-reduce/semaphore sets | Slower than sync. Host decode 584.073 us; tt-perf decode device/total 572.267/1213.491 us. Rejected. |
| Fused matmul reduce-scatter | Current TTNN API requires persistent intermediate/output buffers and yields sharded RS output. Keeping the current replicated inter-layer contract would need immediate all-gather, equivalent to the measured in-layer all-reduce path plus extra buffer management. Holding sharded residual across layers would require distributed RMSNorm and fused all-gather matmul changes outside the completed decoder contract. Rejected for this pass. |
| Output subblock tuning | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` in this checkout exposes `in0_block_w`, `per_core_M`, `per_core_N`, and `fused_activation`; output subblock knobs are not available for this path. |
| Prefill BF8 CCL | Decode-applicable, but prefill PCC/perf regressed. Prefill CCL default remains BF16. |
| All-path LoFi | Decode improved, but prefill PCC regressed. Final code applies LoFi only to decode by default. |
| Inter-layer sharded residual | Would add inter-layer collectives or require broader full-stack contract changes. Final contract is replicated residual between layers. |
| Watcher smoke variants | Earlier ad hoc watcher smokes either stalled with Ethernet watcher disabled or hit idle-ERISC code-size overflow without `TT_METAL_WATCHER_NOINLINE=1`. The final accepted watcher run mirrors the completed multichip stage and passes. |

## Performance Accounting

Decode roofline estimate:

- Per-device decode weights and KV read estimate: 9,474,048 bytes.
- Assumed per-device Wormhole DRAM peak from tt-perf DRAM percentage rows: about 288 GB/s.
- Aggregate 1x8 lower bound: about 0.0329 ms/token.

Accounting evidence:

| Source | Roofline | Device decode | Host traced decode | Notes |
| --- | ---: | ---: | ---: | --- |
| Accepted final evidence | 0.0329 ms | 0.543 ms | 0.559 ms | `final_split_fidelity` tt-perf plus final no-profiler host run. |
| Same-run accounting profile | 0.0329 ms | 0.546 ms | 0.782 ms | `final_accounting_short`; host timing is profiler-perturbed but proves the same optimized path. |

The gap to roofline is mostly CCL, small-op, and layout overhead. The optimized traced host latency is close to device time; avoidable host dispatch overhead was reduced by tracing. A 100-iteration same-run profile was attempted and rejected because profiler buffers overflowed and ARC lock waits appeared; the usable same-run accounting profile uses one host timing iteration.

## Verification

Passed:

- Static mesh plan and runtime fallback audit:
  `logs/final_static_fallback_audit.log`
- Final default real-weight layer PCC and host traced decode:
  `logs/final_default_host_timing_real_layer0.log`
- Repeated deterministic multichip path after reset:
  `logs/final_repeated_determinism_retry.log`
- Final performance profile and tt-perf-report outputs:
  `logs/final_split_fidelity_tracy_perf.log`
  `tracy/final_split_fidelity/reports/2026_06_15_15_01_00/ops_perf_results_2026_06_15_15_01_00.csv`
  `perf/final_split_fidelity_decode_perf_human.txt`
  `perf/final_split_fidelity_decode_perf_report.csv`
  `perf/final_split_fidelity_prefill_perf_human.txt`
  `perf/final_split_fidelity_prefill_perf_report.csv`
- Final same-run accounting profile:
  `logs/final_accounting_short_tracy_host_timing.log`
  `tracy/final_accounting_short/reports/2026_06_15_16_15_30/ops_perf_results_2026_06_15_16_15_30.csv`
  `perf/final_accounting_short_decode_perf_human.txt`
  `perf/final_accounting_short_decode_perf_report.csv`
  `perf/final_accounting_short_prefill_perf_human.txt`
  `perf/final_accounting_short_prefill_perf_report.csv`
- Final watcher-clean run:
  `watcher/2026_06_15_optimized_1x8_ring_real_watcher10/pytest.log`
  `watcher/2026_06_15_optimized_1x8_ring_real_watcher10/generated/watcher/watcher.log`

## Limitations

- Prefill is not trace-optimized in this pass. Its device time stayed flat, but host/op-gap profile noise remains high.
- The decoder is dense Phi, so MoE active-expert handling is not applicable.
- Full-model and vLLM stages were intentionally not started.

## Optimize Checklist

| Item | Status |
| --- | --- |
| Functional checks and PCC | Done; final real-weight prefill PCC 0.999991788840335, decode PCC 0.999975693890481. |
| Paged KV-cache and trace replay | Done; real-weight traced decode and repeated deterministic path passed. |
| Runtime fallback audit | Done; static audit passed. |
| Stress/repeated-run coverage | Done; repeated determinism retry passed after reset. |
| Warmed prefill/decode latency before and after | Done; tables and logs above. |
| tt-perf-report advice and CSV/provenance | Done; baseline, trials, rejected options, and final reports are in `perf/` and `tracy/`. |
| Watcher clean | Done; `TT_METAL_WATCHER=10` real-weight 1x8 path passed and watcher log scan had no matches. |
| Decoder path traced/no host fallback | Done for traced decode; runtime hot callables contain no Torch/from-device fallbacks. |
| Decode activation layout | Done; width-sharded L1 internally, replicated residual only at layer boundary by contract. |
| Prefill layout | Done; prefill remains DRAM interleaved with 2D prefill matmul configs. |
| Optimized composite ops | Done; SDPA/SDPA decode and TTNN composite head/cache ops are used. |
| Explicit configs | Done; key memory, program, compute-kernel, CCL, and dtype configs are explicit. |
| Shard/core-grid legality | Done; O/down decode shard widths were adjusted to make `in0_block_w=2` legal. |
| DRAM-sharded decode matmuls | Done; QKV/O/gate-up/down decode matmuls use DRAM-sharded weights. |
| Fused matmul-CCL | Tried/rejected with API/contract evidence. |
| MoE | Not applicable; Phi-3.5-mini is dense. |
| LM head/sampling | Not applicable to this decoder-layer goal. |
| Precision/fidelity | Done; decode BF8 CCL and LoFi accepted, prefill reductions rejected. |
| Performance accounting | Done; roofline/device/host accounting recorded above. |
