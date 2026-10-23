# Qwen3.8-27B fused decoder

Stage complete. Independent review returned **clean-pass**; see `STAGE_REVIEW.md`. Local checkpoint SHAs are recorded in `work_log.md`.

Target: `Qwen/Qwen3.8-27B`, checkpoint revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Real layers 0 (`linear_attention`) and 3 (`full_attention`), one Blackhole p300c ASIC, physical device 0, 11x10 available compute grid. Four ASICs are available on the host; only one participates.

`tt/fused_decoder.py:FusedDecoder` is the delivered runtime. Tests import it directly. The optional `--compare-functional` flag creates a separate functional control outside measured windows; it is not a runtime fallback. Functional implementation and custom FP32 compute kernel remain unchanged.

## Results

Headline: **4096 input positions + 128 supplied-hidden-input decode steps, batch 1, one request**. This is single-layer work, not full-model token generation. Decode measures the final warmed traced step at context 4224.

| Layer kind | Functional prefill us | Fused prefill us | Speedup | Functional traced decode us | Fused traced decode us | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| linear_attention | 284743.541 | 83008.778 | 3.430x | 2660.428 | 2325.657 | 1.144x |
| full_attention | 108787.730 | 93214.924 | 1.167x | 2456.269 | 2187.320 | 1.123x |

| Layer kind | Functional prefill PCC | Fused prefill PCC | Functional decode PCC | Fused decode PCC |
|---|---:|---:|---:|---:|
| linear_attention | 0.999975860 | 0.999960254 | 0.999983549 | 0.999975682 |
| full_attention | 0.999962094 | 0.999962310 | 0.999983369 | 0.999983968 |

The acceptance bar remains **PCC >= 0.995**. The small linear delta follows different chunk algebra, fused normalization and rounding; it does not require a relaxed threshold. The historical linear functional statistic used FP32 correlation, whereas final fused statistics use FP64. Full-attention baseline PCC uses the prior FP64 control. Direct fused/functional equivalence is separately verified, rather than inferred from these HF scores.

Sources: `history_slice_final.json`, `final_full.json`, `tracy/{history_slice_final,final_full}/whole_layer_timing.json`, and `../functional_decoder/{profile_linear_b1_complete,full_b1_fp64}.json`. Baseline timing directories are `../functional_decoder/tracy/{linear_b1_complete,full_b1}`. `performance.json` and `tests/fused_report.py` reproduce the calculations.

## API, state and context

The constructor, `allocate_state`, `snapshot`, `prefill_forward` and `decode_forward` APIs match the functional stage. Prefill input is tiled BF16 `[B,S,5120]`; decode is `[B,1,5120]`. Output includes every logical input position. Positive logical sequence lengths need not align to a tile, page or chunk. Continuation starts at a 128-token boundary, as in the functional contract; decode handles individual positions.

- Weights remain BF16, matmuls HiFi4 with FP32 destination accumulation. Fused causal-convolution and gated RMS kernels explicitly disable unused L1 packer accumulation. FP32 recurrent state/arithmetic remains selected.
- Linear convolution history is now **row-major BF16 `[B,3,10240]`**, consumed directly by fused causal convolution. History updates exclude physical padding. State is 3145728 recurrence bytes + 61440 history bytes = **3207168 bytes/request**, saving 593920 bytes from the tiled functional layout. No second weight copy is retained.
- Full-attention KV remains BF16, `[physical_pages,4,32,256]`, with an INT32 paged table. Physical table capacity rounds to eight pages; logical lengths remain unrestricted. Current positions and RoPE tensors are refreshed on device-bound inputs before replay.
- Advertised and supported context remains **262144**. The state-layout change improves memory capacity and imposes no capability reduction. Exact/nonaligned maximum-context evidence is indexed in `../context_contract.json`.
- Linear prefix hits retain convolution/recurrent state for the exact prefix and position. Full-attention hits retain page ownership and absolute-position/RoPE metadata. `snapshot` clones state for isolation. Caller owns page aliasing, prefix reference counts and trace-bound lifetimes. This is a layer contract, not a serving scheduler.

Internal convolution/chunk-GDN padding uses physical multiples of 32. Padded GDN rows have zero q/k/v/g/beta, leaving state unchanged. Request batches are split only around recurrence kernels whose head count is limited by available cores; projections and public batching remain batched.

## Chosen graph and rejected alternatives

`PATTERNS.md` assesses every graph-fusing skill pattern and additional relevant dedicated op. `work_log.md` records exact attempts and source snapshots.

Selected changes: shared-input mixer projection packing; dedicated causal convolution + SiLU; flat-input chunked delta rule with L2/GQA handling; head-major gated RMSNorm in prefill; FP32 RMSNorm formulation of decode L2; native logical reshapes around the verified FP32 recurrent kernel; dedicated full-attention QKV split/prefill head concatenation and rotate-half RoPE; binary input-activation fusion; gate-matmul SiLU fusion; row-major history storage with conversion before the logical tail slice.

Key rejected options:

| Candidate | Evidence and reason |
|---|---|
| Dedicated minimal-matmul SwiGLU | Correct but slower: full decode 2685.596 us. Adapted M/K/N geometry under BF16/HiFi4 remains slower or exceeds L1. `candidate3/4.py`, `swiglu_k*n*`, `tracy/c3_full` |
| Packed MLP gate/up | Linear 2377.982 us; full 2227.382 us. Separate gate matmul with fused SiLU wins both. `tracy/{packed_final_linear,packed_full}` |
| Minimal-matmul split | Direct equal-width gate/up test passes; whole traced host screen 2789.998 us versus 2357.664 us gate-activation control. `minimal_split.json/log` |
| Decode-specific head ops + decode-mode RoPE | Correct after adapting padded outputs; no speed gain after required memory transitions. `decode_heads_crop.json`, `decode_heads_native_concat.json` |
| Fused gated RMSNorm in decode | Correct after 32-row adaptation, but slower than selected decode path. `norm_decode.json` |
| Whole-MLP fused expert kernel | Adapted to one dense expert, row-major input and supported compute settings. Needs 7497408 bytes CB L1 versus 1461120 available on maximum 11x10 grid. `expert_mlp_supported.log` |

The selected runtime beats the best correct measured alternatives in this stage. The final graph inventory has no unresolved fusion candidate. Broader precision, matmul-geometry, sharding, multichip, full-model and serving work remains in its designated later stages and was not started.

## Runtime movement and performance interpretation

`performance.json:operation_audit` lists measured ops. Final full decode has 35 device ops, no tilize/untilize, and two K/V height-sharding conversions required by paged cache update. Final linear decode has 56 ops, zero tilizes, and one untilize to transfer the newly projected token into the row-major convolution history. Linear prefill now has 931 device ops; moving conversion before the logical history slice removes 32 redundant tilize/untilize pairs (64 ops), reducing its whole-layer time from 83381.832 to 83008.778 us. That decode conversion is required by the selected fused-convolution input contract; the five redundant history-layout ops have been removed. No Torch, `from_torch`, `to_torch` or host tensor fallback occurs inside audited forward paths.

All selected dominant matmuls are BF16/HiFi4 and classified DRAM-bound at approximately 81% per-op bandwidth. These percentages are **not** the whole-layer roofline. Whole-layer useful-prefill FLOPs/peak are 22.970% linear and 21.066% full; estimated compulsory-decode DRAM/peak is 64.935% and 68.025% respectively.

The denominator is the complete layer's firmware timestamp envelope, including attention, normalization, movement and gaps, divided by the captured profiler header's 1350 MHz. It is not matmul-only time, summed op utilization or host time. Peak is one participating p300c ASIC: 120 physical cores * 4096 FLOP/cycle * 1.35 GHz / 4 for HiFi4 = 165.888 TFLOP/s, and 512 GB/s DRAM. Runtime exposes 110 compute cores; peak still uses all physical cores. Prior-stage source basis is retained in `../functional_decoder/README.md` and `tests/roofline.py`.

Useful FLOPs count canonical dense layer work, causal attention or recurrent updates and approximate scalar operations; no padding or sparse/MoE work. DRAM is an explicit lower-bound estimate: weights, cache/state, residual input/output, including a/b projection padding and the final row-major history. It omits intermediate spills, repeated within-plane reads and small metadata. No percentage is clamped.

Each selected profile has `ops.csv`, `whole_layer_timing.json`, `{prefill,decode}_perf_report.csv` and human-readable `.txt`. Raw captures remain local. The table's kernel-time sum is not substituted for the complete-layer denominator. The rejected packed-full control reported AICLK 1331 MHz versus requested/header 1350 MHz; using the lower clock would make that control slower still. Selected profile logs have no such warning. Optional GUI export, pandas and Matplotlib warnings are classified in the work-log anomaly ledger; numeric windows remain complete.

## Validation and reproduction

Use the existing `qwen38-functional-build` container, `/workspace` checkout and `/opt/venv` environment. Hardware commands run serially.

```bash
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/fused_run.sh check <label> --layer <0|3> --seq 129 --resume --compare-functional
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/fused_run.sh profile <label> --layer <0|3>
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/validate_fused.sh
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/validate_fused_linear_state.sh
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/validate_fused_history_slice.sh
python3 -m models.autoports.qwen_qwen3_8_27b.tests.fused_report --conv-row-major
```

The wrapper performs packaged environment setup. Profile mode runs exactly 4096+128, disables watcher/tracking and extracts tt-perf-report. Correctness mode enables `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; no allocation exemptions are used. Watcher runs are separate. Tests restore state after warmup/capture and prove bitwise repeatability, changed inputs/positions, page remapping, continuation and no runtime host fallback. Prepare all supported shapes before capture; weights, constants, state and input/output allocations must remain alive until trace release.

`final_validation.log` records the complete candidate15 suite: both watcher runs; exact 262144 and nonaligned 262142 plus two final-position decodes for both kinds; 4096+128 workloads at batches 8/16/32; odd linear batch 3; ragged full batch 8; and 18/18 synthetic cases. `linear_state_validation.log` records passing affected checks after the final row-major state change: watcher/boundaries, both maximum-context cases, all three batch workloads, odd batch 3 and 9/9 linear synthetic cases in 86.39 seconds. After the final review history-slice fix, `history_slice_validation.log` repeats those affected checks successfully, including 9/9 synthetic cases in 87.76 seconds. The 9 unaffected full-attention synthetic cases are reused from the original 18-case suite. `validation_summary.json` indexes the final evidence. `unaffected_full_attention.json` proves all full-attention code, allocation and MLP dependencies are unchanged, allowing its valid evidence to be reused.

Long prefill compares the final 128 outputs using the entire prefix; all TT outputs are computed. Recurrent-state PCC is about 0.998998 versus functional 0.999974; output and final-position decode remain above the unchanged bar. Different chunk rounding is the plausible mechanism, not an experimentally isolated attribution to one rounding site. Full long-context prefill remains approximately 0.9964, matching the functional control. No text-generation quality claim is made.

The agentic planning workload remains 32 concurrent conversations, 32768 live tokens each, typical context 4096. Batches 1/8/16/32 are validated. Supplemental 4096+128 batch rows in `performance.json` explicitly use host-wall latency around blocking trace execution; they are not copied into device-time telemetry. No full-model memory-pool or serving-cache performance claim is made. `../agentic_serving.json` retains prior-stage evidence and records this stage separately.

Final review, local checkpoint SHAs and the exact telemetry packet are recorded in `work_log.md`. Nothing is pushed.
