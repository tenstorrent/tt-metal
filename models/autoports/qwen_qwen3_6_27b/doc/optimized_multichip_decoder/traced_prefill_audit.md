# Separate TP4 traced-prefill probe

Status: source and CPU preparation complete; **no device run, build, Torch import, or TTNN import was performed by the implementing subagent**. No measured latency or hardware pass is claimed here. Production code, helpers, the existing six same-harness baselines, and their timing semantics are unchanged.

Implementation: `tests/probe_traced_multichip_prefill.py`. This follows the installed [tracing skill](/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/tt-enable-tracing/SKILL.md) and `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`. Parent owns device acquisition, serialized execution, watcher/profiler separation, and any recovery.

## Boundary and reference

The probe loads actual pinned Qwen/Qwen3.6-27B layer weights, opens **1×4 FABRIC_1D**, and requires the sharded residual contract (1280 local hidden channels). Every readback requires all four rank tensors. There is no single-chip or replicated execution branch.

`--layer --batch --length --policy --reference --output` are required. Policy is an explicit JSON file; requested policy, resolved model policy, file hash, all Python implementation hashes under `tt/`, and reference file hash are recorded. A historical example policy must not be mistaken for the eventual current default: the operator should pass the precise policy being evaluated. Partial policies inherit production defaults, so both resolved policy and source hashes are essential provenance. No policy is silently selected by this probe.

Input/page generation follows `run_multichip_decoder` exactly: Torch seed17, pinned real weights loaded first, BF16 `randn * .1` prefill and decode inputs, then shuffled page IDs with `(length + 33 + 31) // 32` pages/user. Metadata must exactly match the supplied **frozen accepted single-chip TTNN** reference. The probe neither creates nor substitutes HF or fresh candidate references. A stale reference causes failure before opening devices.

Logical length33/513 is accepted unchanged; padding remains implementation-owned. CPU plan coverage includes lengths1,2,31,32,33,512,513. The recorded tile length is descriptive, not a public input restriction. The second input bank uses `-x,-d` at the same positions and fixed page mapping; it consumes no extra RNG and has distinct hashes.

The measured boundary is one complete prefill forward plus copying its final tensor into a preallocated output slot. Both eager and traced paths include that same copy. Restore of empty state, input-bank selection, and the synchronization before the timer are outside the interval; completion synchronization is inside. There are **seven synchronized single-call samples per path**, alternating eager-first/trace-first order. Each sample restores empty state. No recurrence is advanced hundreds of times and reported as prefill throughput. Decode trace exists for continuity and allocation-safety checks; this probe does not claim a new decode latency.

Tracing uses the actual capture branch. Eager may use the certified bulk cache fill while capture uses ordered cache fill. Those are the actual paths being compared, and dispatch counts explicitly identify the difference. This separate experiment does not replace or revise the existing warmed eager prefill / traced warmed decode baseline table.

## Ownership, warmup, and capture

All two-bank inputs, active inputs, empty state, both saved prefilled states, rotary/position tensors, and **both** persistent output slots are allocated before the first capture. Initial uncaptured forwards discover output specs. `empty_like` inherits dtype, layout, memory configuration, mesh distribution, and logical shape in this checkout (`operations/creation/creation.cpp`). Transient returned outputs are destroyed inside `invoke` before capture ends; only the preallocated slots survive between captures.

Full-attention state may have a certified private page-table lease. `MultichipDecoder._prefill_cache_fill` explicitly prohibits the bulk branch while `ttnn.is_trace_capture_active(mesh)` is true. To warm the legal ordered branch, the probe constructs an ordinary uncertified `PagedState` view sharing the key/value tensors and the internal read-only `page_table_for_attention(state)` handle. It does not expose the certified state's public `page_table` property, mutate the table, recertify a lease, fake capture-active state, or bypass an ownership check. Eager calls on the original state retain normal bulk eligibility. The original lease's certification is asserted unchanged through warmup, capture, and final replay. Each ordered result/cache must exactly match the same-bank eager result/cache.

Both banks, normal and ordered prefill, decode, all state-restoration copies, and saved-state copies are warmed before either capture. `mesh.set_program_cache_misses_allowed(False)` then rejects late compilation. There is exactly one prefill capture and one decode capture per process; there is no recapture workaround. Captured bodies contain forward plus device copy only. Public tensor conversion and device-synchronization APIs are patched to reject use inside forward/capture. The source has no host read, upload, synchronization, or reset inside the capture interval.

The state object and captured input/output addresses remain stable. Captured decode consumes the actual preceding traced prefill state at the next logical position, with no state restore between the two. After changed-bank replays and optional timings, uncaptured eager prefill/decode runs execute with both traces still alive. The final device forward is decode replay following prefill replay. This catches program-cache or persistent scratch allocations that appear only after eager interleaving. Program-cache entry count must remain unchanged.

## Gates and artifacts

For bank0, eager and traced prefill and subsequent decode must retain the existing >=.995 gates for total output, contribution (output minus input), every user, and each complete reconstructed cache/recurrent-state tensor against the frozen accepted TTNN reference. Finite/defined PCC is required. Conv state reconstruction preserves rank-local grouped Q/K/V ordering. The changed bank is checked exactly against its own warmed eager output **and both state tensors**, twice; every original-bank replay is also exact against eager and separately PCC-gated against the accepted reference. Timing samples undergo the same exact output/cache check outside the timed interval. No threshold, dtype, or gate is relaxed.

With `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, both traces must pass the tracker after capture and before replay. The positive control allocates one deliberately live tensor after capture, confirms its exact buffer ID is unsafe, requires the documented tracker rejection without replaying, frees it, then requires both traces clean. No acknowledged-buffer, corruptible scope, exception-based exemption, or invalidation is used. Program-cache skipping is rejected even if tracking is off. The positive control is not run in uninstrumented latency processes; those are explicitly labeled performance evidence only for this gate.

Timing is disabled whenever watcher, device profiler, allocation tracking, or `--quality-only` is active. Watcher and profiler together are rejected. The output JSON includes all quality/exact records, dispatch counts, signatures, source/reference/policy hashes, persistent addresses, cache counts, timing samples and medians, tracker positive-control result, and exception/traceback on failure. `run_job.sh` adds command/environment/commit provenance and the console/watcher logs.

This probe uses extra state and input banks solely as test fixtures. It does not change production memory/context contracts. At B32S513, a local BF16 residual input alone is approximately40.1MiB/rank; two banks, an active input and a stable prefill output add staging pressure. Empty and two saved prefilled cache/state copies also remain live. A probe allocation failure must be distinguished from the production context limit, with actual allocator evidence before any context-contract change.

## Parent launch commands

From checkout root, using the parent's established serialized `run_job.sh` device queue. The following existing reference/policy paths are concrete examples; set `P` to the **current explicit candidate** (or its fully resolved saved policy) before execution. Begin with the available B1S512 reference, then both layer kinds at B32S33/B32S513. Do not regenerate the accepted reference to bypass a mismatch.

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
R=models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder
P="$D/integrated14_profile_l0_b32_s33.policy.json"
TT_METAL_WATCHER=10 bash "$D/run_job.sh" traced_prefill_l0_b1_s512_quality_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_traced_multichip_prefill \
  --layer 0 --batch 1 --length 512 --policy "$P" \
  --reference "$R/delivery_perf_ref_l0_b1_s512.pt" \
  --output "$D/traced_prefill_l0_b1_s512_quality_v1.json"

bash "$D/run_job.sh" traced_prefill_l0_b1_s512_perf_v1 \
  env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_traced_multichip_prefill \
  --layer 0 --batch 1 --length 512 --policy "$P" \
  --reference "$R/delivery_perf_ref_l0_b1_s512.pt" \
  --output "$D/traced_prefill_l0_b1_s512_perf_v1.json"
```

For full-attention nonalignment/ownership, use `--layer 3 --batch 32 --length 33` and the current layer3 policy with `delivery_perf_ref_l3_b32_s33.pt`, then length513 with the matching reference. Repeat both tracking+watcher quality and uninstrumented timing. Likewise evaluate layer0 B32S33/513. A trace, shape, API, program-cache, or allocation failure is unfinished work; preserve its JSON/log, adapt the harness/model lifetime or warmup as necessary, and retry without removing a gate.

## Completed CPU validation

- `python3 -m py_compile .../tests/probe_traced_multichip_prefill.py`: pass.
- AST/import audit: module-level dependencies are Python stdlib; hardware imports are inside `main` after the `--cpu-only` return; captured invocation contains no explicit allocation, host conversion, or synchronization API.
- Pure-Python launch plans: 42 valid combinations B1/3/9/10/17/32 × S1/2/31/32/33/512/513, and rejection of B0/B33/S0: pass.
- Actual `--cpu-only` CLI exercised at layer3 B32S513 using existing policy/reference paths and a temporary output JSON; emitted `cpu-plan-pass`, no Torch/TTNN present in `sys.modules`: pass.

Pending evidence: both device signatures compile and fit trace memory, tracker positive control, both layer kinds' frozen-reference gates, changed-bank/cache exactness, watcher-clean logs, uninstrumented seven-sample timing, and final decode-after-eager replay. This document records no hardware success until those parent-owned artifacts exist.
