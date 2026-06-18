# Kimi chunked-prefill: the H2D-stream-service per-op dispatch tax

## TL;DR
A resident **H2D stream service** (the request-loop's host→device token streamer) was adding **~0.2 s/chunk**
to Kimi chunked prefill — about **+50–70% per chunk**, and the bulk of the runner-vs-test gap. It is **not**
a compute or on-device-dispatch problem: it is a **host-side per-op validation loop** in
`tt_metal/distributed/distributed.cpp::EnqueueMeshWorkload` that turns on the moment *any* service core is
claimed. The fix is to claim the service core **"isolated"** so model workloads keep the fast enqueue path,
and launch the service's own kernel via direct slow-dispatch (exactly how the realtime profiler already
behaves).

## Symptom
- Profiled L10, 1 chunk, warm: **nosvc 0.264 s/chunk → with H2D service 0.48 s/chunk** (+0.22 s).
- Tracy per-op diff (per device, warm chunk): **on-device kernel time unchanged** (224 ms → 213 ms, noise);
  **op2op (gap-between-ops) doubles** (174 ms → 408 ms, +234 ms). The whole penalty is op2op.
- The extra op2op is spread **flat across every op type**, scaling with op **count**, not compute weight
  (~0.2–1 ms extra per op launch; tiny ops like BinaryNg/RoPE/Slice hit hardest). Classic per-op-launch tax.

## What "op2op gap" means and why this shows up there
`op2op` = wall time from the previous op's kernel finishing on device to the next op's kernel starting. On a
warm, program-cached run the device is fed by the host: each `EnqueueMeshWorkload` call (one per op) submits
the next program. If the **host** is slow to issue that enqueue, the device sits **idle** in between — and
that idle shows up as op2op gap, with the kernels themselves unchanged. So a host-side cost *inside the
enqueue call* is indistinguishable, at the device, from a "dispatch gap."

That is exactly what happens here. The cost is **host-side, inside `EnqueueMeshWorkload`** — it is NOT in the
device dispatch kernels (`cq_dispatch*`), NOT go-signal multicast, NOT worker-completion. (We tried
`--profile-dispatch-cores`; `DISPATCH GO SEND WAIT TIME` was ~0. We also ruled out NOC contention, the
resident kernel itself, sub-device clear, the compute-grid cap, and core-reservation timing — see Evidence.)

## Root cause — `distributed.cpp::EnqueueMeshWorkload`
```cpp
// tt_metal/distributed/distributed.cpp, EnqueueMeshWorkload(...)
if (svc.impl().has_any_claims()) {            // <-- true the instant ANY service core is claimed
    for (auto& [device_range, program] : programs) {
        for (const auto& coord : device_range) {            // up to 32 devices on an 8x4 mesh
            for (const auto& per_type : program.impl().logical_cores()) {
                for (const auto& core : per_type) {          // every core the program uses
                    ++total_cores;
                    if (svc.impl().is_service_core(device->id(), core)) ++service_cores;  // 2 hash lookups
                }
            }
        }
        // ... TT_FATAL no-mixing asserts ...
    }
}
```
This block is a **"no-mixing" safety check** (a workload must be entirely on claimed service cores or
entirely on the worker grid) plus **SD-routing** for service workloads. Its own comment says it best:
*"Common case — no service claimed so skip entirely."*

The instant the H2D service calls `ServiceCoreManager::claim()`, `has_any_claims()` becomes true, so **every
subsequent model op** pays this `O(programs × coords × cores)` scan on the host before its command is
enqueued. On an 8x4 mesh with the Kimi model that's thousands of `is_service_core` lookups per op × hundreds
of ops/chunk ≈ **~0.2 s/chunk** of pure host enqueue latency → op2op gaps → device idle.

Crucially, `claim()` itself is **pure host bookkeeping** (it just sets up an L1 allocator + a grid snapshot,
no device work). It is not the claim's *work* that costs — it is that the claim **flips this per-op code path
on** for the whole process.

## Why the realtime profiler (also a resident service kernel) is free
The realtime profiler runs a persistent kernel on a reserved dispatch-column core too, yet adds ~0 ms. The
difference is **it never calls `ServiceCoreManager::claim()`** — it reserves its core via
`dispatch_core_manager` at device init and launches its program with direct slow-dispatch. So
`has_any_claims()` stays false and `EnqueueMeshWorkload` keeps the fast skip-path. (It is *not* about
init-reservation per se — it's about not claiming.)

## The fix — `isolated_claim`
Make the H2D service behave like the realtime profiler:
1. **Claim the service core "isolated"** — it still gets its per-core L1 allocator, but is **excluded from
   the per-op routing gate**. New `ServiceCoreManager::has_any_non_isolated_claims()` is used at the
   `EnqueueMeshWorkload` gate; `has_any_claims()`/`is_service_core()`/`claimed_cores()` still include
   isolated cores (program-placement and circular-buffer validation need to see the service core).
2. **Launch the persistent receiver via direct slow-dispatch** (`detail::LaunchProgram(..., force_slow_dispatch=true)`)
   instead of `EnqueueMeshWorkload`, since the isolated claim means the SD-routing branch won't fire for it.

No `dispatch_core_manager` init-reservation is needed: `get_claimable_cores()` already returns dispatch-column
spares that FD doesn't use, so there is never an FD conflict — the cost was only ever the per-op validation.

Exposed as an opt-in: `H2DStreamService::Config::isolated_claim` (default false) → `ttnn.H2DStreamService(...
isolated_claim=)` → `build_h2d_service(..., isolated_claim=False)`. The prefill runner enables it via
`PREFILL_H2D_ISOLATED_CLAIM` (default on).

## Evidence (plain runs, L10, 1 chunk, iter-1 warm; baseline nosvc = 0.264 s/chunk)
| configuration | s/chunk | note |
|---|---|---|
| nosvc (no service) | 0.264 | baseline |
| nosvc + `clear_loaded_sub_device_manager()` only | 0.266 | sub-device clear is innocent |
| h2dsvc, service built but **persistent kernel not launched** | 0.459 | the kernel is ~0 of it |
| h2dsvc, service on an **init-reserved** core (still claimed) | 0.459 | core/reservation is ~0 of it |
| h2dsvc, built only up to **backing tensor** (no claim) | 0.263 | = baseline → claim is the trigger |
| h2dsvc, built up to **claim only** (no sockets/kernel) | 0.459 | claim alone = full tax |
| h2dsvc, full service, **legacy claim** | 0.48 | the tax |
| h2dsvc, full service, **`isolated_claim=True`** | 0.265 | **fix → back to baseline** |

The two middle rows isolate it: stop *before* the claim = baseline (0.263); stop *right after* the claim
(nothing else) = full tax (0.459). The cost is the claim flipping `has_any_claims`, nothing else.

## How to reproduce / verify
Test: `models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc`
with three service modes: `nosvc`, `h2dsvc` (legacy claim → taxed), `h2dsvc_isolated` (the fix).
See the manual run recipe shared alongside this doc. Expected warm (iter 1) chunk time, L10 / 1 chunk:
nosvc ≈ 0.27, h2dsvc ≈ 0.48, h2dsvc_isolated ≈ 0.27.

## Open item for the dispatch owners
The `has_any_claims()` gate is what makes *any* claimed (non-isolated) service tax every model op. The H2D
fix sidesteps it (isolated claim). A complementary, general improvement would be to make the per-op
no-mixing check cheap (it's `O(programs×coords×cores)` today; the claimed set is tiny, so it can be
`O(coords×claimed)` or cached per program). That helps any service that legitimately needs a non-isolated
claim, but is **not** required for the H2D fix.
