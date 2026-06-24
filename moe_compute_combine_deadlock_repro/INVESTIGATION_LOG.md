# moe_compute fused-combine hang — investigation log (2026-06-24)

Galaxy: WH 6U `UF-EV-A5-GWH01`, mesh `(4,8)`, `cluster_axis=0`, COL dispatch, `FABRIC_1D_RING`.
Commit `ec16987276e` (branch `session-on-remote`, off `mvasiljevic/moe-compute-combine-deadlock-repro`).
Container `tt-xla-ird-mvasiljev`, runtime `TT_METAL_HOME=/home/mvasiljev/tt-metal`.
Tracking issue: tenstorrent/tt-metal#47523.

---

## TL;DR (most important conclusion)

The fused-combine deadlock (`ttnn.experimental.moe_compute` Full mode → `selective_reduce_combine`,
`synchronize_device` never returns) is **INTERMITTENT and device/fabric-state-dependent**, NOT a
deterministic function of the mux core range, routing seed, fabric topology, kernel cache, or build.

The decisive evidence is a **drifting hang rate for the exact same config** across one ~2.5h session:

| Window (2026-06-24) | config (default mux `(3,0,4,7)`) | result | confirmed by triage? |
|---|---|---|---|
| 11:23–11:35 | baseline / cleancache / sigcheck | **HANG 3/3** | YES (11:37 sigcheck: dev 16/20/24/28, writer.cpp:365) |
| ~11:41–13:17 | e2e test + mux A/B + 13-trial stats + no-settle ctrl | **PASS 14/14 valid** | n/a (passes) |
| 13:35–14:07 | fixed stress harness, 5 attempts | stall ("HANG" 4/4) | **NO — UNVERIFIED** (see caveat) |
| 16:16–16:52 | probe + catch (single-shot, canonical) | **PASS 9/9** | n/a (passes) |

**CAVEAT on the 13:35–14:07 window (added after review):** these were NOT confirmed hangs. They were
classified HANG solely by a host-side log-stall heuristic in the freshly-modified device-open stress
harness (`run_stress.sh`), which had just had two L1-clash harness bugs (13:23, 13:27). All four runs
stopped after the first `moe_compute`'s 35 placement prints with NO `moe_compute ok` and NO
`STRESS iter 1 OK` — i.e. stuck at/after the first enqueue+synchronize on iteration 0. **No `tt-triage`
was run, so devices 16/20/24/28 and `writer.cpp:365` were NOT confirmed for this window.** The exact
unmodified single-shot test (`run_verdict`) was NOT re-run during 13:35–14:07; the only canonical
single-shot runs after 13:17 were the 16:16+ probe/catch, which all PASSED. So 13:35–14:07 may be a
genuine return of the deadlock OR a stress-harness/device-open-path artifact — currently undetermined.

What IS solidly established: a confirmed-and-triaged HANG regime (11:23–11:37) and a long confirmed
PASS regime (11:41–13:17, plus 16:16–16:52, 23/23 single-shot passes). The intermittency between
those two is real.

### UPDATE 17:0x — stall verifier result (catch_and_verify, run 15:03–15:16)

Caught and **LIVE-triaged a stall on attempt 2** of the device-open inducer:
- inducer verdict=STALL, **SIGMATCH=YES**: op `15: MoEComputeDeviceOperation` RUNNING on devices
  **16, 20, 24, 28**, combine writer at **writer.cpp:365:27** — *identical* to every prior capture.
- Stalled with `moe_compute_ok=0` → it deadlocked on the FIRST op of the device-open session.
- Attempt 1 = the post-reset fabric-init flake (`topology_mapper.cpp:518`) → ERROR.

**This confirms the device-open stress reproduces the GENUINE canonical deadlock** (not a harness
artifact), and retroactively makes the 13:35–14:07 stalls very likely the same real hang (same path,
same signature). The earlier retraction was the correct caution, but the signature is now verified.

Cross-check nuance: immediately after the caught stall, `tt-smi -r` + 30s settle + the EXACT
unmodified single-shot test (`run_triage_sig.sh`) **PASSED** → result classified `STRESS_STALL_ONLY`.
CONFOUND: the cross-check resets the HARDWARE before the single-shot, so this cannot distinguish
"the reset recovered the device" from "the single-shot path is genuinely less hang-prone than the
device-open multi-iter path." Both remain possible. Open follow-up: cross-check WITHOUT a hardware
reset (kill only the python process, reopen device, rerun single-shot) to tell these apart.

Net: the hang is real and 100%-consistent in signature; the device-open inducer is now a reliable
*live* repro for triage; whether single-shot is truly less prone (vs just reset-recovered) is TBD.

### UPDATE 17:51 — DECISIVE A/B (ab_now, 15:21–15:51): path-deterministic, NOT random

Alternating single-shot vs device-open stress, IDENTICAL `tt-smi -r` + 30s settle before each, same
window:

```
ROUND 1: single-shot PASS | stress STALL (sigmatch=YES)
ROUND 2: single-shot PASS | stress STALL (sigmatch=YES)
ROUND 3: single-shot PASS | stress STALL (sigmatch=YES)
```

3/3 clean separation. All three stress stalls were the **verified canonical deadlock** (sigmatch=YES:
dev 16/20/24/28, writer.cpp:365) and all hung on the **FIRST op** (`moe_compute ok=0`, zero
iterations completed). So:

- This is **NOT pure fabric-state intermittency** — in one fixed window the outcome is *deterministic
  by code path*: single-shot always passes, device-open stress always hangs.
- It is **NOT iteration wear-down** — the stress hangs on op #1, not after N iterations.

**Why this matters (likely explains the whole issue):** the real model runs the **device-open,
repeated, alloc/free** pattern — exactly the stress path — and hangs; isolated single-shot tests use a
simpler one-shot pattern and pass. amorrisonTT's passing e2e and our passing single-shots are the
"easy" path; the model is the "hard" path.

**Pinned-down difference (only material one):** the two first-op call sites are otherwise identical
(same `num_links=4`, `worker_mode=DIRECT`, `SPARSE_MCAST_SHORTEST_PATH`, same `dispatch_sem`/
`combine_sem` created once at smoke L247-248, same mux `(3,0,4,7)`, same combine prealloc shape/memcfg,
default Ring topology). The stress path differs ONLY in that it **deallocates the uploaded inputs
`x/idx/scr` and the pre-built dispatch `prealloc` (L325-332) and reallocates fresh `xi/ii/si` + `pa =
build_dispatch_prealloc` before op #1** → a different L1/DRAM memory map and dispatch-prealloc
placement. Hypothesis: the dealloc/realloc shifts dispatch-prealloc / moe_compute core+buffer
placement onto a layout whose cross-device combine routing deadlocks.

**Reconciliation with earlier windows:** single-shot DID hang in the morning (11:23–11:37) but passes
now → single-shot has a residual fabric-state component (only hangs in bad windows). The device-open/
realloc path is *strictly more prone* — it hangs even in a window good enough for single-shot to pass.

**Next step — bisection (code edit):** make the stress branch reuse the original `x/idx/scr/prealloc`
for iteration 0 (skip the pre-op dealloc/realloc). If op #1 then PASSES, the dealloc/realloc memory-map
change is the trigger; if it still hangs, the trigger is the device-open loop context itself. Either
way this isolates "test content" from "fabric state" definitively.

### Provenance: which path is the ORIGINAL repro?

`git show HEAD:.../moe_compute_smoke.py` has NO `SMOKE_ITERS`/stress loop → **the committed original
repro is the SINGLE-SHOT path** (dispatch → fused moe_compute → sync, once). The device-open stress
loop (+75 lines) is a working-tree addition from THIS session. So the test the issue/branch was filed
with is the single-shot one — which now PASSES (it hung in the morning; fabric-state-dependent),
while the session's device-open variant hangs reliably.

### Bisection setup (running, ~18:0x): SMOKE_REALLOC knob

Added `SMOKE_REALLOC=1` to the single-shot path (smoke L380): right before the single op it
deallocates `x/idx/scr` + the dispatch `prealloc` and rebuilds fresh ones — i.e. injects ONLY the
device-open path's churn into the ORIGINAL single-shot test, nothing else. A/B `bisect_realloc.sh`:
3 rounds of single-shot REALLOC=0 vs REALLOC=1 under identical reset+settle, + a triaged confirm of
REALLOC=1. Expectation if the memory-map churn is the trigger: REALLOC=0 PASS, REALLOC=1 HANG with the
canonical signature → a deterministic one-shot repro derived from the original test.

**RESULT (18:26) — CONFIRMED, 3/3 deterministic:**
```
ROUND 1: REALLOC=0 PASS | REALLOC=1 HANG
ROUND 2: REALLOC=0 PASS | REALLOC=1 HANG
ROUND 3: REALLOC=0 PASS | REALLOC=1 HANG
REALLOC=1 triaged: CONFIRMED HUNG — op15 MoECompute RUNNING, dev 16,20,24,28, writer.cpp:365
```
The ONLY change between PASS and HANG is **deallocating + reallocating the inputs (`x/idx/scr`) and
the dispatch `prealloc` before the single fused `moe_compute`**. Same reset+settle, same window, same
everything else. So:

- **The trigger is the L1/DRAM memory-map / allocator state, NOT fabric-window intermittency.** In the
  very same windows where the clean single-shot passes 3/3, injecting the realloc churn hangs 3/3 with
  the verified canonical signature.
- **We now have a DETERMINISTIC one-shot repro** derived from the original test: `SMOKE_REALLOC=1`.
- **This explains the model:** a real decode loop deallocates/reallocates tensors every step, so it
  runs against a churned (non-pristine) memory map → hits the deadlock; isolated one-shot tests start
  from a clean allocation and pass. That is the "passes for them, hangs for me" mechanism.

Mechanism hypothesis: the reallocated layout shifts moe_compute's combine/mux core+buffer placement
(or a semaphore L1 address) onto a configuration whose cross-device combine routing deadlocks at the
ring barrier. Always the same four devices (16/20/24/28) → a specific physical region is sensitive.

**Remaining narrowing (not yet done):** split the churn — realloc inputs ONLY vs dispatch prealloc
ONLY — to see which one flips it; and capture the moe_compute placement (`selected ... cores`) +
buffer addresses in REALLOC=0 vs =1 to see exactly what moves.

### Split bisection RESULT (18:48): the INPUT realloc is the trigger, not the prealloc

```
inputs-only   -> HANG  (round 2 = fabric-init flake, not a crash → 1/1 valid HANG)
prealloc-only -> PASS, PASS  (2/2)
both          -> HANG   (positive control)
none(0)       -> PASS   (negative control)
```
And the **moe_compute placement is IDENTICAL in every mode** (tilize 4, combine 16, matmul 12) → the
hang is **NOT a core-placement shift**. So reallocating the dispatch prealloc is harmless; reallocating
the **router inputs `x/idx/scr`** is what flips PASS→HANG.

Key reframing: "realloc inputs" = **re-upload the cluster-axis-sharded `x/idx/scr` across the mesh**
(host→device redistribution that uses the fabric), whereas "realloc prealloc" is a pure device-side
allocation (no host traffic). So the suspect is now **fabric perturbation from re-sharding the inputs
right before the op**, NOT the L1/DRAM address map. This aligns with the original FINDINGS note that an
incorrectly-sharded input "leaves the fabric in a state that deadlocks the subsequent CCL."

### Mechanism test (running, 18:5x): host re-upload vs device-only clone

Added `SMOKE_REALLOC=inputs_clone`: changes the input ADDRESSES via on-device `ttnn.clone` (no host
re-upload, no re-shard). A/B vs `SMOKE_REALLOC=inputs` (host re-upload), 3 valid reps each (fabric
flakes auto-retried). Decision:
- inputs HANG + inputs_clone PASS → the **host re-upload / re-shard over the fabric** is the trigger
  (a fabric-state effect), not the address change.
- inputs_clone also HANG → the **device-side address / memory-map change** is the trigger.

**RESULT (19:16) — inputs_clone ALSO HANGS 3/3:**
```
REP 1: inputs(upload) HANG | inputs_clone HANG
REP 2: inputs(upload) HANG | inputs_clone HANG
REP 3: inputs(upload) HANG | inputs_clone HANG
```
`inputs_clone` changes the input tensors via on-device `ttnn.clone` — NO host re-upload, NO re-shard,
NO cross-device/fabric traffic — and still hangs. So it is **NOT** fabric perturbation from re-sharding.

**Established trigger:** *recreating the router input tensors `x/idx/scr`* (free + re-allocate, by ANY
means) deterministically causes the fused-combine deadlock; recreating the dispatch prealloc does not;
moe_compute core placement is unchanged. So the op is fragile to the **input tensors' allocation state
(DRAM buffer address / allocation order / buffer identity)** — when they are NOT at the pristine
first-allocation layout, the combine ring barrier deadlocks.

**This is the full "works for them, not me" mechanism:** the real decode loop never has a pristine
allocator state at moe_compute (it allocates/frees every step), so its inputs land at non-pristine
addresses → deadlock. Isolated one-shot tests allocate inputs first → pristine layout → pass.

Open sub-question (address vs buffer-identity vs alloc-order): not yet separated; needs the actual
`x/idx/scr` buffer addresses logged in PASS vs HANG. But the actionable trigger is pinned.

### Culprit trace (19:34) — all-core callstacks under SMOKE_REALLOC=inputs

Ran `live_hang_triage.sh` (full `tt-triage --run=dump_callstacks --all-cores -vv`) with
`SMOKE_REALLOC=inputs`. Hung on attempt 1; callstack log
`logs/livehang_triage_callstacks_20260624_172714.txt` (31 MB). Distribution of parked kernel cores —
**three distinct moe/combine stall points, 12 cores each**, plus matmul:

| parked at | count | meaning |
|---|---|---|
| `moe_compute/.../dm1.cpp:359` | 12 | moe_compute data-movement: `noc_semaphore_wait(combine_semaphore_ptr, combine_semaphore_val)` — waits for the combine to signal the output buffer segment is free (producer↔consumer handshake) |
| `selective_reduce_combine/.../writer.cpp:271` | 12 | combine writer: token-search busy-wait (looking for data that never arrives) |
| `selective_reduce_combine/.../writer.cpp:365` | 12 | combine writer: cross-device ring-barrier `noc_semaphore_wait` (peers waiting for an atomic-inc that never comes) |
| `matmul.h:335` (+ pack_untilize, llk_*) | 12 | compute cores parked in matmul |

The `writer.cpp:365` cores are in EARLIER device sections of the dump (~L14.7k) than the
`writer.cpp:271`/`dm1.cpp:359` cores (~L22k) → they are on **different devices**. So the deadlock is a
cross-device producer↔consumer + collective tangle:

**dm1.cpp:359** (moe_compute DM waits for combine to free the buffer) ↔ **writer.cpp:271** (combine
waits for the data dm1 should produce) on the stuck devices → those devices never reach the barrier
send (`writer.cpp:339`) → **writer.cpp:365** peers wait forever for the missing atomic-inc.

The dm1↔combine handshake at dm1.cpp:359 uses `combine_semaphore_ptr` / `combine_semaphore_val` and
buffer offsets. The input-allocation sensitivity most plausibly desyncs THIS handshake (a semaphore
value/address or a buffer offset derived from the input/buffer layout), producing the producer↔consumer
deadlock that then cascades into the cross-device ring barrier. This is the concrete code locus to
hand to the moe_compute / selective_reduce_combine owners, together with the deterministic
`SMOKE_REALLOC=inputs` repro.

### Address capture (17:56) — it is NOT the address/memory map

`SMOKE_ADDRS=1` prints `buffer_address()` of all inputs/outputs right before the op:
```
MODE=0            (PASS): x=0x895180 idx=0x89a180 scr=0x89a1c0  sparse=0x886180 disp_idx=0x165c00 ...
MODE=inputs       (HANG): x=0x895180 idx=0x89a180 scr=0x89a1c0  sparse=0x886180 disp_idx=0x165c00 ...  ← IDENTICAL to PASS
MODE=inputs_clone (HANG): x=0x89a200 idx=0x89f200 scr=0x89f240  sparse=0x886180 disp_idx=0x165c00 ...  ← different, still hangs
MODE=prealloc     (PASS): x=0x895180 idx=0x89a180 scr=0x89a1c0  sparse=0x886180 disp_idx=0x165c00 ...
```
`inputs` (HANG) has **byte-identical addresses** to `mode 0` and `prealloc` (PASS); `inputs_clone`
(HANG) has different addresses. So **address/memory-map is NOT the factor**, and the input DATA is
identical too. The ONLY discriminator is whether `x/idx/scr` were **freed + recreated right before the
dispatch** (by re-upload OR device clone).

This kills the "memory-map placement" hypothesis. Leading explanation now: a **write-before-read
ordering / dependency miss** — the recreate (re-upload/clone) of `x/idx/scr` is not properly ordered
before `all_to_all_dispatch_metadata` reads them, so the dispatch reads stale/incomplete input →
produces token maps inconsistent with the activations → combine token-search (writer.cpp:271) never
matches → deadlock. (Note: `prealloc` churn passes because the dispatch *inputs* are untouched.)

### Decisive ordering test (running, 18:0x): SMOKE_REALLOC_SYNC

Added `SMOKE_REALLOC_SYNC=1`: `synchronize_device` immediately after the recreate, before the dispatch.
A/B `inputs` vs `inputs+SYNC`, 3 valid reps each.
RESULT: `inputs` HANG 3/3, `inputs+SYNC` HANG 3/3. **A full device sync after the recreate does NOT
fix it** → it is NOT a drainable write-before-read ordering race. The trigger is the allocator-state
perturbation from freeing+recreating the dispatch inputs, not timing.

### Dispatch metadata content differs PASS vs HANG (18:34) — SMOKE_DUMP

`SMOKE_DUMP=1` reads back `disp_idx`/`disp_scr` (dispatch metadata, feeds the combine token maps)
right after the dispatch, before moe_compute:
```
disp_idx MODE=0      (PASS): sum=267801704  first16=[62,17,109,33,102,43,74,94, 16058,16083,15733,15995,48703,16074,48775,15888]
disp_idx MODE=inputs (HANG): sum=646176     first16=[106,71,44,20,23,156,56,7,  0,0,0,0,0,0,0,0]
```
Token0 populated in both; in the HANG run the remainder of `disp_idx` is **all zeros** (sum ~400x
smaller — not a reordering artifact, since sum is order-independent). So recreating the dispatch inputs
makes `all_to_all_dispatch_metadata` **under-populate the token-routing metadata**. The combine then
spins forever at writer.cpp:271 searching for a token id that the dispatch never recorded. (Caveat:
the readback also touches uninitialized tile padding — `disp_scr` shows nan/garbage in both — so treat
the absolute values loosely; the populated-vs-zero structural difference and the 400x sum gap are the
reliable signal.)

### CONCLUSION (root cause)

The hang is a deterministic, allocator-state-triggered desync between `all_to_all_dispatch_metadata`
and `selective_reduce_combine`:
1. Freeing + recreating the dispatch INPUT tensors (`x/idx/scr`) — by re-upload OR on-device clone,
   at the SAME or a different address — makes the dispatch emit incomplete/inconsistent token metadata.
2. The combine writer's token search (`writer.cpp:271`, `while (expert_token_activations_ptr[0] != st)`)
   never finds the missing token. Its only termination is `ASSERT(guard++ < global_num_tokens)` at
   line 273, which is **compiled out in release** → infinite loop instead of a fast failure.
3. That combine core never reaches the cross-device ring barrier (`writer.cpp:365`), so peer combine
   writers wait there forever, and the moe_compute data-mover deadlocks at `dm1.cpp:359` waiting for
   the combine to free its buffer. → full multi-device deadlock (the triaged signature on devs 16/20/24/28).

Ruled OUT as the trigger: buffer address / memory map (identical in PASS & HANG), input data values
(identical), host-upload vs device-clone (both hang), write-before-read ordering (sync doesn't fix),
mux core range geometry, core-placement counts.

Two distinct defects for the owners:
- **A (robustness):** the compiled-out `ASSERT` at `writer.cpp:273` turns a metadata desync into an
  unrecoverable hang instead of a fast, debuggable failure.
- **B (root cause):** `all_to_all_dispatch_metadata` (and/or moe_compute's consumption of it) is
  sensitive to allocator state when its input tensors are recreated, producing incomplete token
  metadata. Deterministic repro: `SMOKE_REALLOC=inputs` (free+recreate `x/idx/scr` before the op).

My earlier "the mux core range is the root cause" claim (from a clean-looking 3/3-HANG vs 2/2-PASS
A/B) was a **small-sample artifact** and is RETRACTED — see Experiment 6.

---

## The bug (what hangs)

- Op: `ttnn.experimental.moe_compute(..., optional_output_tensor, optional_cross_device_semaphore)`
  in Full mode; the fused combine is `selective_reduce_combine` internally.
- Symptom: `moe_compute` enqueues ("moe_compute ok ... combine (8,16,5120)"), then the immediate
  `ttnn.synchronize_device` never returns.
- Triage signature (consistent across 7+ captures): op `9: MoEComputeDeviceOperation` RUNNING on
  **devices 16, 20, 24, 28**; combine writer cores parked in a NOC-semaphore wait at the
  cross-device ring barrier `selective_reduce_combine/.../writer.cpp:~352-365`
  (`noc_semaphore_wait(semaphore_ptr, expected_dispatch_device_inc)`).
- A separate but related post-reset failure: `set_fabric_config(FABRIC_1D_RING)` intermittently
  `TT_FATAL`s "Graph specified in MGD could not fit in the discovered physical topology"
  (`tt_metal/fabric/topology_mapper.cpp:518`). Same "fabric not deterministically ready" root.

---

## Session timeline (2026-06-24)

Pre-session: issue filed 2026-06-19; hangs reproduced 06-23 (axis/cycle/topology sweeps) and
06-24 morning (livehang triages 09:52, 10:07, 10:08, 11:00). So hangs predate this session.

This session:
- 11:23 `baseline` (warm cache, default mux) — **HANG**
- 11:28 `cleancache` (wiped `~/.cache/tt-metal-cache`, `TT_METAL_FORCE_JIT_COMPILE=1`) — **HANG**
  → rules out stale/poisoned kernel cache.
- 11:35 `sigcheck` live triage — **HANG**, signature = op9 on devices 16,20,24,28 (matches all prior).
- ~11:41 amorrisonTT e2e test `test_tt_moe_decode.py` (their commit `ddeb7b52f5a`, glm_47, (4,8),
  cluster_axis=0, FABRIC_1D_RING) — **PASS** → rules out machine/firmware/topology/fabric as a
  deterministic cause.
- 11:49 / 11:52 `mux_model` (mux→`(1,1,3,3)`) — **PASS x2**  ← looked like a fix (it was not).
- 12:00–12:13 `sweep_mux`: `(1,0,2,7)` PASS, `(1,1,3,3)` PASS, `(5,0,6,7)` HANG, `(3,0,4,7)` &
  `(3,1,4,2)` early-exit (fabric flake / too-few-mux FATAL).
- 12:18 `rerun_mux` (+25s settle): default `(3,0,4,7)` — **PASS** ← first contradiction of the mux story.
- 12:22–13:03 `stats_mux` (reset+settle each): A default+seed1234 **4/4 PASS**; B default+random
  **4/4 PASS** (1 fabric flake); C model+random **5/5 PASS**. → 13/13 valid PASS regardless of mux.
- 13:04–13:17 `nosettle_ctrl` (default mux, no settle): **4 PASS, 0 HANG, 2 fabric flake**.
- 13:22–13:32 stress-harness bring-up: runs `132300`/`132717` crashed on L1-clash harness bugs
  (`TT_THROW Statically allocated circular buffers ... clash`, Traceback) → ERROR, not hang.
- 13:35 `stress` (fixed harness, default mux) — log-stall after first `moe_compute` placement, no
  `moe_compute ok` / no iter progress. Classified HANG by stall heuristic; **NOT triaged**.
- 13:41–14:07 `stress_until` (5 attempts): 4 stalls (verdict "HANG", `last='none'`) + 1 ERROR. All
  stopped at the first enqueue (35 placement prints, then silence). **None triaged; none confirmed as
  the combine deadlock.** UNVERIFIED — see TL;DR caveat. Single-shot `run_verdict` was NOT run here.
- ~16:15 `probe_now` (3 trials, reset+settle): see RESULT appended at bottom.

---

## Experiments and what each ruled OUT

1. **Kernel cache** — wiped `~/.cache/tt-metal-cache` (1.6G) + `TT_METAL_FORCE_JIT_COMPILE=1`,
   fresh JIT (508M, our uid): still HANG. The pre-existing cache was owned by a different uid; the
   fresh rebuild by our uid hung identically. → cache/binary provenance is NOT the cause.
2. **Topology missing from `SelectiveReduceCombineParams::attributes()`** — real latent bug (cache
   key omits `topology`), but irrelevant on a fresh compile (nothing to mis-reuse). Not this hang.
3. **Machine / firmware / fabric / mesh** — amorrisonTT's e2e glm_47 decode test PASSES on our
   machine at the identical mesh/axis/topology. → not a fixed hardware/topology defect.
4. **Routing data** — default mux + fixed `SMOKE_SEED=1234` ran 4/4 PASS (mid-session); random
   routing hung early and passed mid. Outcome is not determined by the input seed.
5. **Mux geometry** — see Experiment 6; mux has a hard floor (`num_links*neighbors` cores, =8 here;
   `(3,1,4,2)`=4 cores → clean FATAL at `selective_reduce_combine_program_factory.cpp:145`) and it
   sets *where* the 16 combine cores land (`moe_core_placement.cpp` avoid-set), but it does NOT
   deterministically gate the hang.
6. **RETRACTED: "mux is root cause."** Initial A/B looked clean (default `(3,0,4,7)` HANG 3/3 vs
   model `(1,1,3,3)` PASS 2/2), but repeated trials showed the SAME default mux PASS 8/8 (stats A+B)
   and later HANG 4/4 (stress). The 3/3-vs-2/2 was an intermittent failure lining up with test order.

---

## Conclusion

Intermittent, device/fabric-state-dependent race in the fused combine's cross-device ring barrier.
Rate varies over time on the same config. Two observable symptoms of the same underlying "fabric not
deterministically ready/stable": (a) the combine `noc_semaphore_wait` deadlock, (b) the
`set_fabric_config` topology-mapping FATAL after reset.

### When did tests start hanging?
They were already hanging before this session (issue 06-19; 06-23 sweeps; 06-24 morning triages) and
at the start of this session (11:23). There is **no single "started hanging" moment** — it is a
recurring intermittent condition. Within the session it then entered a clean window (~11:41–13:17,
14/14 pass). The apparent "bad window again" at 13:35–14:07 is UNVERIFIED (untriaged stress-harness
stall, not cross-checked with the single-shot test) — see TL;DR caveat; do not rely on it.

### Are all tests hanging now?
See `probe_now` RESULT appended below.

### Repro to ENTER / EXIT the hang state?
**We do NOT have a deterministic recipe.** Observed facts:
- The state drifts on its own across tens of minutes (high→low→high) with identical inputs.
- `tt-smi -r` does NOT reliably clear it: late-session it hung across many resets; mid-session it
  passed across many resets. Reset is necessary to recover from an *individual* wedged hang, but
  does not change the regime.
- A ~25–30s post-reset settle reduced the fabric-init FATAL but did not prevent the iter-1 hangs in
  the late-session bad window.
- No correlation found with mux/seed/topology/cache.
This is consistent with a hardware/fabric-level condition (link training, accumulated reset state,
or thermal) rather than anything controllable from the workload. Next step to characterize the
device state directly: `./build/test/tt_metal/tt_fabric/test_system_health` in good vs bad windows,
plus fabric link/telemetry inspection.

---

## Key source references

- `ttnn/cpp/.../selective_reduce_combine/device/kernels/dataflow/writer.cpp` — ring-barrier
  `noc_semaphore_wait` (~L352-365); token-search busy-wait (~L271).
- `ttnn/cpp/.../moe_compute/moe_core_placement.cpp` — `select_moe_compute_cores`; mux added to
  avoid-set (`build_moe_compute_avoid_set`, L161); combine strip prefers eastern columns.
- `ttnn/cpp/.../selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp:145` —
  `needed_cores <= available_cores` ("Not enough mux cores: num_links*neighbors").
- `models/common/modules/moe/tt_moe_decode.py:920` — model's `moe_compute` call (default mux
  `(1,1)-(3,3)` from `MoEComputeConfig`, `tt_moe_decode_config.py:215`).
- `tt_metal/fabric/topology_mapper.cpp:518` — post-reset fabric-init FATAL.

## Tools created this session (all in this repro dir)

- `moe_compute_smoke.py` — added `SMOKE_MUX` (mux override; `=model` → `(1,1,3,3)`) and `SMOKE_ITERS`
  (in-process, no-reset stress loop; self-contained per-iter alloc/free).
- `run_verdict.sh` — single-shot HANG/PASS verdict + strict hygiene (kill own PID, no stragglers, no
  device fds, clean container `/dev/shm`).
- `run_stress.sh` — `SMOKE_ITERS` stress runner with progress-stall hang detection + fabric retry.
- `sweep_mux.sh` / `rerun_mux.sh` — mux-geometry sweep (reset between, settle, fabric retry).
- `stats_mux.sh` — repeated-trial hang-rate campaign (mux × seed).
- `nosettle_ctrl.sh` — no-settle control batch.
- `stress_until.sh` — retry the 100-iter stress until it clears iter 1 (or N attempts).
- `run_triage_sig.sh` / `live_hang_triage.sh` — live tt-triage capture of the hang signature.
- `verify_stall.sh` (in-container) + `catch_and_verify.sh` (host) — **stall verifier**: device-open
  inducer that, on a progress-stall, runs `tt-triage` against the LIVE wedged runtime BEFORE killing
  and reports `SIGMATCH` (writer.cpp:365 + devices 16,20,24,28). On a caught stall the host wrapper
  then resets and runs the EXACT unmodified single-shot test (`run_triage_sig.sh`) as a cross-check
  → classifies `CONFIRMED_BAD_REGIME` (single-shot also hangs) vs `STRESS_STALL_ONLY` (single-shot
  passes → stall was harness/device-open-specific, NOT the canonical deadlock). Built specifically to
  avoid repeating the 13:35–14:07 mistake of trusting an untriaged stall heuristic.

Hygiene protocol used throughout: kill own PID (TERM→KILL), verify no `moe_compute_smoke`/`tt-triage`
stragglers, verify no `/dev/tenstorrent/*` fd holders, clean container `/dev/shm/TT_UMD_LOCK.*` and
`tt_device_*_memory`, then `tt-smi -r` on the HOST between runs.

---

## Recommendations

1. Reframe issue #47523 as an **intermittent fused-combine deadlock (race), variable rate**, not a
   config/mux bug. Include the within-session rate-drift table as the key evidence.
2. To pin it down: long unattended stress that **keeps the device open** (no per-iter reset), with
   watcher enabled, run during a "bad window" to catch the first stall deterministically; correlate
   with `test_system_health` / fabric link telemetry across good vs bad windows.
3. Independent hardening (real bugs, not the hang): include `topology` in
   `SelectiveReduceCombineParams::attributes()`; have `moe_compute` validate mux geometry up-front
   (it already FATALs on too-few cores — good); investigate the post-reset `set_fabric_config`
   topology-mapping FATAL (fabric readiness after `tt-smi -r`).

---

## probe_now RESULT (current state, 16:16–16:26)

3 trials, `tt-smi -r` + 30s settle + single-shot default mux `(3,0,4,7)`:

```
PROBE trial1 -> VERDICT: PASS
PROBE trial2 -> VERDICT: PASS
PROBE trial3 -> VERDICT: PASS
```

**Not all tests are hanging now — currently PASSING 3/3.** This is the *fourth* regime flip observed
today (HANG 11:23–11:35 → PASS 11:41–13:17 → HANG 13:35–14:07 → PASS 16:16–16:26), with no workload
change between them. Confirms the intermittent, self-drifting nature of the hang.

---

## RESOLUTION (20:30) — root cause confirmed in-kernel + fix validated

The `SMOKE_REALLOC=inputs` knob gives a deterministic repro (independent of the regime drift above).
With it, the root cause is now pinned in-kernel and a robustness fix is validated.

### Root cause (confirmed by DPRINT)

The fused combine writer (`selective_reduce_combine/.../kernels/dataflow/writer.cpp`) iterates, per
expert `e`, `token_split_counts[e]` entries of `dense_token_maps` and searches the activations stream
for each token id `st`:
```cpp
while (expert_token_activations_ptr[0] != st) { ... ASSERT(guard++ < global_num_tokens); }  // was line 271/273
```
`dense_token_maps` = moe_compute tilize output `tilize_e_t_output_tensor`, the `[E,T]` expert→token
buffer **`-1`-padded** ("capped by -1 to indicate no more tokens", `moe_compute_program_factory.cpp:571`).
`token_split_counts` = tilize output `tilize_per_expert_total_tokens_output_tensor`.

When the dispatch inputs are recreated, the tilize stage emits a `(count, map, activations)` triple
that is internally inconsistent: `count[e]` over-runs the valid (non-`-1`) map entries, so the combine
reads the `-1` sentinel as a token id and searches forever. The release-compiled-out `ASSERT` was the
only loop exit → infinite spin → combine never reaches the ring barrier → whole-mesh deadlock.

### In-kernel proof (DPRINT, `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_CHIPS=all`)

`logs/fix_validate_20260624_202943.txt`, dev 24 (one of the originally-hung 16/20/24/28), expert 0:
```
24:5-4:BR: token-map desync expert=0 dt=0 missing_token_id=4294967295   (0xFFFFFFFF = -1 sentinel)
24:6-0:BR: token-map desync expert=0 dt=0 missing_token_id=33           (real token, activation absent)
24:6-4:BR: token-map desync expert=0 dt=5 missing_token_id=4294967295
```

### Fix (Bug A) — implemented + validated

`writer.cpp`: bounded the token search; on overrun it `DPRINT`s `(expert, dt, missing token id)`,
`ASSERT(false)` (debug), and `break`s so the core still reaches the ring barrier. Result:
- before: `SMOKE_REALLOC=inputs` → 100% HANG.
- after: **`SMOKE PASSED`** (no hang) + the desync is logged. (Output still wrong for dropped tokens —
  that requires Bug B, the tilize/dispatch count-vs-map consistency.)

### Still open (Bug B)

Why the tilize stage / `all_to_all_dispatch_metadata` produce an inconsistent count/map when inputs are
recreated (address+data byte-identical to a passing run — NOT an address/data bug). See `BUG_REPORT.md`
§7. The deterministic `SMOKE_REALLOC=inputs` repro + the DPRINT make this directly debuggable by the op
owners.
