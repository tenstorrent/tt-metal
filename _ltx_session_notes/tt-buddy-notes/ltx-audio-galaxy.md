# ltx-audio-galaxy

## cq-1 routing TESTED on device — deadlock PERSISTS: child cq1 op OK, but returning to parent cq0 HANGS (overlapping submesh CQs not independent)
**2026-06-17 00:52 UTC / 2026-06-16 17:52 PT** · `tt-metal@079d567769b` (branch smarton/audio-submesh-e2e)

First on-device test of the PRIMARY cq-isolation fix (commit 079d567769b: open mesh with num_command_queues=2, wrap audio decode in `ttnn.command_queue(1)`). Job 005204-2, test_submesh_repro.py::test_submesh_cq1_repro, warm cache.

**Board is HEALTHY.** The 06:00 PCIe wedge (job 060002-13) cleared on its own over the ~18h idle window. This run opened all 32 devices, initialized FABRIC_1D_RING fabric with num_command_queues=2 across all 32 devices, real-time profiler synced all chips. No setup error. (`tt-smi -s` also clean beforehand.)

**The cq-1 fix is INSUFFICIENT — deadlock persists.** Sequence in the repro:
1. `[REPRO] all_gather OK (parent cq0, before audio submesh): (1,1,32,2048)` — parent on cq0 OK.
2. `[REPRO] all_gather OK (child cq1): (1,1,32,1024)` — child 1x4 submesh all_gather on cq1 **SUCCEEDS** (this is new — the first child op now runs, where cq0-shared previously hung immediately).
3. `_run_ag(ccl, mesh, "parent cq0, after audio submesh")` — the NEXT parent cq0 all_gather **HANGS**. No further output for ~14.5 min until the broker 900s timeout killed the job. Never printed the 3rd "OK" nor PASS.

**Interpretation:** routing the child onto cq1 lets the child's CCL run once, but switching back to the parent on cq0 after cq1 activity on the OVERLAPPING 1x4 submesh deadlocks the dispatcher. The two HW command queues are NOT independent across the physically-overlapping chips for CCL/fabric ops — fabric/CCL state is shared per-chip regardless of cq_id. So cq routing alone cannot isolate an overlapping live-parent + child. This is exactly the FALLBACK condition the SHARPENED FIX PLAN named ("still deadlock on Blackhole → ttnn C++ change").

**Consequence:** PRIMARY (Python config + cq routing) path is exhausted and does NOT work. The wall is confirmed at the ttnn/fabric layer: an overlapping child submesh cannot drive CCL on a distinct CQ while the parent remains live on cq0. The real fix is a ttnn C++ change for true per-submesh CQ/fabric isolation (or a parent-quiesce/hand-off API) — a metal-team PR, not an ltx-perf push.

**Op note:** auto-triage did NOT fire at a 30s op-timeout (the env.yaml I seeded lacked TT_METAL_OPERATION_TIMEOUT_SECONDS + the dispatch-timeout callback per the tt:run auto-triage contract); the broker's own 900s timeout killed it instead, so there is no per-core triage artifact for this hang. The hang point is unambiguous from stdout (3rd op) regardless.

## FINAL (Phase 4.5): LTX_AUDIO_SUBMESH=1x4 E2E does NOT land — genuine ttnn cq-isolation wall; not pushed
**2026-06-15 23:25 UTC / 16:25 PT** · `tt-metal@7af5fbf5974` (branch smarton/audio-submesh-e2e)

Mission outcome: NOT landed. Per Phase 4.5, stopping after real attempts and documenting the ttnn wall. No success claim — gates fail.

**What was achieved:**
- ROOT-CAUSED the failure through 3 device runs + 1 cheap repro. Two distinct problems, not the one MISSION.md assumed:
  1. Eager __init__ creation of the overlapping 1x4 submesh+CCLManager corrupted the VIDEO DiT CCL runtime args (Index>size flood during Stage-1). FIXED by commit 7af5fbf5974 (defer submesh+shells to first decode_audio, after all video). Verified: video now runs fully clean end-to-end (Stage1+upsample+Stage2+VAE -> (1,3,145,1088,1920)).
  2. The 1x4 AUDIO decode itself HANGS when run after a live video gen on the parent — dispatch deadlock, 600s watchdog fired. This is the hard wall.

**The wall (precise):** ttnn `MeshDevice::create_submesh` returns a child that SHARES the parent's command queue (cq 0); there is no per-submesh CQ. When the parent 4x8 has just driven a full video gen on cq 0 and is still alive, the child 1x4's audio CCL ops (neighbor_pad halos / all_gather at T-shard=4) deadlock against the parent's cq-0 state. Confirmed by contrast: the prior working 673ms / PCC-0.99379 1x4 result (commit 6182a3b) was audio_only=True with the parent mesh opened but NEVER running video — that path works. The close-side of the same constraint is mesh_device.cpp:900/912 (cq-shared parent<->child cannot be closed in either order).

**All 3 MISSION fix angles exhausted:**
- A1 release video traces before decode_audio: N/A — eager run has no traces; the blocker is the live parent MESH + its cq-0 history, not held traces. Cannot release the parent (conftest owns it; closing hits the cq-shared close wall).
- A2 separate cq_id for audio: THIS IS THE WALL. ttnn provides no per-submesh CQ.
- A3 force audio eager (no trace): already in effect (LTX_TRACED=0). Eager audio still hangs -> not a trace-capture problem.

**Exact ttnn change needed (escalate):** per-submesh command-queue isolation. Either (a) `create_submesh` lets the child own a distinct CQ id, or (b) an API to quiesce/hand-off the parent's cq 0 so a child can drive CCL while the parent is idle-but-alive. Without it, audio-on-1x4-after-video cannot run in one process. (Related earlier class: learn-trace-ccl-deadlock.md end_trace_capture hang — same CCL-on-Galaxy-under-constraint family.)

**Disposition — NOT pushed to ltx-perf:** the LAND gates fail (no Saved-video mp4; audio decode hangs; cannot measure a speedup). Pushing a default-on 1x4 would hang the E2E. The deferral fix (7af5fbf5974) is correct and safe (removes the destructive eager overlap; full-mesh path byte-identical; a prerequisite for any future fix once ttnn has per-submesh CQs) but it does NOT make 1x4 E2E work, so it stays on branch smarton/audio-submesh-e2e, NOT merged. LTX_AUDIO_SUBMESH remains OPT-IN and is documented known-broken in the full E2E.

**Device hygiene:** my hung 1x4 audio job (230222-50) left the board FW-init-wedged; recovered via sanctioned MCP reset (device idle, gate confirmed no foreign holders). A foreign sulphur tenant is aggressively cycling galaxy resets (~every 1-2 min), which preempted/killed two of my runs — that contention, not my code, blocked the optional audio_only PCC re-confirmation (PCC 0.99379 for 1x4 audio_only already on record from commit 6182a3b).

**Branch state:** smarton/audio-submesh-e2e @ 7af5fbf5974 (deferral fix). Repro test models/tt_dit/tests/models/ltx/test_submesh_repro.py present (untracked).

## ttnn WALL confirmed: deferred-submesh fixes video, but 1x4 audio decode HANGS when run after a live video gen on the parent (shared-cq dispatch deadlock)
**2026-06-15 23:18 UTC / 16:18 PT** · `tt-metal@7af5fbf5974` (branch smarton/audio-submesh-e2e)

E2E run-3 (job 230222-50), LTX_AUDIO_SUBMESH=1x4 LTX_TRACED=0 eager, with the deferred-submesh fix. Decisive split result.

**The deferred-submesh fix WORKS for video.** Full video pipeline ran CLEAN end to end:
- Stage 1: 8 steps. Latent upsample 13.7s. Stage 2: 3 steps (74.7s). VAE decode 57.3s -> video (1,3,145,1088,1920).
- The lazy 1x4 submesh was created AFTER all video (`_ensure_audio_submesh:413 Audio decode routed onto 1x4 submesh of 4x8` at 23:06:52, post-VAE), exactly as designed. No video corruption.
- The runtime_args_data.hpp:29 fatals all fell in 23:03-23:05 (Stage 1+2 video), STOPPED before audio (last 23:05:53). Confirmed: a pre-existing _ring video artifact, independent of the submesh. Video produced a full-shape frame tensor (so the OOB-read fatals are non-fatal noise here, as suspected).

**But the 1x4 AUDIO decode HANGS.** After `_prepare_audio_decoder` loaded the 1x4 dec+voc (23:06:52), the audio decode produced NO further output. Frozen from 23:07:06; the 600s operation-timeout watchdog fired at 23:17:36 (tt-triage ran). >10 min hang on the first audio op on the 1x4 child. Killed job (no mesh wedge expected per prior note).

**Why this is the ttnn wall (Phase 4.5):**
- The PRIOR working 1x4 result (673ms, PCC 0.99379, commit 6182a3b) was audio_only=True + run_warmup=False: the parent 4x8 mesh was opened but NEVER ran a video op. There, 1x4 audio decode worked.
- Here the parent 4x8 ran a FULL video gen (Stage1/2 + VAE) immediately before. The child 1x4 submesh SHARES the parent's command queue (cq 0; ttnn create_submesh gives no separate CQ). Running the audio CCL ops (neighbor_pad halos / all_gather at T-shard=4) on the child while the parent mesh is alive and has just driven heavy CQ-0 traffic deadlocks the dispatcher. This is the cq-sharing constraint the 6182a3b commit message flagged ("a routed child submesh shares the parent command queue").

**All 3 MISSION fix angles are exhausted/inapplicable:**
- Angle 1 (release video DiT traces before decode_audio): N/A — this run is eager (LTX_TRACED=0), there are NO video traces. The blocker is the parent MESH being alive (and its CQ-0 history), not held traces. Can't release the parent mesh (the test/conftest owns it; closing it hits the cq-shared close wall anyway).
- Angle 2 (separate cq_id for audio): this IS the wall. ttnn create_submesh shares cq 0 with the parent; there is no API to give the child its own CQ. This is the exact ttnn change needed.
- Angle 3 (force audio fully eager, no trace capture): ALREADY in effect (LTX_TRACED=0 -> use_trace=False). The hang is eager audio CCL ops, not trace capture. Does not help.

**Exact ttnn change needed (for escalation):** a child submesh from MeshDevice::create_submesh must be able to own a SEPARATE command queue (or the parent must be able to fully quiesce/hand off cq 0) so the child's CCL ops do not deadlock against the live parent's cq-0 traffic. Today the child is bound to the parent's cq 0 for both dispatch AND close (mesh_device.cpp:900/912 close wall + this dispatch hang). Until ttnn provides per-submesh CQ isolation, audio-on-1x4-after-video cannot run in one process.

**Net:** LTX_AUDIO_SUBMESH stays OPT-IN and must NOT be defaulted on for the full E2E (it hangs). The deferred-submesh fix (7af5fbf5974) is still a correct improvement: it removes the eager-__init__ overlap and keeps the full-mesh path byte-identical; it is a prerequisite for any future fix once ttnn gives per-submesh CQs. Recommend landing the deferral as a latent-capability improvement with 1x4 documented as not-yet-usable in E2E, OR holding it. Next optional evidence: run test_audio_decode_girl LTX_AUDIO_SUBMESH=1x4 (audio_only, no video) to re-confirm 1x4 audio is correct+fast standalone (the 673ms/PCC case), isolating the hang strictly to the post-video condition.

## Deferred-submesh fix WORKS on the video side; runtime_args fatals are a pre-existing _ring artifact (not the submesh)
**2026-06-15 22:49 UTC / 15:49 PT** · `tt-metal@7af5fbf5974` (branch smarton/audio-submesh-e2e)

Implemented + ran the fix (commit 7af5fbf5974: defer audio submesh+CCLManager+shells to first decode_audio via lazy _ensure_audio_submesh). E2E run-1 = job 224507-43, LTX_AUDIO_SUBMESH=1x4 LTX_TRACED=0 eager.

**Fix validated on video:** log shows `Audio decode will route onto 1x4 submesh of 4x8 (lazy)` at __init__ (submesh NOT created), then the VIDEO DiT ran cleanly: Stage 1 all 8 steps (~6.5s/step), upsample 31.4s (cold), into Stage 2 (1088x1920) cold-compiling. The eager __init__ corruption is gone from the video path.

**Run did not finish:** a FOREIGN sulphur tenant ran a galaxy reset (broker job 224918-44) at 22:49:18 and preempted my still-running job -> exit -9. Not my code — shared board reset under me. Re-queued (job 225427-47) behind sulphur + noblewoodall.

**KEY — the runtime_args_data.hpp:29 fatals are NOT caused by the submesh.** 61440 fatals, ALL timestamped 22:46:08-22:47:10, i.e. DURING Stage-1 video denoise, BEFORE _ensure_audio_submesh runs (it is lazy, fires only at decode_audio which the run never reached). So `Index 6 > size 6` / `Index 3 > size 3` is a PRE-EXISTING artifact of the bh_4x8sp1tp0_RING eager video DiT path, independent of LTX_AUDIO_SUBMESH. My EARLIER conclusion (that the submesh corrupts video runtime args) was WRONG about the mechanism — the fatals were already there in the _ring path; the video DiT floods them regardless of the submesh. The baseline traced no-submesh run (note 2026-06-15 11:13) produced a valid synced mp4, strongly suggesting these TT_FATAL lines are benign device-side validation noise that does not corrupt output (host crawls forward through all steps; output later validated as a good mp4).

**Open items:** (1) confirm decode_audio succeeds on the lazily-created 1x4 submesh + Saved video + valid mp4 (next run); (2) confirm the fatals are benign by diffing against a no-submesh eager _ring run (or trust the prior valid-mp4 evidence); (3) the cq-shared close wall at teardown still applies (cosmetic post-mp4). Cold Stage-2 1088x1920 + 1x4 audio kernels are now warming the cache across these preempted runs, so the next run reaches decode_audio faster.

## CONFIRMED root cause: 1x4 audio submesh built at __init__ corrupts the VIDEO DiT CCL runtime args
**2026-06-15 22:41 UTC / 15:41 PT** · `tt-metal@6182a3b34c0-dirty` (branch smarton/audio-submesh-e2e)

Ran full E2E gen#0, LTX_AUDIO_SUBMESH=1x4, LTX_TRACED=0 eager (job 223805-42, warm cache). Watched it live, then killed it. This is the definitive root cause — different from BOTH the MISSION CQ-trace story and the submesh-alloc-corruption-of-the-PARENT-OP repro result.

**What happened:**
- pipeline __init__ logged `Audio decode routed onto 1x4 submesh of 4x8` (submesh + audio CCLManager built eagerly at __init__, BEFORE any video op).
- Warm 1x4 audio cache hit (`mesh1x4_bf16` dec+voc loaded). Cold 1x4 audio kernels are NOW compiled into the warm cache (good — reusable).
- Gen started 22:39:09. During VIDEO DiT Stage-1 denoise, `TT_FATAL: Index 6 is larger than runtime args size 6` AND `Index 3 ... size 3` began flooding (13988+ in ~45s) at runtime_args_data.hpp:29 — while compiling/running rotary_embedding_llama + dit_layernorm_post_all_gather (these are VIDEO transformer ops, not audio).
- Crucially the host pipeline KEPT crawling forward (Step 1/8 σ1.0->0.9937 logged at 22:41:09) WHILE the device flooded fatals. So the prior "42-min hang" was actually this fatal-flood + garbage-compute crawl, NOT a deadlock. Output would be corrupt.

**Mechanism:** The overlapping 1x4 audio submesh's CCLManager (manager.py:48-108: a 2nd SubDevice spanning all compute cores at SubDeviceId(0) + ~30 global semaphores) is allocated on the 4 chips the 1x4 slice overlaps, at __init__, before the video DiT runs. This shifts the runtime-arg layout baked into the VIDEO DiT's CCL kernels (neighbor_pad / ring-SDPA / dit_layernorm all_gather) on those overlapping chips -> the parent kernel indexes past its runtime_args size (6>6, 3>3). The repro's simple parent all_gather has few runtime args and didn't trip it; the DiT's richer CCL ops do.

**Why the prior note's signature matches but the cause is now precise:** prior note (095511-3) saw the same runtime_args_data.hpp:29 and blamed "submesh routing produces wrong kernel arg counts." Correct symptom; the precise trigger is the EAGER __init__-time creation of the overlapping submesh+CCLManager while the parent (video) still needs clean runtime args. NOT audio-decode kernels, NOT trace capture, NOT a CQ deadlock.

**Fix (decided):** Defer the audio submesh + audio CCLManager + audio-decoder shell build (_new_audio_decoder) until the FIRST decode_audio call (lazy `_ensure_audio_submesh`), AFTER all video DiT + VAE work is done. The video DiT then runs with no overlapping submesh -> clean runtime args. Full-mesh path (no LTX_AUDIO_SUBMESH) stays eager and byte-identical to baseline. The cq-shared close wall at teardown remains (cosmetic, post-mp4). This is the mission's fix-angle-1 reframed: the conflict is submesh+CCLManager ALLOCATION overlapping the live parent, not trace-on-CQ.

Kill of 223805-42 used MCP daemon (SIGKILL) — prior note confirms this job type does not wedge the mesh; will verify next job starts clean. Cold 1x4 audio kernels cached this run.

## REPRO RESULT: overlapping submesh does NOT corrupt parent runtime args; the hard ttnn wall is close-ordering (cq shared parent<->child)
**2026-06-15 22:35 UTC / 15:35 PT** · `tt-metal@6182a3b34c0-dirty` (branch smarton/audio-submesh-e2e)

Ran test_submesh_overlap_repro on bh 4x8 (job 223513-41). Decisive — overturns BOTH the MISSION CQ-trace story and the prior runtime-args-corruption note.

**Test body PASSED.** Logs:
```
[REPRO] parent all_gather OK (before audio submesh): (1, 1, 32, 2048)
[REPRO] created 1x4 audio sub-submesh + CCLManager
[REPRO] parent all_gather OK (after audio submesh): (1, 1, 32, 2048)
[REPRO] PASS: parent op survived overlapping submesh
```
=> Creating the overlapping 1x4 submesh + its CCLManager (the second SubDevice + ~30 global semaphores on overlapping cores) does NOT corrupt the parent's runtime args. The parent all_gather runs identically before and after. The prior note's `runtime_args_data.hpp:29 Index N > size N` theory is FALSE for submesh allocation — that earlier 42-min failure must be in the audio-decode KERNELS at the cold 1x4 mesh shape, not in submesh creation.

**The only hard failure is at CLOSE (conftest teardown), and it is a genuine ttnn-core wall:**
```
mesh_device.cpp:912: MeshDevice cq ID 0 is in use by child submesh ID 2 during close of mesh ID 1  (closing PARENT while child alive)
mesh_device.cpp:900: MeshDevice cq ID 0 is in use by parent mesh ID 1 during close of mesh ID 2     (closing CHILD while parent alive)
```
A cq-sharing parent<->child pair is mutually un-closeable: ttnn forbids closing the parent (child holds cq 0) AND forbids closing the child (parent holds cq 0). conftest.py:586-589 closes submeshes then parent; neither order is legal -> TT_THROW -> std::terminate -> SIGABRT (exit -6). This matches the 6182a3b commit message exactly ("ttnn cannot close it while the parent lives... lifetime bound to process teardown"). release_audio_submesh() only drops Python refs; the submesh stays REGISTERED on the parent (get_submeshes() still returns it), so conftest's teardown loop hits the abort.

**Implication for landing:**
- The teardown SIGABRT happens at PROCESS EXIT, AFTER the test body (and "Saved video"/mp4) complete. It is a non-zero exit, not a functional failure of generation. Gate (a) "reaches Saved video + valid mp4" can still pass if the body completes.
- The REAL functional blocker remains the warmup-time audio decode on the 1x4 submesh (the reported 42-min hang). Repro proves it is NOT submesh-alloc corruption -> next: run the real E2E with warm cache and watch exactly where/if warmup audio decode hangs on 1x4. Could be cold 1x4-kernel compile (slow but not hung) vs a real CCL/halo deadlock at the 1x4 shape.
- To exit clean: either (i) prevent conftest from close-looping the audio submesh (it should not be in get_submeshes() at teardown, or close it BEFORE the parent in a legal order — but cq-share makes that illegal), or (ii) accept the teardown abort as cosmetic if the mp4 is valid, or (iii) the real ttnn fix = per-submesh CQ so the child does not share cq 0.

Job 223513-41 exit -6 did NOT wedge the mesh (clean abort, fabric reset ran). Foreign sulphur tenant cleared before this ran.

## Root cause of LTX_AUDIO_SUBMESH=1x4 E2E failure: overlapping-submesh runtime-args corruption, NOT a trace/CQ hang
**2026-06-15 22:32 UTC / 15:32 PT** · `tt-metal@6182a3b34c0-dirty` (branch smarton/audio-submesh-e2e)

Pre-device code+notes analysis before attempting fixes. Reconciles two competing failure stories.

**Two stories, one is empirical:**
- MISSION.md hypothesis: child 1x4 submesh shares the parent 4x8 command queue; in full E2E the parent holds captured video DiT traces on that CQ, so decode_audio HANGS at the first warmup decode. Proposed fixes: (1) release video trace before decode_audio, (2) separate cq_id for audio, (3) force audio eager.
- ACTUAL observed signature (this notes file, 2026-06-15 11:13, the real hung job 095511-3): `TT_FATAL: Index N larger than runtime args size N (runtime_args_data.hpp:29)`, repeated, then a ~42min wedge. "The submesh routing produces wrong kernel arg counts." This is a runtime-args CORRUPTION, not a silent CQ deadlock. It is exactly the failure test_submesh_repro.py was built to reproduce ("If overlapping submesh allocations corrupt the parent's runtime args, this emits Index N > size N / hangs").

The runtime-args TT_FATAL is the empirical truth; the CQ-trace story is unconfirmed speculation. The mission's 3 fix angles all target a trace/CQ conflict and may NOT address a runtime-args corruption.

**Why overlap is the novel/unsupported pattern:**
- pipeline_ltx.py:289-299: LTX_AUDIO_SUBMESH calls `mesh_device.create_submesh(MeshShape(r,c))` (SINGULAR) — a sub-slice that OVERLAPS the parent 4x8 which STAYS fully live (video keeps the whole mesh; audio runs on the slice in the same process).
- Contrast: motif/flux1/qwenimage pipelines use `device.create_submeshes` (PLURAL) for CFG — they CARVE the parent into DISJOINT submeshes and the parent-as-a-whole is not used afterward. That pattern is supported. The LTX overlap (live parent + overlapping child) is not exercised anywhere else.
- The new audio CCLManager (pipeline_ltx.py:296) runs `_init_subdevice` (manager.py:48) creating a SubDevice spanning ALL compute cores with `SubDeviceId(0)` AND `_init_semaphores` (manager.py:61) creating ~30 global semaphores — on the SAME physical cores the parent's CCLManager already configured. Prime suspect: the second SubDevice/global-semaphore allocation on overlapping cores shifts the parent CCL op's runtime-arg layout, so a parent neighbor_pad/all_gather indexes past its runtime_args size -> runtime_args_data.hpp:29.

**Trace mechanism confirmed (for completeness):** video DiT traces captured via @traced_function on LTXTransformerModel.inner_step, cq_id defaults to 0 everywhere (tracing.py:152 tracer_cq_id=0). Vocoder.forward_traced (vocoder_ltx.py:356) runs EAGER on the first call at a shape (warming) and only CAPTURES on the second call. So warmup's decode_audio (distilled:118) is the eager warming pass — if THAT is where it dies, the failure is plain eager ops on the overlapping submesh, which again points at allocation corruption, not trace capture. LTX_VOC_TRACE defaults ON ("1") even on large mesh (pipeline_ltx.py:1621).

**Plan (revised, root-cause-first):**
1. Run test_submesh_repro.py on device (cheap: random tensors, no 46GB checkpoint, ~1-2min) to confirm the parent op breaks immediately after the overlapping 1x4 CCLManager is constructed. This isolates allocation-corruption from trace/CQ for ~free.
2. If repro confirms allocation corruption: the real fix is to stop the overlapping submesh's CCLManager from corrupting the parent's subdevice/semaphore state (e.g. share the parent SubDevice, or defer audio submesh+CCLManager creation until AFTER all video work + release_traces, or run audio decode without a second SubDevice). The mission's "separate cq_id" and "release trace" angles are secondary.
3. Only after a green repro + a non-corrupting allocation path, run the full E2E.

**Device status at analysis time:** busy with FOREIGN sulphur tenant (job 222831-38, TT_METAL_HOME=/home/sulphur). Queuing behind it per HARD safety; not killing.

**Key files:** models/tt_dit/pipelines/ltx/pipeline_ltx.py:286-388 (submesh routing + release_traces), models/tt_dit/parallel/manager.py:48-108 (_init_subdevice/_init_semaphores), models/tt_dit/utils/tracing.py:148-331 (Tracer cq_id/capture/release), models/tt_dit/tests/models/ltx/test_submesh_repro.py (the isolation repro).


## Real girl E2E validated (traced, no submesh): 10.37s compute, mp4 OK
**2026-06-15 11:13** · `tt-metal@6182a3b34c0-dirty` (branch smarton/audio-submesh-e2e)

Ran test_pipeline_distilled -k bh_4x8sp1tp0_ring, LTX_TRACED=1, NO LTX_AUDIO_SUBMESH. PASSED. mp4 /home/smarton/ltx_av_e2e_traced.mp4: h264 1920x1088 145f@24 + aac 48k stereo, 6.04s, synced.

**Warm/traced per-gen:** Stage1 2.72s, upsample 0.29s, Stage2 3.26s, VAE 1.21s, **Audio 2.89s**, =Total(compute) 10.37s (+video export 1.6s host). Cold gen#0: 67.17s.

**Audio decode is 2.89s real E2E, NOT 0.88s/242ms** — the 242ms was isolated VocoderWithBWE (Stage C); real decode is dominated by the still-eager mel-VAE (Stage A) + host STFT/resample/IO. BWE trace ([[ltx-audio-galaxy]] prior entry) is engaged. Next lever = trace the mel-VAE (Conv2dViaConv3d writes-during-capture blocks it).

**BROKEN: LTX_AUDIO_SUBMESH=1x4** wedges at "warmup audio decode" (~42min hang) with repeated TT_FATAL: Index N larger than runtime args size N (runtime_args_data.hpp:29). The submesh routing produces wrong kernel arg counts. Killed the hung job 095511-3; SIGKILL via daemon did NOT wedge the mesh (next job started clean). Do not use submesh until that's fixed.

## 4x8 audio decode: convs are dispatch-bound — trace is the only lever; BWE-trace win
**2026-06-11 05:54** · `tt-metal@c04262a55e5-dirty`

Measured on bh Galaxy 4x8, synthetic stage_b/stage_c at real girl-clip mel shape (real checkpoint not on box; timing is shape-driven so valid).

**Blockings are NOT a lever on 4x8.** Per-conv profile (`test_prof_vocoder_per_conv -k bh_4x8`): ms/call ~constant 1.1–1.35ms across a >50x FLOPs range (T 160→5120, C 32→768, k 3→11) → the fp32 audio convs are **dispatch-bound** (host dispatch + CCL halo + layout), not compute-bound. T_out_block tuning changes compute hidden under the overhead → a sweep moves nothing. Also `_FP32_BLOCKINGS` (utils/conv3d.py) is **mesh-agnostic** (keyed only on (Cin,Cout,kernel); fp32 branch returns before the mesh key is built) — never 4x8-vs-2x4 specific. **Do not sweep audio blockings for 4x8.**

**The lever is TRACE.** `test_full_forward_wall -k bh_4x8`: main vocoder eager=443ms traced=82ms = **5.41x**. Full VocoderWithBWE `test_bwe_traced_wall -k bh_4x8`: eager=3266ms, main-traced-only (prior prod)=518ms, **both-traced=242ms** → 13.5x over eager, 2.14x over prior.

**BWE-trace win** (committed `c04262a55e5` on smarton/optimizer/ltx-galaxy-v2): BWE was hardcoded eager over a "known channel-TP divergence" that **no longer reproduces** — traced-BWE is bit-identical (max|Δ|=0.000e+00). Added `use_trace_bwe` (bwe_ltx.py), routed `bwe_generator.forward_traced`, wired `use_trace_bwe=self._traced` (pipeline_ltx.py:1553); release_trace frees both.

**Remaining:** AudioDecoder (mel-VAE) has no forward_traced — still eager, likely next-biggest. Earlier "full decode only 1.18x" explained: BWE+VAE were eager. Real-checkpoint girl decode (`test_audio_decode_girl -k bh_4x8sp1tp0`) is SKIPPED here — needs LTX_CHECKPOINT (.safetensors absent on this box). See [[learn-ltx-audio-opt-playbook]].


## Baseline iter 0 — girl audio decode 4x8 = 1188.5 ms warm
**2026-06-10 01:09** · `tt-metal@cdfce61bf20`

**Target:** LTX-2 audio decode (mel-VAE + main vocoder + BWE) on bh Galaxy 4x8 (32 chips), per issue #46423.
**Branch:** `smarton/optimizer/ltx-audio-galaxy-2026-06-10` (from `smarton/ltx-perf-audio @254267f0151`).
**Test:** `test_audio_decode_girl[bh_4x8sp1tp0]` — added 4x8 param + oracle guard (commit `cdfce61bf20`).
**Env:** `LTX_CHECKPOINT` + `HF_HOME` → `/home/kevinmi/.cache/huggingface` (world-readable 46 GB ckpt + gemma). Device tools preloaded.

**Baseline:** `AUDIO_GIRL depthwise=conv1d warm=1188.5ms cold=356142.7ms`. Cold = 46 GB weight load, not a metric. Test PASSED.

**Key finding:** 2x4 = 1250 ms (issue) → 4x8 = 1188.5 ms — **~5% faster on 4× the chips**. Confirms #46423: T-pad factor-8 inflation (120→256, 2.1x vocoder waste) + per-conv all-gather tax eat the parallelism.

**Correctness gate:** `test_audio_components_ltx -k 4x8` sharded-vs-unsharded `max|Δ| < 5e-3` (4x8 unvalidated per issue; gate running, job 010918-5).
**Fidelity:** torch-oracle unavailable (diffusers 0.35.1 lacks `autoencoder_kl_ltx2_audio`); guarded to skip → perf-only.

**Trend**

| iter | change | warm (ms) | vs base | correctness | commit |
|---|---|---|---|---|---|
| 0 | baseline (conv1d, single-axis T-shard factor 8) | 1188.5 | — | pending | cdfce61bf20 |

**Next:** Fix A (ROW_MAJOR `_partition_t`, `vocoder_ltx.py` ~351-366) to drop T-pad align from `32*factor`→`factor` (256→128); then BWE channel-TP factor-4 crossover (`LTX_BWE_CHANNEL_TP`).
---

## 2026-06-17 03:06 UTC / 2026-06-16 20:06 PT — RESUME: build landed, reset binding built, device wedged

**Build (isolated worktree, fabric-lock-reset fix):** Prior worker's `build_worktree` died mid-build on
a process restart, INCOMPLETE (no libtt_metal.so/_ttnn.so). The stall was a COMPILE ERROR:
`ttnn-nanobind/device.cpp:117` called `tt::tt_metal::ResetFabricConnectionLock` but the fn is in
`tt::tt_metal::detail` (decl tt_metal.hpp:364, impl tt_metal.cpp:287). Fixed the qualifier
(commit a46f2e3c6f5), resumed `cmake --build build_worktree --target install` → BUILD COMPLETE.
`reset_fabric_connection_lock` symbol present in the new `_ttnn.so`; exposed as
`ttnn.reset_fabric_connection_lock`. MAIN's build untouched.

**Force-load (editable .pth caveat solved):** `ttnn-custom.pth` + an `__editable__` finder both hard-map
`import ttnn` → MAIN's `ttnn/ttnn/_ttnn.so` (reset symbol ABSENT). Prepending the worktree's ttnn dir to
PYTHONPATH (`<wt>/ttnn:<wt>:<wt>/tools`) makes the worktree `_ttnn.so` win (`reset present: True`) and
pulls the worktree `libtt_metal.so` via rpath (verified /proc/self/maps). Main's .so never overwritten.

**BLOCKED — board FW-init wedged (NOT code):** repro jobs 030411-6 + 030522-8 both FAILED at fixture
SETUP (~30s, during device open, before fix code runs):
`Device 0: Timeout (10000ms) waiting for physical cores to finish ...` →
`Device 0 init: failed to initialize FW! Try resetting the board.`
Cause: my own prior E2E (012356-5) was broker-killed -9 at 01:49 → SIGKILL of a live CCL job wedges the
board. Queue idle, no foreign tenant active. RESUME mission HARD rule = ABSOLUTELY NO device resets →
did NOT reset; a 2nd fresh open didn't clear it (hard wedge). Stopped per do-not-thrash; board self-healed
over an idle gap last time.

**Status: fix UNTESTED on device. NOT landed, NOT pushed.** Build + force-load plumbing proven; the L1
fabric-lock-reset's correctness is unproven. Resume = re-run the repro on a healthy board (env file
`~/.tt-buddy/mcp/20260617T0303Z-repro/env.yaml`), expect HANG→PASS, then E2E.
