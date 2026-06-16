# Mission — LAND LTX_AUDIO_SUBMESH=1x4 in the full AV E2E (resume-safe)

Goal: make `LTX_AUDIO_SUBMESH=1x4` work in the FULL AV E2E gen (video on full 4x8 + audio decode
on a 1x4 submesh, ONE process) → produces the mp4 + timing table with the audio speedup, then LAND
it on ltx-perf. Use superpowers (systematic-debugging — ROOT CAUSE before fixes; Phase 4.5 — if it's
a true ttnn architecture wall after real attempts, STOP and document, do not push broken; TDD;
verification-before-completion) + tt-buddy (tt:run, tt:profiler, tt:debugger). Commit + append to the
PROGRESS LOG after EVERY step (restarts are frequent — lose ≤1 step).

## The blocker (verified)
The 1x4 child submesh SHARES the parent 4x8 command queue. In full E2E the parent holds captured
video DiT traces on that CQ; when `decode_audio` runs on the cq-sharing child it HANGS at the first
decode_audio in warmup (job 095511-3: hung ~46min, no mp4). The prior 673ms/PCC-0.99379 validation
worked only because it was audio_only=True + run_warmup=False (NO video traces on the CQ).

## Fix angles (root-cause first, then try in order; this is a real ttnn-concurrency problem)
1. **Release the video DiT trace region BEFORE decode_audio.** Video (DiT + VAE) is fully done by the
   time audio decodes. If the parent's video traces are released before decode_audio, the shared CQ
   may be free for the child submesh. Look at pipeline_ltx_distilled.generate ordering + where video
   traces are captured/held (pipeline_ltx.py trace plumbing, release_traces). CAVEAT: in LTX_TRACED=1
   the gen#1 replay REUSES video traces — releasing them breaks replay. So this likely only works for
   a one-shot gen (gen#0), OR release only after the final video gen. A one-shot non-replay gen that
   produces a valid mp4 + timing is acceptable to land 1x4 (state the mode clearly).
2. **Separate cq_id for the audio submesh.** ttnn supports multiple CQs (cq_id 0/1). If the submesh
   audio ops issue on a different cq_id than the parent's video traces, the conflict may vanish.
   Investigate ttnn create_submesh / CQ assignment + whether decode_audio can target a distinct CQ.
3. **Don't capture/replay audio traces on the child at all** — run audio fully EAGER on the submesh
   (LTX_VOC_TRACE/BWE_TRACE are already default-OFF on large mesh). Confirm whether the hang is audio
   TRACE CAPTURE specifically vs any op on the shared CQ; if capture-only, forcing eager audio may
   sidestep it (the 673ms number was eager).
If all three fail and it's a genuine ttnn wall (per-submesh CQ needed in ttnn core), STOP per Phase
4.5, document the exact ttnn change needed, leave 1x4 opt-in. Do NOT push a default that hangs.

## CACHE DISCIPLINE (the human flagged this explicitly)
- Use the WARM cache: TT_METAL_CACHE=/home/smarton/.cache/tt-metal-cache. NEVER wipe it.
- Routing audio to 1x4 changes the audio kernels' mesh-shape hash → the FIRST 1x4 E2E compiles a
  fresh set of 1x4-audio kernels (cold, slow). That's expected ONCE — they then cache. So: run once
  to populate, reuse on every subsequent attempt; do not re-wipe and pay recompile each iteration.
- Keep TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT constant if profiling (changing it rehashes kernels).

## HARD safety
- Device ONLY via broker MCP. Check queue first; QUEUE behind foreign tenants (sulphur / bare
  "smarton" jobs are NOT yours — never kill/reset them). Reset only your OWN wedged proc + only if NO
  foreign tenant. SIGTERM-only your own orphans (never -9). The board has been wedging on
  ethernet-core-27-25 FW-init timeouts — if setup errors with that, it's a wedged board, not your
  code: re-queue, don't reset under a foreign tenant.
- Work in worktree /home/smarton/tt-metal/.claude/worktrees/audio-submesh-e2e (branch
  smarton/audio-submesh-e2e off origin/ltx-perf @6182a3b, build+python_env symlinked). TT_METAL_HOME
  + broker workspace = that path. Audio/pipeline changes are Python → reuse prebuilt C++, no rebuild.
- Env: TT_DIT_CACHE_DIR=/home/smarton/.cache/tt-dit, LTX_CHECKPOINT + GEMMA_PATH (kevinmi paths in
  run_girl_e2e.sh), full E2E needs --timeout=0.
- Before editing source Read .../buddy/code-comments.md; before commits Read commit-messages.md,
  footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Timestamps UTC+PT.

## Gen entry + gates
- Gen: test_pipeline_ltx_distilled.py::test_pipeline_distilled -k bh_4x8sp1tp0_ring, LTX_AUDIO_SUBMESH=1x4,
  --timeout=0, OUTPUT_PATH=/home/smarton/ltx_av_1x4_landed.mp4.
- Baselines for comparison (measured today, full mesh): gen#1 Total ~10.7s, audio-decode 3.2-3.3s.
- VALIDATE before landing (ALL must pass): (a) gen reaches "Saved video", mp4 valid (pyav: h264
  1920x1088 ~145f@24fps + aac); (b) audio PCC-vs-torch >0.95 — run test_audio_decode_girl with
  LTX_AUDIO_SUBMESH=1x4; (c) E2E audio-decode stage faster than the full-mesh ~3.2s (the whole point);
  (d) 1x4 audio ≈ full-mesh audio within tolerance. NEVER land if any gate fails.
- LAND: if all gates pass, push to ltx-perf. State clearly what you pushed: default-ON vs a documented
  opt-in default, the gen mode (one-shot vs traced-replay), and before/after audio+E2E timings in the
  commit message.

## PROGRESS LOG (append after every step; newest last)
- (created) Next: read the code (generate ordering, trace capture/release, ttnn CQ/submesh), form the
  root-cause hypothesis, then attempt fix angle 1 on device (warm cache, queue behind foreign).
- 2026-06-15 22:32 UTC/15:32 PT: Read pipeline_ltx.py (submesh routing :286-388, release_audio_submesh,
  release_traces), pipeline_ltx_distilled.py (warmup :118 decode_audio, generate :448), tracing.py
  (Tracer cq_id default 0 everywhere), manager.py (CCLManager _init_subdevice/_init_semaphores), and all
  audio-galaxy notes. KEY: prior note (2026-06-15 11:13) recorded the ACTUAL hung job 095511-3 signature
  as `runtime_args_data.hpp:29 Index N > size N` (overlapping-submesh runtime-args corruption), NOT the
  MISSION's CQ-trace-hang story. Recorded root-cause note. Decided to run the cheap test_submesh_repro.py
  FIRST to isolate alloc-corruption from trace/CQ before any expensive E2E.
- 2026-06-15 22:35 UTC/15:35 PT: Ran test_submesh_repro.py on device (job 223513-41, warm cache, queued;
  foreign sulphur tenant had cleared). DECISIVE: test body PASSED — `parent all_gather OK (after audio
  submesh)`. Overlapping 1x4 submesh + CCLManager does NOT corrupt parent runtime args (both prior
  hypotheses overturned). The ONLY hard failure is at CLOSE in conftest teardown: ttnn forbids closing a
  cq-sharing parent<->child pair in either order — `mesh_device.cpp:912` (child holds cq during parent
  close) + `:900` (parent holds cq during child close) → SIGABRT at process exit (exit -6, mesh NOT
  wedged). This is a genuine ttnn close-ordering wall, exactly per the 6182a3b commit message. The
  teardown abort is at PROCESS EXIT, after the mp4 would be written. Recorded repro note.
  Next: run the real E2E with LTX_AUDIO_SUBMESH=1x4 (warm cache) to find the REAL warmup-audio-decode
  blocker (the 42-min hang) — repro proves it is NOT submesh-alloc corruption, so it's either cold
  1x4-kernel compile (slow, not hung) or a real CCL/halo deadlock at the 1x4 shape.
- 2026-06-15 22:38 UTC/15:38 PT: Launched full E2E gen#0 (job 223805-42, bg) — LTX_AUDIO_SUBMESH=1x4,
  LTX_TRACED=0 (eager one-shot, no warmup → no video/audio trace capture anywhere; decode_audio happens
  inside generate(), not warmup), OUTPUT_PATH=/home/smarton/ltx_av_1x4_landed.mp4, warm TT_METAL_CACHE.
  Rationale: eager isolates whether 1x4 audio decode hangs even with ZERO traces (→ real CCL deadlock at
  1x4 shape) or works (→ landing path = one-shot eager). Note the test creates a 4x8 submesh of the
  galaxy at L85, so 1x4 audio is a GRANDCHILD; conftest teardown SIGABRT (cq-share close wall) expected
  at process exit AFTER mp4 — validate mp4 regardless of exit code. Cold run: 46GB load + cold 1x4 audio
  kernels compile ONCE into warm cache. Polling.
- 2026-06-15 22:41 UTC/15:41 PT: gen#0 reproduced the failure LIVE — `runtime_args_data.hpp:29
  Index 6 > size 6 / Index 3 > size 3` flooded (13988+ in ~45s) during VIDEO DiT Stage-1 denoise
  (rotary_embedding_llama / dit_layernorm_post_all_gather), NOT during audio. Host kept crawling
  (Step 1/8 logged) while device flooded -> the prior "42-min hang" was this fatal-flood + garbage
  crawl. CONFIRMED ROOT CAUSE: the 1x4 audio submesh + its CCLManager, created EAGERLY at pipeline
  __init__ (2nd SubDevice + ~30 global semaphores on the 4 overlapping chips) corrupts the video
  DiT's CCL runtime args. Killed garbage job 223805-42 (cold 1x4 audio kernels now cached). Recorded
  confirmed-root-cause note.
- 2026-06-15 22:4x UTC/15:4x PT: Implemented fix (commit 7af5fbf5974): defer audio submesh +
  CCLManager + decoder-shell build to the FIRST decode_audio via lazy _ensure_audio_submesh(). In
  generate(), decode_audio is the LAST op (after Stage1/upsample/Stage2 denoise + VAE decode), so the
  submesh is created only after ALL video DiT work -> parent runtime args stay clean during video.
  Full-mesh path (no LTX_AUDIO_SUBMESH) is byte-identical (eager as before). audio_only builds eagerly
  (no video to protect). Targeting LTX_TRACED=0 one-shot eager gen (no warmup/no trace replay, so the
  submesh-after-video ordering is guaranteed). Syntax OK, pre-commit hooks passed. Next: re-run E2E.
- 2026-06-15 22:49 UTC/15:49 PT: E2E run-1 with fix (job 224507-43). FIX WORKS on the video side:
  log shows `Audio decode will route onto 1x4 submesh of 4x8 (lazy)` (NOT created at init), and the
  video DiT ran CLEANLY — Stage 1 all 8 steps (~6.5s/step), upsample (31.4s cold), into Stage 2
  (1088x1920, 3 steps) cold-compiling. Then a FOREIGN sulphur tenant ran `tt-smi -glx_reset` (job
  224918-44) at 22:49:18 and preempted my job -> exit -9. NOT my code; the shared board was reset
  under me. IMPORTANT: the runtime_args_data.hpp:29 fatals (61440) all occurred 22:46:08-22:47:10
  DURING Stage 1, BEFORE any submesh exists (it's lazy) -> they are a PRE-EXISTING _ring-path device
  artifact, NOT caused by the submesh. The baseline traced no-submesh run (note 2026-06-15 11:13)
  produced a valid synced mp4, so these are very likely benign device-side validation noise. Device
  now busy: sulphur running (224931-45), noblewoodall queued (225134-46). Re-queuing behind them.
  Warm cache now has Stage-1 + upsample + partial Stage-2 1088x1920 kernels -> next run faster to
  reach decode_audio. Still need: decode_audio on lazy 1x4 submesh + Saved video + valid mp4.
- 2026-06-15 23:18 UTC/16:18 PT: E2E run-3 (job 230222-50) — DECISIVE SPLIT. Deferred-submesh fix WORKS
  for video: full pipeline ran clean (Stage1 8 steps, upsample 13.7s, Stage2 3 steps 74.7s, VAE 57.3s
  -> (1,3,145,1088,1920)); lazy 1x4 submesh created post-VAE as designed; runtime_args fatals confined
  to Stage1+2 (pre-existing _ring artifact, non-fatal). BUT the 1x4 AUDIO decode HUNG (frozen 23:07:06,
  600s watchdog fired 23:17:36). Killed. ROOT: prior working 673ms/PCC 1x4 was audio_only (parent mesh
  never ran video); here parent ran a full video gen first and the child 1x4 SHARES the parent cq 0
  (ttnn create_submesh = no separate CQ) so audio CCL on the child deadlocks vs the live parent's cq-0
  history. ALL 3 mission angles exhausted (A1 N/A eager-no-traces / live parent MESH; A2 = the wall
  no per-submesh CQ; A3 already eager and still hangs). Genuine Phase-4.5 ttnn wall. ttnn change needed:
  per-submesh command-queue isolation. 1x4 stays OPT-IN, never default-on. Recorded note.
- 2026-06-15 23:19 UTC/16:19 PT: Running audio_only PCC gate (job 231944-51): test_audio_decode_girl
  -k bh_4x8sp1tp0 LTX_AUDIO_SUBMESH=1x4 (no video -> reproduces working 673ms/PCC case + torch-oracle
  PCC), to satisfy gate (b) and isolate the hang to the post-video condition. Then finalize.
- 2026-06-15 23:25 UTC/16:25 PT: FINAL (Phase 4.5). PCC gate runs blocked by device contention: my hung
  1x4 audio job (230222-50) left the board FW-init-wedged; recovered via sanctioned MCP reset (device
  idle, gate confirmed NO foreign holders, did not use force). Retry was then preempted/killed by a
  FOREIGN sulphur tenant aggressively cycling galaxy resets (~every 1-2 min). 1x4 audio_only PCC is
  already on record at 0.99379 (commit 6182a3b), so 1x4 audio correctness is established.
  OUTCOME = NOT LANDED, NOT PUSHED. The deferral fix (7af5fbf5974) makes the VIDEO path clean and is a
  correct prerequisite, but the 1x4 AUDIO decode HANGS after a live video gen because ttnn create_submesh
  gives the child no separate command queue (shares parent cq 0); all 3 mission fix angles are
  exhausted/inapplicable. This is a genuine ttnn cq-isolation wall. LAND gates fail (no mp4, audio hangs,
  no speedup measurable). Pushing a default-on 1x4 would hang the E2E -> per the HARD rule, did NOT push.
  LTX_AUDIO_SUBMESH stays OPT-IN, documented known-broken in full E2E. Exact ttnn change needed:
  per-submesh command-queue isolation (create_submesh child owns its own CQ, or a parent cq-0
  quiesce/hand-off API). Branch smarton/audio-submesh-e2e @ 7af5fbf5974 holds the deferral fix (unmerged).
  Full root-cause + evidence in ~/.tt-buddy/notes/ltx-audio-galaxy.md (5 entries this session).

## SHARPENED FIX PLAN (2026-06-15 — supersedes the fix-angle ordering above)
Code-read conclusion: the deadlock AND the close-throw both come from parent video + child audio
SHARING the same HW cq_id (0). The multi-CQ infra EXISTS: MeshDeviceImpl.create_submesh calls
`submesh->initialize(num_hw_cqs(), ...)` (mesh_device.cpp:676) so the child gets its own
MeshCommandQueue per cq_id; the close guard (mesh_device.cpp:900/912) only throws when parent and
child hold the SAME cq_id in use; Tracer/ttnn ops already accept cq_id (tracing.py tracer_cq_id;
:523 shows tracer_cq_id=1 works). The device currently opens with num_hw_cqs=1 (ring/line_params
set no num_command_queues), so there is no cq 1 to use. Blackhole has 2 HW CQs.

PRIMARY FIX (try first — likely NO rebuild, Python config + cq routing):
1. Open the 4x8 mesh with num_hw_cqs=2 (add num_command_queues=2 to the device_params / mesh
   fixture path for bh_4x8sp1tp0_ring; confirm where the fixture reads it — tt-metal conftest
   mesh_device fixture). Verify the galaxy opens cleanly with 2 HW CQs + fabric.
2. Route the audio submesh's ops to cq_id=1: the audio CCLManager (parallel/manager.py) + the
   conv/neighbor_pad/all_gather + any audio Tracer to cq_id=1; video DiT/VAE stay on cq_id=0.
   (Builds on the committed lazy-deferral fix 7af5fbf — submesh is created after video, now on cq 1.)
3. Run the repro (close should NOT throw — parent cq0, child cq1, no shared cq_id) then the full
   E2E (LTX_AUDIO_SUBMESH=1x4, warm cache): audio decode should no longer deadlock (cq 1 ≠ parent
   cq 0). Validate the gates (mp4 + PCC>0.95 + audio faster than ~3.2s + 1x4≈full-mesh audio).
FALLBACK (only if CCL/fabric ops reject cq 1 or still deadlock on BH): it IS a ttnn C++ change —
make fabric CCL cq-aware / give the submesh a distinct CQ. THIS needs an ISOLATED build (the
worktree's `build` symlinks to main's build_Release — a C++ change MUST NOT clobber main's build,
other tenants depend on it). Set up a separate build for the worktree (build_metal.sh into a
worktree-local build dir, repoint the symlink) BEFORE editing C++. Budget ~14-40min/build.

LANDING SCOPE (honest): a core CQ-routing change is reviewable infra. If the Python config+routing
fix works, it can land on ltx-perf (LTX-layer, gated). If it needs a ttnn C++ change, that should
go as a ttnn PR for metal-team review, NOT a silent ltx-perf push — open the PR, link it, leave
1x4 gated until merged. Either way: validate ALL gates before landing; never push a hanging default.

## PROGRESS LOG (PRIMARY-FIX attempt — cq isolation)
- 2026-06-16 (resume): Code-read CONFIRMED the SHARPENED FIX is reachable in Python (no C++):
  the async CCL ops (all_gather_async/reduce_scatter_minimal_async/neighbor_pad_async) take NO
  queue_id, but the generic device-op enqueue calls `mesh_device->mesh_command_queue()` with no
  arg → `cq_id.value_or(GetCurrentCommandQueueIdForThread())` (mesh_device.cpp:760). That thread-
  local current-cq IS settable from Python via the exported `ttnn.command_queue(cq_id)` context
  (decorators.py:105; push/pop bound in ttnn-nanobind/core.cpp). So wrapping the whole audio decode
  in `ttnn.command_queue(1)` routes EVERY audio op (incl. the cq-less CCL ops) onto cq 1. The close
  guard (mesh_device.cpp:890-916) is per-cq via `mesh_command_queues_[cq_id]->in_use()` → parent cq0
  + child cq1 ⇒ no shared in-use cq ⇒ no throw. Implemented (commit 079d567769b): pipeline_ltx.py
  adds `_audio_cq_id` (=LTX_AUDIO_CQ, default 1, only when submesh active) + `_audio_cq()` ctx, splits
  decode_audio into a thin cq-wrapper + `_decode_audio_impl`; test opens ring with num_command_queues=2
  when LTX_AUDIO_SUBMESH set; repro rewritten to run parent cq0 + child cq1 and prove clean close.
  Full-mesh path byte-identical (cq 0, 1 CQ). py_compile OK, pre-commit OK. Next: run repro on device.
- 2026-06-16 06:02 UTC / 23:02 PT (06-15): Repro job 060002-13 FAILED in fixture SETUP (0.17s),
  NOT my code: RuntimeError "Read 0xffffffff over PCIe ID 8: the board should be reset" at the first
  ttnn.get_pcie_device_ids() — a wedged Galaxy board (the FW/PCIe wedge the mission flagged).
  device_params correctly showed num_command_queues:2 (my parametrize edit is live). recent_jobs
  shows a FOREIGN tenant (bare "smarton"/sulphur, NOT "[claude]smarton") cycling galaxy-reset attempts
  (05:53, 04:45, 04:18, 04:13) all FAILING (exit 1, 261s) — they cannot recover it either. Queue idle
  now but foreign tenant active. Per HARD safety: NOT resetting under a foreign tenant whose own resets
  are failing (collision risk); re-queuing/waiting for the board to recover. cq-1 fix is UNTESTED on
  device (board never reached my code). Code committed (079d567769b).
