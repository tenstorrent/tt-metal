# PLAN: Fused NP+Conv3d Cleanup & Perf Restoration

## Goal
Land the branch in a shippable state: every halo-needing VAE conv fused by default, true T-pipelining across H and W halo signalling, and a focused test that proves fused ≥ standalone on perf.

## Scope / Hardware
- Primary target: Blackhole 2x2 (bh-lb-09, locally reproducible).
- Correctness gate: PCC=1.0 vs standalone *and* PCC≈1 vs PyTorch-fp32 reference on all-ones latent.
- Perf gate: fused end-to-end VAE decoder wall-clock faster than standalone by a clear margin on the same shape.
- 2x4 LoudBox: out of scope for this plan (untestable locally); CI handoff after Phase 3.

## Decisions
| Decision | Reason | Rejected alternative |
|----------|--------|----------------------|
| Every halo-needing VAE layer fused by default | Fused is meant to be the fast path; partial fusion was diagnostic only | Leave some layers standalone |
| Time-conv stays standalone | It has no spatial halo, so fusion has nothing to fuse | Force fuse everywhere |
| Plumb `use_fused` from `WanDecoder3d.__init__` | Single toggle for tests/debug; removes per-layer DIAG hard-codes | Model-wide monkey-patch from test |
| Per-T-batch progress-sem signalling on **both** H and W + per-batch wait in conv3d reader | Current H-per-batch signal is wasted because reader waits once for the total; W is not per-batch at all | Revert H to one-shot (loses latency) |
| Fix 1 (in-kernel progress-sem reset) kept as defensive no-op on 2x2 | Real 2x2 fix is Fix 2 (W-reader input_addr RTA); Fix 1 may still matter on 2x4 | Remove before 2x4 CI validates |
| w_neighbor_sem per-batch splitting only if it's clean | Thresholded `wait_min` on a monotonically-increasing sem is elegant; actual reset per batch is messy | Force per-batch reset + full rewrite |

## Phases

### Phase 1 — Restore production defaults *(risk: low; unlocks Phase 2)*
1. `models/tt_dit/models/vae/vae_wan2_1.py`: plumb `use_fused` through `WanDecoder3d.__init__`; thread it down to every halo-needing conv (conv_in, mid_block, up_blocks, conv_out). **Time-conv** (WanResampleConv with `kernel_size[0] > 1` and no spatial halo) stays standalone.
2. Remove all `# DIAG:` hard-codes (lines 1269, 1282, 1320, 1345).
3. `models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py`: strip the monkey-patch scaffolding. Drive `use_fused` via the new constructor arg. Keep the test parametrize IDs `fused` / `standalone_np_conv3d`.
4. Nuke local-only debug artifacts (*not* committing them; `rm` or ignore):
   - `SEAM_DEBUG_HANDOFF.md`
   - `bh_loudbox_2x4_mesh.textproto`
   - `pytorch_vae_ones_ref.py` (diagnostic; its job is done)
   - `wan_decoder_ones_480p_81f_*.mp4` and `.pt.bak`
5. Delete `W_SEAM_MINIMAL_TEST.md`. Replace `W_HALO_VERTICAL_LINES.md` with a ~30-line `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/README.md` covering only: halo layout, per-T-batch signalling contract, and progress-sem lifecycle. Investigation history stays in `git log`.

**Verify:** `test_decoder_ones_input.py -k "2x2_h0_w1 and multi_block"` — both `fused` and `standalone_np_conv3d` IDs pass; fused matches standalone (PCC=1.0).

---

### Phase 2 — Full-fusion correctness gate *(risk: medium; may surface up_block-specific bugs)*
With every halo-needing conv fused, run the ones-input decoder and verify:
- `fused` output == `standalone_np_conv3d` output bit-for-bit.
- `fused` output matches the PyTorch reference (single-frame) at col 416 and interior.

If a new seam surfaces only with up_blocks fused, it's almost certainly another instance of the RTA-not-refreshed class of bug on some kernel I didn't update. Bisect with the `use_fused` constructor toggle (on/off per up_block stage) and apply the same RTA-refresh pattern.

**Do not start Phase 3 until Phase 2 passes.**

---

### Phase 3 — True T-pipelining across H and W halo signalling *(DEFERRED — see Session 7 note below)*

**Status:** First attempt committed the wrong threshold formula in the conv3d reader. Correctness regressed (`FUSED vs STANDALONE: Boundary PCC=98.3%, max_diff=1.4`) while H-only boundary stayed at 100%, interior stayed at 100%. Reverted.

**Why the threshold was wrong:** The per-T-block wait I used (`(t_iter + 1) * signals_per_batch`) only covers the W-halo for the CURRENT T-input frame. A causal 3D conv with `kernel_t=3`, `padding_t=1` actually reads T-input frames `[k-1, k, k+1]` for T-output block `k`, so it needs W-halo *up to batch `k+1`* — i.e. the threshold needs a look-ahead.

**Correct formula (for future retry):**
```
last_t_in_needed(k) = k * T_block_size * stride_t + kernel_t - 1 - padding_t
t_batch_offset     = ceil((T_block_size * stride_t + kernel_t - 1 - padding_t) / progress_t_batch_size)
threshold(t_iter)  = min(t_iter + t_batch_offset, total_batches) * signals_per_batch
```
For VAE mid_block (T_block_size=1, stride_t=1, kernel_t=3, padding_t=1, progress_t_batch_size=1):
- `t_batch_offset = 2` → threshold at t_iter=0 is `2 * signals_per_batch`, at t_iter=1 is `3 * signals_per_batch`, etc.
- Without the cap, the last t_iter over-waits → hang.

**Hang-avoidance protocol — blocking when retry:**
A mismatch between what signals the writers emit and what the reader waits for turns into a device hang that eats ~2 minutes per test run. To keep dev loop fast:
1. **Test at the unit level first.** Before touching the full decoder, run `test_fused_production_shapes` (a single dispatch) for every shape after each kernel-side change. One dispatch can't hang on per-T-batch mismatch — a multi-dispatch test can.
2. **Run with Watcher enabled** (`TT_METAL_WATCHER=60` or similar) during Phase 3 development so hangs abort instead of waiting out the pytest timeout. **Caveat observed 2026-04-22:** Watcher currently fails fabric firmware init with "Program size (27872) too large for kernel config buffer (25600) on ACTIVE_ETH" on bh-lb-09 — debug separately before relying on it.
3. **Start `input_progress_signal_count` conservative.** Temporarily set it to the *expected total across all T-batches* (i.e., current behaviour) and verify the kernel logic still reaches the end, then flip to per-batch math.
4. **Add a debug counter** next to each `noc_semaphore_wait_min` during development (number of iterations it waited for, logged via `DPRINT`). Removed before commit.
5. Keep the existing `test_fused_ones_input_seam` unit test passing after each kernel edit — it's the canary.

Current state is *asymmetric and broken*:
- `minimal_default_writer.cpp` (H-writer): signals `progress_sem` **per T-batch** (good intent) + tail signal.
- `phase2_w_reader.cpp` (W-reader): signals `progress_sem` **once, at the very end**.
- `reader_vol2col.cpp` (conv3d reader): `noc_semaphore_wait_min` **once, before the T-loop** — i.e., waits for all signals up front, so the H-per-batch signals are wasted.

Net: no actual pipelining; conv3d can't start T-batch `k` until H+W halo for all T-batches is done.

**Changes:**

1. **W-reader: add per-T-batch signalling.** Add `progress_t_batch_size` as a CT arg (mirroring H-writer). Split the end-of-kernel signal into a per-T-batch signal inside the outer_dim loop, gated by `(outer_dim + 1) % progress_t_batch_size == 0`, plus a tail signal. Each signal is a `noc_semaphore_inc` across all conv3d reader cores.
2. **conv3d reader: move the wait inside the T-block loop.** Replace the single `wait_min(sem, total)` with an incremental `wait_min(sem, cumulative_threshold_for_this_t_block)` call each iteration of the `for (t_block ...)` loop. `cumulative_threshold_for_this_t_block` = `(t_block_batches_so_far + 1) * signals_per_batch` where `signals_per_batch = num_h_fabric_cores + num_w_fabric_cores`.
3. **Host factory (`neighbor_pad_conv3d_program_factory.cpp`):** update `input_progress_signal_count` to `(num_h_fabric_cores + num_w_fabric_cores)` *per batch* (or plumb a per-batch step + num_batches separately). Pass `progress_t_batch_size` into the W-reader compile args. Verify H-writer already has it (it does).
4. **Progress-sem reset placement:** with per-batch waits, the final reset moves to *after* the last T-block's wait, not before the first (current Fix 1 location). Single reset is sufficient since the sem is monotonically increasing.

**w_neighbor_sem per-batch splitting** *(only if clean):*

Current `phase2_w_reader.cpp`:
```cpp
noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);  // wait for ALL sticks across all T-batches
noc_semaphore_set(w_neighbor_sem_ptr, 0);
```
The *elegant* per-batch version uses the same monotonic `wait_min` with a cumulative threshold (no reset until end-of-kernel), exactly like the conv3d reader's progress-sem pattern:
```cpp
for (t_batch = 0; t_batch < num_batches; ++t_batch) {
    uint32_t threshold = (t_batch + 1) * sticks_per_batch;
    noc_semaphore_wait_min(w_neighbor_sem_ptr, threshold);
    // ... process this batch's W halo ...
    // ... emit per-batch progress_sem signal to conv3d readers ...
}
noc_semaphore_set(w_neighbor_sem_ptr, 0);  // single reset at end
```
This is elegant — no per-batch reset required, and the threshold stays correct because the fabric in-order delivery guarantees all writes for batch `k` land before any signal for batch `k+1`. **Ship it.**

**Verify:**
- Correctness: full-fused decoder still PCC=1.0 vs standalone and PyTorch on 2x2.
- Perf: end-to-end decoder wall-clock *and* per-op latency (fused NP+conv3d alone) both lower than standalone. See Phase 5 for the measurement harness.

---

### Phase 4 — Test suite consolidation *(risk: low)*
Keep:
- `test_decoder_ones_input.py` — end-to-end correctness (fused vs standalone vs PyTorch).
- `test_neighbor_pad_conv3d_fused.py::test_fused_production_shapes` — per-layer correctness on random input.
- `test_neighbor_pad_conv3d_fused.py::test_fused_ones_input_seam` — minimal boundary-artifact unit test.

Audit `test_fused_decoder_boundary.py` (1311 lines, added by c7aa748d3c for the C_in_block investigation). Extract anything still load-bearing into the two test files above; delete the rest.

---

### Phase 5 — Focused perf test: fused vs standalone *(risk: low)*
New test `test_neighbor_pad_conv3d_fused_perf.py`:
- Same parametrize shapes as `test_fused_production_shapes`.
- For each shape: 1 warmup + N measured dispatches of **fused** and **standalone** back-to-back on the same mesh device; measure end-to-end wall-clock time via `time.perf_counter_ns()` with `ttnn.synchronize_device()` bracketing (no per-kernel instrumentation needed at this stage).
- **Log-only** — print `PERF: fused=<X.XX>ms standalone=<Y.YY>ms ratio=<X/Y> shape=<...>`. No hard assertion; a regression is visible but won't fail the test. Promote to a hard assert once numbers stabilize.
- Run for 2x2 480p shapes first; add 2x4 shapes once CI confirms 2x4 correctness.

---

### Phase 6 — Docs & Fix 1 decision *(risk: low)*
1. Delete `W_HALO_VERTICAL_LINES.md`, `W_SEAM_MINIMAL_TEST.md`. History is in commits `d8a939bf1e` (W-reader RTA refresh) and `c1dbd82c85` (progress-sem reset).
2. Add a short `ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/README.md`:
   - Pipeline diagram: H-writer / W-reader / conv3d-reader progress-sem contract.
   - "Every per-dispatch RTA must be refreshed in `override_runtime_arguments`" checklist.
   - Per-T-batch signalling protocol.
3. Fix 1 decision: if 2x4 CI passes without Fix 1 (the `noc_semaphore_set` reset in `reader_vol2col.cpp`), remove it and update the commit message in a squash-or-followup. If it's load-bearing on 2x4, update the comment to cite the specific 2x4 failure mode.

---

## Commits-per-phase plan
- One commit per phase (or logical sub-step within a phase). No phase squashed with another.
- Phase 3 likely splits into 3 commits: (a) W-reader per-batch signal, (b) conv3d reader per-batch wait, (c) factory arg-passing wiring.
- Force-push only at end (same as today).

## Open questions
- [ ] Any up_block layer currently broken when fused that we don't yet know about? (Phase 2 will answer.)
- [ ] Expected fused-vs-standalone speedup on 2x2 480p — to be measured in Phase 5 and promoted to a hard assert once stable.

## State
- [x] Phase 1 — Restore production defaults  *(committed 2e140a5ba8)*
- [x] Phase 2 — Full-fusion correctness gate  *(same commit — full-fused decoder verified PCC=1.0)*
- [ ] Phase 3 — True T-pipelining (H + W + reader)  *(deferred; first attempt reverted after correctness regression, formula fixed above but not shipped)*
- [x] Phase 4 — Test-suite consolidation  *(deleted test_fused_decoder_boundary.py, 1311 lines of investigation scaffolding)*
- [x] Phase 5 — Perf test (log-only)  *(new test_neighbor_pad_conv3d_fused_perf.py; first run: mid_res_2x2_480p fused/standalone=1.200 — fused 20% SLOWER, quantifies gap T-pipelining would close)*
- [x] Phase 6 — Docs & Fix 1 decision  *(deleted W_HALO_VERTICAL_LINES.md + W_SEAM_MINIMAL_TEST.md; added neighbor_pad_conv3d/README.md with pipeline diagram + RTA-refresh checklist. Fix 1 kept defensive pending 2x4 CI.)*
