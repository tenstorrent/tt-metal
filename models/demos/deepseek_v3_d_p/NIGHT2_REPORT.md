# Overnight-2 run report — per-element-tensor metadata redesign + rebase + per-op split

Branch `ppopovic/trace_experiments`. Plan: `/data/ppopovic/.claude/plans/i-want-to-do-twinkly-volcano.md`.
Pre-squash HEAD (recovery): `2b16e59002bfebdd0467e4c32c7a350e1b095c4d`.

## Phase 1 — squash + force-push ✅
- `git reset --soft 51bbd92` → 1 commit `326056b` (squashed tree byte-identical to pre-squash 2b16e59).
- Excluded the 2 untracked files from another task (dflash_draft_prefill.py, mtp_prefill.py).
- `git push --force-with-lease origin ppopovic/trace_experiments` → forced update OK.

## Phase 2 — per-element-tensor redesign (commit 2) — in progress
Replacing the single packed `metadata` tensor with N 1-element uint32 tensors per op. New signatures:
- update_padded_kv_cache(cache, input, slot_idx:Tensor, kv_actual_global:Tensor, layer_idx, num_layers, cluster_axis)
- rotary_embedding_indexed(input, cos, sin, trans_mat, kv_actual_global:Tensor, cluster_axis, ...)
- zero_padded_kv_cache(cache, slot_idx:Tensor, valid_global:Tensor, layer_idx, num_layers, chunk_size_global, cluster_axis, pad_align)
- ring_mla(..., slot_id:Optional[Tensor], kv_actual_isl:Optional[Tensor], kv_cache_num_layers:int, kv_cache_layer_idx:int)
Each per-element tensor = 1-element uint32 replicated DRAM. Scalar (int) overloads kept; the single-
metadata overload is replaced by the per-element-tensor overload. Order: rotary (template) → update_cache
→ zero_pad → ring_mla; then mla.py/pipeline/tests; one ttnncpp+ttnn rebuild; bit-exact per-op gate.

Progress:
- rotary_embedding_indexed: DONE (kernel reads elem[0]; hpp/cpp/nanobind metadata→kv_actual_global; device op
  field `metadata` kept internally, now a 1-element tensor; no shape assert so compatible).
- update_padded_kv_cache, zero_padded_kv_cache, ring_mla: delegated to parallel subagents (edit-only).

GLUE DESIGN (mla.py/transformer/block/pipeline/tests): keep the `metadata` param name but it becomes a
3-TUPLE of 1-element tensors `(slot_t, actual_start_t, actual_end_t)` (or None), minimal signature churn.
mla.py unpacks: rotary←kv_actual_global=meta[1]; update_cache←slot_idx=meta[0],kv_actual_global=meta[1];
zero_pad←slot_idx=meta[0],valid_global=meta[2]; ring_mla←slot_id=meta[0],kv_actual_isl_tensor=meta[1].
Pipeline capture_trace builds the 3 persistent 1-element replicated-uint32-DRAM tensors, updates in place
per chunk. Tests build 3 tensors instead of the packed [slot,start,end,0].

Status (mid-Phase-2):
- DONE (edits): rotary (me) + update_padded_kv_cache (subagent, reads both 4B into one CB page at off 0/4)
  + zero_padded_kv_cache (subagent, common args 10/11) — all C++ + their op equivalence tests (now
  per-element-tensor == scalar, torch.equal). mla.py (3-tuple unpack), pipeline (capture_trace builds 3
  tensors, prefill updates 3, release frees 3), test_prefill_transformer_chunked.py (both trace sites build
  3 tensors). All Python AST-clean.
- IN PROGRESS: ring_mla subagent (kernels done: ring_joint_reader + all_gather_reader; sdpa.hpp/cpp/nanobind
  + ring_joint_sdpa device op + test still finishing).
- transformer/block/runner UNCHANGED (forward the metadata tuple / pipeline owns the 3 tensors).
- NEXT: after ring_mla lands → ONE `cmake --build build_Release --target ttnncpp` (+ `--target ttnn`) +
  .so refresh → run the 4 per-op equivalence tests (bit-exact gate) → fix → commit 2 → push → Phase 3.

### Build + first gate run (DONE)
- ttnncpp + ttnn built clean (exit 0); `.so` refreshed (build_Release/ttnn/*.so → lib + ttnn/ttnn).
- First gate: zero_pad 4/4 PASS; rotary FAIL (test outdated); update_padded 3/3 FAIL.
- **Bug 1 (update_padded writer kernel)**: read slot into dst offset 0 AND kv_actual into dst offset 4
  of one CB page, single barrier. The +4 dst offset violates NoC/DRAM read dst-alignment → kv landed
  wrong (case slot=1,kv=1024 failed, all-zero case passed). FIX: mirror zero_pad — read slot into
  offset 0, barrier+extract, read kv into offset 0 (overwrite), barrier+extract. Kernel-only (JIT).
- **Bug 2 (rotary TEST not converted)**: test still fed the packed H2D `[slot,start,end]` tensor; new
  kernel reads element [0] → got slot(0) not kv_actual. FIX: rewrote test to build a 1-element uint32
  kv_actual_global tensor (per-element pattern, like the other 3 op tests); dropped H2D-service
  apparatus + unused `struct` import + `_H2D_METADATA_SIZE_BYTES`.
- Re-run rotary + update gate after fixes: **4/4 PASS**.

### ring_mla gate (DONE diagnosing)
- First ring_mla gate: rotation kv64/256/320 PASS, indexed[slot0] PASS, **indexed[slot1] FAIL** (diff 1584).
- DPRINT in the all-gather reader showed the gather reads `slot_id=1` correctly on ~all cores but ONE
  core (dev10 core y=2) intermittently read `slot_id=0` (base=0) → gathered garbage input slot 0.
- **Bug 3 (all-gather reader race)**: slot_id + kv_actual were read into the SAME `meta_l1` (offset 0)
  and `slot_id` was first *used* far later (the input_batch_base loop) — after the kv read clobbers
  meta_l1. The optimizer hoisted the volatile slot load past the kv `async_read` (NoC DMA writes are
  invisible to the compiler), so on cores where kv data landed first, slot read 0. Intermittent
  per-core. Packed baseline did ONE 16B read so never hit it. FIX: read slot→offset 0, kv→offset 16
  (16B-aligned, separate non-aliasing L1 slots) under a single barrier; c_in3 is 32B so it fits.
  Kernel-only (JIT). SDPA ring_joint_reader is safe (it consumes slot into kv_cache_batch_idx BEFORE
  the kv read). update writer reads both into offset 0 but the test passed 3/3 — see note below.

### Bug 3 deep-dive (2026-07-01, resumed) — ring_mla all-gather reader reads slot_id=0 on ~6-13/128 cores
Test: `test_ring_mla_metadata_matches_scalar_indexed[slot1]` (bit-exact). slot0 passes (0==0), rotation passes.
Reader runs on 4 cores/device (x=11,y=0..3) × 32 dev = 128 lines; fwd(dir=1) + bwd(dir=0) both read metadata.
DECISIVE DIAGNOSTICS (all with DPRINT):
- Sentinel (0xDEADBEEF pre-write): NEVER survives => the NoC read ALWAYS lands (not stale L1).
- Same L1 addr (144896) + same DRAM addr (5893760) for all 128 cores => address is correct/identical.
- Host readback: `slot_id AFTER op = all 1` on 32 devices => DRAM value is stably 1 before AND after (NOT a
  persistent clobber / buffer aliasing). out_kv buffer per-bank range does NOT overlap slot_id.
- `ttnn.synchronize_device` before the op does NOT help => NOT a host-write race (also: run(False) fully
  runs+syncs before run(True), so write is long-landed).
- read size 4B vs 32B (aligned, in-bounds) => NO change in failure rate => not read-size/alignment/contention
  (packed version had identical 128-core page-0 contention and PASSED bit-exact).
- +8 / +16 second reads always returned 0 => platform needs 32B-aligned L1 dst (that diag was an artifact).
CONCLUSION so far: DRAM holds 1 throughout, the read lands, correct address — yet ~10% of cores' reads return
0, different cores each run. SDPA reader reads the SAME slot_id tensor CORRECTLY (0 failures) but runs LATER
(after the gather). Strongly points to NoC-transaction corruption during the all-gather's fabric/EDM storm at
op start. Reader slot-read code is essentially identical to the packed version (which passed this exact test).
NEXT: dumping full 32B (words w0..w3) on failing vs good cores to see if failing read is all-zeros (wrong/empty
source) or word0-specific. If NoC-storm, fix = issue read at a quieter point / different NoC / read-verify-retry.

### Bug 3 RESOLVED (2026-07-01) — transient drop-to-zero NoC read, fixed with max-of-K re-read
Final root cause: the metadata NoC read COMPLETES (barrier returns; sentinel always overwritten;
spin-until-landed count = 0 => not late-landing) but under the all-gather's op-start NoC/DRAM contention
it INTERMITTENTLY DELIVERS 0 on ~6% (no DPRINT) to ~15% (DPRINT-all) of the reader cores that all hit the
one tiny replicated metadata DRAM page. DRAM stably holds the true value (host readback before+after = 1).
Corruption is always a full drop-to-zero; a re-read once contention subsides returns the true value.
Ruled out: host-write race (synchronize_device no help; 2nd run anyway), buffer aliasing (slot_after=1,
out_kv per-bank range clears slot_id), read size/alignment (4B vs 32B identical), late barrier (spins=0).
FIX: re-read K=8 times, zero L1 before each, take the MAX (drop-to-zero => max recovers true value; a true
0 stays 0; values only ever drop, never rise). Applied to all 3 metadata readers of the fused ring_mla:
  - ring_attention_all_gather_reader.cpp (slot_id + kv_actual_isl)
  - ring_attention_all_gather_writer.cpp (kv_actual_isl)
  - transformer/sdpa/.../ring_joint_reader.cpp (slot_id + kv_actual_isl)
Removed all debug DPRINTs + dprint.h includes; reverted test-file diagnostics.
RESULT: test_ring_mla_metadata_matches_scalar_indexed[slot1] PASSES bit-exact (was diff ~1584-1832), all
128 cores read slot_id=1. Running full metadata equivalence suite (indexed slot0/1 + rotation kv64/256/320).
NOTE (for later root-causing / PR reviewers): max-of-K is a robust mitigation, not a platform fix. The
underlying "completed same-address NoC read returns 0 under peak fan-out contention" likely warrants a
metal-level look. K=8 residual miss ~0.06^8/read => negligible over a full 61-layer/11-chunk run.

### Bug 3 FINAL fix scope (2026-07-01) + NEW pre-existing rotation bug found
Bug 3 fix is SURGICAL: max-of-K=8 re-read applied ONLY to the all-gather reader's SLOT read (the kernel's
FIRST NoC read, issued at peak op-start contention -- the only read that suffers drop-to-zero). The kv_actual
reads (gather reader/writer, SDPA reader) run a few instructions later once the storm clears and are reliable
with a single read (max-of-K on kv is unnecessary AND was briefly suspected of a regression). kMetadataReadBytes
stays 4. SDPA reader slot read left as single (it runs after the gather, never observed to drop).
RESULT: test_ring_mla_metadata_matches_scalar_indexed[slot0]+[slot1] PASS bit-exact.

NEW (separate, pre-existing) BUG: test_ring_mla_metadata_matches_scalar_rotation[kv64/256/320] FAIL with
DETERMINISTIC diffs (20.375 / 24.625 / 24.75 -- byte-identical across 3 runs, and identical whether the slot
fix is present or not, and the tests run at slot=0 so the slot fix cannot affect them). => the metadata-path
kv_pad ROTATION DERIVATION in the SDPA reader (logical_nt / q-mapping / ring masks derived on-device from
kv_actual_isl, handed to compute via cb_kv_pad_derived) does not bit-match the scalar path's host-computed
q-mapping. This is a task-4 derivation bug inherited from the dead session (NOT introduced here). NEXT: diff
the on-device derivation vs the host scalar computation to find the mismatch.

### Rotation bug ROOT-CAUSED (2026-07-01) — SDPA reader kv read at a 32B-MISALIGNED L1 offset returns 0
Diagnosis path: compute-kernel DPRINT showed metadata-path logical_nt=16 vs scalar-path=18 (device-0 qmap
qpre=0/qprec=2 vs scalar qpre=0/qprec=0/qpost=16) -> metadata path derives from kv_actual_isl=0 not 64.
Reader DPRINT confirmed kv_actual read = 0 from the CORRECT address (kvaddr distinct from slotaddr by 64).
ROOT CAUSE: ring_joint_reader read kv_actual_isl into `meta_l1 + kKvDstOffset` with kKvDstOffset=16. This
platform requires the L1 DESTINATION of these small (4B) DRAM reads to be 32B-ALIGNED -- a read into a
merely-16B-aligned address silently lands 0 (independently confirmed earlier: a +16 re-read in the gather
reader ALWAYS returned 0 while +0 worked). slot read at +0 (32-aligned) works; kv at +16 (not 32-aligned)
returns 0. The indexed tests can't catch it (kv_actual=0 there, so 0 is "correct"); only rotation (kv!=0)
exposes it. The writer reads kv at offset 0 (32-aligned) so it was always fine.
FIX: kKvDstOffset 16 -> 32 (both slot@0 and kv@32 now 32B-aligned + non-aliasing). One-line kernel change.
This is a DETERMINISTIC bug, unrelated to Bug 3's transient drop-to-zero (max-of-K does not help/matter here).
Verifying full ring_mla suite (indexed slot0/1 + rotation kv64/256/320) now.

### Rotation bug FIXED (2026-07-01) — CORRECTED root cause: kv read must land at CB offset 0, not +16
My earlier "32B-alignment" / "shared-accessor" theories were WRONG (both changed nothing -- and I was ALSO
running a STALE _ttnncpp.so: the loaded lib is ttnn/ttnn/_ttnncpp.so, NOT build_Release/lib; host changes
were silently ignored until I copied to ttnn/ttnn/. See updated ttnn-so-refresh memory.).
ACTUAL root cause (proven by RJKV2 DPRINT: kv=0 at off=16/32, kv=64 at off=0; tensor host-verified = 64):
the SDPA ring_joint_reader read kv_actual_isl into `meta_l1 + kKvDstOffset` with kKvDstOffset=16 (later
tried 32) -- reading into ANY nonzero offset of that CB deterministically lands 0 on this platform; only
the CB page base (offset 0) works. FIX: kKvDstOffset 16 -> 0 (slot is read+consumed into kv_cache_batch_idx
BEFORE the kv read reuses offset 0, so no aliasing; mirrors the all-gather reader which reads both scalars
at offset 0). Also gave the reader its own kv accessor (mirrors the writer; correct + tested, though
offset-0 was the operative fix). RESULT: all 5 ring_mla equivalence tests PASS (indexed slot0/1 + rotation
kv64/256/320). Files: ring_joint_reader.cpp (kernel, JIT) + ring_joint_sdpa_program_factory.cpp (host, .so).

### Bug 3 CORRECTED root cause + proper fix (2026-07-01) — supersedes the max-of-K entries above
The max-of-K re-read was a HACK and is REMOVED. Real root cause: the gather reader read the metadata
scalars into the tiny dedicated `c_in3` CB (cb_meta_id) via get_write_ptr(). That 32-byte CB's L1 region
is transiently CLOBBERED by concurrent fabric/packet-header activity during the op-start read window ->
intermittent drop-to-zero on ~10% of cores (varying run-to-run: clean-run diffs 1584/720/1584). PROPER
FIX (1-line, no retry): read the metadata into the OUTPUT CB (cb_output_id / c_in0) L1 as scratch -- a
large real data CB that is not touched until the gather loop reserves+overwrites it -- exactly as the
proven SDPA ring_joint_reader does with cb_q_in. Single read, reliable: indexed[slot1] 3/3 clean pass
(was 3/3 fail); full ring_mla suite (5 tests) 3/3 = 15/15 pass. The writer still reads kv from c_in3 and
is reliable (rotation 3/3), so it is left unchanged (reads at a less-contended point).
BRANCHING: per user, this work is NOT on ppopovic/trace_experiments (I mistakenly committed+pushed there;
reverted the remote to commit 1 326056b7ac8). Work now lives on ppopovic/per_element_metadata (local).
