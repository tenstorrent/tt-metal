# Rebasing the autoport onto the new skills base

`mvasiljevic/qwen38-autoport-qb2` was brought up against a tt-metal from
merge-base `107623bb9dd`. `agentic-research/fast-models-fast` is 2,580 commits
past that point and carries the newer agent skills. This records what the move
broke, what was fixed, and what is still broken.

Result branch: `mvasiljevic/qwen38-deltanet-kda`.

Everything below is sorted by **who caused it**, because that determines who
should fix it. Three of these are not rebase regressions at all and are marked
as such; they are included because they cost time during this work and the
distinction was not obvious up front.

## How the replay was done

202 commits, replayed with `git rebase --onto agentic-research/fast-models-fast
107623bb9dd -X ours --empty=drop`. 165 landed, 37 dropped as empty, 31
conflicts auto-resolved in favour of the new base.

`-X ours` is the right default here: where the old branch and the new base
touched the same lines, the base wins, which is the entire point of moving onto
the new skills. But it only decides *conflicting* hunks. Non-conflicting edits
from the replayed commits still apply on top, and that is where the damage was
-- see A1. Always diff the rebased tree against the base afterwards and read
every file outside the payload directory.

After the fixes below the delta from the base is Python-only, so a build made
on the base stays valid.

---

## A. Caused by the rebase mechanics

### A1. The replay silently reverted three dispatch/fabric kernels

- `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp`
- `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`
- `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`

The autoport branch had edited the idle-erisc early-exit path in all three
against the old base. Upstream has since folded that path into the
`IDLE_ERISC_HEARTBEAT_AND_RETURN` / `IDLE_ERISC_RETURN` macros. The replayed
commits re-expanded the macro back into its old inline form, and because the
surrounding lines had moved, the hunks applied **without conflicting** -- so
`-X ours` never saw them. The result was a silent revert of an upstream
refactor, in C++ kernels, in a tree that otherwise looked clean.

It also would have forced a rebuild of a tree whose delta was supposed to be
Python-only.

**Fix:** take the base's version of all three (`git checkout
agentic-research/fast-models-fast -- <files>`). Commit `e542603b2e3`.

**Lesson:** a rebase strategy option cannot protect files where history moved
underneath the edit. Grep the post-rebase diff for `.cpp` / `.hpp` / `CMakeLists`
before trusting "Python-only".

### A2. The old branch's `.agents/` skills layered on top of the new ones

The autoport branch carried its own evolved copy of the skills; the new base
carries the newer ones. Conflicting hunks resolved to the base as intended, but
purely additive old-branch content applied cleanly on top -- `+153` lines in
`optimize/SKILL.md`, plus additions to `tti-release`, `full-model`,
`datatype-sweep` and `tt-enable-tracing`.

**Decision: kept, not reverted.** Reviewed and it is additive prose (the
performance-accounting and full-model-decode-closure sections), not a
contradiction of the new skills' approach. Flagging it because "rebased onto the
new skills" does not mean the skills are byte-identical to the base, and anyone
diffing them will find this.

Same shape, also kept: `models/common/readiness_check` gained the `P150_X4`
mesh label and the `--tensor-cache` pass-through; `tools/tracy/process_ops_logs.py`
gained a fix for a trace `END` marker with no captured `BEGIN`.

---

## B. Caused by the new base — still broken

### B1. The TP4 multichip path did not run at all: three L1 clashes — FIXED

Every entry point into `multichip_decoder.py` failed on this base with
statically-allocated circular buffers colliding with L1 buffers:

| Path | Call site | Cores | CB region ends | L1 buffer at | Overlap |
|---|---|---|---:|---:|---:|
| prefill | `_rms_norm` (input layernorm) | `[0-0 - 3-0]` | 1,188,864 | 916,992 | 271,872 B |
| decode | `_tp_linear("mlp_down_decode")`, `in0_block_w=17` | `[0-0 - 7-9]` | 811,776 | 777,728 | 34,048 B |
| decode | `_all_reduce` after the MLP down projection | `[0-0 - 0-0]` | 177,152 | 163,328 | 13,824 B |

**Was it the rebase? Not confirmed.** The circumstantial case is strong, but I
never found a change in tt-metal that causes it, so this is recorded as
unattributed.

What supports "regression":

- `doc/multichip_decoder/artifacts/tracy/linear_b32_dram_sharded/provenance.txt`
  records the exact command, and its `trace_result.json` records
  `candidate=default`, `baseline_candidate=default`, `pcc: [1.0, 1.0, 1.0]`,
  `multichip_trace_median_ms=4.4718`. Same command, same candidate, no env vars,
  no setup step. Running it on the rebased tree failed.
- The harness performs no collective warm-up for `candidate=default`
  (`_ccl_buffers` is only populated for `multichip_preallocated_ccl`, and holds
  output tensors, not semaphores), so those green runs took the same path.
- The single-chip baseline the same harness prints is unchanged (15.8581 ms
  recorded, 15.8382 ms now), so the environment is comparable.

Candidate causes checked and **ruled out**:

| Candidate | Verdict |
|---|---|
| `ccl/all_reduce/all_reduce.cpp` changed | No. Only its nanobind docstring moved. |
| Fabric reserves more L1 | No. `total_bytes_per_bank` is 1,461,248 with `FABRIC_1D_RING` both disabled and enabled. |
| The composite-vs-RS+AG path flipped for this shape | No. `composite_common` changed only by a namespace refactor and a rank-1 edge case. |
| Reduce-scatter CBs grew via the `packet_size_bytes` switch in `e4afdb1a8e6` | No. `tile_granularity = min(4 * min(4, pages_per_packet), max_dst_size)` saturates at 8 either way. |
| `use_l1_small_for_semaphores` is new | No. Present on both bases, defaulting to `false` on both. |

**Not** ruled out: the `reduce_scatter_minimal_async` rewrite (+2,192/−1,800
across the two bases) could have changed the size or ordering of the transient
that displaces the semaphores. Settling that needs a build of the old base to
measure the same low-water mark, which this branch did not do.

So the honest position is: the failure is real, reproducible, and fully
characterized on the current base; whether the current base *changed* it is
unproven.

**Root cause — one thing, not three.** `ttnn.all_reduce` passes no persistent
semaphores (it forwards `std::nullopt` and falls through to the non-persistent
`ttnn::reduce_scatter` / `ttnn::all_gather` wrappers), so the op mints its own
global semaphores and keeps them, cached per collective configuration.

*That caching is not the bug, and is not a leak.* Repeating one shape holds
steady at nine 64-byte blocks across six calls; only a new shape adds another
set. In a real model, where the collective shapes are fixed, it is bounded and
small.

The problem is **where the first set lands**. The op also allocates a large L1
transient: a width-5120 `all_reduce` needs roughly 700-900 KiB contiguous, and
below ~600 KiB it fails inside `bank_manager` rather than degrading — measured
by reserving L1 with a blocker tensor and starving it. That transient is
allocated *before* the semaphores, and semaphores are placed top-down into
whatever is free. So on the first collective, with L1 empty, the transient takes
the top of L1 and forces the semaphores underneath it. Because they are cached,
freeing the transient does not move them, and contiguous L1 stays capped there
for the rest of the process:

| first call width | 32 | 512 | 5120 (the real residual width) |
|---|---:|---:|---:|
| largest contiguous L1 after | 1,461,120 B | 1,329,664 B | **805,376 B** |

805,376 B is not enough for this layer. The decode MLP down projection's static
CBs alone reach 811,776 B. The prefill norm needs ~1,080,320 B contiguous,
measured by squeezing L1 and bisecting; the same `ttnn.rms_norm` with a plain
tiled `[1, 1, 1, W]` weight instead of the row-major `[1, 1, W/32, 32]` contract
needs only ~564,224 B, which is why the single-chip path — which runs no
collectives and so never loses the L1 — never saw any of this.

**Fix:** run one minimal collective (width 512) at load time, while L1 is empty.
It leaves its nine 64-byte blocks near the top; a later full-width call then has
to put its transient below them, and can place its own semaphores in the holes
they left. Measured, the mark then stays at 1,329,664 B across repeated
width-5120 calls. One extra program at setup, no numerics change, nothing added
to the hot path. Commit `386e8b8ba9c`.

| Kind | Batch | recorded | now |
|---|---:|---:|---:|
| linear attention | 32 | 4.4718 ms | 4.3344 ms |
| linear attention | 1 | 0.9008 ms | 0.7930 ms |
| full attention | 32 | 0.7223 ms | 0.7241 ms |
| full attention | 1 | 0.5958 ms | 0.6084 ms |

All at PCC 1.0 against the single-chip baseline, prefill smoke green at S128.

**What I could not establish** is which side of the collective grew. Both
candidates were rewritten between the two bases: `reduce_scatter_minimal_async`
(+2,192/−1,800, including its program factory and a new chunk-paged contiguous
staging intermediate plus a "penult" intermediate — both DRAM, so not directly
the L1 cost), and `normalization/layernorm` (+2,581/−1,889, including the
interleaved multi-core path and the CB helpers). Settling that needs a build of
the old base to compare against, which this branch did not do. The fix does not
depend on the answer: it removes the fragmentation rather than trying to fit
underneath it.

### B1a. `ttnn.all_reduce` cannot reach the knob that avoids this

`ttnn.reduce_scatter` and `ttnn.all_gather` both take
`use_l1_small_for_semaphores`, which puts the semaphores in the L1-small region
and leaves the main heap completely alone — measured, largest contiguous L1 is
1,428,480 B both before and after a width-5120 reduce-scatter plus all-gather.
That is the intended mechanism for exactly this problem.

`ttnn.all_reduce` does not expose it, and `all_reduce_async` calls
`ttnn::reduce_scatter` with thirteen arguments, omitting the fourteenth, so it
always takes the `false` default. Any caller using `ttnn.all_reduce` therefore
has no way to keep the semaphores out of the main heap. This gap exists on both
bases.

It is also not a drop-in fix for the model, because it needs the *caller* to
open the mesh with an `l1_small_size`; at the default of 0 it fails with
`Not enough space to allocate 1760 B L1_SMALL buffer ... bank size is 0 B`.
The decoder does not open the mesh — the harness, the demo and vLLM do — so the
load-time warm-up stays the model-side fix, and the knob is the upstream ask.

### B2. Eager batch-32 decode fails in the shared synthetic harness

```
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --mode decode --optimized --batch 32
TT_FATAL: Shard height 32 must match physical height 1024 for width sharded
```

Fails for `default`, `linear_final`, `linear_packed_dram` and
`linear_state_fp32`. Also confirmed pre-existing relative to the KDA work by
reverting the three files. The traced harness (`traced_synthetic_pcc.py`) runs
batch-32 decode fine, and that is what the autoport's own batch-32 evidence
used, so **I did not establish whether this combination ever worked before the
rebase** -- it may simply never have been exercised. Recorded, not diagnosed.

---

## C. Not rebase regressions

Included because each cost time and each looked like a rebase problem at first.

### C1. The device profiler's program cap is not new

Tracy post-processing aborts with:

```
AssertionError: Device data missing: Op 1095683 not present in
cpp_device_perf_report.csv for device 3 (trace_id=None)
```

The dropped ops are real matmuls in the affine-scan loop -- 84 of them, starting
at program index 1070 of the ~1,440 the sequence-128 prefill harness dispatches
-- so the run cannot be reported with a hole in it.

I assumed the newer base had tightened `_enrich_ops_from_perf_csv`. **It had
not**: the assertion and the function are present on the pre-rebase autoport
branch too. What changed was the workload. The autoport's own profiled runs use
`--sequence 5` for linear prefill, which stays under the default cap; profiling
a sequence-128 prefill does not.

**Fix:** `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=8192`, wired into
`doc/kda_conv_swap/run_ab.py`.

### C2. Missing perf tooling

Neither tool was installed. Both were needed and both were installed into the
repo venv with `uv pip install --python python_env/bin/python`:

| Package | Version | Side effect |
|---|---|---|
| `tt-perf-report` | 1.2.9 | pins `matplotlib==3.10.9`, downgraded from 3.11.1 (repo `requirements-dev.txt` leaves matplotlib unpinned, so this is in policy) |
| `tt-smi` | 6.3.0 | pulled `tt-umd` 0.9.9 → 0.9.8 |

Note `pip` is not available in the venv -- it is a `uv` venv, and plain
`python -m pip` fails with "No module named pip" while system `pip` refuses with
PEP 668.

### C3. Boards were in the ERISC-timeout state

```
TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to
become active again. Try resetting the board.
```

The documented recoverable fault from the `tt-device-usage` skill. `tt-smi -r`
followed by a mesh smoke cleared it. Hardware state, unrelated to the rebase.

### C4. Two defects surfaced by the KDA work, not by the rebase

Both are recorded in `doc/kda_conv_swap/README.md` and fixed there:

- The shared synthetic weights set `conv[:, 0, -1] = 0.5` and leave taps 0-2 at
  zero, degenerating the convolution to `silu(0.5 * x_t)`. No conv-history bug
  can show under that weight. `doc/kda_conv_swap/check_conv_taps.py` covers it.
- `traced_synthetic_pcc.py`'s cache restore hardcoded `layout=ttnn.TILE_LAYOUT`
  while already deriving `dtype` from the destination, so a row-major cache
  could not be restored. It now takes both from the destination tensor.

---

## D. Repo mechanics worth knowing

The repo-root `.gitignore` ignores `*.csv` and `*.log` globally, so derived perf
evidence must be `git add -f`'d -- which is what the existing autoport doc
artifacts did. A pre-commit hook rejects files over 500 KB, which a traced
decode's `profile.log` exceeds at 783 KB; `run_ab.py` now retains only the
provenance lines. Raw captures (`.logs/`, `reports/`) are not committed per this
autoport's `.gitignore` -- `profile_log_device.csv` alone is 347 MB per arm.

## Open work

1. Decide whether B2 is a real regression or an unsupported combination.
2. Establish which side of the collective grew its L1 footprint, by building the
   old base and measuring the same first-collective low-water mark. The fix does
   not depend on it, but the answer belongs upstream: any model that runs a
   collective before a large program inherits this on this base.
3. Consider whether the collective should allocate its cached semaphores before
   its large L1 transient rather than after. The caching itself is bounded and
   correct; it is only the placement of the first set, underneath a transient
   that is then freed, that is load-bearing for every program that follows.
   That is a sharp edge for any caller who runs a collective before a large
   program, and it is invisible until something else fails to fit.

## Note for anyone triaging the MLP down projection

TP4 splits it to K=4352, which over 8 cores is **17 K tiles per core**, so
`in0_block_w` may only be 1 or 17. The existing `final_down_w4` /
`final_down_w34` / `final_down_w68` candidates are all illegal on the multichip
path and raise `ValueError: in0_block_w=N must divide 17 K tiles/core` before
reaching the device. `final_down_w1` was added during triage as the family's
only other legal TP4 value. It is not needed for the fix -- shrinking those CBs
only moved the failure to the next site, because 811,776 B was already above the
805,376 B floor.
