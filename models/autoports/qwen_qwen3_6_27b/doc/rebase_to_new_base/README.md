# Rebasing the autoport onto the new skills base

`mvasiljevic/qwen38-autoport-qb2` was brought up against tt-metal at merge-base
`107623bb9dd`. `agentic-research/fast-models-fast` is 2,580 commits past that
and carries the newer agent skills. This records what the move broke and what
was done about it.

Result branch: `mvasiljevic/qwen38-deltanet-kda`.

Sorted by cause, because half of what cost time here was not a rebase
regression, and the difference was not obvious up front.

## How the replay was done

202 commits, replayed with `git rebase --onto agentic-research/fast-models-fast
107623bb9dd -X ours --empty=drop`. 165 landed, 37 dropped as empty, 31 conflicts
auto-resolved in favour of the new base.

`-X ours` is right here — where the old branch and the new base touched the same
lines the base wins, which is the point of moving onto the new skills. But it
only decides *conflicting* hunks, and that is where the damage was (A1). After
the fixes below the delta from the base is Python-only, so a build made on the
base stays valid.

---

## A. Rebase mechanics

### A1. The replay silently reverted three dispatch/fabric kernels

`tt_metal/impl/dispatch/kernels/cq_dispatch.cpp`,
`tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`,
`tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`.

The autoport had edited the idle-erisc early-exit path in all three against the
old base. Upstream has since folded that path into the
`IDLE_ERISC_HEARTBEAT_AND_RETURN` / `IDLE_ERISC_RETURN` macros. The replayed
commits re-expanded the macro into its old inline form and, because the
surrounding lines had moved, applied **without conflicting** — so `-X ours`
never saw them. A silent revert of an upstream refactor, in C++ kernels, in a
tree that otherwise looked clean. It would also have forced a rebuild of a tree
whose delta was meant to be Python-only.

Fixed by taking the base's version of all three — commit `e542603b2e3`.

**Lesson:** a rebase strategy option cannot protect files whose history moved
under the edit. Grep the post-rebase diff for `.cpp` / `.hpp` / `CMakeLists`
before trusting "Python-only".

### A2. The old branch's `.agents/` skills layered on top of the new ones

Conflicting hunks resolved to the base as intended, but purely additive
old-branch content applied cleanly on top — `+153` lines in
`optimize/SKILL.md`, plus additions to `tti-release`, `full-model`,
`datatype-sweep`, `tt-enable-tracing`.

**Kept deliberately.** It is additive prose (performance accounting,
full-model decode closure), not a contradiction of the new skills. Flagged
because "rebased onto the new skills" does not mean byte-identical to the base.

Same shape, also kept: `models/common/readiness_check` gained the `P150_X4` mesh
label and a `--tensor-cache` pass-through; `tools/tracy/process_ops_logs.py`
gained a fix for a trace `END` marker with no captured `BEGIN`.

---

## B. The regression: `reduce_scatter`'s new "direct" op strands semaphores in L1

Every entry point into `multichip_decoder.py` died with statically-allocated
circular buffers colliding with L1 buffers:

| Path | Call site | Cores | CB region ends | L1 buffer at | Overlap |
|---|---|---|---:|---:|---:|
| prefill | `_rms_norm` (input layernorm) | `[0-0 - 3-0]` | 1,188,864 | 916,992 | 271,872 B |
| decode | `_tp_linear("mlp_down_decode")` | `[0-0 - 7-9]` | 811,776 | 777,728 | 34,048 B |
| decode | `_all_reduce` after MLP down | `[0-0 - 0-0]` | 177,152 | 163,328 | 13,824 B |

### The op

`ttnn.reduce_scatter` now dispatches small shapes to
`reduce_scatter_minimal_direct`, which **does not exist on the old base**: 0
files there, 12 on this one. It arrived in `b76ea5de9b3` *"[Performance]
reduce_scatter \"direct\" algorithm for small shapes (#51741)"*, an ancestor of
the new base and not of the old.

`all_gather` has no such op and strands nothing — isolating the two halves of
`all_reduce` pointed straight at reduce-scatter.

### Why it breaks

Three properties combine:

1. **Selected by default for exactly this shape.** The gate is
   `k_direct_rs_max_input_bytes = 512 KiB` of per-device input; the residual
   all-reduce at 32x5120 bf16 is 320 KiB.
2. **It stages in L1 by design** — height-sharded L1, or the interleaved
   fallback at `total_bytes <= k_l1_staging_budget_bytes ? L1 : DRAM` (4 MiB).
3. **Its semaphores fall back to the main L1 heap with no L1-small region.**
   `reduce_scatter_minimal_direct_factory.cpp`:
   `sem_buffer_type = l1_small_size > 0 ? BufferType::L1_SMALL : BufferType::L1`.

The staging is allocated *before* the semaphores. With the default
`l1_small_size = 0`, the persistent semaphores land beneath a large transient L1
staging buffer; freeing the staging does not move them, so contiguous L1 stays
capped for the rest of the process. The semaphores are cached per collective
shape, not leaked — repeating one shape holds steady at nine 64-byte blocks.

Measured, first collective in a fresh process, `reduce_scatter` at width 5120:

| variant | largest contiguous L1 after |
|---|---:|
| default (direct op) | **805,376 B** |
| `intermediate_memory_config=DRAM` → ring op | 1,460,992 B |
| `use_l1_small_for_semaphores=True` + `l1_small_size=32768` | 1,428,480 B (unchanged) |
| input 640 KiB, above the 512 KiB gate → ring op | 1,460,992 B |

The last row is the clean tell: the same op on a *larger* input is harmless,
because a larger input is not eligible for the direct path.

805,376 B is not enough for this layer. The decode MLP down projection's static
CBs alone reach 811,776 B. The prefill norm needs ~1,080,320 B contiguous
(measured by squeezing L1 with a blocker tensor and bisecting); the same
`ttnn.rms_norm` with a tiled `[1,1,1,W]` weight instead of the row-major
`[1,1,W/32,32]` contract needs only ~564,224 B — which is why the single-chip
path, running no collectives, never saw any of this.

### Fix

Run one minimal collective (width 512) at load time, while L1 is empty. It
leaves its blocks near the top; a later full-width call then has to put its
staging below them and can place its own semaphores in the holes they left. One
extra program at setup, no numerics change, nothing in the hot path. Commit
`386e8b8ba9c`.

Not `intermediate_memory_config=DRAM` or `use_l1_small_for_semaphores`: both
opt back out to the ring op, forfeiting whatever #51741 bought, and the L1-small
route additionally needs the *caller* to open the mesh with a non-zero
`l1_small_size` (at the default it fails with `bank size is 0 B`). The decoder
does not open the mesh — the harness, demo and vLLM do.

Restores the documented numbers, against
`doc/multichip_decoder/artifacts/tracy/linear_b32_dram_sharded/trace_result.json`
which recorded this exact command and candidate:

| Kind | Batch | recorded | now |
|---|---:|---:|---:|
| linear attention | 32 | 4.4718 ms | 4.3344 ms |
| linear attention | 1 | 0.9008 ms | 0.7930 ms |
| full attention | 32 | 0.7223 ms | 0.7241 ms |
| full attention | 1 | 0.5958 ms | 0.6084 ms |

All at PCC 1.0 against the single-chip baseline; prefill smoke green at S128.

### Worth reporting upstream

Item 3, the semaphore fallback — not the L1 staging, which is a deliberate
performance choice. Putting *persistent* semaphores into the main heap
underneath the op's own *transient* staging permanently fragments L1 for every
later program. Either allocate them before the staging, or keep them out of the
main heap when there is no L1-small region. Any caller running a small
reduce-scatter before a large program inherits this, and it is invisible until
something unrelated fails to fit.

Related: `ttnn.reduce_scatter` and `ttnn.all_gather` both expose
`use_l1_small_for_semaphores` and `intermediate_memory_config`; `ttnn.all_reduce`
exposes neither, so a caller using it has no way to avoid the direct path.

---

## C. Not rebase regressions

Each of these cost time and each looked like a rebase problem at first.

### C1. The device profiler's program cap

Tracy post-processing aborts with `Device data missing: Op ... not present in
cpp_device_perf_report.csv`. The dropped ops are real matmuls in the affine-scan
loop — 84 of them, from program index 1070 of the ~1,440 a sequence-128 prefill
dispatches.

I assumed the newer base had tightened `_enrich_ops_from_perf_csv`. **It had
not** — the function and its assertion are on the pre-rebase branch too. What
changed was the workload: the autoport profiled linear prefill at
`--sequence 5`, under the default cap; sequence 128 is not.

Fix: `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=8192`, wired into
`doc/kda_conv_swap/run_ab.py`.

### C2. Missing perf tooling

Installed into the repo venv with `uv pip install --python python_env/bin/python`
(`pip` is absent — it is a `uv` venv, and system `pip` refuses under PEP 668):

| Package | Version | Side effect |
|---|---|---|
| `tt-perf-report` | 1.2.9 | pins `matplotlib==3.10.9`, down from 3.11.1 (repo `requirements-dev.txt` leaves it unpinned, so in policy) |
| `tt-smi` | 6.3.0 | pulled `tt-umd` 0.9.9 → 0.9.8 |

### C3. Boards in the ERISC-timeout state

`Timed out while waiting for active ethernet core 29-25 to become active again`
— the documented recoverable fault from the `tt-device-usage` skill. `tt-smi -r`
plus a mesh smoke cleared it. Hardware state, unrelated to the rebase.

### C4. Two defects surfaced by the KDA work

Both fixed and recorded in `doc/kda_conv_swap/README.md`:

- The shared synthetic weights set `conv[:, 0, -1] = 0.5` and leave taps 0-2 at
  zero, degenerating the convolution to `silu(0.5 * x_t)`, so no conv-history
  bug can show. `doc/kda_conv_swap/check_conv_taps.py` covers it.
- `traced_synthetic_pcc.py`'s cache restore hardcoded `layout=ttnn.TILE_LAYOUT`
  while already deriving `dtype` from the destination, so a row-major cache
  could not be restored. It now takes both from the destination.

### C5. Eager batch-32 decode fails in the shared synthetic harness — not diagnosed

```
linear_attention_synthetic_pcc.py --mode decode --optimized --batch 32
TT_FATAL: Shard height 32 must match physical height 1024 for width sharded
```

Fails for `default`, `linear_final`, `linear_packed_dram`, `linear_state_fp32`,
and is pre-existing relative to the KDA work (confirmed by reverting those
files). The traced harness runs batch-32 decode fine and is what the autoport's
own batch-32 evidence used, so this combination may simply never have been
exercised. Recorded, not diagnosed.

---

## D. Repo mechanics

The repo-root `.gitignore` ignores `*.csv` and `*.log` globally, so derived perf
evidence needs `git add -f` — as the existing autoport artifacts did. A
pre-commit hook rejects files over 500 KB, which a traced decode's `profile.log`
exceeds at 783 KB; `run_ab.py` now retains only the provenance lines. Raw
captures (`.logs/`, `reports/`) are not committed — `profile_log_device.csv`
alone is 347 MB per arm.

## Open work

1. Report the semaphore fallback upstream (see B).
2. Diagnose or dismiss C5.

## Note for anyone triaging the MLP down projection

TP4 splits it to K=4352, which over 8 cores is **17 K tiles per core**, so
`in0_block_w` may only be 1 or 17. The existing `final_down_w4` / `final_down_w34`
/ `final_down_w68` candidates are all illegal there and raise
`ValueError: in0_block_w=N must divide 17 K tiles/core` before reaching the
device. `final_down_w1` was added during triage as the only other legal TP4
value. It is not part of the fix — shrinking those CBs only moved the failure to
the next site, because 811,776 B was already above the 805,376 B floor.
