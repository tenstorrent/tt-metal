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

### B1. The TP4 multichip path does not run at all: three L1 clashes

**This is the significant one, and it is unfixed.** Every entry point into
`multichip_decoder.py` fails on this base with statically-allocated circular
buffers colliding with L1 buffers. Three distinct sites:

| Path | Call site | Cores | CB region ends | L1 buffer at | Overlap |
|---|---|---|---:|---:|---:|
| prefill | `functional_decoder._rms_norm` (input layernorm), from `prefill_forward` | `[0-0 - 3-0]` | 1,188,864 | 916,992 | 271,872 B |
| decode | `_mlp_decode` → `_tp_linear("mlp_down_decode")`, `in0_block_w=17` | `[0-0 - 7-9]` | 811,776 | 777,728 | 34,048 B |
| decode | `_tp_linear` → `_all_reduce` after the MLP down projection | `[0-0 - 0-0]` | 177,152 | 163,328 | 13,824 B |

Reproduce:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/multichip_traced_decode.py --kind linear --batch 32 --steps 4
python models/autoports/qwen_qwen3_6_27b/tests/multichip_linear_attention_smoke.py --mode prefill --sequence 128
```

Both `--kind full` and `--kind linear`, both `--batch 1` and `--batch 32`, fail.

**Confirmed pre-existing relative to the KDA work on this branch.** Reverting
`functional_decoder.py`, `optimized_decoder.py` and `traced_synthetic_pcc.py` to
`e542603b2e3` reproduces the identical failure -- same program id (354), same
addresses. Nothing in the KDA change touches `multichip_decoder.py`.

Diagnosis so far: the decode sites are reachable in sequence. Setting
`mlp_down_in0_block_w=1` shrinks the down projection's in0 CB enough to clear
the second row of the table, and execution then advances to the third. So this
is a multi-site L1 budget regression, not one bad program config. The most
likely cause is that the L1 unreserved base moved up on the newer metal,
leaving less room than the multichip decoder was tuned against; the smallest
overlap is only 13.5 KB.

A related trap while triaging: TP4 splits the MLP down projection to K=4352,
which over 8 cores is **17 K tiles per core**, so `in0_block_w` may only be 1 or
17. The existing `final_down_w4` / `final_down_w34` / `final_down_w68`
candidates are all illegal on the multichip path and raise
`ValueError: in0_block_w=N must divide 17 K tiles/core` before reaching the
device. `final_down_w1` was added to give the family its only other legal TP4
value; it is a triage aid, **not a fix** -- it advances the failure, it does not
clear it.

**Consequence for the KDA work:** the fused conv could not be applied to
`multichip_decoder.py` with a measurement, because the path it would be measured
on does not execute. The single-chip results stand on their own; the multichip
port is blocked behind this L1 re-tune.

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

1. **Re-tune the multichip decoder's L1 budget for the new base** (B1). Blocks
   all TP4 evidence, and blocks porting the fused KDA conv to
   `multichip_decoder.py`.
2. Decide whether B2 is a real regression or an unsupported combination.
3. If the multichip path comes back, port the fused conv into
   `multichip_decoder._linear_attention_decode`. The per-device widths are
   Q/K/V = 512/512/1536 with `conv_width` 2560, all tile-aligned, and `K*B` is
   128 at batch 32, so the same user-major packing applies. The one extra
   consideration is `_active_mask`: the composite computes the convolution from
   the *unmasked* advanced state and only masks what it stores, so a port must
   read the advanced window before blending, not after.
