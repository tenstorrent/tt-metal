# laneMK galaxy streamer — 2^32 sem-vs-hand bit-exactness on silicon

Converts the single-input fp32/int32 unary corpus rows recorded INFEASIBLE-2^32 (the
2^16 sim sweep would need ~2^20 device runs/leg ≈ 60 days) into real silicon verdicts, by
streaming BOTH certified legs (sem = `fresh_cpp` impl 1, hand = production impl 0) over the
entire 2^32 input space in ONE open device session and comparing per-leg SHA-256.

It reuses tt-polynomial-fitter's proven flow — streaming attestation (SHA over the output
stream, never retaining 16 GiB) + `--fp32-start-bit/--count` sharding + fleet work-stealing
— but runs the **certified corpus ELF unchanged** (object identity preserved), not a
re-implementation.

## Pieces
- `fp32_stream_lib.py` — device-independent core: band enumeration/coverage, per-leg
  streaming SHA, input sum64/xor32, first-divergence bisection, .text identity gate.
- `selftest_fp32_stream.py` — MUST PASS before any galaxy run: known-equal, deliberately-
  divergent (witness bisection), .text-gate (match passes / mismatch+absent refuse).
- `elf_text_sha.py` — dependency-free `.text`-section sha256 of an ELF (no objcopy needed
  on the cluster). Matches `riscv-tt-elf-objcopy -O binary --only-section=.text | sha256sum`.
- `build_identity_gate.sh` — compiles each op's sem+hand (pinned cc1plus) and records the
  object-identity map (op → sem/hand variant + `.text` sha; asserts sem≠hand).
- `fp32_stream_sweep.py` — single-op orchestrator (resume-safe bands, per-band SHA compare,
  coverage assert, witness-band flag). Good for one op on one chip (quietbox).
- `galaxy_numeric_admission.py` — whole-campaign numerical admission. It folds
  every matching sem/hand golden sidecar, requires the semantic arm to satisfy
  the absolute oracle, and requires its global maximum ULP to be no worse than
  hand in every populated input class. Hand absolute tolerance is report-only:
  a valid semantic uplift may improve a production kernel that is out of contract.
- `run_op.sh` / `run_op_array.sh` — the fan-out as it actually ships: one Slurm
  job per op (`run_op_array.sh` is the job-array shim; Slurm is the queue), node-local
  RUNNER_TEMP, resume-safe from cached band SHAs, and a dead job only affects its own op.
  `run_op.sh` takes `SWEEP=fp32` (default, one-operand) or `SWEEP=binary`.
  This fan-out is across OPS, not chips: `run_op.sh` passes `--chip 0`, so one
  galaxy node runs one op on one chip.  For a true 32-chip shard of a single op see
  `galaxy_shard.sh` (binary ops today; `fp32_stream_sweep.py` accepts the same
  `--start-bit`/`--total`, so the unary equivalent is a small generalization away).
  The work-stealing fleet this section used to describe (`lanemk_worker.sh`,
  `lanemk_fleet.sh`, `lanemk_submit.sh`) was retired 2026-09-04 and deleted.
- The device leg is the env-gated hook in `python_tests/test_sfpu_unary.py`
  (`SFPU_STREAM` runs the in-session chunk loop; `SFPU_TILE_DIM` sizes the dispatch;
  `SFPU_WAIT_TIMEOUT` the per-dispatch Math wait) — additive and inert when unset.

## Object identity (the whole point — do not skip)
A verdict is only meaningful on the exact certified pin-59 kernel. `.text` is
farm-path-dependent (profiler `li` immediates embed the source path), so the gate is
**in-farm**: compile the certified node here, and before streaming assert the ELF's `.text`
== the recorded reference AND sem≠hand. The worker refuses (`REFUSED-IDENTITY`) otherwise.
Never run a verdict on an unverified ELF. Cross-farm `.text` hashes are provenance only,
never byte-equal.

## Route (galaxy) — see [[mac-relay-exabox]]
quietbox cannot resolve exabox DNS; the owner's Mac relays: `ssh mac-relay` then, on the
Mac, `SSH_AUTH_SOCK=$HOME/.ssh/qz-exabox-agent.sock ssh nkapre@slurm-login.exabox...`.
Two-stage rsync (quietbox→mac-relay:staging→exabox:/data). Etiquette: idle glx only, only
as many as needed, NEVER touch drain/reserved/customer nodes or kill others' jobs, BH reset
= `tt-smi -r` never `glx_reset`. Known-poisoned rack: `glx-110-c` (bh_sc36_5) — salloc there
times out — exclude it yourself with `sbatch --exclude=`.  (The retired fleet had a
`LANEMK_NODE_EXCLUDE` knob for this; it went with the fleet and nothing reads it now.)

## One-op re-run (quietbox, one chip)
```
# 1. compile the op's sem+hand (pinned toolchain) into a shared build; record identity
RUNNER_TEMP=/tmp/b pytest --compile-producer <sem-node> <hand-node>
# 2. full 2^32 sweep, resume-safe bands, per-band sem==hand compare
python3 fp32_stream_sweep.py --op sign --sem-node '<sem>' --hand-node '<hand>' \
  --farm <python_tests> --venv <py> --llk-home <tt-llk> --runner-temp /tmp/b \
  --tile-dim 256,256 --band-bits 28 --chip 0 --out <evdir>
# -> <evdir>/sign-VERDICT.txt : BIT-EXACT-ALL-INPUTS (covered==2^32) or DIVERGENT+witness bands
```

## Galaxy fan-out (all ops) — a per-op runner + ONE `sbatch --array`
Stage the tree (with the hook) + the prebuilt shared ELF build to `/data`, write the ops
that lack a verdict one-per-line to `remaining.txt`, then submit ONE array:

    export OPS_LIST=.../remaining.txt \
           GALAXY_SHARD=1 SWEEP=fp32 OPS_TSV=... IDMAP=... FLAGS_TSV=... \
           FARM_ROOT=... VENV=... OUT=... NPAR=32 BAND_BITS=23 \
           SFPU_WAIT_TIMEOUT=600
    sbatch --array=1-$(wc -l < remaining.txt) --requeue --export=ALL -J run_op \
           -p <glx-partitions> --time=720 run_op_array.sh

`run_op_array.sh` maps `$SLURM_ARRAY_TASK_ID` → that line of `remaining.txt` → one op and
runs `run_op.sh <op>` beside it: object-identity gate → stream the full 2^32 (resume-safe from
cached band SHAs) → write `<OUT>/<op>/<op>-VERDICT.txt` → **exit, which frees the galaxy**.
**Slurm is the scheduler, queue and refill**: it runs as many tasks as there are idle
galaxies at once, queues the rest, and `--requeue` retries a died task. No supervisor, no
passes, no waits. Re-submit the still-missing ops if any task dies (`ops.tsv` =
op⇥sem_node⇥hand_node; `idmap` from `build_identity_gate.sh`).

> Design lesson (why this shape): a work-stealing fleet with a supervisor leaked idle
> galaxies (a worker out of ops held its node) and abandoned ops (a crashed worker's claim
> was never re-stolen); a hand-rolled submit LOOP that batched-and-waited per pass
> serialized and ignored a wide-open cluster. The array has neither problem — one task owns
> one op and one node and frees it on exit, and the Slurm scheduler does the fan-out and
> refill. Do not reintroduce claims / work-stealing / a supervisor loop.

For per-LLK tuning, `FLAGS_TSV` is `op<TAB>exact compiler flag string`.
`galaxy_shard.sh` exports the matching string before invoking
`--compile-consumer`; without it pytest computes the stock build key and cannot
load an ELF staged under the selected configuration. The identity map still
checks the resulting semantic and handwritten `.text` independently.

## Plan the remaining exhaustive campaign

`exhaustive_campaign_plan.py` joins the current corpus, tuning search, and
validation ledger without running hardware:

```
python3 exhaustive_campaign_plan.py \
  --corpus ../sweep_2x2_ops.tsv \
  --search <search.json> \
  --validation <validation.json> \
  --out <new-plan-directory>
```

The output separates unary BF16 (`2^16`), unary 32-bit (`2^32`), joint
BF16-by-BF16 (`2^32`), class-stratified, and structural rows. Its executable
rosters use the explicit tri-arm contract: selected semantic (A), frozen
semantic (B), and frozen handwritten (C). A/B is the compiler-correctness gate;
B/C is the semantic/numeric gate. The planner refuses baseline disagreement and
will not emit the old ambiguous two-arm roster.

With `GOLDEN=1` (the default), the final status is the global numerical
admission, not sem-vs-hand equality. `<op>-VERDICT.txt` independently records
`BIT-EXACT` or `DIVERGENT`; `<op>-NUMERIC-ADMISSION.{json,tsv}` records oracle
availability, semantic and hand absolute status, global per-class ULP status,
and the admission. A divergent row may pass only when the semantic absolute
oracle passes and candidate max ULP is no worse than hand for every class.
Missing/partial oracle evidence and partial input-space coverage fail closed.
The per-slice correctness verdicts are diagnostics only; their local ULP
comparisons are never ANDed into a campaign claim.

## Measured (BH silicon)
~2.5M patterns/s per chip ⇒ **~27.7 min/leg, ~55 min/op** full 2^32 on ONE chip (chunk size
barely matters — ttexalens debug-bus L1 I/O + per-dispatch soft-reset bound; sharding across
chips is the lever). `sign` proved BIT-EXACT-ALL-INPUTS (16 bands tiling [0,2^32), 0 witness
bands) on quietbox and reproduced byte-identically on an exabox glx host (cross-farm).

## Gotchas banked
- Sharing one `RUNNER_TEMP` on NFS races on conftest `order_records` mkdir → node-local
  per-chip `RUNNER_TEMP` under a unique per-job temporary root. The root is
  removed when the driver exits, so a later job cannot reuse a stale staged
  build.
- Galaxy hosts need a generous `SFPU_WAIT_TIMEOUT` (harness default 2 s times out on a
  cold first dispatch); band-0 (all-denormal patterns) is slow-or-hangs for a few ops
  (softplus/hardshrink/add1/softshrink) even at 120 s — investigate per-op, don't force.
- The mac relay is flaky (laptop sleep). The fleet runs detached (`setsid nohup`) and
  persists verdicts to `/data`, so a relay drop loses observability, not the run; re-collect
  when it returns. Nodes auto-reap on completion.

## Wired into prove_all (plumbing landed; routing has not)
`prove_all.py` now has a `silicon_stream` engine that shells out to `galaxy_shard.sh`
verbatim and translates its combined verdict into the existing ledger classes
(`BIT-EXACT-ALL-INPUTS` -> `SILICON-EXHAUSTIVE`, `DIVERGENT` -> `DIVERGENCE-CERTIFIED`,
anything else -> `UNSWEPT`, i.e. a reported operational failure).  It re-enters the shard
on every run rather than caching a verdict, because the driver's cache key cannot see the
shard geometry while the streamers' own per-band resume (`stream_resume.py`) can.

    python3 prove_all.py --engine silicon_stream \
      --silicon-farm-root <staged tree> --silicon-venv <py> --silicon-idmap <idmap.tsv>

**No manifest row routes here yet.** The 31 `INFEASIBLE-2^32` rows still say
`engine=classify` and still carry `sem_node`/`hand_node` = `-`.  Flipping them is a
measurement decision (device time on a galaxy node) and needs their pytest node ids plus an
in-farm `IDMAP` from `build_identity_gate.sh`; the plumbing above does not supply either.
