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
- `lanemk_worker.sh` / `lanemk_fleet.sh` — the galaxy fan-out: idle-glx-only salloc, one
  work-stealing worker per host, NFS-atomic claims, node-local RUNNER_TEMP, resume-safe,
  `trap reap EXIT` auto-releasing its own allocations.
- The device leg is the env-gated hook in `python_tests/test_sfpu_unary.py`
  (`LANEMK_STREAM` runs the in-session chunk loop; `LANEMK_TILE_DIM` sizes the dispatch;
  `LANEMK_WAIT_TIMEOUT` the per-dispatch Math wait) — additive and inert when unset.

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
times out; the fleet excludes it by default (`LANEMK_NODE_EXCLUDE`).

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

    export LANEMK_OPS_LIST=.../remaining.txt LANEMK_RUN_OP=.../lanemk_run_op.sh \
           OPS_TSV=... IDMAP=... BUILD=... VENV=... LLK_HOME=... PYDIR=... OUT=... \
           LANEMK_WAIT_TIMEOUT=600
    sbatch --array=1-$(wc -l < remaining.txt) --requeue --export=ALL -J lanemk_op \
           -p <glx-partitions> --time=720 lanemk_array.sh

`lanemk_array.sh` maps `$SLURM_ARRAY_TASK_ID` → that line of `remaining.txt` → one op and
runs `lanemk_run_op.sh <op>`: object-identity gate → stream the full 2^32 (resume-safe from
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

## Measured (BH silicon)
~2.5M patterns/s per chip ⇒ **~27.7 min/leg, ~55 min/op** full 2^32 on ONE chip (chunk size
barely matters — ttexalens debug-bus L1 I/O + per-dispatch soft-reset bound; sharding across
chips is the lever). `sign` proved BIT-EXACT-ALL-INPUTS (16 bands tiling [0,2^32), 0 witness
bands) on quietbox and reproduced byte-identically on an exabox glx host (cross-farm).

## Gotchas banked
- Sharing one `RUNNER_TEMP` on NFS races on conftest `order_records` mkdir → node-local
  per-host RUNNER_TEMP (workers copy the prebuilt ELFs local).
- Galaxy hosts need a generous `LANEMK_WAIT_TIMEOUT` (harness default 2 s times out on a
  cold first dispatch); band-0 (all-denormal patterns) is slow-or-hangs for a few ops
  (softplus/hardshrink/add1/softshrink) even at 120 s — investigate per-op, don't force.
- The mac relay is flaky (laptop sleep). The fleet runs detached (`setsid nohup`) and
  persists verdicts to `/data`, so a relay drop loses observability, not the run; re-collect
  when it returns. Nodes auto-reap on completion.

## Fast-follow TODO
Integrate as `prove_all.py --fleet <op-set>` so there is ONE census pipeline (route the
INFEASIBLE-2^32 rows to this streamer, fold SILICON-EXHAUSTIVE-2^32 verdicts into the
ledger). Deferred from the first landing to avoid wiring prove_all mid-campaign; the tools
run standalone as above meanwhile.
