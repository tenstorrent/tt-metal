# Methods: code position, toolchain, bisect

Use these when the same build is bit-exact, but a change that does no work moves values (P2),
or when you must find the commit where a move started. `A` and the environment are the
common setup of `methods-order-and-state.md` (`CHIP_ARCH`, `TT_LLK_DISABLE_ASSERTS=1`,
`RUNNER_TEMP`, `LLK_PERF_RUN_TYPES=L1_TO_L1`).

## NOP and code-shift insertion

When: you suspect the value follows the code address, not the work. Typical: a change outside
the measured loop (an init, a new helper, a toolchain bump) moves TILE_LOOP of unrelated
configs, in both directions.

Two ways:

1. **Shift all TRISC code** (stage-2 tooling, branch `nstojictt/llk-perf-gate-stage2-bh` until
   #62981 merges). `LLK_CODE_SHIFT=<bytes>` adds `-Wl,--defsym=__llk_code_shift=<bytes>` at link
   time; objects are reused, so each position is cheap. Measure positions 0, 4, 20, 36, 52 with
   `tests/run_llk_perf_placement.sh` (see `quick-checks.md` § 3), or one position by hand:

   ```bash
   LLK_CODE_SHIFT=20 PERF_RUN_TAG=p20 pytest "${A[@]}" --compile-producer -n 10 <nodes or files>
   LLK_CODE_SHIFT=20 PERF_RUN_TAG=p20 pytest "${A[@]}" --compile-consumer --dist loadgroup -n 15 <nodes or files>
   ```

2. **One NOP in one place** (any tree). Add `asm volatile("nop");` at the top of an init that
   runs once before the loop, for example `_llk_pack_init_` in
   `tt_llk_wormhole_b0/llk_lib/llk_pack.h`, rebuild, and compare with the build without it.
   N = 1..4 NOPs covers all four 4-byte positions in a 16-byte block.

Read: the same build is bit-exact, but +4 B moves values: a placement effect. Expect discrete
levels (fast / mid / slow), not a continuous spread. One NOP moved WH L1_TO_L1 by up to 16 %
and Wormhole stays above 2 % on some configs at every fixed threshold below ~19 %. A change
that is slower at every position is real.

Pitfalls:
- `TEXT_SIZE` may not change: section alignment absorbs a 4-byte NOP. Use `objdump -d` on the
  two ELFs to prove that only your NOP moved.
- A NOP inside code that the harness places out of line (an INIT placed in its own section)
  does not move the loop. That test is vacuous. Put the NOP where it shifts the loop.
- `-fpatchable-function-entry` (one NOP before every function) did not assemble on the
  stability stack; it was replaced there by other no-work arms.
- Asserts on (no `TT_LLK_DISABLE_ASSERTS=1`) change INIT code. Shifts measured with asserts
  on do not transfer to CI.
- The 5 shifts are 0 or 4 modulo 16. If functions are 16-byte aligned, the real shifts differ.
  Read the TILE_LOOP start address per position with `objdump` (issue #63548).

## SFPI version pin / bump

When: a gate report follows an SFPI bump, or a same-code control changed after one. A new
SFPI changes code generation broadly (7.92: 16 of 20 math ELFs of a matmul sample changed),
which changes both the work and the code layout. It can also change how sensitive a kernel is
to the previous kernel: on Wormhole, SFPI 7.89 code moved 57 points (max 0.02 %) when the
predecessor changed, 7.90 code 1,861 points (max 10.1 %).

Pin a tree to another SFPI (same commit, only the toolchain differs):

```bash
git show <old commit>:tt_metal/sfpi-version > tt_metal/sfpi-version
cat tt_metal/tt-llk/tests/sfpi/sfpi.version          # what is installed
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh)   # installs the pinned sfpi into tests/sfpi
```

Then measure both trees with the same harness and compare.

Read:
- With both sides on the same SFPI, a "regression" that came from the bump disappears. In the
  replays, 3 SFPI PRs gave 0 candidates with 7.89 pinned on both sides; #59612 (a real SFPU
  change) stayed confirmed in every arm.
- Codegen vs noise: an SFPI regression shows a `TEXT_SIZE` change and the same delta in every
  repeat. Treat two runs as replicates only when their `TEXT_SIZE` is equal.

Disassembly diff: `tt_metal/tt-llk/.claude/scripts/sfpi_disasm.sh` on branch
`nstojictt/sfpi-disasm-tool` builds one commit with two toolchains and diffs the disassembly:

```bash
sfpi_disasm.sh --work <dir> --python <venv python> --commit <sha> --other-sfpi <sha with the other sfpi-version> \
  [--test perf_math_matmul.py] [--select <-k expr>] [--jobs N]
```

Pitfalls:
- `setup_testing_env.sh` also installs pre-commit hooks into the repo. In a shared checkout
  run a copy without `setup_precommit`.
- A worktree with a symlinked `tests/sfpi` builds with the other checkout's compiler. Check
  `tests/sfpi/sfpi.version` against `tt_metal/sfpi-version` before every run.
- SFPI 7.9x needs glibc 2.34+ (Ubuntu 22.04+). Use a container on older hosts.
- ELF bytes embed the build path. Compare disassembly, or builds made in the same directory.
- Newer code may not build with an old SFPI (two replays could not pin 7.89).

## Bisect over LLK-perf-input commits

When: a same-code control was clean at one commit and is not at a later one, or a perf move
started somewhere in a range.

1. List the commits that can move an LLK perf number:

   ```bash
   git rev-list --reverse --first-parent GOOD..BAD | while read c; do
     git diff --name-only --diff-filter=ACMRT "$c^" "$c" | .github/scripts/llk-perf-inputs-changed.sh | grep -q true && echo "$c"
   done > commits.txt
   ```

2. Probe the middle commit with a same-code control (`quick-checks.md` § 1) on a board, or in
   CI with `perf_bisect.py` (branch `nstojictt/llk-perf-bisect`,
   `tt_metal/tt-llk/.claude/scripts/perf_bisect.py`):

   ```bash
   perf_bisect.py measure <sha> --arch wormhole --runs 2 --compare-ref origin/main
   perf_bisect.py bisect --good <sha> --bad <sha> --arch wormhole --signal L1_TO_L1 --compare-ref origin/main
   perf_bisect.py cleanup          # deletes the perf-bisect/<sha> branches it pushed
   ```

   `measure` dispatches `llk-perf.yaml` twice for one commit and counts fires between the two
   runs. Useful options: `--workers`, `--dist`, `--maxschedchunk`, `--stable-groups`,
   `--test-filter`, `--split-into`, `--apply <ref>` (measure a candidate fix on top), `--tag`
   (repeat a configuration), `--use-runs` (reuse finished runs), `--sol`.

Read: the first commit where the control moves. Two bisects so far:
- Blackhole, 2026-09: `fd8a72b6cbd` (#53642) raised `--maxschedchunk` 10 → 2000; test order
  on each core became timing-dependent. Proven by reverting only that line.
- Wormhole, 2026-10: `a02a3fea40c` (#62327, SFPI 7.90.0). Probes before it 9 of 9 clean,
  from it 4 of 4 moved.

Pitfalls:
- The default `--compare-ref` branch no longer exists; pass `--compare-ref origin/main`.
- `measure` forces SoL off unless `--sol`. A small `--test-filter` does not reproduce
  scheduling effects (small xdist chunks); keep the full shard for order problems.
- With `--run-count`, `min(` columns enter the point key of the compare and every point looks
  clean. Re-derive the views by hand.
- Do not kill collectors with `pkill -f` on a shared machine: other sessions use the same
  command line. Kill by PID.
- A commit that is "bad" may only expose a mechanism that was there before (SFPI 7.90 exposed
  the order effect). Say so in the report.
