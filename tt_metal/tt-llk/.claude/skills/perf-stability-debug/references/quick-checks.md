# Quick checks: is the move real or noise?

Do these in order. Each one is cheap compared with the methods that follow.

## 0. Read the gate report first (no board)

1. Download the gate artifacts of both sides (`gh run download <run id> -R tenstorrent/tt-metal`).
2. Open `perf_gate_report.regressions.csv` and `.points.csv`.
3. Look at the shape of the move:
   - every config of one module moves by the same cycles, and `TEXT_SIZE` changes: a code
     change. Probably real. Go to step 3 (stage 2) to confirm.
   - a few configs of `perf_math_matmul` move by 5–40 %, in both directions, `TEXT_SIZE`
     equal: the classic order/placement signature (LoFi configs first). Go to step 1.
   - the largest improvement is as large as the largest regression: noise is likely.
4. Check that both sides used the same toolchain (`tt_metal/sfpi-version`) and the same run
   types. A toolchain bump changes code everywhere (see `methods-code-and-toolchain.md`).

## 1. Same-code control on one board

Measure ONE build twice. Compile once, measure twice, so the ELFs are identical.

```bash
cd tt_metal/tt-llk/tests/python_tests
export CHIP_ARCH=wormhole            # or blackhole
export TT_LLK_DISABLE_ASSERTS=1      # as CI; otherwise INIT code differs from CI
export LLK_PERF_RUN_TYPES=L1_TO_L1   # what the gates measure
export RUNNER_TEMP=/tmp/stab-ctl     # build dir = $RUNNER_TEMP/tt-llk-build; keep it for both passes
A=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=60)

pytest "${A[@]}" --compile-producer -n 10 perf_math_matmul.py
for t in ctl-a ctl-b; do
  PERF_RUN_TAG=$t pytest "${A[@]}" --compile-consumer --dist loadgroup -n 15 perf_math_matmul.py
done

S=../../.claude/skills/perf-stability-debug/scripts/perf_bitexact.py
python3 $S ../../perf_data/runs/ctl-a ../../perf_data/runs/ctl-b --marker TILE_LOOP
```

Use the module(s) the gate flagged. For a few configs, pass node ids instead of the file
(quote them: the ids contain `[` and `]`).

How to read it:
- **0 points not bit-exact**, or only a few tens of ~118,000 points below 0.005 % (Blackhole
  full suite, 2026-10): the harness is repeatable for this module. A move between two commits is
  either a code change or a P2 effect. Go to step 3.
- **Hundreds of points, up to ~10 %, mostly `perf_math_matmul`** on Wormhole: the known
  order effect (state left by the previous kernel). Main does not have the warm-up fix yet.
  See `methods-order-and-state.md` § warm-up. Do not trust any WH L1_TO_L1 move below ~15 %
  from one run.
- **The script warns that TEXT_SIZE differs**: the two passes did not use the same build.
  Check `RUNNER_TEMP`.

Pitfalls:
- `perf_compare_commits.sh` (skill `perf-regression-check`) caches per commit. Calling it with
  the same ref twice reuses the cache and is not a control.
- `--dist loadgroup` with no `xdist_group` marks behaves like dynamic `load`: the test order on
  each core still depends on timing. That is what CI does, so keep it for a control.
- A subset of a module does not reproduce scheduling effects of the full shard: xdist's chunk
  is `pending // workers // 2`, so a small selection gets small chunks. Measure the module or
  the shard the gate measured.

## 2. The same control on a second board

Run step 1 on another board (or as two CI runs). If one board shows 1,000+ points and the
other shows 0–3, the first board is noisy. In the 2026-10 campaign one Wormhole n150
(of six) showed ~1,200 not-bit-exact points even with a fixed order; it was dropped.
Several lab boards also ran ~1.8× slower than others. Never mix board timings.

## 3. Stage 2: re-measure the candidates at 5 code positions

Stage 2 re-measures each flagged test on base and head with all TRISC code shifted by
+0, +4, +20, +36 and +52 B. A real regression is slower at all 5 positions. A placement
effect is slower at some positions only.

Where it lives: PR #62981 (Blackhole merge gate, report-only). Until it merges, use branch
`nstojictt/llk-perf-gate-stage2-bh`: `tests/run_llk_perf_placement.sh`,
`perf/placement_check.py`, `LLK_CODE_SHIFT`, `LLK_WINDOW_HASH`, `--node-list`.

On a board, per side (base tree and head tree, both with stage-2 support):

```bash
cd tt_metal/tt-llk
# nodes.txt: one pytest node id per line. To make it from a stage-1 compare, measure the
# stage-1 head pass with LLK_WINDOW_HASH=1 (it then writes *.windowhash files that map
# report rows to node ids), then:
#   python3 perf/placement_check.py nodes --candidates stage1.regressions.csv \
#     --map "perf_data/runs/<stage-1 head tag>/**/*.windowhash" --out nodes.txt
LLK_PERF_RUN_TYPES=L1_TO_L1 LLK_PERF_NODE_LIST=$PWD/nodes.txt RUNNER_TEMP=/tmp/s2-head \
  ./tests/run_llk_perf_placement.sh            # writes perf_data/runs/p0, p4, p20, p36, p52
cp -r perf_data/runs/p{0,4,20,36,52} /tmp/s2/head/
```

Then decide:

```bash
python3 perf/placement_check.py decide --threshold 0.02 --min-cycles 30 --module-threshold "" \
  --candidates stage1.regressions.csv --baseline /tmp/s2/base --current /tmp/s2/head \
  --arch wormhole --report stage2.md
```

How to read `stage2.md`:
- **regression** (slower at all 5, loop code changed): real.
- **review** (slower at all 5, loop code identical): real cost moved into the loop from
  somewhere else, or a timing effect. Look at it.
- **partial / placement**: the value follows the code address. A hardware effect, not a
  regression of the change. Keep it visible; do not drop it.
- **not reproduced**: no position is slower. Noise or order. On Wormhole without the
  warm-ups this is common (230 points in 32 of 76 replayed reports).

Pitfalls:
- A base that predates stage 2 or pins another SFPI cannot be re-measured: stage 2 skips it.
- On Blackhole CI legs, stage 2 has a few minutes. It measures the 10 largest tests first; the
  rest of a big regression shows as "not re-measured".
- The 5 shifts are 0 or 4 modulo 16. Whether they cover every 16-B phase depends on the
  function alignment (issue #63548).
- The loop fingerprint only labels rows. Blocking follows "slower at all 5".
