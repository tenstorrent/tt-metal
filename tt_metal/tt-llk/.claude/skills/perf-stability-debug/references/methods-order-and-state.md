# Methods: test order, kernel state, workers, L1 traffic

Use these when the same build is not bit-exact (P1). Base every arm on the same compiled
build (one `--compile-producer` pass, several `--compile-consumer` passes with their own
`PERF_RUN_TAG`), and compare with `scripts/perf_bitexact.py`.

Common setup for the commands below:

```bash
cd tt_metal/tt-llk/tests/python_tests
export CHIP_ARCH=wormhole TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP=/tmp/stab
A=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=60)
LLK_PERF_RUN_TYPES=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE \
  pytest "${A[@]}" --compile-producer -n 10 perf_math_matmul.py   # build every run type once
```

Building all run types once lets every arm below reuse the same ELFs.

## 1 worker vs 15 workers

When: first split of a P1 move. One worker runs all tests back to back on one core, in a
fixed order.

```bash
LLK_PERF_RUN_TYPES=L1_TO_L1 PERF_RUN_TAG=w1-a pytest "${A[@]}" --compile-consumer -n 1 perf_math_matmul.py
LLK_PERF_RUN_TYPES=L1_TO_L1 PERF_RUN_TAG=w1-b pytest "${A[@]}" --compile-consumer -n 1 perf_math_matmul.py
```

Read: 1 worker bit-exact and 15 workers not means the value depends on what ran before on
the core (order) or on what the other cores do. The next two methods separate the two.
2026-10 example (WH, SFPI 7.90, same ELFs): 1 worker 22,680 of 22,680 bit-exact; 15 workers
~1,300 not bit-exact, up to 9.2 %.

Pitfalls: 1 worker is slow (the whole module serially). It shows a deterministic history, not
"no history".

## Fixed order vs dynamic order

When: 1 worker is clean, 15 are not. Find out whether the per-core sequence matters.

Tools on main:
- `--dist loadgroup` plus `xdist_group` marks fixes which worker runs which tests. Without
  marks it behaves like `load` (dynamic).
- `--test-order-file <json> --rewind-runner <worker>` replays one worker's recorded sequence
  (format: `{"<worker>": [{"test": "<nodeid>"}, ...]}`; run it with `-n 0` or `-n 1`).
- `perf_bisect.py --stable-groups N --dist loadgroup` (branch `nstojictt/llk-perf-bisect`)
  patches a stable crc32 grouping into a CI run.

Read: fixed order bit-exact, dynamic order not: the measured value depends on the preceding
kernel on the core. Then compare two DIFFERENT fixed orders (forward and reversed): if values
differ between orders, the order picks the value. Fixed order is a workaround, not a fix.

Pitfalls:
- With `loadgroup`, xdist gives group k to whichever worker registered k-th, and the worker
  number picks the core. Bind the core to the group if you need the same core.
- `--record-test-order` reads the Tensix state after every test (it halts the TRISCs to read
  registers). On Wormhole that read itself changes the carried state. Do not record the
  order in a run whose numbers you want to keep.

## Warm-up on/off (Wormhole)

When: WH L1_TO_L1 is not repeatable with 15 workers. The gates measure only L1_TO_L1, so each
measured kernel follows a kernel of ANOTHER test, picked by the scheduler.

The fix is two commits of PR #58068 (not on main on 2026-10-30):
- `c75a4733ceb` run every kernel of a test once, unrecorded, before measuring it;
- `be61600d8bb` in a run narrowed to L1_TO_L1, also warm up the first isolate kernel.

Apply both (`git cherry-pick c75a4733ceb be61600d8bb`) and repeat the same-code control.
2026-10 results: perf_math_matmul 0 of 68,040 points not bit-exact (2 boards), full WH suite
0 of 108,366; without them 696–853 TILE_LOOP points on the same boards. Cost: about +18 %
measure time on WH matmul; Blackhole stage 1 about ×1.19.

Pitfalls:
- The first commit alone is not enough: one config needed two X kernels after a bad
  predecessor before it converged.
- Any ONE isolate kernel before X makes X independent of the predecessor, but on its own
  level (a different value from the warmed-up X). Compare like with like.
- The carried state is not identified. Ruled out: the TRISC branch predictor contents (the
  TRISC reset clears them in Versim) and the SFPU lane enables. Turning the math or pack
  predictor off DURING the predecessor flips X, so the state depends on how the predecessor ran.

## MATH_ISOLATE prelude (alternative to the warm-up)

When: you cannot use the warm-ups. Measure `MATH_ISOLATE,L1_TO_L1` instead of `L1_TO_L1`;
the measured L1_TO_L1 kernel then follows an isolate kernel.

```bash
LLK_PERF_RUN_TYPES=MATH_ISOLATE,L1_TO_L1 PERF_RUN_TAG=pre-a pytest "${A[@]}" --compile-consumer --dist loadgroup -n 15 perf_math_matmul.py
```

Results: full WH suite 7 of 36,050 not bit-exact (0.01 %); +32 % measure time; CI legs +39 %.
On Blackhole it adds +72 % with no gain. Draft PR #61761 carries it.

## L1 traffic from the host or BRISC

When: values change run to run with a fixed order and 1 worker, or a small timing change
(release gap, BRISC delay) moves them.

Mechanism: one L1 access by another core at the wrong cycle can move the packers into a slow
rhythm for the whole loop (WH). On BH, BRISC and TRISC clock reads share one register block.
Sources: the host polling mailboxes in L1, BRISC polling its command, profiler zone records.

How to test: compare main with the stability stack PR #61357 ("host and BRISC stay off L1 while
a kernel runs": host waits on overlay registers, BRISC polls every 100 µs on NOPs). Build both,
run the same-code control on each. Blackhole: main + #61357 alone was bit-identical run to run.

Pitfall: a test-only fast BRISC poll (1 µs) changes the results. Keep the production firmware.

## loadgroup on/off

`run_llk_perf_*.sh` on main measure with `--dist loadgroup` (#57015, a temporary fix for the
Blackhole order effect, with `--maxschedchunk=200` in `pytest.ini`). On top of the stability
stack, Blackhole is as repeatable without it (CI run 40725553397: 82 vs 78 of 118,601 points
not bit-exact, all < 0.005 %). On Wormhole it never had an effect. To test it, edit the
`--dist` argument in the runner script on a branch, or use `perf_bisect.py --dist load`.

## Board quality and CI vs lab

- Same control on 2–3 boards. Drop a board that is not bit-exact with a fixed order.
- CI runners are 2.1–2.4× slower than lab cards. Some CI-only residual movers did not
  reproduce on a lab card or in Versim. Confirm a CI-only effect with two CI runs on fresh
  runners (`perf_bisect.py measure <sha> --arch wormhole --runs 2`).
