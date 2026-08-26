# Approximate exp: opt-in `SKIP_NEGATIVE_SANITIZE`, and the harness bug it exposed

Branch: `sfpu-exp-skip-negative-sanitize`

Two write-ups. They ship together because the second was found by the first: the one input that
provokes the exp kernel's known `ITERATIONS` limitation also wedges the Tensix, and the test
harness turned that single wedged core into a run's worth of failures attributed to unrelated
kernels.

| # | Document | Change | Result |
|---|---|---|---|
| 01 | [exp-approx-skip-negative-sanitize.md](01-exp-approx-skip-negative-sanitize.md) | opt-in `SKIP_NEGATIVE_SANITIZE` on approx exp, Wormhole and Blackhole | **2.67x** fewer issue slots, opt-in, default unchanged |
| 02 | [harness-wedge-cascade.md](02-harness-wedge-cascade.md) | stop one wedged core reporting as dozens of kernel failures | **fixed** — 3 desync/recovery defects |

## In one paragraph each

**01.** `calculate_exponential<APPROXIMATION_MODE=true, CLAMP_NEGATIVE=true>` — the default, and
what SDPA's softmax uses — spends 24 SFPU issue slots on 8 datums, and 15 of those 24 are the
−88.5 clamp pass. `SKIP_NEGATIVE_SANITIZE` (new, default `false`) drops it: 9 slots per 8 datums,
bit-identical for every input ≥ −88.5. It is a caller contract rather than a hint, and violating
it fails loudly in the wrong direction (≈3.4e38, not ≈0), so **it is not enabled at any call
site** — the reasoning, including why flash attention's additive mask makes `sdpa.h` unsafe for
it, is in 01. Same numbers on both arches, read off the compiler.

**02.** The host↔BRISC command handshake desynchronises in two ways and never recovers from
either: a timed-out command leaves the host counter permanently ahead (and writing the wrong
double-buffer slot), and a *successful* mid-run bring-up resets BRISC's counter to 0 without
resetting the host's. On top of that, nothing dropped the cached bring-up state, so no test ever
retried. One wedge therefore produced 57–73 independent-looking failures whose count drifted
between runs. Fixed, plus a latch that skips the remainder with a reason once a bring-up has
demonstrably failed to help — because a RISC soft reset cannot clear Tensix-level state, so only
a board reset recovers the core and the honest thing the harness can do is say so.

## Relationship to the SFPU LUT branch

`sfpu-wh-lut-accuracy-and-tti-scheduling` carries the approximate tanh and sigmoid work
(6-entry `SFPLUTFP32` tables, raw TTI scheduling) and its own report. It was originally one
branch with this exp work folded in; the two were split so each reviews on its own terms. The
kernels do not overlap — that branch touches `ckernel_sfpu_tanh.h` and
`ckernel_sfpu_sigmoid_appx.h`, this one touches `ckernel_sfpu_exp.h` and the harness — so the two
can land in either order.

## Reproducing

```bash
cd tt_metal/tt-llk/tests && source .venv/bin/activate && cd python_tests
pytest test_sfpu_unary.py -k "Exp"
```

If any test reports `TENSIX TIMED OUT`, **reset the board (`tt-smi -r`) before drawing any
conclusion from the rest of the run** — see 02 for why that is not optional, and what the harness
now does about it.

Issue-slot counts are read from the generated kernel rather than argued from the source; 01 shows
the disassembly and how to get it.
