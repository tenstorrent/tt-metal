<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Quasar binary-op qualification — runbook

How we test and qualify experimental-quasar binary ops on a foundation (LLK + craq-sim) that is
itself under active development. The goal is to (a) **localize** any failure to its owner and
(b) **re-baseline** on every foundation bump so regressions and newly-unblocked capability both
surface immediately.

This is the tt-metal-side companion to the foundation's own CI. We do **not** re-qualify the
simulator: craq-sim's `quasar-llk.yml` already runs the whole `test_*_quasar.py` LLK suite on the
QSR sim (8 shards, structured pass/fail/skip + failure signatures), and tt-llk gates its own
Quasar paths. We consume that and add a thin local layer.

## The two-layer model

Every binary-op capability (a "cell" = `op × dtype × compute-path`) sits on two layers:

| Layer | Probe | Answers |
|---|---|---|
| **Foundation** | LLK binary micro-tests (`test_eltwise_binary_quasar.py` FPU, `test_eltwise_binary_sfpu_quasar.py` SFPU), run **on the sim** so they transitively exercise craq-sim | "Is the primitive alive on the sim for this cell?" |
| **Op (ours)** | `test_binary_ng_no_bcast.py` (Quasar-runnable), each test asserting against a torch golden | "Did we build on it correctly?" |

"Correct" = the op test's PCC-vs-torch assertion. There is no Quasar silicon yet, so the sim is
the Quasar oracle; the sim's own fidelity-vs-hardware is craq-sim CI's concern, cross-checkable
via craq-sim's WH/BH sim builds (out of scope here).

## Triage rule (localize a failure)

When an op cell fails on the sim, run the **mapped LLK probe** for that cell:

```
op FAILS on sim
├── foundation (LLK) PASSES  → OP BUG. Fix in ttnn/.../experimental/quasar/binary_ng/.
├── foundation (LLK) FAILS   → FOUNDATION GAP. Not an op fix. Cross-check craq-sim quasar-llk.yml
│                              failure signatures; file against tt-llk or craq-sim (see below).
└── no foundation primitive  → KNOWN GAP (e.g. fp32 add/sub: SFPU float family is MUL/DIV only).
                               Keep the op test skipped with a reason; record in QUASAR_PARITY_GAPS.md.
```

This formalizes the manual bf16-mul investigation (PCC ~0.02): run the LLK SFPU float test for the
same `(op, dtype)`; if it passes, the gap is in the op, not the LLK/sim.

**Reading foundation-probe failures.** `FAIL(2)` is a *compile-step* failure (not a numeric one) and is
transient-prone on a cold build cache or the first probe in a session — the driver retries it once. If
it still fails, re-run that single `--test-id` in isolation (`run_test.sh run --test-id ...`): a clean
isolated pass means it was transient; a repeat failure is a real tt-llk compile gap. A foundation
`FAIL` while the op `PASS`es (the "FOUNDATION-ONLY FAIL" verdict) means the failing LLK path is one our
op does not exercise — not a blocker for us, but worth confirming it is not a narrow LLK gap.

## Filing a foundation gap (which repo)

Use craq-sim's simulator error taxonomy (the sim is intentionally stricter than silicon):

- `UnimplementedFunctionality` / `Unsupported` / `MissingSpecification` → **craq-sim** (ISA/sim gap),
  branch `quasar`. craq-sim also has a `/quasar-fix` autofix flow.
- LLK fails to JIT-compile (`#ifndef ARCH_QUASAR` guard) or computes a wrong result on the sim →
  **tt-llk** (`tt_llk_quasar/`), op unported or numerically wrong.
- `UndefinedBehavior` / `AssertionFailure` → simulated SW illegal / simulator bug respectively.

## The driver: `qualify_quasar_binary.py`

The authoritative op-cell ↔ LLK-test mapping lives in the driver. It runs a **single representative
LLK variant** per cell (not the full sweep — that is craq-sim CI's job) by collecting LLK ids
through the harness venv (`tt-llk/tests/.venv`), filtering by exact id-substrings, and running the
first match as one `--test-id` through `run_test.sh` (flock-serialised sim access). Op probes run
under the QSR-sim env (`TT_METAL_SIMULATOR` + slow dispatch; **not** `TT_METAL_MOCK_CLUSTER_DESC_PATH`,
which expects a cluster — not soc — descriptor).

```bash
# See the mapping (op cell -> LLK probe + op probe):
python qualify_quasar_binary.py --show-mapping

# Confirm every cell resolves (collection-only, no sim) — run after editing the mapping or when
# an LLK/op test is renamed or reparametrized:
python qualify_quasar_binary.py --validate

# Full grid on the sim; save a baseline:
python qualify_quasar_binary.py --run --out baseline.json

# Just one layer / a subset:
python qualify_quasar_binary.py --run --layer op         --cells '*.bf16.*'
python qualify_quasar_binary.py --run --layer foundation --cells 'mul.bf16.sfpu'
```

Prerequisites: the QSR sim built at `/workspaces/craq-sim/src/_out/release_qsr/libttsim.so`
(`cd /workspaces/craq-sim && ./make.py src/_out/release_qsr/libttsim.so`) and the LLK harness venv
(`cd tt_metal/tt-llk/tests && CHIP_ARCH=quasar ./setup_testing_env.sh`). Bound every run with a
hard `timeout` — a sim hang ignores pytest's `--timeout`; recover a wedged device/sim before re-running.

## Discovering what the foundation supports (the two qualification questions)

Both answers are derived **live from the LLK test collection**, so they self-update as tt-llk /
craq-sim evolve — no hand-maintenance:

**"What does LLK support that our op does not cover/test?"** (headroom to broaden into):
```bash
python qualify_quasar_binary.py --coverage
```
Lists every `(op, dtype)` the LLK no-broadcast binary suite tests, marking which our op covers and
which is headroom — e.g. `gt/lt/le/ge @ int32`, `max/min` (all dtypes), `add/sub/mul @ fp16/int8/Mx*`,
int `add/mul`. A headroom row you want is the prompt to add a cell + an op test.

**"Before I build op X (e.g. `pow`), does LLK already support it?"**:
```bash
python qualify_quasar_binary.py --supports pow       # NOT in the binary suite -> needs an LLK primitive/composite
python qualify_quasar_binary.py --supports max       # SUPPORTED @ bf16/fp16/fp32/int32 -> wiring only
python qualify_quasar_binary.py --supports subtract  # ttnn names alias to LLK ops (sub/mul/div/...)
```
SUPPORTED ⇒ broadening the op is wiring (add a cell + op test). NOT FOUND ⇒ it is an LLK/sim ask —
the report points at any other place the name appears in tt-llk (a unary family, the sources) so you
can tell "different family" from "unported". Scope: the FPU (`test_eltwise_binary_quasar.py`) and SFPU
(`test_eltwise_binary_sfpu_quasar.py`) no-broadcast suites; broadcast is a separate LLK file.

## Verify-on-bump procedure

When bumping the tt-llk or craq-sim pin (or rebuilding the sim):

1. Rebuild the sim if craq-sim changed.
2. `python qualify_quasar_binary.py --run --baseline <prev>.json --out <new>.json`.
3. Read the diff:
   - **REGRESSIONS** (was PASS, now not): if foundation regressed, file upstream; if only the op
     regressed, it is ours.
   - **PROGRESSIONS** (was not PASS, now PASS): enable the cell in op routing, drop the op-test skip,
     and update `QUASAR_PARITY_GAPS.md`.
4. Keep this baseline JSON as the new reference.

The `expected` field in the mapping (`PASS` / `KNOWN_GAP` / `OP_COVERAGE_GAP`) flags a deviation
even on a first run with no baseline.

## Pointers

- Op parity inventory & known gaps: `QUASAR_PARITY_GAPS.md` (in the op dir,
  `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`).
- Foundation CI: craq-sim `.github/workflows/quasar-llk.yml` (LLK on sim), `quasar-ops.yml` (op lane,
  currently a matmul/resnet scaffold — our binary tests are not wired in upstream).
- LLK test runner: tt-llk's `run-test` skill / `.claude/scripts/run_test.sh`.
- craq-sim build/run recipe & error taxonomy: see the team's craq-sim notes.
