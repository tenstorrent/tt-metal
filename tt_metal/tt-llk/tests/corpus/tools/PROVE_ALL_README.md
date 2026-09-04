# prove_all — prove every SFPU board op, one command

`prove_all.py` runs **both** proof engines across **all 134 kernel-decided
board ops** at the current compiler pin and emits the master coverage ledger.
It is the paper's artifact-evaluation entry point: **one command reproduces the
proof-coverage census.**

```
make prove-all            # or: python3 prove_all.py --all
make prove-all-selftest   # guards routing + census math (must pass)
make prove-all-manifest   # print the routing census, no proving
```

## What it does

For each of the 134 ops (the laneFM `FINAL-BOARD.tsv` set, asserted 1:1 against
the checked-in routing manifest) it picks one engine and records a provability
class, then joins in two provenance-pinned overlays under a strict precedence.

| engine / overlay | what it proves | how |
|---|---|---|
| `formal_equiv.py` (laneJO) | per-lane bit-exact equivalence over **all** inputs | z3 QF_BV translation validation on the **final emitted SFPU stream**, on the pinned *instrumented* craq-sim; VALIDATION GATE = concrete replay reproduces every trace snapshot before any verdict |
| `bitexact_sweep.py` (laneJN) | single-input **2^16** exhaustive equivalence | sweep on the pinned craq-sim (sim-only `probe/sweep/verdict`) |
| classify (no run) | 2^32 single-input / cross-lane | recorded infeasibility / one-lane-model scope refusal, with reason |
| KC-silicon overlay | device-exhaustive 2^16 | **recorded** laneKC silicon sweeps (a device campaign; not re-run here) |
| JO-domain overlay | documented-deliverable-domain equivalence | **recorded** laneJO z3 re-proofs (upgrade clamp / mulint32) |

**The two engines are re-run live from scratch every invocation.** The two
overlays are recorded, sha-pinned inputs (silicon needs a device lane; the
domain proofs are z3 on the identical instrument) and are clearly sourced in
every ledger row.

Precedence (strongest first):

```
SILICON-EXHAUSTIVE > SMT-PROVEN-ALL-INPUTS > SMT-PROVEN-DOMAIN >
DIVERGENCE-CERTIFIED > SIM-BIT-EXACT-16 > UNDECIDED-Z3-TIMEOUT >
INFEASIBLE-2^32 > NOT-EXHAUSTIBLE > SCOPE-REFUSED > UNSWEPT
```

`machine-certified-equal := SILICON-EXHAUSTIVE ∪ SMT-PROVEN-ALL-INPUTS`.

## Routing is data-driven and auditable

`prove_all_manifest.tsv` (checked in) has one row per op:
`op, board_class, arity_space, engine, sem_node, hand_node, expected_class_ref,
reason`. Every routing decision is visible; `expected_class_ref` is a
**reconciliation reference only** (the live run re-derives the actual class).
`make prove-all-manifest` prints the engine census and re-asserts op-set == board.

## Provenance gate (fails loudly)

On every run the driver verifies by sha256 and records into `RUN-MANIFEST.json`:

* active **cc1plus** — must be pin-59 (`b013967fffaa…`);
* **JO instrumented sim** — must equal `ba23c3f16912…` (+ its `soc_descriptor.yaml`);
* **bitexact pinned sim** — must be `1d162f0adf67…`;
* both engines, the board, the manifest, both overlays, the harness venv.

A missing or mismatched required instrument aborts with exit 3 (use `--no-gate`
to record-without-enforce for diagnostics only). The ON flag set is imported
from the canonical `sweep_2x2.ON_FLAGS` (pin-59 ON-39).

## Re-run / resume / budget

* **Resume-safe:** a valid `verdicts/<op>/prove_all_verdict.json` is reused;
  `--force` re-proves from scratch.
* **Per-op budget:** `--timeout` (default 1800 s, like laneJO); on expiry a
  formal row records `UNDECIDED-Z3-TIMEOUT` and never hangs.
* **Subsets:** `--only '<glob>[,<glob>...]'`, `--engine {formal_equiv,bitexact,classify}`.
* **Parallelism:** `--jobs N` drives the bitexact sim workers (the shared
  instrumented sim + deep z3 queries run serially by design).
* Deterministic given the same pin, so drift across runs is detectable.

## Outputs (under `--out`, default `~/sfpi-uplift/laneMH-evidence-20260903/run`)

* `MASTER-COVERAGE-LEDGER.tsv` — one row/op: class, engine, arity, evidence-ptr, reason.
* `SUMMARY.txt` — class census, machine-certified headline, and the paper's
  `36 fast → 25/11` recomputation.
* `verdicts/<op>/prove_all_verdict.json`, `formal/<op>/…`, `bitexact/…` — per-op evidence.
* `RUN-MANIFEST.json` — pin, all shas, args, census, wall.
* `SHA256SUMS`.
