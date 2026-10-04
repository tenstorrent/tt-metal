# prove_all — historical pin-59 coverage reproduction

> This tool is not the current compiler/knob admission path. It reconstructs a
> historical coverage census by joining pin-59 engines and recorded overlays.
> Use [`formal_campaign.py`](FORMAL_README.md) for current-tuple formal results.

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
| `bitexact_sweep.py` (laneJN) | single-input **2^16** exhaustive equivalence | sweep on the pinned craq-sim; VALIDATION GATE = the sim executor must reproduce the row's **device anchors** bitwise on both legs (see below) |
| `galaxy_shard.sh` (laneMK) | single-input **2^32** exhaustive equivalence on **real silicon** | the op's whole space sharded across a galaxy node's 32 chips, both identity-gated legs per chip, folded back by `galaxy_combine.py`; `BIT-EXACT-ALL-INPUTS` -> `SILICON-EXHAUSTIVE` |
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

## The bitexact leg needs device anchors (`--allow-hardware`)

`bitexact_sweep.py` only counts a sweep whose executor it has **validated**:
the pinned craq-sim must reproduce, bitwise on both legs, the row's
`rows/<row>/anchor-{sem,hand}.npz` — dumps of the *same* registered stimuli
taken from a **silicon** run of the same two pytest nodes. Without them the
engine reports `EXECUTOR-UNVALIDATED`, which the driver maps to
`SCOPE-REFUSED`: the whole 32-row leg then asserts nothing.

The anchors are produced by the engine's own `anchor` stage, and until now
nothing ever invoked it, so under the driver's `bitexact/` tree they never
existed. `prove_all.py --allow-hardware` runs that stage (serial silicon
pytest runs under `/tmp/tt-device.lock` + `/tmp/tt-llk-sfpu-silicon.lock`)
into the same `--out` tree before the sim batch:

```
make prove-all-anchor          # silicon: anchor + prove, one command
```

* **One-time cost, then reusable.** The anchors are ~KB `.npz` dumps and the
  stage skips any leg already present, so after one anchoring run a plain
  `make prove-all` on the same `--out` tree validates with **no device**.
  Anchors are therefore a *recordable* artifact — but they must be produced on
  silicon once; there is no way to synthesize them from the tree.
* Rows whose anchors did not appear keep the honest
  `EXECUTOR-UNVALIDATED` / `SCOPE-REFUSED` verdict — the flag adds evidence,
  it never relaxes the gate.
* `--allow-hardware` always re-proves its own rows: a verdict cached before
  any anchor existed says `SCOPE-REFUSED`, and the cache key cannot see that
  anchors have since appeared.

Note that every op routed to `bitexact` is also covered by the KC-silicon
overlay, which outranks `SIM-BIT-EXACT-16`. A validated sim leg therefore
never changes a `provability_class`; it appears as independent corroboration
in the ledger's `equal_evidence` / `engine_verdict` columns. That is the
intended design (device evidence supersedes simulator evidence), not a
discarded result.

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

A missing or mismatched required instrument aborts with exit 3. `--no-gate` is
retained as a compatibility spelling but no longer bypasses provenance. Current
compiler experiments use `formal_equiv_row.sh` and remain labeled
`CURRENT-CANDIDATE-NOT-PIN59`; they do not inherit pin-59 overlays. The ON flag
set is imported from the canonical `sweep_2x2.ON_FLAGS` (pin-59 ON-39).

## Re-run / resume / budget

* **Resume-safe:** a valid `verdicts/<op>/prove_all_verdict.json` is reused;
  `--force` re-proves from scratch.  `silicon_stream` rows are the exception:
  this driver's cache key cannot see the shard geometry (`NPAR`/`BAND_BITS`/
  `SPACE`), so instead of widening it they always re-enter `galaxy_shard.sh`,
  which resumes per band under the streamers' own provenance record
  (`stream_resume.py`).  Re-running costs only what it has not already streamed.
* **Per-op budget:** `--timeout` (default 1800 s, like laneJO); on expiry a
  formal row records `UNDECIDED-Z3-TIMEOUT` and never hangs.
* **Subsets:** `--only '<glob>[,<glob>...]'`, `--engine {formal_equiv,bitexact,classify,silicon_stream}`.
* **Silicon:** `silicon_stream` needs a staged galaxy node —
  `--silicon-farm-root` (tree with `tests/` + `build/tt-llk-build`),
  `--silicon-venv`, and `--silicon-idmap` (from `build_identity_gate.sh`).
  `NPAR`/`BAND_BITS`/`STAGGER`/`GOLDEN` pass through to `galaxy_shard.sh`.
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
