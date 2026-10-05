# Current-tuple SFPU formal validation

`formal_campaign.py` answers two independent questions about the exact compiler
configuration being considered now:

1. **Compiler gate:** selected flags versus the frozen compiler baseline on the
   same semantic source. This is an exact, unrestricted equivalence claim.
2. **Semantic gate:** the semantic source versus the handwritten source, both
   compiled with the frozen baseline. A documented source-level domain may
   constrain this comparison.

The deployment gate passes only when both pass. A numeric/ULP license may admit
a source-level semantic uplift elsewhere, but it can never excuse a compiler
gate divergence. The runner does not read the historical performance board,
pin-59 overlays, or recorded formal verdicts.

## Build the trace instrument

The trace hook is committed in `craq-sim` on `nkapre/sfpi`. Build and stage it
from that repository:

```sh
python3 scripts/build-formal-instrument.py --out /tmp/formal-sim --jobs 16
```

The staged directory contains `libttsim.so`, `soc_descriptor.yaml`, the exact
`tensix_isa.json` used to decode instructions, and `formal-instrument.json`. The
manifest records the source commit, build command, target, observed artifact
hashes, and trace schema. The formal runner validates the schema and records
these identities; it does not require an obsolete binary hash.

## Validate selected LLK knobs

Run from `tt_metal/tt-llk/tests/corpus/tools` after the LLK harness and toolchain
have been installed under `tests/sfpi`:

```sh
python3 formal_campaign.py \
  --sim /tmp/formal-sim/libttsim.so \
  --selection /path/to/search.json \
  --jobs 8 \
  --out /tmp/formal-results
```

`search.json` is the full output of the LLK knob search, not the abbreviated
`selected.tsv`: the formal campaign consumes both the exact selected flag
string and the frozen baseline (`settings.baseline_flags`, checked against each
proposal's `frozen_baseline_flags`). A JSON without that baseline is refused
unless `--baseline-flags` supplies it explicitly. A TSV must contain
`op`, `flags`, and `baseline_flags`, or be paired with `--baseline-flags`.
Use `--ops 'abs,mulint32-*'` for a subset. `--jobs` runs
independent per-operation capture/proof tasks concurrently; output records are
still written in canonical operation order.

For a uniform compiler profile instead of a search result:

```sh
python3 formal_campaign.py \
  --sim /tmp/formal-sim/libttsim.so \
  --flags '-mtt-tensix-optimize-example' \
  --baseline-flags '-mtt-tensix-optimize-frozen' \
  --ops 'sqrt-fresh' \
  --out /tmp/formal-sqrt
```

Omitting `--baseline-flags` in uniform mode makes it identical to `--flags` and
records the compiler gate as `NOT_APPLICABLE_IDENTICAL_CONFIGURATION`; it does
not manufacture a proof by compiling the same configuration twice.

Documented input restrictions are explicit current-run inputs:

```json
{
  "mulint32-fresh": [
    {"which": "all", "int_min": 1, "int_max": 40000}
  ]
}
```

The checked-in `formal_domains.json` supplies the reviewed contracts for
`clamp-fresh` and `mulint32-fresh` by default; pass `--domains domains.json` to
use another explicit contract set. The runner always proves the unrestricted
claim first. If it diverges, it separately runs the domain query and preserves
both verdicts. A recorded historical domain overlay never promotes a current
result.

## Verdicts

The runner emits `formal-results.json`, `formal-results.tsv`, up to three traces
and pytest logs (`selected-sem`, `baseline-sem`, `baseline-hand`), ELF text
identities, and separate native `formal_equiv.py` verdicts for the compiler and
semantic pairs. A row without a handwritten reference still runs the compiler
gate; only its semantic status is `NO_REFERENCE`.
Statuses have deliberately narrow meanings:

- `PROVEN_EQUIVALENT`: Z3 found no different output over the modelled inputs.
- `PROVEN_EQUIVALENT_ON_DOMAIN`: the same claim under the supplied domain.
- `DIVERGENT`: Z3 produced a witness. This is a result, not a runner failure.
  A compiler-gate divergence is wrong-code and cannot route to ULP admission;
  a semantic-gate divergence may route only through an explicit numeric license.
- `UNSUPPORTED`: the operation is outside the formal model, for example a
  cross-lane instruction, and must route to another validation engine.
- `TIMEOUT`: the query was not decided within the budget.
- `TRACE_VALIDATION_FAILED`: the independent concrete executor did not replay
  the simulator snapshots, so no proof was admitted.
- `NO_REFERENCE`: the corpus has no distinct handwritten leg.

Each result records `compiler_status`, `semantic_status`, and their independent
gate states, plus `deployment_gate`. The legacy top-level `status` and proof
fields continue to describe the semantic comparison for existing readers.

The campaign-level fields answer completion separately from admission. While the runner is
checkpointing, `status` is `RUNNING`, `selected` is the completed count, and
`expected_total` is the requested count. `status: COMPLETE` is emitted only
after every requested row ran without an operational failure; it does **not**
mean that every row was proved. `compiler_admission`, `semantic_admission`, and
`deployment_admission` report the three independent rollups. The legacy
`formal_admission` remains an alias for semantic admission. For a
domain-admitted semantic row, the JSON retains the unrestricted `DIVERGENT`
verdict and witness alongside the separate domain verdict; compiler proofs
never use that domain fallback.

The compatibility wrapper `formal_equiv_row.sh` invokes this same runner for one
operation. There is no second capture or provenance implementation.

## Historical coverage

`prove_all.py`, its manifest, and its overlays reproduce the old pin-59 coverage
census. They are not an admission path for current compiler or knob selections.
