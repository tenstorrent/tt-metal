# Current-tuple SFPU formal validation

`formal_campaign.py` proves or refutes the exact compiler configuration being
considered now. It does not read the historical performance board, pin-59
overlays, or recorded formal verdicts.

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
  --out /tmp/formal-results
```

`search.json` is the full output of the LLK knob search, not the abbreviated
`selected.tsv`: the formal campaign consumes the exact selected flag string for
each operation. Use `--ops 'abs,mulint32-*'` for a subset.

For a uniform compiler profile instead of a search result:

```sh
python3 formal_campaign.py \
  --sim /tmp/formal-sim/libttsim.so \
  --flags '-mtt-tensix-optimize-example' \
  --ops 'sqrt-fresh' \
  --out /tmp/formal-sqrt
```

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

The runner emits `formal-results.json`, `formal-results.tsv`, both traces, both
pytest logs, ELF text identities, and the native `formal_equiv.py` verdict.
Statuses have deliberately narrow meanings:

- `PROVEN_EQUIVALENT`: Z3 found no different output over the modelled inputs.
- `PROVEN_EQUIVALENT_ON_DOMAIN`: the same claim under the supplied domain.
- `DIVERGENT`: Z3 produced a witness. This is a result, not a runner failure;
  numeric transformations must route to ULP admission.
- `UNSUPPORTED`: the operation is outside the formal model, for example a
  cross-lane instruction, and must route to another validation engine.
- `TIMEOUT`: the query was not decided within the budget.
- `TRACE_VALIDATION_FAILED`: the independent concrete executor did not replay
  the simulator snapshots, so no proof was admitted.
- `NO_REFERENCE`: the corpus has no distinct handwritten leg.

The top-level fields answer two different questions. `status: COMPLETE` means
the requested rows ran without an operational failure; it does **not** mean
that every row was proved. `formal_admission: ALL_PROVEN` means every requested
row is either unrestricted-equivalent or equivalent on its declared domain.
Otherwise `formal_admission` is `FOLLOWUP_REQUIRED`, with exact
`formally_admitted` and `followup_required` counts. For a domain-admitted row,
the JSON retains the unrestricted `DIVERGENT` verdict and witness alongside
the separate domain verdict.

The compatibility wrapper `formal_equiv_row.sh` invokes this same runner for one
operation. There is no second capture or provenance implementation.

## Historical coverage

`prove_all.py`, its manifest, and its overlays reproduce the old pin-59 coverage
census. They are not an admission path for current compiler or knob selections.
