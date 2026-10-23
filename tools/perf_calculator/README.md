# Reusable transport estimation tools

This directory contains the reusable part of the performance calculator:

- `routing.py`: directed NoC torus routes.
- `noc_routing_model.py`: read/write transport event model with link/NIU queues and shared bank byte service.
- `read_service.py`: independent per-NoC/bank request-admission characterization.
- `shortlist.py`: rank and deduplicate predicted configurations, retaining provenance.
- `core_sweep.py`: generate coarse core-count products and refine them using measured saturation, duration changes, or prediction errors.
- `topology_bh.json`, `model_parameters.json`: captured hardware topology and parameter provenance.
- `read_service_bh.json`, `read_probe_bh.jsonl`: measured read service profile and its source evidence.
- `write_probe_bh.jsonl`: single-core write measurements, including issue scopes.
- `write_validation_bh.json`: single-core and multicore validation results; balanced multicore writes remain substantially underestimated.
- `validate_transport.py`: write/mixed comparison CLI with separate issue calibration input.
- `WRITE_MODEL.md`: write semantics, assumptions, and validation limits.
- CPU test modules: routing/resource invariants, characterization, and candidate ranking.

The retained raw read probes contain no timed tilize compute or full output writes.
Their applicability is limited to the measured topology/clock and documented probe
conditions; rates are effective measurements, not universal ISA constants.

The operation-specific tilize simulator, compute/scatter profiles, kernels, and
hardware harnesses have been removed. **There is no complete generic operation
simulator yet.** The transport model is a building block, not an end-to-end estimate
for an arbitrary operation, and its original validation CLI consumes read-probe measurements.
For writes, use `validate_transport.py`; see [WRITE_MODEL.md](WRITE_MODEL.md) for
semantics, validation results, and limitations.

The candidate skill must generate each operation's workload and schedule model,
correctness tests, kernels, component characterizations where needed, and benchmark
artifacts in a separate operation/campaign directory. Do not regenerate those inside
the reusable calculator directory. Missing operation coverage means no supported
end-to-end prediction until that layer has been implemented and validated.

For ranking, call `make_shortlist(rows, count, provenance)` with rows of the form:

```python
{"configuration": {"operation": "copy", "shape": [1024], "cores": 4,
                   "dtype": "bfloat16", "placement": "explicit coordinates here",
                   "schedule": {"batch": 8}},
 "estimated_us": 2.5}
```

Include all semantic and executable settings in `configuration`, including input/
output placement and format. The helper neither generates nor executes kernels.
`file_hashes(paths)` can capture model/profile provenance. Freeze predictions before
hardware measurements and select the winner from comparable, correct implementations.

```sh
python_env/bin/python -m unittest discover -s tools/perf_calculator -p 'test_*.py'
python_env/bin/python tools/perf_calculator/noc_routing_model.py \
  --topology tools/perf_calculator/topology_bh.json \
  --measurements tools/perf_calculator/read_probe_bh.jsonl \
  --read-service-profile tools/perf_calculator/read_service_bh.json \
  --output /tmp/read_transport_validation.json
```

## Automatic core-count sweeps

`core_sweep.py` generates probe-compatible JSON cases and a separate decision/provenance
plan. It does not launch hardware jobs. Import `coarse_cases()` and `refine_cases()`
from a campaign script, or use the CLI below. Keep campaign outputs outside this directory.

The coarse stage selects powers of two, an upper-range 1.5× power-of-two point if
it fits, and the available-worker cap. On the captured 110-worker topology this gives
**1, 2, 4, 8, 16, 32, 64, 96, 110**. `--max-cores` optionally lowers the cap for
workload parallelism or a smaller experiment; it cannot increase it above the topology's
worker count. Core counts enter the Cartesian product with the requested traffic,
payload, batch, NoC and bank choices. Duplicate cases are removed. Incompatible
mixed/non-split and odd-batch/split combinations are skipped and recorded in the plan.

```sh
python_env/bin/python tools/perf_calculator/core_sweep.py coarse \
  --topology tools/perf_calculator/topology_bh.json \
  --traffic write --payload-bytes 128 1024 --batch 16 \
  --modes noc0 noc1 split --banks balanced hotspot \
  --output /tmp/core-sweep/coarse.json

# 108 cases: 9 core counts × 2 payloads × 3 NoC modes × 2 bank mappings.
# Use a fresh report path: the hardware probe APPENDS rows.
NOC_TRANSPORT_CASES=/tmp/core-sweep/coarse.json \
NOC_TRANSPORT_REPORT=/tmp/core-sweep/coarse-measured.jsonl \
  scripts/run_safe_pytest.sh --no-precompile \
  tests/ttnn/unit_tests/operations/examples/test_noc_transport.py

python_env/bin/python tools/perf_calculator/core_sweep.py refine \
  --topology tools/perf_calculator/topology_bh.json \
  --observations /tmp/core-sweep/coarse-measured.jsonl \
  --budget 24 --output /tmp/core-sweep/refined.json
```

By default the companion plan is `OUTPUT.plan.json` (replacing the output suffix);
`--plan` overrides its location. It records source hashes, selected intervals,
reasons, thresholds and the number of proposals deferred by the budget.

Refinement groups observations by traffic, payload, batch, NoC and bank mapping.
Within each group it considers adjacent measured core counts and proposes their integer
midpoint if any of the following applies:

- **Bandwidth saturation:** relative bandwidth gain / relative core-count gain ≤ 0.25.
  Bandwidth is proportional to `cores / measured_us` because work per core is fixed.
- **Duration change:** absolute relative duration change ≥ 0.20.
- **Prediction error:** either endpoint has absolute relative error ≥ 0.10, when
  an estimate is present. This trigger is unavailable with raw timing-only observations.

These are experiment-selection heuristics, not proven hardware saturation diagnoses.
Override them with `--saturation`, `--duration-jump`, and `--error-threshold` (fractions,
not percentages). Wider intervals are considered first within a family; the global
`--budget` is allocated round-robin across families. Repeated measurements of a core
count are summarized by their median. Already-measured counts are never proposed,
and refinement does not extrapolate beyond the measured range. A one-point family
cannot be refined; an empty output may also mean no threshold triggered or all gaps
are adjacent integer counts. Inspect the plan before running an empty sweep.

`--observations` accepts one or more raw probe JSONL files, lists from the read
validation CLI, or JSON objects from `validate_transport.py`. To use the error trigger,
first freeze predictions and produce a validation comparison with independent issue
calibration as described above. Pass **all previous rounds** into later refinement calls
so previously sampled counts are retained. The dimension flags (`--traffic`, etc.)
configure only the coarse stage; refinement inherits its scenario families from observations.
Raw measurements are checked for clock/bank-count compatibility when those fields exist;
compact comparison files lacking that metadata must come from the same topology campaign.

Scope: the current hardware probe fixes **64 requests/core, four iterations, and row-prefix
placement**. Sweep payloads are restricted to 64-byte multiples through 16384 bytes;
batches must divide 64 and be even for split mode. Fixed-total-work algorithm scaling,
arbitrary placement, and hotspots on banks other than bank 0 require extensions to the
probe and workload model. Mixed scenarios can be generated explicitly, but remain
experimental and do not express a dependent copy. Refinement improves coverage; it does
not correct the known balanced-multicore write prediction error.
