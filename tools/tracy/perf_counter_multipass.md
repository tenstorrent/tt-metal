# Multipass counter completeness contract

`merge_perf_counter_device_logs(pass_csvs, out_csv, pass_bitfields, execution_manifests)`
raises `ValueError` when completeness cannot be established. It validates every
scheduled pass before atomically replacing `out_csv`. Raw pass paths cannot be
used as the output. `run_perf_counter_passes` returns `False` on failed capture or
validation; Tracy's existing CLI dispatch exits **4** before report generation.

## Independent execution evidence

For each pass the existing workload adapter must write JSON to the absolute path
in `TT_METAL_PROFILER_EXECUTION_MANIFEST`. The runner supplies this path separately
for each launch. Write it only after the final drain and bind it to that pass's
exact raw `profile_log_device.csv` bytes:

```json
{
  "schema_version": 1,
  "arch": "blackhole",
  "bitfield": 1,
  "device_log_sha256": "<64 lowercase hex characters>",
  "executed_readouts": [
    [0, 1, 1, "BRISC", 7, null, null],
    [1, 2, 1, "BRISC", 8, 3, 0],
    [1, 2, 1, "BRISC", 8, 3, 1]
  ]
}
```

Each tuple is `[physical_chip, core_x, core_y, risc, run_host_id, trace_id, replay]`.
Core coordinates and runtime IDs must match the device CSV coordinate/ID space.
Eager executions use `null, null`; traced replay zero is distinct from eager.
The manifest must enumerate **every executed BRISC readout** in the requested
scope, including every chip/core and each executed replay. Derive it from the
validated dispatch schedule, program/core mapping and actual execution/replay
records, independently of surviving counter rows. A captured but never replayed
program is not an execution; any uncertainty in that distinction must reject the
manifest. Repeated identical tuples are ambiguous and rejected. Do not invent
identities, infer completeness from a nonempty file, or reconstruct expectations
by copying the observed counter keys. Ordinary timing markers alone cannot prove
that an entire chip or invocation was not lost.

The caller owns verification of that external execution evidence. The merger
checks its structure, raw-log hash, architecture, scheduled mask, unique readout
identities and equality of execution scopes across passes. The SHA binds evidence
to a log; it does not independently prove that the evidence producer is correct.
The caller must also attest that the native build uses the source counter tables
read by this module. Currently supported architectures are `blackhole` and
`wormhole_b0`; unsupported/missing native tables fail closed.

`perf_counters.hpp` emits each enabled group's native `hw_counters.h` array once
per BRISC readout. Other RISCs do not emit these counters. The merger requires
exactly that set of named records per identity, including zero-valued counters;
retired enum slots are not expected. It rejects wrong RISCs, unknown/unrequested
counter types, malformed values, duplicates, extra identities and any missing
identity/counter. Groups are disjoint and can have different record counts:
Blackhole FPU has 3 records per readout while PACK has 5. Equal raw row counts are
neither required nor sufficient. All passes must match the execution scope before
later counter timestamps can be anchored to the corresponding pass-0 readout.

## Result and preservation

The runner creates `.logs/perf_counter_passes/` exclusively and retains
`pass_<index>.csv`, `pass_<index>.execution.json`, and `merge_result.json`.
A workload exception or missing manifest also leaves any produced raw log intact.
On validation failure it removes the canonical last-pass device log so it cannot
be mistaken for a complete merged capture. The result contains `schema_version: 1`,
`complete`, scheduled `pass_bitfields`, and absolute `raw_pass_logs`. Failure adds
`error`; success adds `merged_log` with absolute `path` and `sha256`.

A preexisting pass directory is refused without changing earlier artifacts.
Use a fresh output directory per attempt. Always check the current call's return
value/exit status; an older `merge_result.json` is not a result for a refused run.
Direct merge callers must keep raw inputs immutable during validation/publication.

## Integration with profiling acceptance adapters (tasks19/20)

This module supplies a completeness check, not a runner, broker adapter or global
profiling acceptance gate. The existing capture adapter must provide the execution
manifest and retain all pass artifacts. Until it does, multipass capture fails
closed; no permissive fallback infers execution from counters.

Consumers must require a successful current invocation and a hash-bound complete
merge **in addition to** the existing attestation, expected-chip, final-drain and
loss-accounting gates. Unknown coverage or positive loss still invalidates the
affected metrics even when every requested counter row is present. Numerical
quality, device capacity, synchronized timing and measured power remain separate
acceptance gates. CPU synthetic captures validate this contract, not the native
execution-evidence producer or device performance.
