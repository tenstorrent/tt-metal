# From a frozen eval candidate to a validated C++ port

The flow is general; the C++ translation itself is authored and reviewed using
[MAPPING.md](MAPPING.md), not guessed by a retrieval script. Run IDs are inputs
to the exporter, never baked into migration code or tests.

1. Export exact candidate source and evidence with `tools.generic_op_to_factory.export_run`.
2. Reproduce its recorded-runtime baseline using `tools.generic_op_to_factory.migration_workflow`.
3. Select the target revision and its compatible existing toolchain. Keep the
   recorded evaluator revision. `tools.generic_op_to_factory.prepare_target` verifies the frozen
   preparation, inventories changed target references, and installs only the
   absent source operation. It never replaces canonical helpers with old copies.
4. Author a named C++ operation: native validation/output allocation, reflected
   structural attributes, tensor arguments, program factory, explicit cache-hit
   updates, binding and source registration. A descriptor factory with typed
   address bindings is also suitable where its adapter can retain the required
   cache state. Fetch hardware geometry/format sizes from canonical APIs; obtain
   resource IDs from allocation and share named argument schemas with kernels.
   Inventory kernels/helpers and preserve the algorithm; record formatter-only
   changes separately from independently justified ABI/API adaptations. No Python planner dispatch or
   fallback to `generic_op` is permitted in the native path.
5. Supply a native entry-point mapping and operation-specific cache regression
   tests to `tools.generic_op_to_factory.validate_port`. That driver checkpoints build → source smoke
   → target-source golden suite → DB comparison → native smoke → native golden
   suite → source/native comparison → cache tests → independent code-quality /
   host-performance review → scoped completion evidence. Use the independent
   agent task and receipt contract in [REVIEW.md](REVIEW.md). Confirmed findings
   must be fixed and revalidated before completion; changed source needs a new
   validation workspace and a fresh review bound to that plan.

All migration tools live under `tools/generic_op_to_factory/` in tt-metal.
Commands below run from this repository root; runtime build/tests run in the
chosen isolated tt-metal checkout. The evaluator is a pinned historical input,
not a second repository that needs tool changes. The target checkout must already have its environment, submodules and
compatible build configuration. The driver does not install a host toolchain.

```bash
python3 -m tools.generic_op_to_factory.prepare_target --preparation /absolute/prepared \
  --runtime /absolute/target --target-revision FULL_TARGET_COMMIT
# After reviewing the reference differences, add --output /absolute/new-evidence
# to install the absent operation and save its input inventory.

python3 -m tools.generic_op_to_factory.validate_port plan --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port init --config /absolute/port.json
python3 -m tools.generic_op_to_factory.validate_port run --workspace /absolute/new-validation --through cache
# Independent agent reviews the final source and evidence; reconcile findings,
# then write /absolute/new-validation/review.json as described in REVIEW.md.
python3 -m tools.generic_op_to_factory.validate_port run --workspace /absolute/new-validation
python3 -m tools.generic_op_to_factory.validate_port status --workspace /absolute/new-validation
```

Example config (replace placeholders, including the complete commit):

```json
{
  "runtime": "/absolute/target",
  "target_revision": "FULL_TARGET_COMMIT",
  "preparation": "/absolute/prepared",
  "export": "/absolute/frozen-export",
  "phase": "initial",
  "workspace": "/absolute/new-validation",
  "source_entry": "ttnn.operations.sample_op:sample_op",
  "native_entry": "ttnn:sample_op_native",
  "smoke_nodeid": "eval/golden_tests/sample_suite/test_golden.py::test_case[case]",
  "cache_test": "tests/ttnn/unit_tests/operations/test_sample_native.py",
  "build_argv": ["./build_metal.sh", "--enable-ccache"],
  "precompile": true,
  "precompile_workers": 6,
  "allow_recorded_failures": false,
  "environment": {"CMAKE_BUILD_PARALLEL_LEVEL": "6"}
}
```

The source module must be the frozen operation package. The adapter replaces
its exported function before golden-suite collection, retaining the recorded
registry metadata, golden inputs and tolerances. Native mode rejects calls to
`ttnn.generic_op`; the adapter is test scaffolding, not a production dispatcher.
The selected smoke case must pass. Cache tests must all pass, with no skips.
Their assertions must cover the port's actual address/scalar/optional/alias
transitions; a green generic test count is not a substitute for that review.

The validation workspace must be outside the target repo and frozen inputs.
It records tracked diffs, untracked non-ignored source hashes, input/reference
hashes, commands, logs, JUnit and build libraries. Source/configuration drift
blocks resume. `--through STAGE` stops at a checkpoint. Failed/interrupted
stages require inspection followed by explicit `--retry`; old evidence is
retained, and potentially live unfinished commands block retry. Changes to a
port require a new validation workspace, not edited receipts. A completed
comparison is not a request to rerun a completed device stage.

Completion requires a recorded independent review and is scoped to the recorded
golden outcomes/tolerances and supplied cache tests. It does not claim universal support, tracing compatibility,
performance improvement or production readiness. A failing historical baseline
requires explicit acceptance and remains reported as failing after migration.
