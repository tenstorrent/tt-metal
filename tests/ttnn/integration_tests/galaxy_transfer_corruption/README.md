# Galaxy transfer-corruption reproducer

Filter model-tier unit CI with `model=galaxy-transfer-corruption`, tier 3, and
`sku=wh_galaxy_perf`. In **All Model Tests**, select only unit tests and tier 3,
with `skus=wh_galaxy_perf` and the same model filter. This selects one test job.

Each job runs the handoff's stock `fast_reduce_nc` reproducer on UMD physical
device 11 for up to 100,000 iterations, stopping after preserving the first
mismatch. Fabric and Watcher are disabled; no model weights are needed. The
comparison is bitwise against the fixed healthy reference, with no tolerance
or retry-until-pass. A run covers this device selector on its assigned host,
not every device in that Galaxy. Concurrent jobs may reuse the same host as
capacity becomes available; use the recorded runner names to count distinct hosts.

From a built checkout:

```bash
TT_METAL_HOME="$PWD" bash tests/scripts/run_galaxy_transfer_corruption.sh
# CPU-only input verification and command inspection (no Torch/TTNN import):
TT_METAL_HOME="$PWD" bash tests/scripts/run_galaxy_transfer_corruption.sh --dry-run
```

Use a fresh checkout/output directory for each invocation. Results go under
`generated/test_reports/galaxy-transfer-corruption/<run-id>-<attempt>/`:
`runner.json` records host/run identity, and `capture/` contains the original
`report.json`, `launcher.json`, and any `fault-*.pt.gz` captures. Existing model
CI artifact uploads retain these on success or failure. Exit 0 means the bounded
run was clean, 1 means a qualified numerical mismatch, and 2 means setup/runtime
failure. A clean run is not a hardware health certificate.

The portable `repro/` directory comes from the
`device11-transfer-corruption-2026-09-20` handoff. Its source files and fixture
are unchanged; `repro/SHA256SUMS.json` verifies the original handoff hashes.
The fixture is unchanged (5,119,861 bytes; activation/reference tensors, not
model weights), SHA-256:
`8c644b2ee1e87d8b46a9d3b866f4dcbd497ff822de36544c062991fb7e7dbe34`.
The v86 script SHA-256 is
`53186e995f1311f1403dfab523f0b68970a891f43c72d31cd133eda0dc09eace`.

The reference was qualified on the handoff's edited checkout at
`609714c3ce2769165ee94334bf652277725dbb00`. This CI branch exercises its own
build. A consistent mismatch on a new implementation needs separate reference
qualification; do not automatically regenerate the reference on the device
under test. An intermittent mismatch alone does not identify faulty hardware.

For parallel submissions, build this branch once with the **Build tt-metal
artifacts** workflow and `build-inplace-wheel=true`. Pass its successful run ID
as `use-artifacts-from-run` to each filtered unit dispatch. Use artifacts from
the exact same commit and build configuration. This reuses the existing artifact
download path and avoids the shared same-commit compilation mutex serializing
the ten builds before their tests can reach different machines.
