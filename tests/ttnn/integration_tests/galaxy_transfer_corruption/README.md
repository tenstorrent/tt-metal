# Galaxy transfer-corruption reproducer

Filter model-tier unit CI with `model=galaxy-transfer-corruption`, tier 3, and
`sku=wh_galaxy_perf`. In **All Model Tests**, select only unit tests and tier 3,
with `skus=wh_galaxy_perf` and the same model filter. This selects one test job.

Each job sweeps **all 32 UMD physical devices (0-31)** sequentially, running the
handoff's stock `fast_reduce_nc` reproducer in a separate process for each device.
Each device has a 100,000-iteration budget and stops after preserving its first
mismatch; the sweep continues to the remaining devices after numerical or setup
failures. A clean job therefore completes **3,200,000 iterations**. Fabric and
Watcher are disabled; no model weights are needed. The comparison is bitwise
against the fixed healthy reference, with no tolerance or retry-until-pass.
Concurrent CI jobs may reuse the same host as capacity becomes available; use
the recorded runner names to count distinct hosts. The 60-minute test timeout
allows for 32 device startups and iteration budgets; an interrupted sweep retains
its progress and must not be treated as a complete clean result.

From a built checkout:

```bash
TT_METAL_HOME="$PWD" bash tests/scripts/run_galaxy_transfer_corruption.sh
# CPU-only input verification and command inspection (no Torch/TTNN import):
TT_METAL_HOME="$PWD" bash tests/scripts/run_galaxy_transfer_corruption.sh --dry-run
```

Use a fresh checkout/output directory for each invocation. Results go under
`generated/test_reports/galaxy-transfer-corruption/<run-id>-<attempt>/`:
`runner.json` records host/run identity, and `device-00/` through `device-31/`
contain the original `report.json`, `launcher.json`, and any `fault-*.pt.gz`
captures. `sweep.json` is checkpointed before and after each device and lists
clean, corrupt, and setup/runtime-failing devices, with completed iteration counts.
Existing model CI artifact uploads retain these on success or failure. Exit 0
requires qualified clean results for all 32 devices, 1 means at least one qualified
numerical mismatch, and 2 means at least one setup/runtime or unqualified result
(even if another device also had a numerical mismatch). Dry runs are explicitly
marked and never counted as clean hardware results. A clean run is not a hardware
health certificate.

The sweep orchestration can be tested without hardware:

```bash
python3 -m unittest discover \
  -s tests/ttnn/integration_tests/galaxy_transfer_corruption -p test_run_sweep.py
```

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
