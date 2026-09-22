# AutoFix: coordinate-aware native DRAM descriptors

## Starting evidence

See `AUTODEBUG_dram_mesh.md`. The original three-reader TP4 decode fails on
the factory's parent MeshDevice in the unit-mesh hop-distance assertion.
The existing descriptor adapter already has coordinate-specific dispatch.

## Hypothesis and experiment

Hypothesis: opting the DRAM factory into per-coordinate descriptor construction
and resolving that coordinate's physical device for multiple readers removes
the observed API incompatibility while preserving each chip's reader/NoC
metadata. One-reader device selection remains the original mesh path.

Source experiment: inspect the adapter's four-argument signature detector,
per-coordinate build path, binding cache, and cache-hit rebuild path. Verified
that the new signature is selected automatically, each tensor coordinate is
provided, and MeshTensor bindings remain tracked per program.

Implementation worktree:
`/home/mvasiljevic/qwen38-full-rerun/dram-mesh-fix`, detached from
`1623f9cb595dd1e87cb17abc6f3449838c1762ee`.

Patches:

- `dram_mesh_native.patch.gz`: three native files, 29 additions / 8 removals.
- `dram_mesh_tests.patch.gz`: 154 lines of focused regression coverage.
- `dram_mesh_complete.patch.gz`: both, for application to an unmodified checkout.

Do not apply the combined patch after the native-only patch has been applied.

The native change replaces the unused fourth CoreRangeSet argument with
an optional MeshCoordinate. Only multiple readers resolve a physical device;
missing coordinates on a multi-device direct call and remote multi-reader
coordinates fail explicitly. The old direct Python CoreRangeSet argument
remains accepted/ignored; a new trailing coordinate argument is optional.
Native reader-count defaults, reader placement logic, kernels, precision,
tensor ownership, and buffer bindings are unchanged.

## Checks performed

This repair agent ran the following checks in its isolated worktree:

```bash
/home/mvasiljevic/qwen38-full-rerun/tt-metal/python_env/bin/python -m black --check --target-version py312 tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py
/home/mvasiljevic/qwen38-full-rerun/tt-metal/python_env/bin/python -m py_compile tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py
git diff --check
```

All passed. This agent did not build shared libraries or use hardware.

The parent reports integrating the native patch and compiling target `ttnn`
successfully in the existing build tree. Its initial device retries still
showed the old `optional<CoreRangeSet>` factory symbol because Python loaded
installed library copies, not the newly linked build products. That was a
materialization mismatch, not a refutation of the repair.

The parent reports installing both components:

```bash
cmake --install build_Release --prefix $PWD/build_Release --component ttnn-runtime
cmake --install build_Release --prefix $PWD/build_Release --component tt_pybinds
```

It then confirmed the new `optional<MeshCoordinate>` symbol with `nm`.
These compile/install observations were supplied by the parent, not executed
by this repair agent. Preserve the installed binary hashes with the device
evidence so later results identify the actually loaded implementation.

## Focused regression command

```bash
python_env/bin/python -m pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -k test_matmul_dram_sharded_mesh_readers_cache -q
```

Six cases cover readers 1/2/3 on a 1x4 mesh and an offset 1x2 submesh. Each
case retains three sets of allocations, verifies different activation/weight/
output addresses, checks stable program-cache entry counts, and compares every
rank against its own reference. Multi-reader cases inspect the native
descriptor at every coordinate, validate its primary reader assignment and
secondary-reader minimum physical-hop invariant, and test the missing-coordinate
guard while exercising the backward-compatible positional binding.

This is device-required regression coverage and was not run by this agent.
On a homogeneously harvested device set it exercises coordinate handling but
cannot prove behavior when per-device harvesting differs. Trace/eager replay
and full-layer latency remain the parent's separate model controls.

## Status and remaining risks

Source mechanism verified; isolated patch delivered; parent reports native
compile and installed-symbol checks passed. Installed-native device checks
were starting when this handoff was written. No hardware pass or performance
improvement is claimed here.

Per-coordinate dispatch also produces singleton programs for one-reader
workloads, although their original device selection and descriptor contents
remain unchanged. The parent will measure a one-reader whole-layer control
for grouping overhead. Multihost multiple-reader operation remains explicitly
unsupported; the one-reader path retains its existing behavior.
