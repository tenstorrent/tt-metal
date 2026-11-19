# Opt-in sampling capacity 64: isolated repair handoff

Status: host implementation/build checks pass; hardware correctness and P2 remain unverified. No device, server, model/plugin runtime, current checkout or original build output was modified. No commit/push. Parent owns device serialization and integration decision.

Checkout: `/home/ttuser/ornith-9b/tt-metal-sampling64`, detached at `d1c644f3`. Artifact directory: `sampling64_validation/`. Work began 2026-11-13 20:04:14 UTC; final actual elapsed is recorded in `elapsed.json` (about 17 minutes). No delay was used to claim work time.

## Diagnosis and repair

The original sampler sorts and consumes one 32-candidate tile. Merely widening the declared k limit would keep the wrong merge schedule and writer layout. Add trailing/defaulted `max_top_k=32` to the public/prim operation and keyword-only Python binding; reject capacities other than 32/64 and input widths below capacity. Capacity participates in program-cache specialization. For explicit64, reuse the canonical top-k two-tile sorting/merge helpers (K64/logk6), normalize/mask two tiles, and index both tiles with the actual tiled face layout. Existing circular-buffer capacities accommodate two candidate tiles; no allocation enlargement is needed. Preserve the old32 compute/writer branch, PRNG threshold/seed logic and independent greedy path.

No model capability has been widened. Common sampler/model formatting still needs separately approved plumbing after device validation. The repair is an opt-in primitive, not P2 evidence or a waiver.

## Completed checks

- Four modified standalone host translation units compiled with existing build flags and no PCH reuse/write (`host_build.json`, `host_build.log`).
- Three replacement unity objects and two isolated shared libraries compiled/linked sequentially in 93.917 seconds, linker/ThinLTO jobs=1; unchanged main objects were read-only (`isolated_build.json`, `isolated_build.log`). Full commands and per-command elapsed are preserved in JSON. All outputs are isolated.
- Original and isolated import-only provenance checks pass. Patched `_ttnn.so` and `_ttnncpp.so` both resolve under the isolated artifact `lib/`; patched binding advertises `max_top_k`. No device opened (`import.log`, reusable `run_checked.py`).
- Actual Blackhole firmware-wrapper compilation to assembly passed 12 variants: original32/patched32/patched64 times unpack/math/pack/writer (`kernel_compile.json`). This uses installed SFPI and existing generated headers copied into isolated fixtures; fixture bank/accessor constants are compile inputs, not device runtime evidence. Original32 and patched32 assembly are byte-identical after removing only `.file` directives for all four processors (`default32_assembly.json`, empty `default32_*.diff`). Other architectures and actual JIT/program runtime remain untested.
- One standalone host unittest passed with ASan/UBSan: 2048 tiled-address checks, k boundaries including 33/40/49/64, threshold midpoints, strict nucleus cutoff and BF16 p (`host_tests.log`). Both falsifying mutants (omit second-tile address and truncate to32) were detected (`host_mutants.json`).
- C++ formatting, Python formatting and final `git diff --check` completed. No pytest collection was used on host because repository fixtures may inspect devices.

## Parent-only device commands (after current measurements finish)

The following executable script fixes the environment and verifies loaded Python/native paths **inside the same process before pytest**. It clears inherited LD_PRELOAD. Original uses only original Python/source/lib paths. Isolated uses its own TT_METAL_HOME/PYTHONPATH/lib and `TT_METAL_CACHE=sampling64_validation/device_cache`. Installed compiler/firmware assets are reused through the isolated `runtime` symlink to main runtime; these assets are read-only inputs. This avoids duplicating the large caches. Do not rebuild through that symlink.

```bash
V=/home/ttuser/ornith-9b/tt-metal-sampling64/sampling64_validation
# Untouched main build: 1 case with 27 fixtures / 81 seeded draws. Refuses overwrite.
"$V/run_device_tests.sh" original-record
# Gate: identical fixture source and exact outputs against untouched artifact.
"$V/run_device_tests.sh" isolated-compare
# 22 remaining cases: explicit64 and default32/cache/trace/validation coverage.
"$V/run_device_tests.sh" isolated-suite
```

Optional pytest options can be appended unchanged to each command. Preserve the test file after recording: its full SHA256 is embedded in the baseline and recorded in `fixture.sha256`. Baseline includes users1/17/32, widths64/128/256 and ordered/equal/signed-zero values with mixed per-user k. New64 tests cover k40/64, p1/.95, second-tile membership/distribution, row identity, stable ties, mixed-k buffer reload, explicit/implicit32 with cache switches and trace replay. Total 23 collected cases (one baseline case plus22 suite cases). Budget an estimated 5–10 minutes including cold JIT for all three commands; actual duration is unmeasured and no runtime guarantee is implied.

Artifacts produced by these future runs: `legacy32_original.json`, `device_original32.log`, `device_isolated-compare.log`, `device_isolated-suite.log`, private `device_cache/`. Stop on baseline difference or failing64 check; no support declaration until those pass. The baseline tests only sampling compatibility; full model/P2 routing and performance require separate parent validation.

## Patch files

Under `ttnn/cpp/ttnn/operations/reduction/sampling/`:

- `sampling.hpp`, `sampling.cpp`, `sampling_nanobind.cpp`
- `device/sampling_device_operation_types.hpp`, `device/sampling_device_operation.hpp`, `device/sampling_device_operation.cpp`, `device/sampling_program_factory.cpp`
- `device/kernels/compute/sampling.cpp`
- `device/kernels/dataflow/writer_interleaved.cpp`, new `sampling_64_utils.hpp`

Tests: `tests/ttnn/unit_tests/operations/reduce/test_sampling_64.py` and `test_sampling_64_host.py`.

`sampling64.patch` includes all 12 source/test files; local build/report/launcher artifacts are not part of the integration patch. Worktree remains uncommitted.

## Follow-up: isolated Python opt-in plumbing

Added only `models/common/sampling/generator.py`, `models/common/sampling/tt_sampling.py`, and `models/common/tests/test_sampling_capacity_host.py` to the isolated patch. Existing `args.max_top_k=64` now reaches the final primitive draw; default32 omits the new native keyword, preserving compatibility with older extensions. The formatter accepts keyword-only `max_top_k=32|64`; both common decode reload branches use the instantiated sampler's capacity. Invalid capacities fail before device use. Top-k construction, gathered offsets and all buffer sizing already use the same capacity. No new model capability is advertised.

Run the focused host checks directly (no TTNN imports or pytest/device collection):

```bash
/home/ttuser/ornith-9b/tt-metal/python_env/bin/python \
 /home/ttuser/ornith-9b/tt-metal-sampling64/models/common/tests/test_sampling_capacity_host.py
```

Six tests passed in 0.019 seconds with `PATH=/nonexistent` (`python_host_tests.log`): explicit all-field default32 expectations over27 combinations, requiring no Git executable or repository history; explicit64/k40, greedy sentinel and padding; invalid-capacity validation; actual constructor rejection before device access; single/multiple-chunk decode reload; actual final primitive call with unchanged32 kwargs versus explicit64. Methods/functions are extracted from production AST and executed with host mocks, so these checks do not validate device allocation, CCL, top-k extraction or trace behavior. No heavy build or device use was performed. `python_plumbing.patch` is separate from the frozen original12-file `sampling64.patch`. Built native library mtimes and the device fixture SHA remain unchanged (`python_preservation.log`).

### Exact Ornith integration points (read-only inspection of main checkout)

All paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`:

- `tt/generator.py:127`: instantiate `SamplingGenerator` with selected `args.max_top_k`; keep physical `max_batch_size=32`.
- `tt/generator.py:135`, `:229`, `:340`: constructor/default params, request validation formatting, and reset-state formatting must pass that selected capacity explicitly.
- `tt/generator.py:218` and `:232`: requested and formatted stochastic-k bounds currently stop at32; both must match selected capacity after device validation. Temperature0 still maps to k1.
- `tt/generator.py:549`: `_raw_parameters(sampling_params, 32)` also needs the selected candidate capacity; the second positional32 here is slot count, not top-k.
- `tt/sampling_state.py:73`/`:87`: `_raw_parameters` currently calls the formatter with default32; add a separate candidate-capacity argument and pass it through. Preserve its raw-temperature reconstruction.
- `tt/sampling_state.py:418`/`:419` and `:429`: both prior/update `_raw_parameters` calls and merged final upload must use `self.sampler.tt_sampling.max_top_k`. Otherwise scoped reload silently recaps k40 to32. Keep `CAPACITY=32` and all slot/seed buffers unchanged.
- `tt/generator_vllm.py:34`: `model_capabilities["max_device_top_k"]` remains32 until the integrated primitive/common/model device path has passed. No new plugin interface is needed for a truthful later capability change. Unlimited-k requests remain outside finite64 support.
- `doc/full_model/serving_contract.json:585`/`:587`: existing evidence explicitly records top_k1..32; retain that historical evidence until a separately identified new device result supersedes it.

Model, adapter, sampling-state helper, serving contract and main checkout are unchanged. New64 native tests alone do not validate TP4 candidate gathering or the complete common sampler path; those need a parent-owned explicit k40 common-sampler/model check before routing P2 to device.

Durability correction: the default32 production test now checks explicit expected values for every field across27 cases. It no longer calls Git or compares against HEAD. The original exact prechange comparison evidence is preserved separately as `python_host_tests_prechange.log`, with pinned baseline source `generator_prechange.py` and commit/SHA metadata in `python_prechange_baseline.json`. Native libraries, device fixtures and device commands remain unchanged.

## Pending model integration scratch

`ornith_integration/README.md` records the unapplied three-file Ornith patch, eight passing actual-function CPU checks, source hash preservation, and a prepared but unexecuted main-native rebuild/deployment script. The latter uses three main unity objects and two serialized links with worker limits1, then verifies main-only import provenance. Current ordinary Ninja dry-run wants CMake regeneration, so the exact recorded replay avoids an uncontrolled wider rebuild. No main source/build modification or device run was performed.
