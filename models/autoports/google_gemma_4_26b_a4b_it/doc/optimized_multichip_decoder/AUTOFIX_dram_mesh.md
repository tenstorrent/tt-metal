# AutoFix: DRAM-sharded matmul mesh reader placement

## Starting evidence

- Source-only diagnosis: `AUTODEBUG_dram_mesh.md`, produced by a fresh xhigh AutoDebug collaboration agent before implementation edits. The prescribed `.agents/scripts/autodebug.sh` CLI was attempted first, but its nested shell could not run because the sandbox wrapper `bwrap` was missing. Its incomplete investigation was stopped; `autodebug_dram_mesh_runner.log` records that blocker.
- Original failure: `dram_qkv_r2_sliding.command.json` / `.log`, real Gemma4 sliding layer 0, TP4 P300, 4096-token prefill and 128 advancing traced decode steps, QKV DRAM layout, two readers per bank, four storage cores. Host descriptor creation reaches `get_worker_noc_hop_distance()` with the full four-device mesh and fails its unit-mesh guard.
- Passing contrast: `dram_qkv_r1_sliding.json`, same reader/storage candidate family with one reader; output and cache checks pass.
- Starting revision: `adcb0e8f21704ff7e0090812a2f7b8a7d359f979`. The model and runner already had parent-owned Stage05 edits; this investigation did not edit them.

## Hypothesis experiment

**Hypothesis:** Secondary-reader placement incorrectly passes a multi-device mesh to a physical-device hop API. The primary-reader query already uses the mesh's first local device, while the one-reader path returns before any hop query.

**Focused command:**

```bash
python3 /tmp/tt_dram_mesh_probe/driver.py
```

The driver extracts the actual baseline placement function and actual public hop API body, compiles them with `/usr/bin/clang++-20 -std=c++20 -Wall -Wextra -Werror`, and supplies minimal host-only raw/unit-mesh/four-device-mesh fixtures. It uses no accelerator or TTNN runtime. `dram_mesh_host_before.log` confirms:

- One reader on mesh4 matches the raw-device assignment.
- Two and three readers on mesh4 reproduce the exact public API unit-mesh error.
- Two and three readers on a raw device and unit mesh agree.

**Verdict:** Verified caller-contract bug. This experiment proves the host failure boundary, not silicon execution or numerical correctness.

## Fix

The production change is confined to `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp`. In the multi-reader branch, secondary hop scoring uses the mesh's first local physical device, matching its existing primary-bank reader assignment. The change adds the explicit mesh-device include and formats one pre-existing extra blank line. A focused device regression is added beside the existing reader-count tests in `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py`.

The implementation is narrower than the report's suggested whole-helper normalization: it preserves the original primary-reader query, the one-reader early return, and the mesh's common compute-grid validation. Only the hop-distance argument changes. The public hop API guard, NOC0 restriction, placement exclusions, tie order and bank/worker mapping remain intact. No kernels, mesh descriptor adapter or model Python files change.

**Post-fix command:**

```bash
python3 /tmp/tt_dram_mesh_probe/driver.py patched
```

`dram_mesh_host_after.log` passes raw/unit/mesh4 comparisons for reader counts 1, 2 and 3; reader uniqueness; storage-core exclusions; deterministic repeats; direct multi-device public-hop rejection; and multi-reader NOC1 rejection.

The diagnostic driver, generated C++ fixture and build/install helpers are archived as `dram_mesh_host_probe.tar.gz`. Extract into `/tmp` to reproduce the listed commands. The baseline extraction is pinned to the starting revision.

## Build and formatting

The required build command was attempted:

```bash
.github/scripts/copilot-build.sh
```

It cannot run here: Docker is unavailable (`dram_mesh_wrapper_build.log`). No compilers or dependencies were installed.

The existing Release build supplies Clang 20, generated headers, dependencies and Ninja compile/link commands. The following fallback compiled the **complete changed matmul unity translation unit**, then linked the complete `_ttnncpp.so` with the changed object and all other existing objects:

```bash
python3 /tmp/tt_dram_mesh_probe/build_object.py
python3 /tmp/tt_dram_mesh_probe/link_library.py
clang-format --dry-run --Werror ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp
git diff --check
```

All passed. Exact compiler/linker argv and cwd are in `dram_mesh_object_build.command.json` and `dram_mesh_link.command.json`; successful exit codes are in the corresponding `.log` files. The fallback changes only output/dependency paths and substitutes the newly compiled object at link time. It does not reconfigure CMake or rerun unrelated compilation.

Artifacts before parent installation:

- `/tmp/tt_dram_mesh_probe/_ttnncpp.so`, SHA256 `3c6e55d5335dcfe106bc5ec4ece7422e6508cfd01983d6b3b1f028ef8705eba5`.
- `/tmp/tt_dram_mesh_probe/matmul_unity_0.o`, SHA256 `39518e56e5ac0d84bd7e82a5fb17757373b3e00b2d3cfdbfb6e27d21bf3fa6b0`.
- Patched utility source SHA256 `c97504644b95b8ade64522597fb9f7c4405dcc6211fc53da5a10adcdadd805e1`.

The AutoFix agent did not change loaded libraries or run hardware. After its device job closed, the parent reported backing up `build/lib/_ttnncpp.so` to `build/lib/_ttnncpp.so.stage05baseline`, installing the candidate, and starting `dram_qkv_r2_fixed_sliding` under its serialized hardware ownership.

## Parent hardware verification

After closing the preceding device run, the parent installed the candidate in `build/lib/_ttnncpp.so` only, retaining `build/lib/_ttnncpp.so.stage05baseline` as the backup. The source build-tree copy `build/ttnn/_ttnncpp.so` remains unchanged and is not used by the parent runtime.

The original failing configuration was rerun with real weights. Artifacts: `dram_qkv_r2_fixed_sliding.json` and `.log`. The parent reported exit 0; independent artifact inspection confirms `passed: true`, all replicas equal, 128 advancing traced decode positions, minimum output PCC **0.997589630569692**, and minimum cache PCC **0.9999966556897151**. The log reaches normal device closure.

The exact runner argv saved in that result is:

```json
[
  "/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/tests/run_multichip_decoder.py",
  "--layer",
  "0",
  "--length",
  "4096",
  "--steps",
  "128",
  "--trace",
  "--check-cache",
  "--attention-dram",
  "qkv",
  "--dram-readers",
  "2",
  "--dram-storage-cores",
  "4",
  "--output",
  "models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/dram_qkv_r2_fixed_sliding.json"
]
```

Observed TP4 host traced-decode median: **661.8984043598175 us**. The earlier one-reader control recorded **655.0061516463757 us**. This repair removes the failure; it does not demonstrate a performance improvement. The parent continues the remaining reader-count and projection-role matrix independently.

## Durable device regression

The parent executed the new regression with the patched runtime on the four local Blackhole devices. `dram_mesh_regression.log` records **3 passed in 2.75s**, covering one, two and three readers per bank on an explicit `(1, 4)` mesh, followed by normal device closure.

```bash
python_env/bin/python -m pytest -q tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py::test_matmul_in1_dram_sharded_worker_counts_mesh
```

The test uses replicated BF16 inputs and BFP8 weights, HiFi2 compute, a `32x512 @ 512x1536` matmul, four L1 storage cores and eight DRAM banks. Six width tiles per bank divide evenly among all three reader counts. It checks each of the four output replicas against the PyTorch reference with PCC at least 0.999 and relative Frobenius error at most 0.02. The existing mesh fixture handles insufficient-device skips, and the test explicitly restricts architecture and bank count. It needs no fabric collective, new fixture or dependency.

Black, AST parsing, static shard geometry checks and `git diff --check` passed before hardware execution. Final scope inspection found only the placement fix, its explicit include, the formatter's blank-line cleanup and this standalone regression; no additional implementation edits were needed. The standalone mesh case is necessary because the existing single-device helper's output conversion cannot validate all replicas.

## Final status

**Fixed and verified for the original two-reader real-layer failure.** Source cause and host boundary are verified; the complete changed matmul translation unit compiled and shared library linked; the parent-run original real4096/128 sliding-layer configuration passes output, cache and replica checks after the patch. The durable mesh device regression also passes all three reader counts on hardware.

Remaining limits: the helper preserves existing first-local-device placement, so it does not provide per-coordinate optimal placement on heterogeneously harvested meshes. Hardware proof includes the original reader2 QKV model configuration and the small reader1/2/3 mesh regression. The remaining model candidate matrix, any new retained-candidate Watcher validation and overall Stage05 acceptance belong to the parent's serialized validation. No performance improvement or broader stage acceptance is claimed.
