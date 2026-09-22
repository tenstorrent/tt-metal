# AutoDebug: native TP4 multi-reader DRAM matmul

Source-only AutoFix diagnosis, 2026-09-12. This agent read the current source,
prior reports, original failure log, and build graph. It ran no hardware,
build, or runtime commands and changed no implementation. Commands below are
proposed validation, not claimed results.

## Finding

The assertion is caused by using the parent MeshDevice as the device for a
DRAM-reader placement calculation that requires physical, device-specific
NoC geometry. The native descriptor adapter already supports per-coordinate
descriptors. Opt this factory into that path, resolve the physical device for
each multi-reader descriptor, and retain the existing kernels, MeshTensors,
buffer bindings, and mesh-owned execution. No change to the hop-distance API
or tensor ownership is needed.

The prior one-reader workaround is a valid control, not fulfillment of this
stage's multi-reader optimization requirement. No speedup is established by
this diagnosis.

## Evidence and mechanism

- Starting reports: `../multichip_decoder/AUTODEBUG_dram_mesh.md` and
  `../multichip_decoder/AUTOFIX_dram_mesh.md`.
- Original command:
  `bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh replicated_fixed_l0 --layer 0`.
  `../multichip_decoder/replicated_fixed_l0.log:23-25` shows prefill completed
  and the first eager decode failed. Lines 50-62 show the native call chain.
- `Tensor::device()` returns the owning MeshBuffer's MeshDevice, including
  for coordinate-restricted views (`ttnn/core/tensor/tensor.cpp:507`).
- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:937-979`
  exposes an unused fourth `optional<CoreRangeSet>` argument and selects
  `IDevice* device = a.device()`. Its helper receives that same pointer.
- The helper calls `get_dram_bank_reader_assignments` at line 124. That
  utility returns before hop-distance queries for one reader, but uses the
  supplied device for every secondary-reader candidate's physical cost for
  two or three readers (`matmul_utilities.cpp:404-463`).
- `tt_metal/impl/device/experimental/device.cpp:17-21` rejects meshes with
  more than one device. This occurs before device kernels launch.
- The same descriptor helper also uses its device argument for worker-grid
  size, multicast endpoints, and sender/output NoC coordinate arrays
  (factory lines 95-107, 675-679, 754-758). Resolving only the hop-distance
  call would leave those other device-specific queries on the mesh.
- The mesh's old primary-reader API explicitly documents that it returns
  only the first device's assignment and is incorrect for heterogeneous
  harvesting (`tt_metal/api/tt-metalium/mesh_device.hpp:121-137`). Equal
  logical grid dimensions do not prove equal physical NoC distances.

The earlier report's broad statement that native coordinate-aware descriptor
handling would be required remains true, but the required dispatch machinery
already exists in this checkout; a new dispatch mechanism is unnecessary.

## Exact isolated repair proposal

Change only the native, non-Quasar DRAM factory declaration/definition and
its existing descriptor binding. The Quasar factory is a different namespace
and is not in this failure's call path.

1. Replace the factory's unused fourth argument in its `.hpp` and `.cpp`
   with:

   ```cpp
   const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt
   ```

   The default belongs in the header only. Add the necessary mesh header
   explicitly if transitive includes do not provide the used API.

2. Move extraction of the existing DRAM `program_config` and reader count
   before selection of `device`. Preserve the one-reader device selection
   exactly; resolve the physical device only for multi-reader descriptors:

   ```cpp
   auto* mesh = a.device();
   tt::tt_metal::IDevice* device = mesh;
   if (program_config.num_workers_per_dram_bank > 1) {
       TT_FATAL(
           mesh_dispatch_coordinate.has_value() || mesh->num_devices() == 1,
           "Multi-reader DRAM matmul requires a mesh dispatch coordinate");
       const auto coord = mesh_dispatch_coordinate.value_or(
           ttnn::MeshCoordinate::zero_coordinate(mesh->shape().dims()));
       TT_FATAL(mesh->get_view().contains(coord), "Mesh coordinate {} is out of bounds", coord);
       TT_FATAL(
           mesh->get_view().is_local(coord),
           "Multi-reader DRAM matmul requires a local physical device at {}", coord);
       device = mesh->get_device(coord);
       TT_FATAL(device != nullptr, "No physical device at mesh coordinate {}", coord);
   }
   ```

   This is proposed code, not a compiled patch. `get_device(coord)` and
   `get_view().is_local(coord)` are currently deprecated public interfaces,
   but are present and used by existing native factories. Do not suppress
   or hide that limitation. The local-only multi-reader scope is deliberate:
   the old multi-reader path already fails on every non-unit mesh. One-reader
   execution keeps its original parent-mesh path, including its existing
   multihost behavior. Do not silently substitute a local reference device
   for a remote multi-reader coordinate.

   Pass this selected `device` through the existing helper. Keep tensor
   equality validation against `a.device()` and keep all `MeshTensor`
   references unchanged. Use the one selected physical device consistently
   for primary readers, secondary-reader costs, and NoC runtime arguments.

3. Adapt the direct Python binding in
   `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1308-1324`.
   Preserve its existing positional/keyword `core_range_set=None` argument
   as an ignored compatibility argument, because the old factory ignored it.
   Append `mesh_dispatch_coordinate=None` and forward that coordinate as the
   native factory's fourth argument. Unit-mesh direct calls without the new
   argument still work. Multi-reader direct calls on larger meshes must pass
   the coordinate. The ordinary `ttnn.linear`/`ttnn.matmul` path supplies it
   automatically through the adapter.

Do **not** add a second C++ `create_descriptor` overload to retain the old
argument type: `ProgramDescriptorFactoryConcept` currently tests
`&T::create_descriptor` (`ttnn/api/ttnn/operation_concepts.hpp:73`), which is
ambiguous for an overload set. Also, merely adding the coordinate as a fifth
argument does not opt into the adapter's four-argument signature test.

## Why this routes and caches correctly

`DescriptorMeshWorkloadAdapter::create_descriptor_uses_mesh_dispatch_coordinate`
checks for exactly the callable fourth `optional<MeshCoordinate>` argument
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:424-447`). `invoke_per_coord`
passes it at lines 517-530. On a cache miss, lines 607-614 build a separate
descriptor/program for each actual tensor coordinate when that signature is
present; otherwise they build once per tensor range, which is the current
behavior. The defaulted new signature still satisfies the existing factory
concept.

The DRAM descriptor already uses `emplace_runtime_args` with MeshTensor
references for weight/bias addresses (factory lines 909-916) and tensor-backed
CB descriptors. The adapter stores bindings per coordinate range and patches
those bindings on cache hits; a descriptor rebuild, when required, receives
that range's coordinate again (adapter lines 645-767). No host copy, new
unit mesh, buffer rebind, or trace-owner change is needed.

One-reader descriptor contents and device selection remain unchanged, but
opting into this adapter branch changes a uniform four-device workload from
one range program to four singleton programs, including for one reader.
That can increase compile/cache metadata and host construction cost. Measure
it; do not claim identical host overhead. If preserving one-reader range
grouping is mandatory, use a `create_workload_descriptor` entry point instead:
keep the current once-per-range construction for one reader, emit singleton
coordinate programs for multiple readers, and retain the old public
`create_descriptor` binding. That is a somewhat larger alternative and needs
its own binding/cache verification; it is not required to repair device
correctness for the target local TP4 mesh.

## Coordinates, heterogeneous devices, and multihost limits

- Resolve coordinates through the tensor's owning mesh, never through the
  root mesh or a hard-coded `(0, 0)`. A coordinate-restricted tensor on a
  parent mesh retains parent-relative coordinates. A tensor allocated on an
  offset submesh uses that submesh's local coordinates; `create_submesh`
  installs a correctly rebased view (`mesh_device.cpp:677-735`).
- Do not use `get_devices().at(coord.to_linear_index(shape))` to resolve
  coordinates. `get_devices()` contains only local devices, while a mesh
  coordinate indexes local plus remote slots. The existing coordinate-aware
  hop overload currently uses that unsafe indexing (`experimental/device.cpp:51-52`);
  it is unnecessary for this repair and should remain outside this patch.
- Per-coordinate selection handles different physical harvesting and primary
  reader placement when the shared tensor layouts are valid on every device.
  It does not establish support for differing logical worker-grid sizes,
  DRAM bank counts, architectures, allocator geometry, or shard capacities.
  Preserve the existing Blackhole/NOC0 and weight-width validation. Do not
  advertise arbitrary heterogeneous-mesh support from a homogeneous TP4 run.
- The adapter iterates global tensor coordinates, including remote coordinates
  in multihost configurations. Reject unsupported multi-reader remote queries
  explicitly. Skipping them with empty descriptors or substituting a local
  device requires a separate multihost workload/collective audit. The
  one-reader branch above avoids introducing a new physical-device lookup
  into that previously working path.

## Discriminating validation

### Host/build checks

The changed factory and nanobind binding are C++; both must compile. The
mandatory repository wrapper is:

```bash
.github/scripts/copilot-build.sh --build-ttnn-tests
```

The parent reports it attempted the wrapper and Docker is unavailable. It
also reports an installed native toolchain and an existing `build_Release`
tree. The inspected Ninja graph defines `ttnn_op_matmul`, `ttnncpp`, and
`ttnn`. A narrow native fallback that includes the factory, template
instantiation, Python binding, and dependent links is:

```bash
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --build build_Release --target ttnn --parallel 4
```

For early isolation, `--target ttnn_op_matmul` compiles the native op but is
insufficient by itself because it omits `matmul_nanobind.cpp`. Existing builds
use Unity sources, so do not assume a one-file object target exists. Record
the actual linked library paths and hashes before launching a device test;
compiling an unused tree does not verify the loaded Python module.

Host-only mock descriptor tests can separate dispatch selection from device
execution. Explicitly configure Blackhole mock mode before any context/device
initialization; never use `configure_mock_mode_from_hw`. The public mock API
supports Blackhole 1/2/4/8-chip configurations
(`tt-metalium/experimental/mock_device/mock_device.hpp`). Build descriptors
without enqueuing kernels and check:

1. A full local 1x4 tensor creates descriptors for all four coordinates and
   each multi-reader assignment equals the assignment obtained from that
   coordinate's physical device. Check complete `(worker, bank, index)`
   tuples, unique secondary cores, excluded storage cores, and three readers
   per active bank. This fails on the old code at the unit-mesh assertion.
2. A nonzero restricted coordinate and an offset submesh use their correct
   owning-mesh mapping. This discriminates against a first-device shortcut.
3. Missing coordinate on a direct multi-device multi-reader call and a
   remote multi-reader coordinate fail clearly. A one-reader descriptor on
   the existing path must still build without the new local-device query.
4. If a custom heterogeneous mock descriptor is available, vary harvesting
   across same-size grids and compare per-coordinate assignments with
   independent physical-device queries. A uniform mock alone cannot prove
   this property. Mock descriptor success proves no kernel correctness.

### Device checks for the parent

First isolate native matmul using BF16 activation/output, LoFi BFP4 weights,
and FP32 destination accumulation. Test readers 1/2/3 with distinct input and
weight values on each rank, replicated-input and TP-sharded-weight mappings,
and compare every rank against its own Torch product. Include the original
local packed projection `[32,5120] @ [5120,4160]`, with weight storage width
padded to `32 * banks * readers` and exactly matching program reader count.
Also test a small output width with padding-only banks and a reader with no
output shard; keep current layout validation enabled. Existing worker-count
tests in `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py`
are useful single-device regressions but do not currently establish TP4
coverage.

Use fresh same-shape inputs/weights on a program-cache hit, then capture/replay
with refreshed inputs. Check all ranks, not only an aggregate PCC. Distinct
rank payloads detect duplicated first-rank data; changed allocations detect
stale buffer bindings. A bitwise one-reader/two-reader equality requirement
would be unjustified if arithmetic ordering changes; compare against the
appropriate quantized-reference tolerance and require eager/trace equality
for the same policy.

After the projection control passes, rerun both decoder attention types:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh native_reader3_l0 --layer 0 --policy '{"attention_readers":3,"output_readers":3,"gate_readers":3,"up_readers":3,"down_readers":3}'
bash models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh native_reader3_l3 --layer 3 --policy '{"attention_readers":3,"output_readers":3,"gate_readers":3,"up_readers":3,"down_readers":3}'
```

Use the stage's device safety workflow, separate watcher/correctness runs
from profiler/performance runs, then compare warmed complete-layer trace
latency for readers 1/2/3 under identical precision and input shapes. The
original historical failure used a less optimized storage-core policy;
retain that original shape/policy as an assertion regression as well as
testing current selected storage/core layouts. Keep multi-reader defaults
only where the measured full-layer result and correctness gates support them.

## Verdict

The failure mechanism and existing coordinate-dispatch repair boundary are
verified by source inspection and the archived traceback. The proposed repair
is unimplemented and uncompiled by this agent. Multi-reader device correctness,
trace behavior, and performance remain unverified until the parent executes
the focused tests. No source or kernel changes outside this matmul factory
and its direct binding are needed for the proposed local TP4 repair.
