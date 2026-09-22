# Native fix source review

Bounded AutoFix validation of the five final root diffs against HEAD
`1623f9cb595`, 2026-09-12. This is not the final stage review. No additional
correctness defect was found in the changed paths for the supported local
Blackhole mesh configuration. Compatibility limits and incomplete test
coverage are stated below; this report does not certify the stage.

## API, ABI, and dispatch

- Native/Python `num_workers_per_dram_bank` still defaults to 1
  (`matmul_program_config_types.hpp:80`, `matmul_nanobind.cpp:570`). No model
  shape allowlist or default multi-reader selection was introduced.
- Python `create_descriptor` retains its optional fourth positional/keyword
  `core_range_set`, whose ignored behavior is unchanged. The new optional
  `mesh_dispatch_coordinate` is fifth and defaults to None. Existing
  three-argument and legacy fourth-argument Python calls remain accepted.
- **C++ ABI/source compatibility limit:** the internal factory's fourth
  parameter changes from `optional<CoreRangeSet>` to
  `optional<MeshCoordinate>`. Its mangled symbol changes, so binaries must
  be rebuilt/installed together; out-of-tree C++ callers passing a concrete
  CoreRangeSet must adapt. The only qualified in-tree call found is the
  updated nanobind wrapper. Defaulted three-argument calls remain valid.
  Do not claim binary compatibility with an old installed `_ttnn`/library.
- That exact fourth-argument signature opts into the existing adapter's
  coordinate loop (`mesh_device_operation_adapter.hpp:424-447,607-614`).
  No overload ambiguity is introduced. One-reader device selection remains
  the MeshDevice, including the pre-existing remote-coordinate path. However,
  one-reader programs are now grouped per coordinate rather than per uniform
  range; host construction/cache overhead is a measurement question, not
  proof of identical performance.

## Mesh coordinates and cache binding

The multi-reader branch resolves `a.device()->get_device(coord)` only after
checking presence/unit mesh, containment, locality, and a nonnull device
(factory `:988-1006`). Placement, harvesting translation, and output NoC
coordinates all receive that physical device. Tensor ownership, tensor-backed
CBs, and weight/bias buffer bindings remain MeshTensor based (`:918-925`).
The adapter retains bindings per singleton range and patches new buffers on
cache hits (`mesh_device_operation_adapter.hpp:597-603,735-760`).

An offset submesh is safe: `MeshDeviceImpl::create_submesh` copies the selected
parent devices into a newly based view (`mesh_device.cpp:716-734`), and the
factory queries the tensor's owning view. It never linear-indexes a local-only
`get_devices()` vector or substitutes root coordinate zero. A unit submesh's
omitted-coordinate fallback resolves its local zero.

**Support limit:** multi-reader remote coordinates fail explicitly. The
adapter iterates global tensor coordinates, so a multi-host mesh containing
remote tensor coordinates is not supported by this new path. One-reader
behavior avoids that new physical-device lookup. Different harvesting with
compatible shared geometry is addressed by physical-device placement; mixed
architectures, bank counts, grid/shard capacities, and allocator geometry
remain unproven. Existing Blackhole, NoC0, reader-count, and shard-width
validation remains in place (`factory:117-149`).

## Output-tail safety

Reader assignments are ordered bank then reader (`matmul_utilities.cpp:429-459`),
so reader `i` starts at `i * reader_width`. The added exhausted-storage branch
asserts this start is beyond N before setting write count zero
(`factory:815-822`). It neither removes the worker nor changes its DRAM reads
or compute participation. The existing argument-padding code supplies all 11
fixed slots. The writer still waits for and pops its output DFB while its
zero-count write loop performs no stores
(`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:210-246`).

The ordinary split branch clips each segment to an existing output shard.
Output specs allocate full shards (`matmul_device_operation.cpp:2536-2552`),
and `Buffer::num_dev_pages()`/`aligned_size_per_bank()` use the full shard page
count (`buffer.cpp:715-722,765-771,833-835`). Therefore padding written inside
the last shard is allocated storage. For N4160/two readers, nine 15-tile
shards provide 135 output tiles; reader 15 starts at tile 135 and writes zero.
The N32 two-tile final shard is also allocation-safe; its later conversion
validator issue and the excluded broader patch remain documented in
`AUTODEBUG_narrow_output.md`. This review does not expand that patch's scope.

## Packet-tag cleanup and architecture compatibility

The router's one added call restores only `noc_index` after write/atomic
barriers and before its final peer synchronization (`fabric_erisc_router.cpp:2949`).
It preserves all exit assertions, termination stores, and handshakes. The
saved failing-state decoder verifies all 32 target kernels' NoC ownership:
ERISC0 services NoC0, ERISC1 services NoC1. Clearing both interfaces from each
ERISC would violate this ownership; the submitted patch does not do so.

`noc_clear_packet_tags(uint32_t)` exists in both Blackhole and Wormhole NoC
headers and as the existing no-op in both Quasar API versions. The new host
factory references architecture-independent MeshDevice/IDevice APIs; its
header already includes the distributed coordinate definitions transitively
through `device_operation.hpp`. No added symbol requires a Blackhole-only
preprocessor branch. This is source compatibility inspection, not compilation
or runtime validation on other architectures. In particular, it does not
establish complete tag cleanup for unrelated single-ERISC/mixed-NoC fabrics.

## Regression strength and evidence boundaries

The native test uses the correct nested factory binding, distinct per-rank
Torch inputs, readers 1/2/3, full 1x4 and offset 1x2 meshes, native conversion,
and per-rank numerical checks. It retains three generations of buffers,
asserts distinct addresses and stable program-cache entry counts, and checks
coordinate-specific primary/minimum-distance secondary placement, exclusions,
uniqueness, legacy Python argument acceptance, missing-coordinate rejection,
and padding-only zero-write arguments. These are substantive regression
checks. Uniform hardware cannot prove heterogeneous harvesting behavior;
remote-coordinate rejection and invalid-coordinate calls are source-reviewed
but not exercised by this test.

**Coverage gap at review time:** the current parameterization is 18 cases
(three shapes, two meshes, three reader counts). The archived
`native_mesh_tests_retry.log` records 12 passes before N256 was added. Do not
attribute that old log to all 18 current cases; the final native regression
must retain its own result. N32 conversion remains excluded for the separate
validator issue described above.

`watcher_full_eth_tags_fix_repeat.exit_status` is 0, with full watcher10,
O3/noinline, and no ETH disable in the saved environment. The first
`watcher_full_eth_tags_fix.exit_status` is 2; its native shutdown evidence must
not be described as a fully successful process. The repeat supplies the
complete-process success evidence reported by the parent. Final model
regressions are still parent-owned work, outside this source-review verdict.

Checks performed here: `git diff --check`, Python AST parsing of the test
without importing it, source/call-site review, and read-only saved-artifact
inspection. No TTNN import, linked probe, build, or hardware operation ran.
Reviewed source SHA-256 prefixes, for detecting later changes: factory cpp
`958261a7fe8640f4`, factory hpp `3b43f714cfd010e8`, nanobind
`de18afec262b9f98`, router `af26a8b164b3ed36`, test `99843072beffbe82`.
