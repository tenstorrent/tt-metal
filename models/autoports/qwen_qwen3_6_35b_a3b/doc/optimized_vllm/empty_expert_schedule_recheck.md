# Empty-expert protocol recheck and exact build plan

Second source review only. No candidate source was applied, no target was built, and no device was opened.

## Shared mailbox lifetime

The three CB views deliberately alias one64-byte allocation: `CB_X_STAGE`, `CB_MAILBOX_WRITER`, and `CB_MAILBOX_COMPUTE` (`moe_fused_swiglu_geometry.cpp:329–331,401`). Each has independent FIFO state and one page. The new header pushes only the two mailbox views. It does not reserve/push/pop the tilization-completion view, so it cannot shift that existing producer/consumer protocol.

The reader reserves **both** mailbox views before writing the header or any later expert publication. Writer snapshots the complete `Mailbox` into a local struct and copies its two bitmap fields before popping. UNPACK snapshots those fields, sends them to MATH and PACK using the existing two-word mailbox pattern, then pops. Therefore the reader cannot reuse the shared header payload until both consumer views have released it. All TRISCs retain their own bitmap words; subsequent expert publications cannot overwrite the retained schedule.

The existing BF16 tilization path confirms that `CB_X_STAGE` is a completion event, not a payload writer: compute reserves it, calls `tilize_row` targeting `cb_x_tiles`, and pushes the completion page (`compute.cpp:608–626`); reader waits/pops it (`reader.cpp:780–781`). Adding one header to the other independent FIFO views does not change that sequence or its physical address.

The header writes words0–4 only. The reader’s initialization of `MBOX_HSEND_DONE` and `MBOX_UP_SCATTER_DONE` (words5/6) and row-aggregator coordinates (words8+) is unchanged. Header magic `MAILBOX_MAGIC + EXPERTS_PER_CHIP` is distinct from every legal per-expert magic. The at-most64 guard keeps the two-word bitmap valid, including bits31 and63. Each launch rebuilds the bitmap from freshly broadcast routing metadata; capture does not cache a host schedule.

The reader’s `gb`, `hfree_seq`, `hsend_seq`, `wd_done` and peer counters are incremented only by retained M-blocks, exactly as before. The compact loop preserves original ascending IDs and per-expert weight addresses. Empty experts previously had zero blocks and zero `h_cursor`, so skipping their body does not remove a hidden alignment pad.

The final reader `noc.async_write_barrier()` still executes on the all-empty path and drains metadata multicast acknowledgements before kernel completion. Writer’s final `out_pending` wait/write barrier/pop still executes when an active expert left output pending, even if all later experts are empty. No existing read, write or atomic barrier was removed. All-empty compute/writer still consume the header and complete; the enclosing program completion waits for those kernels, so no extra reader-side final header-ACK round is required by the existing dispatch contract.

No source-level lifetime defect was found. This is not hardware proof: cold kernel compilation, changed-mask traces, all-empty ranks, dense traffic, and repeated back-to-back dispatches remain required.

## Exact configured rebuild and installation

Observed configuration:

- `build` is a symlink to `build_Release`; build typeRelease, generatorNinja.
- `CMAKE_COMMAND=/usr/local/bin/cmake` (4.0.2), `CMAKE_MAKE_PROGRAM=/usr/bin/ninja` (1.11.1), compiler`/usr/bin/clang++-20`.
- `ENABLE_TTNN_SHARED_SUBLIBS=OFF`, `WITH_PYTHON_BINDINGS=ON`, `TT_INSTALL=ON`.
- Install prefix is this checkout’s `build_Release`, not a system prefix.

Minimal dependency-aware build target is **`ttnn`**, not just the operation object or `ttnncpp`:

```bash
/usr/local/bin/cmake --build build_Release --target ttnn --parallel 4
/usr/local/bin/cmake --install build_Release --component ttnn-runtime
/usr/local/bin/cmake --install build_Release --component tt_pybinds
```

This sequence is a plan, not an executed build. The parallelism4 setting bounds simultaneous unity compiles; it is not a measured fastest build setting.

The configured graph provides the following dependency chain:

1. Native operation unity source `.../moe_fused_swiglu/CMakeFiles/ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu.dir/Unity/unity_0_cxx.cxx` includes the public wrapper, device operation and program factory.
2. That object is linked into `build_Release/ttnn/_ttnncpp.so`. Header dependencies may also rebuild other C++ callers; let Ninja determine them rather than manually compiling a single object.
3. `build_Release/ttnn/CMakeFiles/ttnn.dir/Unity/unity_37_cxx.cxx` contains the changed nanobind source. Target`ttnn` links `build_Release/ttnn/_ttnn.so` and depends on `_ttnncpp.so`.
4. Component`ttnn-runtime` updates `build_Release/lib/_ttnncpp.so`, `build_Release/lib/_ttnn.so` and registered runtime files, including kernels. Component`tt_pybinds` updates `ttnn/ttnn/_ttnn.so` used by the editable checkout.

`readelf` confirms the editable `_ttnn.so` needs `_ttnncpp.so` and searches `$ORIGIN/../../build/lib`; merely relinking the uninstalled build-tree libraries would leave that runtime dependency stale. Installing only the Python binding would likewise leave its native dependency stale. The changed native function signature requires a fresh Python/server process after both installs. No additional full Metalium target is requested; unchanged dependencies are left to the configured graph.

A read-only `/usr/bin/ninja -C build_Release -n ttnn` reports glob recheck and CMake regeneration are pending. Consequently that dry run cannot establish a final object count or build time before regeneration; no such estimate is claimed. It did not execute either step. Use CMake’s configured Ninja rather than the different virtual-environment Ninja selected by bare`ninja` on PATH.

## Probe datatype and harness patch

`doc/datatype_sweep/selected_precision_config.json` selects `weight_groups.expert_ep=bfp8`, with LoFi, FP32 destination disabled, approximate math disabled and packer L1 accumulation disabled. The probe’s compute settings already match those four fields. Its weight default is now **BFP8** rather than the first draft’s BF4. `--weight-dtype bfp8|bfp4|bf16` makes alternate coverage explicit, and every artifact records `weight_dtype`. CPU metadata mode passed with the new default and `hardware_verified=false`; it performed no weight conversion or device execution.

`empty_expert_schedule_harness.patch` is a separate archived patch (three files,13 insertions/1 deletion):

- Adds `--compact-empty-experts` to reduced bucket and full-logit harnesses, forwarding `compact_empty_moe_experts=True` only to candidate construction.
- Forces a separate fixed32 reference when that flag alone is selected; records that the reference compaction flag isFalse. Existing ND/route/state/row reference decisions are unchanged.
- Adds the effective `compact_empty_moe_experts` value to the native matched helper’s result. It does not silently enable a native override or change the adapter.

The temporary harness copies compile and Black checks pass; `git apply --check` passes. Active harness files were not edited. After native microgates, the owner may apply this test patch and add `--compact-empty-experts` to the established candidate reduced/full40 invocations. HTTP mode still reports a requested harness flag, not proof that an external server enabled it; match actual server construction separately.
