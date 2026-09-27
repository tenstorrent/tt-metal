# AutoFix: multicast scatter-state initialization

## Starting evidence
The full-model Watcher run aborts on `multicast_writer.cpp`, core (0,0) BRISC, assertion line 279. Source inspection identifies `populate_unicast_scatter_write_fields` in `tt_metal/fabric/hw/inc/api_common.h`: scatter writes require at least two chunks. `FabricWriter` nevertheless initializes scatter headers unconditionally even when `use_scatter_write` is false. A uint32 tile has a 4096-byte page; the 4352-byte fabric packet fits one page, while the actual transfer correctly selects unicast.

## Discriminating experiment
Baseline command:
`TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 timeout 90 python bringup/artifacts/reference-fix/probe_watcher_gather.py --output bringup/artifacts/reference-fix/watcher_gather_before.json`
Console: `watcher_gather_before.log`; Watcher: `watcher_gather_before_watcher.log`.

Both cases use the real common sampler `_perform_all_gather` with common TT_CCL, TP4 vocabulary sharding, local shape [1,1,32,32]. BF16 (2048-byte page, two chunks) passes exact output comparison on all four chips. Uint32 (4096-byte page, one chunk) then aborts with exit 134 and the same BRISC core (0,0), multicast_writer.cpp, assertion 279 signature. Hypothesis verified.

## Repair
Only initialize scatter state under `if constexpr (use_scatter_write)` at both the normal route and alternate-route initialization sites in `multicast_common.hpp`. Unicast initialization remains unconditional, preserving both the one-page path and the singleton-tail flush from the scatter path. No precision, routing, packet size, or sampler behavior was changed.

Patch: `multicast_scatter_init.patch`, applied to the main worktree by the parent after review. The patch also adds `models/autoports/google_gemma_4_26b_a4b_it/tests/probe_multicast_page_contract.py` as the durable regression. Pre-commit including clang-format passed. Parent's required `.github/scripts/copilot-build.sh --build-ttnn-tests` attempt exited 1 because Docker was unavailable (stage `native_build.log`). Device tests below compiled the affected kernels through the normal JIT.

## After validation
For both default fabric and `--ring`:
`TT_METAL_WATCHER=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 90 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multicast_page_contract [--ring] --output <artifact.json>`

Both commands exit 0. BF16 and uint32 each pass eager execution and two changed-input trace replays on all four chips: six pass rows per topology, twelve total. Artifacts: `watcher_gather_after_linear.json/.log`, `watcher_gather_after_ring.json/.log`, corresponding `_watcher.log` files.

Alternate-route execution is verified from the ring-run compiled ELF DWARF, which identifies `FabricWriter<2048, 4352, true>` and `FabricWriter<4096, 4352, true>`. Exact ELF paths and extraction results are in `watcher_gather_ring_template_evidence.txt`; kernel-to-run mapping is `watcher_gather_after_ring_kernels.yaml`. The native all_gather topology argument is deprecated and ignored: `get_axis_topology` derives Ring from FABRIC_1D_RING plus the physical closing link. Therefore no ineffective topology override was added.

## Device recovery
After the expected baseline abort: bounded list, reset, list, and TP4 mesh smoke all exit 0. All four devices are visible; no second reset or lock removal was needed. Evidence: `watcher_before_reset_list.log`, `watcher_reset.log`, `watcher_after_reset_list.log`, `watcher_mesh_smoke.log`. Baseline process already exited; no process was killed. After both passing tests the mesh closed successfully. Devices returned to the parent.

## Final status
Fixed and verified at the exact common-sampler CCL component boundary under Watcher, including both route variants and trace replay. The parent must rerun the original full-model Watcher gate to establish integration evidence. No full-model performance claim is made.
