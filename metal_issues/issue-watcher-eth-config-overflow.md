# DRAFT upstream issue — watcher + fabric: kernel config buffer overflow on ACTIVE_ETH at open_mesh_device (Blackhole 1×4)

> Draft for filing at `tenstorrent/tt-metal`. **Not submitted** — local draft only. Needs a current-`main`
> confirmation (the observation is on a checkout ~1089 commits behind main).

**Component / Area:** tt-metal / dispatch + fabric + watcher
**Board:** Blackhole P300 (qb2), 1×4 mesh

## Observed
With `TT_METAL_WATCHER` enabled, opening a 1×4 mesh with a fabric config fails **at `open_mesh_device`**, before
any op runs:
```
critical | Always | TT_FATAL: Program size (27776) too large for kernel config buffer (25600) on ACTIVE_ETH (assert.hpp:104)
RuntimeError: TT_FATAL @ tt_metal/impl/program/program.cpp:2483: state.offset <= max_size
  ttnn::distributed::open_mesh_device(...)
```
Program sizes seen: **27776 / 27920** vs the **25600 B** `ACTIVE_ETH` kernel-config buffer. Reproduced with both
`FabricConfig.FABRIC_1D` and `FABRIC_1D_RING`. Without watcher the same `open_mesh_device(1×4)` succeeds (4
devices). Also observed **in real model runs** during QB2 stage-05 bring-up (falcon3-7b ×1, mistral ×2), so it is
not purely a synthetic-repro artifact.

## Impact
Watcher instrumentation inflates the ethernet dispatch/fabric program past the `ACTIVE_ETH` config-buffer
capacity, so **watcher cannot be used on 1×4 fabric meshes** on this configuration — which in turn blocks
watcher-gated CCL repros/gates (e.g. the one-tile scatter-write assert repro must run under watcher to fire).

## Steps (minimal)
```python
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)   # or FABRIC_1D_RING
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))   # under TT_METAL_WATCHER=120 -> TT_FATAL at program.cpp:2483
```
Run: `TT_METAL_WATCHER=120 TT_METAL_HOME=$METAL PYTHONPATH=$METAL python3 -c '<above>'`.

## Expected
Watcher-enabled `open_mesh_device` on a 1×4 fabric mesh should not overflow the ACTIVE_ETH kernel-config buffer
(raise the eth config-buffer budget, reduce the watcher eth-dispatch footprint, or provide a watcher-lite eth
mode).

## Notes
- No existing upstream issue found (searched: "kernel config buffer ACTIVE_ETH", "Program size too large kernel
  config buffer", "watcher ethernet dispatch program size").
- Please confirm on current `main` — the config-buffer size or watcher eth footprint may have changed.
