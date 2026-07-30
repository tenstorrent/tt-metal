# move_stick_layout_interleaved_with_overlap.cpp

Path: `ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp`
Role: row-major (stick) twin of `move_interleaved_with_overlap.cpp`. One pass.
API era: legacy free-function.

## Block instances
**Structurally identical** to `move_interleaved_with_overlap.cpp`; only the data-movement differs (page/stick reads & writes with `aligned_page_size` instead of tile reads).

### Block 1 — SENDER half (controller) — lines 63-77
- `noc_semaphore_wait(semaphore_addr_ptr, control_value)` (L64).
- mcast cluster: `get_noc_multicast_addr`+`noc_semaphore_set_multicast` for range0 (L67-69), range1 (L70-72), optional range2 (L73-77).

### Block 1 — RECEIVER half (non-controller) — lines 78-84
- `noc_semaphore_inc(controller_noc_address, 1)` (L81).
- `noc_semaphore_wait(semaphore_addr_ptr, control_value)` (L83).

Data write barrier: `noc_async_write_barrier()` (L91) guards dst writes (L88-93).

## Mapping / Forks / HOLEs
Identical to `move_interleaved_with_overlap.md`. F1=barrier(on data, not mcast); F2=counter, single-shot, no reset; F3=opaque (legacy API), assume EXCLUDE_SRC; pre_handshake=dest reused (same L1 word as counter+flag).

This + the tile variant are a **single shared block in two skins** — strong evidence the helper must be layout-agnostic (operate on the sem flag only, leave data movement to the caller).
