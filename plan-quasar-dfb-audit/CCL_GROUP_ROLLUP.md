# CCL Group — CB→DFB Kernel Audit Rollup

**Date:** 2026-07-15
**Group:** CCL ops from `gchoudhary_likely_to_do_list.csv` (3 ops)
**Audit spec:** `plan-quasar-dfb-audit/cb_dfb_kernel_audit_helper.md`

## Group verdict: GREEN

**Bottom line — how hard is a kernel-only CB→DFB port for the CCL group? Easy.** All three ops are GREEN. No GATE (`get_local_cb_interface` field access), no silent-wrong (`get_cb_tiles_*_ptr`), no Quasar-blocked runtime dependency (`read_tile_value`/`get_tile_address`), and no LTA prerequisite anywhere in scope. Data CBs are canonical Class 1 linear FIFOs (mechanical `CircularBuffer` → `DataflowBuffer` rename); the only non-FIFO CBs are private-L1 scratch that map autoportably to `ScratchpadSpec`.

## Per-op rollup

| Op | Factory (slice) | In-scope kernels | CBs (FIFO / scratch) | GATE | Quasar-blocked | Verdict | Report |
|----|-----------------|------------------|----------------------|------|----------------|---------|--------|
| `ccl/all_broadcast` | `AllBroadcastProgramFactory` | 4 donor (from `ccl/broadcast`) | 1 / 0 | none | none | GREEN | `ccl/all_broadcast/CB_DFB_KERNEL_AUDIT.md` |
| `ccl/all_to_all_dispatch` | `AllToAllDispatchSparse` | 2 | 3 / 3 | none | none | GREEN | `ccl/all_to_all_dispatch/CB_DFB_KERNEL_AUDIT.md` |
| `experimental/ccl/all_gather_async` | `AllGatherViaBroadcastFactory` | 2 | 1 / 0 | none | none | GREEN | `experimental/ccl/all_gather_async/CB_DFB_KERNEL_AUDIT.md` |

## Notes & follow-ups (kernel-only scope)

- **Shared donor kernels:** `all_broadcast`'s 4 kernels live under `ccl/broadcast/device/kernels/` — a port covers both ops. `ccl/broadcast` is not separately on the do-list.
- **Unaudited `all_gather_async` factories:** `AllGatherAsyncDefaultProgramFactory` (`minimal_default_*` kernels) and `AllGatherAsyncLlamaShardedProgramFactory` (`llama_shapes_sharded_*` kernels) were OUT OF SCOPE for the `AllGatherViaBroadcastFactory` slice. Audit them as separate slices before porting those code paths.
- **`all_to_all_dispatch` scratch CBs:** `c_3`/`c_4`/`c_5` are Class 6 private-L1 scratch → `ScratchpadSpec`. Confirm `c_5`'s reader→writer metadata handshake is covered by an existing semaphore/fabric barrier; if a real cross-kernel sync edge exists it becomes `ScratchpadSpec + SemaphoreSpec` (still autoportable/GREEN).
- **Host-side feasibility not covered here.** This is a kernel-only audit; SPSC/endpoint legality and `DataflowBufferSpec` fit are tracked by the host audit (`port_op_to_metal2_audit.md`).
