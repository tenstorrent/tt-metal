# welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp — MIGRATED

- Tier: 2 (refactor-low), worklist item 3.
- Role: receiver (group_norm sharded welford).
- Validation nodeid (gn_v2 welford): `tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-welford-N=1-C=1280-H=16-W=16-num_groups=32-device_params={'l1_small_size': 0}]`.
- Result: **PASS** (1 passed in 3.71s; 12 fresh JIT builds → kernel recompiled). Commit `e7bc619869c`.

## Chosen template
`dataflow_kernel_lib::Pipe<>` — i.e. defaults `<EXCLUDE_SRC, Staging::Flag, PRE_HANDSHAKE=true, LINK=true>`.
Receiver-side MCAST/LINK are irrelevant (only STAGING + PRE_HANDSHAKE matter on receive()).
Constructed with `McastRect::single_core(mcast_sender_noc_x, mcast_sender_noc_y)`.

Semaphore mapping (GN naming is flipped vs matmul):
- `reduce_sender_sem`   = data-ready S->R level flag (cleared+waited)  → `data_ready` ctor arg
- `reduce_receiver_sem` = consumed R->S counter (up'd)                → `consumed` ctor arg

This mirrors EXACTLY the already-migrated non-welford twin
`reader_mcast_receiver_unary_sharded_gn_v2.cpp` (Tier 1), which uses the same `Pipe<>` + same sem
mapping + same single_core rect.

## Diff summary
- Added `#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"`.
- Constructed `reduce_pipe` right after the two Semaphore<> declarations.
- Replaced the per-group handshake block:
  ```
  reduce_receiver_sem.up(noc, mcast_sender_noc_x, mcast_sender_noc_y, 1);
  reduce_sender_sem.wait(VALID);
  reduce_sender_sem.set(INVALID);
  ```
  with a single `reduce_pipe.receive();`.
- Net: 1 file changed, 15 insertions, 6 deletions. Lines of raw handshake removed: 5 (3 sem ops +
  2 comment lines) replaced by 1 call + ctor.

## Notes
- The helper's `receive()` order is up → wait(VALID) → set(INVALID), which matches the kernel's
  original up → wait → clear exactly (the audit's "clear-after-wait anomaly" for welford is the SAME
  order the standardized helper uses, so no behavioral change).
- Counterpart sender (item 2) is multi-rect FAILED and left RAW; the migrated receiver is fully
  protocol-compatible with that untouched raw sender (validated green).
