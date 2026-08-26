# moe_compute tilize_writer.cpp + tilize_reader.cpp — DEFERRED (design-gap)

Kernels (co-compiled by `test_moe_compute_single_card_gpt_oss`, 1x1 mesh):
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_reader.cpp`
Tier: 2d. Status: deferred (design-gap). No code change.

## tilize_writer.cpp — value-carrying flag (the listed design gap)
The sender broadcasts an ARBITRARY VALUE in the semaphore word, not VALID/INVALID and not a +1 counter:
- `matmul_chunk_ready_semaphore_set_value` starts at 1 and is `++`'d every chunk (lines 408, 696-697).
  Each round: `noc_semaphore_set(addr, set_value)` then `noc_semaphore_set_multicast(addr, mcast_addr, ...)`
  (lines 694-702) — broadcasting the running chunk count. Receivers `noc_semaphore_wait(addr, <that value>)`.
- `matmul_chunk_available_semaphore_wait_value += num_matmul_cores` (line 516) accumulates a runtime
  value that is rebroadcast via `set_multicast` (lines 522-524).

v7 has only: Flag = fixed VALID/INVALID (`data_ready.set(VALID)` + exact `wait(VALID)`), or Counter =
`inc_multicast(+1)` + `wait_min(++round)`. There is no `set_multicast(arbitrary_value)` verb. This is the
"arbitrary-value set_multicast flag (value-carrying flag)" gap called out in the task's KNOWN HELPER
DESIGN GAPS list.

Additionally the rects use RUNTIME num_dests (`num_matmul_cores`, etc.).

## tilize_reader.cpp — runtime multi-rect counts (companion)
Its data broadcasts target three DIFFERENT rectangles with RUNTIME counts (lines 795-857):
`tilize_bounding_box_num_cores - 1`, `matmul_bounding_box_num_cores`,
`all_worker_cores_bounding_box_num_cores - 1`. v7 `SenderPipe` is single-rect with a COMPILE-TIME
`NUM_ACTIVE_RECEIVER_CORES` — same runtime/multi-rect gap that deferred gn_v2. Co-compiled with the
writer; deferred together.

## Coverage note
`test_moe_compute_single_card_gpt_oss` does run on a 1x1 mesh on WH, BUT requires an unharvested grid
>= 7x10 (70 cores). A standard n150 (8x8 = 64-core worker grid) hits `grid.y < 10` and `pytest.skip`s.
So coverage on this specific chip is also doubtful — but the binding blocker is the design gap above.

## Verdict
DESIGN-GAP (value-carrying flag + runtime multi-rect count). Do NOT touch the helper. Both deferred.
