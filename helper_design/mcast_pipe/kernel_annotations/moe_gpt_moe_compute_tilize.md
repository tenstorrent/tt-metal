# moe_gpt + moe_compute tilize kernels — annotation

Files (intra-chip mcast leg of CCL ops; the local rectangle mcast is IN SCOPE, fabric leg out of scope):
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_reader.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/tilize_writer.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/tilize_writer.cpp`

All three carry a **local mcast helper pair** at the top of the file:
- `get_safe_multicast_noc_addr(...)` — NOC0/NOC1 coord-swap wrapper around `get_noc_multicast_addr`. (reader 20–22, writer 20–22 / 22–25)
- a `multicast(...)` helper that loops `noc_async_write_multicast` over `NOC_MAX_BURST_SIZE` chunks (writer 29/34, moe_compute 32/37). **New spelling: chunked-burst data mcast loop.**

## Canonical BLOCK — moe_gpt/tilize_reader.cpp, drain-core branch, lines 660–763
SENDER (is_drain_tilize_core):
| step | lines | call |
|------|-------|------|
| build 3 mcast addrs (data) | 660–674 | `get_safe_multicast_noc_addr(...)` ×3 |
| **mcast DATA block ×3** (e_t buffer, per-expert counts, total_chunks) | 677–689 | `noc_async_write_multicast(...)` ×3, num_dests = N-1 |
| set local sem flag = 1 | 693 | `noc_semaphore_set(metadata_ready_semaphore_addr, 1)` |
| build sem mcast addr | 695–700 | `get_safe_multicast_noc_addr(...)` |
| **mcast SEM flag** | 703–704 | `noc_semaphore_set_multicast(...)` |
| **flush** | 708 | `noc_async_writes_flushed()` |
| (2nd target: MM cores) write per-expert sems, mcast count block | 724–746 | `noc_async_write_multicast` + `noc_async_write_barrier` (line 747) |
| mcast ready flag to MM cores | 753–763 | `noc_semaphore_set(...,1)` + `noc_semaphore_set_multicast` |

## Forks (across all three)
- **F1 = BOTH**: `noc_async_writes_flushed` (reader 708, used when the local sem value is mutated again right after) AND `noc_async_write_barrier` (reader 747). The choice is data-dependent: flush when the source sem L1 will be overwritten for a *different* mcast next; barrier when fully done.
- **F2 = flag** (set local to 1, mcast, receiver waits/resets). Per-expert "counts encoded into the semaphore value" (lines 718–728) is a **value-carrying flag** variant — the sem payload is data, not just 1.
- **F3 = EXCLUDE_SRC** (num_dests = bounding_box_num_cores − 1). No loopback.
- **pre_handshake = ABSENT** on the drain→non-drain leg; the non-drain→drain count-collect (lines 765+) is a separate atomic-inc gather.

## HOLE flags
- Two *different* destination rectangles per kernel (tilize cores vs matmul cores), each with its own coord-swap + flag. Pipe must support multiple dest rectangles per sender, or be instantiated twice.
- Value-carrying semaphore (counts packed into the sem word) — Pipe's F2 must allow an arbitrary 32-bit payload, not only a fixed flag constant.
- Chunked-burst `multicast()` loop (NOC_MAX_BURST_SIZE) — Pipe's data path must chunk large blocks.

## tilize_writer.cpp (both moe_gpt + moe_compute)
Same helper pair; sem-mcast sites at moe_gpt 499/558/570, moe_compute 538/716/728. Receiver-side and multi-target sem-flag mcasts — same F1=flush/barrier, F2=flag, F3=EXCLUDE_SRC profile. No new fork.
