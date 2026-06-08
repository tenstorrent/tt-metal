# activation_reader_width_sharded.cpp — MIGRATED (partial)

- Tier: 2 (refactor-low), worklist item 4.
- Role: hybrid (round-robin self-mcast sender + receiver in one kernel; conv2d width-sharded).
- Validation nodeid (conv WIDTH_SHARDED): `tests/ttnn/unit_tests/operations/conv/test_conv2d.py::test_conv_features[output_layout=Layout.TILE-math_fidelity=MathFidelity.HiFi4-filter=3-padding=(1, 2, 2, 3)-packer_l1_acc=True-fp32_accum=True-input_dtype=DataType.BFLOAT16-output_dtype=DataType.BFLOAT16-output_channels=353-input_channels=384-input_height=8-input_width=8-shard_layout=TensorMemoryLayout.WIDTH_SHARDED-config=None-batch_size=2-stride=2-device_params={'l1_small_size': 16384}]`.
- Result: **PASS** (PCC 0.9999992597771459, threshold 0.997; 10 fresh JIT builds → kernel recompiled). Commit `967cb69b01a` (re-committed after clang-format hook reformatted the ctor split).

## What was migrated (partial)
ONLY the SENDER half's data + flag broadcast (the INCLUDE_SRC loopback). Wrapped as a single
`Pipe::send()`.

## Chosen template
`dataflow_kernel_lib::Pipe<Noc::McastMode::INCLUDE_SRC, dataflow_kernel_lib::Staging::Flag, /*PRE_HANDSHAKE=*/false>`
(LINK defaults to true).
- INCLUDE_SRC: sender is inside the mcast rectangle and lands the block in its own act_cb via
  hardware loopback (kernel used `async_write_multicast<INCLUDE_SRC>` + `set_multicast<INCLUDE_SRC>`).
- Staging::Flag: S->R is a level flag (set VALID / wait / clear).
- PRE_HANDSHAKE=false: the R->S readiness counter is left RAW (see below).

`McastRect{mcast_rect.noc_x_start, .._y_start, .._x_end, .._y_end, num_reader_cores}` — note the
conv kernel has its OWN 4-field `struct McastRect` (from conv_reader_common.hpp); the helper's is
fully-qualified `dataflow_kernel_lib::McastRect` (5 fields incl. num_dests) so there is no collision.

Semaphore mapping:
- `act_mcast_receiver_sem` = S->R data-ready VALID flag → `data_ready` ctor arg
- `act_mcast_sender_sem`   = R->S readiness counter      → `consumed`  ctor arg (driven RAW, not by Pipe)

`send(tilized_in0_cb.get_read_ptr(), act_cb.get_write_ptr(), act_mcast_sender_size_bytes)`.

## Why PARTIAL (what was left RAW and why)
KNOWN LIMIT #1(b): divergent counts. The data/flag mcast population is `num_reader_cores`
(= `all_reader_cores.size()`, the full reader grid), but the R->S readiness handshake uses
`act_mcast_sender_sem.wait_min(num_mcast_cores - 1)` / `set(0)` where
`num_mcast_cores = max(num_input_cores, num_output_cores)`. The ack count
(num_mcast_cores-1) differs from the mcast population (num_reader_cores), AND the receiver branch
acks per-core with `up(...,1)`. A single-`num_dests` Pipe with PRE_HANDSHAKE=true cannot express the
divergence (it would wait `num_dests` = num_reader_cores).

Resolution: set PRE_HANDSHAKE=false so `send()` does ONLY {data mcast + raise flag + flush}, all of
which use `num_reader_cores` consistently — a clean Pipe verb. Everything else stays raw:
- sender branch: `act_mcast_sender_sem.wait_min(num_mcast_cores-1)` + `set(0)` (R->S drain)
- sender branch: `act_mcast_receiver_sem.set(INVALID)` pre-clear before send
- receiver branch (whole `else`): `set(INVALID)` / `up(sender,1)` / `wait(VALID)` — untouched raw

## Fence change (validated safe)
The original used `noc.async_write_barrier()` after the data+flag mcast. The helper's Flag+LINK path
uses the baked-in `async_writes_flushed()` (F1 = flush, the bake-off winner over barrier). Data and
flag ride the same NoC/VC (the kernel comment confirmed static-VC ordering), so flush is sufficient:
flag arrival proves data arrival (INV4). PCC 0.9999992 confirms correctness; no hang.

## Diff summary
- Added `#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"`.
- Removed the now-unused `MulticastEndpoint mcast_ep;` + `McastDst mcast_dst{...}` locals (would
  otherwise be unused-variable).
- Constructed `act_mcast_pipe` once, after `act_mcast_receiver_sem.set(VALID)`.
- Replaced the sender block (tilized_src endpoint setup + async_write_multicast<INCLUDE_SRC> +
  set(VALID) + set_multicast<INCLUDE_SRC> + async_write_barrier) with one `act_mcast_pipe.send(...)`.
- Net (post clang-format): 1 file changed, 25 insertions, 43 deletions.
