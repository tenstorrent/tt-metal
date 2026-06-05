# mcast.hpp (deepseek_v3_b1 unified_kernels) — annotation

File: `models/demos/deepseek_v3_b1/unified_kernels/mcast.hpp`
Role: **Already a hand-rolled two-sided Pipe.** Strongest existing evidence for the Pipe shape. INTRA-CHIP. IN SCOPE (reference, not a migration target — it IS the prior art).

This is the `deepseek_b1_ops::Mcast` struct: a persistent-state mcast micro-op with `init()` / `operator()` / `teardown()`, sender on BRISC, receiver on NCRISC (or BRISC via `ReceiverOnBrisc`). Built on raw `NOC_CMD_BUF_WRITE_REG` set-state/with-state primitives (lines 88–288), NOT on the `noc_async_*` API.

## SENDER block — `sender_impl`, lines 444–507
| step | lines | what |
|------|-------|------|
| preprogram data mcast state (addr+size, BRCST_PACKET) | 446–456 | `mcast_send_set_state<...,write_cmd_buf>` |
| preprogram sem-flag mcast state (4B) | 457–475 | `mcast_send_set_state<...,write_reg_cmd_buf>` or just inc counters if shared cmd_buf |
| wait source L1 ready | 477 | `cb_wait_front(src_cb, src_num_pages)` |
| **issue DATA mcast** | 479–489 | `mcast_send_with_state<...,write_cmd_buf>` |
| **issue SEM flag mcast** | 490–500 | `mcast_send_with_state<...,write_reg_cmd_buf>` |
| flush | 502 | `noc_async_posted_writes_flushed()` |
| pop source | 504–506 | `cb_pop_front` (if `pop_src`) |

## RECEIVER block — `receiver_impl`, lines 509–516
`cb_reserve_back(dst_cb)` → `noc_semaphore_wait(recv_sem, VALID)` → `noc_semaphore_set(recv_sem, INVALID)` → `cb_push_back(dst_cb)`.
`mcast_grid_impl` (518–523): same wait/reset without the CB push (for cores in the rectangle that aren't true CB receivers).

## init / teardown
- `init()` lines 396–421: sets sender sem INVALID, calls `init_persistent_mcast_sender` (lines 222–260) to preprogram NOC cmd-buf state once, then sets sender sem VALID.
- `teardown()` lines 431–440 → `teardown_persistent_mcast_sender` (262–288): includes `riscv_wait(10000)` "due to posted mcast hw bug".

## Forks
- **F1 = flush** (line 502 `noc_async_posted_writes_flushed`), teardown uses **barrier** (line 286).
- **F2 = flag** (level flag VALID/INVALID; receiver waits ==VALID then resets to INVALID). Lines 513–514.
- **F3 = template `loopback`** (CT bool). `loopback ? NOC_CMD_BRCST_SRC_INCLUDE : 0` line 162. `static_assert(is_part_of_receiver_grid)` when loopback (line 200). So **both F3 variants** are parameterized.
- **pre_handshake = ABSENT** — comment line 313: "Sender assumes that receiver's dst_cb is ready to receive at the beginning of execution." No receiver-ready wait before sending. (Contrast kv_cache_update/flash_mla which DO pre-handshake.)

## API shape (Pipe evidence)
- `posted=true, linked=true` hardcoded (lines 547–548).
- Compile-time fan-out: `mcast_num_cores`, `loopback`, `is_part_of_receiver_grid` as CT template args.
- Runtime `SenderArgs`: dest rectangle (start/end x/y), sender+receiver sem addrs, data size, src_cb, src pages, input data addr, mcast receiver data addr (lines 336–348).
- `ReceiverArgs`: receiver sem addr, dst_cb, dst pages (lines 350–354).
- **Send surface**: `op.init(args)` (once) → `op(args)` (repeatable) → `op.teardown(args)` (once).
- **Receive surface**: same `op(args)` dispatches to `receiver_impl` by RISC + CT flags.
- Data + sem mcast fused into one send; both halves carried in one unified `DMArgs` so the op can be placed on either RISC.

## HOLE flags
- NOC coord swap handled in `get_noc_multicast_addr<noc>` (lines 68–86) — noc 0 vs 1 swap baked in. Pipe must keep this.
- Shared-cmd-buf branch (`mcast_is_shared_write_cmd_buf`, line 20) doubles the control flow — a real complexity the Pipe will inherit or hide.
