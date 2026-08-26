# unified_routed_expert_ffn_reader.cpp — annotation (added by reconcile 2026-06-19)

File: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`
Family: ccl / deepseek (deepseek_prefill). Role: **hybrid (multi-mcast sender)**. Tag: **refactor**.

A deepseek_prefill FFN reader that multicasts **several distinct blocks** (in0, in1, gate, up) to a
receiver rectangle, each with its own ready/valid flag handshake. Same lineage as the existing
`reader_dispatch` / `reader_combine` annotations, but **wider** — multiple data-mcast + flag pairs per
iteration.

## BLOCK (sender legs)
| step | lines | call |
|------|-------|------|
| build in1 / in0 valid-sem mcast addrs | 208, 215 | `get_noc_multicast_addr(...)` ×2 |
| wait in0 local ready (= num receivers) | 282 | `noc_semaphore_wait(in0_ready_local, in0_num_receivers)` |
| build in0 data mcast addr | 312 | `get_noc_multicast_addr(...)` |
| **mcast DATA (in0)** | 315 | `noc_async_write_multicast(...)` |
| **mcast FLAG (in0 valid)** | 321 | `noc_semaphore_set_multicast(in0_valid_sem_addr, in0_mcast_valid_noc, in0_num_receivers)` |
| wait in1 local ready | 325 | `noc_semaphore_wait(in1_ready_local, in1_num_receivers)` |
| build gate/up data mcast addrs + **mcast DATA (gate, up)** | 367–379 | `get_noc_multicast_addr` + `noc_async_write_multicast` ×2 |

## Forks
- **F2 = FLAG** (`noc_semaphore_set_multicast`) — set-flag handshake (contrast the counter `inc_multicast`
  in reader_dispatch).
- **F3 = EXCLUDE_SRC** (`*_num_receivers` = recipient count, sender separate).
- **pre_handshake** = receiver-ready wait *before* each data send (`semaphore_wait` at 282/325).

## Migration-blocker audit → REFACTOR
- **Multiple independent data+flag legs** (in0, in1, gate, up) in one kernel → multiple `SenderPipe`
  instances (or repeated `send()`/`send_signal()` on distinct CBs/sems), like the matmul didactic
  2-Pipe kernels. Not a single-block clean migration.
- Confirm whether the gate/up legs share the same recipient rectangle as in0/in1 (likely) → one rect,
  several sends.
- Sem ids: confirm compile-time vs runtime (deepseek_prefill historically used runtime sem ids — see
  conv-weights P6 caveat) before deciding template vs method-arg.
