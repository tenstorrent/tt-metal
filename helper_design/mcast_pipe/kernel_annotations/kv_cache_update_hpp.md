# kv_cache_update.hpp (deepseek_v3_b1 unified_kernels) — annotation

File: `models/demos/deepseek_v3_b1/unified_kernels/kv_cache_update.hpp`
Role: SENDER+RECEIVER, BRISC/NCRISC. Two-sided block built on `dataflow_utils.hpp` set-state/issue-txn helpers. INTRA-CHIP. IN SCOPE.

## THE BLOCK (NopeSender), lines 150–221
SENDER half (`IsNopeSenderCore`):
| step | lines | call |
|------|-------|------|
| set local sender flag VALID | 159 | `noc_semaphore_set(sender_sem_ptr, VALID)` |
| build mcast addr | 161–166 | `get_noc_multicast_addr<0>(...)` |
| preprogram DATA mcast state | 168–176 | `unified_kernels::noc_async_write_multicast_preprogram_all_state<posted=false>` (VC `NOC_DISPATCH_MULTICAST_WRITE_VC`) |
| preprogram SEM mcast state (4B, write_reg_cmd_buf) | 178–191 | same helper, guarded `!mcast_is_shared_write_cmd_buf` |
| wait source L1 ready | 193 | `cb_wait_front(kv_rmsnorm_output_cb, num_tiles)` |
| **issue DATA mcast** | 197–198 | `noc_async_write_multicast_issue_txn<posted=false>` |
| **issue SEM mcast** | 200–218 | `multicast_write_with_state` (shared cmd_buf) OR `noc_async_write_multicast_issue_txn<...,write_reg_cmd_buf>` |
| barrier | 220 | `noc_async_write_barrier(MCAST_NOC_INDEX)` |

RECEIVER half (`IsNopeCore`), lines 222–231:
`cb_reserve_back` → `noc_semaphore_wait(recv_sem, VALID)` → `noc_semaphore_set(recv_sem, INVALID)` → `cb_push_back`.

(Earlier mcast at lines 136/159 etc. in the file: a separate non-block atomic-inc handshake — note but not THE BLOCK.)

## Forks
- **F1 = barrier** (line 220).
- **F2 = flag** (VALID/INVALID level flag; receiver resets). Lines 159/228/229.
- **F3 = EXCLUDE_SRC** (no loopback; `posted=false`, num_dests excludes sender).
- **pre_handshake = ABSENT in this nope block** — but the file's other mcast site (k-chunk, see flash_mla-style) does receiver-ready waits via `unicast_atomic_inc`. Mixed across the file.
- Uses the **preprogram_all_state → cb_wait → issue_txn** split: state set up BEFORE the producer's data lands, issue only after `cb_wait_front`. This split is the perf-relevant style the bake-off should test.

## HOLE flags
- `mcast_is_shared_write_cmd_buf` branch (line 178/200) doubles the control flow exactly as in mcast.hpp. Same hazard.
- Custom VC `NOC_DISPATCH_MULTICAST_WRITE_VC` + `static_assert(noc_mode==DM_DYNAMIC_NOC ...)` (line 152) — Pipe must expose VC + noc selection.
