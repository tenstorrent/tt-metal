# reader_indexer_score.cpp — annotation (added by reconcile 2026-06-27)

File: `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/reader_indexer_score.cpp`
Family: **transformer + sdpa**. Role: **hybrid** (per-direction role rotates: none / sender / receiver).
Tag: **refactor**. Status: **deferred**. Flag: **`design-gap`**.

Reader for `indexer_score` (a banded q·k attention-scoring DMA reader; includes the sdpa
`dataflow_common.hpp`, builds the causal `[diag, full]` -inf mask tiles). Each core owns a
(group-phase × k-band) rectangle. Banded-product mcast: a grid ROW shares q/w (q-mcast), a grid COLUMN
shares the k-band (k-mcast). The two directions are INDEPENDENT — either may be off. Per direction a core
plays `role none` (plain DRAM read), `role sender` (read DRAM + mcast), or `role receiver` (L1→L1 copy).

## THE BLOCK — chain-link STAR with cross-id relay
The sender/receiver halves are factored into `mcast_send` (L67-84) and `mcast_recv` (L87-93), driven by
`read_block_or_mcast` (L98-118). The doc-comment on `mcast_send` says it "Mirrors chain_link."

### Sender half — `mcast_send<send_sem, recv_sem, valid_sem>` (L67-84)
| step | lines | call |
|------|-------|------|
| wait all receivers ready (pre-handshake) | 70 | `Semaphore<>(send_sem).wait(d.ndst)` |
| reset the ready counter | 71 | `s.set(0)` |
| **mcast DATA block** | 72-79 | `noc.async_write_multicast(CoreLocalMem(addr), MulticastEndpoint{}, bytes, d.ndst, {}, {rect..., .addr=addr}, /*linked=*/true)` |
| **relay VALID** (cross-id) | 81-82 | `Semaphore<>(valid_sem).relay_multicast(noc, Semaphore<>(recv_sem), d.xs,d.ys,d.xe,d.ye, d.ndst, /*linked=*/false)` |
| flush | 83 | `noc.async_writes_flushed()` |

### Receiver half — `mcast_recv<send_sem, recv_sem>` (L87-93)
| step | lines | call |
|------|-------|------|
| arm own doorbell INVALID | 90 | `Semaphore<>(recv_sem).set(0)` |
| signal sender "I'm ready" | 91 | `Semaphore<>(send_sem).up(noc, d.sx, d.sy, 1)` |
| wait the relayed VALID | 92 | `r.wait(1)` |

Two independent Pipes per core (q-row direction, k-col direction); the role and rectangle for each are
unpacked per direction from runtime args via `read_mcast_dir` (L53-63). `read_q_rows` (L162-185) runs a
per-row variant (one rendezvous per q-row); `read_q_block`/`read_w_group`/`read_k_chunk` run one
rendezvous per block.

## Forks / invariants
- **F4 = LINKED-PAIR** (H10) — the DATA write is `linked=true` (L79) and the VALID relay (L81-82) is
  issued **back-to-back** with NO barrier between (an intervening flush/barrier would DEADLOCK, kernel
  comment L80); a single `async_writes_flushed()` (L83) closes the pair. This is the linked-pair contract,
  not unlinked-barrier-between.
- **F1 = FLUSH** (`noc.async_writes_flushed`, L83). No barrier.
- **F2 = FLAG (relayed VALID) + counter pre-handshake.** Data-ready is a level VALID delivered by relay;
  the pre-handshake R→S is a counter (`up` / `wait(ndst)` + reset to 0).
- **F3 = EXCLUDE_SRC** — the sender reads its own block from DRAM and is excluded from the receiver rect.
- **KNOB pre_handshake = present** — sender waits all `ndst` receivers ready before mcasting (dest slot
  reused across the band/group loop).

## Migration-blocker audit → REFACTOR, but DEFERRED on a structural design gap

**GAP = CHAIN (INV12).** This is the explicit Pipe capability gap the topology survey records.
`mcast_send` uses `Semaphore::relay_multicast` (A5′): a **cross-id** broadcast where a **write-once
`valid_sem`** (pinned VALID once, never re-stored) is the broadcast source, and it lands in a **different**
destination semaphore (`recv_sem`) on each receiver. This is exactly INV12 — needed because each receiver's
`recv_sem` doorbell is mutable (`receive()` drives it INVALID then waits VALID), so it cannot also serve as
the broadcast source. Combined with **`linked=true`** on the data write (F4), this block needs:
1. a SECOND, write-once sem id (`valid_sem`) distinct from the doorbell (`recv_sem`), and
2. a linked data + cross-id relay pair with no barrier between.

The current `SenderPipe` / `ReceiverPipe` API uses a **single shared `data_ready` sem id** with A5
same-cell `set_multicast` — it **structurally cannot express** the distinct write-once `valid_sem` +
`relay_multicast` cross-id pattern, nor the linked-pair contract. → **DEFER** (`design-gap`), same verdict
as the `chain_link.hpp` CHAIN family. INV12 documents what the abstraction must grow to cover it
(cross-id relay source + linked-pair F4). Note this is a rotating-role STAR using the relay spelling, not a
literal store-and-forward chain; it would be migratable once the Pipe grows the relay/linked path.
