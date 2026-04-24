# KV Chain Forwarding in Single-Chip SDPA

L1→L1 store-and-forward of K and V chunks between cores that share a
(batch, head), so a chunk is read from DRAM once and handed off across
multiple Q workers instead of each worker re-reading it. Reduces DRAM read
pressure on the MLA-shaped regime where `Sq_chunk_t == Sk_chunk_t` is small
and `DHt` is large — K/V are dense and cheap to move core-to-core, but
expensive to re-fetch from DRAM.

This is a port of the ring-joint SDPA chain logic into the single-chip
`SDPAProgramFactory` / `reader_interleaved.cpp` / `sdpa.cpp` path.

## When it runs

Host-side gate:

    chain_enabled = !is_chunked && (!is_causal || (flatten_work && lightweight_causal))

- **Non-causal** — K is walked fully for every Q, so every Q worker for a
  given (batch, head) wants the exact same K/V bytes. Chain them.
- **Flat-work + lightweight causal** — the iter-0 proxy. Chain participants
  over-read past each Q's `q_high`, and the lightweight causal mask zeroes
  softmax columns past the diagonal. The hierarchical causal path keeps its
  tuned per-Q truncated-K (no-chain) code because without the lightweight
  mask there's no way to cheaply discard the extra columns.
- **Paged / chunked** — not supported; paged reads need per-batch page-table
  indirection that doesn't compose with the chain's K address pre-compute.

## Topology

Chains form per `(batch, head)`. If multiple cores are assigned Q chunks of
the same head:

```
injector ─data─▶ receiver ─data─▶ receiver ─data─▶ sink
         ◀sema─          ◀sema─          ◀sema─
```

Data flows L1→L1 along `next_physical_{x,y}`. Credit flows back along
`sender_semaphore_noc_addr` so an upstream core knows the downstream one
has consumed its previous chunk (`sender_wait_count` credits are required
before the next forward).

The host-side `core_chain_info[i]` per-core record carries:
- `participates` / `is_injector` / `is_sink` — role within the chain.
- `batch` / `head` / `q_chunk_start` / `q_chunk_count` — which head this
  core chains on and how many Q slots it owns on that head.
- `prev_physical` / `next_physical` — NOC neighbours.
- `next_core_q_chunks` — how many Q slots the downstream core has. The
  sender only forwards for `min(own_q_slots, next_core_q_chunks)` iterations
  so a heavier sender correctly serves a lighter receiver.
- `use_mcast` / `mcast_num_dests` / `mcast_sender_wait` — set when the
  whole chain qualifies for a one-to-many mcast instead of unicast hops.

### Mixed q_chunk_counts

Unicast chains tolerate mixed per-core `q_chunk_count` by sorting cores in
**descending** order: heavier senders come first, and the per-chunk
`should_forward` guard `(q_iter_local < next_core_q_chunks)` stops the
extra iterations from blocking. Mcast chains require **uniform**
`q_chunk_count` across the whole chain (single `sender_wait` count), so
they're gated on that check in the eligibility pass.

### Mcast eligibility (all-or-nothing)

Mcast flips on only when every multi-core chain in the program passes:
1. All cores on the same physical Y.
2. No non-chain worker inside the `[min_x, max_x]` rectangle on that row.
3. Uniform `q_chunk_count` across the chain.

`mcast_enabled` is a single compile-time flag in the reader (set in the
`sem_args_offset + 3` slot after topology build). If any chain fails, the
whole grid falls back to unicast — avoids per-chain runtime branching.

## Kernel flow (reader)

Two compile-time knobs on the reader:

- `chain_enabled` — is the kernel even linked against the chain code?
- `mcast_enabled` — if chained, is the transport mcast or unicast?

### Setup

Per-core chain record comes in as tail runtime args (gated by host-side
`chain_enabled`). Reader assembles a local `ChainState` with: role flags,
NOC addresses, semaphore pointers, and (on injector-mcast) the mcast base
address + sender wait count. Three semaphores are created host-side and
shared via compile-time IDs:

- `sender_semaphore` — ticked by receivers to credit the upstream sender.
- `receiver_semaphore` — set to VALID by the sender after a data write.
- `valid_semaphore` — local "data ready" signal, always VALID on the
  receiver side so the sender's mcast can set it remotely.

### Per Q-chunk

On each `(nb, nq, q_chunk)` work item, the reader computes:

```cpp
should_forward = chain.participates && !chain.is_sink
              && on_chain_head && (q_iter_local < chain.next_core_q_chunks);
should_receive = chain.participates && !chain.is_injector && on_chain_head;
```

then loops over K chunks. For each K chunk (and, symmetrically, for V):

**If `should_receive`** — wait for the upstream mcast/unicast to land:
```
cb_reserve_back(cb, tiles)
noc_semaphore_set(receiver_sem, INVALID)
noc_semaphore_inc(sender_sem_noc_addr, 1)   // credit upstream
noc_semaphore_wait(receiver_sem, VALID)     // wait for data
cb_push_back(cb, tiles)
```

**Else** — read from DRAM directly into the CB.

**If `should_forward`** — wait for the downstream credit, then initiate the
forward write:
```
noc_semaphore_wait(sender_sem, sender_wait_count)
noc_semaphore_set(sender_sem, 0)
// mcast: linked data write followed immediately by linked semaphore
// mcast — the companion sema mcast MUST follow without an intervening
// noc_async_read_barrier, or the read barrier blocks while the linked
// write waits for its companion and the chain deadlocks.
// unicast: async write, then explicit noc_semaphore_set_remote.
```

The K-forward sequence has a mask read interleaved between the write start
and the final flush-and-signal — that overlaps the mask read with the
outgoing write and fits the injector's read barrier within the write pair.

### K-loop bound

```cpp
if constexpr (proxy_mode == RingProxyMode::Up) {
    k_chunk_end = k_num_chunks / 2;               // UP proxy halves K
} else if (chain_enabled && chain.participates) {
    k_chunk_end = k_num_chunks;                   // stay in lockstep w/ chain
} else {
    k_chunk_end = (q_high_idx + Sk_chunk_t - 1)
                  / Sk_chunk_t;                   // truncate past diagonal
}
```

Chain participants must walk the full K range so injector + receivers stay
aligned chunk-for-chunk; the lightweight causal mask zeroes the softmax
columns past each Q's diagonal. Non-chain causal cores still truncate their
K loop at the diagonal, preserving the untouched hierarchical performance.

## Compute interaction

The compute kernel's `sdpa_inner_loop` mirrors the reader: chain
participants walk the full `k_num_chunks` and skip past-diagonal K chunks
via a CB-pop fast path rather than doing masked matmul/softmax cycles:

```cpp
if (is_causal && k_chunk >= causal_k_limit
    && (sdpa_type == RING || (sdpa_type == STANDARD && is_chain_participant))) {
    cb_wait_front(cb_k_in, k_chunk_tiles);
    cb_wait_front(cb_v_in, v_chunk_tiles);
    cb_pop_front(cb_k_in, k_chunk_tiles);
    cb_pop_front(cb_v_in, v_chunk_tiles);
    continue;
}
```

`is_chain_participant` rides in as a single per-core runtime arg guarded by
the same compile-time `chain_enabled`.

## What this commit changed

This commit is strictly a cleanup of an already-working chain path — no
behavior changes expected, and the 3 proxy tests produce the same math util
numbers as before the refactor (31.0% / 53.0% / 56.8%).

### `SDPA_KV_CHAIN_ENABLED` define → compile-time arg

Before, the host set `defines["SDPA_KV_CHAIN_ENABLED"]` when chains were on,
and the kernel gated the chain-specific runtime-arg reads with
`#if defined(SDPA_KV_CHAIN_ENABLED)`. The preprocessor gate made chain
state invisible to IDE tooling on non-chain builds and forced the two
kernels (reader + compute) to coordinate on a non-typed symbol.

Now both kernels take a single CT bool `chain_enabled`:

- Reader reads it at `chunk_start_idx_args.next_compile_time_args_offset() + 1`
  (right after `proxy_case`).
- Compute reads it at index 34.
- Kernel ifdef blocks become `if constexpr (chain_enabled)`.
- Host drops the define and pushes the bool onto both CT arg vectors.

### `ChainState` struct in the reader

The 24-line dump of zero-initialized chain variables (`is_chain_participant`,
`chain_batch`, `sender_semaphore_addr_ptr`, `mcast_base_noc_addr`, …) is
replaced by a single local `ChainState` struct with default member
initializers. The `if constexpr (chain_enabled)` block populates it; every
downstream forward/receive site reads `chain.sender_sem_ptr`,
`chain.mcast_num_dests`, etc.

### Tightened dead-code elimination

Before, `should_forward` and `should_receive` were computed unconditionally
from runtime chain fields that happened to be zero when chains were off,
relying on the optimizer to prune. Now they're initialized to `false`
outside the `if constexpr (chain_enabled)` block and only reassigned when
chains are on — the compiler can prove the K/V forward/receive bodies are
dead on non-chain builds. The `k_chunk_end` branch short-circuits on
`chain_enabled && chain.participates` so the runtime participates field
isn't even read when chains are off.

### Docs / comments

- Stale comment on the host-side chain topology build removed ("non-causal
  only" no longer accurate — chain also runs on flat-work causal with the
  lightweight mask).
- Reader's commented-out "Semaphore IDs … always present in compile args"
  crumb removed; the CT args now speak for themselves.
