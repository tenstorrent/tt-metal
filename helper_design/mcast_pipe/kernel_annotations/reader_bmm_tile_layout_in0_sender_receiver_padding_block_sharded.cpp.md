# reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp — annotation

Role: **THE GENERALITY STRESS TEST.** A SINGLE kernel binary that is, per-block, EITHER the sender OR a receiver, with loopback when the sender is in the receiver rect, and which dispatches across 4 distinct mcast shapes by compile-time predicate. Object API.

## Compile-time axes that reshape the block
- `core_in_in0_receiver_mcast_grid` (cta1): is this core inside the receiver rectangle? (controls loopback + INVALID reset + receiver-wait participation)
- `core_has_output_block_work` (cta0): does it consume, or only forward? (controls extra `pop_front`)
- `extract_shard_sub_blocks` (derived): mcast from-and-to the SAME cb_in0 (must keep all rect cores participating) vs from a different CB.
- `in0_mcast_num_cores == 1`: degenerate single-core → unicast or compiled-out flag.

## Fork signature (multi-valued within one file)
- **F1**: FLUSH, conditional. `noc.async_writes_flushed()` (L334) guarded so it is skipped only when `core_in_in0_receiver_mcast_grid && num_cores==1`. The L327-332 comment is the clearest H-statement in the group: flag mcast reads `receiver_sem`'s L1 as source; without flush the CPU overwrites it to INVALID before the NoC reads VALID → receivers hang. Final `async_write_barrier()` L360.
- **F2**: LEVEL FLAG. `receiver_sem.set(VALID)` pre-loop (L103); per-iter `set(INVALID)` for rect cores (L141); sender re-sets `set(VALID)` before flag mcast (L286, L314); receivers `wait(VALID)` (L343). R→S side: `sender_sem.wait(num_dests or num_dests-1)` (L224/L227) + `set(0)` reset (L229); receivers `sem.up(+1)` (L338).
- **F3**: ALL THREE PATHS in one block:
  - sender ∈ rect, extract_shard_sub_blocks → EXCLUDE_SRC `num_cores-1`, mcast same CB (L240-251), guarded by `num_cores>1` (else self already has data).
  - sender ∈ rect, different CB, num_cores==1 → **plain unicast** `noc.async_write` (L259-266).
  - sender ∈ rect, different CB, num_cores>1 → **INCLUDE_SRC loopback** (L270-281) + INCLUDE_SRC flag mcast (L288).
  - sender ∉ rect → EXCLUDE_SRC `num_cores` (L300-311) + plain flag mcast (L315).
- **KNOB pre_handshake**: YES. `sender_sem.wait` precedes data mcast (L224/227 before L232+).

## Protocol steps
- L68-75: ctors; `sender_sem` cta9, `receiver_sem` cta10.
- L77-102: precompute `remote_sender_noc_x/y[]` — per-block sender coordinates (receivers must know WHICH core to signal back since the sender rotates by `block_id`). [invariant: multi-sender rotation]
- L103: **invariant** `receiver_sem.set(VALID)`.
- L105-107: `cb_in2.reserve_back` whole batch of sharded source; `in0_tensor_shard_read_addr`.
- Per block (L125): `block_id = block / num_blocks_per_shard`; **the sender for this block is the core whose `sender_id == block_id`** (rotating sender).
  - L133: `cb_in0.reserve_back`.
  - L139-142: if rect core → **receiver arm** `receiver_sem.set(INVALID)`.
  - L144 `if (block_id == sender_id)` → **SENDER arm**: extract/copy sub-blocks (L148-217, self-read via UnicastEndpoint + pad) = sender-fill; **R→S wait+reset** (L224-229); **data-mcast** (F3 branches L232-322); **flag-mcast** + `set(VALID)`; **flush** (L333-335).
  - L336 `else if rect core` → **RECEIVER arm**: `sender_sem.up(remote_sender_noc_x/y[block_id],1)` (L338) = receiver-signal-back to the rotating sender.
  - L341-344: rect cores **receiver-wait** `receiver_sem.wait(VALID)`.
  - L345: `cb_in0.push_back`.
  - L351-353: if `!core_has_output_block_work` → `cb_in0.pop_front` immediately (keep lockstep CB ptrs for send-only cores). [invariant: lockstep CB across send-only and compute cores]
- L360: final `async_write_barrier()`.

## HOLEs
- L77-102 rotating-sender coordinate table — not a primitive call, but it is load-bearing protocol state the Pipe API must accommodate: **the sender identity changes per block** (block-sharded inner-dim split across cores). A two-sided Pipe where one side is always sender and the other always receiver does NOT fit; here a core is sender for block b and receiver for block b'.
- L256-266 the `num_cores==1` → unicast fallback: the "mcast" degenerates to `async_write`. Pipe `send()` must internally pick unicast when rect is a single core (else hang, per L237 comment).

## Generality verdict
This is the kernel that breaks the "two objects, one sends one receives" model: it is **one object that is both, selected per-iteration by `block_id == sender_id`**, with loopback, with rotating sender identity, with a unicast degenerate case, and with a send-only-core CB lockstep requirement. See migration_audit: `refactor` (high cost) or `defer`.
