# reader_bmm_tile_layout_in0_sender_padding.cpp — annotation

Role: **SENDER half** of the Pipe block (data + flag mcast). Canonical interleaved/sharded in0 sender. Object/wrapper API (`Noc`, `Semaphore<>`, `MulticastEndpoint`, `CoreLocalMem`).

## Fork signature
- **F1 (flush vs barrier)**: FLUSH. Data mcast then flag mcast on same noc/vc/cmd_buf; relies on in-order NoC (lines 362-364). BH-only extra `async_writes_flushed()` between data and flag (line 368). Final `async_write_barrier()` only once at kernel end (line 421).
- **F2 (flag vs counter)**: LEVEL FLAG. `receiver_sem` carries VALID; sender pre-sets VALID (line 142), and the R→S readiness sem (`sender_sem`) is a level checked with `wait(num_dests)` then `set(0)` reset (lines 343-344). Both are reset-style, not monotone counters.
- **F3 (loopback)**: EXCLUDE_SRC. `noc.async_write_multicast(...)` default mode; num_dests = `in0_mcast_num_cores` and comment "num_dests must not include source, since we are NOT really doing a local copy" (line 348). Sender fills its own CB locally (the read loop) — self-fill, not loopback.
- **KNOB pre_handshake**: YES (pre-handshake / dest-reuse). The `sender_sem.wait(in0_mcast_num_dests)` at line 343 is BEFORE the data mcast — sender waits for all receivers to signal "I consumed the previous block / I'm ready" before clobbering the shared source region.

## Protocol steps (line refs)
- L104-107: object construction — `Noc noc`, `CircularBuffer cb_in0`, `sender_sem` (R→S readiness, cta 15), `receiver_sem` (S→R VALID flag, cta 16). [invariant: two-semaphore handshake]
- L142: **invariant establish** — `receiver_sem.set(VALID)` once before loop; the local L1 value that will be mcast as the flag.
- L171-173 (sparsity batch-valid broadcast, get_batch_from_reader path): **R→S wait** `sender_sem.wait(num_dests)` + **reset** `sender_sem.set(0)`; then `receiver_sem.set(VALID/IGNORE_BATCH)` chooses the flag payload; this is a *flag-only* mcast (no data) of a batch-valid signal.
- L174-180: **flag-mcast** `receiver_sem.set_multicast(...)`.
- L181: **flush** `noc.async_writes_flushed()` (mitigates H: source-clobber of the flag L1 value before NoC reads it).
- L183: **reset/restore** `receiver_sem.set(VALID)` back to default after the one-off IGNORE_BATCH case.
- L218: `cb_in0.reserve_back` — CB slot for fresh block.
- L228-229: **sender-fill bookkeeping** capture `in0_start_address = cb_in0.get_write_ptr()` (the mcast source).
- L233-286: **sender-fill** read in0 block tile-by-tile into CB (`noc.async_read`), pad last ktile, `noc.async_read_barrier()` (L286) — source ready.
- L343-344: **R→S wait + reset** `sender_sem.wait(in0_mcast_num_dests)` / `sender_sem.set(0)` (pre_handshake: receivers consumed prior block).
- L347-360: **data-mcast** `noc.async_write_multicast(CoreLocalMem(in0_start_address), mcast_dst, in0_block_size_bytes, in0_mcast_num_cores, ...)`. EXCLUDE_SRC.
- L362-369: **flush** — comment explains same-noc/vc/cmd_buf ordering means no barrier needed between the two mcasts; BH adds `async_writes_flushed()` (source-clobber mitigation).
- L373-379: **flag-mcast** `receiver_sem.set_multicast(...)` — signals receivers the data is VALID.
- L383: `cb_in0.push_back` — hand block to compute.
- L421: **final barrier** `noc.async_write_barrier()` — drain all outstanding writes at kernel end.

## SKIP_MCAST
When `SKIP_MCAST` defined, the entire mcast/handshake block is compiled out (single-core / no receivers); only local fill + push remains. This is the degenerate "Pipe of width 1" → trivially clean.

## HOLEs
- None. Every line maps to fill / data-mcast / flag-mcast / flush / R→S-wait+reset / final-barrier. The sparsity batch-valid mcast (L169-184) is a *second, flag-only* invocation of the block (no data payload) — note for API: the Pipe `send()` must support a flag-only / data-less mode.
