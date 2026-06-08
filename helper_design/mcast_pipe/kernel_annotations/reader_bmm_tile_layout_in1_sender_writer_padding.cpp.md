# reader_bmm_tile_layout_in1_sender_writer_padding.cpp — annotation

Role: **SENDER half** (reader portion) + an unrelated WRITER portion in the same file. The block appears TWICE in the reader: once for in1 data, once for in3/bias data. Object API.

## Fork signature (identical for both in1 and in3 invocations)
- **F1**: FLUSH. BH-only `async_writes_flushed()` between data and flag (L450, L581); final `async_write_barrier()` at L764. Same in-order-NoC reliance comment (L443-445).
- **F2**: LEVEL FLAG. `receiver_sem` pre-set VALID (L212). `sender_sem.wait(num_dests)` + `set(0)` reset (L424-425, L557-558).
- **F3**: EXCLUDE_SRC (default `async_write_multicast`, num_dests = num_cores, "must not include source" comment L429). Self-fill via local reads.
- **KNOB pre_handshake**: YES — `sender_sem.wait` precedes the data mcast (L424 before L430; L557 before L563).

## Protocol steps
- L177-181: object ctors — `Noc`, `cb_in1`, `cb_out`, `sender_sem` (cta 10), `receiver_sem` (cta 11).
- L212: **invariant** `receiver_sem.set(VALID)`.
- **in1 block** (per inner-dim block):
  - L285/341/374-380: `cb_in1.reserve_back` + capture `in1_start_address` (source). [sender-fill bookkeeping]
  - L300-417: **sender-fill** — three variants by build flag (DRAM-width-sharded set_async_read_state/async_read_with_state, DRAM-height-sharded async_read, interleaved async_read) all ending in `noc.async_read_barrier()`. Source ready.
  - L424-425: **R→S wait + reset** `sender_sem.wait(in1_mcast_num_dests)` / `set(0)`.
  - L430-441: **data-mcast** `noc.async_write_multicast(...)` EXCLUDE_SRC.
  - L446-451: **flush** (BH).
  - L455-461: **flag-mcast** `receiver_sem.set_multicast(...)`.
  - L465: `cb_in1.push_back`.
- **in3/bias block** (FUSE_BIAS, first batch or multi-block, L476-600): structurally identical second instance of the block — fill (L491-549) → R→S wait+reset (L557-558) → data-mcast (L563-574) → flush (L578-582) → flag-mcast (L586-592) → `cb_in3.push_back` (L595).
- L603-732: **WRITER portion** (out tensor stores). Uses `noc.async_write` + `noc.async_write_barrier()` (L670, L716) + `cb_out` pops. NOT part of the Pipe block — plain unicast output drain. (incidental barriers, not the family.)
- L764: final `noc.async_write_barrier()`.

## SKIP_MCAST
Both block instances guarded by `#ifndef SKIP_MCAST`; compiled out for single-core.

## HOLEs
- None for the mcast block. The writer half (L603-732) is out-of-family and should NOT be wrapped by the Pipe — it is a separate concern co-resident in the file (migration cost note: this kernel mixes reader-sender + writer in one file, so a Pipe migration only touches the reader half).
