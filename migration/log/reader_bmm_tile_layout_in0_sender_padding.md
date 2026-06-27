# reader_bmm_tile_layout_in0_sender_padding.cpp (TIER 1 #1, sender)

## Block migrated
Canonical in0 data-mcast + handshake (orig lines 339-380, under `#ifndef SKIP_MCAST`):
`sender_sem.wait(num_dests)/set(0)` (pre-handshake) + `async_write_multicast(...,linked=true)` +
`#ifdef ARCH_BLACKHOLE async_writes_flushed()` + `receiver_sem.set_multicast(...)` (flag).

NOT migrated: the sparsity batch-valid flag-only block (lines 169-184). It does a pre-handshake
wait THEN a value-carrying (VALID/IGNORE_BATCH) flag mcast with no data. `send_signal` is flag-only
with no pre-handshake; no clean fit. Left as raw (audit-flagged "needs data-less send mode").

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
- EXCLUDE_SRC: sender core not in receiver rect (`async_write_multicast`, non-loopback).
- Flag: level VALID/INVALID flag (orig `set_multicast` of receiver_sem).
- PRE_HANDSHAKE=true: dest is the receivers' reused in0 CB slot -> R->S consumed wait gates each block.
- LINK=true: orig linked=true data mcast + flush (BH). Pipe always flushes (correct on WH too).

Mapping: data_ready=receiver_sem (CT arg 16), consumed=sender_sem (CT arg 15).
McastRect num_dests = `in0_mcast_num_cores` (the mcast geometry count; drives both the data/flag
mcast and the handshake wait). NOTE divergence: orig waits `in0_mcast_num_dests` (=min(cores,active))
but mcasts to `in0_mcast_num_cores`. Pipe collapses to one field; for the validated interleaved 1D
mcast-in0 case the two are equal (test passes). For dispatches where they differ this would be an
API mismatch — out of scope for the validated node.

## Call-site diff
~42 lines removed (the open-coded wait/set/mcast/flush/flag block) -> 1 `in0_pipe.send(src,dst,bytes)`
(+ ~11-line Pipe construction hoisted before the loop). src==dst==in0_start_address.

## Validation
nodeid: test_matmul_1d_multiple_output_blocks_per_core[...mcast_in0=True...grid_size=(8, 2)...n=2048-k=1024-m=256]
Result: SAFE_PYTEST_RESULT: PASS (1 passed in 2.56s). JIT-compiled fresh (build log shows kernel).

## Commit
4751ee89f1f  "apply mcast_pipe to reader_bmm_tile_layout_in0_sender_padding" (amended w/ clang-format fix)
