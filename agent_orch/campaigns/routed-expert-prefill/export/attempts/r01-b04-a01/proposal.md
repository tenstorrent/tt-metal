# r01-b04-a01: software-pipeline the weight streams — issue the next K-block's DRAM read before multicasting the current one

## Motivation
No prior nodes (round 1, first attempt on every branch). Evidence from the baseline profile
(`reports/baseline_1`) and the code:

* Every shape runs with per_core_M = 1 (counts are 74-272 tokens per expert, i.e. 3-9 tile-rows,
  spread over the 8 M-rows), so each core does exactly one tile-mult per weight tile its column
  sender delivers. Per expert the measured time works out to ~29-40 cycles per tile-mult on every
  bf16-silu shape (DS-V3 33, M3 31, V4-pro 29.5, K3 40), about 2x the ~16-cycle LoFi matmul tile
  rate, while the delivered weight bandwidth is only ~200-265 GB/s against the ~370 GB/s the
  rotating ND-sharded layout is documented to reach. So neither math nor DRAM is saturated: the
  per-K-block pipeline itself is the limit.
* On each column's weight-sender core the two DM RISCs are strictly serial per K-block:
  - gate/up phase, writer (BRISC): `wait up_go -> read up(kb) -> barrier -> wait mcast_go ->
    wait in1_ready -> multicast gate+up(kb) -> flush -> valid`, and only then `read up(kb+1)`.
    Block time on that RISC = T_read(~55 KB) + T_mcast(~110 KB) + handshakes (~3 us on DS-V3),
    about the same as one block of compute, with only 2 slots of slack.
  - down phase, reader (NCRISC): `wait in1_ready -> read down(kb) half -> barrier -> multicast
    down(kb) -> act multicast`, then `read down(kb+1)`. Same read-then-multicast serialisation,
    and the DRAM read even waits for the *receivers'* ready although it only writes the sender's
    own (already reserved) slot.

## Mechanism
Overlap the DRAM read of K-block kb+1 with the multicast of K-block kb on the sender core, using
the slot the double-buffered CB already has; no CB/L1 sizes, formats or math change.

1. `unified_routed_expert_ffn_writer.cpp` (UP_SPLIT + IN1_WRITER_MCAST, the active config): after
   up(kb) lands, while waiting for mcast_go / in1_ready and before the multicast, poll whether the
   local compute has popped block kb-1 (the previous occupant of kb+1's slot) by reading
   cb_in1_up's tiles_acked stream register; as soon as it has, issue up(kb+1)'s DRAM reads into
   the other slot (non-blocking) and skip the up_go wait/read at the top of the next iteration.
   If the slot is not free in time it falls back to the original blocking path.
2. `unified_routed_expert_ffn_reader.cpp` down phase (weight-sender cores): move the in1_ready
   wait after the DRAM read (the read targets the sender's own reserved slot), and after down(kb)
   lands, while waiting for the receivers, poll `pages_reservable_at_back(2 blocks)` and as soon
   as the other slot is free issue down(kb+1)'s reads (and release the writer's NoC-1 half via
   down_go) before multicasting kb. The next iteration then skips the reserve/go/read.

## Why this is not a repeat
First round, no earlier nodes. Within the op's own history (code comments) UP_SPLIT, DOWN_SPLIT and
IN1_WRITER_MCAST split reads and multicasts across RISCs/NoCs, but every RISC that both reads and
multicasts still does them back to back per K-block; this adds the missing cross-block overlap.

## Expected effect and risk
If the sender RISC chain is the bottleneck, ~10-25% on all shapes (largest on the bias-free
bf16-silu shapes with long K loops: DS-V3, Kimi, GLM, V4-pro, M3). Bit-identical math, so PCC /
mean_abs should be unchanged. Risk: a protocol mistake shows up as a hang (eval timeout) or as
wrong weights in a slot (accuracy_fail); the fallback path keeps the original semaphore order, so
a correct implementation cannot deadlock. If the gain is within noise, the bottleneck is
elsewhere (compute-side per-block overhead or receiver coupling), which is itself informative.
