# r03-b04-a04: down-weight sender issues block kb+1's DRAM read (both DOWN_SPLIT halves) right after multicasting block kb, into its own already-free ring slot, instead of after all column receivers ack kb+1

## Motivation
- Parent reflection (r03-b04-a03) found the group tail is mostly in the DOWN phase. In mistral call 7, the y8-9
  group's DN took 130 µs, against 35-83 µs for the other groups doing identical work. Its stated suspect: the down
  loop issues each block's DRAM read only after every column receiver has acked. No read-ahead, so DRAM latency is
  exposed once per down block.
- The reader's phase-4 loop on the column sender (`unified_routed_expert_ffn_reader.cpp`), per down K-block kb:
  1. Reserve its own `cb_in1_down` slot and release the writer's DOWN_SPLIT half (NoC 1).
  2. **Wait for `in1_ready` from all receivers.**
  3. Then issue its own half of the block's DRAM reads (NoC 0), barrier, wait for the writer's half, multicast,
     and send the valid sem.
  4. Act multicast and wait.
- Receivers ack kb+1 only after they have received block kb's weights AND kb's activated multicast. So each block
  pays this chain in series:
  - the act multicast of kb, then the receivers' ack round trip;
  - the reader-half DRAM read of kb+1: ~1.2-2.7k cycles of issue (r02-b02 measured ~43 cycles per 576 B bfp4 tile)
    plus the DRAM latency;
  - the multicast of kb+1.
- The read only touches the sender's OWN ring slot. Gating it on the receivers is unnecessary.
- Evidence the chain sets the DN cadence on the less DRAM-bound shapes:
  - mistral DN is ~50 µs for 11 down blocks, ~4.6 µs per block, against ~2.6 µs of LoFi compute (pcM 3,
    216 tile-mults);
  - kimi-k3 DN runs 61 cycles per tile-mult (r03-b02-a03), the worst of any phase;
  - mistral, gpt-oss and minimax-m3 move only ~214-350 GB/s of weights in total, below the ~430-450 GB/s that the
    DRAM-bound shapes reach.
- r03-b04-a01 found these shapes lose a large fixed cost per expert that does not scale with per_core_M. That is
  the signature of a serial DM chain, not of matmul rate. r03-b02-a03's "stream x instead of weights" idea assumes
  the matmul rate is the limit, but r03-b04-a01's per_core_M fit puts the marginal per-tile-row cost at the LoFi
  ideal. So I did not take that idea.

## Mechanism
Reader kernel only (`kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, phase-4 loop). No host or compute
change.
- On the column weight sender, after block kb's down multicast and valid sem, and before the act multicast / act
  wait:
  - if kb+1 < num_blocks_d and its own `cb_in1_down` has room for two blocks (non-blocking
    `pages_reservable_at_back`: the unpushed block kb plus block kb+1), it starts block kb+1 now;
  - it releases the writer's DOWN_SPLIT half for kb+1 (`down_go`), and the writer is unchanged: it reads slot
    `(down_blk + kb) % IN1_D_SLOTS` when released;
  - it issues its own half of kb+1's reads into the next ring slot (write pointer + one block, wrapping at
    fifo_limit, the same slot the push will move to).
- In iteration kb+1 the sender then only waits for the acks, barriers and multicasts. If the slot was not free yet
  (compute-bound case), it issues the reads at the top of kb+1, still before the ack wait.
- Block 0 of every chunk keeps today's order (acks first, then read). So nothing is read during the activation
  gap, and the per-group DRAM race that sank r02-b02-a03 is not reopened.
- Ring depth, slots, semaphores and data are unchanged. Every block lands in the same slot with the same bytes, so
  the result is bit-identical.

Safety:
- Receivers' slots are written only by the multicast, which still waits for every receiver's ack.
- `in1_ready` acks for kb+1 cannot arrive before the sender's in1_valid for kb, so the wait / set(0) pairing is
  unchanged.
- The pre-issued reads are on `noc_read`. The only barrier before kb+1's multicast is the kb+1 read barrier
  itself.
- No read is left in flight past the down loop: the last block never pre-issues.

## Why this is not a repeat
- r02-b02-a03 (down look-ahead across the activation gap, -1.3%) used a deeper, down-only ring and read Dw-2 blocks
  ahead, including into the activation gap. Groups that left gate/up first grabbed DRAM share, which widened the
  group race.
  - This node keeps the 2-3 slot ring and reads at most one block ahead.
  - It never reads block 0 early (so nothing across the activation gap).
  - It only removes the receiver-ack gate from a read that targets the sender's own free slot.
- r02-b04-a02 (writer-owned down stream, -17%) put all down reads on one NoC. Here both DOWN_SPLIT halves stay on
  their own NoCs.
- r01-b03-a01 / r01-b04-a01 (gate/up read-ahead, root code) were a different phase, on code before UP_SPLIT and
  DOWN_SPLIT. In gate/up the writer-multicast mode already decouples the reads from the receivers' acks.
  Phase 4 is the one weight stream still gated on them.

## Expected effect and risk
- **Expected:**
  - DN per block on mistral, minimax-m3, gpt-oss and dsv4-flash falls by up to the reader-half read time (~1-2 µs of
    ~4.6 µs), i.e. up to ~10-20 µs per expert's DN where the group is not held by aggregate DRAM;
  - kimi-k3 (57 KB down blocks, many experts) may gain a few %;
  - dsv3, glm, kimi-k2.7 and dsv4-pro (DRAM-aggregate-bound) should stay within noise;
  - overall about +0.5-1.5% geomean.
- **Risks:**
  - the faster down streams take DRAM share from other groups and widen the tail on 1-expert-per-group shapes,
    as in r02-b02-a03 (visible as max - mean growing in the RE-DN zones);
  - a slot / sequence mistake would give wrong down weights (accuracy_fail) or a hang on the down_go / down_done
    sequence.
- **Expected numerics:** bit-identical PCC / mean_abs, which confirms the slot math.
