# r03-b04-a02: take the row-shared x look-ahead DRAM read off the gate/up weight path: tag the x reads with their own NoC transaction id, so the gate-sender's weight barrier and every reader's per-block weight push no longer wait for them, and push the x block as soon as it lands

## Motivation
The few-expert shapes are the furthest from the DRAM floor. At ~450 GB/s:
- mistral 192 µs against a ~133 µs floor;
- gpt-oss 262 against ~150;
- minimax-m3 373 against ~310;
- dsv4-flash 322 against ~265.

Their per-group gate/up stream runs at only ~7-8 GB/s per column sender. For example, mistral needs 14.6 µs per 110 KB gate+up block (r02-b01-a02 RE-GU zone: 117 µs over 8 blocks) against ~6.8 µs of LoFi compute per block at per_core_M 3. Earlier nodes ruled out these causes: DRAM read latency (read-ahead, r01-b03-a01 / r01-b04-a01), ring depth (r01-b04-a03), request VCs (r02-b02-a02) and down-stream ownership (r02-b04-a02). The group-tail lottery (r02-b04-a03) also looks like a gate/up-phase delivery effect: slow groups have a long RE-GU, sometimes only one row of a group is slow, and "that row's own x stream is the suspect".

Reading the reader kernel (`unified_routed_expert_ffn_reader.cpp`, gate/up loop) shows two places where the row-shared x look-ahead read serialises into the weight path. Each one fires on every K-block:

1. **Sender rows.** Every core of a group's first row is the in1 (gate) sender of its column. The x owner of block kb+2 rotates over all 11 columns, so on every K-block one gate sender also issues the x(kb+2) read: pcM x 32 row-major sticks of ~1 KB, e.g. 96 KB at pcM 3, issued at the top of the iteration.
   - Its gate read (16 x 3.4 KB = 55 KB) then ends in a plain `noc_read.async_read_barrier()`. That barrier waits for ALL outstanding NoC-0 reads, so it also waits for the x block.
   - That column's gate data is therefore ready only after ~150 KB instead of ~55 KB have landed. Its writer's multicast (`mcast_go`) is late, its receivers start block kb late, and through the row x multicast and the column ring (2-slot slack) the whole group slips.
   - A different column is hit on every block.
2. **Every row.** After its own work, the owner of x(kb+2) barriers the x read at the end of iteration kb (`x_rm_pending`), and only then reaches step 3, which waits for and pushes this block's x and gate/up.
   - If the 96 KB x read takes longer than the weights for block kb take to arrive, the core's matmul of kb starts late. Under ~400 GB/s of aggregate DRAM load, a per-RISC share is a few GB/s, so the read takes roughly 10-20 µs against a ~13-15 µs block period.
   - This core's slot then frees late, and the column multicast of kb+2, which waits for every receiver, slips with it.
   - Yet the compute needs x(kb+2) only for its tilize before matmul kb+1, a whole block later.

## Mechanism
One file: `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, row-major x path only. The TILE path keeps its behaviour.
- **Separate transaction ids** on the reader's NoC-0 read command buffer, set with the raw sticky `noc_async_read_set_trid`. This is the same pattern moe_fused_swiglu_reader.cpp uses: TXN_ID on Noc::async_read would throttle at 128 outstanding per id, and the sticky form is free.
  - The x look-ahead reads (`issue_x_rm_read`) are tagged X_TRID (15).
  - The gate reads are tagged GATE_TRID (14).
  - The tag returns to 0 after each burst, so every other read (counts, start, bias, down, TILE x) keeps tag 0 and the plain barriers it already uses.
- **The gate barrier waits only for the gate reads**: `noc_async_read_barrier_with_trid(GATE_TRID, 0)` instead of the global barrier.
- **On a gate-sender core, the x(kb+2) read is issued right after its gate reads** rather than before them, so the gate requests reach the DRAM queues first. Non-sender owners issue it at the top of the iteration, as today.
- **The x block is pushed as soon as it lands, not at a fixed blocking point.** A pending x block j (issued in iteration j-2, or in the chunk prologue for j = 1) is pushed into cb_x_rm in one of three ways:
  - non-blocking (`ncrisc_noc_read_with_transaction_id_flushed`) at the poll points of iterations j-2 and j-1: after the acks, after the sender / x-owner work, and inside the step-3 valid-semaphore waits (the waits poll the semaphore value while the x read is still pending);
  - otherwise, with a blocking trid barrier at the END of iteration j-1, after this block's weights and x have been pushed. The compute needs x(j) only for its tilize before matmul j-1, so this is the latest safe point and it never delays the weight push.
- The chunk prologue keeps x(0) blocking, because the compute tilizes it before its first matmul. x(1) becomes pending in the same way.
- Deadlock check: one cb_x_rm slot per core, and a core owns every GRID_X-th block. With GRID_X >= 2, the previous owned block was pushed at least one iteration before the next issue. The deferral is compiled only when GRID_X_NOC >= 2. With 1 column the old blocking push stays.
- **Measurement (NCRISC zones, which do record):**
  - RE-GW around the sender's gate-read trid barrier;
  - RE-XW around the residual blocking x barrier.

  With them, the eval itself shows whether x still stalls the reader and how long the gate wait is.

No numerics change: the same bytes land in the same CB slots in the same order, and only the barrier scope and the push time move. Kernel-only change, so no host rebuild.

## Why this is not a repeat
- r01-b03-a01 / r01-b04-a01 issued the NEXT weight block's DRAM read before the current multicast. Here no weight read moves. The node removes an x read from the weight barrier and from the weight push.
- r01-b03-a02 batched x into 4-block reads, which made x burstier with the same lead time and lost. Here x keeps its size and gets more lead time.
- r01-b01-a04 moved the x pipeline across unit boundaries (neutral). It kept the blocking barrier at the end of the iteration and the shared global read barrier.
- r02-b02-a02 changed request VCs (neutral). r02-b04-a03 suggested this x-stream suspect (its next #3 b: "move the x stick reads off the reader RISC of weight-sender rows"). This node is the first attempt to decouple it, and it does so without moving the reads to another RISC.

## Expected effect and risk
- Shapes where gate/up delivery paces the group should gain: mistral, minimax-m3, dsv4-flash, gpt-oss, and probably dsv3 / glm / kimi-k3. The expected size is roughly the per-block x-read overhang, a few µs per K-block on some cores, which shows up as a shorter RE-GU and a smaller max-vs-median spread.
- Guess: +0.5..+2% geomean. Some of it may be capped where the aggregate DRAM rate (~400-450 GB/s) is already the limit (kimi-k2.7, dsv4-pro, dsv3).
- It will read as neutral if the x reads already land within the block period, i.e. the RE-XW zone shows ~0 and RE-GW is unchanged. That would rule out the x-stream suspect for the lottery, which is still worth knowing.
- Risks:
  - **Accuracy.** A wrong trid accounting would push x before it lands and corrupt the tilize, which would fail PCC on the row-major shapes (all of them).
  - **Hang.** A missed push would leave the compute waiting on cb_x_rm forever, which would show as a hang in the eval.
  - Both are checked by the eval's accuracy gate and timeout.
