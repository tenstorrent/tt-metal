# r01-b01-a02: pipeline the reader's input row with per-block NoC transaction ids and interleave the gamma reads into the gaps

## Motivation
r01-b01-a01 (best node, 1.1089) moved x*gamma under the AG wait, but its reflection measured two remaining reader
problems at h7168 (chip-relative zones):
- R_INPUT ~7.5 us for 56 tiles: `read_input_pass` barriers every block_size (=4) tiles, so only 4 x 2 KB reads are
  ever in flight. That is latency bound (~0.5 us per 8 KB block, ~15 GB/s per core), not DRAM-bandwidth bound.
  Everything downstream (PRE, stick push, AG, POST, drain) is serialized behind it.
- Gamma is late: its 112 face-row reads are issued only after the last input block's barrier, so cb_weight lands
  ~3 us after the input row (NCRISC end 10.7-12.0 us vs PRE end ~8.6-9.9 us), and the pre-AG x*gamma pass waits
  on it instead of starting right after PRE.
r01-b02-a01's and r01-b04-a01's reflections also point at the latency-bound input read.

## Mechanism
Reader only (`kernels/dataflow/dit_rmsnorm_fused_reader.cpp`), for resident INPUT_FIRST rows (the campaign config):
1. Reserve the whole row in input_cb (it holds 2 rows, never wraps), issue every input block's tile reads up front,
   block b tagged with NoC transaction id 1 + b % 14 (sliding window of 14 blocks for wider rows).
2. Walk the blocks in order: `async_read_barrier<TXN_ID>(trid_b)` then `push_back(block)`, so compute's PRE still
   starts on block 0 as soon as it lands but the rest of the row is already in flight.
3. Between block waits, issue ceil(56/14)-ish pages of the broadcast gamma face-row reads under trid 15; after the
   last input block, issue any remainder, barrier trid 15 and push the whole gamma row (same single push as a01).
4. Reset the read trid to 0 afterwards. Streaming / SPLIT / DEFER_ALL schedules keep the old path.

## Why this is not a repeat
a01 batched the gamma read but still after the input barrier and left the input per-block barriered. No node has
changed the input read depth. b02/b03 split columns across cores (different lever, dispatch-skew problems); b04
is the same compute reorder as a01. This keeps a01's compute and changes only the reader's issue schedule.

## Expected effect and risk
If the read becomes DRAM-bandwidth bound (20 cores x 112 KB = 2.2 MB/chip), R_INPUT at h7168 could drop from
~7.5 us to ~4-5 us and gamma would land with the input, so PRE/AG/POST all start ~2-3 us earlier at the wide
shapes and ~1-1.5 us at the narrow ones. The output drain (DRAM write contention) may absorb part of it.
Risks: a trid/push mismatch would hang (fail_class=hang) or give PCC failure if a block is pushed before its data
lands; per-read TXN_ID setup adds a few cycles of issue overhead; deeper reads from 20 cores could contend in DRAM
so the gain is smaller than the latency math suggests. Judge with the R_INPUT zone end and NCRISC end in the profile.
