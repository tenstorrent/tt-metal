# r04-b03-a02: decouple the down phase's two input streams in the reader: the row's activated stream and the column's down-weight stream each advance on their own (receivers ack a stream's next block as soon as that stream's previous block landed and its slot is free; the act sender multicasts its block as soon as its row acks are in), via one non-blocking polling loop instead of the lock-step per-block sequence

## Motivation
The parent (r04-b03-a01) measured the compute's UNPACK-thread waits: in the down window the compute waits on
`cb_in0_down_full` (activated) for **30-58%** of the window (dsv3 32%, kimi-k2.7 47%, kimi-k3 58%, dsv4-flash 50%,
dsv4-pro 50%, mistral 34%, mm3 30%, gpt-oss 39%), plus 3-11% more on the down weights. The matmul itself runs at
21-25 cycles per tile-mult with no data wait.

The reader's down loop is lock-step. Per block kb every core: reserve both slots -> ack the column's down-weight
sender AND the row's act sender kb -> (sender duties) -> wait act_valid(kb) -> wait in1_valid(kb) -> next kb.
So:
- a receiver's ack for act(kb+1) waits on the arrival of the DOWN WEIGHTS of kb (a column event): a late weight
  block shows up one block later as an activated wait, which is what the parent's tally attributes to "activated";
- the act sender of kb+1 can only multicast after it has itself received both streams of kb;
- the column's weight ack for kb+1 waits on the row's activated kb.

Every down block is therefore a grid-wide barrier coupling one row stream with one column stream. r04-b02-a01
(best, 1.6275) showed the down weight delivery latency matters (down read look-ahead: DN -4..-30 µs mean core);
this node attacks the coupling, not the read timing.

## Mechanism
Reader only (`kernels/dataflow/unified_routed_expert_ffn_reader.cpp`), phase-4 loop rewritten as a polling loop
over two independent streams, each with "at most one outstanding ack" per core:
- act (row) stream: `act_rx` blocks pushed, `act_ack` blocks offered. Receiver of block j: ack its sender
  (reset act_valid first) once block j-1 is pushed and one cb_in0_down_full slot is reservable; push when
  act_valid == 1. Own block (j == my_nt_d): multicast as soon as block j-1 is pushed, a slot is free,
  cb_activated holds the block and all GRID_X-1 row acks are in (same INCL_SRC linked multicast, valid sem,
  write barrier as today).
- weight (column) stream: same for cb_in1_down. The sender reserves its slot and releases the DOWN_SPLIT writer
  as soon as a slot frees (as today at the loop top), then, once the column acks are in, does exactly today's
  read -> barrier -> down_done wait -> multicast -> valid sequence.
Block order, slot addresses, bytes, semaphores and the compute kernel are unchanged, so outputs are bit-identical.
Also: the parent's profiler-only matmul probe (`re_probe_gu`) is compiled out (`RE_PROBE` define, off) because
it delays the first gate/up by 1-2 µs; its RE-UW tallies stay to judge this node.

## Why this is not a repeat
- r04-b04-a01 decoupled the x / weight acks in GATE/UP (neutral): there the next block's senders were not ready
  anyway (owner tilize, serial writer up read). In DOWN the act senders' data (cb_activated) is ready for the
  whole down phase, so an early ack can actually be served; and here the act sender's own send is decoupled too,
  not just the acks.
- r04-b02-a01 / r03-b04-a04 changed when the down-weight DRAM read starts; this node keeps the read order
  (after the column acks) and changes only the handshake coupling. The two should stack.
- r02-b04-a02 moved the down weights to the writer (worse); r02-b02-a03 deepened the down-weight ring.

## Expected effect and risk
- Expect RE-DN mean core -5..-15% where the activated wait is large (kimi-k3, dsv4-flash, dsv4-pro, kimi-k2.7,
  gpt-oss); RE-UW activated share should fall and some of it may move to the down-weight share on DRAM-bound
  shapes (dsv3/glm/k2.7), where the net may be ~0. Geomean +0.5..+2%.
- Risks: a hang if the stream bookkeeping is wrong (e.g. an ack before the previous block's valid was seen, or a
  multicast into a slot whose write pointer differs between cores). The valid semaphores stay binary because each
  core still has at most one outstanding ack per stream and the sender's write pointer equals the receivers'
  (all have pushed blocks 0..j-1). Reader code grows ~0.5-1 KB; the probe removal frees TRISC space.
- Tell from the eval: hang / PCC fail = bookkeeping bug; RE-UW DN activated share vs parent.
