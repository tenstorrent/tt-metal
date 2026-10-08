# r01-b03-a04: split the output drain across both NoCs: BRISC writes even CB positions on NoC0, the idle reader (NCRISC) writes odd positions on NoC1

## Motivation
Parent r01-b03-a03 (1.1313; k=4 column split, 80 workers, bank-rotated slices) is drain-bound. Per-core medians
for h7168 (`reports/r01-b03-a03`, all 4 chips, measured calls 3-12, µs relative to the last W_AGWAIT end;
/tmp/r01b03a04/percore.py):

```
W_DRAIN end - AG     x=1    2    3    4    5    6    7   10   11   12   13   14   15
y=2                  3.6  3.7  3.9  4.1  4.4  4.7  5.6  5.8  6.2  3.9  6.4  7.3  7.5
y=5                  5.5  5.6  5.9  6.1  6.4  7.5  7.9  7.9  8.7  8.9  9.3  9.6  9.7
y=8                  6.5  6.6  7.0  7.4  7.6  8.1  8.6  8.8  9.6  9.7 10.0 10.2 10.2
TRISC end - AG       3.3 ... 5.2 everywhere (flat-ish)
```
Compute finishes AG+3.3-5.2 µs on every core, but the drain ends anywhere from AG+3.6 (top-left) to AG+10.2
(bottom-right), a smooth gradient in x and y. The parent's reflection showed the drain is not DRAM-bank bound
(bank de-phasing cut per-step bank load 2-8x and the drain tail moved only 0.4-0.6 µs). A gradient in core position
is the signature of NoC0 write-path congestion: all 80 BRISC writers inject 2 KB tiles on NoC0 (routes +x then +y,
torus), so far cores share more loaded links. NoC1 is unused during the drain: the reader (NCRISC, NoC1) finishes its
input + gamma reads before the AG and then sits idle until the kernel ends.

## Mechanism
Only for the plain column-split layout (`col_split_ok && !streaming_low_l1 && !block_major_post`, which already
implies one tile-row per worker core and a resident, whole-row output_cb that never wraps):
- `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: new trailing CT args after the recip accessor:
  `split_drain`, `output_cb`, `drain_sem_id`, output TensorAccessorArgs; reader common arg 6 = output addr.
  After its reads, when `split_drain`, the reader waits on output_cb cumulatively per block (it never pops),
  NoC1-writes the ODD CB positions p of the row to the same rotated output column the writer uses
  (`tile_row*row_stride + col_start + (p+col_rot)%slice`), flushes, then sets a local L1 semaphore `drain_sem` = 1
  and finally does a write barrier.
- `device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: new trailing CT args `split_drain`, `drain_sem_id`.
  When set, W_DRAIN waits cumulatively per block, writes only EVEN positions on NoC0, doesn't pop per block;
  after its own flush it waits for `drain_sem == 1`, resets it to 0 (trace replay), and pops the whole padded row.
  (BRISC must not pop before NCRISC finishes waiting, since pops move tiles_acked under NCRISC's cumulative wait.)
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`: compute `split_drain`, create `drain_sem` on the
  worker cores, append the CT args to both kernels, add output_addr as reader common arg 6 and refresh it in
  `override_runtime_arguments`.
Compute is unchanged. Every other layout keeps the old single-RISC drain (split_drain=0 -> reader skips the drain).

## Why this is not a repeat
- r01-b02-a02 tried one flush per row (drain depth): neutral. Showed the tail is contention, not flush serialization.
- r01-b03-a03 (parent) de-phased DRAM banks: fixed the read hot spot, drain only -0.4..0.6 µs: not bank bound.
- No node has moved output traffic onto NoC1. This is the parent's #1 recommendation and also b04-a01/b04-a03's #2.

## Expected effect and risk
If the drain tail is NoC0 link congestion, halving NoC0 write traffic and putting the other half on NoC1 (opposite
routing direction, so the far-core gradient should flatten) should pull the slowest core's drain end from ~AG+10 µs
toward ~AG+6 µs on h7168/h6144 (-2..-4 µs), less on h3584/h4096 (-1..-2 µs). Rough score 1.20-1.30.
Risks: hang if the reader's cumulative wait or the drain_sem handshake is wrong (fail_class=hang); wrong output
columns -> accuracy_fail; if NoC1 routes are equally congested the gain is smaller (judge with the same per-core
table: drain end - AG by x,y). The read-side CT-arg append must keep the existing accessor offsets.
