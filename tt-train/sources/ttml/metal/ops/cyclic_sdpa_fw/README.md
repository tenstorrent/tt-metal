# `cyclic_sdpa_fw` -- the cyclic attention forward, for users

The forward pass of scaled-dot-product attention (`O`, and the per-row
log-sum-exp the backward consumes) as one fused kernel over the *cyclic
schedule* of `cyclic_sdpa_bw`: every core owns one block of keys for the
whole launch, and the query blocks travel from core to core in packets,
carrying their online-softmax state (`O`, running max `m`, running sum `l`)
with them. Nothing is read twice from DRAM, the statistics and the
accumulator stay in Float32 from the first key to the last, and there is no
chip-wide barrier. It is tt-flash-attn's Algorithm 8, built there first
(`forward.py`, `forward_relay.py`, `docs/forward.md`) and ported here.

| entry point | what it is for |
|---|---|
| `ttml::metal::cyclic_sdpa_fw` | one chip: `(output, intermediates)` from `Q, K, V` |
| `ttml::metal::ring_cyclic_sdpa_fw` | one step of a context-parallel ring forward; drop-in for `ring_ttnn_sdpa_fw` and `ring_sdpa_fw` |

The ring driver `ttml::ops::distributed::ring_attention_sdpa` uses the
second when asked (`forward_kind = Cyclic`); the trainer's config says
`cp_forward: cyclic`.

## What it computes

For each (batch, head) slice, with the scale `a = 1 / sqrt(d)` and the mask
causal or none:

    S = a Q K^T          O = softmax(S) V          lse = log sum exp(S) per row

`lse` is returned as tt-train's intermediates layout, `(B, H, N, 32)`
Float32 with the value in column 0 and zeros elsewhere -- what `sdpa_fw`
returns and what `cyclic_sdpa_bw` and the ring's `ring_softmax_merge` take.

## Inputs and outputs

| tensor | shape | type | notes |
|---|---|---|---|
| `query` | `(B, H, N, d)` | bfloat16 | tile layout, interleaved, on device |
| `key`, `value` | `(B, G, N, d)` | bfloat16 | `G = H`, or a divisor of `H` for grouped-query attention: the query heads of a key head are independent slices here (no shared output), so there is no group cap |
| output (returned) | `(B, H, N, d)` | bfloat16 | the attention output |
| intermediates (returned) | `(B, H, N, 32)` | Float32 | `lse` in column 0 |

Both outputs can be preallocated. The op also allocates two Float32 scratch
tensors of `(B, H, N, d)` and `(B, H, N, 64)` for the state a row spills
between its streaks; the caller never sees them.

The constraints are the backward's: `d` a multiple of 32, `N` a multiple of
`2 * rows_per_block_tiles * 32`, `Causal` or `None`, bfloat16 operands. The
knobs `rows_per_block_tiles`, `mask_type`, `max_groups` and the chunk pairs
(`sequence_chunks`, `row_chunks`, `col_chunks`) mean what they mean in the
backward's README, with one difference: two chunk pairs of one launch may
share a *column* chunk (columns have no outputs) but not a row chunk (both
would finish the same rows; the ring merges such partials from separate
launches).

## How it works, in two paragraphs

Per timestep a core holds `K_j, V_j` (and `V_j^T`, transposed once per
residency interval) and receives a packet: `Q_i` first, then the state
`(O_i^T, m_i, l_i)` after the previous consumer has updated it. It forms
`S^T = K Q^T` for the block and the column maximum of every query tile,
then, per query tile, `P^T = exp(a (S^T - m))`, `l += colsum P^T` and
`O^T += V^T P^T`, all in Float32 in the destination registers (the scores
go through the source registers' 19 bits once, on the way into the maximum
and the exponential, which is the backward's polynomial at 9.5e-5). At a
row's last visit the consumer finishes it -- `O = (O^T / l)^T` in bfloat16,
`lse = a m + ln l` -- and the write kernel stores it. A row's first visit
starts from nothing; between streaks the raw state spills to the scratch
tensors and the endpoint words of the backward order the reload after the
spill.

The running maximum is *lazy*, as in FlashAttention-4: `m` is only a
reference point, and the finished row comes out the same for any reference
as long as `exp(a (S - m))` cannot overflow, so a query tile whose block
maximum stays within 8 (in units of the scaled scores; e^8 in Float32 is
nowhere near overflow) of the `m` it carries keeps that `m`. No rescale
factor is computed, the block sum and the products are added onto `l` and
`O^T` where they lie by the packer's L1 accumulate, and nothing of the state
passes through the destination registers. Only when a tile's maximum grows
past the threshold -- the first visits of a row, then rarely -- does it
take the exact path: `m_new = m_old + max(colmax - m_old, 0)`, `r =
exp(-a max(colmax - m_old, 0))`, `l = r l_old + colsum`, `O^T = r O^T + V^T
P^T`, every factor exact. The check is the FPU's `colmax - m` compared on
the unpack thread and broadcast to the other two through the mailboxes.
`TTML_CYCLIC_FW_EXPERIMENT=NO_LAZY` rescales every timestep (the same
results, slower); `LAZY_TAU=<x>` sets the threshold.

## What to expect

Accuracy first, since that is what this forward buys. Against a Float32
host reference from bfloat16 inputs (`CyclicSdpaFwTest`, one chip):

| | `cyclic_sdpa_fw` | `sdpa_fw` | ttnn's kernel with lse |
|---|---|---|---|
| output, relative RMS | 1.6e-3 to 1.8e-3 | 1.8e-3 to 3.3e-3 | 2.5e-2 to 3.2e-2 |
| lse, max absolute error | 4e-4 to 7e-4, independent of N | 1.7e-3 at N = 64, 5e-3 at 1024, 1e-2 at 2048 | 2.6e-2 |

The output's error is the bfloat16 rounding of the result; the lse's is the
19-bit rounding of the scores on their way into the exponential and does
not grow with the sequence, because every rescale factor is exact and the
lazy rescale changes nothing but the reference point.

Speed, one p150, median of five (`CyclicSdpaBwTimingTest.DISABLED_CompareForwardsWithTtnn`
and `CyclicSdpaFwTimingTest.DISABLED_TimeTheForward`), `Bt = 4`; the
cyclic column varies by about 3% from run to run:

| shape | `sdpa_fw` | `cyclic_sdpa_fw` | ttnn with lse (chunk 256) |
|---|---|---|---|
| 4/4 heads, 4096, d 64, causal | 1.01 ms | 0.77 ms | 0.54 ms |
| 20/10 heads, 5632, d 64, causal | 7.38 ms | 3.75 ms (22 TFLOP/s) | 1.28 ms (63 TFLOP/s) |
| 32/8 heads, 5632, d 128, causal | 22.3 ms | 8.1 ms (32 TFLOP/s) | 3.57 ms (73 TFLOP/s) |

Eight chips, zigzag layout, direct shifts, the cyclic backward, forward
time of one ring step (`LoudboxRingSDPATest.DISABLED_CompareStepTimes`,
`TTML_LOUDBOX_RING8=1`):

| shape | `sdpa_fw` forward | cyclic forward | ttnn forward |
|---|---|---|---|
| 4/4 heads, 4096 rows/chip, d 64 | 10.2 ms | 12.7 ms | 29.9 ms |
| 20/10 heads, 5632 rows/chip, d 64 | 66.8 ms | 43.3 ms (-35%) | 32.9 ms |
| 32/8 heads, 5632 rows/chip, d 128 | 191 ms | 93 ms (-51%) | 40.7 ms |

In training (`training_shakespeare_nanollama3_cp8_char.yaml`, 20/10
heads, 45056 tokens over 8 chips, zigzag, direct shifts, the cyclic
backward in place, 40 steps): the losses match the ttnn-forward run step
for step (2.7676 at step 10, 2.5176 at 20, 2.4922 at 30, 2.4688 against
2.4707 at 40) at 462 ms a step against 412 ms with `cp_forward: ttnn`
and 534 ms with this kernel's first version (1875 ms with the original
two-pass ring); with the fused ring shifts (`ring_shift_fused`) the step
is 392 ms, against 354 ms with `cp_forward: ttnn` and the same shifts. At
Llama 8B's attention shape (32/8 heads, d 128, 20 steps) the step is 596
ms with this forward and 526 ms with ttnn's, both with fused shifts.

So: as accurate as the ring's driver could want, half of `sdpa_fw`'s time
at the model shapes, and 2.3 to 3 times slower than ttnn's kernel on
one chip (1.3 to 2.3 in the ring, where the relay's overlap helps). Where
the time goes at the 20/10 shape, per timestep of a core (15.5 us; the
device profile, `CyclicSdpaFwProfileTest`): the exponential 4 us on the
vector unit, the two matmuls about 7 us on the FPU at HiFi3 (HiFi2 would
save a fifth but costs the accuracy edge: lse 1.2e-3, output 3.4e-3),
the block maximum and its check 3 us, the rest synchronisation. The
probabilities, sums and output products of one query tile run while the
next tile's exponential runs; the scores of the next timestep do not, and
that overlap -- worth perhaps a fifth -- is the next item.
`TTML_CYCLIC_FW_EXPERIMENT=NO_EXP,NO_STATS,...` switches stages off for
timing (the results are then wrong); `NO_LAZY`, `LAZY_TAU=<x>` and
`EXP_GUARD=<0|1|2>` are exact-result variants (0 is not: no underflow
guard).

## The fast variant: what the design reaches without the exact statistics

`cyclic_sdpa_fw_fast` (`CyclicSDPAForwardParams::fast`, the kernel
`cyclic_sdpa_fw_fast_compute.cpp`, a copy of the exact one) answers one
question: how far the schedule itself is from ttnn's kernel once it gives up
what the accuracy costs. It keeps the relay and the Float32 packet but runs
bf16 destination registers (16 tiles instead of 8), two matmul phases, and
so blocks of 8 tiles (256 rows), which the Float32 kernel's registers cannot
hold. The exact op is unchanged; the variant is selected by name.

One chip, 20/10 heads, 5632 rows, d 64, causal (ttnn with lse: 1.28 ms):

| | time | output RMS | lse max error |
|---|---|---|---|
| the exact kernel, blocks of 4 | 3.75 ms | 1.7e-3 | 4.9e-4 |
| fast, blocks of 4 | 3.66 ms | 6.8e-3 | 4.9e-2 |
| fast, blocks of 8 | 2.51 ms (32 TFLOP/s) | 7.1e-3 | 6 to 9e-2 |
| fast, blocks of 8, exponential removed (timing only) | 2.13 ms | | |

So the registers alone buy nothing; the block height they allow buys a
third; and even a free exponential would leave the design at 1.7x ttnn.
What remains is on the FPU side and in the timestep's fixed passes: the
matmuls stream one operand tile per tile product and run at about 50
cycles a phase where the FPU could do 20 (a block that reuses both
operands would help, and needs the registers the exponential's half of DST
takes), the block maximum's 64 reductions a timestep, and the relay's
synchronisation. ttnn's kernel has none of these because it has no relay:
it re-reads K and V from DRAM per query chunk, which is the trade the
schedule was built to avoid, and pays for it in bandwidth rather than in
per-core time. The variant's accuracy is ttnn's on the output (7e-3 against
2.5e-2, better) and a little worse on the lse (6e-2 against 2.6e-2), so it
is not a forward to train with; it is the measurement. Tests: the forward's
suite with `TTML_CYCLIC_FW_FAST=1` (looser lse grading) and
`CyclicSdpaFwTest.FastTallBlocks`.

## Testing

Single chip: `ttml_tests --gtest_filter='CyclicSdpaFwTest.*'` (8 tests,
about a minute, against the host reference and `sdpa_fw`; every block
height, causal and dense, grouped heads, batches, capped groups, 64 cores).
Loudbox: `LoudboxRingSDPATest.CyclicForward*` (4 tests, 2x4 mesh by default,
`TTML_LOUDBOX_RING8=1` for the 1x8 ring) run the ring forward and backward
end to end against a host reference, graded like `sdpa_fw`. Run from the
tt-metal root with `TT_METAL_HOME` and `TT_METAL_RUNTIME_ROOT` set.
