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

## How it works, in one paragraph

Per timestep a core holds `K_j, V_j` (and `V_j^T`, transposed once per
residency interval) and receives a packet: `Q_i` first, then the state
`(O_i^T, m_i, l_i)` after the previous consumer has updated it. It forms
`S^T = K Q^T` for the block, reduces the column maximum, `m_new = max(m_old,
colmax)`, `r = exp(a (m_old - m_new))`, `P^T = exp(a (S^T - m_new))`,
`l_new = r l_old + colsum P^T`, `O^T <- r O^T + V^T P^T`, all in Float32 in
the destination registers (the scores go through the source registers'
19 bits once, on the way into the maximum; `m` is a maximum of those
rounded scores and so exact; the exponential is the backward's polynomial,
2.9e-6). At a row's last visit the consumer finishes it -- `O = (O^T / l)^T`
in bfloat16, `lse = a m + ln l` -- and the write kernel stores it. A row's
first visit starts from nothing; between streaks the raw state spills to
the scratch tensors and the endpoint words of the backward order the
reload after the spill.

## What to expect

Accuracy first, since that is what this forward buys today. Against a
Float32 host reference from bfloat16 inputs (`CyclicSdpaFwTest`, one chip):

| | `cyclic_sdpa_fw` | `sdpa_fw` | ttnn's kernel with lse |
|---|---|---|---|
| output, relative RMS | 1.7e-3 to 2.0e-3 | 1.8e-3 to 3.3e-3 | 2.5e-2 to 3.2e-2 |
| lse, max absolute error | 4e-4 to 7e-4, independent of N | 1.7e-3 at N = 64, 5e-3 at 1024, 1e-2 at 2048 | 2.6e-2 |

The output's error is the bfloat16 rounding of the result; the lse's is the
exponential's and does not grow with the sequence, because every rescale
factor is exact.

Speed, one p150, median of five (`CyclicSdpaBwTimingTest.DISABLED_CompareForwardsWithTtnn`
and `CyclicSdpaFwTimingTest.DISABLED_TimeTheForward`), `Bt = 4`:

| shape | `sdpa_fw` | `cyclic_sdpa_fw` | ttnn with lse (chunk 256) |
|---|---|---|---|
| 4/4 heads, 4096, d 64, causal | 1.01 ms | 1.90 ms | 0.54 ms |
| 20/10 heads, 5632, d 64, causal | 7.38 ms | 6.97 ms (11.6 TFLOP/s) | 1.28 ms (63 TFLOP/s) |
| 32/8 heads, 5632, d 128, causal | 22.3 ms | 19.9 ms (13.1 TFLOP/s) | 3.57 ms (73 TFLOP/s) |

Eight chips, zigzag layout, direct shifts, the cyclic backward, forward
time of one ring step (`LoudboxRingSDPATest.DISABLED_CompareStepTimes`):

| shape | `sdpa_fw` forward | cyclic forward | ttnn forward |
|---|---|---|---|
| 4/4 heads, 4096 rows/chip, d 64 | 10.3 ms | 21.7 ms | 28.8 ms |
| 20/10 heads, 5632 rows/chip, d 64 | 66.8 ms | 61.4 ms (-8%) | 30.9 ms |
| 32/8 heads, 5632 rows/chip, d 128 | 191 ms | 143 ms (-25%) | 40.6 ms |

In training (`training_shakespeare_nanollama3_cp8_char.yaml`, 20/10
heads, 45056 tokens over 8 chips, zigzag, direct shifts, the cyclic
backward in place, 40 steps): the losses match the ttnn-forward run step
for step (2.7676 at step 10, 2.5176 at 20, 2.4922 at 30, 2.4688 against
2.4707 at 40) at 534 ms a step against 412 ms with `cp_forward: ttnn`
(1875 ms with the original two-pass ring).

So: as accurate as the ring's driver could want, a little faster than
`sdpa_fw` at the model shapes, and two to three and a half times slower
than ttnn's kernel. The kernel is the first version; where its time goes
at the 20/10 shape (10 ms before the exponential was replaced): the
exponential 4.3 ms (now ~1.2), the statistics passes 2.6 ms, everything
else -- the two matmuls, the packs and the relay -- 2.7 ms against ttnn's
1.3 ms for the whole launch. The backward kernel reaches 44 TFLOP/s per
chip on the same relay with the same block height; what it does that this
kernel does not yet: the pack thread's SFPU overlapped with the FPU, groups
of two key tiles in a half-synchronised destination file, no per-stage
round trips of the statistics. `TTML_CYCLIC_FW_EXPERIMENT=NO_EXP,NO_STATS,...`
switches stages off for timing (the results are then wrong).

## Testing

Single chip: `ttml_tests --gtest_filter='CyclicSdpaFwTest.*'` (8 tests,
about a minute, against the host reference and `sdpa_fw`; every block
height, causal and dense, grouped heads, batches, capped groups, 64 cores).
Loudbox: `LoudboxRingSDPATest.CyclicForward*` (4 tests, 2x4 mesh by default,
`TTML_LOUDBOX_RING8=1` for the 1x8 ring) run the ring forward and backward
end to end against a host reference, graded like `sdpa_fw`. Run from the
tt-metal root with `TT_METAL_HOME` and `TT_METAL_RUNTIME_ROOT` set.
