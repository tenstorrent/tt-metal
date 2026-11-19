<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hy4-preview fused decode kernels

Custom kernels for the decode layer of tencent/Hy4-preview on one Blackhole Galaxy (32 chips, 8x4 mesh), first
version. The ttnn-op decode in `../tt/` is the functional path and the comparison; `../PERF.md` has its measured
latency by stage and why it cannot reach the target (8 users x 1M context at 75 tokens/s per user: 13.3 ms per
batched step, 0.171 ms per layer). It issues 200 to 315 programs and 15 to 21 cross-chip collectives per layer per
user. The pieces here replace whole stages of a layer with one program each, built in Python from kernel sources
and launched with `ttnn.generic_op`; kernels are JIT-compiled, so nothing here needs a host C++ change.

The composition tooling and several kernel patterns come from `models/demos/deepseek_v3_b1` (unified kernel
descriptors) and `models/demos/gpt_oss/tt/fused_decode` (non-persistent programs under fast dispatch, replayed as a
trace, row-placed DRAM readers); the LLKs used are in `tt_metal/hw/inc/api/compute/experimental/`.

## Design decisions
1. Programs are non-persistent and run under fast dispatch inside one device trace, so they share the open device
   and the caches with the ttnn prefill.
2. All 32 chips work on the same layer and the 8 users are 8 rows of one pass (`[8, 32]` row tiles), so that every
   weight byte is read from DRAM once per step.
3. The 4 fp32 iHC residual streams of the 8 rows live whole on one root chip: pre-mix, post-mix and the norm
   statistic are chip-local and need no collective. What crosses chips is the collapsed sublayer input (broadcast
   out) and the sublayer output (reduced back).
4. Decode reads the prefill KV storage in place: packed rows at a 704-byte stride (512 FP8 latent, 4 fp32 scales,
   64 BF16 RoPE values) in 128-token block-cyclic stripes over the 32 chips, and the BFP8 index keys.
5. Attention never moves rows between chips: each chip scans its own stripe of index keys, keeps its own share of
   the global top-2048, reads its own rows from DRAM by id, attends in latent space with all 64 heads, and the
   partial results merge across chips by log-sum-exp, the sink being one more peer.
6. Routed experts are streamed from DRAM for the union of the 8 rows' picks only. How an expert is split over the
   chips (whole experts per chip, or a 1/N slice of every expert) is a parameter of the MoE piece and is being
   measured.

## Pieces
| # | directory | what | state |
|---|---|---|---|
| 1 | `collective/` | cross-chip sum / gather inside a program: sender cores, L1 landing buffers, persistent counters; standalone and as a fragment appended to another program | runs on 8x4, exact; standalone latency measured; fragment form not yet used by another piece |
| 2 | `ihc_site/` | iHC pre-mix (+ 1/RMS), post-mix and post+pre for 8 rows on one chip, fp32 streams | runs on one chip, 188 cases |
| 3 | `sparse_mla/` | read selected rows by id from the packed KV storage, FP8 to BF16, latent attention with sink for 8 query rows, partial outputs for the cross-chip merge; single-row append | runs on one chip, 22 cases |
| 4 | `indexer_scan/` | index-key scan and chip-local top-2048 with positions for 8 users; cut program (global cut in, selection table out) | runs on one chip, 41 cases, exact ids |
| 5 | `moe/` | router, union of picks, DRAM-streamed routed experts, shared expert, dense FFN for 8 rows on one chip | routed experts run on one chip; rate and accuracy work in progress; board runs are gated by `HY4_MOE_BOARD=1` |

Not here yet: the cross-chip merges (global top-2048, log-sum-exp), the dense attention projections, a layer built
from the pieces, and anything on more than one chip except `collective/`.

## Measured so far
One chip unless said, traced replay (20 replays back to back, one synchronize), 2026-11-19. The ttnn column is the
same stage with ttnn ops from `../PERF.md` (traced).
| stage | this | ttnn ops |
|---|---|---|
| index-key scan + local top-2048, 8 users x 32,768 positions | 120 to 150 us; key stream 387 GB/s | 29.2 ms |
| the same, 1 user | 66 to 82 us | 3.65 ms |
| iHC pre-mix with 1/RMS, 8 rows | 15 us on 48 cores, 16 us on 24 | 0.30 + 0.10 ms |
| iHC post-mix + next pre-mix, 8 rows | 16 us on 48 cores, 18 us on 24 | 0.09 + 0.30 + 0.10 ms |
| read 64 selected rows + attention, 8 query rows | 26 us (9 us of it is the launch) | 1.6 ms per user |
| the same with 2048 selected rows on the chip | 288 us (112 us of it is the read) | 1.6 ms per user |
| single-row append, 8 users | 5.6 us | 0.06 ms per user |
| cross-chip sum of 4 KB, one mesh axis / both axes (8x4) | 18 us / 24 us | 14 to 23 us per axis |
| routed experts, union of 64 picks, 1/32 of each expert on the chip (42.5 MB) | 257 us, 165 GB/s whole program | 0.85 ms (all 8 of a chip's experts) |

Accuracy against the references in `../reference/`: scan scores within 0.13 % to 0.31 % of the score scale with
ids equal to the top-k of the device's own scores in every case; attention partial outputs PCC 0.9990 or better;
the iHC site's 1/RMS, mix coefficients and output within 1.3e-7 relative of float64 at Hy4 width (the ttnn form:
5e-6 to 2e-5); routed experts PCC 0.998 against exact arithmetic on
the BFP4-rounded weights (gain 0.979, being worked on).

Known limits of this version: a program's semaphores are rewritten at every launch and cost launch time (6 us
without, 31 us with 6 semaphores on 64 cores; `sparse_mla/` uses global semaphores for its hand-overs for that
reason, the other pieces do not yet); the scan's scores are BF16 because the fused top-k word is
`[bf16 score | stamp]`; an FP32 circular buffer without unpack-to-dest is unpacked at BF16 precision; a NoC
multicast on NOC1 needs the rectangle's end corner first on Blackhole.

## Running
Each piece has `tests/`. Compile-only on a simulated 32-chip Galaxy (no board; kernels are JIT-compiled and the
circular-buffer, L1 and kernel-size limits checked):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml \
        pytest models/experimental/hy4_preview/fused/<piece>/tests

On a board, run a piece's smallest case first, alone, with `TT_METAL_OPERATION_TIMEOUT_SECONDS` set (20 to 30 s):
a kernel that deadlocks otherwise holds the device until the process is ended. Tests that read the prefill KV
layout need the Hy4 prefill adapter (`models/demos/deepseek_v3_d_p/tt/hy4`, tenstorrent/tt-metal#65935 and the
PRs stacked on it), which is not on this branch.
