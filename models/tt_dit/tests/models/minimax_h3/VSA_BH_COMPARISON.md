# VSA vs dense attention: Blackhole comparison handoff (exploration, not for merge)

## Goal

On Wormhole Galaxy 4x8, MiniMax-H3 ref2va with VSA (`ttnn.transformer.sparse_ring_sdpa`) is slower than dense ring
SDPA at 5 s and 10 s and faster only at 15 s. Measure the same things on Blackhole 4x8 to find which of the causes
below depend on the hardware. Wormhole reference numbers are at the end.

## Wormhole findings to test

1. **64-row Q chunks (expected to depend on the hardware).** One mask row per 64-token VSA tile forces
   `q_chunk_size=64` (`attention_minimax_h3.py` `_vsa_program_config`; `static_assert(Sq_chunk_t == block_tiles)` in
   `sparse_ring_sdpa.cpp`). On WH, dense ring SDPA at q64 is 2.68x slower than at q256. The sparse kernel with every
   block kept is 1.36x dense-q256 (but 2x faster than dense-q64). WH-only code in this branch adds cost here: the
   resident-step tables live in L1 instead of on the TRISC stack, the unpack thread's step code is built at -Os,
   and P@V output packs go one tile at a time (`MIN_BLOCKED_PACK_TILES` is 8 on WH, 4 on BH). See
   `sparse_ring_streaming.hpp`.
2. **Sparsity 0.9 keeps about 39% of the work (expected to be the same on both).** Prefix (text, reference, audio)
   rows keep every key, and every video row keeps every prefix key (`vsa_prepass.py` `vsa_routing_request`). On WH
   the s0.9 time is 55% of the keep-everything time.
3. **Work around the kernel (expected to be the same in ms on both).** Routing prepass (`vsa fused: *` sections),
   the coarse-branch gate (inside `heads + gate`), and the larger rung from VSA padding (`vsa_min_padded_len`).
4. **Load imbalance (fixed on WH by `MINIMAX_H3_VSA_INTERLEAVE=4`).** Prefix tiles sit first in each shard and cores
   take contiguous Q-block runs. This branch does not have the automatic load balancing from PR #65585
   (`d18270796f0`).

## Setup

- Branch `jameslee/vsa_sparse_wh_bh_explore` (built normally; the WH-only kernel changes are behind `ARCH_WORMHOLE`).
- Test: `models/tt_dit/tests/models/minimax_h3/test_scratch_vsa_breakdown.py`. One transformer block at the
  campaign's request layouts, random activations, compile run, then 3 unsynced block runs and 3 synced per-section
  runs. Results are the `RESULT` lines.
- Always set `TIME_VSA=1`; without it the per-section timings are missing.
- **Fixture selector differs by hardware.** On Blackhole the 4x8 row is `4x8sp1tp0nl2_ring_is_fsdp0` (2 links,
  fsdp off). WH used `4x8sp1tp0nl4_ring_is_fsdp1`. Select with `-k "4x8sp1tp0nl2"` on BH; `fsdp1` selects nothing there.
- Select attention with `dense-` or `vsa-` in `-k` (plain `vsa` also matches the file name).
- Cases: `uc4_5s_54211`, `uc7_5s_81640`, `uc5_5s_96515` (prefix-heavy), `uc4_10s_89887`, `uc4_15s_125563`.
- `uc4_15s` needs `SCRATCH_K_CHUNK=384` (or 512) for VSA: at 256 the op hits its 64-K-pieces-per-shard TT_FATAL
  (an error, not a hang). The pipeline avoids this by falling back to dense above that size.

## Runs (in priority order; UC4 10 s first)

```bash
T=models/tt_dit/tests/models/minimax_h3/test_scratch_vsa_breakdown.py
SEL='4x8sp1tp0nl2 and uc4_10s'
# A  dense, default chunks, at VSA's rung
TIME_VSA=1 pytest -s $T -k "dense- and $SEL"
# A2 dense at its own rung (the fair block-level baseline)
SCRATCH_DENSE_OWN_RUNG=1 TIME_VSA=1 pytest -s $T -k "dense- and $SEL"
# B  dense at the sparse op's Q geometry (finding 1)
SCRATCH_DENSE_Q=64 SCRATCH_DENSE_K=256 TIME_VSA=1 pytest -s $T -k "dense- and $SEL"
# C  dense q256/k256 (separates the q effect from the k effect)
SCRATCH_DENSE_Q=256 SCRATCH_DENSE_K=256 TIME_VSA=1 pytest -s $T -k "dense- and $SEL"
# D  sparse, keep every block (finding 1: kernel cost at full width)
SCRATCH_SPARSITY=0.0 MINIMAX_H3_VSA_INTERLEAVE=4 TIME_VSA=1 pytest -s $T -k "vsa- and $SEL"
# E  sparse, production sparsity, balanced (findings 2 and 3)
MINIMAX_H3_VSA_INTERLEAVE=4 TIME_VSA=1 pytest -s $T -k "vsa- and $SEL"
# F  sparse, production sparsity, no interleave (finding 4)
TIME_VSA=1 pytest -s $T -k "vsa- and $SEL"
# G  sparsity sweep for the work-vs-time curve
for s in 0.5 0.75 0.97; do SCRATCH_SPARSITY=$s MINIMAX_H3_VSA_INTERLEAVE=4 TIME_VSA=1 pytest -s $T -k "vsa- and $SEL"; done
```

Then repeat A, A2, E and F for `uc4_5s`, `uc7_5s` and `uc5_5s`, and for `uc4_15s` with `SCRATCH_K_CHUNK=384` on the
VSA runs.

## How to read the results

| Question | Ratio | WH value |
|---|---|---|
| How much do 64-row chunks cost dense? | B / A (ring sdpa) | 403.4 / 150.5 = 2.68 |
| How much does the sparse kernel pay at full width? | D / A (vsa fine / ring sdpa) | 204.4 / 150.5 = 1.36 |
| How well does sparsity turn into time? | E / D (vsa fine) | 112.8 / 204.4 = 0.55 |
| How much does balance matter? | F / E (vsa fine) | 170.3 / 112.8 = 1.51 |
| What is added around the kernel? | `vsa fused: *` + extra `heads + gate` in E, ms | 8.4 + 5.6 |
| Block total, sparse vs dense at its own rung | E block / A2 block | 198.8 / 181.0 = 1.10 |

If B/A and D/A are much smaller on BH, finding 1 is mostly WH-specific. If E/D and the added ms match WH, findings
2 and 3 are confirmed as hardware-independent. Compare attention sections across hardware, not block totals: BH runs
fsdp0 with 2 links and WH ran fsdp1 with 4, which changes to_qkv, to_out and the feed-forward.

## Wormhole 4x8 reference (fsdp1, 4 links, sparsity 0.9 unless noted)

Attention op ms (`ring sdpa` for dense, `vsa fine` for sparse); block ms in brackets (unsynced, fsdp1).

| Case | Dense, own rung | Dense, VSA's rung | VSA, no interleave | VSA, interleave G=4 | Best VSA found |
|---|---|---|---|---|---|
| UC4 5 s | 45.5 [87.8] @55296 | [110.6] @65280 | 80.7 [134.5] @65280 | 52.9 [106.4] | 48.5 [96.4] @59392 |
| UC4 10 s | 122.8 [181.0] @90368 | 150.5 [224.0] @106496 | 170.3 [256.3] | 112.8 [198.4] | 103.9 [183.0] @98304, k384 |
| UC7 5 s | 111.6 [170.7] @90368 | same rung | 162.7 [230.9] | 132.8 [200.6] | |
| UC5 5 s | 156.6 [231.1] @106496 | same rung | 221.7 [307.6] | 199.1 [285.6] | |
| UC4 15 s | 292.7 [394.0] @147712 | same rung | 274.0 [394.1] k512 | 208.3 [328.1] k512 | 193.7 [313.5] k384 |

UC4 10 s at rung 106496, attention op ms: dense q256/k512 150.5, q256/k256 168.0, q64/k256 403.4, q64/k512 425.8;
sparse keep-all G4 204.4 (k512: 192.7); sparse G4 by sparsity 0.5 / 0.75 / 0.9 / 0.97: 171.1 / 137.2 / 112.8 / 99.2.
`SCRATCH_QBPP` 32 / 21 / 14 / 11 / 7 (1 to 6 passes): 112.7 / 112.8 / 112.8 / 111.1 / 114.3, so passes cost nothing.

UC4 10 s section deltas, sparse G4 @106496 vs dense @90368 (block 198.8 vs 181.0): sparse kernel -10.1, prepass
(route 6.2 + pool 1.2 + gather 1.0) +8.4, heads + gate +5.6, to_qkv 25.5 vs 17.3, rest of block 37.1 vs 32.4.

## Cautions

- Activations are random, so top-k routing is close to random and kept blocks more scattered than real requests.
  The same is true on WH, so the comparison is fair, but absolute VSA numbers may be pessimistic.
- `MINIMAX_H3_VSA_INTERLEAVE` changes the tile layout; the geometry unit tests that assert the old layout fail with
  it set. Oracle correctness tests passed on WH (PCC given mask >= 0.9997).
- On WH, a hang in this op once wedged PCIe on all 32 chips and needed a chassis power cycle. On BH, recover with
  your system's usual reset if a run stalls.
