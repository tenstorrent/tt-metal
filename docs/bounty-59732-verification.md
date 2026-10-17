# Bounty 59732 - ttnn.sampling distribution bias fix verification

## Issue
- ID: 59732
- Title: Fix ttnn.sampling distribution bias from low-precision random threshold
- Repo: tenstorrent/tt-metal
- Reward: $3,000
- URL: https://github.com/tenstorrent/tt-metal/issues/59732

## Root cause (per issue + PR #59759)
`ttnn.sampling` compared cumulative probabilities against a random threshold packed to BF16 (8-bit mantissa, step 1/256 near 1.0, max ~0.998/255/256). Tokens whose cumulative slice lay in the last 0.4% of mass could never be drawn; tokens at ~2% were under-sampled.

## Fix (already merged to origin/main as c0b2affd, PR #59759)
Two kernel files, no API change:
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp`: rand_scale 0x3F7F7FFF -> 0x43800000 (256.0f), SFPU floor_tile before packing so each element is an exact integer 0..256 in BF16
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp`: threshold = (hi*256+lo)/65536 over 65536 equiprobable values in [0, 1-2^-16]; hi=element 0, lo=element 2 (element 1 dead on Wormhole face 0)

## Host repro (this ask, no device)
Ran /tmp/repro_sampling_bias.py (host-only BF16-quant simulation + lattice simulation, no ttnn import, no hardware):
- Tail token p=0.00193 (gap 6.25): BF16 fraction above cutoff 0.00000 vs expected 0.00193 (0.00x) -> matches issue "gap-6 at 0.00x". Lattice: 0.00189 -> 0.98x (correct).
- Gap sweep replicated: gap 5 0.33x, gap 6 0.00x with BF16; lattice ~1.0x for all.
- Current HEAD file markers verified: compute contains 0x43800000U + floor_tile + rounding.h; writer contains RAND_HI_ELEMENT/RAND_LO_ELEMENT/RAND_LATTICE_SCALE and (rand_hi*256+lo)/65536

## Regression tests already in tree (from fix)
- `tests/ttnn/unit_tests/operations/reduce/test_sampling_distribution.py`: 4 tests covering reference distribution (32 users x 3000 draws, 3-sigma), threshold uniformity (m=4,10,20 within 4 sigma), low-digit tail, softmax gap. Requires hardware; gated by TT_METAL_SIMULATOR.
- `tests/ttnn/unit_tests/operations/reduce/test_sampling.py`: determinism/valid-index tests

## This branch
Branch `bounty-59732-sampling-fix-repro` from current origin/main (14565689). No code change (fix already in main); this doc + commit records the independent host repro required by the ask. Push is via fork luoyu2475 for PR visibility.

## Wallet (per ask)
0x1BF1f65711DB933047E67b1F3F959bb42dc6af64
