# t240: DiffVAE next lever toward 1 s (after S5_LEAN)

Base: ttp/t48-ltx25-integrated @ f0585103844 (S5_2D, key phase and S5_LEAN on by default; blx01 job 879: 4.261 s).

## Lever: gather rephase (DIFFVAE_NA_GATHER_REPHASE=1)
The stage-5 profile (t238 deep tree) shows 68 ms per block for the K/V key-phase rebrick:
to_natural -> slice -> zero-frame concat -> to_bricked -> tilize. That is about 0.54 s over 8 blocks.
Commit fdd5374f3e7 replaces the chain with one `ttnn.embedding` per lane. A cached uint32 index
maps each phased bricked site to its resident bricked row, and the op writes TILE directly. Pad frames
read the clamped real frame. They are masked keys, so the output should be bit-identical to the default.
CPU test: test_key_phase_gather_index_* in test_neighborhood_sdpa.py (3 pass).

## A/B (blx01 job 882)
Driver: g15blx01:/var/tmp/fasth3/t240/drv/driver.sh (marker driver.marker, log driver.log, out ../out).
It reuses the t238 Release build tree /var/tmp/fasth3/t238/b, checked out at fdd5374f3e7 (no C++ change since t238).
Arms: def vs gather. Each runs warm-up + 2 timed seeds + deep profile + host-noise seeds 0,1. Scoring: md5 gather vs def, and cmp vs diffvae/ref.

## Next step
On `T240_DRIVER_DONE stage=done`:
- If gather is identical (or PCC/PSNR-neutral) and faster: flip the default (=0 opts out), make a -land branch
  from origin/ttp/t48-ltx25-integrated, cherry-pick the code commits, run `ttp push --detach`, then publish the own branch.
- If it crashed: read out/run.log. Likely causes are the embedding constraints (weights must be interleaved
  row-major, last dim a multiple of 32) or a sharded exchange output.
