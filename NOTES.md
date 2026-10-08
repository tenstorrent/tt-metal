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

## Result (job 882, completed, 209 s, no drop)
- def 4.265/4.264 s (mean 4.264). gather 3.775/3.769 s (mean 3.772): -0.49 s.
- Not bit-identical. gather vs def: PCC 0.99999, PSNR 62.8/62.5 dB, worst frame 1 at 53.0/53.2 dB.
  The pad frames now hold real keys, so masking leaves a trace near the front frame. That is far above the seed floor (43.7 dB).
  Gather vs def: out/cmp_gather_vs_def.json. The driver writes out/cmp_{def,gather}.json vs diffvae/ref.
- Default flipped in 96734409a57 (=0 opts out).

## Next step
1. Check out/cmp_gather.json vs ref is within ~0.1 dB of cmp_def.json (t238: def vs ref PCC 0.99996, 55.6/55.2 dB).
2. Make a -land branch from origin/ttp/t48-ltx25-integrated, cherry-pick fdd5374f3e7 and 96734409a57 (code only, not the notes),
   run `ttp push --detach` onto ttp/t48-ltx25-integrated, then `ttp push --own --detach` for this branch.
3. Next lever: fuse the stage-5 qkv-lanes slice+norm+rope (48 ms per block), or stage-1 de-replication (606 ms replicated).
