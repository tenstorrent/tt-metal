# t227 notes (DiffVAE next step after C4)

Branch reset onto origin/ttp/t48-ltx25-integrated @ 036247a8eb6 (DIFFVAE_S5_2D=1 opt-in, 4.927 s production decode).

Why not fused MLP / uint8 pull / tracing first: t222 job E855 2-D stage tree (tt-project/t222/stage_tree_jobE855_2d.txt
on origin/ttp/t222-...): decode 5009 ms = det stages 1169 (stage 1 alone 514) + stage 5 3840; per stage-5 block
attention 417 ms, MLP 25 ms, context 5 ms; pixel pull 99 ms. Attention is 3.33 s (66%). Fusing the MLP saves <= ~0.2 s,
the pull overlap <= 0.1 s, tracing tens of ms. So the next step is inside attention, and it needs the op split first
(PLAN 1.3 #1: never measured).

Job F (blx01 broker 863, submitted 22:00:30 UTC): driver /var/tmp/fasth3/t227/drv/driver.sh (from t222 driverE, same
health/drop rules), run.sh + decode227.py, code = /var/tmp/fasth3/t219/ov overlay (md5 equal to t48 head for NA/stage5
files; diffvae_ltx.py differs only by the default-off DIFFVAE_GNA_STRIDE env). -t 500.
  arm hifi2: DIFFVAE_S5_2D=1, 2 timed seeds (device noise) + one deep-span (TT_DIT_BLOCK_PROF-level) profiled decode.
  arm lofi : same with DIFFVAE_NA_FIDELITY=lofi (P11, env knob exists already), + host-noise yuv for seeds 0-4.
  Driver then scores lofi vs diffvae/ref (unoptimized, host noise) -> drv/cmp_lofi_vs_ref.json and vs the 2-D hifi2
  host-noise outputs of job 853 (t219/out) -> drv/cmp_lofi_vs_hifi2.json.
Marker: blx01 /var/tmp/fasth3/t227/drv/driver.marker. Outputs: /var/tmp/fasth3/t227/outF (run.log, stage_tree_{hifi2,lofi}.txt).
Next: read marker, run.log DECODE lines, both deep trees (split of the 417 ms attention), cmp jsons. Then pick the code
step from the split (halo exchange / permutes / sdpa / qkv) and record it here.

## Job F result (blx01 863, 22:00-22:07 UTC, 242 s, no drops). Driver marker rc=13 was only the scorer's python3 lacking numpy; rescored by hand with t48 python_env.
- hifi2 (default) 4.927 s mean; lofi 4.898 s (-29 ms). lofi vs unoptimized ref (host noise, seeds 0-4): PCC 0.99986-0.99987, PSNR 48.1-49.0 dB (floor 43.7). vs 2-D hifi2: same. Gain too small to bother defaulting.
- Stage-5 block (8 blocks, ~471 ms each, deep profile): attention 429 = neighborhood-sdpa 294 (69%) + qkv-lanes slice+norm+rope 58 + halo/brick k,v 51 (halo-exchange 2x10.4) + q-to-seq 9 + proj 10. MLP 25.
- Conclusion: the NA SDPA kernel itself is 2.35 s of 4.93 s. lofi barely moves it, so it is not math-fidelity bound: look at masked/padded tile waste in the brick (tile narrowing, PLAN), K/V reads and core grid utilization. Second target: qkv-lanes 58 ms/block (0.46 s) -> fuse slice+norm+rope.
- Next step (standard run): profile the neighborhood-sdpa op (op-level: core count, per-core tiles, useful vs computed QK tiles) and implement NA tile narrowing behind an env knob. Also run the #226 follow-up tests (needs a ttnn build at/after 5c1635d733d on blx01).
Files: tt-project/t227/stage_tree_{hifi2,lofi}.txt, cmp_lofi_vs_{ref,hifi2}.json.

## Run 2 (2026-10-07 late): design for the next code step, no code yet
- Rebased onto origin/ttp/t48-ltx25-integrated @ 5c1635d733d (clean).
- blx01 unreachable (ssh: No route to host), so there was no device job and the #226 follow-up tests
  (test_choose_sharded_brick_regression + GNA_STRIDE+S5_2D guard) did NOT run. No ttnn build at/after
  5c1635d733d exists on g15blx02 (/home budget: ~92 GB used, no new build dir allowed).
- Next step chosen: "key phase" (offset K/V brick grid), see tt-project/t227/KEY_PHASE.md + key_phase_calc.py.
  It is exact. Brick (2,4,4), phase (1,1,1), chunk (2,1,1): 56 K slots per Q brick vs 98 today (1.75x).
  Estimated NA SDPA 294 -> ~168 ms/block, decode 4.93 -> ~3.9 s. Tile narrowing (#213) was only ~3%.
- Next: implement planner + reader query_phase + Python opt-in DIFFVAE_NA_KEY_PHASE=1 (list in KEY_PHASE.md),
  build on blx01 /var/tmp/fasth3 (or blx03 ~/fasth3), job 1 = tests, job 2 = decode A/B phase off/on.
