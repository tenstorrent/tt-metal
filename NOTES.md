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
