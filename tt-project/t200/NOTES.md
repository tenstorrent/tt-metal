# t200: 6-step S1 that keeps the high-sigma head, stacked on the S2x2 default

## Setup (blx01, all under /var/tmp/fasth3/t200)
- Python: overlay tree at t48 b21f12b93a2 (S2 2-step default; t48 head ba4682afbd0 only changes a README), built by
  mkoverlay.sh from t195's files/. C++ build + JIT cache /var/tmp/fasth3/t48 (bf7db12a149). Same stack as ref_t48_s2x2 (job 764).
- Default S1 (8 steps): 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
- Arm s1x6b: LTX_S1_SIGMAS=1.0,0.99375,0.9875,0.975,0.909375,0.725,0.0 (drops 0.98125 and 0.421875).
  #186's rejected s1x6 dropped 0.99375 and 0.98125 instead.
- One broker job: run_cfg.sh s1x6b LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0, -t 240,
  PYTEST_S=220, -e /var/tmp/fasth3/t159/env.yaml (job 764 took 156.7 s).
- driver.sh (t186's): waits for blx01 ready (broker RUNNING section, not HELD, no smarton job; 2 passes), submits
  under `ttp lock blx01-device`, reruns a dropped job once, skips after 2 drops, then post.sh.
- post.sh -> tt-project/data/g15/t200_s1x6b/: seed<N>.mp4, vis/ (seed0 t=2.6-3.4 s at 10 fps, ref and cand;
  1 fps ref|cand strips per seed), sbs/, eval_vs_ref_t48_s2x2/ (PCC/PSNR + VBench 5 dims), POST.done.
- Marker: t200/DRIVER.done = "<rc> <reason>".
