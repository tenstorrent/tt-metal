# t884 notes (low-res first steps on by default where they passed)

- Code: branch `ttp/t884-low-res-first-steps-on-by-default-where-`, one code commit 986a8c96921 on 98bac7b10f3
  (origin/ltx-rt). `ttp checks --target origin/ltx-rt`: 4 passed. Opus review: pass (2 `should`, 3 `nit`, see result.json).
- Default on (first 2 forwards at 2/3 = 896x512, 1344x768 t2v only): 10Eros-Max beta7_fix1 6 s at 4 forwards
  (sulphur's "h3-10eros"), HyperFlow v1.0 at 8 forwards, 6 s and 10 s. Off elsewhere. Off switch
  `MINIMAX_H3_LOWRES_STEPS=0`.
- Drivers (project root, `tt-project/t884/`, commit 89ddda70cd6 on fasth3-opt): `test_t884_drive.py`, `run884.sh`,
  `prep884.sh`, `chain884.sh`, `spec-*.txt`, `audio884.py` (loudness raw and through sulphur's export, PSNR).
- Device run: `chain884` detached on the blx04 broker (17:35 PST 11-27), cells eros158, hyperflow158, hyperflow243,
  one job each (first job 913). State `tt-project/t884/chain884.state`, log `chain884.log`, job lines `chain884.jobs`;
  end marker `/home/smarton/fasth3/data/t884/chain.done`; results `/home/smarton/fasth3/data/t884/out/<cell>/`
  (`results.jsonl`, `<arm>_s<seed>.mp4`, `.audio.npy`, `done`), job log `out/batch-<cell>/run.log`.
- Arms per cell: default (no switch set), off (`MINIMAX_H3_LOWRES_STEPS=0`), explicit (2 at 2/3: what #832 judged).

## Next steps once chain.done exists
1. Per cell: `python tt-project/t884/fasth3_ab.py table <results.jsonl>` and
   `... --reconcile off default forwards,set-up | tt-project/harness/bin/fasth3-reconcile`; quote one steady clip's
   "MINIMAX-H3 PERFORMANCE RESULTS" block from `run.log`.
2. Proofs: "low-res first stage: 2 of N forwards at latent 32x56" in the default arm's log and not in off; default
   and explicit `video_md5`/`audio_md5` equal per seed.
3. `tt-project/harness/bin/fasth3-cpu python_env/bin/python -I tt-project/t884/audio884.py` (from the project root).
4. `ttp note --to sulphur` with the preset (under 4,000 characters), then `ttp obligation receipt o166 "<evidence>"`.
5. `ttp task FastH3 show 875`: the 10 s 4-forward cell stays off unless it shows a pass.
6. tp-commit the chain's files; write result.json.

## Run 3305 (19:21 PST 11-27): driver fix, chain relaunched
- Jobs 913 and 924 (eros158) failed exit 2 in 5 s at collection: test_t884_drive.py imported fasth3_ab, which
  pytest.ini's importlib mode does not put on the path. Fixed (the driver adds its own folder to sys.path),
  prep884.sh rerun, collect-only checked on CPU with run884.sh's env.
- chain884 relaunched from run 3305 with new job keys (<cell>.r2.<try>); end marker
  /home/smarton/fasth3/data/t884/chain.done (0 = all three cells done), failed cells in fail-<cell>.
- Next steps: unchanged (list above), then tp-commit chain884.state/log/jobs in the project root.

## Run 3374 (about 00:10 PST 11-28): one commit per cell, on current origin/ltx-rt
- The branch was rebuilt on origin/ltx-rt 006e48b30d3 (4 commits newer than the old base 98bac7b10f3). The old head is
  kept as local tag `t884-pre-split` (code 986a8c96921). Code commits, in landing order:
  - 9dc9ac462dc: the table, the weight naming and the 10Eros-Max 6 s row (4 forwards, first 2 at 896x512).
  - 9cb0bd195fe: the two HyperFlow rows (8 forwards, 6 s and 10 s).
  Together they equal 986a8c96921 except: test_decode_order keeps its `pytest.raises` line (the old commit changed
  it to `expect_error`, which made the hand-off check count it as a new test), and one more off case for HyperFlow.
  Commits made with SKIP=check-models-do-not-import-tests (fails repo-wide on an LTX tool file) and, for the first,
  SKIP=prefer-expect-error (a line this change does not touch).
- CPU tests: `tmp/h3t.sh` runs named H3 host tests the way harness/checks/h3-host-tests.sh does; 30 passed on
  9cb0bd195fe (test_lowres_steps_minimax_h3.py and test_decode_order_minimax_h3.py).
- New tests on the base (pipeline files of 006e48b30d3, test file of the commit): collection ends in
  `ImportError: cannot import name 'EROS_MAX_BETA7_FIX1'`; with the two constants stubbed in a scratch copy, 7 fail
  (cells_that_passed x3, transformer_file, adapter x3) and the 7 off cases pass (nothing is on by default there).
- Device results: jobs 955 (eros158), 959 (hyperflow158), 963 (hyperflow243) on blx04, code 986a8c96921 on
  98bac7b10f3; tables, timing sections, loudness and md5 proofs in `tt-project/t884/RESULT.md` (project root).
  Not rerun on the rebuilt head: the review's e2e runs on the landing head.
- #875 (10Eros-Max 10 s at 4 forwards): look passes, VBench verdict not finished; the row stays out.
