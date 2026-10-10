# t313 notes (run 1245, 2026-10-10)
- Worktree is on branch ttp/ltx-rt-next (no upstream set; never push to ltx-rt).
- ltx-rt b9f8587ce6c == merge base; t48 f6547442b30 is 371 commits (356 non-merge) ahead; ltx-rt 0 ahead -> fast-forward, no conflicts.
- ltx23-main-pr: all 7 gains already in t48 or ltx-rt (mapping in GAINS.md). Nothing cherry-picked.
- GAINS.md committed: 49ffb24aa8b (tt-project/t-ltxrt-next/GAINS.md).
- Build: tmp/t313/build.sh detached as t313-build; log /home/smarton/fasth3/tt-metal/tt-project/state/runs/1245/t313-build.log, rc /home/smarton/fasth3/tt-metal/tt-project/state/runs/1245/t313-build.rc.
Next: check rc; import smoke of LTXDistilledPipeline + LTX25DistilledPipeline with PYTHONPATH/TT_METAL_HOME=worktree;
ttp checks -- 'pytest -q models/tt_dit/tests/unit/test_ltx_*.py ...'; ttp push --own --detach; then remove build_Release/.cpmcache/build dirs.

## Run 1253 (2026-10-10 03:30 UTC)
- blx01 driver: /var/tmp/fasth3/t313/drv313.sh 356, pgid 1934642 (setsid nohup). Log drv313.done.log, marker drv313.done.
- Broker job 356 = arm next (overlay f6547442b30), running since 03:28. Arm ltxrt (overlay b9f8587ce6c) follows, then cmp313.py -> t313/cmp.txt.
- /home write incident: ttp detach --remote wrote /home/smarton/.ttp-detach/1253 on blx01; driver killed by pgid, dir removed, job 356 kept.
- Next: when drv313.done exists, read cmp.txt, run_*_job*.log (PERFORMANCE table gen #1), view still_ltxrt_vs_next_gen1_f72.png.
  If pass: git fetch; origin/ltx-rt must still be b9f8587ce6c; git push origin f6547442b30:refs/heads/ltx-rt (no force).
  Then clean blx01 t313 (keep cmp.txt, stills, logs), remove g15blx02 build dirs in worktree.

## Run 1254 (2026-10-10 03:40 UTC)
- Job 356 (next, f6547442b30, unmodified test): PASSED 276 s; gen#1 warm Total 7.33 s (900 MHz clamp); mp4 md5 df83c3e0... (gen0 == gen1).
- Job 357 (ltxrt b9f8587ce6c): FAILED in 9 s: origin/ltx-rt's own test passes image_conditioning= to LTXPipeline -> TypeError
  (fixed on t48 by 5006c6aba6e). Output kept in out_ltxrt_job357.
- Reference rerun: ov_ltxrt test patched to drop only that kwarg (ov_ltxrt/PATCH). Driver drv313b.sh (pgid 2006587), broker job 359.
  Marker /var/tmp/fasth3/t313/drv313b.done, compare -> cmpb.txt + still_ltxrt_vs_next_gen*_f72.png.
- Next: if PCC/PSNR fine and still OK: git fetch; origin/ltx-rt still b9f8587ce6c -> git push origin f6547442b30:refs/heads/ltx-rt.

## Run 1255 (2026-10-10 03:48 UTC)
- Job 359 (ltxrt b9f8587ce6c, test minus image_conditioning kwarg): completed, exit 0, 8+3, seed 10, default prompt.
- cmpb: ltxrt vs next gen0/gen1 PCC=0.014 PSNR=11.4 dB: DIFFERENT CONTENT (ltxrt: woman with guitar, warm studio;
  next: man in leather jacket, purple studio). Both clean. Still: tmp/t313/vis/still_ltxrt_vs_next_gen1_f72.png
  (blx01 /var/tmp/fasth3/t313/still_ltxrt_vs_next_gen1_f72.png). Not pushed to ltx-rt: the gate cannot be judged on this.
- Suspect: t48 defaults LTX_FUSE_GATE_ON_DEVICE=1 + LTX_FUSE_NORM_ADALN=1 (numerics change; ltx-rt kept gate fold off over a VBench loss, #296).
- Diagnosis job 360: arm nextoff (next code, both flags =0) via run313c.sh / drv313c.sh (pgid 2093545), marker drv313c.done, cmp -> cmpc.txt,
  still_ltxrt_vs_next_off_gen*_f72.png.
- Next: if nextoff ~ ltxrt (PCC > ~0.95), content change = those 2 flags. Then for the ltx-rt push: flip their defaults to 0 on
  ltx-rt-next (keeps ltx-rt behaviour; spec: keep upstream behaviour where our change need not replace it), and the job 360 run
  is that config's e2e evidence. If nextoff still differs: find the other numerics change before any push.

## Run 1256 (2026-10-10 03:55 UTC)
- Job 360 (nextoff: next code, LTX_FUSE_GATE_ON_DEVICE=0 LTX_FUSE_NORM_ADALN=0) completed; vs ltxrt still PCC 0.014 / PSNR 11.45 dB.
  So the two fusion flags are NOT why the scene differs.
- Real cause: DEFAULT_LTX_PROMPT (models/tt_dit/utils/ltx.py) differs. ltx-rt b9f8587ce6c: "A young woman ... acoustic guitar";
  next f6547442b30: "A confident rapper in a black leather jacket ...". Both videos match their own prompt. Not a quality issue.
- Job 361 failed in 0 s (my run313d.sh arm check refused 'nextp'; no device time, not a drop). Fixed line 12.
- Job 363 = arm nextp: next code (f6547442b30, defaults incl. fusion flags on), PROMPT = ltx-rt default prompt
  (blx01 t313/prompt_ltxrt.txt), seed 10, 8+3, unmodified test. Driver drv313d.sh (setsid, pgid 2173641), marker drv313d.done,
  compare -> cmpd.txt + still_ltxrt_vs_next_p_gen*_f72.png.
- Next: read cmpd.txt + still. If PCC/PSNR vs job 359 is good (same content, clean): this is the push evidence.
  Open question for the push: should ltx-rt-next keep ltx-rt's DEFAULT_LTX_PROMPT (spec: keep upstream behaviour where our
  change need not replace it)? Check who uses DEFAULT_LTX_PROMPT (ltx-server default?) before pushing.
  Then git fetch; if origin/ltx-rt still b9f8587ce6c -> git push origin <next head>:refs/heads/ltx-rt (fast-forward only).
