# t511 notes (device A/B LTX25_S2_ANCESTRAL off vs on, then ltx-rt push)

- Code: 923fffb239d = cherry-pick of a22dc027ee2 onto origin/ltx-rt f6547442b30 (clean). CPU test
  models/tt_dit/tests/unit/test_ltx25_s2_ancestral.py: 4 passed.
- Box: blx03 (only box with LTX-2.5 weights: /var/tmp/fasth3/t503/ltx25; blx01's copy is gone).
  Tree ~/fasth3/t511/tt-metal on blx03 = `cp -al` of t503's tree + runtime/hw/lib from g15blx02 t313
  (t503's copy lacked runtime/hw/lib/blackhole/*.o: its job died at fabric JIT link) + models/ at 923fffb239d.
- Jobs (blx03 serial runner, FIFO behind t503-ab-r2):
  1. t511-e2e-r1: standard LTX-2.5 e2e, test defaults (pre-push gate, also fills JIT /var/tmp/fasth3/t511/jit).
     Cold JIT may not fit 575 s (t373 on blx01: cold 545 s timeout, warm 293 s): then requeue as -r2.
  2. t511-ab-r1: SEED=0, LTX_E2E_SEEDS=0,0+, LTX_E2E_AB_ENV=LTX25_S2_ANCESTRAL=1:
     out_ab/ltx_av_fast_1920x1088_1.mp4 = off, _2.mp4 = on (both pure trace replays).
- Probe: ssh g14blx03 'bash ~/fasth3/runner/probe.sh <ID>'; marker /var/tmp/fasth3/runner/done/<ID>.done
- Next: check logs (noise_seed=20000 only in gen #2), quote Stage 2 denoise for gen #1/#2, copy mp4s to
  g15blx02, run ltx_eval video --vbench-ref, look at stills; if ok, ttp push to ltx-rt (ff only).
- Cleanup at end: ~/fasth3/t511 and /var/tmp/fasth3/t511 on blx03.
- 2026-10-15 04:13 PDT: queued; runner restarted pid 49491 (blx03 rebooted ~03:51 PDT), t503-ab-r1/-r2 ahead
  (they stop early on t503's last_fail guard). ttp checks started in run 1958 (checks.out/.rc there).
