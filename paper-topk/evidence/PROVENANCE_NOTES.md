# Evidence provenance notes (2026-08-16)

- These reports are the archived primary artifacts of the two agent campaigns
  (previously volatile /tmp scratchpad). Companion measured data lives in
  tests/ttnn/unit_tests/operations/reduction/baselines/ (scope51, smallk_routefix,
  comp3 incl. scenarios1_table.csv and psweep4_full.csv).
- Roofline-v2 derivation ("fits 11 silicon points within 14.1%") is NOT lost:
  it lives in the llm_perf repo, branch nkapre/topk-roofline-v2, commit f51abb3
  ("BH top-k roofline v2: two-family analytical model replaces the v1 lookup
  table"). Local-only branch on this box (~/llm_perf).
- Clock: idle AICLK measured 0x320 = 800 MHz via tt-smi. The 1.35 GHz used for
  cyc->us conversions is the assumed BUSY clock and has NOT been captured under
  load on this board. TODO before paper submission: capture busy AICLK during a
  long kernel (Tracy device frequency field or telemetry poll mid-run) and
  re-derive every cyc->us figure from the measured value.
- Remaining measurement gaps (from evidence.md): WH datapoint, energy,
  per-run chunk-skip skip-rate telemetry, error bars beyond 3 trials.
