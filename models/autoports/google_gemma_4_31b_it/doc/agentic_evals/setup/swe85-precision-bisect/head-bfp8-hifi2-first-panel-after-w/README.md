# First head-BFP8 HiFi2 fixed-panel candidate

Candidate only; no admission or device execution. This proposal uses the frozen, independently reviewed P common-cap4096 manifest (108 trials, 15 target pairs) and existing head-BFP8 HiFi2 config. Forecast was frozen before execution at 3,524.69 seconds, planning range 2,515.76–5,878.86 seconds.

The inert driver waits for W’s successful terminal queue and both successful mode completions, verifies V terminal success, then rechecks pinned source hashes. Immediately before the panel it requires empty TT device owners and no active `precision_repro_hf.py` process. It will not launch unless root changes the admission placeholder to `admitted`. Timeout is 7,200 seconds, above the 5,878.86-second forecast upper bound.

The candidate panel is diagnostic only. It does not change scores or select a precision policy. All 15 target pairs must complete without semantic anomaly, progressing caps remain inconclusive, all 108 outputs must be reviewed for regressions, and the root-defined fastest-passing and later vLLM/full85 requirements remain in force.
