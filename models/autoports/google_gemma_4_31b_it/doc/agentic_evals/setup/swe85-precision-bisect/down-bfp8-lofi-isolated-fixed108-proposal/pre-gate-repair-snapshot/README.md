# Isolated down BFP8 LoFi fixed108 candidate

Prepared host-side only; no admission or launch. The one full-mode panel reuses the exact 108-row fixed manifest: 36 source targets with seeds 42/43/44, no teacher-forced branch, one free run per row, and the per-case cap recorded in each fixture (256 through 4096 tokens).

This isolates one precision group relative to `setup/head_bfp8_precision_config.json`: attention, MLP, and down start as BFP4 LoFi, and the served lm_head is BFP8 LoFi. Only down changes from BFP4 LoFi to BFP8 LoFi. It does not carry P2's BF16 HiFi4 lm_head setting.

The forecast uses the clean P1 fixed108 completed wall with broad explicit planning multipliers. P2 elapsed time is excluded from forecast calibration and policy-speed ranking because an unintended HF CPU process overlapped that run. P2 terminal reconciliation remains an ordering prerequisite, not a policy baseline.

Execution is blocked by `STOP_BEFORE_LAUNCH`, successful P2 terminal/root forecast review, root-reviewed matched sklearn suffix512 TT plus any conditional HF outcome, no active HF process, no device owners, and explicit root admission. The command performs one serial full-mode panel; there is no automatic retry. Static validation does not import TTNN.
