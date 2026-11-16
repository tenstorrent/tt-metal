# P8 fixed108 full-panel candidate (inert)

This candidate uses the exact byte-identical 108-row P3 manifest under P8 precision: BFP8-LoFi for attention, MLP gate/up, down, and lm_head, with other policy fields unchanged. It remains unadmitted and unrun.

The prior long-subset extrapolation is preserved verbatim in `forecast-initial-long11-extrapolation.json`. The revised workload-weighted planning point and sensitivity range are in `forecast.json`; direct per-row P6 workload inventory and all 11 matched P8/P6 timing rows are in `matched-workload-validation.json`. The focused 11 are long-context/anomaly-selected, so the resulting interval is a planning sensitivity envelope, not a confidence interval or measured bound.

P8 currently fails the natural-accuracy gate: 447/512 (87.30%) versus the 90% threshold; AIME is 98%, top-5 is 510/512, and top-100 is 512/512. Therefore the P8 full108 and acceptance commands are conditional only and must not run without a passing accuracy rerun, independent full108 review, and explicit root admission. `setup/attention-down-acceptance/commands-p8-conditional.json` preserves the P6 command and uses a separate output directory.
