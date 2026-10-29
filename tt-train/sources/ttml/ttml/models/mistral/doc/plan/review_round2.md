# Phase review, round 2 (train-plan), 2026-10-29

Verdict: clean-pass

Every round-1 required item verified resolved (retention-model budget and 1/2/4/8 depths; no multi-device work, single-chip substitutes stated; LoRA term at measured rates with the missing efficient path in the inventory; dispatch, loss definition, FLOPs convention, block-0 note). Independent re-derivation of the retention model (788/468/264 MiB per layer), the achievable target (38.16 ms/layer, 35.5 fixed, 58.2 dispatch, 1620 ms, 40.0 % MFU) and the matmul calibration agrees with numbers.py and mm_eff.py. Inventory citations spot-checked and correct.

Other concerns (fixed in the same commit): fp64 reference size 35.6 GB not 4.4 GB for 8 layers; wasted dW is 46.9 TFLOP/step = 49 %, not 55 %; workspace line mislabelled "dX of gate_up 134 MiB"; glue/layout rows priced at nominal bandwidth (noted); v double-counted in retained() (removed); skinny-matmul rates are an extrapolation in K/N (claim lowered); loss-trajectory comparison wording clarified.

Hard-check gaps: LoRA term (28 % of target) and 70 % bandwidth / 50 % SDPA are extrapolated or provisional, to be calibrated by the first Tracy run; retention table to be calibrated with the allocator peak at 2 and 4 layers in the composite phase.

Residual risk: the composite-bringup gate "full model runs on one chip" cannot be met for this model; the plan offers the 8-of-40-layer substitute, which the user has not yet explicitly accepted.
