# FLUX.2 generation failure investigation

## Confirmed primary defect

The pinned September 8 base predates a FLUX.2 correctness fix on main:
`2a0f4c55e6727826200082083163df68e5a3270a` (September 10, PR #55225).
The SDPA research commits do not change `models/tt_dit` relative to their
main merge base `2ba6fc2339d53300ae87c5202f335ef56492cfb3`.

The dual-stream block passes a residual and gate into the attention output
projection. On the Linear topology used by our eight-chip allocation,
`ColParallelLinear.forward` falls through to `minimal_matmul` and silently
ignores the requested `addcmul_a` and `addcmul_b`. It returns the projection
alone instead of `residual + gate * projection`. Both image and text streams
are affected, on every dual-stream block. This is missing model math, not an
SDPA precision problem.

## Controlled evidence

The existing `test_transformer[single_blocks]` retains one dual-stream and one
single-stream block, plus input projections, conditioning and output head.
Tests use the pinned checkpoint, 2x4 mesh, SP2/TP4, Linear topology, 4096 image
tokens and 512 prompt tokens. Torch reference calculations and original
thresholds (PCC >= 0.996, RMSE/reference standard deviation <= 9%) are unchanged.
The wrapper only resolves the checkpoint locally and limits CPU threads.

| Configuration | Final PCC | Relative L2 | Existing test |
|---|---:|---:|---|
| Original stock model | 0.00320143 | 100.9885% | Fail |
| Restore only residual/gate math | 0.99989172 | 3.4096% | Pass |
| Only enable per-head Q/K normalization | 0.00369365 | 100.7193% | Fail |
| Restore residual/gate + per-head Q/K normalization | 0.99997153 | 3.1552% | Pass |
| Above + guidance scaling / FP32 sinusoidal phases | 0.99997885 | 1.2210% | Pass |
| Main-style fused residual + per-head normalization, stock conditioning | 0.99997139 | 3.3779% | Pass |

Baseline intermediate checks: input projections have PCC 0.9999929 and L2
0.467%; first dual-stream image output has PCC 0.08564 and L2 159.19%.
Restoring only the missing residual/gate, with no conditioning or normalization
changes, is sufficient to recover the original test's acceptance threshold.

`model_fixes.py` uses an experiment-only **unfused** `addcmul` to establish
causality independently of the fused kernel. Main's actual fix uses
`dit_minimal_matmul_addcmul_fused`. A separate control using that fused operator
and main's config-selection logic also passes. The slight numerical difference
from the unfused residual path is measured above. These diagnostic timings are
not a model performance baseline and no full-main binary test is claimed.

## Secondary issues

- Missing per-head Q/K normalization is independently confirmed as a secondary
  accuracy issue. It is fixed by the same main commit, but alone cannot recover
  the broken residual computation (see controls above).
- The conditioning mismatch is separate: guidance is not scaled by 1000 in the
  TT path, and its sinusoidal phases use BF16 factors. Correcting these helps
  numerical agreement, but did not fix the original noise generation alone.
- The test with main's fused residual operator passed in 80.21 seconds.

## End-to-end confirmation

`stock-main-fixes-01` completed in 428.61 seconds, with 50 denoising steps at
1024x1024, prompt "A photo of a cat sitting on a windowsill at sunset", seed 0,
guidance 4, stock attention and **stock conditioning**. It uses the fused
residual correction and per-head normalization, not the optional conditioning
repair. Visual inspection confirms a coherent cat on a windowsill at sunset,
instead of the prior noise. The model executed untraced with no dynamic loading.

Image: [stock-prompt0-seed0.png](stock-main-fixes-01/stock-prompt0-seed0.png).
The manifest, finite-latent check, image/latent hashes and execution log are in
the same run directory and adjacent `stock-main-fixes-01.log`.

This confirms the generation failure's diagnosis; it is not yet the requested
multi-variant image-quality evaluation or a full-model reference accuracy test.
No full-main build or git bisect was necessary: the controlled arithmetic
ablation isolates the defect, and main already contains its fix. The experiment
does not rebase or modify production model code to apply the fix.

## Implication for the frontier evaluation

Apply the same main-style model corrections to every attention variant and the
stock control, qualify full-block/reference and trace paths, then regenerate all
images and timings. Choose the conditioning setting independently and hold it
fixed across that suite. Main's third related fix (trace lifetime across dynamic
weight evictions) is not exercised here because dynamic loading is disabled.

The earlier D and stock 50-step images are invalid model-quality results; both
produce noise. Their CLIP and transformer performance should not be presented
as a functioning-model frontier comparison. The standalone attention-kernel
accuracy/performance experiments remain separate from this model-level bug.
