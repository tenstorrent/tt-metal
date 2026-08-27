# SFPU LUT retune — figure assets

Rendered figures for **PR #54602** (`ldjurovic/sfpu_constants_retune`), which retunes the
Wormhole SFPU LUT coefficient tables for `sigmoid_appx`, `tanh` (APPROXIMATION_MODE) and
`gelu_appx`.

They live on this orphan branch rather than in the PR so the review diff stays limited to the
kernel change and its writeup. The PR description embeds them from
`raw.githubusercontent.com`.

Measured on an n300 (Wormhole B0). Regenerate with `sfpu_lut_data/plot_lut.py` from the PR
description's harness listing; the plots are a pure function of the captured JSON.

Nothing builds against this branch. It is documentation media only, safe to delete once
PR #54602 is closed.
