# SFPU LUT retune — figure assets

Rendered figures for **PR #54602** (`ldjurovic/sfpu_constants_retune`), which retunes the
Wormhole SFPU LUT coefficient tables for `sigmoid_appx`, `tanh` (APPROXIMATION_MODE) and
`gelu_appx`.

They live on this orphan branch rather than in the product tree so the review diff — and the
merged tree — stay limited to the kernel change, its writeup and one regression test. Both the
PR description and the committed writeup
(`tech_reports/SFPU_LUT_Retune_Wormhole/SFPU_LUT_RETUNE_WORMHOLE.md`) embed them from
`raw.githubusercontent.com`, **pinned to a commit SHA rather than to this branch name**, so a
later commit here cannot silently change what the writeup shows.

Both formats are here for a reason: the `.svg` files are the originals, and the `.png` renders
are what the links use, because GitHub serves `raw.githubusercontent.com` SVG as `text/plain`
and its image proxy will not render it.

> **Retention:** this branch is **not** safe to delete. `tech_reports/SFPU_LUT_Retune_Wormhole/`
> is merged into `main` and its five figures resolve here. Deleting this branch, or garbage-
> collecting the commit the links pin, leaves the merged writeup with five broken images.
> Nothing *builds* against this branch — it is documentation media only — but it is load-bearing
> for the docs.

Measured on an n300 (Wormhole B0). Regenerate with `sfpu_lut_data/plot_lut.py` from the PR
description's harness listing; the plots are a pure function of the captured JSON.
