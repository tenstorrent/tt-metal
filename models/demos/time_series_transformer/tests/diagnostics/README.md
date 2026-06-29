# Diagnostics

These scripts were used to root-cause two bugs during development:

1. **Attention scale factor bug** — self-attention and cross-attention were
   scaling by the padded tile width (`HEAD_DIM_PADDED ** -0.5`, i.e. 32) instead
   of HF's true per-head width (`HEAD_DIM_TRUE ** -0.5`, i.e. 13). Confirmed via
   direct hardware measurement (ratio 0.637377 = sqrt(13/32)). Fixed in
   `tt/tst_attention.py`.

2. **out_proj weight layout bug** — `_build_out_proj`'s input-dimension padding
   assumed `concatenate_heads`' output packs real head data contiguously
   (`[0:26]`), but it actually packs per-head (`[0:13]` real, `[13:32]` padding,
   `[32:45]` real, `[45:64]` padding). This silently dropped head 1's entire
   contribution. Fixed in `tt/tst_model.py` (`_pad_input_per_head`).

Both bugs are fixed in the shipped code (`tt/`). These files are **not** part
of the bounty acceptance test suite — see `tests/test_tst_pcc.py`,
`tests/test_tst_e2e.py`, and `tests/test_tst_perf.py` for that, and
`README.md` at the repo root for the documented `pytest tests/` command,
which intentionally does not descend into this folder by default.

Kept for provenance / debugging history. Several of these scripts predate
fixes that are now in `tt/` and will report stale or misleading results if
re-run (e.g. `test_out_proj_weight_layout.py` investigates a bug that no
longer exists in `_build_out_proj`).
