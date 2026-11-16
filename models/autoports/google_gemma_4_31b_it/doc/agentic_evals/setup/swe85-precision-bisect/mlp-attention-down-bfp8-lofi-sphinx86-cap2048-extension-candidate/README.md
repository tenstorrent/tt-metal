# P8 Sphinx call86 cap2048 extension (inert)

One-case, full-mode cap-extension candidate for P3 fixed108 index74, Sphinx11510 call86 seed44. It uses the exact existing 1024-cap fixture and P8 policy, requesting 2048 free-run tokens. The input fixture is unchanged.

The completed P8 cap1024 output produced 1024 tokens without EOS50. Its SHA-pinned token sequence is recorded in `forecast.json` as the expected prior prefix check. This command reruns from the prompt; it does not resume from token1024. Point scenario 244.7 s, range 190–450 s, including one measured P8 model load; estimate is planning-only.

No admission or run artifacts are present.
