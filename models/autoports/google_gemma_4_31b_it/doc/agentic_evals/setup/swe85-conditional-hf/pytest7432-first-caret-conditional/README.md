# Pytest7432 post-first-caret conditional HF candidate

This candidate asks whether pinned HF continues the full-TT seed44 caret branch after being conditioned on the first full-TT generated token. The original shortened suffix512 prompt IDs are copied exactly, then full-TT token ID `172109` is appended; the chosen continuation is shifted from 64 IDs to the remaining 63. The frozen command requests one HF CPU free sample with seed44 and cap512.

The prior H9 HF output remains preserved. Its seed44 free sample took a different first-token branch (`236884`) and naturally stopped at307 tokens. Its teacher-forced result gives full-TT's first token ID172109 probability0.10442692, with top1 agreement at all64 chosen positions and 0.070656 first-position HF-minus-TT log-probability delta; positions1–63 have top1 token172109. This makes branch divergence a plausible hypothesis, not a proven mechanism.

This is a conditional continuation probe only. It cannot establish equal-input incidence or native-prompt equivalence, confirm a stack bug by itself, replace the prior HF sample, alter a score, or change policy. No result file exists. The candidate has no driver or admission. The command is frozen for review only; the host validation used `--validate-only`, which reported that no model was loaded.

Forecast uses a prior one-case HF512 measurement: 1032.38s free-run plus about19.46s non-free load/prefill/teacher-forcing overhead; planning range841.91–1268.86s. The actual input is one token longer and may stop earlier or later.
