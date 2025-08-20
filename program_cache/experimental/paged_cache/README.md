### Program-cache review for `experimental/paged_cache`

Findings:
- Identified a cache-hit issue when toggling `batch_idx` mode (tensor vs scalar) for FILL across runs.
- The writer kernel compile-time args encode the mode; the override cannot adjust compile-time paths on cache-hit. Hash currently does not ensure separation for the two modes in FILL.

Tests:
- failures/paged_fill_toggle_batch_idx_mode/test_paged_fill_toggle_batch_idx_mode.py: exposes PCC mismatch on cache-hit when switching modes.

Suggested fixes:
- Add mode to program hash or unify paths to a single runtime mechanism.
