# Latest decode perf profile — `ropecse` (HEAD: router fusion + stat-gather tile repack + RoPE CSE)

Per-phase device-time profile of **one decode step** on the **2-layer graph** (1 dense + 1 MoE),
captured with tracy + `tt-perf-report` windowed on the signposts. DeepSeek-V3, Blackhole 4×8 galaxy.

Files per phase (`full`, `prologue`, `attn0`, `dense`, `attn1`, `moe`, `lmhead`):
- `*.summary.csv` — per-op device-time, structured.
- `*.summary.png` — stacked device-time bar chart (Compute / DM / TM / Other).
- `*.stdout.txt` — full per-op report incl. inline SLOW / BW / FLOPs / Math-Fidelity hints.

Notes:
- This is the **measured 2-layer** profile (`full` total ≈ 7418 µs); the full-61-layer projection
  (≈158.5 ms) is computed from it via `prologue + attn×61 + dense×3 + moe×58 + lm_head`.
- On this particular run the **MoE-internal op attribution is noisy** (`AllBroadcast`,
  `UntilizeWithUnpadding` inflated vs other runs — run-to-run re-attribution of `moe_compute`
  internals; `MoECompute` itself is stable ~698 µs). The per-op deltas of the *changed* ops
  (AllGather, Repeat, ReduceScatter) are the reliable signal, not single-run phase totals.
