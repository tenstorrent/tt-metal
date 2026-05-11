# MoE PCC sweep — partial results

Sweep status: **stopped** at user request after positions 60-77 fully covered + pos 78 partial (layers 3-8). Did not extend to pos 128 since the data through pos 78 already shows no per-step PCC drop that would explain the kv_cache divergence.

Setup:
- `test_moe_fused_with_reduce` parametrized dynamically from `accepted_experts.json` so each (layer, pos) combo loads exactly the 8 experts the compressed pod selected at that layer/position.
- **Real weights** (`USE_RANDOM_WEIGHTS=False`, `DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized`).
- **Real BSPM** (`BSPM_DIR=/data/bliu/bit_sculpt/results/deepseek-r1-0528`, variant B, budget 3.5).
- TP8 compressed, slow dispatch, allocator mode HYBRID.

## Coverage (186 rows)

| Metric | Value |
|---|---|
| Position range covered | 60–78 (19 positions; pos 78 partial: layers 3-8 only) |
| Layers | 3-12 (all MoE layers) |
| Total rows | 186 |
| Status | 186/186 PASSED |
| **PCC range** | **0.975372 – 0.991904** |
| Mean PCC | 0.986492 |
| Rows with PCC < 0.97 | **0** |

## Per-layer PCC range

| Layer | min | max | mean | n | observation |
|---|---|---|---|---|---|
| 3  | 0.9852 | 0.9870 | 0.9864 | 19 | mid |
| 4  | 0.9769 | 0.9806 | 0.9789 | 19 | **second lowest** |
| 5  | 0.9854 | 0.9874 | 0.9864 | 19 | mid |
| 6  | 0.9814 | 0.9853 | 0.9826 | 19 | mid-low |
| 7  | **0.9754** | 0.9813 | **0.9784** | 19 | **lowest consistently** |
| 8  | 0.9914 | 0.9919 | 0.9916 | 19 | high |
| 9  | 0.9914 | 0.9919 | 0.9917 | 18 | high |
| 10 | 0.9866 | 0.9890 | 0.9882 | 18 | mid-high |
| 11 | 0.9896 | 0.9904 | 0.9901 | 18 | high |
| 12 | 0.9911 | 0.9915 | 0.9913 | 18 | high |

**Key finding so far**: layers 4 and 7 consistently have the worst PCC (~0.978-0.981). Other layers cluster around 0.987-0.992. No layer × position combination yet drops below 0.97 — i.e., the test still passes its 0.99 threshold (PCC > 0.97 for the routed-output PCC checked).

## Hypothesis vs evidence

The investigation goal was: find layers/positions where TP8/BSPM compression error in MoE output is bad enough to explain the kv_cache divergence (PCC=0.86 at seq-len 128 in `kv_cache_stage_01_layer_0.pt`).

Through positions 60-74, **no single layer × position MoE output dips below PCC 0.97**. Layers 4 and 7 are persistently lower than others (~0.978) but consistent across positions. This suggests:

- MoE per-step output error is modest and roughly constant across the prefill/decode boundary (no dramatic spike at the divergence position 67).
- Accumulated MoE error through the residual stream over many positions could still cause downstream KV-cache divergence even if each step's PCC is "fine".
- Or the divergence isn't primarily MoE compression — could be attention/dense compression, accumulating rounding, or specific expert sensitivity that this test setup doesn't surface.

## Raw data

`moe_pcc_sweep.jsonl` — one JSON line per test: `{layer, pos, pcc, status, elapsed_s}`.

The sweep will append rows as it runs. To reproduce or extend: see `models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py` (now parametrizes from this dir's `accepted_experts.json`) and `/data/bliu/sweep_moe_pos.sh` (idempotent — skips rows already in the results file).
