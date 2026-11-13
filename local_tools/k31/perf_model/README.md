# K3.1 analytical prefill perf model (local only)

Removed from PR #64830 (tt_metal_tracker-8v1.1, review M1); kept here as a local diagnostic.
Source: tt-metal branch mvasilijevic/k31/main at 70d4fe79170 (last change b5eb39172d3).

- `kimi_k31_perf_model.py` was `models/demos/deepseek_v3_d_p/utils/kimi_k31_perf_model.py`
- `test_k31_perf_model.py` was `models/demos/deepseek_v3_d_p/tests/kimi_k3/test_k31_perf_model.py`

To use: copy both back to those paths in a tt-metal checkout. The model reads
`models/demos/deepseek_v3_d_p/common/prefill/runners/ci/perf_baselines.json` relative
to its own location and imports `kimi_k3_1_config` and `kv_cache_utils`.
