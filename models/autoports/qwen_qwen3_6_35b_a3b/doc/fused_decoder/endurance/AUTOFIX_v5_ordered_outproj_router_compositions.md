# Ordered output-projection plus alternate-router controls

These two diagnostic snapshots are rejected after completed model tests, as detailed below. They start from [ordered output projection K512](v5_ordered_outproj_k512.py.snapshot), SHA256 `7f7fe4000c728ab1f0450c52d430898676b317c2556115da329f0f910394c60d`. They replace **only** the decode `s == 1` router body in `_moe`, retaining all output-projection, upstream precision, residual, prefill-router and expert-consumer code from that base. No added persistent allocations or host runtime operations are introduced.

- [v5_ordered_outproj_k512_router_bf16_topk.py.snapshot](v5_ordered_outproj_k512_router_bf16_topk.py.snapshot), SHA256 `c3a8afc865ea8d509ce79b3d79e3aa27c5ede48300e07e98273bcdc49f341e37`; exact router donor [v5_router_bf16_topk.py.snapshot](v5_router_bf16_topk.py.snapshot).
- [v5_ordered_outproj_k512_router_fp32_softmax_topk.py.snapshot](v5_ordered_outproj_k512_router_fp32_softmax_topk.py.snapshot), SHA256 `e9802cb27352d78eecbbe3a3a7bb8580763a4f104e3bd95397c08963f4bbf220`; exact router donor [v5_router_fp32_softmax_topk.py.snapshot](v5_router_fp32_softmax_topk.py.snapshot).

Each sibling `.py.provenance.json` records the base/donor/candidate hashes and two independent AST checks: the inserted router body matches the donor, and restoring the original router body recovers the entire base module. CPU syntax and both composition checks pass. This agent performed no device execution and changed no production or core harness source.

The isolated donor routers previously failed actual layer 34 model gates and are not claimed to repair this combined source. These controls tested their interaction with ordered output projection; the completed unchanged model gates fail for both combinations. Native top-k tie behavior is unchanged by these model-level compositions.

Parent-only commands (use fresh outputs):

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_outproj_k512_router_bf16_topk.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer 34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_outproj_k512_router_bf16_topk_layer 34_seed154.json

HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.stress_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_outproj_k512_router_fp32_softmax_topk.py.snapshot \
  --layer 34 --seed 154 --prompt-length 128 --decode-length 8192 --continuation-length 33 --repeats 1 \
  --oracle-cache models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/oracle_cache/v5_layer 34_seed154_8192.pt \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v5_ordered_outproj_k512_router_fp32_softmax_topk_layer 34_seed154.json
```

## Completed component and model decisions

The [ordered output-projection component report](v5_ordered_outproj_device.json) shows 335 native BF16 differences versus CPU HF on the same captured input, and **zero across all 2048 columns** for both ordered K512 and K1024. Sampled ordered FP32 references also match. This is a local arithmetic result; neither schedule passes the full model gate, and component latency was not measured. Each ordered schedule adds 16 MiB DRAM weights, with capacity qualification pending.

| Actual layer 34 seed 154, 8192+33 control | Failing rows | Minimum contribution PCC | Minimum position | Position 181 contribution | Position 7944 contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordered_outproj_k512 | 201 | 0.912519296376 | 7944 | 0.999988913130 | 0.912519296376 |
| ordered_outproj_k1024 | 201 | 0.912519296376 | 7944 | 0.999988913130 | 0.912519296376 |
| combined_accurate_residual_explicit_gate_norm | 219 | 0.912637764596 | 7944 | 0.992933785854 | 0.912637764596 |
| ordered_outproj_k512_router_bf16_topk | 203 | 0.946055341978 | 5799 | 0.999989058666 | 0.986205319036 |
| ordered_outproj_k512_router_fp32_softmax_topk | 200 | 0.930944200802 | 8112 | 0.992917669139 | 0.986205319036 |

**All five completed model controls fail the unchanged 0.995 quality gate and remain unselected.** The BF16-topk composition repairs 181 but not 7944 or the stream; the FP32-softmax/topk composition still fails 181. Better worst-token PCC, exact local output projection, or restored membership at one position is insufficient for decoder acceptance. No tolerance waiver or completion claim is made.

The [decision ledger](v5_ordered_outproj_router_compositions.decision.json) and composition provenance retain exact source identities and raw JSON/provenance/log hashes. Both one-repeat cached-oracle compositions are diagnostic controls, not final acceptance. The explicit gated-norm result is included to close the preceding precision hypothesis: its component BF16 normalization improvements still leave 219 failing model rows. A separate fresh aggregate layer 34 regression is parent-owned and remains outside this report. No additional candidate is prepared by this task.
