# d1-3B on N150 — accuracy and performance

Single Wormhole chip (N150), batch 1, prefill only, bf16 weights. Measured on the first bring-up; numbers are
reported, not promised.

## Accuracy

Device output vs the Hugging Face reference (`LiquidAI/d1-3B`, `trust_remote_code=True`), over the 59 golden
questions in `tests/examples.json` (yes/no, choice, score; text, JSON, nested, multilingual, long states, and
images). PCC, top-1 token and the decision are checked against the **fp32** reference; probability closeness is
gated against the **bf16** reference (the device runs bf16).

| Set | Questions | min hidden PCC | min logits PCC | top-1 token | decision (argmax) | max \|Δp\| | mean \|Δp\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text | 53 | 0.9990 | 0.9993 | 100% | 100% | 0.033 (vs bf16) | 0.0052 |
| Image | 6 | 0.9993 | 0.9998 | 100% | 100% | 0.005 (vs fp32) | 0.0017 |

Every decision matches the reference. The largest probability drift (0.033, one high-confidence 4-way choice)
is at the bf16 rounding floor: HF's own fp32→bf16 drift reaches 0.028 with no decision flips
(`test_decision.py::test_bf16_reference_drift`).

Per-module PCC vs the torch reference (reused from `lfm25_vl`, run against the d1-3B checkpoint on N150):

| Module | PCC |
|---|---:|
| ShortConv | ~0.9998 |
| MLP | ~0.9996 |
| Projector | ~0.9999 |

## Performance

Traced prefill, device-only (trace replay + answer-row readback), p50 of 20 runs:

| Bucket (tokens) | p50 (ms) | p90 (ms) |
|---:|---:|---:|
| 128 | 52.4 | 52.7 |
| 256 | 65.9 | 66.4 |
| 512 | 92.1 | 92.4 |
| 1024 | 151.0 | 153.7 |
| 2048 | 283.1 | 293.2 |
| 4096 | 604.7 | 612.6 |

End-to-end one short question (39 tokens: render + tokenize + prefill + readout): **52.6 ms** p50.

### Context

Vendor anchors for one short question (from the model card): 8 ms on an RTX 4090 (bf16, CUDA graphs), 50 ms on
an NVIDIA Jetson Orin Nano. The N150 short-question latency (~52 ms) is in Orin Nano territory for this first
bring-up.

A short question lands in the 128-token bucket, which is weight-bound: reading the ~4.9 GB bf16 text weights
once at 288 GB/s is ~17 ms, so there is headroom below the measured 52 ms. Next steps: BFP8 MLP weights, and
packing several questions over one state into a single pass (as the HF tree path does) instead of one prefill
per question.

Reproduce:

```bash
MESH_DEVICE=N150 pytest models/demos/multimodal/d1_3b/tests/test_model.py -s   # accuracy
MESH_DEVICE=N150 pytest models/demos/multimodal/d1_3b/tests/test_perf.py  -s   # latency
```
