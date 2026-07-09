# HunyuanImage-3.0 — TTNN transformer pipeline (`tencent/HunyuanImage-3.0`)

A native TTNN bring-up of the transformer decoder-block of
**HunyuanImage-3.0** (`HunyuanImage3ForCausalMM`), an 80B-class mixed-MLP MoE
causal multimodal model (text + image tokens; `model_type=hunyuan_image_3_moe`).

- 32 decoder layers · hidden 4096 · 32 q-heads / 8 kv-heads · head_dim 128
- MoE: **64 routed + 1 shared** SwiGLU experts, **top-8** routing, `norm_topk_prob`
- 2D-RoPE · qk-norm · RMSNorm · non-causal joint attention

The real `model.generate()` is a 50-step **diffusion image** loop over the full
80B stack — not device-feasible and not the gate target. The bring-up graduated
the transformer decoder-block internals, so the faithful, device-feasible e2e is
the model's **real transformer forward** (the exact `HunyuanImage3Model.forward`
path that feeds the CausalMM / image heads).

## Call-1 — `hunyuan_image3_transformer_prefill`

```
input_ids (HF tokenizer)  --ttnn.embedding-->  inputs_embeds
    --> image3_decoder_layer x N  -->  last_hidden_state   (+ summed MoE l_aux)
```

Graduated stubs, composed along the real HF nesting
(`HunyuanImage3DecoderLayer.mlp == HunyuanMoE`; `HunyuanMoE.gate == HunyuanTopKGate`):

| graduated stub | role | on the forward path |
|---|---|---|
| `image3_decoder_layer` | RMSNorm + GQA attn + 2D-RoPE + qk-norm + SDPA + residuals | layer output = `last_hidden_state` |
| `mo_e` | shared SwiGLU + 64 routed expert SwiGLUs, combined by the router | feeds the post-attn residual |
| `top_k_gate` | softmax + top-8 → **router weights** (feed expert combine) + `l_aux` | router → experts → hidden (main path) |

Every graduated stub is invoked on this one real forward path with its output
feeding downstream computation (no coverage sweep). All ops are native ttnn.

## Package layout

```
models/demos/vision/generative/hunyuanimage_3_0/
  tt/pipeline.py        ONE shared chained pipeline (build_pipeline factory,
                        run_prefill, per-stage trace+2CQ hooks, selftests).
                        BOTH the demo and the e2e test import & call this.
  _stubs/               the three graduated stubs (composed along the HF nesting)
  demo/demo_image3_prefill.py   runnable `python -m ...demo.demo_image3_prefill`
  tests/e2e/test_e2e_prefill.py Gate 1/2/3 on the shared pipeline
  tests/e2e/test_trace_2cq.py   Command-3 trace + host-op selftests
  tests/pcc/            per-component PCC tests (from bring-up)
  e2e_plan.json         the planner sketch (Command 1)
```

## Results (Blackhole, single chip, N=1 layer, seq_len=64)

| gate | metric | result |
|---|---|---|
| Gate 1 (native) | `host_op_selftest` host aten ops | **0** (fully on device) |
| Gate 2 (invoked) | graduated stubs invoked | `image3_decoder_layer`, `mo_e`, `top_k_gate` (all) |
| Gate 3 (PCC) | e2e PCC(last_hidden_state) vs HF golden | **0.9997** (≥ 0.95) |
| — | MoE `l_aux` (tt vs ref) | 37.50 vs 37.15 |
| per-component | image3_decoder_layer / mo_e / top_k_gate PCC | 0.99999 / 0.9996 / 1.0 |
| Command 3 (trace) | prefill / decode trace capture, host-free PCC | 1.0 / 1.0 |

## Run

```bash
# e2e gates (real input -> chained stubs -> real output, PCC vs HF golden)
./python_env/bin/python -m pytest \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_e2e_prefill.py -s

# trace+2CQ + fully-on-device selftests (Command 3)
./python_env/bin/python -m pytest \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_trace_2cq.py -s

# per-component PCC
./python_env/bin/python -m pytest \
  models/demos/vision/generative/hunyuanimage_3_0/tests/pcc -s

# demo (real prompt -> TT prefill -> real last_hidden_state; --compare adds PCC)
./python_env/bin/python -m \
  models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_prefill \
  --prompt "A serene mountain lake at sunrise" --compare
```

Tunables: `HUNYUAN_E2E_NUM_LAYERS` (default 1 — the full 32-layer stack is the
whole 80B model and does not fit on one chip; each graduated layer is a real
transformer block), `HUNYUAN_E2E_SEQ_LEN` (default 64).

## Notes / honest simplifications

- **Golden** = the real `HunyuanImage3DecoderLayer` forward over the first `N`
  layers on the same `inputs_embeds` + real 2D-RoPE (`build_2d_rope`), plus the
  reference `HunyuanTopKGate` `l_aux`. `attention_mask=None` (matches the
  graduated component's non-causal reference); image gen skips `ln_f`, so the
  stack output IS `last_hidden_state`.
- **decode stage:** the graduated attention is full-SDPA with no incremental KV
  cache (the reference component ran non-causal), so the `decode` PIPELINE_STAGE
  reuses the pinned-C decoder block reading the resident buffers rather than an
  incremental single-token KV read. It is a real host-op-free fixed-shape
  forward and is printed by `trace_capture_selftest` (never silently dropped).
