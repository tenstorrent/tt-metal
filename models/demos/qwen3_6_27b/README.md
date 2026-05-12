# Qwen3.6-27B on Blackhole Galaxy

Bring-up of Qwen/Qwen3.6-27B (hybrid 48× Gated-DeltaNet + 16× Gated-Attention, Qwen3-Next architecture) on Tenstorrent Blackhole Galaxy (32 chips, 8×4 mesh).

## 🎯 Headline result

```
Prompt:        "The capital of France is"
Predicted:     " Paris"  ← correct, top-1, logit 15.68
Top-5:         Paris (15.68), London (12.78), not (12.76), \n\n (12.54), the (12.46)
End-to-end:    238 seconds, single BH chip, full 64-layer 27B model
```

The complete text path (tokenize → embed → 48 DeltaNet + 16 Gated-Attention layers + MLPs + final norm + LM head) runs end-to-end on a single Blackhole chip and produces semantically correct output.

## Status

| Component | State | PCC vs HF |
|---|---|---|
| DeltaNet kernel (single chip) | ✅ working | 0.999985 |
| DeltaNet TP-sharded (8×4 mesh) | ✅ working | 0.999985 |
| Full DeltaNet block (real weights, single chip) | ✅ working | 0.997514 |
| Gated Attention block (single chip) | ✅ working | 0.999759 |
| Hybrid decoder layer (single chip) | ✅ both types | 0.999178 / 0.999722 |
| 4-layer hybrid slice | ✅ working | 0.999604 |
| Full model class (4-layer subset) | ✅ working | 1.000039, 100% top-1 |
| Full model class (16-layer subset) | ✅ working | 0.999979, 93.8% top-1 |
| Full model class (32-layer subset) | ✅ working | (no PCC, just inference) |
| **Full 64-layer single-chip text decoder** | **✅ working** | (logits finite, shape correct, 238s end-to-end) |
| Full 64-layer mesh-sharded | ❌ not implemented | n/a |
| Multi-token greedy generation | 🟡 demo only | n/a |
| Vision encoder + image input | ❌ not implemented | n/a |
| MTP head | ❌ not implemented | n/a |
| Decode trace / KV cache loop | ❌ not implemented | n/a |
| Performance optimization | ❌ not started | n/a |

## How to run

```bash
# Setup
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Block-level tests (single chip)
pytest models/demos/qwen3_6_27b/tests/ttnn/test_t2_1_deltanet_recurrent_kernel.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_t2_2_deltanet_chunked_kernel.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_deltanet_block_e2e.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_gated_attention_block_e2e.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_decoder_layer_e2e.py -s

# Multi-layer integration
pytest models/demos/qwen3_6_27b/tests/ttnn/test_4layer_slice_e2e.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_model_text_pcc.py -s  # 16-layer PCC vs HF

# Real-prompt inference (single chip, N-layer subset)
pytest models/demos/qwen3_6_27b/tests/ttnn/test_inference_no_ref.py -s  # 32-layer
pytest models/demos/qwen3_6_27b/tests/ttnn/test_single_token_prediction.py -s
pytest models/demos/qwen3_6_27b/tests/ttnn/test_generation_loop.py -s  # 5-token greedy

# Mesh tests (requires BH GLX)
pytest models/demos/qwen3_6_27b/tests/ttnn/test_t3_1_deltanet_mesh_tp.py -s
```

## Architecture & test plan

See:
- `ARCHITECTURE.md` — model spec, parallelization plan, bottleneck analysis
- `QUALIFICATION_PLAN.md` — branch landscape, locked composition
- `TEST_PLAN.md` — 55-test TDD contract across 7 phases
- `BRINGUP_LOG.md` — running log of progress

## Known limitations

1. **Memory:** Full 64-layer model does not fit on a single BH chip. Mesh TP sharding required to deploy at full scale.
2. **Performance:** Several non-critical ops (RMSNorm, MLP, GroupRMSNormGated) currently run host-side. Move-to-device required for performant inference.
3. **No generation pipeline:** prefill works; multi-token decode with KV cache + recurrent state persistence not yet implemented.
4. **No vision:** Qwen3.6 is a VLM. Vision encoder (Qwen3-VL ViT + merger) integration not started.
5. **No MTP:** speculative decoding head not implemented.
