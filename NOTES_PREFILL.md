# GPT-OSS Prefill MoE Integration Notes

## Current State (2026-04-07 13:30)
**Track A COMPLETE**: EP=8 ndg=1 combine hang fixed.
**Track B COMPLETE (v1)**: DeepSeek prefill path integrated and wired into model.

### Validated:
- Isolated ops: PCC=0.9996 (random weights, mesh 4x8 and 8x4)
- Real weights: PCC=0.9954 (GPT-OSS layer 1, mesh 4x8, bfloat4_b quantized)
- Per-layer latency: 8.3ms (seq=128), 14.5ms (seq=512) on mesh (4,8)
- Model wiring: create_tt_model → Model → DecoderLayer → MLP → ThroughputExperts

### Remaining:
- Full demo test with use_deepseek_prefill=True (the model-level prefill input
  formatting is handled by the generator, not manually)
- To enable: add use_deepseek_prefill=True to create_tt_model call in text_demo.py

## Files Changed

### C++ (Track A — combine hang fix):
- reader_combine.cpp: num_dispatch_groups CT arg 33, mesh_col % ndg
- writer_combine.cpp: TensorAccessorArgs shift, array[2]→[4]
- combine_program_factory.cpp: compute + push num_dispatch_groups

### Python (Track B — integration):
- NEW: models/demos/gpt_oss/tt/experts_throughput/prefill_deepseek.py
- models/demos/gpt_oss/tt/experts_throughput/__init__.py
- models/demos/gpt_oss/tt/mlp.py (creates DeepSeekPrefillConfig)
- models/demos/gpt_oss/tt/layer.py (threads params)
- models/demos/gpt_oss/tt/model.py (threads params)
- models/demos/gpt_oss/tt/common.py (threads params)

### Test scripts:
- test_prefill_4x8.py: Isolated per-op timing + PCC (PASS)
- test_prefill_integration.py: Real-weight PCC test (PASS, 0.9954)
- test_prefill_realweights.py: Real-weight pipeline test (PASS)

## How to enable in demo
In text_demo.py, add to create_tt_model call:
  use_deepseek_prefill=True, prefill_seq_len=128
