# Goal: GLM-4.7-Flash TTNN Modularization and Optimization

## Objective
Refactor the monolithic GLM-4.7-Flash (`glm4_moe_lite`) TTNN implementation into a clean, modular architecture, then optimize decode performance.

## Model
- **Name**: GLM-4.7-Flash (ChatGLM4 MoE variant)
- **HuggingFace**: `zai-org/GLM-4.7-Flash`
- **Architecture**: 47 decoder layers, MLA attention, SwiGLU MLP, MoE (sparse + shared experts)
- **Key dims**: hidden=2048, q_lora_rank=512, kv_lora_rank=512, qk_rope_head_dim=64, 32 heads, 8 routed experts

## Success Criteria

### Phase 2 (Profile Baseline)
- [ ] Tracy ops report for full model decode (4 tokens)
- [ ] Tracy ops report for single layer decode
- [ ] Per-op kernel durations recorded in baseline.yaml
- [ ] Top-5 bottleneck ops identified

### Phase 3 (Modularize)
- [ ] `runtime_config.py` extracted (all env vars in one place)
- [ ] `linear_helpers.py` extracted (matmul wrappers)
- [ ] `attention/` package extracted (Q path, KV path, FlashMLA, output proj)
- [ ] `mlp/` package extracted (dense MLP, DRAM-sharded MLP)
- [ ] `decoder_layer_tt.py` simplified to ~150-line orchestrator
- [ ] `model_tt.py` split (decode_trace.py, mtp.py extracted)
- [ ] ALL existing tests still pass after each extraction

### Phase 4 (Optimize)
- [ ] Decode latency per layer < target (TBD after profiling)
- [ ] No PCC regressions (> 0.999 for dense layers, > 0.99 for MoE)

## Constraints
- No regression on existing tests
- Preserve all env var knobs (move to runtime_config, don't remove)
- Keep vLLM integration working
- Keep fused ops (kv_cache_branch, pre_sdpa) working

## Preferred Implementation Order
1. Profile baseline (establish numbers before changing anything)
2. Extract `runtime_config.py` (lowest risk, highest impact on readability)
3. Extract `linear_helpers.py` (prerequisite for attention/mlp extraction)
4. Extract `attention/` package
5. Extract `mlp/` package
6. Simplify `decoder_layer_tt.py`
7. Split `model_tt.py`
8. Profile again, compare to baseline
9. Optimize bottlenecks

## Key Test Commands
```bash
# Smoke test (fastest)
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
  pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_decode_update_cache_optional.py -v

# MoE layer
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
  pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_optional.py -v

# Full model
python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py --max-new-tokens 4

# Profile single layer (pytest is a module, so -m works)
TT_METAL_DEVICE_PROFILER=1 TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
  python -m tracy -v -r -p -n glm4_layer0 \
  -m "pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_decode_update_cache_optional.py -v"

# Profile full model (standalone script, no -m flag)
TT_METAL_DEVICE_PROFILER=1 \
  python -m tracy -v -r -p -n glm4_full \
  models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py --max-new-tokens 4
```
