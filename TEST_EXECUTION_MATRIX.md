# Test Execution Matrix

**Purpose:** Test schedule and expected results by week  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md Part C

---

## Test Suite Overview

### Total Test Count by Category

| Category | Count | Location |
|----------|-------|----------|
| Unit Tests | 30+ | `comfyui_bridge/tests/test_*.py` |
| Integration Tests | 20+ | `comfyui_bridge/tests/test_integration_*.py` |
| Performance Tests | 10+ | `comfyui_bridge/tests/benchmark_*.py` |
| Regression Tests | 10+ | `comfyui_bridge/tests/test_regression_*.py` |
| **Total** | **70+** | |

---

## Phase 0 Tests

### Day 1-2: Scheduler Sync Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_timestep_passed_correctly` | Verify timestep transfer | timestep matches input |
| `test_sigma_passed_correctly` | Verify sigma transfer | sigma matches input |
| `test_full_20_step_sequence` | Verify complete sequence | All 20 steps process |
| `test_error_on_missing_timestep` | Error handling | Clear error message |
| `test_error_on_invalid_sigma` | Error handling | Clear error message |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_scheduler_sync.py -v
```

### Day 1: IPC Baseline Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `benchmark_full_loop_latency` | Measure baseline | ~2000-3000ms for 20 steps |
| `benchmark_ipc_roundtrip` | Measure IPC overhead | < 10ms P99 |
| `calculate_per_step_budget` | Calculate budget | Per-step < 150ms |

**Run Command:**
```bash
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_ipc.py
```

### Day 3-5: ControlNet Feasibility Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_control_hint_transfer` | Verify tensor transfer | Data integrity preserved |
| `test_control_hint_shape` | Verify shape preservation | Shape matches original |
| `test_control_hint_integration` | Verify UNet accepts | No errors, output produced |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_controlnet_feasibility.py -v
```

---

## Week 1 Tests

### Per-Step API Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_denoise_step_single_basic` | Basic execution | Success response |
| `test_denoise_step_single_format` | Output format | [B, C, H, W] shape |
| `test_denoise_step_single_scheduler_state` | Scheduler params | Timestep/sigma used |
| `test_model_agnostic_config_sdxl` | SDXL config | 4 channels |
| `test_model_agnostic_config_sd35` | SD3.5 config | 16 channels |
| `test_format_conversion_roundtrip` | Format helpers | Lossless conversion |
| `test_per_step_matches_full_loop` | SSIM validation | SSIM >= 0.99 |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_per_step.py -v
```

### Pass Criteria

| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| SSIM vs full-loop | >= 0.99 |
| Format conversion | Lossless |

---

## Week 2 Tests

### Session Management Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_session_create` | Create session | session_id returned |
| `test_session_step` | Execute step | Step completes |
| `test_session_complete` | Complete session | Stats returned |
| `test_session_lifecycle` | Full lifecycle | Create->Step->Complete |
| `test_session_timeout` | Timeout expiry | Session expires |
| `test_session_cleanup` | Resource release | Memory freed |
| `test_concurrent_sessions` | Multiple sessions | All work correctly |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_session.py -v
```

### Error Handling Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_error_session_not_found` | Invalid session | Clear error |
| `test_error_model_mismatch` | Wrong model | Clear error |
| `test_error_step_out_of_order` | Wrong step | Warning, continues |
| `test_error_format_invalid` | Bad tensor | Clear error |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_error_handling.py -v
```

### Performance Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_per_step_latency_overhead` | Overhead measurement | < 10% |
| `test_memory_stability_100` | Memory test | No leaks |

**Run Command:**
```bash
python /home/tt-admin/tt-metal/comfyui_bridge/tests/benchmark_per_step.py
```

### Pass Criteria

| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| Latency overhead | < 10% |
| Memory stability | 100 gen, no leaks |

---

## Week 3 Tests

### ControlNet Integration Tests

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_controlnet_canny_basic` | Canny ControlNet | Output produced |
| `test_controlnet_canny_ssim` | Canny quality | SSIM >= 0.90 |
| `test_controlnet_depth_basic` | Depth ControlNet | Output produced |
| `test_controlnet_depth_ssim` | Depth quality | SSIM >= 0.90 |
| `test_controlnet_openpose_basic` | OpenPose ControlNet | Output produced |
| `test_controlnet_openpose_ssim` | OpenPose quality | SSIM >= 0.90 |
| `test_multi_controlnet` | Multiple ControlNets | Combined output |

**Run Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_controlnet.py -v
```

### Human Validation Tests (Manual)

| Test | Raters | Question | Target |
|------|--------|----------|--------|
| Canny validation | 5 | "Does output follow edges?" | 5/5 correct |
| Depth validation | 5 | "Does output follow depth?" | 5/5 correct |
| OpenPose validation | 5 | "Does output follow pose?" | 5/5 correct |

### Pass Criteria

| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| Canny SSIM | >= 0.90 |
| Depth SSIM | >= 0.90 |
| OpenPose SSIM | >= 0.90 |
| Human validation | 5/5 per type |

---

## Week 4 Tests

### Comprehensive Test Suite

| Test Category | Count | Run Command |
|---------------|-------|-------------|
| Unit Tests | 30+ | `pytest tests/test_*.py -v` |
| Integration Tests | 20+ | `pytest tests/test_integration_*.py -v` |
| Performance Tests | 10+ | `python tests/benchmark_*.py` |
| Regression Tests | 10+ | `pytest tests/test_regression_*.py -v` |

**Full Suite Command:**
```bash
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/ -v --tb=short
```

### Pass Criteria

| Metric | Target |
|--------|--------|
| Overall pass rate | >= 95% |
| Zero critical failures | 0 |
| Performance within budget | Yes |

---

## Week 5 Tests

### Final Validation Suite

| Test Name | Purpose | Expected Result |
|-----------|---------|-----------------|
| `test_ssim_100_prompts` | 100 prompt validation | All SSIM >= 0.99 |
| `test_controlnet_30_images` | 30 image validation | All SSIM >= 0.90 |
| `stress_test_1000_generations` | Robustness | 0 crashes |
| `memory_test_1000_generations` | Memory stability | 0 leaks |
| `regression_txt2img` | txt2img unchanged | Pass |
| `regression_img2img` | img2img unchanged | Pass |
| `regression_vae_decode` | VAE unchanged | Pass |

**Run Commands:**
```bash
# SSIM validation
python /home/tt-admin/tt-metal/comfyui_bridge/tests/validate_ssim.py

# Stress test
python /home/tt-admin/tt-metal/comfyui_bridge/tests/stress_test.py

# Regression tests
pytest /home/tt-admin/tt-metal/comfyui_bridge/tests/test_regression_*.py -v
```

### Pass Criteria

| Metric | Target |
|--------|--------|
| Per-step SSIM | >= 0.99 (100 prompts) |
| ControlNet SSIM | >= 0.90 (30 images) |
| Stress test | 1000 gen, 0 crashes |
| Memory | < 100MB growth |
| Regression | All pass |

---

## Master Test List

### All Tests (70+)

**Unit Tests (30+):**
1. `test_denoise_step_single_basic`
2. `test_denoise_step_single_format`
3. `test_denoise_step_single_scheduler_state`
4. `test_model_agnostic_config_sdxl`
5. `test_model_agnostic_config_sd35`
6. `test_model_agnostic_config_sd14`
7. `test_format_conversion_roundtrip`
8. `test_format_conversion_shape`
9. `test_format_conversion_dtype`
10. `test_session_create`
11. `test_session_step`
12. `test_session_complete`
13. `test_session_timeout`
14. `test_session_cleanup`
15. `test_error_session_not_found`
16. `test_error_model_mismatch`
17. `test_error_step_out_of_order`
18. `test_error_format_invalid`
19. `test_error_missing_params`
20. `test_timestep_validation`
21. `test_sigma_validation`
22. `test_guidance_scale_validation`
23. `test_control_hint_parsing`
24. `test_control_hint_shape`
25. `test_control_hint_dtype`
26. `test_shm_read_write`
27. `test_tensor_transfer`
28. `test_config_lookup`
29. `test_config_missing_model`
30. `test_logging_format`

**Integration Tests (20+):**
31. `test_per_step_matches_full_loop`
32. `test_session_lifecycle`
33. `test_concurrent_sessions`
34. `test_controlnet_canny_basic`
35. `test_controlnet_canny_ssim`
36. `test_controlnet_depth_basic`
37. `test_controlnet_depth_ssim`
38. `test_controlnet_openpose_basic`
39. `test_controlnet_openpose_ssim`
40. `test_multi_controlnet`
41. `test_10_step_generation`
42. `test_20_step_generation`
43. `test_50_step_generation`
44. `test_seed_reproducibility`
45. `test_different_resolutions`
46. `test_batch_size_1`
47. `test_cfg_scale_variations`
48. `test_negative_prompt`
49. `test_empty_prompt`
50. `test_long_prompt`

**Performance Tests (10+):**
51. `benchmark_full_loop_latency`
52. `benchmark_per_step_latency`
53. `benchmark_ipc_roundtrip`
54. `benchmark_format_conversion`
55. `benchmark_shm_transfer`
56. `benchmark_session_overhead`
57. `benchmark_controlnet_injection`
58. `benchmark_memory_100_gen`
59. `benchmark_concurrent_sessions`
60. `benchmark_throughput`

**Regression Tests (10+):**
61. `test_regression_txt2img`
62. `test_regression_img2img`
63. `test_regression_vae_decode`
64. `test_regression_clip_encode`
65. `test_regression_model_load`
66. `test_regression_api_compatibility`
67. `test_regression_error_format`
68. `test_regression_logging`
69. `test_regression_config`
70. `test_regression_shm`

---

## Pass/Fail Criteria Summary

| Phase | Tests | Required Pass Rate | Critical Failures |
|-------|-------|-------------------|-------------------|
| Phase 0 | ~15 | 100% | 0 |
| Week 1 | ~10 | 100% | 0 |
| Week 2 | ~15 | 100% | 0 |
| Week 3 | ~10 | 100% | 0 |
| Week 4 | ~70 | >= 95% | 0 |
| Week 5 | ~70 | >= 95% | 0 |

---

**Document Version:** 1.0  
**Created:** December 16, 2025  
**Reference:** PHASE_1_BRIDGE_EXTENSION_PROMPT.md
