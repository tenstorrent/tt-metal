# Quick Start Guide - AllGather_preff1/3 Fused Op Test

## Prerequisites
- In docker container with python_env activated
- All environment variables set (see full guide for details)

## Quick Test Commands

### 1. Run Basic Unit Test (Decode)
```bash
cd /home/models-team/hzhou/tt-metal
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "decode and 1 and program_cache and eager" -v
```

### 2. Run Basic Unit Test (Prefill)
```bash
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "prefill and 128 and program_cache and eager" -v
```

### 3. Run All Unit Tests (Except Long Seq)
```bash
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3 -k "not 131072" -v
```

### 4. Run Device Perf Test
```bash
pytest models/demos/deepseek_v3/tests/unit/fused_op_unit_tests/mlp/test_ds_fused_all_gather_preff1_3.py::test_ds_fused_all_gather_preff1_3_device_perf -k "decode and 1" -v
```

### 5. Verify with Module Test
```bash
# First, temporarily modify mlp.py (see VERIFICATION_GUIDE for details)
pytest models/demos/deepseek_v3/tests/test_mlp.py::test_forward_pass[decode-32-MLP-None-device_params0] -v
# Then revert mlp.py changes
```

## Expected Results

- **PCC:** Should be > 0.9999 (likely 1.0)
- **Test Status:** PASSED
- **Time:** ~1-2 minutes per test

## Environment Setup (If Needed)

```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/models-team/hzhou/tt-metal
export PYTHONPATH=/home/models-team/hzhou/tt-metal
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache
export MESH_DEVICE=TG
export TT_METAL_RUNTIME_ROOT=/home/models-team/hzhou/tt-metal
```

## Troubleshooting

### Test Hangs
```bash
# Kill test and reset device
tt-smi -glx_reset
```

### Import Errors
```bash
# Verify PYTHONPATH
echo $PYTHONPATH
# Should output: /home/models-team/hzhou/tt-metal
```

## Full Documentation

- **Detailed Guide:** `VERIFICATION_GUIDE_ALLGATHER_PREFF1_3.md`
- **Complete Summary:** `ALLGATHER_PREFF1_3_SUMMARY.md`
- **Automated Script:** `verify_fused_all_gather_preff1_3.sh`
