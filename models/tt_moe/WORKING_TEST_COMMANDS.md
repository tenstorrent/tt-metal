# Working Test Commands for MoE Block

## Environment Setup (Required for All Tests)

### Full Environment Setup (One-liner)
```bash
source python_env/bin/activate && \
export PYTHONPATH=/home/ntarafdar/tt-moe/tt-metal && \
export TT_METAL_HOME=/home/ntarafdar/tt-moe/tt-metal && \
export MESH_DEVICE=TG && \
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 && \
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%s) && \
mkdir -p $DEEPSEEK_V3_CACHE
```

### Step-by-Step Environment Setup
```bash
# 1. Navigate to project root
cd /home/ntarafdar/tt-moe/tt-metal

# 2. Activate Python virtual environment
source python_env/bin/activate

# 3. Set Python path
export PYTHONPATH=/home/ntarafdar/tt-moe/tt-metal

# 4. Set TT Metal home
export TT_METAL_HOME=/home/ntarafdar/tt-moe/tt-metal

# 5. Set mesh device for Galaxy
export MESH_DEVICE=TG

# 6. Set DeepSeek model path (only needed for DeepSeek tests)
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528

# 7. Create unique cache directory (important to avoid conflicts)
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%s)
mkdir -p $DEEPSEEK_V3_CACHE
```

## Verified Working Test Commands

### 1. DeepSeek Decode Test
```bash
# After environment setup
python -m pytest models/tt_moe/tests/test_moe_block.py::test_forward_pass[True-decode-128-deepseek-device_params0] -xvs

# With PCC output filter
python -m pytest models/tt_moe/tests/test_moe_block.py::test_forward_pass[True-decode-128-deepseek-device_params0] -xvs 2>&1 | grep -E "(PCC:|PASSED|FAILED)"
```
**Expected PCC:** ~0.9909

### 2. DeepSeek Prefill Test
```bash
python -m pytest models/tt_moe/tests/test_moe_block.py::test_forward_pass[True-prefill-128-deepseek-device_params0] -xvs
```
**Expected PCC:** ~0.9913

### 3. GPT-OSS Decode Test
```bash
# Note: GPT-OSS doesn't need DEEPSEEK env vars
python -m pytest models/tt_moe/tests/test_moe_block.py::test_forward_pass[True-decode-128-gptoss-device_params1] -xvs
```
**Expected PCC:** ~0.9259

### 4. GPT-OSS Prefill Test
```bash
python -m pytest models/tt_moe/tests/test_moe_block.py::test_forward_pass[True-prefill-128-gptoss-device_params1] -xvs
```
**Expected PCC:** ~0.9178

## Running All Tests

### All MoE Block Tests
```bash
python -m pytest models/tt_moe/tests/test_moe_block.py -xvs
```

### Specific Backend Tests
```bash
# DeepSeek only
python -m pytest models/tt_moe/tests/test_moe_block.py -xvs -k "deepseek"

# GPT-OSS only
python -m pytest models/tt_moe/tests/test_moe_block.py -xvs -k "gptoss"
```

### Specific Mode Tests
```bash
# Decode mode only
python -m pytest models/tt_moe/tests/test_moe_block.py -xvs -k "decode"

# Prefill mode only
python -m pytest models/tt_moe/tests/test_moe_block.py -xvs -k "prefill"
```

## Quick Validation Commands

### Check JSON Config Loading
```bash
source python_env/bin/activate && export PYTHONPATH=/home/ntarafdar/tt-moe/tt-metal && python -c "
from models.tt_moe.moe_block import MoEBlock
config = MoEBlock._get_base_config('deepseek', 'decode')
print('DeepSeek config:', config.get('architecture', {}).get('num_experts'))
config = MoEBlock._get_base_config('gptoss', 'decode')
print('GPT-OSS config:', config.get('architecture', {}).get('num_local_experts'))
"
```

### Check Memory Config Conversion
```bash
source python_env/bin/activate && export PYTHONPATH=/home/ntarafdar/tt-moe/tt-metal && python -c "
from models.tt_moe.moe_block import MoEBlock
import ttnn
config = MoEBlock._get_base_config('deepseek', 'decode')
print('L1 Memory loaded:', config['decode'].get('memory_config') == ttnn.L1_MEMORY_CONFIG)
config = MoEBlock._get_base_config('gptoss', 'prefill')
print('DRAM Memory loaded:', config['prefill'].get('memory_config') == ttnn.DRAM_MEMORY_CONFIG)
"
```

## Common Issues and Solutions

### Issue: "python: command not found"
**Solution:** Always activate the virtual environment first:
```bash
source python_env/bin/activate
```

### Issue: "mkdir: cannot create directory '': No such file or directory"
**Solution:** Use explicit paths and separate export statements:
```bash
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%s)
mkdir -p $DEEPSEEK_V3_CACHE
```

### Issue: Tests hang with dispatch core warnings
**Solution:** Reset the Galaxy devices:
```bash
pkill -9 pytest  # Kill hanging processes
source python_env/bin/activate
tt-smi -glx_reset  # Reset devices
```

### Issue: ImportError or module not found
**Solution:** Ensure PYTHONPATH is set correctly:
```bash
export PYTHONPATH=/home/ntarafdar/tt-moe/tt-metal
```

## Test Output Filtering

### Get just PCC values
```bash
pytest <test_path> -xvs 2>&1 | grep "PCC:"
```

### Get summary only
```bash
pytest <test_path> -xvs 2>&1 | tail -20
```

### Get pass/fail status
```bash
pytest <test_path> -xvs 2>&1 | grep -E "(PASSED|FAILED|ERROR)"
```

## Notes

1. **Always use unique cache directories** to avoid conflicts between test runs
2. **python -m pytest** is more reliable than just `pytest`
3. **Use explicit absolute paths** in exports to avoid issues
4. **The -xvs flags** mean: -x (stop on first failure), -v (verbose), -s (show print statements)
5. **Test times:** Each test takes approximately 2.5-3 minutes to complete
