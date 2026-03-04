# Claude Development Environment Setup

This document contains the necessary environment setup for working with TT-Metal and TT-MoE projects.

## Environment Variables

Before running any tests or code, set up the following environment:

```bash
# Navigate to tt-metal root
cd /home/ntarafdar/tt-moe/tt-metal

# Activate Python virtual environment
source python_env/bin/activate

# Set required environment variables
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export MESH_DEVICE=TG

# DeepSeek model paths (if working with DeepSeek models)
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528

# Cache directory should be test-specific to avoid conflicts
# Create a unique cache directory for each test run:
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p $DEEPSEEK_V3_CACHE

# Or for a specific test:
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_moe_block_test
mkdir -p $DEEPSEEK_V3_CACHE
```

## Quick Setup Script

You can source this as a one-liner (with test-specific cache):
```bash
cd /home/ntarafdar/tt-moe/tt-metal && source python_env/bin/activate && export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$ && mkdir -p $DEEPSEEK_V3_CACHE
```

## Running Tests

After setting up the environment:
```bash
# Create test-specific cache directory
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p $DEEPSEEK_V3_CACHE

# Run TT-MoE tests
pytest models/tt_moe/tests/test_moe_block.py -xvs

# Run specific test
pytest models/tt_moe/tests/test_moe_block.py::test_moe_block_forward -xvs

# Clean up cache after test (optional)
rm -rf $DEEPSEEK_V3_CACHE
```

## Important Notes

- Always activate the Python environment before running any code
- The MESH_DEVICE=TG is required for multi-device tests
- The DeepSeek model paths are required when running tests that load model weights
- Use test-specific cache directories to prevent cache conflicts between tests
- Always create the cache directory with `mkdir -p` before running tests
- Consider cleaning up cache directories after tests complete to save disk space
