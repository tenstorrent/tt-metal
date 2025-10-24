# Building TT-Metal

Simple steps to build tt-metal:

## Steps

```bash
cd tt-metal

./install_dependencies.sh

./create_venv.sh

source ./env_vars_setup.sh

./build_metal_with_flags.sh

pip install -e .

python -c "import ttnn"
```

## What Each Step Does

1. **`cd tt-metal`** - Navigate to the tt-metal directory
2. **`./install_dependencies.sh`** - Install system dependencies
3. **`./create_venv.sh`** - Create Python virtual environment
4. **`source ./env_vars_setup.sh`** - Set up environment variables
5. **`./build_metal_with_flags.sh`** - Build metal with custom flags
6. **`pip install -e .`** - Install tt-metal in editable mode
7. **`python -c "import ttnn"`** - Verify the build by importing ttnn

## Important Note

**Every time C++ source code changes, you need to rebuild and reinstall:**

```bash
./build_metal_with_flags.sh
pip install -e .
```

## Running Qwen3-32B Demo

To run Qwen3-32B with batch-1:

```bash
export HF_MODEL=Qwen/Qwen3-32B
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

**For more detailed instructions and additional configurations, see:**
`models/tt_transformers/README.md`
