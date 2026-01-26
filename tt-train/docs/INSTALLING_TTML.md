# Installing TT-Train (ttml)

TT-Train provides the ttml module (`ttml`), written using nanobind, which provides python bindings to tt-train C++ operations.

## Prerequisites

- **ttnn** must be installed first. See [INSTALLING.md](../INSTALLING.md) for TT-Metal installation instructions.

### Option A: pip install (standalone build)

```bash
pip install .
```

Or for editable/development install:

```bash
pip install --no-build-isolation -e .
```

### Option B: Using pre-built ttml (recommended for development)

If you built tt-metal with `build_metal.sh --build-tt-train` or `--build-all`, ttml is already compiled. Create two `.pth` files to make Python find it:

```bash
# One-time setup - adds paths to your virtualenv's Python path

# 1. Add ttml Python source code
echo "/path/to/tt-metal/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/ttml.pth

# 2. Add the built _ttml C++ extension (.so file)
echo "/path/to/tt-metal/build/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/_ttml.pth
```

This avoids rebuilding and reflects Python source changes immediately.

## Verification

```python
import ttml
print("ttml installed successfully")
```
