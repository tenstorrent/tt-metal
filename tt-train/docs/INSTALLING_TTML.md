# Installing TT-Train (ttml)

TT-Train provides the ttml module (`ttml`), written using nanobind, which provides python bindings to tt-train C++ operations.

## Prerequisites

- **ttnn** must be installed first. See [INSTALLING.md](../INSTALLING.md) for TT-Metal installation instructions.

### Option A: Using pre-built ttml (recommended for development)

If you built tt-metal with `build_metal.sh --build-tt-train` or `--build-all`, ttml is already compiled.

**Automatic setup:** The `create_venv.sh` script automatically creates the necessary `.pth` files. Just run:

```bash
cd /path/to/tt-metal
./create_venv.sh
./build_metal.sh --build-tt-train
```

**Manual setup:** If needed, create the `.pth` files manually:

```bash
cd /path/to/tt-metal
echo "/path/to/tt-metal/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/ttml.pth
echo "/path/to/tt-metal/build/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/_ttml.pth
```

This avoids rebuilding and reflects Python source changes immediately.

### Option B: uv pip install (standalone build)

```bash
cd /path/to/tt-train
uv pip install .
```

Or for editable/development install:

> **Note:** `--no-build-isolation` Disables isolation when building a modern source distribution, saves time when re-building often

```bash
cd /path/to/tt-train
uv pip install --no-build-isolation -e .
```

## Verification

```python
import ttml
print("ttml installed successfully")
```
