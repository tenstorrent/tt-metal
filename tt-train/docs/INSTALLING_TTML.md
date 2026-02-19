# Installing TT-Train (ttml)

TT-Train provides the ttml module (`ttml`), written using nanobind, which provides python bindings to tt-train C++ operations.

## Prerequisites

- **ttnn** must be installed first. See [INSTALLING.md](../INSTALLING.md) for TT-Metal installation instructions.

## Installation

tt-train must be built as part of tt-metal using `build_metal.sh --build-tt-train`.

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

## Verification

```python
import ttml
print("ttml installed successfully")
```
