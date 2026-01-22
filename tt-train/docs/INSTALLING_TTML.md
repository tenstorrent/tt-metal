# Installing TT-Train (ttml)

TT-Train provides the ttml module (`ttml`), written using nanobind, which provides python bindings to tt-train C++ operations.

## Prerequisites

- **ttnn** must be installed first. See [INSTALLING.md](../INSTALLING.md) for TT-Metal installation instructions.

### 1. Install Python Package

> **Important:** tt-metal uses `uv` for virtual environment management. You must use `uv pip` instead of `pip`. See [INSTALLING.md](../INSTALLING.md) for details.

```bash
uv pip install .
```

Or for editable/development install:

```bash
uv pip install --no-build-isolation -e .
```

## Verification

```python
import ttml
print("ttml installed successfully")
```
