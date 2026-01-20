# Installing TT-Train (ttml)

TT-Train provides the ttml module (`ttml`), written using nanobind, which provides python bindings to tt-train C++ operations.

## Prerequisites

- **ttnn** must be installed first. See [INSTALLING.md](../INSTALLING.md) for TT-Metal installation instructions.

### 1. Install Python Package

```bash
pip install .
```

Or for editable/development install:

```bash
pip install --no-build-isolation -e .
```

## Verification

```python
import ttml
print("ttml installed successfully")
```
