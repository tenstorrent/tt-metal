# Build tt-metal

## Standard build

```bash
./build_metal.sh
```

Builds tt-metal, ttnn, and the Python bindings. After this, `import ttnn` works
from the repo root.

## Build flags

- `--release` / `--debug` — build type (default: release)
- `--build-tests` — also build C++ test binaries
- `--build-ttnn-tests` — only TTNN test binaries
- `--build-metal-tests` — only metal test binaries

## Clean build

```bash
./build_metal.sh --clean
```

## Model-specific dependencies

Some models have extra Python deps:

```bash
pip install -r models/demos/<model>/requirements.txt
```

## CI convention

```bash
mkdir -p generated/test_reports
```

JUnit XML output goes here (configured in pytest.ini).
