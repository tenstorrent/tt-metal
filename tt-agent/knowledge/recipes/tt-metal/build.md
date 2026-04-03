# Build tt-metal

## Prerequisites

```bash
git submodule update --init --recursive
```

## First-time build

```bash
bash build_metal.sh -e          # -e flag sets up build environment
bash create_venv.sh             # creates python_env/ with all Python deps
```

After this, activate the venv and `import ttnn` works.

## Incremental rebuild

```bash
bash build_metal.sh
```

Use after code changes to C++ source. Fast — only recompiles what changed.

## Build flags

- `--release` / `--debug` — build type (default: release)
- `--build-tests` — also build C++ test binaries
- `--build-ttnn-tests` — only TTNN test binaries
- `--build-metal-tests` — only metal test binaries

## Clean build

```bash
bash build_metal.sh --clean
```

## Kernel iteration (no rebuild needed)

Kernels (`tt_metal/kernels/`) are compiled JIT at runtime. If you're only
editing kernel code, skip the rebuild — just re-run the test.

## Pull and sync

After pulling new changes, always sync submodules before rebuilding:

```bash
git pull
git submodule update --init --recursive
bash build_metal.sh
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
