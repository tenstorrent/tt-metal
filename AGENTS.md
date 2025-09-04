# Repository Guidelines

## Project Structure & Module Organization
- `tt_metal/`: C++ Metalium core, device/runtime APIs and kernels.
- `ttnn/`: Python frontend and bindings exposed as the `ttnn` package.
- `models/`: Demo models and example pipelines.
- `tests/`: C++ gtests and Python pytest suites (`tests/tt_metal`, `tests/ttnn`, etc.).
- `docs/`: Sphinx documentation; see `docs/Makefile`.
- Tooling: `cmake/`, `scripts/`, `infra/`, `third_party/` and top‑level `CMakeLists.txt`.

## Build, Test, and Development Commands
- Install deps: `sudo ./install_dependencies.sh` (Linux) and hardware drivers per `INSTALLING.md`.
- Build (recommended): `./build_metal.sh --build-tests --development`
  - Manual: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo && ninja -C build && ninja -C build install`
- Python env: `./create_venv.sh && source python_env/bin/activate && pip install -e .`
- Run tests:
  - Pytest: `pytest -q` or targeted file, e.g. `pytest tests/ttnn/test_*.py`.
  - Post‑commit suite: `./tests/scripts/run_tests.sh --tt-arch <ARCH> --pipeline-type post_commit`
  - C++ gtests: `./build/test/tt_metal/unit_tests_api --gtest_filter=Suite.Test`

## Coding Style & Naming Conventions
- C++: `.clang-format` and `.clang-tidy` enforced; run via pre‑commit.
- Python: `black` + `isort` (120 cols) per `pyproject.toml`.
- Naming: Python `snake_case` for funcs/vars; C++ types `PascalCase`, methods `camelCase` as in existing code.
- Headers: new files include SPDX license headers (see CONTRIBUTING.md).
- Pre‑commit: `pre-commit run --all-files` before pushing.

## Testing Guidelines
- Frameworks: `pytest` (Python) and gtest (C++).
- Python tests live under `tests/**/test_*.py`; use markers from `pytest.ini` (e.g., `@pytest.mark.post_commit`, `@pytest.mark.slow`).
- Build tests with `./build_metal.sh --build-tests`. Some tests require Tenstorrent hardware; validate with `tt-smi`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, reference issues/PRs (e.g., `#12345`, `(#25230)`); use `[skip ci]` for docs‑only changes.
- PRs: link the issue, describe changes and impact, include test commands/results and any env flags; keep diffs focused and update docs as needed.
- CI must be green and code owners must approve before merge.

## Environment Tips
- Set `TT_METAL_HOME=$(pwd)` and `PYTHONPATH=$TT_METAL_HOME` for local runs.
- Verbose logs: `TT_LOGGER_LEVEL=Debug`.
