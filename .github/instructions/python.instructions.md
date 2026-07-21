---
description: 'Python coding standards and review rules for all Python files in tt-metal'
applyTo: '**/*.py'
excludeAgent: "cloud-agent"
---

# Python Review Rules

## Review Priorities (Python specific)

### 🔴 CRITICAL — must fix before merge

- **Public API breakage**: changing `ttnn.*` function signatures, removing parameters, or altering return types without deprecation
- **Incorrect tensor shape/dtype assumptions**: operations that will silently produce wrong results on device
- **Missing device resource cleanup**: tensors or meshes allocated but never freed in test teardown
- **Hardcoded paths or environment assumptions**: absolute paths, missing `os.environ` guards for CI-only variables

### 🟡 IMPORTANT — strong preference to fix

- **Missing `@pytest.mark` annotations**: tests requiring specific hardware (`requires_mesh_topology`, `requires_grid_size`) or long runtime (`timeout`) without appropriate marks
- **Bare `assert` without message**: `assert x == y` gives no diagnostic context on failure — prefer `assert x == y, f"expected {y}, got {x}"`
- **Broad exception handling**: `except Exception` or bare `except:` that swallows device errors
- **Torch-to-ttnn conversion without validation**: missing shape/dtype checks before `ttnn.from_torch()`
- **Test parametrize explosion**: `@pytest.mark.parametrize` with >100 combinations without `@pytest.mark.nightly` gating

### 🟢 SUGGESTION — nice to have

- Type hints on public function signatures
- Docstrings on non-trivial helper functions
- Using `pathlib.Path` over `os.path` string manipulation
- Extracting repeated test setup into fixtures

## Formatting & Style

- Line length: **120** (configured via `black` and `ruff` in `pyproject.toml`)
- Formatter: **black**; import sorting: **isort** (profile: black)
- Do not flag style issues that `black`/`isort`/`ruff` would auto-fix

## Import Hygiene

- `import ttnn` is the public entry point — test and model code must not import from `ttnn._ttnn` or other internal modules
- Avoid `from ttnn import *`; use explicit imports so readers can trace symbol origins
- Group imports: stdlib → third-party (`torch`, `numpy`, `pytest`) → first-party (`ttnn`, `models.*`, `tests.*`)

## Pytest Conventions

- Tests requiring Tenstorrent hardware must use the appropriate fixture (`device`, `mesh_device`) or mark (`@pytest.mark.requires_mesh_topology`)
- Use `@pytest.fixture` with appropriate scope (`function` for stateful, `session` for expensive one-time setup)
- Prefer `@pytest.mark.parametrize` over manual for-loops inside tests
- Nightly-only or slow tests must carry `@pytest.mark.nightly` or `@pytest.mark.timeout(N)`
- Tests must not depend on execution order — each test should be independently runnable

## Tensor & Device Patterns

- Always validate tensor shape and dtype before sending to device: `ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)`
- Specify `device` and `memory_config` explicitly rather than relying on defaults that may change
- Close or deallocate device tensors in test cleanup to avoid resource leaks on CI runners
- Use `ttnn.to_torch()` for comparisons — never index device tensors directly from Python

## Review Checklist

- [ ] No internal (`_ttnn`) imports in test/model code
- [ ] Hardware-dependent tests are properly marked
- [ ] Parametrized tests have reasonable cardinality or nightly gating
- [ ] Tensor conversions specify dtype and layout explicitly
- [ ] No bare `except:` or `except Exception:` without re-raise
- [ ] Test does not assume a specific working directory or absolute path
