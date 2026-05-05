# ACE-Step v1.5 (Torch ref + TTNN)

This folder provides:

- `torch_ref/`: PyTorch reference implementation
- `ttnn_impl/`: TTNN implementation with one-to-one module mapping
- `tests/`: per-module PCC validation (Torch vs TTNN)

## Mandatory constraints (enforced by design)

- **TTNN device purity**: TTNN modules must not call PyTorch ops inside their `forward()`; the only allowed transfers are:
  - Host → device at the start of the run (inputs + weights)
  - Device → host at the end (final outputs for PCC comparison)
- **One-to-one mapping**: every Torch module has a TTNN equivalent.

## Layout

```
ace_step_v1_5/
  torch_ref/
  ttnn_impl/
  tests/
```

## Running tests

From repo/workspace root:

```bash
python -m pytest ace_step_v1_5/tests -q
```

If you have TT hardware/runtime, set:

```bash
export MESH_DEVICE=N150   # or N300 / T3K
```

## Notes

- PCC threshold in tests is set to `>= -0.9` per request (very lenient). You can tighten it later.
