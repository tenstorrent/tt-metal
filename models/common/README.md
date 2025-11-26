# TTNN Validation & Testing Utilities

Helpers for validating TTNN computations against reference implementations and
for moving tensors between TTNN and PyTorch. The public API is implemented
across `models.common.validation_tools`, `models.common.metrics`,
`models.common.auto_compose`, and `models.common.distribute_as`, and is
exercised in:

- `models/common/tests/test_validation_tools.py`
- `models/common/tests/test_metrics.py`
- `models/common/tests/test_auto_compose.py`
- `models/common/tests/test_distribute_as.py`
- `models/common/tests/host/test_metrics_pytorch_only.py`

The examples in these tests are the most up‑to‑date reference for usage.

## Quick Start – host reference (`compare_to_torch`)

Use `compare_to_torch` when your reference implementation is a PyTorch function.
Inputs and outputs are automatically converted between TTNN and PyTorch.

```python
import torch
import ttnn
from models.common.validation_tools import compare_to_torch, Metric, get_validation_registry


@compare_to_torch(
    reference_fn=torch.matmul,
    metric_tolerances={
        Metric.MAX_ABS_ERROR: 1e-1,
        Metric.PCC: 0.99,
    },
)
def ttnn_matmul(a, b):
    # a, b are TTNN tensors (possibly sharded)
    return ttnn.matmul(a, b)


def run_example(device: ttnn.MeshDevice):
    m, n, k = 16, 24, 12
    a = torch.randn(1, m, k, dtype=torch.bfloat16)
    b = torch.randn(1, k, n, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    _ = ttnn_matmul(a_tt, b_tt)

    registry = get_validation_registry()
    registry.print_report()
```

Note:
- When the signature of the decorated function is different from the reference function, `input_to_torch` and `output_to_torch` can be used to map the inputs and outputs between the decorated function and the reference function.
- See `models/common/tests/test_validation_tools.py::test_validation_matmul` for a real test using
this pattern.

## Quick Start – TTNN reference (`compare_to_ttnn`)

Use `compare_to_ttnn` when both your implementation and reference are TTNN‑based
and you want metrics computed directly on device.

```python
import torch
import ttnn
from models.common.validation_tools import compare_to_ttnn


def torch_rms_norm(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    return weight * x * torch.rsqrt(var + eps)


class DeviceValidatedRMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, device: ttnn.MeshDevice):
        self.eps = eps
        self.device = device
        self.weight_torch = weight
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _reference_impl(self, x):
        x_torch = ttnn.to_torch(x).squeeze(0)
        y_torch = torch_rms_norm(x_torch, self.weight_torch, self.eps)
        return ttnn.from_torch(
            y_torch.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    @compare_to_ttnn(reference_fn=lambda self, x: self._reference_impl(x))
    def __call__(self, x):
        x_sq = ttnn.mul(x, x)
        mean_x_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_sq, self.eps))
        x_norm = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_norm, self.weight)
```
Note:
- When the signature of the decorated function is different from the reference function, `input_to_ttnn` and `output_to_ttnn` can be used to map the inputs and outputs between the decorated function and the reference function.
- This mirrors the pattern used by `DeviceValidatedRMSNorm` in
`models/common/tests/test_validation_tools.py`.

## Features

- **Decorator‑based validation** – Wrap TTNN functions or methods and compare
  them against PyTorch or TTNN references.
- **Host and device modes** – `compare_to_torch` (PyTorch reference) and
  `compare_to_ttnn` (TTNN reference).
- **TTNN‑native metrics** – When both outputs are TTNN tensors, metrics are
  computed on device with minimal host transfer.
- **Configurable tolerances** – Per‑metric tolerances via the `Metric` enum,
  string keys, or `MetricSpec`.
- **Custom metrics** – Inject your own metric functions.
- **Global registry** – Collects all validation runs for reporting.
- **Easy disabling** – Turn validation on/off globally without changing call
  sites.

## Core Components

### Validation decorators

All decorators live in `models.common.validation_tools`:

- `compare_to_torch(reference_fn, *, input_to_torch=None, output_to_torch=None, metric_tolerances=None, enabled=True, raise_exceptions=False, return_reference_output=False)`
  - Use when `reference_fn` is a PyTorch implementation.
  - By default, all TTNN tensors in the arguments/outputs are converted to
    PyTorch via `to_torch_auto_compose`.
  - Optional `input_to_torch(*args, **kwargs)` lets you override how inputs
    are mapped to the reference.
  - Optional `output_to_torch(output)` converts the implementation output
    before metrics are computed.

- `compare_to_ttnn(reference_fn, *, input_to_ttnn=None, output_to_ttnn=None, metric_tolerances=None, enabled=True, raise_exceptions=False, return_reference_output=False)`
  - Use when `reference_fn` consumes and returns TTNN tensors.
  - Optional `input_to_ttnn(*args, **kwargs)` lets you override how inputs
    are mapped to the reference.
  - Optional `output_to_ttnn(output)` converts the implementation output
    before metrics are computed.
  - If both implementation and reference return TTNN tensors, metrics run
    entirely on device.

In both cases, decorating a function records a `ValidationResult` in the global
`ValidationRegistry` every time the function is called (unless disabled).

### Metrics

Metric utilities are implemented in `models.common.metrics`:

- `compute_max_abs_error(impl, ref)` – max absolute error.
- `compute_mean_abs_error(impl, ref)` – mean absolute error.
- `compute_pcc(impl, ref)` – Pearson correlation coefficient; uses TTNN
  operations when possible and falls back to host.
- `comp_allclose(impl, ref, rtol=..., atol=...)` – allclose check plus a
  detailed delta string.
- `DEFAULT_METRICS` – dict with built‑in metrics (`"max_abs_error"`,
  `"mean_abs_error"`, `"pcc"`).

Metrics support both TTNN and PyTorch tensors.

### Registry and control functions

From `models.common.validation_tools`:

- `get_validation_registry() -> ValidationRegistry`
  - Holds all `ValidationResult` objects.
  - Provides `get_summary()` and `print_report(verbose: bool = False)`.

- `enable_validation(enabled: bool = True)`
  - Globally enable/disable validation; when disabled, decorators become
    transparent wrappers.

- `clear_validation_results()`
  - Clear all accumulated validation results.

`ValidationResult` includes:

- `function_name`
- `passed` (bool)
- `metrics` – map of metric name → per‑metric result (value, passed, error)
- `execution_time_impl`, `execution_time_ref`
- `timestamp`
- `logs` – optional debug strings

### Auto‑compose helper

`to_torch_auto_compose` lives in `models.common.auto_compose`.

It converts an arbitrary TTNN tensor (including sharded/replicated multi‑device
tensors) to a single PyTorch tensor by automatically choosing the appropriate
mesh composer.

It is heavily used in:

- `test_auto_compose.py`
- `test_distribute_as.py`
- all `compare_to_torch`‑based examples.

## Usage Patterns

High‑level patterns illustrated in the tests:

1. **Host reference with explicit input mapping**
   - See `HostValidatedRMSNorm` in `models/common/tests/test_validation_tools.py`.
   - Uses `compare_to_torch` with `input_to_torch` to map TTNN inputs and
     TTNN weights to a pure‑PyTorch reference function.

2. **TTNN reference (on‑device metrics)**
   - See `DeviceValidatedRMSNorm` in `models/common/tests/test_validation_tools.py`.
   - Uses `compare_to_ttnn` where both implementation and reference return
     TTNN tensors; metrics run on device.

3. **Simple library calls**
   - See `ttnn_matmul` and `ttnn_matmul_reverse` in `models/common/tests/test_validation_tools.py`.
   - `compare_to_torch(reference_fn=torch.matmul, ...)` with optional
     `input_to_torch` remapping.

4. **Checkpoint / `from_torch` validation**
   - See `from_torch_checkpoint` in `models/common/tests/test_validation_tools.py`.
   - Validates a direct `ttnn.from_torch(...)` call using `compare_to_torch`
     and `output_to_torch`.

5. **Custom metric via `MetricSpec`**
   - See `ttnn_matmul_metric_spec` in `models/common/tests/test_validation_tools.py`
     and `MetricSpec` usage in `models/common/tests/host/test_metrics_pytorch_only.py`.
   - Use `MetricSpec(tolerance=..., higher_is_better=..., compute_fn=...)`
     in `metric_tolerances`.

6. **Non‑decorator usage**
   - `test_validation_non_decorator_class_vs_class_torch` demonstrates calling
     `compare_to_torch` in a more manual, non‑decorator style between two
     callable classes.

## Default Metrics and Tolerances

When `metric_tolerances` is omitted, the framework uses sensible defaults:

- `Metric.MAX_ABS_ERROR` with tolerance `1e-2`
- `Metric.PCC` with tolerance `0.99`

If you pass a `metric_tolerances` dict, keys can be:

- `Metric` enum members (recommended), e.g. `Metric.MAX_ABS_ERROR`
- strings (`"max_abs_error"`, `"mean_abs_error"`, `"pcc"`)
- arbitrary names when used with `MetricSpec`

Values can be:

- a float tolerance (uses the built‑in metric)
- a `MetricSpec` instance to define a custom metric and tolerance

Example:

```python
from models.common.validation_tools import Metric, MetricSpec
from models.common.metrics import compute_pcc


@compare_to_torch(
    reference_fn=torch.matmul,
    metric_tolerances={
        Metric.PCC: MetricSpec(tolerance=0.99, higher_is_better=True, compute_fn=compute_pcc),
        Metric.MAX_ABS_ERROR: 1.5e-1,
    },
)
def ttnn_matmul_metric_spec(a, b):
    return ttnn.matmul(a, b)
```

## Testing

The local test suite in `models/common/tests` shows end‑to‑end usage:

- `test_validation_tools.py`
  - Core decorator usage, registry behaviour, error handling, custom metrics.
- `test_metrics.py`
  - Numerical correctness of device and host metric functions.
- `host/test_metrics_pytorch_only.py`
  - Pure‑PyTorch metric tests.
- `test_auto_compose.py`
  - Auto‑composition of sharded/replicated TTNN tensors into PyTorch.
- `test_distribute_as.py`
  - Distribution helpers (`from_torch_dist_as`) that mirror an existing TTNN
    tensor’s topology.

Example commands (run from the repo root, with TTNN available):

```bash
python -m pytest models/common/tests/test_validation_tools.py -v
python -m pytest models/common/tests/test_metrics.py -v
python -m pytest models/common/tests/host/test_metrics_pytorch_only.py -v
```

## API Reference (public surface)

All symbols below are imported from `models.common.validation_tools` and `models.common.metrics`:

- Decorators:
  - `compare_to_torch`
  - `compare_to_ttnn`
- Registry and control:
  - `ValidationResult`
  - `ValidationRegistry`
  - `get_validation_registry`
  - `enable_validation`
  - `clear_validation_results`
- Metrics:
  - `Metric` (enum: `MAX_ABS_ERROR`, `MEAN_ABS_ERROR`, `PCC`)
  - `MetricSpec`
  - `compute_max_abs_error`
  - `compute_mean_abs_error`
  - `compute_pcc`
  - `comp_allclose`
  - `DEFAULT_METRICS`
- Auto‑compose:
  - `to_torch_auto_compose`

For concrete, runnable examples of each API, see the tests listed at the top
of this document.
