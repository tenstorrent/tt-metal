# Extract Subagent — Model → Unit Test

Runs once per session, only when no unit test exists for the target op.
Common case is "unit test already exists" — skip this phase.

## Purpose

Isolate a bottleneck op from a model into a standalone pytest that:

1. Loads the exact input tensors the op receives in the model's forward
   pass (not random data — value distribution matters).
2. Reproduces the op's device time within ±10% of the model's measurement.
3. Runs in seconds, so the optimizer can iterate.

## Inputs

- Model test path + optional `-k` filter.
- Target op identifier (op code or approximate line range in model code).
- Baseline profile note path — identifies which specific op instance.

## Procedure

### 1. Locate the call site

From the baseline profile note, identify the op instance by rank + shape.
Cross-reference with the model code for the exact call site. If multiple
match, ask the developer — do not guess.

### 2. Instrument the model

Add a temporary hook at the target call site that:

- Captures input tensors *just before* the call on the first real
  iteration (not cache-warmup).
- Captures op config (shapes, dtypes, memory configs, program configs).
- Writes to `<workspace>/.tt-agent/tensors/<scope>-<YYYY-MM-DD-HHMMSS>.pt`
  via `torch.save({"inputs": (...), "config": {...}, "fingerprint": {...}})`.
- Exits after capture (skip the rest of the forward pass).

Remove the hook after capture. Do not commit it.

### 3. Fingerprint the capture

Before writing, fingerprint each input:

```python
{"shape": tuple(t.shape), "dtype": str(t.dtype), "mean": float(t.mean()),
 "std": float(t.std()), "min": float(t.min()), "max": float(t.max())}
```

The unit test verifies it on load — swap or corruption must fail loud.

### 4. Write the unit test

Create `tests/optimizer/<scope>_unit_test.py` (or a workspace-local path).
Template:

```python
import torch, pytest, ttnn

TENSOR_PATH = "<absolute path>"

@pytest.fixture(scope="module")
def captured():
    blob = torch.load(TENSOR_PATH)
    for t, fp in zip(blob["inputs"], blob["fingerprint"]):
        assert tuple(t.shape) == tuple(fp["shape"])
        assert str(t.dtype) == fp["dtype"]
        assert abs(float(t.mean()) - fp["mean"]) < 1e-3
    return blob

def test_target_op(device, captured):
    inputs = captured["inputs"]
    config = captured["config"]
    # Run twice — first run populates program cache
    out1 = <op_call>(*inputs, **config)
    out2 = <op_call>(*inputs, **config)
```

Replicate the model's `<op_call>` verbatim — do not paraphrase.

### 5. Verify — hard gate

Profile the new unit test via `tt:profiler`. Compare its `DEVICE FW
DURATION` for the target op against the baseline profile.

- **Within ±10% → pass.** Return test path + tensor path to main.
- **Outside ±10% → fail.** Do NOT iterate. Write a failure note with
  both measurements. Common causes: captured tensors from wrong
  iteration; op config not captured fully; surrounding context (trace
  capture, fabric config, persistent kernel state) not reproduced.

Surface to the developer; they decide whether to refine or abandon.

## Outputs

On success: unit test path, tensor path, ±10% verification measurement.

On failure: mismatch report with both durations; files left in place.

## Scope

- Hook is one-shot — revert after capture.
- One capture run per session; no re-runs of the full model.
- Commits begin at the Iterate phase, not during extraction.
