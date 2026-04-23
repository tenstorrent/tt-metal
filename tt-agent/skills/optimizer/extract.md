# Extract Subagent — Model → Unit Test

Run this subagent once per optimization session, only when no suitable unit
test exists for the target op. The common case is "a unit test already
exists" — skip this phase entirely in that case.

## Purpose

Isolate a bottleneck op from a model into a standalone pytest that:

1. Loads the exact input tensors the op receives in the model's forward pass
   (not random data — value distribution matters).
2. Reproduces the op's device time within ±10% of the model's measurement
   for that op.
3. Runs in seconds, not minutes, so the optimizer can iterate.

## Inputs from caller

- Model test path + optional `-k` filter (the full-model entry point).
- Target op identifier — op code (e.g., `Matmul`), or an approximate line
  range in the model code if the op code alone is ambiguous.
- Baseline profile note path — so we know which specific op instance to
  target (there may be many `Matmul` calls in a forward pass).

## Procedure

### 1. Locate the call site

Load the baseline profile note. Identify the op instance by its rank +
shape metadata. Cross-reference with the model code to find the exact
Python call site. If multiple instances match, ask the developer to
disambiguate — do not guess.

### 2. Instrument the model

Add a one-time hook at the target call site that:

- Captures the op's input tensors *just before* the call on the model's
  first real iteration (not the cache-warmup iteration).
- Captures the op's configuration (shapes, dtypes, memory configs, program
  configs if applicable).
- Writes to `<workspace>/.tt-agent/tensors/<scope>-<YYYY-MM-DD-HHMMSS>.pt`
  using `torch.save({"inputs": (...), "config": {...}, "fingerprint": {...}})`.
- Exits the model after capture (no need to run the rest of the forward
  pass).

The hook is temporary — remove it after capture. Do not commit the hook.

### 3. Fingerprint the capture

Before writing the .pt file, compute a fingerprint of each input tensor:

```python
{"shape": tuple(t.shape), "dtype": str(t.dtype), "mean": float(t.mean()),
 "std": float(t.std()), "min": float(t.min()), "max": float(t.max())}
```

Store the fingerprint alongside the tensors in the .pt file. The unit test
will verify it on load — a corrupted or swapped tensor file must fail loud.

### 4. Write the unit test

Create `tests/optimizer/<scope>_unit_test.py` (or a workspace-local path —
do not pollute the main test tree). Template:

```python
import torch, pytest, ttnn

TENSOR_PATH = "<absolute path to the .pt>"

@pytest.fixture(scope="module")
def captured():
    blob = torch.load(TENSOR_PATH)
    inputs = blob["inputs"]
    for t, fp in zip(inputs, blob["fingerprint"]):
        assert tuple(t.shape) == tuple(fp["shape"]), "tensor shape drift"
        assert str(t.dtype) == fp["dtype"], "tensor dtype drift"
        # fingerprint stats check — tolerant, just to catch swap/corruption
        assert abs(float(t.mean()) - fp["mean"]) < 1e-3, "tensor value drift"
    return blob

def test_target_op(device, captured):
    inputs = captured["inputs"]
    config = captured["config"]
    # Run the op twice — first run populates program cache
    out1 = <op_call>(*inputs, **config)
    out2 = <op_call>(*inputs, **config)
    # Correctness check against captured reference output if available,
    # otherwise skip — the optimizer's PCC check handles this.
```

Adapt the `<op_call>` to the actual op API — read the model source for the
exact call pattern and replicate it verbatim. Do not paraphrase.

### 5. Verify — hard gate

Profile the newly-written unit test with `tt:profiler`. Compare its
`DEVICE FW DURATION` for the target op against the baseline profile's
duration for the same op.

- **Within ±10% → pass.** Extraction done. Return the test path and tensor
  path to the main optimizer.
- **Outside ±10% → fail.** The extracted test is not representative. Do
  NOT proceed to iteration. Write a failure note with both measurements
  and hand back to the main optimizer. Common causes:
  - Captured tensors came from a different iteration than intended.
  - Op config (memory layout, program config) was not captured fully.
  - The op in the model ran under a surrounding context (trace capture,
    fabric config, persistent kernel state) that the standalone test
    does not reproduce.

Surface the mismatch to the developer; let them decide whether to refine
the extraction or abandon and optimize against the model itself.

## Outputs

On success:
- Unit test file path.
- Tensor file path.
- Verification measurement showing ±10% agreement.

On failure:
- Mismatch report with both durations.
- Unit test and tensor files left in place for the developer to inspect.

## Non-goals

- Do not modify the model. The hook is added for one capture run and
  removed.
- Do not re-run the full model after capture. One capture run is the
  contract.
- Do not commit anything during extraction. Commits begin at the Iterate
  phase.
