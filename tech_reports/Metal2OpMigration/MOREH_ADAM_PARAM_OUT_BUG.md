# Bug: `moreh_adam` `param_out` returns garbage (pre-existing, cache-independent)

**Severity:** correctness (silent). **Component:** `ttnn/cpp/ttnn/operations/moreh/moreh_adam` (compute kernel). **Found via:** descriptor slow-path-rebuild migration (#46506) — surfaced, not caused, by it.

## Summary
`ttnn.operations.moreh.adam`'s `param_out` output is wrong: a few percent of elements come back as garbage (~`-1e18`), and even the remaining elements do **not** correlate with a `torch.optim.Adam` reference (PCC ≈ 0.04 vs both a same-step and different-step reference). The moment outputs `exp_avg` / `exp_avg_sq` are correct. So the moment computation is fine; the **param update** (`param -= lr * mhat / (sqrt(vhat) + eps)`) write-back is broken.

## It is NOT a descriptor / program-cache / migration issue
- Reproduces with the **program cache disabled**.
- Reproduces with a **constant gradient** (every element identical) — so alternating-position garbage is not numeric.
- The Adam **compute kernels are unchanged** on the migration branch (`git diff origin/main` over the kernels is empty).
- The descriptor guard (`TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT`) never fires for this op.

## Why it wasn't caught before
The nightly `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adam.py` only `assert`s the **last** `passing` variable (`exp_avg_sq`); its `param` check computes `passing=False` (PCC ≈ −0.06) but the result is overwritten before the assert. So the param_out failure is silently masked.

## Repro (single device)
```python
import torch, ttnn
dev = ttnn.open_device(device_id=0)  # cache disabled by default
shape = (1, 1, 32, 64)
param0 = torch.randn(shape, dtype=torch.bfloat16)
grad   = torch.rand(shape, dtype=torch.bfloat16)
mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
p, g = mk(param0), mk(grad)
ea, eas = mk(torch.zeros(shape)), mk(torch.zeros(shape))
po, eao, easo = mk(param0), mk(torch.zeros(shape)), mk(torch.zeros(shape))
out, _, _, _ = ttnn.operations.moreh.adam(
    p, g, ea, eas, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
    step=4, amsgrad=False, param_out=po, exp_avg_out=eao, exp_avg_sq_out=easo)
res = ttnn.to_torch(out).float()
print("garbage frac:", (res.abs() > 1e6).float().mean().item())   # > 0
# even excluding garbage, res is uncorrelated with the torch.optim.Adam param update.
```

## Impact
`moreh_adam` param updates are unreliable. For the #46506 migration this blocks numerically validating the (structurally sound, guard-verified) `get_dynamic_runtime_args` frozen-args fix, so `moreh_adam` was **left on the descriptor slow-path** in that PR until this is fixed. Fixing it requires a compute-kernel change.
