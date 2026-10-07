# Mixed Precision in TTML

This page describes how TTML keeps a tensor in one precision and computes in another. It is tracked in
[#56513](https://github.com/tenstorrent/tt-metal/issues/56513).

## Two views of one tensor

Every `autograd::Tensor` stores its value in an `autograd::AutocastTensor`
(`sources/ttml/autograd/autocast_tensor.{hpp,cpp}`). It holds the tensor in its native precision, bf16 or fp32, and
creates a copy in the other float precision the first time someone asks for it:

| Call | Returns |
|---|---|
| `get_value(HALF)`, the default | bf16: the native tensor, or the derived copy of an fp32 tensor |
| `get_value(FULL)` | fp32: the native tensor, or the derived copy of a bf16 tensor |
| `get_value(NATIVE)` | the tensor as stored, never a copy |

Forward and backward use the default, so they always run on bf16. Non-float tensors, such as uint32 token ids, are
returned as stored for every precision.

## How the two views stay coherent

The native tensor is the source of truth, and the derived copy is refreshed from it:

- **Writing.** In-place writes go only to the native tensor, through `get_value_for_update()`. It returns a
  `MutableTensorView`; destroying the view bumps a version counter on the tensor.
- **Reading.** The derived copy records the version it was cast from. A read that finds it behind refreshes it in
  place with `ttnn::typecast(native, dtype, std::nullopt, derived)`, into the same buffer, so there is no allocation
  and the buffer address stays the same. A copy that is current is not cast again.
- **Checks.** Each of these is a `TT_FATAL`: asking `get_value_for_update()` for a precision other than the native
  one, taking a second view of a tensor that is being written, reading the other precision while a view is alive,
  and calling `set_tensor()` while a view is alive.
- **Copies.** Copies of an `AutocastTensor` share storage and versioning, so a write through one is seen by all of
  them; a move shares it the same way. `set_tensor()` gives the copy it is called on a new tensor and leaves the
  others alone. When that copy is the only owner, it resets in place, so a reference returned earlier by
  `get_tensor(NATIVE)` follows the new tensor. A reference to the derived copy does not.
- **Limits.** A write that bypasses `get_value_for_update()`, such as a kernel writing through a `get_value()`
  handle, is not seen. The tensor is driven from one host thread.

## Writing a tensor in place

Take the view, pass its tensor to the kernel, and keep the view alive until the kernel is enqueued:

```cpp
auto param = theta->get_value_for_update();
ttml::metal::adamw(param.tensor(), grad, exp_avg.tensor(), exp_avg_sq.tensor(), /* ... */);
// When param goes out of scope after the call, the version moves and the next get_value(FULL) refreshes the copy.
```

These optimizers write in place, and all of them take views:

| Optimizer | What it writes in place |
|---|---|
| `AdamW` (fused) | the parameter and its moments |
| `SGD` (fused) | the parameter and its momentum buffer |
| `AdamWFullPrecision` | its fp32 master weights and moments |
| `MorehAdamW` | the parameter and its moments, as `ttnn::moreh_adamw` output tensors |
| `RemoteOptimizer` | the parameter, received from the aggregator |

`AdamWComposite`, `SGDComposite` and `MuonComposite` compute new tensors and install them with `set_value()`.

Taking the view is checked at run time, not by the compiler, so every in-place write path has a test that reads the
other view after a step (`AutogradTensorTest.*Tracks*` in `tests/autograd/autograd_tensor.cpp`). A new optimizer, or
a new path in an existing one (for example a config option that changes which tensors the step writes), needs one too.
`RemoteOptimizer` is the exception: its step needs a second host, so its write is covered by review only.

## Parameter dtypes the optimizers accept

- Fused `AdamW` accepts bf16 and fp32 parameters. Its moments take the parameter's native dtype, which its kernel
  requires.
- Every other optimizer that updates parameters has a bf16 update only. It rejects an fp32 parameter at
  construction, with an error that names the parameter. `AdamWFullPrecision` keeps its own fp32 master weights for
  bf16 parameters.
- Stochastic rounding (`AdamWConfig::stochastic_rounding`) applies to bf16 parameters only. An fp32 parameter keeps
  the low bits that rounding would lose, so it is updated without it.
- Loading values into an existing tensor keeps its dtype: `Tensor.assign()` (used by the checkpoint and safetensors
  loaders) and the in-place initializers in `ttml.init` cast the new values to the dtype the tensor is stored in.

## Why this design

- **Dropping the other copy on every write** would be the smallest change, but for fp32 parameters it reallocates the
  bf16 compute copy every step, and keeping that copy alive needs the same bookkeeping as a version counter.
- **No cached copy at all** can never be stale, but a weight is read by several ops per step, so it would be cast
  and allocated several times per step, and `get_value()` could no longer return a reference.
- **Letting both copies be written** would round the fp32 tensor to bf16 on write and erase its sub-ulp progress.

PyTorch uses the same idea: one source of truth with a version counter (`c10::VariableVersion`), bumped by in-place
operations in its autograd layer while the kernels take plain tensors. Here the bump happens when the view is
destroyed.
