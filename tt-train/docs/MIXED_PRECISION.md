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
  handle, is not seen. The tensor is driven from one host thread. The ordering also relies on one command queue: the
  view bumps the version on the host after the kernel is enqueued, and the refresh cast is enqueued later on the same
  queue, so it runs after the write. tt-train opens one queue (`sources/ttml/core/mesh_device.cpp`).

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
  construction, with an error that names the parameter.
- Stochastic rounding (`AdamWConfig::stochastic_rounding`) applies to bf16 parameters only. An fp32 parameter keeps
  the low bits that rounding would lose, so it is updated without it.

## fp32 master weights

`training_config.precision.param_dtype: float32` stores every trainable parameter in fp32. It is applied by
`core::apply_precision_config()` (`sources/ttml/core/parameter_dtype.{hpp,cpp}`) right before the optimizer is created,
so the optimizer state follows: in Python by `ttml.common.utils.create_optimizer()` when it is given the whole config
(as in `train.py`), and in C++ by the `nano_gpt` runner and the 3-tier optimizer worker. Runners that pass only the
optimizer section, such as the SFT and GRPO trainers, don't apply it. 3-tier workers keep bf16 parameters for
`RemoteOptimizer`; the optimizer worker holds the fp32 master weights. The fp32 parameter is the master weight: fused
`AdamW` updates it in place with fp32 moments, and forward and backward read its bf16 view, which is refreshed into its
existing buffer the first time it is read after the step. Frozen
parameters and non-float tensors keep their dtype. `bfloat16` stores the trainable parameters in bf16; unset, they keep
the dtype the model builds them in, so existing configs don't change.

## Gradient reduction in fp32

Data-parallel training averages each gradient across the devices after backward. By default the sum runs in the
gradient's dtype, bf16, so every hop of the reduction rounds its running total to bf16.
`training_config.precision.reduce_dtype: float32` sums in fp32 instead: each gradient is cast to fp32, summed and
averaged in fp32, and the mean is written back in the gradient's dtype, so the optimizers still get bf16 gradients, as
the fused kernels require. Unset keeps the bf16 sum.

- **DDP.** `core::apply_precision_config()` stores the value in `AutoContext` (`set_gradient_reduce_dtype()`) when the
  key is present; a config without it leaves the stored value as it is, and `close_device()` clears it. C++
  `synchronize_gradients()` and Python `ttml.sync_gradients()` sum in the `reduce_dtype` they are given, else in the
  stored one. `core::distributed::synchronize_tensor()`, the average they call, takes `reduce_dtype` only as an
  argument.
- **FSDP.** `ttml.fsdp.fully_shard(..., reduce_dtype=...)` makes `backward_post` reduce-scatter in fp32. `train.py`
  reads the key with `ttml.core.reduce_dtype_from_config()` and passes it to every `fully_shard()` call, the way it
  passes `replicate`: sharding happens before the optimizer, and so before `apply_precision_config()`. The fp32 sum
  needs a transient fp32 copy of each gathered gradient, twice its bf16 size; the bf16 gradient is freed once it is
  cast.
- **Routes.** Both use the existing ttnn collectives. A gradient of at least 2^19 elements per device (1 MiB of bf16),
  synced over one mesh axis, with a dim that splits into whole-tile slices that no other mesh axis already splits, is
  reduce-scattered in fp32, its mean is written in the gradient's dtype, and that slice is all-gathered: a quarter fewer
  bytes than an fp32 all-reduce. Every other gradient, and every fp32 gradient, takes one fp32 all-reduce: below the
  threshold a call costs host dispatch rather than link time, and the reduce-scatter route issues more ops. Where both
  routes split the same dim they give the same bits.
- **Not exact.** The FPU adds two values after aligning the smaller one to the larger with 10 fraction bits, even into
  a 32-bit `Dst` (see the `fp32_dest_acc_en` note in
  `docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst`), so the fp32 sum
  is not the exact fp32 sum. On real gradients over 8 devices its error is within 1-3% of the error of rounding the exact mean
  once to bf16, against 2.2-2.7 times that for the bf16 sum, which also inflates gradient magnitudes by about 0.27%.
- **Cost.** Measured on an 8-chip Blackhole machine with `train.py`, median step of steps 6-30:

  | Model, mesh | bf16 sum | fp32 sum |
  |---|---|---|
  | nano DeepSeek, 1x8 line | 127.0 ms | 133.7 ms (+5.3%) |
  | TinyLlama, 1x8 line | 1675.6 ms | 1721.4 ms (+2.7%) |
  | TinyLlama, 1x8 ring | 1651.7 ms | 1679.8 ms (+1.7%) |

  Models with many small gradients pay per call rather than per byte, so fewer, larger collectives would cut most of
  the cost.
- **Not covered.** GRPO calls `fully_shard()` without `reduce_dtype` and builds its optimizer from the optimizer section
  only, and the 3-tier aggregator sums the workers' gradients in bf16, so they keep the bf16 sum.

The mean is cast back to bf16 rather than kept in fp32: a gradient is used once per step, so this costs one rounding,
while the optimizer state and the master weights, which accumulate, stay in fp32. A sum that is exact in fp32 would
need new kernels; over 17 TinyLlama runs per variant it made no difference to the loss against this one.

### How the design was chosen

The options below were measured on an 8-chip Blackhole p150b machine (1x8 line and ring, 2x4, the default 4352-byte
fabric packets, eager dispatch) before the code was written. The bar was a sum clearly better than bf16 that changes
training as little as possible, not a bit-exact one.

**Precision on real gradients** (nano DeepSeek and TinyLlama, captured during training): the share of elements equal
to the exact mean rounded once to bf16, and the mean error relative to that rounding error.

| Devices | bf16 sum | fp32 sum | exact sum |
|---|---|---|---|
| 2 | 80-89%, ≤1.003× | 99.4-99.9%, ≤1.003× | 100%, 1× |
| 4 | 61-67%, 1.6-1.8× | 97.5-98.8%, ≤1.010× | 100%, 1× |
| 8 | 40-54%, 2.2-2.7× | 92.6-97.9%, 1.007-1.032× | 100%, 1× |

The bf16 sum is also biased: it inflates every gradient by about 0.27% on 8 devices (0.08-0.13% on 2); the fp32 sum's
bias is about 60 times smaller. Even on 2 devices the fp32 sum isn't exact, because of how the FPU adds: `add_tiles`
of two bf16 values into an fp32 `Dst` aligns them with 10 fraction bits, so `1 + 2^-20` gives `1`, and only 50.2% of
random bf16 pairs come out exact through `ttnn.add`. Accumulating into `Dst` instead (`copy_tile`, then `add_tiles`
with `acc_to_dest`) adds at fp32 width, and the SFPU adds exactly; per tile on one core they cost 35 ns, 135 ns and
325 ns.

**Training impact:** nano DeepSeek, 2,000 steps on 8 devices, mean loss of the last 50 steps: bf16 sum 0.1068, fp32
0.1066, exact 0.1074, against 0.1076 for a control that only sums the exact copies in reverse order. TinyLlama, 17 seeds per
variant, steps 201-300: fp32 minus exact +0.007 (95% interval −0.085 to +0.100), bf16 minus exact −0.038 (−0.100 to
+0.025). No variant trains measurably better.

**Cost of one collective**, 64 MB of bf16 gradient per device:

| Mesh | bf16 all-reduce | fp32 all-reduce | fp32 reduce-scatter, bf16 all-gather | exact sum |
|---|---|---|---|---|
| 1x8 line | 3.14 ms | 7.15 ms (2.28×) | 4.80 ms (1.53×) | 25.5 ms (8.1×) |
| 1x8 ring | 1.95 ms | 4.76 ms (2.44×) | 3.10 ms (1.59×) | 20.2 ms (10.4×) |

Up to 1 MB per device the two fp32 routes are within 3% of each other: a call there costs 155-190 µs of host time
against 32-49 µs on the device. From 4 MB the reduce-scatter route takes 0.71-0.79 of the all-reduce's time.

**Options and verdicts:**

| Option | What it showed | Verdict |
|---|---|---|
| fp32 all-reduce | twice the bytes of bf16; no slower than the reduce-scatter route up to 1 MB | used below 2^19 elements |
| fp32 reduce-scatter, bf16 all-gather | 1.5 times the bytes; 1.53× bf16 at 64 MB on the line | used from 2^19 elements |
| fp32 all-gather, then cast | same bits, 7.1 against 4.8 ms at 64 MB | not used |
| exact sum (gather, SFPU adds) | +253 ms per nano DeepSeek step, +711 ms per TinyLlama step; no loss difference | not used |
| bf16 on the wire, fp32 accumulation inside the reduce-scatter | would save the casts: at most 2-3% of a TinyLlama step; needs a kernel change, feasible with the accumulate-into-`Dst` add | possible later optimisation |
| ttnn's direct reduce-scatter (≤512 KB) | device time within 10%, host dispatch dominates | no gain |
| chains on a ring, slice exchange (all-to-all) | by a cost model 1.36× today's bytes on an 8-device ring, and 2.29× the busiest-link load of a bf16 reduce-scatter; need kernel work | not pursued |
| fp32 gradients kept after the reduce | the fused optimizers take bf16 gradients; +2 bytes per parameter while gradients live | not done |
| `FABRIC_1D` instead of `FABRIC_2D` | raw links 28.5-30.3 against 22.9 GB/s, but collectives within 1% at 4-64 MB (the ops cap near 21 GB/s per link), and rings would run as lines | no change |
| 10240-byte fabric packets | 1.09-1.32× on 16-64 MB collectives, nothing below 1 MB | separate work |
| trace | the whole nano DeepSeek sync 39.9 → 14.4 ms, TinyLlama unchanged; tt-train can't trace today | later |
| bucketing small gradients | nano DeepSeek's sync is a quarter of its step, the device is busy 28% of it, and one 64 MB call takes 3.1 ms against 14.4 ms of device time for its 249 calls | the main lever for later |

PyTorch's FSDP2 with `reduce_dtype=torch.float32` also reduce-scatters in fp32, but casts the result to the parameter's
dtype, so with fp32 parameters its optimizer gets fp32 gradients; DDP's `bf16_compress_hook` sums in bf16.

## Loading values into a tensor

The checkpoint loaders and `Tensor.assign()` keep the precision a tensor is stored in: the model decides it, not the
file.

- `Tensor::assign()` sets a value cast to the dtype the tensor is stored in. The cast applies between bf16 and fp32.
  A value of the stored dtype, an empty value, or any value for an empty tensor is taken as is; any other dtype
  mismatch is an error. The Python `Tensor.assign()` calls it, so the in-place initializers in `ttml.init` keep each
  tensor's dtype.
- `ttml.autograd.assign_numpy()` loads a numpy array into a tensor in its stored dtype, converted on the host: an
  fp32 tensor gets the exact values, a bf16 tensor the values rounded with `ml_dtypes`. `ttml.checkpointing` (which
  loads parameters and optimizer state into the live tensors), the safetensors loaders, the SFT trainer's default
  checkpoint loader and `LinearLayer.__setstate__` use it.
- C++ checkpoints are written as stored (`get_value(NATIVE)`), and `read_autograd_tensor` loads parameters and
  optimizer state through `assign()`. A checkpoint from an fp32 run resumes into a bf16 model as bf16, and the other
  way round, and the AdamW moments follow the parameters, as the fused kernel requires.
- Transfers through `SocketManager` carry the bf16 view, from C++ and from Python. `SocketManager::recv_value()`
  receives a tensor's value: a bf16 tensor in place through `get_value_for_update()`, any other tensor into a bf16
  buffer that `assign()` installs, so it keeps the dtype it is stored in. `RemoteOptimizer` and the Python
  `SocketManager.recv()` receive values this way.

## Reading values from Python

- `Tensor.to_numpy()` reads the tensor as stored. A bf16 tensor gives a float32 array by default, converted on the
  host, so the read leaves no fp32 copy on the device. `precision=` reads one view instead, and `new_type=` picks
  the array's dtype.
- `Tensor.dtype()` and `Tensor.to_string()` describe the tensor as stored.

## Why this design

- **Dropping the other copy on every write** would be the smallest change, but for fp32 parameters it reallocates the
  bf16 compute copy every step, and the copy's buffer address moves each time.
- **No cached copy at all** can never be stale, but a weight is read by several ops per step, so it would be cast
  and allocated several times per step, and `get_value()` could no longer return a reference.
- **Letting both copies be written** would round the fp32 tensor to bf16 on write and erase its sub-ulp progress.
- **Keeping the fp32 master weights in the optimizer**, next to bf16 parameters, means casting each master into a new
  bf16 tensor and installing it after every step: an allocation per parameter per step and a buffer address that
  moves. With the master in the parameter, the bf16 view is refreshed in place and the step runs the same `AdamW`
  kernel, so one optimizer covers both precisions.

PyTorch uses the same idea: one source of truth with a version counter (`c10::VariableVersion`), bumped by in-place
operations in its autograd layer while the kernels take plain tensors. Here the bump happens when the view is
destroyed.
