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

- Fused `AdamW` accepts bf16 and fp32 parameters, each with a bf16 or an fp32 gradient. Its moments take the
  parameter's native dtype, which its kernel requires.
- Every other optimizer that updates parameters has a bf16 update only. It rejects an fp32 parameter at
  construction, with an error that names the parameter, and rounds an fp32 gradient to bf16 before its step (see
  [fp32 gradient accumulation](#fp32-gradient-accumulation)).
- Stochastic rounding (`AdamWConfig::stochastic_rounding`) applies to bf16 parameters only. An fp32 parameter keeps
  the low bits that rounding would lose, so it is updated without it, and `AdamW` logs a warning at construction when
  the option meets an fp32 parameter.

## fp32 master weights

`training_config.precision.master_dtype: float32` stores every trainable parameter, the master weight, in fp32, while
forward and backward keep running in bf16. This is not PyTorch FSDP2's `MixedPrecisionPolicy(param_dtype=...)`, which
names the compute dtype. It is applied by `core::apply_precision_config()`
(`sources/ttml/core/parameter_dtype.{hpp,cpp}`) after the model is built and before the optimizer is created, so the
optimizer state follows. It is applied once per run, where the optimizer is built:

- **Python.** `ttml.common.utils.create_optimizer()` applies it when it is given the whole config (as in `train.py`).
  `SFTTrainer` takes the block as the keyword argument `precision=`, and `GRPOTrainer` reads it from
  `GRPOConfig.precision`, which `get_grpo_config()` fills from the YAML (in both, the YAML's
  `training_config.precision` as a dict, whose values may also be `ttnn.DataType.BFLOAT16`/`FLOAT32`). `SFTTrainer`
  applies it after the LoRA wrapping and before it builds the optimizer from a dict or its default; given a ready
  optimizer it applies nothing, and a block passed with one raises. `GRPOTrainer` applies it right after it builds the
  model, and under FSDP shards it, before the weights or a trainer checkpoint load. `None` means no block everywhere,
  as a config without one: the parameters keep their dtype and no reduce dtype is stored, clearing one stored earlier
  in the process.
- **C++.** The `nano_gpt` runner applies it.
- **Checking.** `ttml.core.validate_precision_config()` (C++ `core::validate_precision_config()`) checks the block
  without applying it. Python's `ttml.common.config.TrainingConfig`, which also exposes the block as `.precision`,
  `GRPOConfig` (through `ttml.common.utils.validate_precision()`) and the C++ `nano_gpt`
  `parse_config()` call it, so a typo stops the run before the model is built.

The load order is: build the model, convert, load weights, create the optimizer, then load a training checkpoint,
which restores the optimizer state too. Every loader writes the dtype a tensor is stored in, so values loaded after
the conversion keep their fp32 bits; loaded before it, they are rounded to bf16 first. `train.py` and the `nano_gpt`
runner keep this order; `nano_gpt`'s save-and-exit path skips the conversion, so it exports the weights in the dtype
the model builds.
In the Python hierarchical example the workers keep bf16 parameters for `RemoteOptimizer`; the rank that runs the
optimizer holds the fp32 master weights.
The fp32 parameter is the master weight: fused `AdamW` updates it in place with fp32 moments, and forward and backward
read its bf16 view, which is refreshed into its existing buffer the first time it is read after the step. Parameters
managed by `ttml.fsdp.fully_shard` are the exception: FSDP swaps the gathered weight and the cached fp32 shard in with
`set_value()` (`ttml/fsdp.py`), so each managed shard is cast into a new bf16 buffer every step. Frozen parameters and
non-float tensors keep their dtype. `bfloat16` stores the trainable parameters in bf16; unset, they keep the dtype the
model builds them in, so existing configs don't change.

## Gradient reduction in fp32

Data-parallel training averages each gradient across the devices after backward. By default the sum runs in the
gradient's dtype, bf16, so every hop of the reduction rounds its running total to bf16.
`training_config.precision.reduce_dtype: float32` sums in fp32 instead: a bf16 gradient is cast to fp32, summed and
averaged in fp32, and the DDP sync writes the mean back in bf16, while FSDP keeps its mean shard in fp32; an fp32
gradient, such as one accumulated over micro-batches (see
[fp32 gradient accumulation](#fp32-gradient-accumulation)), is summed and written in fp32. Unset keeps the bf16 sum. A
sum runs in `reduce_dtype` or in the gradient's own dtype, whichever is wider, so `bfloat16` never narrows an fp32
gradient.

- **DDP.** `core::apply_precision_config()` stores the value in `AutoContext` (`set_gradient_reduce_dtype()`) every
  time it runs: a config without the key stores none, so nothing carries over from a config applied earlier in the
  process, and `close_device()` clears it too. A script that sets the value itself does so after `create_optimizer()`,
  or after building `SFTTrainer` or `GRPOTrainer`: they apply the block the same way, so they clear a value set
  before.
  C++ `synchronize_gradients()` and Python `ttml.sync_gradients()` sum in the `reduce_dtype` they are given, else in
  the stored one. `core::distributed::synchronize_tensor()`, the average they call, takes `reduce_dtype` only as an
  argument.
- **FSDP.** `backward_post` reduce-scatters in the `reduce_dtype` given to `ttml.fsdp.fully_shard()`, else in the
  stored one, which it reads when it runs, as the DDP sync does. It first runs in the first backward, after the
  optimizer is created, so a script that shards before `create_optimizer()`, as `train.py` does, needs nothing else. A
  `reduce_dtype` given to `fully_shard()` wins over the stored one for this reduce-scatter; how gradients are
  accumulated before it follows the stored value alone (see
  [fp32 gradient accumulation](#fp32-gradient-accumulation)). The fp32 sum holds more while a gradient is reduced: an
  fp32 copy of the gathered gradient (the bf16 one is freed once it is cast), and the reduce-scatter's working buffer,
  which ttnn allocates in its input's dtype at the input's size, twice that on a line
  (`ReduceScatterMinimalAsyncDeviceOperation::compute_output_specs`). On a line that is about 12 bytes per gathered
  element against 6 for the bf16 sum. With `FLOAT32` the mean shard is kept in fp32, twice the size of a bf16 shard;
  with `BFLOAT16` or none, a bf16 gradient's shard is written in bf16.
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
- **Cost.** Measured on an 8-chip Blackhole machine with `train.py`, median step of steps 6-30 for DDP and of steps
  3-12 for FSDP:

  | Model, mesh | bf16 sum | fp32 sum |
  |---|---|---|
  | nano DeepSeek, DDP 1x8 line | 127.0 ms | 133.7 ms (+5.3%) |
  | TinyLlama, DDP 1x8 line | 1675.6 ms | 1721.4 ms (+2.7%) |
  | TinyLlama, DDP 1x8 ring | 1651.7 ms | 1679.8 ms (+1.7%) |
  | TinyLlama, FSDP 1x8 line, batch 16 x 2048 | 755.4 ms | 817.6 ms (+8.2%) |
  | Llama 8B, FSDP 1x8 line, memory-efficient runner, batch 16 x 2048 | 5025.4 ms | 5382.6 ms (+7.1%) |

  FSDP pays more than DDP: its gradients go through the reduce-scatter alone, so fp32 doubles their bytes, while
  DDP's reduce-scatter route pays 1.5 times. Peak device memory under FSDP (`train.py --track-memory`): Llama 8B goes
  from 12.6 to 15.7 GB per device, the peak being the backward pass's reduce-scatter of the 525M-element embedding
  and LM-head gradient; TinyLlama's peak is set by its forward activations and doesn't change. Models with many small
  gradients pay per call rather than per byte, so fewer, larger collectives would cut most of the cost.
- **Next to the sync.** With `float32`, the sums around the cross-device one run in fp32 too: a module parameter's
  sum over micro-batches (see [fp32 gradient accumulation](#fp32-gradient-accumulation)), and under HSDP (`dp` x
  `fsdp`) the `dp` sum of a sharded parameter's gradient, which adds FSDP's fp32 shards. That sum, like every
  collective sum here, runs on the FPU (the "Not exact" bullet above), so the HSDP gradient is not rounded once: in
  the `[2, 2]` test about 98.5% of its elements, rounded to bf16, land on the exact mean, where an exact sum would give
  100% and rounding the shard to bf16 before the `dp` sum 76%. No shipped config enables HSDP, and none of the
  measurements above is an HSDP run.
- **Not covered yet.** The aggregators of 2- and 3-tier training, `aggregator()` and `aggregator_optimizer()` in the
  Python hierarchical example (`sources/examples/python/multihost/hierarchical_parallel/trainer.py`), add the
  workers' gradients and all-reduce them in bf16.

In the DDP sync, a bf16 gradient's mean is written in bf16 rather than kept in fp32: it is used once per step, so this
costs one rounding, while the optimizer state and the master weights, which accumulate, stay in fp32 with
`master_dtype: float32` (with bf16 parameters, fused `AdamW` keeps bf16 moments). A gradient that is already fp32
stays fp32 through the sync, with one fp32 all-reduce (see Routes above). A sum that is exact in fp32 would need new
kernels; over 17 TinyLlama runs per variant it made no difference to the loss against this one.

How this design was chosen, with the precision, training-impact and collective measurements behind it, is in
[a comment on #56513](https://github.com/tenstorrent/tt-metal/issues/56513#issuecomment-6869872779).

## fp32 gradient accumulation

With `reduce_dtype: float32` stored, a module parameter's gradient is summed in fp32 wherever it is summed, not only
across devices: over the micro-batches of a step, and over the ops that use the same weight in one backward, such as
a tied embedding and LM head.

- **Which tensors.** A module parameter is a tensor a module registered (`ModuleBase::register_tensor()` or
  `override_tensor()`; Python modules register theirs through the same calls, and `Tensor.is_parameter()` tells). Its
  gradient is summed in fp32 when the stored default (`AutoContext::get_gradient_reduce_dtype()`) is `FLOAT32`, read
  at every add: set by `core::apply_precision_config()` or by hand with `set_gradient_reduce_dtype()`, it applies to
  every parameter, including one created or unfrozen later. The stored value alone decides it: a `reduce_dtype`
  passed to `ttml.sync_gradients()`, `synchronize_gradients()` or `ttml.fsdp.fully_shard()` sets the dtype of that sum
  across devices and nothing else. With `float32` stored, `fully_shard(reduce_dtype=BFLOAT16)` still sums in fp32 from
  the second micro-batch on; with nothing stored, an explicit `FLOAT32` sums across devices in fp32 while
  `add_grad()` keeps adding bf16 gradients in bf16. Tensors no module registered, such as activations and the
  memory-efficient runner's `input_detached` (`sources/ttml/models/common/transformer_common.hpp`), keep their
  gradients in the dtype they arrive in, because the backward kernels they feed take bf16. A gradient that is already
  fp32 is always summed in fp32.
- **When.** `Tensor::add_grad()` stores the first gradient as it comes, so a parameter added to once, the usual case
  with one micro-batch, costs nothing extra. From the second add on, the stored gradient and the new one are both cast
  to fp32 and added as two fp32 tensors, which gives the IEEE fp32 sum.
- **zero_grad.** Every optimizer's `zero_grad()` drops the gradients, so the first gradient of a step is stored as it
  comes. `SGD`, `SGDComposite` and `MuonComposite`, which used to install zeros there, install bf16 zeros in `step()`
  instead, for a parameter that had a gradient before `zero_grad()`, got none since and is still trainable, so their
  updates don't change.
  `NoOp`, which updates nothing, only drops them.
- **FSDP.** `backward_post` keeps the mean shard in fp32 under an fp32 reduce. The next micro-batch's `backward_pre`
  gathers it back to full size, an fp32 all-gather, twice the bytes of a bf16 one, and the new gradient is added at
  that size. Adding the new mean into the stored shard instead, as PyTorch's FSDP2 does, would drop that gather; it is
  left to the FSDP gradient accumulation work in #56901.
- **One rounding before the step.** Fused `AdamW` reads an fp32 gradient as it is. The other consumers that take bf16
  gradients install a bf16 copy first with `core::cast_fp32_gradients_to_bf16()`, which frees the fp32 gradient:
  `SGD`, `SGDComposite`, `MorehAdamW`, `AdamWComposite` and `MuonComposite` at the start of their step,
  `clip_grad_norm()`, and `RemoteOptimizer` before it sends. With clipping on, fused `AdamW` therefore gets the clipped
  bf16 gradient; a clip op that reads and scales fp32 gradients in place (#64517) would keep them in fp32 at about the
  same bytes. A Python optimizer registered with `ttml.optimizers.register_optimizer()` gets the gradients as stored;
  one whose math takes bf16 calls `ttml.core.cast_fp32_gradients_to_bf16(parameters)` first.
- **Memory.** An fp32 gradient takes 2 bytes per parameter more than a bf16 one while gradients are held. With fp32
  master weights and fused `AdamW` that is 18 instead of 16 bytes per parameter (4 for the master, 2 for its bf16
  view, 8 for the moments, 4 for the gradient); with bf16 parameters, 10 instead of 8.
- **Cost.** Each extra add moves 18 bytes per element in two dispatches, against 6 for a bf16 add, and the second
  micro-batch, whose stored gradient is still bf16, 24 bytes in three: about 30 to 46 ms per micro-batch for every
  billion gradient elements a chip holds, counted at the gathered size under FSDP. A fused op that adds a bf16 gradient
  into the fp32 one in place would move about 10 bytes in one dispatch (#65669).

Why both operands are cast, with the measurements behind these rules, is in
[a comment on #56513](https://github.com/tenstorrent/tt-metal/issues/56513#issuecomment-6886483685).

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
- The C++ GPT-2 and Llama safetensors loaders read the file's values as floats and install each one with
  `serialization::assign_from_floats()`, which builds the host tensor in the parameter's stored dtype: an fp32
  parameter gets the file's values exactly, a bf16 one the values rounded on the host.
- C++ checkpoints are written as stored (`get_value(NATIVE)`), and `read_autograd_tensor` loads parameters and
  optimizer state through `assign()`. A checkpoint from an fp32 run resumes into a bf16 model as bf16, and the other
  way round, and the AdamW moments follow the parameters, as the fused kernel requires.
- Transfers through `SocketManager` carry the bf16 view, from C++ and from Python. `SocketManager::recv_value()`
  receives a tensor's value: a bf16 tensor in place through `get_value_for_update()`, any other tensor into a bf16
  buffer that `assign()` installs, so it keeps the dtype it is stored in. `RemoteOptimizer` and the Python
  `SocketManager.recv()` receive values this way. `RemoteOptimizer` rounds an fp32 gradient to bf16 before it sends
  it.

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
  kernel, so one optimizer covers both precisions. Parameters managed by FSDP still get a new bf16 buffer each step,
  from FSDP's own swap of the gathered weight and the shard.

PyTorch uses the same idea: one source of truth with a version counter (`c10::VariableVersion`), bumped by in-place
operations in its autograd layer while the kernels take plain tensors. Here the bump happens when the view is
destroyed.
