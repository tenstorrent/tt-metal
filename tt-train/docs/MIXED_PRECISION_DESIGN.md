# Mixed Precision Design

This note records the design behind tt-train's mixed-precision policy, tracked in
[#56513](https://github.com/tenstorrent/tt-metal/issues/56513). The target policy is:

- parameters and compute in bf16,
- gradient reduction in fp32,
- master weights and optimizer state in fp32.

Each section describes a design choice: what it is, why, and which alternatives were dropped and why. Questions
that are still open are listed at the end; the PR that settles one moves it into its own section with the
reasoning, so reviewers can check a PR against it.

## Background: two views of one parameter

`autograd::AutocastTensor` (`sources/ttml/autograd/autocast_tensor.{hpp,cpp}`) stores a tensor in its native
precision (bf16 or fp32). It creates the other precision lazily, as a typecast copy, the first time someone
asks for it with `get_value(HALF)` or `get_value(FULL)`. `get_value()` defaults to `HALF`, so forward and
backward always run on a bf16 tensor. `get_value(NATIVE)` returns the stored tensor without casting.

Only `set_value` invalidates the cached copy. The fused `AdamW` and `SGD` optimizers update the `HALF` view in
place, so a cached `FULL` view keeps its old values
([#41657](https://github.com/tenstorrent/tt-metal/issues/41657)). For an fp32-native parameter the fused step
updates only the bf16 copy, and the fp32 tensor never changes. The bf16-native case started with
[#41385](https://github.com/tenstorrent/tt-metal/pull/41385), which made the second copy lazy to save memory
for fp32 tensors. Before it, a bf16 tensor was stored once and returned for both `HALF` and `FULL`.

## How the two views stay coherent

This is built in PR 2 ([#58452](https://github.com/tenstorrent/tt-metal/issues/58452)); until it lands, none
of it exists in the code.

In PR 2 the native slot will be the only one that can be written. `get_value_for_update(precision)` will
return a `MutableTensorView` over it. Only `AutocastTensor` will be able to construct that type, and it will
bump a `uint64` version counter when it is destroyed, after the kernel that writes through it has been
enqueued. Asking for a precision other than the native one will be a `TT_FATAL`. The derived slot will store
the version it was cast from. When a read of it finds an older version, it will be refreshed in place with
`ttnn::typecast(native, dtype, std::nullopt, derived)`, which writes into the existing buffer.
`get_value(NATIVE)` will keep returning the native slot as stored, and `set_value` will install a new native
tensor and reset the derived one. `FULL` keeps its current meaning, fp32 with a cast when the tensor is bf16;
readers that mean "the value as stored" use `NATIVE`.

Five optimizers write parameters in place, and all five will take a `MutableTensorView` for the parameter and
its state. Three of them go through tt-train's fused kernels: the fused `AdamW` and `SGD` steps, and
`AdamWFullPrecision`, which updates its fp32 master weights and moments through the same kernel. Their kernel
wrappers will take the view in their signatures, so passing a plain `get_value()` result will not compile. The
other two write through functions tt-train does not own: `MorehAdamW` passes the parameter and its moments as
output tensors to `ttnn::moreh_adamw`, and `RemoteOptimizer` receives weights straight into the parameter's
buffer. For those two the view is a convention. `MorehAdamW`'s is backed by a test; `RemoteOptimizer`'s step
needs a second host, so its write is covered only by review. `AdamWComposite`, `SGDComposite` and
`MuonComposite` build new tensors and install them with `set_value`, so they do not change.

Two optimizer contracts have to move with it. `AdamW` creates its moments from the `HALF` view
(`optimizers/adamw.cpp`), while its device op requires the moments to have the parameter's dtype
(`adamw_device_operation.cpp`), so for an fp32-native parameter PR 2 will create the moments from `NATIVE`;
the kernel already accepts fp32 moments. The fused `SGD` kernel accepts only bf16 parameters and momentum
(`sgd_device_operation.cpp`). Rather than add an fp32 path to it now, PR 2 makes every optimizer that updates
parameters, except fused `AdamW`, reject an fp32 parameter at construction, with an error that names the parameter, until the precision
config lands. Before, they silently trained a bf16 copy or turned the parameter into bf16 through `set_value`.

One rule covers both storage classes: a bf16-native parameter with an fp32 view, and an fp32-native master
weight with a bf16 compute copy. A cast happens only when a stale derived view is read, which is at most once
per optimizer step, and never for a view nobody reads. Buffer addresses stay stable and there is no per-step
allocation, which keeps the door open for trace capture. Peak memory is the same as PyTorch autocast, which
also keeps the bf16 copy alive for backward. The design is PyTorch's (one source of truth plus a version
counter, `c10::TensorImpl::bump_version`), with the bump moved from the dispatcher into the accessor.

Alternatives we dropped:

- **Drop the other copy on every write.** The accessor would return the slot and discard the other one. It is
  the smallest change, but under the fp32-master policy it reallocates the compute copy on every step, and
  keeping that copy alive ends up needing the same bookkeeping as the version counter.
- **No cached copy at all.** Store one tensor and typecast on every read of the other precision. It can never
  be stale, but an fp32 weight is read by several ops per step, so it would be cast and allocated several times
  per step, and `get_value()` could no longer return a `const&`.
- **Let both copies be written.** Rounding the fp32 master to bf16 and writing it back would erase the master's
  sub-ulp progress, which is the reason to keep an fp32 master in the first place.

The limit of the version counter is that it cannot see writes that bypass the accessor. That is mitigated by
migrating every known in-place writer (the five above), the private constructor of `MutableTensorView` that the
fused kernel wrappers require, the `TT_FATAL`, behavioural tests for the in-place optimizers that run on one
device, and a line in the review instructions. The counter is hidden behind one private query, so a future
buffer-level version in ttnn can replace it. That would also cover writers outside tt-train's own wrappers.

## Open questions

Each is settled by the PR named in its heading, which then moves it into its own section with the reasoning.

### What a checkpoint contains, and what happens to old ones (PR 2)

The C++ writer stored `get_value(FULL)`. Since #41385 that is an fp32 copy for bf16 parameters, so C++
checkpoints written since then hold fp32 tensors, and loading one makes those parameters fp32-native.
[#57863](https://github.com/tenstorrent/tt-metal/pull/57863) (in review) switches the writer to `NATIVE`. The
open part is loading: keep the stored dtype and document the memory cost, or cast each loaded value to the
dtype the parameter currently has, so a checkpoint never changes a parameter's storage class. A format version
field would additionally let a reader tell the older fp32 checkpoints from native ones.

### Where the master weight lives (PR 5)

Either in the optimizer, as `AdamWFullPrecision` does today, with a private map of fp32 tensors and the model
parameter staying bf16-native; or in the parameter itself, which is then fp32-native, with the bf16 view as the
compute copy and every in-place optimizer updating the native slot.

### The future of `AdamWFullPrecision` (PR 5)

Keep it as a separate class; reduce it to a thin wrapper that stores its parameters in fp32 and delegates to
`AdamW`, which only works if the master weight lives in the parameter; or deprecate it with a warning for one
release and then remove it.

### Gradient dtype after the fp32 reduction (PR 4)

Cast back to bf16 after the fp32 all-reduce and scaling, so the fused optimizers keep their bf16-gradient
requirement; or keep gradients in fp32 end to end, with fp32 seeding and accumulation in `add_grad` and an
fp32-gradient path in the `AdamW` kernel.

### What goes over the wire during the reduction (PR 4)

Cast, then reduce, with fp32 on the wire and twice the bytes; or bf16 on the wire with fp32 accumulation
inside the collective, if the CCL kernels support it.
