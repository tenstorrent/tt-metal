# Mixed Precision Design

This note records the design decisions behind tt-train's mixed-precision policy, tracked in
[#56513](https://github.com/tenstorrent/tt-metal/issues/56513). The target policy is:

- parameters and compute in bf16,
- gradient reduction in fp32,
- master weights and optimizer state in fp32.

Each decision below lists the options that were considered and its status. When a PR resolves a decision, it
fills in the rationale here, so reviewers can check a PR against the reasoning.

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

## 1. How the two views stay coherent

**Status: decided, option C. Implemented in PR 2.**

Options:

- **A. Explicit mutating accessor.** `get_value_for_update(precision)` returns the slot and drops the other
  one. Optimizers call it; everyone else calls the read-only `get_value()`.
- **B. No derived cache.** Store one tensor and typecast on every read of the other precision.
- **C. Version stamp.** Keep both slots. The native slot carries a version counter that the mutating accessor
  bumps; the derived slot remembers the version it was cast from and is re-derived on read when it is behind.

The native slot is the only one that can be written. `get_value_for_update(precision)` returns a
`MutableTensorView` over it. Only `AutocastTensor` can construct that type, and it bumps a `uint64` version
counter when it is destroyed, after the kernel that writes through it has been enqueued. Asking for a
precision other than the native one is a `TT_FATAL`. The derived slot stores the version it was cast from.
When a read of it finds an older version, it is refreshed in place with
`ttnn::typecast(native, dtype, std::nullopt, derived)`, which writes into the existing buffer.
`get_value(NATIVE)` returns the native slot as stored, and `set_value` installs a new native tensor and
resets the derived one. The fused `AdamW` and `SGD` steps take a `MutableTensorView` for the parameter and
their state. Composite optimizers already go through `set_value` and do not change.

One rule covers both storage classes: a bf16-native parameter with an fp32 view, and an fp32-native master
weight with a bf16 compute copy. A cast happens only when a stale derived view is read, which is at most once
per optimizer step, and never for a view nobody reads. Buffer addresses stay stable and there is no per-step
allocation, which keeps the door open for trace capture. Peak memory is the same as PyTorch autocast, which
also keeps the bf16 copy alive for backward. The design is PyTorch's (one source of truth plus a version
counter, `c10::TensorImpl::bump_version`), with the bump moved from the dispatcher into the accessor.

Option A reallocates the compute copy on every step under the fp32-master policy and ends up needing C's
bookkeeping anyway. Option B casts and allocates on every read and cannot return a `const&`. Letting both
slots be written was also rejected, because rounding fp32 to bf16 would erase the master's sub-ulp progress.
The limit of C is that it cannot see writes that bypass the accessor. That is mitigated by migrating every
known in-place writer (the fused `AdamW` and `SGD` steps), the private constructor of `MutableTensorView`,
the `TT_FATAL`, a behavioural test per in-place optimizer and storage class, and a line in the review
instructions. The counter is hidden behind one private query, so a future buffer-level version in ttnn can
replace it.

## 2. What a checkpoint contains, and what happens to old ones

**Status: undecided.**

The C++ writer stored `get_value(FULL)`. Since #41385 that is an fp32 copy for bf16 parameters, so C++
checkpoints written since then hold fp32 tensors, and loading one makes those parameters fp32-native.
[#57863](https://github.com/tenstorrent/tt-metal/pull/57863) switched the writer to `NATIVE`. The open part is
old checkpoints.

- **A. Write `NATIVE`, read as-is.** Loading an old checkpoint gives fp32-native parameters; document the
  memory cost.
- **B. Write `NATIVE`, and cast on read** to the dtype the parameter currently has, so a checkpoint never
  changes a parameter's storage class.
- **C. B, plus a format version field** so a reader can tell old fp32-upcast checkpoints from new native ones.

## 3. Where the master weight lives

**Status: undecided.**

- **A. In the optimizer**, as `AdamWFullPrecision` does today: a private map of fp32 tensors, with the model
  parameter staying bf16-native.
- **B. In the parameter**: the parameter is fp32-native, the bf16 view is the compute copy, and any in-place
  optimizer updates the native slot.

## 4. Fate of `AdamWFullPrecision`

**Status: undecided.**

- **A. Keep it** as a separate class.
- **B. Make it an alias**: it sets `param_dtype: float32` on its parameters and delegates to `AdamW`. Requires
  decision 3 = B.
- **C. Deprecate** with a warning for one release, then remove it.

## 5. Gradient dtype after the fp32 reduction

**Status: undecided.**

- **A. Cast back to bf16** after the fp32 all-reduce and scaling. The fused optimizers keep their bf16-gradient
  requirement.
- **B. fp32 gradients end to end**: fp32 seeding and accumulation in `add_grad`, and an fp32-gradient path in
  the `AdamW` kernel.

## 6. What goes over the wire during the reduction

**Status: undecided.**

- **A. Cast, then reduce**: fp32 on the wire, twice the bytes.
- **B. bf16 on the wire, fp32 accumulation inside the collective**, if the CCL kernels support it.
