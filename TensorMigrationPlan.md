
# **PR Breakdown (quick index)**


1. **PR1**: Namespace swap for types (`ttnn` → `tt::tt_metal`, `ttnn::` → `ttsl::`)

2. **PR2**: Move `QueueId` to tt-metal (+ shim)

3. **PR3**: Decouple `TensorID` from `CoreID`

4. **PR4**: Replace `tensor` methods which represent the ops with ops usage.

5. **PR5**: Move `tensor/` dir to `tt-metal` \+ distributed deps

6. **PR6**: Introduce `host-only` ops in tt-metal, wire `ttnn` ops to them


---

# **Phase 1 — “Independent” Changes (Safe, Mechanical)**

## **PR1 — Namespace swap (ttnn → tt::tt\_metal, ttsl) for *types***


**Reason**

* To simplify migration, switch to the tt-metal namespace first for types such as Tensor. Because tt-metal isn’t aware of ttnn, this isolates mechanical edits from the meaningful changes we need to review.

**Scope**

* Substitute `ttnn::SmallVector` with `ttsl::SmallVector`.

* Replace `ttnn::` namespaces for `tensors, shapes, MeshDevice, ...` **types** with `tt::tt_metal::...`.

* No functional or ownership changes.

* We keep existing ttnn aliases, customers can still use `ttnn::` namespace.

---

## **PR2 — Move `QueueId` to tt-metal**

**Reason**

* `QueueId` is widely used by `Tensor`, already lives in the `tt-metal` namespace, and the suggestion to move it to tt-metal comes from its own implementation file

**Scope**

* Move `ttnn/api/ttnn/common/queue_id.hpp` to `tt_metal/api/tt-metalium/common/queue_id.hpp`.

* Update all includes and namespaces.

* Keep existing ttnn header file to forward-includes the new location.



---

## **PR3 — Decouple TensorID from CoreID**

**Reason**
* `tensor_id` is part of `Tensor`, but the counter resides in `CoreId`, which also tracks `python_operation_id` and `device_operation_id` — both specific to `ttnn`. This update moves `tensor_id` out of `CoreId` and assigns it to all tensors.”

**Scope**

* Change `Tensor::tensor_id` type from `std::optional<int64_t>` to **`uint64_t`** .

* `tensor_id` is assign **at Tensor construction** using `static std::atomic<uint64_t>` counter

* **Copy/Move semantics:**

  * **Copy**: It **preserve** `tensor_id` (represents “same logical tensor content”). (keep current behavior)

  * **Move**: transfer ID (moved-from becomes 0 or a sentinel).

* Move `tensor_id` from `CoreId` into `Tensor.{h,cpp}` in tt-metal.

* Provide `get_tensor_id()/set_tensor_id()` **ops** (not methods) in tt-metal host-ops API.

* In `ttnn/ttnn/decorators.py`, switch to using new **ops**;

**Perf**

* Verify there’s no performance regression from this change


---

# **Phase 2 — Breaking Tensor API - Method → Op redirection (keep behavior intact)**

## **PR4 — Replace tensor methods which represent the ops with ops usage.**

**Reason:**

* Don't use `Tensor` methods for **ops**. Once `Tensor` moves to `tt-metal`, its methods will be host-only. If we migrate it now, callers that rely on device dispatch via `tensor methods` would lose that behavior. To preserve device semantics, we should use ops (not tensor methods) when working with tensors; each op will decide whether to run on host or device. This lets us migrate `Tensor` safely while keeping device behavior in `ttnn`, with no user-visible change

**Scope**

One method/op `pull request` per operation.

Use `ttnn/tt-metal` operations instead of Tensor methods. Internally `ttnn` ops call `tt-metal` host ops or device ops based on dispatch policy.

Move the following tensor ops implementation from Tensor API to free functions, replace call-sites that use `Tensor` **methods** with `ttnn/tt_metal` **ops**.
Host-only ops implementation can be created in tt-metal to simplify the migration(optional)

* `Tensor::reshape(Tensor, new_shape)`

* `Tensor::print`

* `Tensor::write_to_string(const Tensor&)`

* `Tensor::to_layout(Tensor, Layout)`

* `Tensor::pad/Tensor::unpad(Tensor, spec)`

* `Tensor::pad_to_tile/Tensor::unpad_from_tile(Tensor)`

* `Tensor::cpu(Tensor)`

**Python binding changes**

* In pybind for Tensor `__repr__`, call `write_tensor_to_string(tensor)`, not `tensor.write_to_string()`.

**Tests**

* Update unit tests.

* Confirm the tests cover the new behavior.



---

# **Phase 3 — Move Tensor code to tt-metal (minimal behavior change)**

## **PR5 — Move `tensor/` directory to tt-metal (plus dependencies)**

**Scope**

* Move `ttnn/api/ttnn/tensor/*` to:

  * Public headers: `tt_metal/api/tt-metalium/tensor/*`

  * Impl: `tt_metal/impl/tensor/*`

* Move distributed `Tensor` deps:

  * `tt_metal/impl/tensor/distributed/tensor_topology.cpp`

  * `tt_metal/impl/tensor/distributed/distributed_configs.cpp`

* Move flatbuffer serialization (optionaly)
  * Keep **flatbuffer dump** in `ttnn` due to dependency from `host_ccl::all_gather`.

**Shims**

* Thin headers in `ttnn/.../tensor/*` that include the new locations (deprecated banners).

**Build**

* Update targets/exports (CMake: install/export)


---

## **PR6 — Introduce host-only ops in tt-metal, wire `ttnn` ops to them**

**Scope**

* Implement the following host only ops for tt-metal
  (if not introduced during `Phase 2:PR4`):

  * `to_dtype(tensor,dtype)`
  * `reshape(tensor,shape)`
  * `to_string(tensor)`
  * `tensor_print(tensor)`
  * `to_layout(Tensor, Layout)`
  * `pad(Tensor, spec)`
  * `unpad(Tensor, spec)`
  * `pad_to_tile(Tensor)`
  * `unpad_from_tile(Tensor)`
  * `cpu(Tensor)`


---

# **Open questions**

* **tensor\_id type**: adopt `uint64_t` instead of `std::optional<int64>`?

* **copy semantics**: preserve `tensor_id` on tensor copy?
