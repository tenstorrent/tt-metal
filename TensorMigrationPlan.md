
# **PR Breakdown (quick index)**


1. **PR1**: Namespace swap for types (`ttnn` → `tt::tt_metal`, `ttnn::` → `ttsl::`)

2. **PR2**: Move `QueueId` to tt-metal (+ shim)

3. **PR3**: Decouple `TensorID` from `CoreID`

4. **PR4**: Implement tensor host-only `to_dtype` operation

5. **PR5**: Implement tensor host-only `reshape` operation

6. **PR6**: Replace the `ttnn` op with the `tensor_ops` equivalent for the `to_string` tensor operation.

5. **PR7**: Move `tensor/` dir to `tt-metal` \+ distributed deps under tt::tt_metal::experimental namespace


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

# **Phase 2 —  Introduce tensor host-only ops**

## **PR4 — Implement tensor host-only to_dtype operation**

**Scope**

* Move ttnn::to_dtype implementation into tensor_ops::to_dtype, ttnn::to_dtype should reuse the implementation.


## **PR5 — Implement tensor host-only reshape operation**

**Scope**

* Replace all tensor_reshape calls with ttnn::reshape (since tensor_reshape is just a wrapper). Introduce host-only tensor_ops::reshape, which will be used by tensor.tensor_reshape(). Tensor.reshape() method can be safely deleted.


## **PR6 — Replace the ttnn op with the tensor_ops equivalent for the to_string tensor operation**

**Scope**

* Use the internal tensor_impl::to_layout and tensor_impl::to_dtype within tensor_impl::to_string (replacing ttnn ops) to allow clean migration to tt-metal.
Verify that the change does not affect the current expected behavior.

Introduce a ttnn::to_string(tensor) operation for pybind and move the ttnn::distributed::get_device_tensors logic from tensor_impl::to_string<T> into it, since distributed tensors will remain in ttnn for now.

---

# **Phase 3 — Move Tensor code to tt-metal (minimal behavior change)**

## **PR7- Move `tensor/` directory to `tt-metal` \+ distributed deps under tt::tt_metal::experimental namespace**

**Scope**

* Move `ttnn/api/ttnn/tensor/*` to:

  * Public headers: `tt_metal/api/experimental/tt-metalium/tensor/*`

  * Impl: `tt_metal/impl/experimental/tensor/*`

* Move distributed `Tensor` deps:

  * `tt_metal/impl/experimental/tensor/distributed/tensor_topology.cpp`

  * `tt_metal/impl/experimental/tensor/distributed/distributed_configs.cpp`

* Move flatbuffer serialization (optionaly)
  * Keep **flatbuffer dump** in `ttnn` due to dependency from `host_ccl::all_gather`.

**Shims**

* Thin headers in `ttnn/.../tensor/*` that include the new locations (deprecated banners).

**Build**

* Update targets/exports (CMake: install/export)


---


# **Open questions**

* **tensor\_id type**: adopt `uint64_t` instead of `std::optional<int64>`?

* **copy semantics**: preserve `tensor_id` on tensor copy?
