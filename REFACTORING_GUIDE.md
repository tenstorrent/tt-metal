# Guide: Refactoring `register_operation` to Direct Function Calls

This guide describes the process for transitioning device operations from `ttnn::register_operation` to explicit function declarations and implementations. This refactoring improves compile times and provides more explicit control over operation dispatch.

## 1. Analysis Phase

### 1.1 Extract Operation Details from `.hpp`
Open the target `.hpp` file (e.g., `layernorm_bw_device_operation.hpp`) and identify two key sections:

1.  **The Registration:** Look for the `ttnn::register_operation` template instantiation.
    ```cpp
    // Example from layernorm_bw_device_operation.hpp
    namespace ttnn::prim {
    constexpr auto ttml_layernorm_bw = ttnn::register_operation<
        "ttnn::prim::ttml_layernorm_bw",
        ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation>();
    }
    ```
    *   **Action:** Note the **name of the object** (e.g., `ttml_layernorm_bw`) and the **Operation Class** (e.g., `LayerNormBackwardDeviceOperation`).

2.  **The Invoke Arguments:** Find the `invoke` function within the operation class (usually static and located at the end of the class).
    ```cpp
    // Example arguments from invoke
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& gamma_tensor,
        // ... more args ...
        const std::optional<ttnn::Tensor>& preallocated_dbeta_components = std::nullopt);
    ```
    *   **Action:** Extract the full list of arguments, including their types and any default values.

---

## 2. Header File (`.hpp`) Update

### 2.1 Replace Registration and Invoke
1.  **Remove `ttnn::register_operation`**: Delete the template instantiation and the `constexpr auto` object.
2.  **Remove `invoke` Member Function**: Delete the `static auto invoke(...)` declaration from the Operation Class.
3.  **Add Global Function Declaration**: Add the new function declaration in the same namespace (e.g., `ttnn::prim`).

**Before:**
```cpp
namespace ttml::metal::ops::layernorm_bw::device {
class LayerNormBackwardDeviceOperation {
    // ...
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(...); // REMOVE THIS
};
}

namespace ttnn::prim {
constexpr auto ttml_layernorm_bw = ttnn::register_operation<...>(); // REMOVE THIS
}
```

**After:**
```cpp
namespace ttnn::prim {
ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation::tensor_return_value_t ttml_layernorm_bw(
    const ttnn::Tensor& input_tensor,
    // ... same arguments as original invoke ...
);
}
```

---

## 3. Implementation File (`.cpp`) Update

### 3.1 Implement the Function and Remove Member Invoke
1.  **Remove `OperationType::invoke` Implementation**: Delete the implementation of the static member function from the `.cpp`.
2.  **Implement the Global Function**: Use the logic from the original `invoke` to prepare the attributes and tensor args, then call `launch_on_device`.

**Implementation Pattern:**
```cpp
namespace ttnn::prim {

ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation::tensor_return_value_t ttml_layernorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    // ... args ...) {

    using OperationType = ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation;

    // 1. Prepare attributes and tensor args (Logic previously in OperationType::invoke)
    auto operation_attributes = OperationType::operation_attributes_t{/* attributes if any */};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        // ...
    };

    // 2. Dispatch to device
    return ttnn::device_operation::detail::launch_on_device<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
```

### 3.2 Required Includes
Ensure the `.cpp` file includes the header for `launch_on_device`:
```cpp
#include "ttnn/device_operation.hpp"
```

---

## 4. Verification
To ensure everything is still intact after your changes, run the following build command:

```bash
./build_metal.sh -c -e --debug --build-tt-train
```

1.  **Build:** Verify the build succeeds with the command above.
2.  **Tracking:** Update `device_ops_to_process.txt` by marking the entry as `[x] done`.
