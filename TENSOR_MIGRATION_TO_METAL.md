# Tensor API Migration from TTNN to Metal

## Executive Summary

This document outlines the comprehensive plan to move core tensor functionality and host-device APIs from the TTNN layer to the Metal layer. This migration will enable users who only need Metal (without neural network operations) to access tensor functionality without depending on TTNN, while maintaining backward compatibility for existing TTNN users.

**Key Goals:**
- Move core tensor types, classes, and host↔device APIs from `ttnn/` to `tt_metal/`
- Enable Metal-only users to work with tensors without TTNN dependency
- Maintain clean separation: Metal = framework-agnostic, TTNN = framework integration
- Preserve backward compatibility for existing TTNN users
- No impact on Python API surface

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Migration Scope](#migration-scope)
3. [APIs to Migrate](#apis-to-migrate)
4. [APIs to Keep in TTNN](#apis-to-keep-in-ttnn)
5. [Architecture After Migration](#architecture-after-migration)
6. [Usage Examples](#usage-examples)
7. [Implementation Strategy](#implementation-strategy)
8. [Timeline](#timeline)
9. [Critical Considerations](#critical-considerations)
10. [Testing Strategy](#testing-strategy)

---

## Current State Analysis

### Current Location
Tensor functionality currently lives in:
- **Headers:** `ttnn/api/ttnn/tensor/*.hpp`
- **Implementation:** `ttnn/core/tensor/*.cpp`
- **Python bindings:** `ttnn/cpp/ttnn-pybind/pytensor.cpp`

### Namespace Paradox
Interestingly, tensor types are **already in the `tt::tt_metal` namespace** despite physically living in the `ttnn/` directory:

```cpp
// ttnn/api/ttnn/tensor/types.hpp
namespace tt {
namespace tt_metal {
    enum class DataType { ... };
    enum class StorageType { ... };
    class Tensor { ... };
}
}

// ttnn/api/ttnn/types.hpp
namespace ttnn {
    using Tensor = tt::tt_metal::Tensor;  // Just an alias!
}
```

This means the **namespace migration is already done** - we only need to move the physical files and build system configuration.

### Dependencies
Current dependencies from `ttnn/api/ttnn/tensor/types.hpp`:
```cpp
#include <tt-metalium/bfloat16.hpp>        // ✅ Already in metal
#include <tt-metalium/core_coord.hpp>      // ✅ Already in metal
#include <tt-metalium/buffer.hpp>          // ✅ Already in metal
#include <tt-metalium/mesh_buffer.hpp>     // ✅ Already in metal
#include <tt-metalium/device.hpp>          // ✅ Already in metal
#include <tt_stl/reflection.hpp>           // ✅ Standard library
#include <tt_stl/span.hpp>                 // ✅ Standard library

#include "ttnn/tensor/shape/shape.hpp"     // ⚠️ Needs to move
```

**Finding:** Tensor types have minimal TTNN-specific dependencies. Most imports are already from Metal.

---

## Migration Scope

### Problem Statement
**Current Blocker:** Users who want to consume only Metal (e.g., custom frameworks, direct hardware programming) are forced to depend on the entire TTNN stack just to use tensors for host-device data transfer.

**Goal:** Enable this use case:
```cpp
// Pure Metal code - no TTNN dependency
#include <tt-metalium/tensor.hpp>
#include <tt-metalium/device.hpp>

auto device = MeshDevice::create_unit_mesh(0);
std::vector<float> data = load_my_data();
auto tensor = Tensor::from_vector(data, spec).to_device(device);
```

### What This Migration Enables

1. **Metal-only users** can work with tensors without TTNN
2. **Framework integrations** (Forge, PyTorch backend) can depend only on Metal
3. **Cleaner architecture** with proper layering:
   - Metal: Core tensor + device management
   - TTNN: Neural network operations + framework integration

---

## APIs to Migrate

All APIs listed below will move from `ttnn/` to `tt_metal/`.

### 1. Core Type Definitions

**Source:** `ttnn/api/ttnn/tensor/types.hpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/tensor_types.hpp`

```cpp
namespace tt::tt_metal {

// Data types supported by hardware
enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT8 = 5,
    UINT16 = 6,
    INT32 = 7,
    INVALID = 8,
};

// Tensor storage location
enum class StorageType {
    HOST = 0,
    DEVICE = 1,
};

// Memory layout format
enum class Layout {
    ROW_MAJOR = 0,
    TILE = 1,
};

// Data type utilities
template <typename T>
consteval DataType convert_to_data_type();

bool is_floating_point(DataType dtype);
bool is_block_float(DataType dtype);
DataFormat datatype_to_dataformat_converter(DataType datatype);
DataType dataformat_to_datatype_converter(DataFormat dataformat);

// Multi-dimensional array types
static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;
using Array1D = std::array<uint32_t, 1>;
using Array2D = std::array<uint32_t, 2>;
using Array3D = std::array<uint32_t, 3>;
using Array4D = std::array<uint32_t, 4>;
using Array5D = std::array<uint32_t, 5>;
using Array6D = std::array<uint32_t, 6>;
using Array7D = std::array<uint32_t, 7>;
using Array8D = std::array<uint32_t, 8>;

// Sharding specification
struct NdShardSpec {
    Shape shard_shape;
    CoreRangeSet grid;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;

    NdShardSpec with_shard_shape(Shape new_shard_shape) const;
    bool operator==(const NdShardSpec& other) const = default;
    bool operator!=(const NdShardSpec& other) const = default;
};

}  // namespace tt::tt_metal
```

### 2. Shape Classes

**Source:** `ttnn/api/ttnn/tensor/shape/shape.hpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/shape.hpp`

```cpp
namespace tt::tt_metal {

class Shape {
    // Multi-dimensional shape representation
    // Supports 1D to 8D tensors
    // Handles both logical and padded shapes for tile alignment
};

class Shape2D {
    // 2D shape specialized for height/width access
    uint32_t height() const;
    uint32_t width() const;
};

}  // namespace tt::tt_metal
```

### 3. Storage Variants

**Source:** `ttnn/api/ttnn/tensor/storage.hpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/storage.hpp`

```cpp
namespace tt::tt_metal {

// Host memory storage
struct HostStorage {
    DistributedHostBuffer buffer_;

    const DistributedHostBuffer& buffer() const;
};

// Device memory storage
struct DeviceStorage {
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
    std::vector<distributed::MeshCoordinate> coords;

    Buffer* get_buffer() const;
    std::shared_ptr<distributed::MeshBuffer> get_mesh_buffer() const;
};

// Storage can be either on host or device
using Storage = std::variant<HostStorage, DeviceStorage>;

}  // namespace tt::tt_metal
```

### 4. Tensor Specification

**Source:** `ttnn/api/ttnn/tensor/tensor_spec.hpp`, `tensor_layout.hpp`, `memory_config/memory_config.hpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/tensor_spec.hpp`

```cpp
namespace tt::tt_metal {

// Page configuration for tiled/row-major layout
struct RowMajorPageConfig { /* ... */ };
struct TilePageConfig {
    Tile tile;
};
using PageConfig = std::variant<RowMajorPageConfig, TilePageConfig>;

// Memory configuration for device tensors
class MemoryConfig {
    TensorMemoryLayout memory_layout;
    BufferType buffer_type;
    std::optional<ShardSpec> shard_spec;
    std::optional<NdShardSpec> nd_shard_spec;

    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;
};

// Complete tensor layout specification
class TensorLayout {
    DataType data_type;
    PageConfig page_config;
    MemoryConfig memory_config;
    std::vector<uint32_t> alignment;

    Layout get_layout() const;
    DataType get_data_type() const;
    MemoryConfig get_memory_config() const;
};

// Tile configuration
struct Tile {
    uint32_t tile_shape_h = 32;
    uint32_t tile_shape_w = 32;
    bool transpose_tile = false;

    std::array<uint32_t, 2> get_tile_shape() const;
    std::array<uint32_t, 2> get_face_shape() const;
    bool get_transpose_within_face() const;
};

// Complete tensor specification
class TensorSpec {
    Shape logical_shape_;
    Shape padded_shape_;
    TensorLayout tensor_layout_;

    const Shape& logical_shape() const;
    const Shape& padded_shape() const;
    const TensorLayout& tensor_layout() const;
    Shape physical_shape() const;

    size_t compute_packed_buffer_size_bytes() const;

    TensorSpec with_memory_config(const MemoryConfig&) const;
    TensorSpec with_tile(const Tile&) const;
};

}  // namespace tt::tt_metal
```

### 5. Core Tensor Class

**Source:** `ttnn/api/ttnn/tensor/tensor.hpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/tensor.hpp`

```cpp
namespace tt::tt_metal {

class Tensor {
public:
    std::optional<std::int64_t> tensor_id = std::nullopt;
    std::shared_ptr<TensorAttributes> tensor_attributes = nullptr;
    std::optional<distributed::MeshDevice*> mesh_device_ = std::nullopt;

    // ==================== Constructors ====================

    [[nodiscard]] explicit Tensor() = default;
    [[nodiscard]] Tensor(const Tensor& other);
    [[nodiscard]] Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // Construct from storage + spec
    [[nodiscard]] Tensor(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);

    // Construct from host buffer + metadata
    [[nodiscard]] Tensor(
        HostBuffer buffer,
        const Shape& shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);

    [[nodiscard]] Tensor(
        HostBuffer buffer,
        const Shape& logical_shape,
        const Shape& padded_shape,
        DataType dtype,
        Layout layout,
        const std::optional<Tile>& tile = std::nullopt);

    [[nodiscard]] Tensor(HostBuffer buffer, TensorSpec tensor_spec);

    // ==================== Static Factory Methods ====================

    // Create from span of typed data (copies data)
    template <typename T>
    [[nodiscard]] static Tensor from_span(
        tt::stl::Span<const T> buffer,
        const TensorSpec& spec,
        distributed::MeshDevice* device = nullptr,
        std::optional<QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    // Create from vector (copies data)
    template <typename T>
    [[nodiscard]] static Tensor from_vector(
        const std::vector<T>& buffer,
        const TensorSpec& spec,
        distributed::MeshDevice* device = nullptr,
        std::optional<QueueId> cq_id = std::nullopt,
        T pad_value = 0);

    // Create with borrowed data (zero-copy, caller manages lifetime)
    template <typename T>
    [[nodiscard]] static Tensor from_borrowed_data(
        tt::stl::Span<T> buffer,
        const Shape& shape,
        MemoryPin buffer_pin,
        const std::optional<Tile>& tile = std::nullopt);

    // ==================== Device Transfer ====================

    // Move tensor to device
    [[nodiscard]] Tensor to_device(
        distributed::MeshDevice* mesh_device,
        ttsl::optional_reference<const MemoryConfig> mem_config = std::nullopt,
        std::optional<QueueId> cq_id = std::nullopt) const;

    // Move tensor to host
    [[nodiscard]] Tensor to_host(
        bool blocking = true,
        std::optional<QueueId> cq_id = std::nullopt) const;

    // ==================== Memory Management ====================

    void deallocate(bool force = false);
    bool is_allocated() const;

    // ==================== Accessors ====================

    const Shape& logical_shape() const;
    const Shape& padded_shape() const;
    DataType dtype() const;
    Layout layout() const;
    const TensorSpec& tensor_spec() const;

    StorageType storage_type() const;
    Storage& storage();
    const Storage& storage() const;

    const HostStorage& host_storage() const&;
    const DeviceStorage& device_storage() const&;

    distributed::MeshDevice* device() const;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer() const;
    Buffer* buffer() const;

    const MemoryConfig& memory_config() const;
    const TensorTopology& tensor_topology() const;

    bool is_sharded() const;
    uint32_t element_size() const;

    // Extract single value (for scalars)
    template <typename T>
    [[nodiscard]] T item(std::optional<QueueId> cq_id = std::nullopt) const;

private:
    void init(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);
    void deallocate_impl(bool force);
};

}  // namespace tt::tt_metal
```

### 6. Tensor Allocation APIs

**Source:** `ttnn/core/tensor/tensor.cpp`
**Destination:** `tt_metal/impl/tensor/tensor_ops.cpp`
**Header:** `tt_metal/api/tt-metalium/tensor/tensor_ops.hpp`

```cpp
namespace tt::tt_metal {

// Allocate tensor on device with specified layout and memory configuration
Tensor allocate_tensor_on_device(
    const TensorSpec& tensor_spec,
    distributed::MeshDevice* mesh_device);

// Allocate tensor on host (for multi-device, allocates one buffer per device)
Tensor allocate_tensor_on_host(
    const TensorSpec& tensor_spec,
    distributed::MeshDevice* mesh_device);

// Legacy API - allocate device tensor (deprecated)
[[deprecated]]
Tensor create_device_tensor(
    const Shape& shape,
    DataType dtype,
    Layout layout,
    IDevice* device,
    const MemoryConfig& memory_config = MemoryConfig{},
    const std::optional<Tile>& tile = std::nullopt);

// Modern API - allocate device tensor
Tensor create_device_tensor(
    const TensorSpec& tensor_spec,
    IDevice* device);

}  // namespace tt::tt_metal
```

### 7. Host-Device Transfer APIs

**Source:** `ttnn/core/tensor/tensor.cpp`
**Destination:** `tt_metal/impl/tensor/tensor_ops.cpp`

```cpp
namespace tt::tt_metal {

// High-level write: supports host→device and device→host
void write_tensor(
    const Tensor& src,
    Tensor& dst,
    bool blocking = true,
    std::optional<QueueId> cq_id = std::nullopt);

// Low-level memcpy overloads
void memcpy(
    distributed::MeshCommandQueue& queue,
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const void* src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    distributed::MeshCommandQueue& queue,
    Tensor& dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    void* dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void memcpy(
    Tensor& dst,
    const void* src,
    const std::optional<BufferRegion>& region = std::nullopt);

void memcpy(
    Tensor& dst,
    const Tensor& src,
    const std::optional<BufferRegion>& region = std::nullopt);

}  // namespace tt::tt_metal
```

### 8. Tensor Implementation Utilities

**Source:** `ttnn/api/ttnn/tensor/tensor_impl.hpp` and `ttnn/core/tensor/tensor_impl.cpp`
**Destination:** `tt_metal/api/tt-metalium/tensor/tensor_impl.hpp`

```cpp
namespace tt::tt_metal::tensor_impl {

// ==================== Buffer Allocation ====================

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
    distributed::MeshDevice* mesh_device,
    const TensorSpec& tensor_spec);

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec);

// ==================== Type Utilities ====================

uint32_t element_size_bytes(DataType dtype);

template <typename OutputDataType, typename InputDataType>
std::vector<OutputDataType> cast_vec(tt::stl::Span<const InputDataType> data_to_convert);

// ==================== Data Packing/Unpacking ====================

template <typename T>
std::vector<uint32_t> pack_vec_into_uint32_vec(tt::stl::Span<const T> data_to_pack);

template <typename T>
std::vector<T> unpack_uint32_vec_into_vec(tt::stl::Span<const uint32_t> data_to_unpack);

template <typename T>
constexpr size_t packed_buffer_size_bytes(size_t volume_unpacked_data);

// ==================== Layout Conversion ====================

template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape,
    const Tile& tile,
    tt::stl::Span<const T> data_to_convert);

template <typename T>
std::vector<T> convert_layout_tile_to_row_major(
    const Shape2D& shape,
    const Tile& tile,
    tt::stl::Span<const T> data_to_convert);

// ==================== Helper Functions ====================

bool logical_matches_physical(const TensorSpec& tensor_spec);

// Template implementations for to_device/to_host
template <typename T>
Tensor to_host(
    const Tensor& tensor,
    bool blocking = true,
    std::optional<QueueId> cq_id = std::nullopt);

template <typename T>
void copy_to_host(
    const Tensor& device_tensor,
    Tensor& host_tensor,
    bool blocking = true,
    std::optional<QueueId> cq_id = std::nullopt);

template <typename T>
Tensor to_device(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config = std::nullopt,
    std::optional<QueueId> cq_id = std::nullopt);

template <typename T>
void copy_to_device(
    const Tensor& host_tensor,
    Tensor& device_tensor,
    std::optional<QueueId> cq_id = std::nullopt);

}  // namespace tt::tt_metal::tensor_impl
```

---

## APIs to Keep in TTNN

These APIs are **framework-specific** and should remain in TTNN as integration layers.

### 1. PyTorch/NumPy Integration

**Location:** `ttnn/cpp/ttnn-pybind/pytensor.cpp` (stays in TTNN)

```cpp
namespace ttnn::tensor {

// Parse Python tensor objects (torch.Tensor or numpy.ndarray)
struct PreprocessedPyTensor {
    DataType data_type = DataType::INVALID;
    py::object contiguous_py_tensor;
    std::size_t num_elements = 0;
    std::size_t py_data_ptr = 0;
};

PreprocessedPyTensor parse_py_tensor(
    const py::handle& py_tensor,
    std::optional<DataType> optional_data_type);

// Create tt::tt_metal::Tensor from Python data
// This wraps Metal's Tensor::from_span with torch/numpy dtype conversion
Tensor create_tt_tensor_from_py_data(
    std::size_t py_data_ptr,
    const Shape& py_data_shape,
    const TensorLayout& tensor_layout,
    MeshDevice* device,
    const MemoryPin& pydata_pin,
    std::optional<QueueId> cq_id,
    float pad_value,
    const distributed::TensorToMesh* mesh_mapper);

// Convert tt::tt_metal::Tensor back to torch.Tensor
py::object tensor_to_py_object(const Tensor& tensor);

}  // namespace ttnn::tensor
```

### 2. High-Level Python API

**Location:** `ttnn/ttnn/operations/core.py` (stays in TTNN)

```python
def from_torch(
    tensor: torch.Tensor,
    dtype: Optional[ttnn.DataType] = None,
    *,
    spec: Optional[ttnn.TensorSpec] = None,
    tile: Optional[ttnn.Tile] = None,
    pad_value: Optional[float] = None,
    layout: Optional[ttnn.Layout] = None,
    device: Optional[ttnn.MeshDevice] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    mesh_mapper: Optional[ttnn.TensorToMesh] = None,
    cq_id: Optional[int] = None,
) -> ttnn.Tensor:
    """
    Converts torch.Tensor to ttnn.Tensor.

    Internally:
    1. Parses torch dtype → ttnn.DataType
    2. Extracts contiguous buffer pointer
    3. Creates MemoryPin to keep Python object alive
    4. Calls tt::tt_metal::Tensor::from_span (Metal API)
    """
    # Implementation calls C++ binding
    # which calls Metal's Tensor::from_span

def to_torch(
    tensor: ttnn.Tensor,
    *,
    torch_rank: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    mesh_composer: Optional[ttnn.MeshToTensor] = None,
    cq_id: int = 0,
) -> torch.Tensor:
    """
    Converts ttnn.Tensor to torch.Tensor.

    Internally:
    1. Calls tensor.to_host() if on device
    2. Extracts buffer from Metal tensor
    3. Creates torch.Tensor with borrowed buffer
    """
    # Implementation accesses Metal tensor's host buffer
```

### 3. Distributed Tensor Creation

**Location:** `ttnn/core/distributed/distributed_tensor.cpp` (stays in TTNN)

```cpp
namespace ttnn::distributed {

// Create distributed tensor across mesh
// This is TTNN-specific multi-device logic built on top of Metal tensors
template <typename T>
Tensor create_distributed_tensor(
    tt::stl::Span<const T> data,
    const Shape& shape,
    const MemoryPin& pin,
    const TensorLayout& layout,
    const TensorToMesh& mesh_mapper,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    std::optional<QueueId> cq_id,
    T pad_value);

}  // namespace ttnn::distributed
```

### 4. Neural Network Operations

**Location:** `ttnn/cpp/ttnn/operations/` (stays in TTNN)

All neural network operations stay in TTNN:
- Matrix multiplication (matmul, bmm)
- Convolution
- Attention mechanisms
- Activations (relu, gelu, softmax, etc.)
- Normalization (layernorm, groupnorm, etc.)
- Pooling operations
- Element-wise operations
- And 200+ other operations

These operate on `tt::tt_metal::Tensor` objects but implement high-level ML semantics.

---

## Architecture After Migration

### Layer Separation

```
┌─────────────────────────────────────────────────────────────┐
│                     User Applications                        │
│  (PyTorch models, JAX models, Custom C++ apps, etc.)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌──────────────────────┐          ┌──────────────────────┐
│       TTNN           │          │   Framework          │
│  (Neural Network)    │          │   Integration        │
│                      │          │   (Forge, etc.)      │
│ • from_torch()       │          │                      │
│ • to_torch()         │          │ • Custom bindings    │
│ • matmul()           │          │ • Direct Metal use   │
│ • conv2d()           │          │                      │
│ • 200+ ML ops        │          │                      │
│                      │          │                      │
│ Uses Metal Tensor ↓  │          │ Uses Metal Tensor ↓  │
└──────────┬───────────┘          └──────────┬───────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │        TT-METAL             │
            │   (Hardware Abstraction)    │
            │                             │
            │ • Tensor class              │
            │ • DataType, Layout, Shape   │
            │ • to_device(), to_host()    │
            │ • allocate_tensor_on_*()    │
            │ • memcpy(), write_tensor()  │
            │                             │
            │ • Buffer, MeshBuffer        │
            │ • Device, MeshDevice        │
            │ • Program, Kernel APIs      │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │      Hardware (Tensix)      │
            └─────────────────────────────┘
```

### Directory Structure After Migration

```
tt-metal/
├── tt_metal/
│   ├── api/
│   │   └── tt-metalium/
│   │       ├── buffer.hpp          # Already exists
│   │       ├── device.hpp          # Already exists
│   │       ├── mesh_buffer.hpp     # Already exists
│   │       ├── mesh_device.hpp     # Already exists
│   │       └── tensor/             # ✨ NEW
│   │           ├── tensor.hpp
│   │           ├── tensor_types.hpp
│   │           ├── tensor_spec.hpp
│   │           ├── tensor_ops.hpp
│   │           ├── tensor_impl.hpp
│   │           ├── shape.hpp
│   │           └── storage.hpp
│   │
│   └── impl/
│       └── tensor/                 # ✨ NEW
│           ├── tensor.cpp
│           ├── tensor_ops.cpp
│           ├── tensor_impl.cpp
│           ├── tensor_spec.cpp
│           ├── storage.cpp
│           ├── types.cpp
│           └── tensor_utils.cpp
│
└── ttnn/
    ├── api/
    │   └── ttnn/
    │       └── tensor/             # 🔄 BECOMES FORWARDING HEADERS
    │           └── tensor.hpp      # #include <tt-metalium/tensor/tensor.hpp>
    │
    ├── core/
    │   ├── distributed/
    │   │   └── distributed_tensor.cpp  # Multi-device logic
    │   └── (other core ttnn logic)
    │
    ├── cpp/
    │   └── ttnn-pybind/
    │       └── pytensor.cpp        # Framework integration
    │           # - parse_py_tensor()
    │           # - Calls Metal's Tensor::from_span()
    │
    └── ttnn/
        └── operations/
            ├── core.py             # Python API wrappers
            │   # - from_torch()
            │   # - to_torch()
            └── (200+ neural network operations)
```

### Include Path Changes

**Before Migration:**
```cpp
// TTNN code
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

// Uses tt::tt_metal::Tensor (already in metal namespace!)
```

**After Migration:**
```cpp
// Metal-only code
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

using namespace tt::tt_metal;
Tensor t = Tensor::from_vector(data, spec);

// TTNN code (backward compatible)
#include "ttnn/tensor/tensor.hpp"  // Forwarding header
// OR
#include <tt-metalium/tensor/tensor.hpp>  // Direct include

namespace ttnn {
    using Tensor = tt::tt_metal::Tensor;  // Already an alias
}
```

---

## Usage Examples

### Example 1: Pure Metal User (No PyTorch Dependency)

```cpp
// main.cpp - Pure C++ Metal application
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <vector>
#include <iostream>

using namespace tt::tt_metal;

int main() {
    // Initialize device
    auto device = distributed::MeshDevice::create_unit_mesh(0);

    // Load data from file (no PyTorch needed!)
    std::vector<float> input_data(1024);
    // ... load from binary file, compute, etc. ...
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create tensor specification
    Shape shape({32, 32});  // 32x32 matrix
    TensorLayout layout(
        DataType::BFLOAT16,
        Layout::TILE,
        MemoryConfig{
            TensorMemoryLayout::INTERLEAVED,
            BufferType::DRAM
        }
    );
    TensorSpec spec(shape, layout);

    // Create tensor from vector
    Tensor host_tensor = Tensor::from_vector(input_data, spec);
    std::cout << "Created host tensor: " << host_tensor.logical_shape() << std::endl;

    // Transfer to device
    Tensor device_tensor = host_tensor.to_device(device.get());
    std::cout << "Transferred to device" << std::endl;

    // Do some computation on device...
    // (using Metal programs/kernels)

    // Transfer back to host
    Tensor result_tensor = device_tensor.to_host(/*blocking=*/true);
    std::cout << "Transferred back to host" << std::endl;

    // Access result data
    const auto& host_buffer = result_tensor.host_storage().buffer();
    auto result_data = host_buffer.get<float>();

    std::cout << "First 5 results: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << result_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

**Compilation:**
```bash
# No TTNN dependency!
g++ main.cpp -ltt_metal -o my_app
```

### Example 2: TTNN User with PyTorch (No Changes)

```python
# Existing TTNN code continues to work unchanged!
import torch
import ttnn

# Create device
device = ttnn.open_device(device_id=0)

# Create tensor from PyTorch (high-level API)
torch_tensor = torch.randn(32, 32, dtype=torch.bfloat16)
ttnn_tensor = ttnn.from_torch(
    torch_tensor,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Do ML operations
output = ttnn.matmul(ttnn_tensor, ttnn_tensor)

# Convert back to PyTorch
result = ttnn.to_torch(output)
print(f"Result: {result}")

ttnn.close_device(device)
```

### Example 3: Custom Framework Integration

```cpp
// custom_framework_backend.cpp
// Framework integration (e.g., Forge, custom PyTorch backend)

#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "my_framework/tensor.hpp"

using namespace tt::tt_metal;

class TTMetalBackend {
private:
    std::shared_ptr<distributed::MeshDevice> device_;

public:
    TTMetalBackend(int device_id) {
        device_ = distributed::MeshDevice::create_unit_mesh(device_id);
    }

    // Convert framework tensor to Metal tensor
    Tensor convert_to_metal(const MyFramework::Tensor& fw_tensor) {
        // Extract raw data pointer from framework tensor
        void* data_ptr = fw_tensor.data_ptr();
        size_t num_elements = fw_tensor.numel();

        // Create Metal tensor spec from framework metadata
        Shape shape = convert_shape(fw_tensor.shape());
        DataType dtype = convert_dtype(fw_tensor.dtype());
        TensorLayout layout(dtype, Layout::ROW_MAJOR, MemoryConfig{});
        TensorSpec spec(shape, layout);

        // Create Metal tensor directly - no TTNN dependency!
        switch (fw_tensor.dtype()) {
            case MyFramework::DType::Float32:
                return Tensor::from_span(
                    tt::stl::Span<const float>(
                        static_cast<float*>(data_ptr),
                        num_elements
                    ),
                    spec,
                    device_.get()
                );
            case MyFramework::DType::BFloat16:
                return Tensor::from_span(
                    tt::stl::Span<const bfloat16>(
                        static_cast<bfloat16*>(data_ptr),
                        num_elements
                    ),
                    spec,
                    device_.get()
                );
            // ... other types
        }
    }

    // Convert Metal tensor back to framework tensor
    MyFramework::Tensor convert_from_metal(const Tensor& metal_tensor) {
        // Bring to host if needed
        Tensor host_tensor = metal_tensor.storage_type() == StorageType::HOST
            ? metal_tensor
            : metal_tensor.to_host();

        // Extract buffer and create framework tensor
        const auto& buffer = host_tensor.host_storage().buffer();

        return MyFramework::Tensor::from_buffer(
            buffer.data(),
            convert_shape_back(host_tensor.logical_shape()),
            convert_dtype_back(host_tensor.dtype())
        );
    }
};
```

### Example 4: Low-Level memcpy

```cpp
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/mesh_device.hpp>

using namespace tt::tt_metal;

void low_level_transfer_example() {
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = device->mesh_command_queue(0);

    // Allocate device tensor
    TensorSpec spec(Shape({1024}), TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR));
    Tensor device_tensor = allocate_tensor_on_device(spec, device.get());

    // Prepare host data
    std::vector<float> host_data(1024);
    for (size_t i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<float>(i);
    }

    // Low-level memcpy: host → device
    memcpy(cq, device_tensor, host_data.data());

    // Do computation...

    // Low-level memcpy: device → host
    std::vector<float> result(1024);
    memcpy(cq, result.data(), device_tensor, std::nullopt, /*blocking=*/true);
}
```

### Example 5: Zero-Copy with Borrowed Data

```cpp
#include <tt-metalium/tensor/tensor.hpp>
#include <sys/mman.h>
#include <fcntl.h>

using namespace tt::tt_metal;

Tensor create_file_backed_tensor(const char* filepath) {
    // Memory-map a file
    int fd = open(filepath, O_RDONLY);
    size_t file_size = 1024 * sizeof(float);

    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Create memory pin that will unmap on destruction
    MemoryPin memory_pin(std::shared_ptr<void>(
        mmap_addr,
        [file_size](void* addr) {
            munmap(addr, file_size);
        }
    ));

    // Create zero-copy tensor with borrowed data
    Tensor tensor = Tensor::from_borrowed_data(
        tt::stl::Span<float>(static_cast<float*>(mmap_addr), 1024),
        Shape({32, 32}),
        memory_pin
    );

    close(fd);
    return tensor;  // Safe: memory_pin keeps mmap alive
}
```

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Move core type definitions and create build infrastructure

**Tasks:**
1. Create new directory structure in `tt_metal/api/tt-metalium/tensor/`
2. Move `DataType`, `StorageType`, `Layout` enums
3. Move type conversion utilities
4. Create forwarding headers in `ttnn/api/ttnn/tensor/`
5. Update CMakeLists.txt for Metal library
6. Verify compilation of existing TTNN code

**Validation:**
- All existing tests pass
- No behavior changes
- TTNN code can include from either location

**Deliverables:**
```
tt_metal/api/tt-metalium/tensor/tensor_types.hpp  ✓
ttnn/api/ttnn/tensor/types.hpp → forwarding header ✓
Updated CMakeLists.txt ✓
```

### Phase 2: Shape and Storage (Weeks 3-4)

**Goal:** Move shape classes and storage variants

**Tasks:**
1. Move `Shape`, `Shape2D` classes
2. Move `HostStorage`, `DeviceStorage`, `Storage` variant
3. Update all internal includes
4. Create Metal implementation files
5. Update tensor_impl utilities that depend on shapes

**Validation:**
- Unit tests for Shape classes
- Unit tests for Storage variants
- Integration tests for TTNN

**Deliverables:**
```
tt_metal/api/tt-metalium/tensor/shape.hpp ✓
tt_metal/api/tt-metalium/tensor/storage.hpp ✓
tt_metal/impl/tensor/types.cpp ✓
tt_metal/impl/tensor/storage.cpp ✓
```

### Phase 3: TensorSpec and Layout (Weeks 5-6)

**Goal:** Move tensor specification infrastructure

**Tasks:**
1. Move `TensorSpec` class
2. Move `TensorLayout` class
3. Move `MemoryConfig` class
4. Move `Tile` struct
5. Move `PageConfig` variants
6. Update serialization code if needed

**Validation:**
- TensorSpec creation and modification tests
- Layout conversion tests
- Memory config tests

**Deliverables:**
```
tt_metal/api/tt-metalium/tensor/tensor_spec.hpp ✓
tt_metal/impl/tensor/tensor_spec.cpp ✓
```

### Phase 4: Core Tensor Class (Weeks 7-8)

**Goal:** Move the main Tensor class

**Tasks:**
1. Move `Tensor` class declaration
2. Move `TensorAttributes` class
3. Move constructors and basic methods
4. Move `from_span`, `from_vector`, `from_borrowed_data`
5. Update all includes in Metal codebase
6. Ensure no TTNN dependencies

**Validation:**
- Tensor creation tests (all factory methods)
- Tensor copying and assignment tests
- Memory management tests (deallocate, is_allocated)
- Accessor tests (shape, dtype, layout, etc.)

**Deliverables:**
```
tt_metal/api/tt-metalium/tensor/tensor.hpp ✓
tt_metal/api/tt-metalium/tensor/tensor_attributes.hpp ✓
tt_metal/impl/tensor/tensor.cpp ✓
```

### Phase 5: Host-Device Transfer (Weeks 9-10)

**Goal:** Move device transfer operations

**Tasks:**
1. Move `to_device()`, `to_host()` methods
2. Move `write_tensor()` function
3. Move all `memcpy()` overloads
4. Move `allocate_tensor_on_device/host()`
5. Move `create_device_tensor()`
6. Move tensor_impl utilities

**Validation:**
- Host to device transfer tests
- Device to host transfer tests
- memcpy tests (all variants)
- Allocation tests
- Performance benchmarks (no regression)

**Deliverables:**
```
tt_metal/api/tt-metalium/tensor/tensor_ops.hpp ✓
tt_metal/api/tt-metalium/tensor/tensor_impl.hpp ✓
tt_metal/impl/tensor/tensor_ops.cpp ✓
tt_metal/impl/tensor/tensor_impl.cpp ✓
tt_metal/impl/tensor/tensor_utils.cpp ✓
```

### Phase 6: TTNN Integration Layer (Weeks 11-12)

**Goal:** Ensure TTNN works seamlessly with migrated Metal tensors

**Tasks:**
1. Update TTNN forwarding headers
2. Update Python bindings to use Metal headers
3. Update `from_torch()` / `to_torch()` implementations
4. Update distributed tensor creation
5. Verify all TTNN operations work with Metal tensors

**Validation:**
- All TTNN Python tests pass
- All TTNN C++ tests pass
- PyTorch integration tests pass
- Model tests (run a few representative models)

**Deliverables:**
```
ttnn/api/ttnn/tensor/*.hpp → all forwarding headers ✓
ttnn/cpp/ttnn-pybind/pytensor.cpp → updated includes ✓
All tests passing ✓
```

### Phase 7: Documentation and Cleanup (Weeks 13-14)

**Goal:** Complete migration and update documentation

**Tasks:**
1. Update API documentation
2. Update tutorials and examples
3. Update METALIUM_GUIDE.md
4. Create migration guide for external users
5. Remove unused forwarding headers (optional)
6. Performance validation
7. Final testing across all platforms

**Validation:**
- Documentation review
- Example code compiles and runs
- Performance benchmarks pass
- All CI tests pass

**Deliverables:**
```
Updated documentation ✓
Migration guide ✓
Performance report ✓
```

---

## Timeline

**Total Duration:** 14 weeks (~3.5 months)

```
Week 1-2:   Foundation (types, build system)
Week 3-4:   Shape and Storage
Week 5-6:   TensorSpec and Layout
Week 7-8:   Core Tensor Class
Week 9-10:  Host-Device Transfer
Week 11-12: TTNN Integration
Week 13-14: Documentation and Cleanup
```

**Milestones:**
- **Week 2:** Basic types moved, all tests still pass
- **Week 4:** Storage layer moved, creating tensors works
- **Week 6:** Complete tensor specification in Metal
- **Week 8:** Metal tensors fully functional
- **Week 10:** Host-device transfer complete
- **Week 12:** TTNN integration complete
- **Week 14:** Migration complete, documented, validated

---

## Critical Considerations

### 1. Namespace Strategy

**Current State:**
- Types already in `tt::tt_metal` namespace
- Physical location in `ttnn/` directory
- TTNN uses `using Tensor = tt::tt_metal::Tensor;`

**Decision:** Keep `tt::tt_metal` namespace, only move physical location
- ✅ Minimal code changes
- ✅ Already correct semantic separation
- ✅ TTNN just needs forwarding headers

### 2. Build System Impact

**Considerations:**
- Metal library will become larger (more symbols)
- Need proper symbol visibility (`TT_METAL_API`)
- Link-time dependencies must be managed
- Shared library vs static library considerations

**Actions:**
```cmake
# tt_metal/CMakeLists.txt
target_sources(tt_metal PRIVATE
    impl/tensor/tensor.cpp
    impl/tensor/tensor_ops.cpp
    impl/tensor/tensor_impl.cpp
    impl/tensor/tensor_spec.cpp
    impl/tensor/storage.cpp
    impl/tensor/types.cpp
    impl/tensor/tensor_utils.cpp
)

target_include_directories(tt_metal PUBLIC
    api/tt-metalium
)
```

### 3. Python Bindings

**Current:** Python bindings in `ttnn/cpp/ttnn-pybind/pytensor.cpp`

**Options:**
1. **Keep in TTNN** (Recommended)
   - Python bindings stay in TTNN
   - They bind to Metal C++ APIs
   - Python users import `ttnn` module

2. **Move to Metal**
   - Create `tt_metal` Python module
   - Expose low-level tensor APIs
   - TTNN wraps Metal module

**Decision:** Option 1 - Keep Python bindings in TTNN
- ✅ No Python API changes for users
- ✅ TTNN remains high-level Python interface
- ✅ Metal stays pure C++

### 4. Template Instantiations

Many functions are templated on data types:
```cpp
template <typename T>
Tensor from_span(Span<const T>, ...);

template <typename T>
Tensor to_host(const Tensor&, ...);
```

**Issue:** Template definitions must be in headers OR explicitly instantiated

**Solution:** Explicit instantiation in Metal library
```cpp
// tensor.cpp
template Tensor Tensor::from_span<float>(...);
template Tensor Tensor::from_span<bfloat16>(...);
template Tensor Tensor::from_span<uint32_t>(...);
template Tensor Tensor::from_span<int32_t>(...);
template Tensor Tensor::from_span<uint16_t>(...);
template Tensor Tensor::from_span<uint8_t>(...);
```

### 5. Circular Dependencies

**Potential Issue:** TTNN operations depend on Tensor, but Tensor might use TTNN utilities

**Solution:** Strict layering
```
Metal Tensor → No TTNN dependencies ✓
TTNN Operations → Can use Metal Tensor ✓
TTNN Utilities → Built on top of Metal ✓
```

### 6. Distributed Tensor

**Current:** Distributed tensor logic in `ttnn/core/distributed/`

**Question:** Should this move to Metal?

**Decision:** Keep in TTNN
- Multi-device sharding strategies are high-level
- `TensorToMesh` mapping is application-specific
- Metal provides `MeshDevice` and `MeshBuffer` (already there)
- TTNN provides distributed tensor creation on top

### 7. Serialization

**Current:** Tensor serialization in `ttnn/core/tensor/serialization.cpp`

**Question:** Move to Metal or keep in TTNN?

**Decision:** Move to Metal
- Serialization is core tensor functionality
- Enables Metal-only apps to save/load tensors
- Uses flatbuffers (no TTNN dependency)

### 8. Backward Compatibility

**Critical:** Existing code must continue to work

**Strategy:**
```cpp
// Option A: Forwarding headers (temporary)
// ttnn/api/ttnn/tensor/tensor.hpp
#pragma once
#include <tt-metalium/tensor/tensor.hpp>

namespace ttnn {
    using Tensor = tt::tt_metal::Tensor;
    using DataType = tt::tt_metal::DataType;
    // ... other aliases
}

// Option B: Direct includes (after migration stabilizes)
// Users update includes:
// #include "ttnn/tensor/tensor.hpp" → #include <tt-metalium/tensor/tensor.hpp>
```

**Recommendation:** Keep forwarding headers permanently for TTNN
- TTNN users don't need to change code
- Clear API boundary: `ttnn::Tensor` vs `tt::tt_metal::Tensor`
- Both refer to same type

---

## Testing Strategy

### 1. Unit Tests

**Metal Tensor Tests** (new, in `tt_metal/test/tensor/`)
```cpp
TEST(MetalTensor, CreateFromVector) {
    std::vector<float> data(1024);
    TensorSpec spec(...);
    Tensor t = Tensor::from_vector(data, spec);
    ASSERT_EQ(t.logical_shape(), Shape({32, 32}));
}

TEST(MetalTensor, HostToDeviceTransfer) {
    auto device = MeshDevice::create_unit_mesh(0);
    Tensor host_tensor = ...;
    Tensor device_tensor = host_tensor.to_device(device.get());
    ASSERT_EQ(device_tensor.storage_type(), StorageType::DEVICE);
}

TEST(MetalTensor, DeviceToHostTransfer) {
    // ... test round-trip
}
```

**TTNN Integration Tests** (existing, updated)
- Verify TTNN operations work with Metal tensors
- Verify from_torch/to_torch still work
- Verify distributed tensor creation

### 2. Integration Tests

**End-to-End Tests:**
```python
def test_pytorch_integration():
    import torch
    import ttnn

    device = ttnn.open_device(0)
    torch_tensor = torch.randn(32, 32)
    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device)
    result = ttnn.to_torch(ttnn_tensor)
    assert torch.allclose(torch_tensor, result)
```

### 3. Performance Tests

**Benchmarks:**
- Tensor creation time
- Host to device transfer bandwidth
- Device to host transfer bandwidth
- Memory allocation overhead

**Validation:** No regression > 5%

### 4. Model Tests

**Representative Models:**
- Run ResNet-50 inference
- Run BERT-Base inference
- Run simple MLP training

**Validation:** Results match pre-migration

### 5. Compatibility Tests

**Build Tests:**
```bash
# Metal-only build (no TTNN)
cmake -DBUILD_METAL_ONLY=ON ..
make

# Full build with TTNN
cmake ..
make
```

### 6. Continuous Integration

**CI Pipeline:**
1. Build Metal library
2. Run Metal unit tests
3. Build TTNN (depends on Metal)
4. Run TTNN unit tests
5. Run integration tests
6. Run model tests
7. Performance benchmarks

---

## Success Criteria

### Functional Requirements
- ✅ All Metal tensor APIs functional and tested
- ✅ All TTNN tests pass without modification
- ✅ Python API unchanged for TTNN users
- ✅ Metal-only C++ applications can use tensors

### Non-Functional Requirements
- ✅ No performance regression > 5%
- ✅ Compilation time not increased significantly
- ✅ Binary size increase acceptable (< 10%)
- ✅ Memory usage unchanged

### Documentation Requirements
- ✅ Metal tensor API documented
- ✅ Migration guide for external users
- ✅ Updated examples and tutorials
- ✅ Updated architecture documentation

---

## Risks and Mitigation

### Risk 1: Breaking Existing Code
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Maintain forwarding headers
- Extensive testing before merge
- Staged rollout

### Risk 2: Performance Regression
**Likelihood:** Low
**Impact:** High
**Mitigation:**
- Benchmark at each phase
- Profile hot paths
- Optimize if needed

### Risk 3: Template Instantiation Issues
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Explicit instantiations in library
- Test all data type combinations
- Clear compiler errors

### Risk 4: Build System Complexity
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Incremental CMake changes
- Test on all platforms
- Clear dependency graph

### Risk 5: Timeline Slippage
**Likelihood:** Medium
**Impact:** Low
**Mitigation:**
- Buffer time in estimates
- Parallel work where possible
- Regular progress reviews

---

## Open Questions

1. **Should serialization move to Metal or stay in TTNN?**
   - Recommendation: Move to Metal (enables Metal-only save/load)

2. **Should distributed tensor utilities move?**
   - Recommendation: Keep in TTNN (high-level multi-device logic)

3. **Should we keep forwarding headers permanently?**
   - Recommendation: Yes (better API boundary)

4. **Should Python bindings expose Metal directly?**
   - Recommendation: No (TTNN remains Python interface)

5. **Should we version the Metal tensor API?**
   - Recommendation: Yes (semantic versioning for stability)

---

## Next Steps

1. **Review and approve this plan**
2. **Create detailed work breakdown for Phase 1**
3. **Set up tracking (GitHub project, Jira, etc.)**
4. **Assign ownership for each phase**
5. **Begin Phase 1: Foundation**

---

## References

- [METALIUM_GUIDE.md](METALIUM_GUIDE.md) - Metal programming guide
- [TT-NN README](ttnn/README.md) - TTNN overview
- [Tensor Layouts Tech Report](tech_reports/tensor_layouts/tensor_layouts.md)
- [Tensor Sharding Tech Report](tech_reports/tensor_sharding/)

---

## Appendix A: Complete File Mapping

### Files to Move

| Current Location | New Location | Status |
|-----------------|--------------|--------|
| `ttnn/api/ttnn/tensor/types.hpp` | `tt_metal/api/tt-metalium/tensor/tensor_types.hpp` | To Move |
| `ttnn/api/ttnn/tensor/shape/shape.hpp` | `tt_metal/api/tt-metalium/tensor/shape.hpp` | To Move |
| `ttnn/api/ttnn/tensor/storage.hpp` | `tt_metal/api/tt-metalium/tensor/storage.hpp` | To Move |
| `ttnn/api/ttnn/tensor/tensor_spec.hpp` | `tt_metal/api/tt-metalium/tensor/tensor_spec.hpp` | To Move |
| `ttnn/api/ttnn/tensor/layout/tensor_layout.hpp` | `tt_metal/api/tt-metalium/tensor/tensor_layout.hpp` | To Move |
| `ttnn/api/ttnn/tensor/memory_config/memory_config.hpp` | `tt_metal/api/tt-metalium/tensor/memory_config.hpp` | To Move |
| `ttnn/api/ttnn/tensor/tensor.hpp` | `tt_metal/api/tt-metalium/tensor/tensor.hpp` | To Move |
| `ttnn/api/ttnn/tensor/tensor_impl.hpp` | `tt_metal/api/tt-metalium/tensor/tensor_impl.hpp` | To Move |
| `ttnn/api/ttnn/tensor/tensor_attributes.hpp` | `tt_metal/api/tt-metalium/tensor/tensor_attributes.hpp` | To Move |
| `ttnn/core/tensor/types.cpp` | `tt_metal/impl/tensor/types.cpp` | To Move |
| `ttnn/core/tensor/storage.cpp` | `tt_metal/impl/tensor/storage.cpp` | To Move |
| `ttnn/core/tensor/tensor_spec.cpp` | `tt_metal/impl/tensor/tensor_spec.cpp` | To Move |
| `ttnn/core/tensor/tensor.cpp` | `tt_metal/impl/tensor/tensor.cpp` | To Move |
| `ttnn/core/tensor/tensor_impl.cpp` | `tt_metal/impl/tensor/tensor_impl.cpp` | To Move |
| `ttnn/core/tensor/tensor_ops.cpp` | `tt_metal/impl/tensor/tensor_ops.cpp` | To Move |
| `ttnn/core/tensor/tensor_utils.cpp` | `tt_metal/impl/tensor/tensor_utils.cpp` | To Move |
| `ttnn/core/tensor/serialization.cpp` | `tt_metal/impl/tensor/serialization.cpp` | To Move |

### Files to Keep in TTNN

| File | Reason |
|------|--------|
| `ttnn/cpp/ttnn-pybind/pytensor.cpp` | Framework integration (torch/numpy) |
| `ttnn/ttnn/operations/core.py` | Python convenience API |
| `ttnn/core/distributed/distributed_tensor.cpp` | High-level multi-device logic |
| `ttnn/cpp/ttnn/operations/**/*.cpp` | Neural network operations |

### Files to Create

| New File | Purpose |
|----------|---------|
| `tt_metal/api/tt-metalium/tensor/tensor_ops.hpp` | Device transfer and allocation APIs |
| `ttnn/api/ttnn/tensor/tensor.hpp` | Forwarding header (includes Metal) |
| Various forwarding headers in `ttnn/api/ttnn/tensor/` | Backward compatibility |

---

## Appendix B: Candidate Metal Tests for Migration

### Overview

Many existing Metal tests manually handle data packing, tilization, and buffer management - operations that Metal Tensor will handle automatically. Migrating these tests will:
1. **Validate the Metal Tensor API** - Tests become validation for the new APIs
2. **Demonstrate simplification** - Show how much cleaner code becomes
3. **Serve as examples** - Provide reference implementations for Metal-only users
4. **Reduce maintenance burden** - Less manual data manipulation code to maintain

### Test Migration Candidates

#### Category 1: Data Movement Tests

**`tests/tt_metal/tt_metal/test_datacopy.cpp`**
- **Current pattern:** Manual buffer creation, random data generation, WriteToBuffer/ReadFromBuffer
- **Lines:** ~180
- **What it does:** Tests basic DRAM→L1→DRAM data copy with backpressure
- **Simplification opportunity:** HIGH ⭐⭐⭐
- **Current complexity:**
  ```cpp
  // Manual buffer setup
  uint32_t dram_buffer_size = single_tile_size * num_tiles;
  auto src_dram_buffer = CreateBuffer(dram_config);

  // Manual data generation and packing
  std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
      dram_buffer_size, 100, seed);
  tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

  // Manual result unpacking
  std::vector<uint32_t> result_vec;
  tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
  ```

**`tests/tt_metal/tt_metal/data_movement/interleaved/test_interleaved.cpp`**
- **Current pattern:** Manual interleaved buffer setup, TensorAccessorArgs, packed data generation
- **Lines:** ~750
- **What it does:** Tests interleaved buffer read/write patterns across cores
- **Simplification opportunity:** HIGH ⭐⭐⭐
- **Current complexity:**
  ```cpp
  // Manual buffer creation
  InterleavedBufferConfig config{...};
  auto input_buffer = CreateBuffer(config);

  // Manual packed data generation
  vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
      -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), seed);

  // Manual TensorAccessorArgs
  TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
  ```

**`tests/tt_metal/tt_metal/data_movement/multi_interleaved/test_multi_interleaved.cpp`**
- **Current pattern:** Similar to above but with multiple buffers
- **Lines:** ~850
- **Simplification opportunity:** HIGH ⭐⭐⭐

**`tests/tt_metal/tt_metal/data_movement/loopback/test_loopback.cpp`**
- **Current pattern:** Manual CB setup, buffer management
- **Lines:** ~300
- **Simplification opportunity:** MEDIUM ⭐⭐

#### Category 2: Matmul Tests (Manual Tilization)

**`tests/tt_metal/tt_metal/test_matmul_single_tile.cpp`**
- **Current pattern:** Manual tilization, packing, identity matrix creation
- **Lines:** ~180
- **What it does:** Tests single tile matrix multiplication
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐
- **Current complexity:**
  ```cpp
  // Manual tensor creation and initialization
  SHAPE shape = {1, 1, 32, 32};
  tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
      shape, tt::deprecated::Initialize::RANDOM, 0, 100, seed);

  // Manual tilization
  auto activations_tile_layout = convert_to_tile_layout(
      tt::stl::make_const_span(tensor.get_values()));

  // Manual packing
  auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
  tt_metal::detail::WriteToBuffer(src0_dram_buffer, activations);

  // Manual unpacking and detilization
  auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
  auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(
      tt::stl::make_const_span(result_bfp16));
  ```

**`tests/tt_metal/tt_metal/test_matmul_multi_tile.cpp`**
- **Similar pattern, more tiles**
- **Lines:** ~250
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐

**`tests/tt_metal/tt_metal/test_matmul_single_core.cpp`**
- **Similar pattern with multiple tiles**
- **Lines:** ~300
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐

**`tests/tt_metal/tt_metal/test_matmul_multi_core_*.cpp` (6 variants)**
- **Pattern:** Manual buffer management, tilization, multicore coordination
- **Lines:** 300-500 each
- **Simplification opportunity:** HIGH ⭐⭐⭐

#### Category 3: Format Conversion Tests

**`tests/tt_metal/tt_metal/test_bfp8_conversion.cpp`**
- **Current pattern:** Manual BFP8 packing/unpacking, tilization
- **Lines:** ~150
- **What it does:** Tests BFP8 data format conversion
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐
- **Current complexity:**
  ```cpp
  // Manual layout conversion
  std::vector<float> tiled_fp32_vec = convert_layout(
      tt::stl::make_const_span(fp32_vec),
      shape_vec,
      TensorLayoutType::LIN_ROW_MAJOR,
      TensorLayoutType::TILED_NFACES);

  // Manual BFP8 packing
  std::vector<uint32_t> packed_bfp8b_tile_vec = pack_as_bfp8_tiles(
      tt::stl::make_const_span(tiled_fp32_vec),
      /*row_major_input=*/false,
      /*is_exp_a=*/false);

  // Manual unpacking
  std::vector<float> unpacked = unpack_bfp8_tiles_into_float_vec(
      packed_bfp8b_tile_vec,
      /*row_major_output=*/false,
      /*is_exp_a=*/false);
  ```

**`tests/tt_metal/tt_metal/test_bfp4_conversion.cpp`**
- **Similar pattern for BFP4**
- **Lines:** ~150
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐

**`tests/tt_metal/tt_metal/api/test_tilize_untilize.cpp`**
- **Current pattern:** Manual tilization/untilization with face transpose
- **Lines:** ~450
- **What it does:** Tests various tile layouts and face configurations
- **Simplification opportunity:** VERY HIGH ⭐⭐⭐⭐

#### Category 4: Specialized Tests

**`tests/tt_metal/tt_metal/test_untilize_eltwise_binary.cpp`**
- **Pattern:** Manual tilization + eltwise operations
- **Lines:** ~300
- **Simplification opportunity:** HIGH ⭐⭐⭐

**`tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/test_compute_mm.cpp`**
- **Pattern:** Performance testing with manual buffer management
- **Lines:** ~1700
- **Simplification opportunity:** MEDIUM ⭐⭐
- **Note:** Keep performance-critical parts, but simplify data setup

### Migration Priority

#### Phase 1 (Quick Wins - Weeks 1-2 after Tensor is in Metal)
1. ✅ **test_bfp8_conversion.cpp** - Clear before/after, validates data type conversions
2. ✅ **test_bfp4_conversion.cpp** - Similar to above
3. ✅ **test_matmul_single_tile.cpp** - Classic example, widely understood
4. ✅ **test_datacopy.cpp** - Simple data movement pattern

#### Phase 2 (Core Validation - Weeks 3-4)
5. ✅ **test_interleaved.cpp** - Validates interleaved buffer patterns
6. ✅ **test_tilize_untilize.cpp** - Comprehensive tilization testing
7. ✅ **test_matmul_multi_tile.cpp** - Multi-tile patterns
8. ✅ **test_matmul_single_core.cpp** - Single core matmul with blocking

#### Phase 3 (Advanced - Weeks 5-6)
9. ✅ **test_matmul_multi_core_*.cpp** (all variants) - Multi-core coordination
10. ✅ **test_multi_interleaved.cpp** - Complex buffer patterns
11. ✅ **test_untilize_eltwise_binary.cpp** - Combined operations

### Before/After Examples

#### Example 1: Simple Data Copy (test_datacopy.cpp)

**Before (Current Metal Test):**
```cpp
// ~50 lines just for data setup
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

int main() {
    auto device = tt_metal::CreateDevice(0);
    tt_metal::Program program = tt_metal::CreateProgram();

    // Manual buffer creation
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    auto src_buffer = CreateBuffer(dram_config);
    auto dst_buffer = CreateBuffer(dram_config);

    // Manual data generation (creates packed bfloat16 data)
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100,
        std::chrono::system_clock::now().time_since_epoch().count());

    // Manual write
    tt_metal::detail::WriteToBuffer(src_buffer, src_vec);

    // Set up kernels, run program...
    tt_metal::SetRuntimeArgs(program, reader_kernel, core,
        {src_buffer->address(), 0, num_tiles});
    tt_metal::detail::LaunchProgram(device, program);

    // Manual read back
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_buffer, result_vec);

    // Manual comparison (packed data)
    bool pass = (src_vec == result_vec);

    tt_metal::CloseDevice(device);
    return pass ? 0 : 1;
}
```

**After (With Metal Tensor):**
```cpp
// ~20 lines - much cleaner!
#include <tt-metalium/tensor/tensor.hpp>
#include <tt-metalium/device.hpp>

int main() {
    using namespace tt::tt_metal;

    auto device = MeshDevice::create_unit_mesh(0);
    Program program = CreateProgram();

    // Create tensor with random data (handles packing internally)
    Shape shape({2048, 32, 32});  // 2048 tiles
    TensorSpec spec(shape, TensorLayout(
        DataType::BFLOAT16,
        Layout::TILE,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}
    ));

    // Generate random data
    std::vector<float> random_data(shape.volume());
    std::generate(random_data.begin(), random_data.end(),
        []() { return static_cast<float>(rand()) / RAND_MAX * 100.0f; });

    // Create tensors - automatic packing, tilization, device transfer
    Tensor src_tensor = Tensor::from_vector(random_data, spec)
        .to_device(device.get());
    Tensor dst_tensor = allocate_tensor_on_device(spec, device.get());

    // Set up kernels (now use tensor's buffer directly)
    auto src_buffer = src_tensor.mesh_buffer();
    SetRuntimeArgs(program, reader_kernel, core,
        {src_buffer->address(), 0, 2048});

    // Run program
    EnqueueMeshWorkload(device->mesh_command_queue(),
        create_workload(program), /*blocking=*/true);

    // Read back (automatic unpacking)
    Tensor result_tensor = dst_tensor.to_host(/*blocking=*/true);

    // Compare (works on unpacked data automatically)
    bool pass = tensors_equal(src_tensor.to_host(), result_tensor);

    return pass ? 0 : 1;
}
```

**Benefits:**
- ✅ 60% less code
- ✅ No manual packing/unpacking
- ✅ No manual tilization
- ✅ Clearer intent
- ✅ Type-safe operations
- ✅ Automatic memory management

#### Example 2: Matmul with Tilization (test_matmul_single_tile.cpp)

**Before (Current):**
```cpp
// ~30 lines of manual data manipulation
#include <tt-metalium/bfloat16.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/tilize_utils.hpp>

// Create input tensor using deprecated utility
SHAPE shape = {1, 1, 32, 32};
tt::deprecated::Tensor<bfloat16> tensor =
    tt::deprecated::initialize_tensor<bfloat16>(
        shape,
        tt::deprecated::Initialize::RANDOM,
        0, 100, seed);

// Manual tilization (converts from row-major to tile layout)
auto activations_tile_layout = convert_to_tile_layout(
    tt::stl::make_const_span(tensor.get_values()));

// Manual packing (bfloat16 → uint32)
auto activations = pack_bfloat16_vec_into_uint32_vec(
    activations_tile_layout);

// Write to buffer
tt_metal::detail::WriteToBuffer(src0_dram_buffer, activations);

// Create identity matrix manually
auto identity = create_identity_matrix(32, 32, 32);
auto weights_tile_layout = convert_to_tile_layout(
    tt::stl::make_const_span(identity));
auto weights = pack_bfloat16_vec_into_uint32_vec(
    weights_tile_layout);
tt_metal::detail::WriteToBuffer(src1_dram_buffer, weights);

// After kernel execution...
std::vector<uint32_t> result_vec;
tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

// Manual unpacking
auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

// Manual untilization
auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(
    tt::stl::make_const_span(result_bfp16));

// Finally compare
bool pass = (tensor.get_values() == result_flat_layout);
```

**After (With Metal Tensor):**
```cpp
// ~10 lines - crystal clear!
using namespace tt::tt_metal;

// Create input tensor - automatic tilization
Shape shape({32, 32});
TensorSpec spec(shape, TensorLayout(
    DataType::BFLOAT16, Layout::TILE, memory_config));

std::vector<float> random_data(1024);
std::generate(random_data.begin(), random_data.end(),
    []() { return rand() % 100; });

Tensor input = Tensor::from_vector(random_data, spec)
    .to_device(device.get());

// Create identity matrix
std::vector<float> identity_data(1024, 0.0f);
for (int i = 0; i < 32; ++i) identity_data[i * 32 + i] = 1.0f;

Tensor weights = Tensor::from_vector(identity_data, spec)
    .to_device(device.get());

// After matmul kernel execution...
Tensor result = output_tensor.to_host(/*blocking=*/true);

// Extract and compare (automatic untilization)
auto result_vec = result.host_buffer().get<float>();
bool pass = std::equal(random_data.begin(), random_data.end(),
                       result_vec.begin());
```

**Benefits:**
- ✅ 70% less code
- ✅ No manual tilization/untilization
- ✅ No manual packing/unpacking
- ✅ All layout conversions automatic
- ✅ Works with natural data types (float) instead of packed uint32

#### Example 3: BFP8 Conversion (test_bfp8_conversion.cpp)

**Before (Current):**
```cpp
// Manual layout conversion
std::vector<uint32_t> shape_vec = {1, 1, 32, 32};
std::vector<float> tiled_fp32_vec = convert_layout(
    tt::stl::make_const_span(fp32_vec),
    shape_vec,
    TensorLayoutType::LIN_ROW_MAJOR,
    TensorLayoutType::TILED_NFACES);

// Manual BFP8 packing
std::vector<uint32_t> packed_bfp8b = pack_as_bfp8_tiles(
    tt::stl::make_const_span(tiled_fp32_vec),
    /*row_major_input=*/false,
    /*is_exp_a=*/false);

// Manual unpacking
std::vector<float> unpacked_bfp8b = unpack_bfp8_tiles_into_float_vec(
    packed_bfp8b,
    /*row_major_output=*/false,
    /*is_exp_a=*/false);

// Compare
bool pass = compare_vectors(fp32_vec, unpacked_bfp8b, tolerance);
```

**After (With Metal Tensor):**
```cpp
// Tensor handles everything
Shape shape({32, 32});

// Create FP32 tensor
TensorSpec fp32_spec(shape, TensorLayout(
    DataType::FLOAT32, Layout::ROW_MAJOR, mem_config));
Tensor fp32_tensor = Tensor::from_vector(fp32_vec, fp32_spec);

// Convert to BFP8 with automatic layout conversion
TensorSpec bfp8_spec(shape, TensorLayout(
    DataType::BFLOAT8_B, Layout::TILE, mem_config));
Tensor bfp8_tensor = Tensor::from_vector(fp32_vec, bfp8_spec);

// Convert back - automatic unpacking and untilization
Tensor result = bfp8_tensor.to_host();
auto result_vec = result.host_buffer().get<float>();

// Compare
bool pass = compare_vectors(fp32_vec, result_vec, tolerance);
```

**Benefits:**
- ✅ 80% less code
- ✅ No manual BFP8 packing knowledge needed
- ✅ Automatic layout conversions
- ✅ Data type conversions handled internally
- ✅ Much harder to make mistakes

### Code Reduction Statistics

| Test File | Current Lines | Estimated After | Reduction |
|-----------|--------------|-----------------|-----------|
| test_datacopy.cpp | 180 | 70 | 61% |
| test_matmul_single_tile.cpp | 180 | 65 | 64% |
| test_bfp8_conversion.cpp | 150 | 45 | 70% |
| test_bfp4_conversion.cpp | 150 | 45 | 70% |
| test_interleaved.cpp | 750 | 300 | 60% |
| test_tilize_untilize.cpp | 450 | 150 | 67% |
| **Total for these 6** | **1,860** | **675** | **64%** |

### Additional Benefits

1. **Better Error Messages**
   - Before: "Buffer size mismatch" (cryptic)
   - After: "Shape [32, 32] requires 1024 elements, got 512" (clear)

2. **Type Safety**
   - Before: Everything is `vector<uint32_t>` (easy to mix up)
   - After: Strongly typed `Tensor` with `DataType` (compiler catches errors)

3. **Easier Debugging**
   - Before: Packed uint32 data hard to inspect
   - After: Tensors can be printed with logical values

4. **Better Documentation**
   - Tests become documentation showing how to use Metal Tensor
   - New Metal users have clear examples

5. **Maintenance**
   - Changes to internal packing format? Update Tensor, not 50 tests
   - New data type? Add to Tensor, existing tests work

### Migration Guidelines

For each test to migrate:

1. **Identify manual operations:**
   - `convert_to_tile_layout` / `convert_layout` → Automatic with Layout::TILE
   - `pack_*_into_uint32_vec` → Automatic in Tensor
   - `unpack_uint32_vec` → Automatic with `to_host()`
   - `create_random_vector_of_bfloat16` → `Tensor::from_vector` with random data
   - `WriteToBuffer` / `ReadFromBuffer` → `to_device()` / `to_host()`

2. **Replace with Tensor API:**
   ```cpp
   // Old pattern
   auto data = create_random_vector_of_bfloat16(size, max, seed);
   auto tiled = convert_to_tile_layout(data);
   auto packed = pack_bfloat16_vec_into_uint32_vec(tiled);
   WriteToBuffer(buffer, packed);

   // New pattern
   auto tensor = Tensor::from_vector(generate_random<float>(size), spec)
       .to_device(device.get());
   ```

3. **Update kernel argument passing:**
   ```cpp
   // Old: Pass buffer address
   SetRuntimeArgs(program, kernel, core, {buffer->address(), ...});

   // New: Still pass buffer (from tensor)
   SetRuntimeArgs(program, kernel, core,
       {tensor.mesh_buffer()->address(), ...});
   ```

4. **Validate same behavior:**
   - Run both versions
   - Compare results bit-for-bit
   - Ensure performance unchanged

---

**Document Version:** 1.0
**Last Updated:** 2025-01-XX
**Authors:** [Your Team]
**Status:** Proposal / Draft
