# Sharded Accessor Guide

## Overview

The [ShardedAccessor](../../tt_metal/hw/inc/accessor/sharded_accessor.h) is a utility for efficiently accessing sharded tensors distributed across multiple memory banks. It provides an abstraction that handles the mapping from logical tensor indices to physical memory locations.

The main thing to keep in mind when working with it is that developer can choose which arguments of accessor are passed through compile time arguments, which through common-runtime arguments.
Parameters consist of [rank, number of banks, tensor shape, shard shape, banks coordinates]


## Host-Side Setup

```c++
// Get accessor arguments based on buffer distribution specification
using tt::tt_metal::sharded_accessor_utils::ArgConfig;
const auto& buffer_distribution_spec =
    std::get<BufferDistributionSpec>(mesh_buffer->device_local_config().shard_parameters.value());

// Choose which parts are compile-time vs runtime
// Options include: CTA (all compile-time), RankCRTA, NumBanksCRTA, TensorShapeCRTA, ShardShapeCRTA, BankCoordsCRTA
const auto accessor_args = tt::tt_metal::sharded_accessor_utils::get_sharded_accessor_args(
    *mesh_device, buffer_distribution_spec, shard_view->core_type(), ArgConfig::NumBanksCRTA | ArgConfig::BankCoordsCRTA); // Number of banks and bank coordinates passed through crta, rest - cta

// Setting up device kernel with these arguments
KernelHandle kernel_id = CreateKernel(
    program,
    "path/to/kernel.cpp",
    grid,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = accessor_args.compile_time_args
    });

// Pass any runtime arguments to the kernel
SetCommonRuntimeArgs(program, kernel_id, accessor_args.runtime_args);
```

### Configuration Options
You can configure which parts of the accessor arguments are passed through CRTA or CTA:

- ArgConfig::CTA: All arguments passed through cta
- ArgConfig::RankCRTA
- ArgConfig::NumBanksCRTA
- ArgConfig::TensorShapeCRTA
- ArgConfig::ShardShapeCRTA
- ArgConfig::BankCoordsCRTA
- ArgConcig::CRTA: All arguments passed through crta

These flags can be combined with bitwise OR (|) to specify multiple runtime parameters.

There is one important limitation: In case size of container (rank/num_banks) is crta, then values of containers (tensor_shape/shard_shape/bank_coords) must also be crta. The reason is that all CTA indecies must be constexpr expressions, and it's impossible to calculate offset for shapes without knowing their sizes.

## Device-Side Usage
Creating an Accessor

```c++

// Base offsets of cta and crta arguments
constexpr uint32_t base_idx_cta = 0;
constexpr uint32_t base_idx_crta = 1;

// Get DistributionSpec type.
using dspec_t = nd_sharding::distribution_spec_t<base_idx_cta, base_idx_crta>;

constexpr uint32_t new_base_idx_cta = base_idx_cta + nd_sharding::compile_time_args_skip<dspec_t>();
// new_base_idx_crta might be constexpr if rank and number of banks are static
uint32_t new_base_idx_crta = base_idx_crta + nd_sharding::runtime_args_skip<dspec_t>();

// Create a ShardedAccessor with compile time page size
auto sharded_accessor = nd_sharding::ShardedAccessor<dspec_t, page_size>(bank_base_address);

// Create a ShardedAccessor with runtime page size
auto sharded_accessor = nd_sharding::ShardedAccessor<dspec_t>(bank_base_address, page_size);
```

**Key Operations**

Address Calculation

```c++
// Get the NOC address for a given page
uint32_t noc_addr = sharded_accessor.get_noc_addr(page_id);

// Get bank ID and offset for a given page
auto [bank_id, bank_offset] = sharded_accessor.get_bank_and_offset(page_id);
```

Data Transfer

```c++
// read a page from memory
uint32_t l1_write_addr = get_write_ptr(cb_id);  // Address to write to in L1 memory
sharded_accessor.noc_async_read_page(page_id, l1_write_addr);
noc_async_read_barrier();  // Wait for read to complete

// write a page to memory
uint32_t l1_read_addr = get_read_ptr(cb_id);  // Address to read from in L1 memory
sharded_accessor.noc_async_write_page(page_id, l1_read_addr);
```

Distribution Spec Information

```c++
// Access information about the tensor / shard / banks
const auto& dspec = sharded_accessor.get_dspec();

auto rank = get_rank();
auto num_banks = dspec.get_num_banks();

// Note: all volumes, shapes and strides are in pages!!!
auto tensor_volume = dspec.get_tensor_volume();
const auto& tensor_shape = dspec.get_tensor_shape();
const auto& tensor_strides = dspec.get_tensor_strides();

auto shard_volume = dspec.get_shard_volume();
const auto& shard_shape = dspec.get_shard_shape();
const auto& shard_strides = dspec.get_shard_strides();

const auto& packed_xy_coords = dspec.get_packed_xy_coords();
```

Note: In case containers size is CTA, then shapes, strides, coords are `std::array<uint32_t, rank/num_banks>`, otherwide `Span<uint32_t>`


## Performance Considerations
- If rank is static, then construction of ShardedAccessor is 0-cost, meaning that everything is precomputed in compile time.
- Calculation of address scales ~lineary with number rank


## Examples:
- Reshard [reader](../../tests/ttnn/unit_tests/gtests/accessor/kernels/reader_reshard.cpp), [writer](../../tests/ttnn/unit_tests/gtests/accessor/kernels/writer_reshard.cpp)
