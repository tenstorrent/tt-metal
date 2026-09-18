# Tensor Accessor Guide

## Overview

The [TensorAccessor](../../tt_metal/hw/inc/api/tensor/tensor_accessor.h) is a utility for efficiently accessing all tensors distributed across multiple memory banks. It provides an abstraction that handles the mapping from logical tensor indices to physical memory locations.

The main thing to keep in mind when working with it is that developer can choose which arguments of accessor are passed through compile time arguments, which through common-runtime arguments.
Parameters may consist of rank, number of banks, tensor shape, shard shape, banks coordinates, etc.


## Host-Side Setup

```c++
const auto accessor_args = TensorAccessorArgs(buffer);
// You can choose which parts are compile-time vs runtime
// Options include: None (all compile-time), RuntimeRank, RuntimeNumBanks, RuntimeTensorShape, RuntimeShardShape, RuntimeBankCoords
const auto accessor_args = TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeNumBanks | tensor_accessor::ArgConfig::RuntimeBankCoords);

// Setting up device kernel with these arguments
KernelHandle kernel_id = CreateKernel(
    program,
    "path/to/kernel.cpp",
    grid,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = accessor_args.get_compile_time_args()
    });

// Pass any runtime arguments to the kernel
SetCommonRuntimeArgs(program, kernel_id, accessor_args.get_common_runtime_args());
```

### Configuration Options
You can configure which parts of the accessor arguments are passed through runtime or compile-time arguments:

- ArgConfig::None: All arguments passed through compile-time arguments
- ArgConfig::RuntimeRank
- ArgConfig::RuntimeNumBanks
- ArgConfig::RuntimeTensorShape
- ArgConfig::RuntimeShardShape
- ArgConfig::RuntimeBankCoords
- ArgConfig::Runtime: All arguments passed through runtime arguments

These flags can be combined with bitwise OR (|) to specify multiple runtime parameters.

There is one important limitation: In case size of container (rank/num_banks) is runtime argument, then values of containers (tensor_shape/shard_shape/bank_coords) must also be runtime arguments. The reason is that all compile-time indices must be constexpr expressions, and it's impossible to calculate offset for shapes without knowing their sizes.

## Device-Side Usage
**Creating an Accessor**

- From compile-time/runtime arguments

```c++

// Base offsets of compile-time and common runtime arguments
constexpr uint32_t base_idx_cta = 0;
constexpr uint32_t base_idx_crta = 1;

// This object keeps track of the location of arguments for the tensor accessor
auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
// runtime base index can be a runtime variable too:
auto args = TensorAccessorArgs<base_idx_cta>(base_idx_crta);

constexpr uint32_t new_base_idx_cta = args.next_compile_time_args_offset();
// new_base_idx_crta might be constexpr if rank and number of banks are static
uint32_t new_base_idx_crta = args.next_common_runtime_args_offset();

// Option 1 (preferred, less error-prone): Using default tensor buffer's aligned_page_size to create a TensorAccessor
auto tensor_accessor = TensorAccessor(args, bank_base_address);
// Option 2: Create a TensorAccessor with custom runtime page size passed in as the 3rd argument. Used when we need to override the compile-time aligned_page_size if we have a program cache hit on a buffer with a different aligned_page_size.
auto tensor_accessor = TensorAccessor(args, bank_base_address, page_size);
```

- Manual arguments
Sometimes you might need more control. For example you want to reuse same bank coordinates between different accessors. In such case you can manually create DistributionSpec from rank, number of banks, tensor/shard shape and bank coordinates:

```c++
using tensor_shape = tensor_accessor::ArrayStaticWrapper<10, 10>;
using shard_shape = tensor_accessor::ArrayStaticWrapper<3, 3>;
// Each number in the bank coordinates represent the coordinates of two banks (x0, y0, x1, y1) compressed into a single uint32_t
using banks_coords = tensor_accessor::ArrayStaticWrapper<1179666, 1245202, 1310738, 1376274, 1179667, 1245203, 1310739, 1376275, 1179668, 1245204, 1310740, 1376276, 1179669, 1245205, 1310741, 1376277>;
auto dspec = tensor_accessor::DistributionSpec<2, 16, tensor_shape, shard_shape, banks_coords>();
auto tensor_accessor = TensorAccessor(std::move(dspec), 0, 1024);

// You can also mix constexpr/runtime values:
uint32_t tensor_shape[2] = {10, 10};
uint32_t shard_shape[2] = {3, 3};
using dyn = tensor_accessor::ArrayDynamicWrapper;
// Each number in the bank coordinates represent the coordinates of two banks (x0, y0, x1, y1) compressed into a single uint32_t
using banks_coords = tensor_accessor::ArrayStaticWrapper<1179666, 1245202, 1310738, 1376274, 1179667, 1245203, 1310739, 1376275, 1179668, 1245204, 1310740, 1376276, 1179669, 1245205, 1310741, 1376277>;
auto dspec = tensor_accessor::DistributionSpec<0, 16, dyn, dyn, banks_coords>(2, 0, tensor_shape, shard_shape, nullptr);
auto tensor_accessor = TensorAccessor(std::move(dspec), 0, 1024);

```

**Key Operations**

Address Calculation

```c++
// Get the NOC address for a given page
uint64_t noc_addr = tensor_accessor.get_noc_addr(page_id);

// Get bank ID and offset for a given page
auto [bank_id, bank_offset] = tensor_accessor.get_bank_and_offset(page_id);

// You can also address pages by nd coordinate (such address calculation is a little bit cheaper)
std::array<uint32_t, 4> page_coord{0, 1, 2, 3};
uint64_t noc_addr = tensor_accessor.get_noc_addr(page_coord);   // <- Anything with operator[] should work

// For sharded tensor, you can get address of shards:
static_assert(args::is_sharded, "Sharded API requires sharded tensor");
uint64_t noc_addr = tensor_accessor.get_shard_noc_addr(shard_id);

std::array<uint32_t, 4> shard_coord{0, 1, 2, 3};
uint64_t noc_addr = tensor_accessor.get_shard_noc_addr(shard_coord); // <- Anything with operator[] should work
```

Data Transfer

```c++
// read a page from memory
uint32_t l1_write_addr = get_write_ptr(cb_id);  // Address to write to in L1 memory
auto noc_addr = tensor_accessor.get_noc_addr(page_id);
noc_async_read(noc_addr, l1_write_addr, page_size);
// Or something like that:
noc_async_read_page(page_id, tensor_accessor, l1_write_addr);
noc_async_read_barrier();  // Wait for read to complete

// write a page to memory
uint32_t l1_read_addr = get_read_ptr(cb_id);  // Address to read from in L1 memory
auto noc_addr = tensor_accessor.get_noc_addr(page_id);
noc_async_write(l1_read_addr, noc_addr, page_size);
// Or something like that:
noc_async_write_page(page_id, tensor_accessor, l1_read_addr);
noc_async_write_barrier();

// Similarly, for sharded tensor, you can read/write the whole shard
auto shard_noc_addr = tensor_accessor.get_shard_noc_addr(shard_id);
noc_async_read_shard(shard_id, tensor_accessor, l1_write_addr);
noc_async_write_shard(shard_id, tensor_accessor, l1_read_addr);
```

Distribution Spec Information

```c++
// Access information about the tensor / shard / banks (only available for sharded tensors)
static_assert(args::is_sharded, "Dspec is only available for sharded tensors");
const auto& dspec = tensor_accessor.dspec();

auto rank = dspec.rank();
auto num_banks = dspec.num_banks();

// Note: all volumes, shapes and strides are in pages!!!
auto tensor_volume = dspec.tensor_volume();
const auto& tensor_shape = dspec.tensor_shape();
const auto& tensor_strides = dspec.tensor_strides();

auto shard_volume = dspec.shard_volume();
const auto& shard_shape = dspec.shard_shape();
const auto& shard_strides = dspec.shard_strides();

// Note: x=(packed >> 8) & 0xFF, y=packed & 0xFF
const auto& packed_xy_coords = dspec.packed_xy_coords();

// You can fetch information about locality of data for sharded tensor
static_assert(args::is_sharded, "Sharded API requires sharded tensor");
bool is_local = tensor_accessor.is_local_bank(virtual_x, virtual_y);
bool is_local = tensor_accessor.is_local_addr(noc_addr);
bool is_local = tensor_accessor.is_local_page(page_id);
bool is_local = tensor_accessor.is_local_shard(shard_id);
```

Note: In case containers size is compile-time, then shapes, strides, coords are `std::array<uint32_t, rank/num_banks>`, otherwise `Span<uint32_t>`

### Contiguous pages

`num_contiguous_pages` reports how many pages, starting at a given page, form one unbroken stretch of memory. Use it to replace a run of per-page transfers with a single large one.

```c++
// Sharded: end_page_id defaults to the tensor volume
uint32_t pages = tensor_accessor.num_contiguous_pages(page_id);
// Interleaved: end_page_id is required, the accessor carries no shape
uint32_t pages = tensor_accessor.num_contiguous_pages(page_id, total_pages);

// Those are the page ids { page_id + k * contiguous_page_stride() } for k in [0, pages),
// living at { get_noc_addr(page_id) + k * aligned_page_size }.
noc_async_read(tensor_accessor.get_noc_addr(page_id), l1_write_addr, pages * aligned_page_size);
```

`contiguous_page_stride()` is the page-id step between those pages. It is a property of the accessor, not of the page, so a caller reads it once. It is `1` for most sharded tensors, `tensor_shape[-1]` for a one-page-wide shard (what row-major width- and block-sharded tensors land on), and `num_banks` for interleaved tensors, where pages are round-robined across banks so what is contiguous within one bank is every `num_banks`'th page. When the stride is greater than 1 the pages arrive in stride order rather than tensor order, and the consumer has to account for that.

Two things to get right when sizing the transfer:

- Pages are `aligned_page_size` apart, not `page_size` apart. The two differ whenever a tensor's page size is not a multiple of the allocator alignment, and a bulk transfer then carries the padding between pages along with the data. That is exactly right when the destination has the same layout, but a consumer that expects tightly packed pages in L1 has to account for it. Padding never shortens a run — the count is the same whether or not the page size is aligned — so a caller that needs tightly packed pages has to fall back to per-page transfers.
- Cap `end_page_id` by what the destination can actually hold, not just by your logical range. A single shard can be far larger than a CB, and `noc_async_read` will happily issue the whole thing.

For sharded tensors the run is the longest one that stays inside one shard, walking the innermost dimension the shard actually spans. It extends past that dimension when the shard covers it exactly, so a shard whose shape matches the tensor in every dimension but the first yields runs a whole shard long. Padding in edge shards and shard boundaries both cut a run short. One case is deliberately conservative: when shards sit next to each other in a bank (shard-contiguous placement, or a single bank), a run could carry on past the shard edge, but it stops at the edge.

A run is always the longest one a constant page-id step can describe, which is not always the longest one in memory. You get the whole shard exactly when the shard is one-dimensional in page terms: every dimension but one is a single page, or the shard matches the tensor exactly in every dimension inside the outermost one it splits. Otherwise the shard is a multi-dimensional block — the rest of it is still contiguous in memory, but its page ids are not an arithmetic sequence, so no stride can reach them. Use [`shard_pages()`](./tensor_accessor_iterator.md) to walk a whole shard in memory order in that case. Between the two, every layout is covered:

| Layout | `num_contiguous_pages` | Use `shard_pages()` instead |
| --- | --- | --- |
| Interleaved | whole bank run | n/a |
| Height sharded (tile and row-major) | whole shard | no |
| Row-major width / block sharded | whole shard, stride `tensor_shape[-1]` | no |
| Row-major ND sharded | whole shard if only one shard dimension exceeds a page, else that dimension's extent | only if more than one does |
| Tile block sharded | one shard row | yes, for the whole shard |

## Tensor Accessor iterators
You can use TensorAccessor iterators to speed up and/or simplify iteration over pages in a tensor.
[Tensor Accessor iterators documentation.](./tensor_accessor_iterator.md)

## Performance Considerations
- If rank is static, then construction of TensorAccessor is 0-cost, meaning that everything is precomputed in compile time.
- Calculation of address scales ~linearly with rank
- Iterator-based approaches are more efficient than repeated calls to get_noc_addr() due to state caching (especially shard_pages)


## Examples:
- Reshard [reader](../../tests/ttnn/unit_tests/gtests/accessor/kernels/reader_reshard.cpp), [writer](../../tests/ttnn/unit_tests/gtests/accessor/kernels/writer_reshard.cpp)
