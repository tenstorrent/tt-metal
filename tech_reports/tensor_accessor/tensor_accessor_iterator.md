# Tensor Accessor (TA) Iterators 📚

## Pages Iterator 📄

### Usage
The pages iterator allows you to iterate over pages in a tensor within a given range of pages. It works for both sharded and interleaved tensors.
- For sharded tensors: `end_page_id` defaults to `tensor_volume()`.
- For interleaved tensors: both `start_page_id` and `end_page_id` must be explicitly provided since the accessor doesn't know the tensor volume.
- If `start_page_id` >= `end_page_id` or `start_page_id` >= tensor_accessor.dspec().tensor_volume(), TA.pages(...).begin() == TA.pages(...).end().

### Performance Considerations 🚀
- For interleaved tensors, performance is identical to just iterating over page IDs and calling `TA.get_noc_addr(...)` for each of them.
- For sharded tensors, performance is better with the iterator since its state allows skipping some expensive address computations.

### Creation ⚙️
```c++
// Sharded tensor examples:
// Iterate over all pages (default behavior)
auto pages = tensor_accessor.pages();

// Iterate over pages starting from page with id=100
auto pages = tensor_accessor.pages(/*start_page_id=*/100);

// Iterate over pages ids from 100 to 500
auto pages = tensor_accessor.pages(/*start_page_id=*/100, /*end_page_id=*/500);

// Interleaved tensor examples:
// Must provide both start and end page IDs
auto pages = tensor_accessor.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);

// Iterate over first half of tensor
auto pages = tensor_accessor.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume / 2);
```

### Examples of Using the Pages Iterator 💡
Tensor copy:
```c++
// For sharded tensors
auto pages_src = tensor_accessor_src.pages(/*start_page_id=*/0);
auto pages_dst = tensor_accessor_dst.pages();
auto page_dst = pages_dst.begin();
for (const auto& page_src : pages_src) {
    noc_async_read(page_src.noc_addr(), page_dst->noc_addr(), page_size);
    noc_async_read_barrier();
    ++page_dst;
}

// For interleaved tensors (must specify tensor volume)
auto pages_src = tensor_accessor_src.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto pages_dst = tensor_accessor_dst.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto page_dst = pages_dst.begin();
for (const auto& page_src : pages_src) {
    noc_async_read(page_src.noc_addr(), page_dst->noc_addr(), page_size);
    noc_async_read_barrier();
    ++page_dst;
}

// Example: Process only middle portion of tensor
uint32_t quarter = tensor_volume / 4;
auto pages_src = tensor_accessor_src.pages(/*start_page_id=*/quarter, /*end_page_id=*/3 * quarter);
for (const auto& page_src : pages_src) {
    // Process middle 50% of tensor
    process_page(page_src);
}
```

### Note on NOC Address Computation ⚠️
`tensor_accessor.get_noc_addr(...)` can be quite expensive, but when using the iterator, `page.noc_addr()` returns a precomputed address. Actual address calculation is done in `PagesAddressIterator::operator++`.

### Note on Padding ⚠️
In case of a sharded tensor where the tensor shape is not divisible by the shard shape, the tensor shape is padded. The Pages Iterator iterates over **logical** pages, so padded pages are ignored.

### Advanced Usage 🔧
The pages iterator is a `std::forward_iterator_tag` iterator. But if you need to access pages in a more complex way, you can increment with any positive step, like this (strided write example):
```C++
auto pages = tensor_accessor.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto page = pages.begin();
while (page != pages.end()) {
    noc_async_write_page(page->page_id(), tensor_accessor_dst, page->noc_addr());
    noc_async_writes_flushed();
    page += stride;
}
```

This is a bit uglier than using a range-based for loop, but you still get better performance in the case of sharded tensors.


## Shard Pages Iterator 🧩

### Usage
You can use the shard pages iterator to iterate over pages in a single shard with a step >= 1 in row-major order. It works **only** for sharded tensors.
- One required argument is `shard_id`, which is the shard's ID in row-major order within the sharded tensor.
- You can specify `start_page_offset` and `end_page_offset`. You can imagine a flattened range of pages iterated over by the shard pages iterator. Then `start_page_offset` and `end_page_offset` would crop this range from the start and end. Note that generally page IDs in one shard are not contiguous.

### Performance Considerations 🚀
- Since it maintains state, address calculation is always more efficient than calling `accessor.get_noc_addr()` at each step.

```c++
// Iterate over all pages in a shard
auto shard_pages = tensor_accessor.shard_pages(shard_id);

// Iterate over pages starting from offset 10
auto shard_pages = tensor_accessor.shard_pages(shard_id, /*start_page_offset=*/10);

// Iterate over pages from offset 10 to 50 within the shard
auto shard_pages = tensor_accessor.shard_pages(shard_id, /*start_page_offset=*/10, /*end_page_offset=*/50);
```

### Examples of Using the Shard Pages Iterator 💡
Sharded tensor copy:
```c++
for (uint32_t i = 0; i < num_shards; ++i) {
    uint32_t shard_id = first_shard_id + i * num_cores;
    auto shard_pages = accessor_src.shard_pages(shard_id);
    for (const auto& shard_page : shard_pages) {
        noc_async_write_page(
            /*id = */ shard_page.page_id(),
            /*addrgen = */tensor_accessor_dst,
            /*src_local_l1_addr = */shard_page.noc_addr()
        );
        noc_async_writes_flushed();
    }
}
```

Or this sharded tensor copy, which should be more efficient, since it uses an iterator to calculate the address for both the src and the dst:
```c++
for (uint32_t i = 0; i < num_shards; ++i) {
    uint32_t shard_id = first_shard_id + i * num_cores;
    auto shard_pages_src = tensor_accessor_src.shard_pages(shard_id, /*start_page_offset=*/0);
    auto shard_pages_dst = tensor_accessor_dst.shard_pages(shard_id);
    auto page_dst = shard_pages_dst.begin();
    for (const auto& page_src : shard_pages_src) {
        noc_async_read(page_src.noc_addr(), page_dst->noc_addr(), page_size);
        noc_async_read_barrier();
        ++page_dst;
    }
}

// Example: Copy only first half of each shard
for (uint32_t i = 0; i < num_shards; ++i) {
    uint32_t shard_id = first_shard_id + i * num_cores;
    uint32_t shard_volume = tensor_accessor_src.dspec().shard_volume();
    uint32_t half_shard = shard_volume / 2;
    auto shard_pages_src = tensor_accessor_src.shard_pages(shard_id, /*start_page_offset=*/0, /*end_page_offset=*/half_shard);
    auto shard_pages_dst = tensor_accessor_dst.shard_pages(shard_id, /*start_page_offset=*/0, /*end_page_offset=*/half_shard);
    auto page_dst = shard_pages_dst.begin();
    for (const auto& page_src : shard_pages_src) {
        noc_async_read(page_src.noc_addr(), page_dst->noc_addr(), page_size);
        noc_async_read_barrier();
        ++page_dst;
    }
}

```

### Note on Padding ⚠️
In case the tensor shape is not divisible by the shard shape, the tensor shape is padded. The Shard Pages iterator iterates over **logical** pages, so padded pages are ignored. This also means two `shard_pages()` calls with the same offsets can have different sizes if they are created for different shards.

### Advanced Usage 🔧
Like with a Page Accessor, you can iterate over pages with a positive step >= 1.

## When Should You Use Each Iterator? 🤔

The regular (non-shard) pages iterator is a generic abstraction for iterating over pages on any kind of tensor.
It works for all kinds of tensors (interleaved/sharded, L1/DRAM, etc.). It's good for improving the readability of generic code that is intended to be used with any kind of input, and it should speed up address calculation when called with sharded tensors.

The shard pages iterator should be used only in specific cases when a developer intends to optimize a kernel for the sharded tensor case.

### Reshard Op Example 📋
The reshard op takes input and output tensors that have different `TensorSpec`s. They can differ in anything: `BufferType` (DRAM/L1), `MemoryLayout` (Interleaved/Sharded), or they can have different sharding specifications (e.g., WIDTH_SHARDED->HEIGHT_SHARDED, change of shard shape, etc.).

Since `TensorAccessor` abstracts away the tensor specification, we essentially need to implement a tensor copy.

#### Pages Iterator
We can create a very simple generic kernel that works on any kind of input with the help of `TA.pages(...)`.

- On the [host side](https://github.com/tenstorrent/tt-metal/blob/f0b96983b1fd5817290c2a5babc1c14f4c60f330/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory.cpp#L13), we simply split all pages that need to be copied evenly between cores (core 0: [0 : n_pages / n_cores], core 1: [n_pages / n_cores : n_pages / n_cores*2], ... core n: [n_pages / n_cores * (n_cores - 1) : n_pages]).
- On the [device side](https://github.com/tenstorrent/tt-metal/blob/f0b96983b1fd5817290c2a5babc1c14f4c60f330/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp#L9), we create a Pages Iterator over a given range of pages for a given core.
  - The reader reads pages from the input tensor and pushes them into a CB
  - The writer pops pages from the CB and writes them to the output tensor

<details>
<summary>Kernels code:</summary>

**Reader**
```C++
void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    constexpr uint32_t base_idx_cta = args_src.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_src.next_common_runtime_args_offset();

    constexpr uint32_t cb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    auto accessor_src = TensorAccessor(args_src, bank_base_address_src, page_size);

    constexpr uint32_t one_tile = 1;
    uint32_t cb_addr = get_write_ptr(cb_id);
    auto pages = accessor_src.pages(start_page, end_page);
    for (const auto& page : pages) {
        cb_reserve_back(cb_id, one_tile);
        noc_async_read(page.noc_addr(), cb_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, one_tile);
    }
}
```

**Writer**
```C++
void kernel_main() {
    auto args_dst = TensorAccessorArgs<0, 0>();
    constexpr uint32_t base_idx_cta = args_dst.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_dst.next_common_runtime_args_offset();

    constexpr uint32_t cb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_dst = get_common_arg_val<uint32_t>(base_idx_crta);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    auto accessor_dst = TensorAccessor(args_dst, bank_base_address_dst, page_size);

    constexpr uint32_t one_tile = 1;
    uint32_t cb_addr = get_read_ptr(cb_id);
    auto pages = accessor_dst.pages(start_page, end_page);
    for (const auto& page : pages) {
        cb_wait_front(cb_id, one_tile);
        noc_async_write(cb_addr, page.noc_addr(), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, one_tile);
    }
}
```

</details>

This kernel will work with any kind of input and output tensors. However, it's not optimal for sharded inputs. Let's review a simple case where it could be optimized.
Let's say the reshard op receives a tiny width-sharded tensor as input:
```
[ 0 1 ]
[ 2 3 ]
```
This tensor has 4 pages (0, 1, 2, 3), and let's say it's sharded on 2 cores (banks). Since it's width-sharded:
- pages 0 and 2 allocated on the first bank (core)
- pages 1 and 3 allocated on the second bank (core)

If you use the generic kernel implementation that uses the Pages Iterator from above, you will have core 0 processing pages 0 and 1, and core 1 processing pages 2 and 3. You can see that in such a case, core 0 reads data residing on both core 0 and core 1, and vice versa. It's much quicker to read core-local data than to issue a NOC transaction to read data from a remote core (bank).

### Shard Pages Iterator
The simplest and quite effective optimization would be to read only core-local data, and then write each page wherever it will be mapped to in the output tensor (possibly a remote core).

To achieve this, we need to iterate over pages inside the local shard (i.e., the shard allocated on the same core as the core processing it).

- On the [host side](https://github.com/tenstorrent/tt-metal/blob/f0b96983b1fd5817290c2a5babc1c14f4c60f330/ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory.cpp#L124), we provide kernels with a range of shards they have to process. Generally, shards are mapped in a round-robin fashion (`shard_id` goes to a core (bank) `shard_id % n_banks`).
- On the [device side](https://github.com/tenstorrent/tt-metal/blob/fb3b1e24b601fe1a6cd712b92388b4ece46bc72d/tests/ttnn/unit_tests/gtests/accessor/kernels/copy_local_shard_iterator.cpp), we iterate over local shards and then use the shard iterator to iterate over all the pages inside a given shard.

<details>
<summary>Kernel code:</summary>

```C++
void kernel_main() {
    uint32_t page_size = get_compile_time_arg_val(0);
    uint32_t input_base_address = get_arg_val<uint32_t>(0);
    uint32_t output_base_address = get_arg_val<uint32_t>(1);
    uint32_t first_shard_id = get_arg_val<uint32_t>(2);
    uint32_t num_cores = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    auto args_src = TensorAccessorArgs<1, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, page_size);
    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address, page_size);

    for (uint32_t i = 0; i < num_shards; ++i) {
        uint32_t shard_id = first_shard_id + i * num_cores;
        auto shard_pages = tensor_accessor_src.shard_pages(shard_id);
        for (const auto& page : shard_pages) {
            ASSERT(tensor_accessor_src.is_local_addr(page.noc_addr()));
            noc_async_write_page(/*id=*/page.page_id(), /*addrgen=*/tensor_accessor_dst, /*src_local_l1_addr=*/page.noc_addr());
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
}
```

</details>

Note that we don't need a CB and a separate kernel for the writer for such an optimized variant, since there is no possibility of having a remote->remote NOC transaction. So the core can issue a direct NOC transaction from local to local or remote core.

In practice, this approach gives a [2x+ speedup](https://github.com/tenstorrent/tt-metal/pull/25902) depending on input/output sharding.
