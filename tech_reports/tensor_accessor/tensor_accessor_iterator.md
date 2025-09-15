# Tensor Accessor (TA) Iterators 📚

## Pages Iterator 📄

### Usage
The pages iterator allows you to iterate over pages in a tensor within a given range of pages. It works for both sharded and interleaved tensors.
- For sharded tensors: `end_page_id` defaults to `tensor_volume()` when set to 0
- For interleaved tensors: both `start_page_id` and `end_page_id` must be explicitly provided since the accessor doesn't know the tensor volume

### Performance Considerations 🚀
- For interleaved tensors, performance is identical to just iterating over page IDs with TA and calling `get_noc_addr` for each of them.
- For sharded tensors, performance is better with the iterator, since its state allows skipping some expensive address computations.

### Creation ⚙️
```c++
// Sharded tensor examples:
// Iterate over all pages (default behavior)
auto pages = tensor_accessor.pages();

// Iterate over pages starting from page 100
auto pages = tensor_accessor.pages(/*start_page_id=*/100);

// Iterate over pages from 100 to 500
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
    noc_async_writes_flushed();
    ++page_dst;
}

// For interleaved tensors (must specify tensor volume)
auto pages_src = tensor_accessor_src.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto pages_dst = tensor_accessor_dst.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto page_dst = pages_dst.begin();
for (const auto& page_src : pages_src) {
    noc_async_read(page_src.noc_addr(), page_dst->noc_addr(), page_size);
    noc_async_writes_flushed();
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

### Advanced Usage 🔧
The pages iterator is a random access iterator. If you need to access pages in a more complex way, you can do so like this (strided write example):
```C++
auto pages = tensor_accessor.pages(/*start_page_id=*/0, /*end_page_id=*/tensor_volume);
auto it = pages.begin();
while (it != pages.end()) {
    noc_async_write_page(it->page_id(), tensor_accessor_dst, it->noc_addr());
    noc_async_writes_flushed();
    it += stride;
}
```

This is a bit uglier than using a range-based for loop, but you still get better performance in the case of sharded tensors.


## Shard Pages Iterator 🧩

### Usage
You can use the shard pages iterator to iterate over a range of pages in a shard with a step >= 1.
- It handles cases when the input shape is not divisible by the shard shape along some dimension(s) (i.e., it skips padding pages)
- Note: works only for sharded tensors

### Performance Considerations 🚀
- Since it maintains state, address calculation is always more efficient than calling `accessor.get_noc_addr` at each step.

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
        noc_async_writes_flushed();
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
        noc_async_writes_flushed();
        ++page_dst;
    }
}

```

## When Should You Use Each Iterator? 🤔

The regular (non-shard) address iterator is a generic abstraction for iteration over pages on any kind of tensor.
It works for all kinds of tensors (interleaved/sharded, L1/DRAM, etc.). It's good for improving readability of generic code that is intended to be used with any kind of inputs, and should speed up address calculation when called with sharded tensors.

The shard address iterator should be used only in specific cases, when a developer intends to optimize a kernel for the sharded tensor case.

### Example 📋

Let's say we have an operation that needs to read a tensor page by page.
It receives a width-sharded tensor as input:
```
[ 0 1 ]
[ 2 3 ]
```
This tensor has 4 pages, and let's say it's sharded on 2 cores:
- 0, 2 are on the first bank
- 1, 3 are on the second bank

If you take a simple "interleaved-style" kernel implementation that just splits pages between cores, and uses `.pages()` to iterate over them, you will have core 0 processing pages 0 and 1, and core 1 processing pages 2 and 3.

With `TensorAccessor::pages()`, you just iterate over a range of pages. This kernel will work for any kind of input (sharded/interleaved), but it won't be the most efficient. The kernel structure will look something like this:

```C++
// ...
auto start_page_id = get_arg_val<uint32_t>(...);
auto end_page_id = get_arg_val<uint32_t>(...);
auto tensor_accessor_src = TensorAccessor<...>(...);
auto pages = tensor_accessor_src.pages(start_page_id, end_page_id);
for (const auto& page: pages) {
    process_page(page);
}
// ...
```

Now, let's say we want to optimize this for a sharded tensor. It would be more efficient to read core-local pages.
We can iterate over local shards, and then inside this loop, use `.shard_pages()` to iterate over pages inside each local shard.
(core 0 processes pages 0 and 2, core 1 processes pages 1 and 3). The kernel structure will look something like this:

```C++
// Read arguments ...
auto tensor_accessor_src = TensorAccessor<...>(...);
for (uint32_t shard_id = start_shard_id; shard_id < n_shards; shard_id += shard_stride) {
    auto pages = tensor_accessor_src.shard_pages(shard_id);
    for (const auto& page : pages) {
        process_local_page(page);
    }
}
// ...
```
