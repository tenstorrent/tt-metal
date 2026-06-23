# bh_dram_read — Design

**Date:** 2026-06-23
**Status:** Approved (design), pending spec review

## Summary

`bh_dram_read` is a minimal, read-only TTNN operation intended as a clean
skeleton for DRAM-read work (e.g. a bandwidth microbenchmark or a custom
gather). The host API takes a single DRAM-resident input tensor and returns
`void`. Internally it launches one device primitive whose program places **one
worker (Tensor) core per DRAM bank**; each core reads the pages of the input
tensor that live in its assigned bank into a circular buffer and discards them.
There is no compute kernel and no writer kernel.

The op is built on the modern descriptor-based program-factory API
(Device 2.0 / "Metal2"): `ProgramDescriptor` + `KernelDescriptor` +
`CBDescriptor`, with nanobind bindings, following the `examples/example`
template.

## Goals

- Provide a working, minimal read-only op that compiles, binds to Python, and
  runs on Blackhole.
- Demonstrate the "one core per DRAM bank" placement pattern.
- Keep the surface area small so it can be grown into a real op later.

## Non-goals

- No output tensor / data transform (read-only, void return).
- No compute kernel, no writer kernel.
- No performance tuning, multi-subchannel handling, or sharded-input support in
  this first cut. Input is assumed DRAM **interleaved**.

## Location & file layout

Mirrors `ttnn/cpp/ttnn/operations/examples/example/`:

```
ttnn/cpp/ttnn/operations/examples/bh_dram_read/
├── bh_dram_read.hpp                       # host API: void bh_dram_read(const Tensor&)
├── bh_dram_read.cpp                       # host wrapper -> prim::bh_dram_read
├── bh_dram_read_nanobind.hpp
├── bh_dram_read_nanobind.cpp              # nanobind binding
└── device/
    ├── bh_dram_read_device_operation.hpp  # device op struct
    ├── bh_dram_read_device_operation.cpp  # select factory / validate / specs / prim
    ├── bh_dram_read_program_factory.cpp   # single descriptor factory
    └── kernels/dataflow/
        └── reader_bh_dram_read.cpp        # reader-only kernel
```

Registration: add the binding call to
`ttnn/cpp/ttnn/operations/examples/examples_nanobind.cpp` (same place
`bind_example_operation` is called), so it is exposed under the `examples`
submodule. CMake source lists for the `examples` operations are updated to
include the new `.cpp` files.

## Components

### 1. Host API (`bh_dram_read.hpp` / `.cpp`)

```cpp
namespace ttnn {
void bh_dram_read(const Tensor& input_tensor);
}
```

Implementation calls the primitive and discards its (aliased) return:

```cpp
void bh_dram_read(const Tensor& input_tensor) {
    ttnn::prim::bh_dram_read(input_tensor);
}
```

### 2. Device operation (`bh_dram_read_device_operation.hpp` / `.cpp`)

Follows the `ExampleDeviceOperation` shape:

- `operation_attributes_t` — empty for now (placeholder for future knobs).
- `tensor_args_t` — `const Tensor& input_tensor;`
- `spec_return_value_t = ttnn::TensorSpec;`
- `tensor_return_value_t = Tensor;`
- One program factory variant: `struct DramBankCore { static ProgramDescriptor
  create_descriptor(...); };` and `program_factory_t = std::variant<DramBankCore>;`
- `select_program_factory` returns `DramBankCore{}`.
- `validate_on_program_cache_miss` asserts the input tensor is on device and in
  DRAM (`input.memory_config().buffer_type() == BufferType::DRAM`) and is
  interleaved.
- `compute_output_specs` returns the input's spec; `create_output_tensors`
  returns the **input tensor unchanged** (no new allocation — read-only).
- `ttnn::prim::bh_dram_read(const Tensor&)` builds attributes + tensor_args and
  calls `ttnn::device_operation::launch<BhDramReadDeviceOperation>(...)`.

The aliased input as the "output" is what lets the host wrapper present a `void`
API while satisfying the framework's requirement for a `tensor_return_value_t`.

### 3. Program factory (`bh_dram_read_program_factory.cpp`)

```
num_banks = input.device()->num_dram_channels()
page_size = input page size (bytes)
total_pages = input.physical_volume() / TILE_HW   (interleaved, tiled)
```

- Select the first `num_banks` worker cores from the compute-with-storage grid
  (row-major), core `i` ↔ bank `i`.
- For each bank `i`, the pages that live in it (interleaved round-robin) are
  `{ i, i+num_banks, i+2*num_banks, ... }`; count
  `pages_in_bank_i = ceil((total_pages - i) / num_banks)` (0 if `i >= total_pages`).
- One `CBDescriptor` (2 tiles deep) on all selected cores.
- One reader `KernelDescriptor` (`ReaderConfigDescriptor`) on all selected
  cores. Compile-time args: `page_size` (raw tile size, the read length) and
  `aligned_page_size` (DRAM-aligned in-bank stride). Per-core runtime args:
  `{ base_addr, bank_id=i, pages_in_bank_i }`.

  Note: for interleaved DRAM buffers each bank stores its pages contiguously at
  the DRAM-aligned page size, so the in-bank offset of the `s`-th page is
  `s * aligned_page_size`, while the bytes read per page is `page_size`. The
  exact stride is confirmed against the buffer's page layout during impl.
- No writer, no compute kernel.

### 4. Reader kernel (`reader_bh_dram_read.cpp`)

```c
// compile-time: page_size, aligned_page_size
// runtime: base_addr, bank_id, num_pages
for (s = 0; s < num_pages; ++s) {
    cb_reserve_back(cb0, 1);
    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(bank_id, base_addr + s * aligned_page_size);
    noc_async_read(noc_addr, get_write_ptr(cb0), page_size);
    noc_async_read_barrier();
    cb_push_back(cb0, 1);
    cb_pop_front(cb0, 1);   // discard; reader-only
}
```

`get_noc_addr_from_bank_id<true>(bank_id, base_addr + offset)` resolves to the
bank's NOC base plus the in-bank byte offset; for an interleaved tiled tensor
the `s`-th page in bank `i` sits at in-bank offset `s * aligned_page_size`.

## Data flow

```
input (DRAM interleaved) ──> N worker cores (1 per bank)
                              each core: read its bank's pages -> CB -> discard
                            ──> void
```

## Error handling

- `validate_on_program_cache_miss`: `TT_FATAL` if input is not on device, not
  DRAM, or not interleaved.
- If `total_pages < num_banks`, trailing cores get `num_pages = 0` and simply do
  no reads (loop body skipped). Still placed so placement stays uniform.

## Testing

- C++/pytest smoke test: allocate a DRAM-interleaved tiled tensor, call
  `ttnn.examples.bh_dram_read(t)` (exact Python path confirmed during impl),
  assert it runs without error and the input tensor is unchanged.
- Run on Blackhole (p150b) per the project build/test workflow.

## Open questions / future work

- Multi-subchannel DRAM (read from all subchannels of a bank) — deferred.
- Sharded-input support — deferred.
- Turning the discard loop into a timed bandwidth measurement — deferred.
