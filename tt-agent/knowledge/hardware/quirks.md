# Tensix Hardware Quirks

Things that look like C++ but aren't. Every TT kernel developer hits these.
Hardware-stable — these reflect silicon behavior, not software conventions.

## No Dynamic Allocation

No `malloc`, `new`, or dynamic data structures in kernels. All buffers (CBs,
semaphores, L1 scratch) are statically configured from the host before launch.
Violating this silently corrupts memory.

## Entry Point is kernel_main(), Not main()

```cpp
void kernel_main() {    // correct
    // ...
}
int main() { }          // wrong — never called
```

## 32-bit RISC-V

No `uint64_t`, `int64_t`, `double`, or 64-bit pointer arithmetic. Addresses are
32-bit. Use `uint32_t` for addresses and sizes.

## Compile-time vs Runtime Args

Two separate arg mechanisms — do not mix them:

```cpp
// Compile-time (template/constexpr, baked into binary at host compile time)
constexpr uint32_t tile_size = get_compile_time_arg_val(0);

// Runtime (passed per-launch, can vary per core)
uint32_t src_addr = get_arg_val<uint32_t>(0);
```

## ALWI: Math Inline Assembly Macro

Math operations in compute kernels use the `ALWI` macro:
```cpp
ALWI void add_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t dst_idx) {
    // ...
}
```
Functions called from within `math_main` must be marked `ALWI`.

## No Floating Point in Dataflow Kernels

DM0 and DM1 (reader/writer) RISC-V cores have no FPU. Do not use `float` or
`double` in dataflow kernels. Address calculations must be integer-only.

## NOC Barriers Are Mandatory

`noc_async_read` and `noc_async_write` are asynchronous. Always follow with
a barrier before reading the transferred data:

```cpp
noc_async_read(src_addr, dst_addr, size);
noc_async_read_barrier();   // required — data not valid until after this
```

## NOC Direction Convention

- NOC0: reads (DM0 pulling from DRAM or remote L1)
- NOC1: writes (DM1 pushing to DRAM or remote L1)

Using the wrong NOC doesn't fail immediately — it causes throughput degradation
or deadlocks under load.

## Deadlock Conditions

Deadlock occurs when:
- Reader and writer block on the same CB simultaneously (classic circular wait)
- `cb_reserve_back` and `cb_wait_front` are called in the wrong order
- A kernel exits without consuming all tiles it was supposed to

Always verify the reader/compute/writer tile count contract before running.
