# Eltwise FPU op chain (AXPY)

`y = a*x + y` chained from two different FPU binary ops (`mul_tiles` followed
by `add_tiles`). This is the FPU counterpart of the `sfpu_eltwise_chain`
example.

## Why this example exists

A newcomer's first compute kernel is usually `eltwise_binary`. Its compute
kernel calls `add_tiles_init()` once before the loop and uses one
`tile_regs_acquire()` / `tile_regs_release()` pair per iteration:

```cpp
binary_op_init_common(cb_in0, cb_in1, cb_out0);
add_tiles_init(cb_in0, cb_in1);     // once, before the loop

for (uint32_t i = 0; i < n_tiles; i++) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, 1);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    tile_regs_release();
}
```

That works because the op never changes — `add_tiles` fires every iteration.
The moment you chain two different ops in the loop body, this minimal pattern
no longer works, and the kernel hangs in a `MathWaitDataDependency` (`MWDD`)
state visible in the watcher log.

The two rules that resolve this are:

1. **`*_init` is called every time the op or its input CBs change.**
   `binary_op_init_common()` only sets up the initial state.
2. **`tile_regs_acquire/release` brackets one math chain plus its `pack_tile()`.**
   You need a fresh pair around every pack — the destination register cannot
   be reused across packs.

The first rule is why `mul_tiles_init` and `add_tiles_init` appear *inside*
the loop in `kernels/compute/axpy.cpp`: mul → add → mul → add → … alternates
on every iteration, so both inits run on every iteration. The second rule is
why there are two `acquire/release` pairs per iteration — one around the mul
+ pack, one around the add + pack.

## Wrong vs. correct

What people try first (and what hangs):

```cpp
// WRONG — both *_init only once, single acquire/release for two ops.
binary_op_init_common(cb_x, cb_a, cb_out);
mul_tiles_init(cb_x, cb_a);
add_tiles_init(cb_ax, cb_y);   // overwrites the mul state!

for (uint32_t i = 0; i < n_tiles; i++) {
    tile_regs_acquire();
    mul_tiles(cb_x, cb_a, 0, 0, 0);
    pack_tile(0, cb_ax);
    add_tiles(cb_ax, cb_y, 0, 0, 0);  // engine is configured for ADD, but
                                       // the previous pack already consumed
                                       // the dst-reg window
    pack_tile(0, cb_out);
    tile_regs_release();
}
```

What works (and is what `axpy.cpp` does):

```cpp
binary_op_init_common(cb_x, cb_a, cb_ax);  // once

for (uint32_t i = 0; i < n_tiles; i++) {
    mul_tiles_init(cb_x, cb_a);            // op switched (back) to MUL
    tile_regs_acquire();
    mul_tiles(cb_x, cb_a, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_ax, 1);
    pack_tile(0, cb_ax);
    cb_push_back(cb_ax, 1);
    tile_regs_release();
    // ... cb_pops ...

    add_tiles_init(cb_ax, cb_y);           // op switched MUL -> ADD
    tile_regs_acquire();                   // fresh acquire after the pack
    add_tiles(cb_ax, cb_y, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    // ... cb_pops ...
}
```

## When you only need one `acquire/release`

You can keep a single `acquire/release` pair across multiple math ops *if*
they all accumulate into the same destination register and you pack only
once at the end. This is exactly how matmul accumulates the K dimension:

```cpp
tile_regs_acquire();
for (uint32_t k = 0; k < Kt; ++k) {
    matmul_tiles(cb_a, cb_b, k, k, dst, /*acc=*/true);
}
tile_regs_commit();
tile_regs_wait();
pack_tile(dst, cb_out);
tile_regs_release();
```

`sfpu_eltwise_chain` does the same thing for SFPU ops because they all read
and write the same destination register in place. AXPY mixes a math op
(`mul_tiles`) with another math op that takes both inputs from CBs
(`add_tiles`), so the natural pattern is the two-phase one shown above.

## Layout

```
kernels/
  dataflow/
    reader.cpp   — streams x[i], y[i]; fills cb_a with broadcast scalar
    writer.cpp   — drains cb_out to DRAM
  compute/
    axpy.cpp     — mul_tiles -> add_tiles with per-op init + acquire/release
eltwise_fpu_chain.cpp  — host code, golden check
```

## Build & run

```
./build_metal.sh --build-tests
./build/programming_examples/metal_example_eltwise_fpu_chain
```
