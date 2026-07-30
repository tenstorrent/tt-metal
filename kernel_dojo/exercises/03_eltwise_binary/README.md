# 03 — Element-wise binary: two inputs on the FPU

**Goal:** compute `c = a + b`. Two input tensors, two circular buffers, and the
matrix engine instead of the vector engine.

> **Background:** [`theory 06 — Compute`](../../theory/06-compute.md) for the
> FPU/SFPU split, and [`theory 05 — Data movement`](../../theory/05-data-movement.md)
> for why one barrier covers several reads.

---

## Theory

### The FPU takes two CBs directly

Lesson 02 needed `copy_tile` to stage data in DST because the SFPU only works on
DST. The FPU is different: `add_tiles`, `sub_tiles` and `mul_tiles` read
**straight from two circular buffers** and write the result to DST.

```cpp
add_tiles(cb_in0, cb_in1, tile_idx0, tile_idx1, dst_idx);
```

No `copy_tile`. The unpacker feeds SrcA from `cb_in0` and SrcB from `cb_in1`, the
FPU adds them element-wise, DST gets the result. One instruction, one tile of
output, and it is considerably faster than the SFPU path.

Note `tile_idx0` / `tile_idx1`: these index *within the CB's currently visible
window*. After `cb_wait_front(cb, 4)` the four available tiles are indices 0..3.
In this lesson we only ever make one visible, so both indices are 0.

### Init for binary ops

```cpp
binary_op_init_common(cb_in0, cb_in1, cb_out);  // once, at the top
add_tiles_init(cb_in0, cb_in1);                 // per-op kind
```

`binary_op_init_common` is the binary counterpart of `init_sfpu` — same rule, it
must be the first compute call and must not be repeated. `add_tiles_init` tells
the FPU which of add/sub/mul to run; switch operation mid-kernel and you call
the corresponding `*_tiles_init` again.

### Two accessors, chained compile-time args

The reader now handles two tensors. Each `TensorAccessorArgs` consumes a
variable number of compile-time args, so the second must start where the first
finished:

```cpp
constexpr auto a_args = TensorAccessorArgs<1>();
const auto a = TensorAccessor(a_args, a_addr);

constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
const auto b = TensorAccessor(b_args, b_addr);
```

Hard-coding the second offset is a bug waiting to happen — the arg count depends
on the memory config, so it changes if the tensor becomes sharded.

### Overlapping the two reads

The obvious loop body is:

```cpp
noc_async_read_page(i, a, a_l1);
noc_async_read_barrier();
noc_async_read_page(i, b, b_l1);
noc_async_read_barrier();
```

which serialises two round trips. Issue both first, then barrier once:

```cpp
noc_async_read_page(i, a, a_l1);
noc_async_read_page(i, b, b_l1);
noc_async_read_barrier();     // waits for *all* outstanding reads on this NoC
```

The barrier is not per-transaction — it drains everything this processor has
outstanding. That makes batching essentially free to write, and it is the
cheapest performance win available in a reader kernel.

---

## Your task

Write two kernels:

- **`kernels/reader.cpp`** — read tile `i` of `a` into CB 0 and tile `i` of `b`
  into CB 1, for each of `n_tiles` tiles. Overlap the two reads.
- **`kernels/compute.cpp`** — add the two tiles, result to CB 16.

The writer is provided.

### What the host gives you

**`reader.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB index for `a` (0) |
| compile-time arg 1 | CB index for `b` (1) |
| compile-time args 2.. | `TensorAccessorArgs` for `a`, then for `b` |
| runtime arg 0 | `a` base address |
| runtime arg 1 | `b` base address |
| runtime arg 2 | number of tiles |

**`compute.cpp`**

| | |
|---|---|
| compile-time arg 0 | CB index for `a` (0) |
| compile-time arg 1 | CB index for `b` (1) |
| compile-time arg 2 | output CB index (16) |
| runtime arg 0 | number of tiles |

### API you need

```cpp
binary_op_init_common(icb0, icb1, ocb);
add_tiles_init(icb0, icb1);
add_tiles(icb0, icb1, itile0, itile1, dst_idx);
```

plus everything from lessons 01 and 02.

### Run it

```bash
./dojo test 03
```

---

## Hints

<details>
<summary>Reserving two CBs at once</summary>

Reserve space in **both** CBs before issuing either read, and push **both**
after the barrier:

```cpp
cb_reserve_back(cb_a, 1);
cb_reserve_back(cb_b, 1);
... two reads, one barrier ...
cb_push_back(cb_a, 1);
cb_push_back(cb_b, 1);
```

Interleaving reserve/read/push per buffer also works, but costs you the overlap.

</details>

<details>
<summary>Output equals `a`, or equals `b`</summary>

Check the argument order of `add_tiles`: it is
`(cb0, cb1, tile_in_cb0, tile_in_cb1, dst)`. A common slip is passing the two
DST-relative indices where the CB-relative ones belong.

Also confirm you're popping *both* input CBs — popping only one desynchronises
the buffers and you'll read stale tiles after the first iteration.

</details>

---

## Going further

- Change `add_tiles` to `mul_tiles` (with `mul_tiles_init`) and update the
  golden in `task.py` to match. Confirm it still passes.
- Try computing `exp(a) + b`: you'll need both engines in one kernel, and
  you'll discover that switching between FPU and SFPU ops needs the right init
  call in between.
