# In-Kernel Handling of Non-Tile-Aligned Shapes in TTNN

A survey of techniques used across TTNN operations for handling tensor dimensions
that are not multiples of 32 (the tile side), with focus on **kernel-side**
mechanisms and actionable guidance for new ops.

---

## 1. The Garbage Contract

Every TTNN op is governed by an implicit contract:

- **Input side**: An op may receive tiles whose padding elements contain
  arbitrary bits — NaN, `+inf`, leftover pixel values, etc. The producing op
  is not required to scrub them.
- **Output side**: An op must produce correct values only inside its declared
  logical shape. Its own padding elements may be left as garbage for downstream
  consumers.

The practical consequence: when a kernel's math mixes valid lanes with padding
lanes (any reduction, any cross-lane op, any op that uses the padded values
later), it **must** neutralize the garbage before it contaminates the valid
lanes. "Neutralize" means replacing with the identity for the operation:
`0` for add/sum, `1` for multiply, `-inf` for max, `+inf` for min.

Everything in this document is a different flavor of neutralization and where
to put it.

---

## 2. Op Classification

| Class | Garbage behavior | Example ops | Needed? |
|---|---|---|---|
| **Independent-lane** | Each output lane depends only on the same lane of the input. Garbage in → garbage out, both in padding columns. | eltwise unary/binary, activations, tilize/untilize | **No neutralization needed.** |
| **Reduction-axis** | Output of a lane depends on many input lanes in the same tile. Garbage contaminates the answer. | reduce_h, reduce_w, mean, sum, max, min, argmax, layernorm, rmsnorm, softmax, groupnorm | **Mandatory neutralization on the reduction axis.** |
| **Structural / indexed** | Output values come from specific positions; boundaries matter. | matmul (K reduction), pool (window), concat, pad | **Mandatory on the reduced / bounded axis.** |
| **Shape-changing / ragged** | Output page has a byte count that is not a multiple of tile size. | untilize_with_unpadding, fill_pad, row-major ops | **Handled via byte-accurate NoC, not masking.** |

The rest of this document is about classes 2 and 3.

---

## 3. Approach Catalog

Eight distinct techniques are used in tree. They are listed roughly in order
of cost (highest first) and generality (broadest first).

### Approach A — Pre-op `fill_implicit_tile_padding` (the "undesirable" baseline)

**Where**: `ttnn/cpp/ttnn/operations/data_movement/fill_pad/fill_pad.cpp`
used by `generic_reductions.cpp:379`, `topk.cpp:352`, `gather.cpp:62`,
`sort.cpp:57`, `slice.cpp:179`, `pad.cpp:302`, and `unary_composite_op.cpp:197`
(for bfloat8).

**Mechanism**: Launches a whole extra op over the input tensor that overwrites
every tile's padding region with a chosen neutral value before the main op
runs. The main kernel then treats every tile as "full" — no masking inside.

```cpp
// ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp
float pad_value = get_pad_value(reduce_type);   // 0 / -inf / +inf
auto input_tensor = is_tiled
    ? ttnn::fill_implicit_tile_padding(input_tensor_arg, pad_value)
    : input_tensor_arg;
```

**Cost**:
- One full read + write of the input tensor through DRAM on every call.
- Doubles DRAM traffic on small ops, and for non-contiguous call patterns,
  breaks the ability to keep tiles resident between ops.
- Host-side op launch overhead.

**Why it exists**: It is the simplest thing that works, and it lets the hot
kernel stay shape-oblivious. Good for operations that call into unmodified
third-party kernels, e.g., `topk` and `sort`.

**Recommendation**: Treat as a last resort. Prefer any in-kernel technique.

---

### Approach B — Reader-side in-place pad (`fill_pad_tile` / `pad_last_ktile`)

**Where**: `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp`,
used by matmul readers, e.g.,
`reader_bmm_tile_layout_in0_sender_padding.cpp:260-276`.

**Mechanism**: The dataflow reader issues a normal `noc_async_read` for the
last tile, then, before `cb_push_back`, directly writes the neutral value
(`0` for matmul K dim) into the garbage faces of the tile in L1. The compute
kernel receives a tile whose padding is already zero — no compute-side work.

```cpp
// Reader kernel, inside the K-loop
if constexpr (in0_last_ktile_w > 0) {
    if ((block == num_blocks_inner_dim - 1) && (w == in0_block_w - 1)) {
        noc_async_read_barrier();
        pad_last_ktile<in0_data_format, in0_last_ktile_w>(write_ptr);
    }
}
```

`pad_last_ktile` ultimately calls `fill_pad_tile<T, unpadded_w, unpadded_h>`
(`pad_tile.hpp:128-158`), which is a face-granular compile-time-unrolled loop
that writes the fill value into right and bottom padding strips of each 16×16
face within the 32×32 tile.

**Cost**:
- One extra scalar pass over ≤ half a tile in L1, only on the last tile of a
  core's work, only when the relevant dim is unaligned.
- Zero extra DRAM traffic, zero extra compute-kernel logic.

**Strengths**:
- Completely hides the issue from the compute kernel — critical when compute
  is using a highly tuned LLK path (matmul inner loop) you don't want to
  touch.
- Template specialization means the scrubbing loop is unrolled and has zero
  branches in the hot path.

**Weaknesses**:
- Requires the fill dims to be known at compile time (they are, for a given
  program). For dynamic shapes you would switch to a runtime variant.
- Doubles the L1 residency window around the last tile by a tiny amount.

**Recommendation**: **Use this as the default for any op whose compute kernel
loops over tiles and accumulates into DST.** Matmul is the canonical example.
Anything whose computation naturally treats zero as identity (sum, matmul,
average with a known denominator) wants this.

---

### Approach C — Host-built mask tile in a CB, compute applies

**Where**:
- Softmax: `softmax_program_factory_attention_optimized.cpp:36-42`, applied
  in `softmax.cpp:221-230` via `add_tiles_bcast_rows`.
- Groupnorm: `groupnorm.cpp:299-326` via `mul_tiles`.

**Mechanism**: The host program factory builds a single 32×32 mask tile and
stuffs it into a dedicated CB (e.g., `cb_mask_padded`). The compute kernel
waits on that CB once, then applies it element-wise to the last reduction
tile of each row.

For softmax the mask contents are `0` in valid columns and `-inf` in padding
columns, so `add_tiles_bcast_rows(input, mask)` replaces padding values with
`-inf` before `exp`. For groupnorm the mask is `1` / `0` and combined with
`mul_tiles`.

```cpp
// softmax.cpp
if (mask_padded_data && wt == Wt - ndst && wt8 == ndst - 1) {
    reconfig_data_format(cb_in0, cb_mask_padded);
    add_bcast_rows_init_short(cb_in0, cb_mask_padded);
    cb_wait_front(cb_mask_padded, 1);
    add_tiles_bcast_rows(cb_in0, cb_mask_padded, wt8, 0, wt8);
}
```

**Cost**:
- One DRAM → L1 transfer of a 2 KiB tile at kernel launch (once per core).
- One extra `add_tiles` / `mul_tiles` call per row on the last width tile.

**Strengths**:
- Very clean compute-kernel logic — masking is a single SFPU instruction.
- The mask can encode arbitrary neutralization (−inf, +inf, 1, 0, scaled
  values) with no kernel changes.
- Works for per-op mask values that differ from the scrubbing neutral (e.g.,
  softmax wants `-inf`, not `0`, which rules out Approach B for softmax).

**Weaknesses**:
- The mask is computed host-side and shipped through DRAM. For tiny problems
  the shipment cost can dwarf the actual compute.
- Only one shape per program — re-using for different last-tile widths
  requires re-generating and re-dispatching.

**Recommendation**: Use when the neutralization value is non-zero (softmax,
min-reduction with +inf, weighted averages with 1.0) and the shape is fixed
per program.

---

### Approach D — Kernel-built mask tile via `generate_mask_h/w`

**Where**: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:213-290`, used in
all `moreh_*` ops: `moreh_mean_h.cpp`, `moreh_layer_norm_small_kernel.cpp`,
`moreh_norm_w`, `moreh_softmax_*`, `moreh_group_norm_*`.

**Mechanism**: The reader kernel calls `generate_mask_h(cb_mask_h, mask_h)`
or `generate_mask_w(cb_mask_w, mask_w)` once at the start. These helpers
build a 32×32 tile in L1 with `1.0` in the first `mask_h` / `mask_w` lanes and
`0.0` elsewhere, respecting the tile's 2×2 face layout — each of the four
16×16 subtiles is filled independently, so the helper correctly handles any
mask value in `[1, 32]`.

```cpp
// Reader side
#ifdef DO_MASK_H
    generate_mask_h(cb_id_mask_h, mask_h);
#endif
#ifdef DO_MASK_W
    generate_mask_w(cb_id_mask_w, mask_w);
#endif
```

```cpp
// Compute side
if (do_mask_h) {
    copy_tile(cb_input, 0, reduce_dst_idx);
    copy_tile(cb_mask_h, 0, mask_dst_idx);
    mask_tile_init();
    mask_tile(reduce_dst_idx, mask_dst_idx);   // DST[reduce] *= DST[mask]
}
```

`mask_tile` is an SFPU intrinsic that does element-wise multiply between two
DST tiles — cheaper than issuing full `mul_tiles`.

**Cost**:
- One short loop per core at launch to fill the mask tile (≈1 k float stores).
- One extra `copy_tile + mask_tile` per last reduction tile per row.

**Strengths**:
- No DRAM round-trip for the mask — built directly in L1 on the core that
  needs it.
- Mask dimensions can be runtime args: `mask_h` and `mask_w` are passed from
  the program factory, so the same kernel binary works for any unaligned
  shape.
- The separation (mask-h tile + mask-w tile, applied only where needed) means
  corner tiles are handled correctly in H, W, and both.

**Weaknesses**:
- Only neutralizes to `0` (the multiply identity), so reductions that need
  `-inf`/`+inf` masking require an extra step (or Approach C).
- Two extra DST registers in play during masking — tightens the DST budget
  on the compute kernel.

**Recommendation**: Use for reductions that want zero-neutral (sum, mean,
variance, L2 norm). This is the pattern most directly reusable when writing
a new reduction-style op from scratch.

---

### Approach E — SFPU-intrinsic mask (`mask_tile`, `mask_posinf_tile`, etc.)

**Where**: `mask_tile` is exposed by `compute_kernel_api` and used by the
moreh ops above. Variants in kernel APIs also include
`mask_posinf_tile_init()` and similar (grep for `mask_` in
`tt_metal/include/compute_kernel_api/mask.h` family).

**Mechanism**: Rather than building a full mask tile and multiplying, some
SFPU paths expose a single op that takes a position threshold and a fill
value, and writes the fill into positions beyond the threshold. This
collapses mask generation and application into one SFPU pass.

**Cost**: One SFPU pass over the tile, no L1 allocation for a mask tile.

**Strengths**: Minimal L1 footprint; fewer DST registers needed than D.

**Weaknesses**: Fewer kernels use this today, so you have to verify support
matrix for target arch and dtype. The threshold is per-axis (row or column),
which is fine for the typical last-tile case but not arbitrary patterns.

**Recommendation**: Prefer when available on the target arch. Fall back to
D if the SFPU primitive isn't exposed for the dtype.

---

### Approach F — Boundary-aware iteration (no mask at all)

**Where**: `ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_tile_layout.hpp:144-221`.

**Mechanism**: Initialize the running accumulator to the neutralization
identity (`-inf` for argmax's running max), then write a loop that only
visits logical positions. Garbage lanes are never loaded into the accumulator,
so masking is unnecessary.

```cpp
// argmax
for (uint32_t face_id = 0; face_id < 4; face_id++) {
    uint32_t rows_to_process = face_width;
    uint32_t cols_to_process = face_height;
    if (has_padding) {
        get_face_data_range(rows_to_process, cols_to_process,
                            tile_x, tile_y, face_id, ctx);
    }
    if (rows_to_process == 0 && cols_to_process == 0) continue;
    for (uint32_t row = 0; row < rows_to_process; row++)
        for (uint32_t col = 0; col < cols_to_process; col++)
            /* compare only valid element */
}
```

**Cost**: Zero — the loop just has shorter bounds on the last tile.

**Strengths**: Cleanest possible solution when the op is scalar-per-tile
(reading tile elements with explicit indexing) rather than tile-parallel
(SFPU/FPU vectorized). Perfect for argmax, topk-tile, scan.

**Weaknesses**: Incompatible with LLK tile-parallel primitives — if you want
`reduce_tile()` or `matmul_tiles()`, this approach doesn't work; the unit of
computation is a whole tile.

**Recommendation**: Use for kernels that naturally iterate element-by-element
anyway. Don't rewrite a vectorized kernel into this form.

---

### Approach G — Byte-accurate NoC transactions (no tiles, just bytes)

**Where**:
- `untilize_with_unpadding` writer (`writer_unary_unpad_dims_split_rows.cpp:70-95`)
- `untilize` writer (`writer_unary_stick_layout_split_rows_multi_core.cpp:73-80`)
- Generic row-major writers
- Permute writers (`writer_permute_interleaved_rm_blocked_generic.cpp:100-135`)

**Mechanism**: Don't think in tiles. Compute the logical byte count
(`elems * elem_size`) per row/stick and call `noc_async_write` with that
exact byte size. Leftover bytes in the L1 source buffer are simply not
written.

```cpp
// writer
uint32_t num_cols_to_write = std::min(
    num_unpadded_cols_per_input_block - processed,
    cols_remaining_in_output_block);
uint32_t bytes = num_cols_to_write * out_elem_size;
noc_async_write(l1_read_addr, dst_noc_addr, bytes);
```

**Cost**: Zero extra work — the writer just has tighter byte counts.

**Strengths**: The authoritative solution for **shape-changing** ops where
the output row is not a tile's worth of bytes. Matches what row-major ops
have to do anyway.

**Weaknesses**: Only applicable when you control the memory layout of the
output (row-major or a layout where ragged writes are legal). Cannot be used
to write a partial tile into a tile-layout tensor — you'd then need the
receiver to know the tile was partial.

**Recommendation**: Always the right answer for untilize-style ops and
anything writing row-major output. Never the right answer for something
handing off to a downstream tile-layout consumer.

---

### Approach H — Face-granular writes inside a tile (`fill_pad` kernel)

**Where**: `ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp:60-94`.

**Mechanism**: This is how `fill_implicit_tile_padding` itself is implemented.
The kernel iterates tile by tile, and for each tile that straddles the logical
boundary, it issues multiple `noc_async_write`s with byte counts that
respect face boundaries: `face_size - (col % face_size)` for ragged starts,
`face_size` for full-face writes.

```cpp
uint32_t elems_to_write =
    col % face_size == 0 ? face_size : face_size - (col % face_size);
uint32_t bytes_to_write = elems_to_write * element_size_bytes;
noc_async_write(src, dst + face_offset, bytes_to_write);
```

**Cost**: Multiple NoC transactions per straddle tile; only worth it
because the op runs once and then every downstream op can be shape-oblivious.

**Strengths**: Lets you set padding values from outside the hot path.

**Weaknesses**: Same as Approach A — you paid for a whole op launch.

**Recommendation**: Keep this primitive as a fallback. Use when the caller
guarantees the input will be consumed by many downstream ops that all want
the same neutralization, so the pad cost amortizes. Do not use for one-shot
ops.

---

## 4. Decision Tree for a New Op

```
Is the op independent-lane (eltwise, tilize, activation)?
  └── Yes → Do nothing. Garbage in, garbage out. (Class 1)
  └── No  → continue

Does the computation naturally treat 0 as identity?
  (sum, mean, matmul K-reduction, L2, variance)
  └── Yes:
        Is the neutralized data needed only inside this kernel's compute?
          └── Yes → Approach B (reader scrubs tile in-place via fill_pad_tile)
                   — default for matmul-shaped ops
          └── No  → Approach D (kernel-built mask + mask_tile SFPU)
                   — default for reductions that want dtype-flexible masking

Does the computation need a non-zero neutralization value?
  (softmax needs -inf, min needs +inf, weighted ops need 1.0)
  └── Yes → Approach C (host-built mask CB + add_tiles_bcast or mul_tiles)
           — prefer over D when mask value ≠ 0 and shape is static

Does the op iterate elements scalar-wise instead of tile-wise?
  (argmax, topk-tile, scan)
  └── Yes → Approach F (bound the loop; never load garbage)

Does the op produce a ragged output (row-major, untilize, partial last row)?
  └── Yes → Approach G (byte-accurate noc_async_write)

Is this a utility op whose whole purpose is to scrub padding?
  └── Yes → Approach H (face-granular writes inside the tile)
```

---

## 5. Actionable Steps for Implementing a New Reduction-Class Op

Concrete recipe for, say, an `rms_norm` over the last dim where `W % 32 != 0`:

### Step 1 — host side (program factory)
1. Compute `mask_w = W % TILE_WIDTH` (or `TILE_WIDTH` if `W % TILE_WIDTH == 0`).
2. Pass `mask_w` as a runtime arg to the reader kernel.
3. Reserve a CB `cb_mask_w` of 1 tile in L1 (2 KiB for bf16, 4 KiB for fp32).
4. Set `-DDO_MASK_W` if `W % TILE_WIDTH != 0`, so the defines path is dead
   code for aligned shapes and pays zero cost.

### Step 2 — reader kernel
```cpp
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    ...
#ifdef DO_MASK_W
    const uint32_t mask_w = get_arg_val<uint32_t>(...);
    generate_mask_w(cb_mask_w, mask_w);   // once per core
#endif
    // read input tiles as usual
}
```

### Step 3 — compute kernel
```cpp
#ifdef DO_MASK_W
    cb_wait_front(cb_mask_w, 1);
#endif

for (uint32_t wt = 0; wt < Wt; ++wt) {
    tile_regs_acquire();
    cb_wait_front(cb_input, 1);
    copy_tile_init_with_dt(cb_input);
    copy_tile(cb_input, 0, /*dst=*/0);

#ifdef DO_MASK_W
    if (wt == Wt - 1) {                   // last width tile only
        copy_tile_init_with_dt(cb_mask_w);
        copy_tile(cb_mask_w, 0, /*dst=*/1);
        mask_tile_init();
        mask_tile(/*dst=*/0, /*dst=*/1);  // DST[0] *= DST[1] → zeros garbage
    }
#endif
    // now square-and-accumulate DST[0] into your running-sum DST register
    ...
}
```

### Step 4 — precision
- Variance / sum reductions must run in `fp32_dest_acc_en = true` regardless
  of input dtype, so the masked zero contributes cleanly and doesn't introduce
  denormals.
- Mask tile dtype must match the compute input dtype — `generate_mask_w`
  takes a template parameter for this.

### Step 5 — validation checklist
- Unit-test with a deliberately dirty input: run `fill_implicit_tile_padding`
  with **NaN** into the padding region of the input tensor, then call your
  op, and verify the output is bit-identical to the aligned case. Any bug
  in the mask produces a NaN that propagates and fails the test loudly.
- Test `W = 33` (one full tile + one valid lane) and `W = 63` (just below
  two full tiles) — the two extreme corners of the mask logic.
- Test a shape where `H % 32 != 0` and `W % 32 != 0` to exercise both masks.

---

## 6. Common Pitfalls

1. **Masking only at the last tile but reading through it.** If you mask the
   input tile *after* it has already been accumulated elsewhere, garbage is
   already in the accumulator. Mask before any op that mixes lanes.

2. **Forgetting `pack_tile` reorders.** When compute writes the masked tile
   back to a CB for a later pass, the pack path must use the same dtype the
   mask was applied in. A silent bf16↔fp32 downconversion around the mask
   can turn `0.0 * garbage` into `+inf * garbage`.

3. **Using `mul_tiles` with `mask_value = 1`.** If your neutralization needs
   `1.0` (e.g., reciprocal reductions), the mask lanes must contain `1`, not
   `0`, in the padding region. `generate_mask_w` hardcodes `0` for padding.
   You need Approach C with a host-built mask in this case.

4. **Relying on "the producer op zeroed it".** Don't. The garbage contract
   explicitly allows the producer to leave anything. Mask inside your op
   even if you "know" the upstream is clean — that knowledge rots the first
   time someone reorders a graph pass.

5. **Asking `matmul_tiles` to handle unaligned K without scrubbing.** Matmul
   LLK primitives do not mask; they will happily multiply by whatever bits
   were in the last K tile. Approach B is not optional for matmul — it is
   the **only** correct kernel-level solution. `fill_implicit_tile_padding`
   also works but costs a full DRAM pass.

6. **Assuming row-major and tile-layout are interchangeable for the output.**
   If your op consumes non-aligned input and writes tile-layout output, the
   output is now itself non-aligned and downstream ops must follow the same
   garbage contract. Document this in the op's requirements.

7. **Building a mask on every call.** The mask tile depends only on
   `mask_h` / `mask_w` — build it once per core at kernel launch and reuse
   it across all iterations. The moreh pattern already does this; don't
   regress when adapting.

---

## 7. Tabular Summary

| Approach | Who builds mask | Where applied | Best for | DRAM cost | Compute cost |
|---|---|---|---|---|---|
| A `fill_implicit_tile_padding` | Host op pre-pass | Separate op | Legacy paths, sort/topk | Full tensor R+W | 0 |
| B `fill_pad_tile` in reader | Reader kernel | In-place L1 | Matmul K dim, sum-like reductions | 0 | Small L1 scribble |
| C Host-built mask CB | Host program | Compute `add_tiles` / `mul_tiles` | Softmax, min (`±inf` masks) | 1 tile | 1 binary op per last tile |
| D `generate_mask_h/w` + `mask_tile` | Reader kernel | Compute `mask_tile` | General 0-neutral reductions | 0 | 1 copy + 1 SFPU mask |
| E SFPU `mask_*_tile` variants | — | Compute SFPU | Arch-dependent shortcuts | 0 | 1 SFPU pass |
| F Bounded loop | — | Compute scalar code | Argmax, topk-per-tile, scan | 0 | 0 |
| G Byte-accurate NoC | — | Writer | Untilize, row-major output, permute | 0 | 0 |
| H Face-granular writes | — | Dedicated op | Utility scrubber (fill_pad) | 0 | Multiple NoC per tile |

---

## 8. Key Source References

- Helpers
  - `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp` — `fill_pad_tile`, `pad_last_ktile`
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` — `generate_mask_h`, `generate_mask_w`, `generate_mask_h_w`
  - `ttnn/cpp/ttnn/operations/data_movement/fill_pad/` — `fill_implicit_tile_padding` and its kernel
- Matmul (Approach B)
  - `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`
  - `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- Softmax (Approach C)
  - `ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory_attention_optimized.cpp`
  - `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp`
- Moreh layernorm/mean (Approach D)
  - `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp`
  - `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp`
- Argmax (Approach F)
  - `ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/argmax_tile_layout.hpp`
- Untilize / row-major (Approach G)
  - `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp`
  - `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core.cpp`
- Pooling (Approach B + init-value fill)
  - `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp`
- Concat (host-side composition, not kernel-level)
  - `ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp`
