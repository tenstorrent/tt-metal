# Quasar IDMA Examples

Two example kernels demonstrating the Quasar IDMA engine for local L1-to-L1 copies.

## Kernels

### `idma_basic_example.cpp` (test ID 910)

Linear copy of `num_elements × elem_size` bytes from `src` to `dst` in a single IDMA transaction using direct address registers (no addrgen).

**Parameters:** `src_base`, `dst_base` (compile-time args)

**Sequence:**
1. Configure `cmdbuf_0` for IDMA copy (`idma_setup_as_copy_cmdbuf_0`)
2. Set src/dst addresses via `set_src_cmdbuf_0` / `set_dest_cmdbuf_0`
3. Set transfer length and issue one transaction
4. Wait for IDMA ack

**Host verification:** host writes a known pattern to `src`, reads back `dst` after kernel completes, and checks `dst == src`.

---

### `idma_1d_strided_example.cpp` (test ID 911)

Strided read of `num_elements` elements from `src` (every `src_stride` bytes) into a linear `dst`. One IDMA transaction per element, driven by the address generator inner loop.

**Parameters:** `src_base`, `dst_base` (compile-time args)

**Fixed constants:**
| Symbol | Value | Meaning |
|--------|-------|---------|
| `num_elements` | 10 | number of elements to copy |
| `elem_size` | 8 B | size of each element |
| `src_stride` | 16 B | stride between consecutive src elements (every other 8 B slot) |

**Address pattern:**

| element | src address | dst address |
|---------|-------------|-------------|
| 0 | `src_base + 0×16` | `dst_base + 0×8` |
| 1 | `src_base + 1×16` | `dst_base + 1×8` |
| … | … | … |
| 9 | `src_base + 9×16` | `dst_base + 9×8` |

**Sequence:**
1. Configure addrgen src inner loop (stride=16, end=160) and dst inner loop (stride=8, end=81)
2. Configure `cmdbuf_0` for IDMA copy, `len = elem_size`
3. Loop `num_elements` times: `push_both_addrgen_0()` + `issue_cmdbuf_0()`
4. Wait for IDMA ack

**Host verification:** host constructs expected output by applying the same strided pattern to the source data, then compares `dst` after kernel completes.

---

## Test Matrix

| Test | Kernel | Description |
|------|--------|-------------|
| `IDMA_Basic` (ID 910) | `idma_basic_example.cpp` | 16×8 B linear copy, single transaction |
| `IDMA_1D_Strided` (ID 911) | `idma_1d_strided_example.cpp` | 10 elements, src_stride=16 B, dst linear |

## Running

```bash
TT_METAL_SIMULATOR=1 TT_METAL_DPRINT_CORES=0,0 \
  ./build/test/tt_metal/unit_tests_data_movement --gtest_filter="*QuasarIdmaOps*"
```

## Dispatch Mode

Uses fast dispatch (Mesh Device API) via `MeshDeviceSingleCardFixture`. Tests are skipped automatically on non-Quasar architectures and when `TT_METAL_SIMULATOR` is not set.
