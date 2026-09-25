---
name: quasar-test-coverage
description: Choose the sweep axes of a Quasar LLK functional test — input dimensions, MX formats, math fidelity — so each variant buys coverage. Use when adding or editing tests/python_tests/quasar/test_[op]_quasar.py, picking input_dimensions or a format list, adding dest bank-switch coverage, or reviewing a test whose variant count looks large.
user_invocable: true
---

# Quasar Test Coverage

## Goal

Spend variants where they change hardware behavior. A Quasar sweep that
multiplies out every dimension, format and fidelity costs `llk-build-quasar`
time without testing more of the pipeline, and routinely misses dest bank
switching entirely. Variant count is not coverage.

Three rules. Apply all three to any new or edited `tests/python_tests/quasar/test_[op]_quasar.py`. Perf tests have their own sweep rules: use `quasar-perf-test` instead.

## 1. Input dimensions

Two requirements, whatever scheme the test uses to express its shapes (#57544):

- Sweep a few representative shapes, not every shape that fits in dest. Examples of
  representative shapes :single tile, a full-width row, a full-height column, a balanced grid and an odd tile count cover the distinct dest layouts.
- Include at least one shape that **overflows dest**. `3 * max_tiles_in_dest`
  walks bank 0 -> 1 -> 0, so it covers the wrap back to the first bank and not
  just the initial switch.

The second is the one that gets missed. A sweep whose shapes all fit in one
dest section tests no bank switching however many variants it emits.

Where the shape axis is a plain input matrix,
`generate_reduced_input_dimensions` already returns that set and is the path of
least resistance. Tests with their own shape parametrization — matmul's
`mt`/`nt`/`kt` output grids, tilize/untilize row blocking — should keep their
scheme and add an overflow case to it rather than adopt the helper.

## 2. MX formats

MX formats do not belong in the regular format cross product of a math-kernel
test. On Quasar they live only in L1: the unpacker decodes every one of them to `
`Float16_b` in SrcA/SrcB, so an op fed an MX input runs the identical math
configuration as the `Float16_b` row beside it in the same sweep.
Multiplying MX through the whole sweep re-measures one unpacker descriptor
field against an unchanged pipeline.

Spot-test instead. `quasar_mx_smoke` is the helper for that, and one pair is
usually enough to keep the test's MX-conditional plumbing exercised — the
unpacker descriptor for the MX format, and whatever the test itself branches on
for MX:

```python
FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float16]) \
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
```

Add more pairs where there is a reason to — a format whose decode path the test
specifically exercises, or a bug being pinned down. The thing to avoid is the
unexamined full product, not a second pair.

Two cases are different:

- `test_pack_quasar` and `test_unpack_unary_operand_quasar` own the MX
  conversions themselves, so they carry the real cross product.
- `MxFp4_2x_A` / `MxFp4_2x_B` are genuine SrcA/SrcB register formats that change
  the math format and the per-tile MVMUL sequence, so they are swept wherever
  they apply (matmul, reduce GAPOOL).

The `quasar_mx_smoke` docstring in `helpers/param_config.py` has the full
reasoning.

## 3. Math fidelity

`Int8` and `Float16_b` inputs are **LoFi only**. Int8 has no mantissa
phases, and `Float16_b`'s 7-bit mantissa occupies the high 8 bits of the TF32
source, so the HiFi low-3 phases add nothing. MX is LoFi-only in **perf**
sweeps but still sweeps LoFi–HiFi4 functionally:

```python
def [op]_math_fidelities(format, *, is_perf=False):
    if format.input_format in (DataFormat.Int8, DataFormat.Float16_b) or (
        is_perf and format.input_format.is_mx_format()
    ):
        return [MathFidelity.LoFi]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]
```

This applies only where fidelity changes the MVMUL sequence — matmul and
eltwise multiplies. Add and sub ignore fidelity, so they are LoFi-only for
every format.

## Kernel side: dimensions alone do not switch banks

Rule 1 only buys bank coverage if the kernel is blocked. An input larger than
dest must be consumed one dest section at a time, releasing the section at the
*block* boundary — otherwise the overflow shape either overruns dest or never
advances past the first bank.

Where the shape is a plain input matrix,
`get_num_blocks_and_num_tiles_in_block` derives the split; a test with its own
shape scheme computes its own block count. Either way the kernel loop is the
same:

```c
for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
{
    for (std::uint32_t i = 0; i < NUM_TILES_IN_BLOCK; ++i)
    {
        // math: _llk_math_eltwise_*_(i, ...)   pack: _llk_pack_(i, block * NUM_TILES_IN_BLOCK + i, ...)
    }
    // release the dest section here, so the next block lands in the other bank
}
```

Either synchronization scheme works, as long as the handover is per block and
not once for the whole kernel. Both take and release the section:

| | takes the section | releases it |
|---|---|---|
| **dvalid** | implicit; the hardware chain stalls the thread. Set up once per thread at init with `set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::{UNPACK,FPU,PACK}>()` | math `_llk_math_set_dvalid_`, pack `_llk_pack_dest_dvalid_section_done_`, unpack `_llk_unpack_dest_dvalid_section_done_` |
| **semaphore** | math `_llk_math_wait_for_dest_available_`, pack `_llk_packer_wait_for_math_done_` | math `_llk_math_dest_section_done_`, pack `_llk_pack_dest_semaphore_section_done_` |

The asymmetry matters when reading a kernel: under dvalid there is no acquire
call to look for, so the only visible per-block statement is the release. Under
semaphores both halves are explicit, which is what
`sources/quasar/semaphore_sync_quasar_test.cpp` shows, and it is the shape the
WH/BH tests use.

The semaphore scheme needs `_llk_math_pack_sync_init_` to seed MATH_PACK first.
That seeds `num_sem = 2` under `SyncHalf` and `1` under `SyncFull` — the count
is what makes a second bank available to claim.

**Do not mix the two schemes** in one kernel; `llk_pack_common.h:134` says so
explicitly.

Under `SyncHalf` the *release* is the bank flip:
`_llk_math_dest_section_done_` (and its dvalid counterpart) calls
`_update_dest_register_offset_` and re-bases the section. Skip it and every
block lands in bank 0.

`NUM_TILES_IN_BLOCK` means tiles per dest section, and means that in every
test. When an op folds several input tiles into one dest tile (`acc_to_dest`),
carry the split in `INPUT_NUM_TILES_IN_BLOCK` / `OUTPUT_NUM_TILES_IN_BLOCK` —
the `NUM_TILES_IN_BLOCK` parameter already emits all three — rather than
overloading the plain name with the accumulation depth. See
`sources/eltwise_binary_test.cpp` and
`sources/quasar/eltwise_binary_reuse_dest_quasar_test.cpp`.

Blocks are expected to be uniform — `get_num_blocks_and_num_tiles_in_block`
raises when the tiles do not divide evenly, so a bad shape fails at parametrize
time rather than silently running a short block. That guarantee is why the
block loops need no partial-tail clamp; a test computing its own split owes the
same check.

## Verify

Count variants before and after, without a device — the combination lists are
module-level:

```bash
CHIP_ARCH=quasar python -c "
from quasar.test_[op]_quasar import ALL_[OP]_COMBINATIONS as C
print(f'{len(C):,}')"
```

Run from `tests/python_tests/`. `pytest --collect-only` boots the device in
`pytest_configure` and will not work on a machine without one.

State the before/after counts in the PR description; all three source PRs do.

## Checklist

- [ ] A few representative shapes, at least one of which overflows dest
- [ ] Kernel loops `NUM_BLOCKS`, releasing the dest section at the block boundary
- [ ] MX spot-tested rather than swept, unless this is the pack or unpack test
- [ ] Fidelity LoFi-only for `Int8` / `Float16_b` (and MX when `is_perf`)
- [ ] Variant count measured before and after, and quoted in the PR
