# PR #53130 — drafted replies

> **Status: drafted, NOT posted.** Written 2026-08-18 by a session with no GitHub write access,
> so every reply below is copy-paste ready and none of it is on the PR. SHAs are post-rebase
> (the branch was rebased onto `main` `b62ff4a6af1` the same day). Committed here rather than
> left in a scratchpad so the work is not lost — see item E in
> [`REMAINING_WORK.md`](REMAINING_WORK.md).
>
> Check each thread is still open before pasting: some may have been answered since.

Base URL for each: `https://github.com/tenstorrent/tt-metal/pull/53130#discussion_r<id>`

Eleven open comments. **6 fixed in this pass**, **3 were already fixed by earlier commits on the
branch** (reply + resolve), **2 are reply-only**.

---

## Fixed in this pass

### r3796908960 — `test_custom_mm_uninit_restore.py:210` — imperative `pytest.xfail()`

> Imperative `pytest.xfail()` raises immediately and aborts the body […] it can never report XPASS.

Correct, and this file was the only place in the suite doing it — `test_sfpu_unary.py`,
`test_sfpu_binary.py` and `test_sfpu_reduce.py` all use the marker form. Fixed as suggested:
`request` added to the signature and

```python
request.node.add_marker(pytest.mark.xfail(reason=_DENSE_FP32_XFAIL, strict=False))
```

so the body actually builds and runs, and the variant flips to XPASS once the W-stride constants
become format-aware — which is what the comment above it promised. Confirmed the local
`parametrize()` helper only parametrizes its own named kwargs, so the `request` fixture injects
cleanly.

---

### r3796909504 — `llk_unpack_A_rmsnorm.h:30` — dead `unpack_{src,dst}_format`

> deleting both end-to-end is 2 files […] prefer that over suppressing the warning.

Agreed, done — the `[[maybe_unused]]` suppressions are gone and both parameters are deleted
end-to-end. It was **3** files, not 2: besides the LLK lib header and the
`llk_unpack_A_rmsnorm_api.h` shim, the new tt-llk driver
`tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp:75` was also passing
`formats.unpack_A_src` / `formats.unpack_A_dst` positionally.

The public `llk_unpack_A_rmsnorm_init` signature is unchanged (it never exposed them), so nothing
above layer 1 moves — `hw/inc/api/compute/experimental/rmsnorm.h` still calls it with 3 args and
there are no other consumers in the tree.

The commented-out `disable_src_zero_flag` TODO in `_llk_unpack_A_rmsnorm_init_` referenced
`unpack_dst_format`, so it now carries a line saying reviving it needs the dst format plumbed back
in.

---

### r3796909555 — `sort_headers_coexist_test.cpp:120` — bare `0`

Fixed: `set_dst_write_addr_offset(0 /*addr*/)`, matching the helper's parameter name and the
annotated-literal convention used elsewhere in the file.

---

### r3796119145 (Copilot) — `llk_unpack_AB_compressed_custom_mm.h:250` — OOB remainder read

> When `kt_dim * ct_dim` is divisible by 10, `rem_iters` is zero […] this still reads
> `meta_ptr[full_iters]` one word past the buffer.

Confirmed and fixed. `full_iters = (kt_dim * ct_dim) / 10` and `rem_iters = (kt_dim * ct_dim) % 10`,
so the buffer holds `ceil(kt_dim * ct_dim / 10)` words and the load is past the end whenever the
product is a multiple of 10 — reachable within the documented ranges (`kt_dim` even 2..256,
`ct_dim` 1..16), e.g. `kt_dim=10, ct_dim=1`. The load and the remainder loop are now inside
`if (rem_iters != 0)`.

---

### r3796909307 — `test_sfpu_add_rsqrt.py:268` — sign of an undefined value

> this one pins the sign of a value the implementation does not define […] Asserting that the two
> `fast_approx` builds differ would pin the same intent.

Fair, and taken. The `fast_approx=True` branch was asserting `negative_lanes > 0` on the *unguarded*
result, which as you say falls out of the current LUT seed rather than from anything
`ckernel_sfpu_sqrt.h` states.

Restructured along the lines you suggested: `fast_approx` is no longer a parametrize axis; the test
now builds both variants and asserts

1. `FAST_APPROX=false` leaves no negative lane — guard-specified, and the load-bearing half; and
2. the two builds differ somewhere — which is the actual intent ("the one case that tells
   FAST_APPROX=true from FAST_APPROX=false") and holds regardless of what the unguarded body
   returns.

Both variants are `prepare()`d before either runs, following `test_topk_xl_rebuild_ascending` —
under `--compile-producer` `run()` skips as soon as the first variant is built, so the second would
otherwise never emit its ELF. The "sign guarantee" wording is gone; the docstring now records the
measured behaviour as an observation and says explicitly why it is not asserted.

---

### r3796909124 — `custom_mm.h:267` — `restore_tile_pack_mop` vs `pack_block_contiguous_uninit`

Every factual claim here checks out, and I verified each one:

- the body is byte-identical to `pack_block_contiguous_uninit()` in
  `experimental/pack_block_uninit.h` (which is already on main);
- `custom_mm_block_init_short` (`matmul.hpp:131`, `kn_sliced_matmul.hpp:114`) programs no pack MOP
  at all, so on the mid-kernel path there is nothing to restore;
- full `custom_mm_block_init` derives pack geometry from `out_cb_id` via `llk_pack_init`, so on a
  non-32x32 output CB this clobbers rather than restores;
- it leaves the `set_packer_strides`/`SETADCXX` state untouched;
- and no caller passes `restore_tile_pack_mop=true` — every site in the tree calls
  `custom_mm_block_uninit<dense_packing>()` / `compressed_custom_mm_block_uninit<...>()`.

**Keeping the flag but fixing the documentation**, rather than removing it: it exists for the
tt-blaze fused chains that call neither `pack_block_contiguous_init` nor its uninit and still expect
the packer left at Default on op exit. The doc table row and the body comment no longer say
"restore" — they now state that it *installs* fixed 32x32/4-face geometry, spell out all four
caveats above, and point at `pack_block_contiguous_uninit()` as the correct pairing for a caller
whose MOP was replaced by `pack_block_contiguous_init`. Same for the copy in
`compressed_custom_mm.h`.

Happy to drop the flag instead if you'd rather not ship the redundant surface — say so and I'll
remove it from both headers and strip the MOP-restore half of the new test (the `dense_packing`
W-stride coverage stands on its own either way).

---

## Already fixed by earlier commits on the branch

### r3783240232 — `test_sfpu_sampling.py:62` — reviewdog import order

Already applied in `f628a3de0be` ("Apply isort/black formatting to test_sfpu_sampling.py") — the
import block reads `SAMPLING_PRGM0_HAZARD, SFPU_UNARY_SCALAR` as suggested. Resolving.

### r3783813739 — `compressed_custom_mm.h:62` — `split_acc`/`finalize` doc tables

Already applied in `795ff816b1f` — every affected table row on this family now carries
"NOT FORWARDED on this family: accepted for call-site compatibility, but the LLK is always
instantiated with `<flag>=false` (see the NOTE in the body). Sibling custom_mm.h does forward it."
Resolving.

### r3783809559 — `sort_headers_coexist_test.cpp:125` — runtime half has no signal

Addressed in `c6c52b16063`, taking the first of your two options but keeping the run:

- the Dst-row offset sweep is gone; the test now sweeps `dest_acc`, which is the one axis that
  changes what the combined TU actually builds;
- the pytest assertion message no longer claims anything about the helper's offset — it says the
  datacopy in the combined TU was corrupted, and points at the docstring;
- the module docstring and the file header both state that this is a compile-time assertion, that
  the datacopy reprograms `DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` before touching DEST so any
  offset the helper left is discarded, and that the helper is covered in its real context by the
  topk_xl and deepseek_top32_rm kernels.

The run is deliberately kept rather than dropping to compile-only: it shows the combined TU
*executes* (a coexistence regression that wedged the math thread would surface as a hang or a
corrupted datacopy), and the helper is called so it is code-generated rather than only parsed.
That is a weaker claim than validating the offset, and it is now the only claim made.

---

## Reply-only

### r3796119098 (Copilot) — "Exercise the real uninit API"

> This block is labeled as the function under test, but it reimplements the two statements instead
> of invoking either changed Compute API entry point.

This is a real limitation and a deliberate one — it is called out in the module docstring of
`test_custom_mm_uninit_restore.py` and in the header of
`sources/custom_mm_uninit_restore_test.cpp`: a tt-llk test cannot include
`tt_metal/hw/inc/api/compute`, so the driver replicates the shared body. The docstring already
states the consequence in the same terms you do — the test pins the behaviour the two uninits
currently share, and a future divergence between them is exactly what it cannot catch, which needs
a test on the metal side that calls the real entry points.

Filing that metal-side test as follow-up rather than doing it here: it belongs in the
tt_metal compute-kernel test suite, not in this tt-llk PR, and the value this test does add
(pinning the two packer-state restores against LLK-level regressions, and catching the
`dense_packing` FP32 W-stride defect) does not depend on it.

Since then, `096ff04e219` adds `test_custom_mm_uninit_parity.py`, a device-free static gate
that closes the specific risk you are pointing at without the metal-side test: it fails if the
two compute-API uninit bodies stop being identical, and if the driver's replicated W-stride
expressions stop matching the headers'. Both mutation-checked. It does not make the test
"exercise the real uninit API" — a text match cannot say the functions work — but divergence
now fails loudly instead of silently, which was the part that could rot unnoticed.

### r3796119166 (Copilot) — PR metadata

Correct — the body is still the template and the title understates the diff. Note the title is also
still tagged `[do not review]`.

**Suggested title**

```
[LLK] Promote the Blackhole custom_mm / top32_rm / rmsnorm blaze kernels and add tt-llk coverage
```

**Suggested Summary**

> Promotes four Blackhole kernel families out of the `deepseek_v3_b1` demo's private
> `kernel_includes/` tree into the real ones, and adds the tt-llk regression coverage they had none
> of.
>
> Promoted (moves, so the diff is mostly renames): `custom_mm`, `compressed_custom_mm`,
> `deepseek_top32_rm`, and the rmsnorm bcast-scalar dest-reuse family — LLK lib into
> `tt-llk/tt_llk_blackhole/llk_lib/experimental/`, LLK API into
> `hw/ckernels/blackhole/metal/llk_api/experimental/`, and the two Compute API headers into
> `hw/inc/api/compute/experimental/`, packaged via `HW_JIT_API_HEADERS` in `hw/sources.cmake`. The
> DeepSeek unified kernels and the `top32_rm_dev_compute*` test kernels are migrated to the promoted
> paths.
>
> Also extracts the `set_dst_write_addr_offset` helper shared by `ckernel_sfpu_topk_xl.h` and
> `ckernel_sfpu_deepseek_top32_rm.h` into
> `sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h`, so a math TU including both no
> longer fails with a redefinition error.
>
> New tt-llk tests: `custom_mm` / `compressed_custom_mm` `_block_uninit` packer-state restore,
> rmsnorm bcast-scalar dest-reuse, the fused `add_rsqrt` SFPU functor, and sort-header coexistence.
> `test_sfpu_sampling.py` and `test_matmul_custom_compressed.py` are extended for the promoted
> paths.

**Suggested Notes for reviewers**

> - The promotion is a move: most of the diff is renames, and the LLK bodies are carried verbatim
>   from the demo tree. The behavioural deltas are called out in the header comments.
> - `test_custom_mm_uninit_restore.py` xfails `dense_packing` on a 32-bit pack source: the W-stride
>   constants in both compute-API headers are hardcoded `* 2` (16-bit pack source), so the uninit
>   does not restore what `_llk_pack_init_` programmed. Pre-existing, but the promotion ships it in
>   packaged metalium — see the comment above `_DENSE_FP32_XFAIL`.
> - `split_acc` and `finalize` are accepted but NOT forwarded on the compressed family; the doc
>   tables and body NOTEs say so.
> - `restore_tile_pack_mop` on the two `_block_uninit`s installs fixed 32x32/4-face pack geometry
>   rather than restoring a caller's — documented on the flag, with a pointer to
>   `pack_block_contiguous_uninit()` as the right pairing when the MOP came from
>   `pack_block_contiguous_init`.
