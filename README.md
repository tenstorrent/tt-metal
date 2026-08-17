# tt-llk test docs for the blaze `experimental/` promotions

Documentation-only branch, so these can be read from anywhere without checking out the work
branch. **No code here** — see below for where that lives.

This branch is now the *only* home for them: the two original documents were briefly checked into
`tt_metal/tt-llk/tests/` alongside the tests and were removed on review, on the grounds that
a point-in-time plan with effort estimates and in-flight PR cross-references is a different
genre from the durable `tests/*.md` usage guides. The durable hardware findings they carried
were moved into code comments next to what they constrain before the deletion.

| File | What it is |
|---|---|
| [`REMAINING_WORK.md`](REMAINING_WORK.md) | **Start here.** Every outstanding item in one place — functional test gaps, the three product issues that need an owner rather than a test, the review comments that landed in `main` before they could be fixed, and PR mechanics — each with a plan and a suggested order. |
| [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) | **Background detail.** Superseded as a to-do list by `REMAINING_WORK.md`; keep it for the forensic record on the two attempted-and-reverted items (§3, §9), which is too long to inline. The 4 remaining items, what is uncovered, two attempts that were reverted and why, open questions for the PR authors, and an appendix of everything learned (harness gotchas, LLK facts established by measurement, method notes). |
| [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md) | **Closed record.** The 5 landed test suites with results, the 4 green verification items, and the 8 findings that came out of building them. |

## Covers

tt-metal PRs **#52747, #52745, #52713, #52727, #52709** — the blaze → tt-metal
`experimental/` LLK promotions.

## Where the code is

Branch **`ldjurovic/llk-tests-blaze-promotions`** (PR
[#53130](https://github.com/tenstorrent/tt-metal/pull/53130)). Eight files added under
`tt_metal/tt-llk/tests/`:

- `sources/sfpu_add_rsqrt_test.cpp` + `python_tests/test_sfpu_add_rsqrt.py`
- `sources/custom_mm_uninit_restore_test.cpp` + `python_tests/test_custom_mm_uninit_restore.py`
- `sources/sort_headers_coexist_test.cpp` + `python_tests/test_sort_headers_coexist.py`
- `sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp` + `python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py`

plus an extension to `sources/sfpu_sampling_test.cpp` / `python_tests/test_sfpu_sampling.py`,
and a build fix to two LLK headers that did not compile under tt-llk at all (Finding 5).

**#52709 merged on 2026-08-14** and the branch has been rebased onto main, so that family's
headers no longer appear in the diff. #52713 and #52727 are still open, so the branch still
carries their promotion payload — 32 files, byte-identical to
`pmilenkovic/promote-top32-rm` and `pmilenkovic/promote-custom-mm`. Rebase again once those
land and the PR reduces to the test files.

## Status at time of writing

- Verification tier V1–V4: **4 of 4 green**
- New test items: **5 landed** (137 new variants passing, 1 xfailed), **2 attempted and
  reverted**, **2 not started**
- **Three things need an owner:**
  - the `dense_packing` W-stride constants in `custom_mm.h` / `compressed_custom_mm.h` are
    hardcoded for a 16-bit pack source (item **D1**, detail as Finding 2);
  - the `eltwise_mul_scalar` HiFi workaround's stated mechanism does not survive reading the
    code it calls (Finding 7, for the #52709 author);
  - a pre-existing `topk_xl` -> `eltwise_binary` reconfig escape, unrelated to the
    promotions but liable to confuse the remaining `top32_rm` work (Finding 8).

All results measured on Blackhole p100a. Last updated 2026-08-17.
