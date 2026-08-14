# tt-llk test docs for the blaze `experimental/` promotions

Documentation-only branch, so these two files can be read from anywhere without checking
out the work branch. **No code here** — see below for where that lives.

| File | What it is |
|---|---|
| [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) | **Open work.** The 5 remaining items, what is uncovered, two attempts that were reverted and why, open questions for the PR authors, and an appendix of everything learned (harness gotchas, LLK facts established by measurement, method notes). |
| [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md) | **Closed record.** The 4 landed test suites with results, the 4 green verification items, and the 4 findings that came out of building them. |

## Covers

tt-metal PRs **#52747, #52745, #52713, #52727, #52709** — the blaze → tt-metal
`experimental/` LLK promotions.

## Where the code is

Branch **`ldjurovic/llk-tests-blaze-promotions`**, which merges #52709 + #52713 + #52727
onto main so the promoted headers exist to compile against. Six files added under
`tt_metal/tt-llk/tests/`:

- `sources/sfpu_add_rsqrt_test.cpp` + `python_tests/test_sfpu_add_rsqrt.py`
- `sources/custom_mm_uninit_restore_test.cpp` + `python_tests/test_custom_mm_uninit_restore.py`
- `sources/sort_headers_coexist_test.cpp` + `python_tests/test_sort_headers_coexist.py`

plus an extension to `sources/sfpu_sampling_test.cpp` / `python_tests/test_sfpu_sampling.py`.

That branch carries the three PR merge commits, so it is **not reviewable as-is** — the test
commits touch only `tt_metal/tt-llk/tests/` and should be rebased onto main once the PRs
land.

## Status at time of writing

- Verification tier V1–V4: **4 of 4 green**
- New test items: **4 landed** (87 new variants passing, 2 xfailed), **2 attempted and
  reverted**, **3 not started**
- **1 defect needs an owner decision** — the `dense_packing` W-stride constants in
  `custom_mm.h` / `compressed_custom_mm.h` are hardcoded for a 16-bit pack source. Carried
  as item **D1** in the open-work document, with the detail as Finding 2 in the done
  document.

All results measured on Blackhole p100a.
