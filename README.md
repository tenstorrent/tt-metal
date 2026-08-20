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
| [`REMAINING_WORK.md`](REMAINING_WORK.md) | **Start here.** Every outstanding item in one place — functional test gaps, the five product issues that need an owner rather than a test, and PR mechanics — each with a plan and a suggested order. Closed items are **not** listed there; they move to the DONE document, so anything still in it is still to do. |
| [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) | **Background detail, and the defect dossiers (§13).** §13 is the one to read before picking up any bug: one entry per open defect with mechanism, measurements, blast radius, reproduction, fix options and tripwire — including two fixes that were tried and reverted. The rest of the document is the forensic record: §3 and §9 on the two attempted-and-reverted test items, the per-PR inventories, open questions for the PR authors, and an appendix of everything learned (harness gotchas, LLK facts established by measurement, method notes). It is **not** a to-do list — `REMAINING_WORK.md` is. |
| [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md) | **Closed record.** The 5 landed test suites with results, the 4 green verification items, and the 8 findings that came out of building them. |

## Covers

tt-metal PRs **#52747, #52745, #52713, #52727, #52709** — the blaze → tt-metal
`experimental/` LLK promotions.

## Where the code is

Branch **`ldjurovic/llk-tests-blaze-promotions`** (PR
[#53130](https://github.com/tenstorrent/tt-metal/pull/53130)). Fourteen files added under
`tt_metal/tt-llk/tests/`, each a `sources/*_test.cpp` + `python_tests/test_*.py` pair unless
noted:

- `sfpu_add_rsqrt` · `custom_mm_uninit_restore` · `sort_headers_coexist` ·
  `rmsnorm_bcast_scalar_dest_reuse` (the original five, with the `sfpu_sampling` extension)
- `set_dst_write_addr_offset` · `mul_reduce_scalar_reenter` ·
  `test_custom_mm_uninit_parity.py` (python-only, static) — 2026-08-18
- `matmul_custom_mm` — 2026-08-19
- `top32_rm` — 2026-08-20

plus extensions to `sfpu_sampling_test.cpp` / `test_sfpu_sampling.py` and
`test_matmul_custom_compressed.py`, and a build fix to **three** LLK headers that did not
compile under tt-llk at all (Findings 5 and 17).

**All three promotion PRs have now merged** — #52709 on 2026-08-14, #52727 on 2026-08-18,
#52713 by 2026-08-20 — and the branch was rebased onto main on 2026-08-20 (53 commits, six
promotion-payload commits dropped), so the PR diff is test files plus LLK-side cleanups only.

## Status at time of writing

- Verification tier V1–V4: **4 of 4 green**
- New test items: **12 landed** (355 new variants passing, 14 xfailed), **1 attempted and
  reverted** (now diagnosed as Finding 9), **1 not started** (A5, gated on C2)
- **Five things need an owner:**
  - the `dense_packing` W-stride constants in `custom_mm.h` / `compressed_custom_mm.h` are
    hardcoded for a 16-bit pack source (item **C1**, detail as Finding 2);
  - the `eltwise_mul_scalar` HiFi workaround's stated mechanism does not survive reading the
    code it calls (**C2** / Finding 7, for the #52709 author);
  - `mul_reduce_scalar` re-entry needs a DEST-section boundary (**C4** / Finding 9), a located
    defect in a shipping op;
  - a pre-existing `topk_xl` -> `eltwise_binary` reconfig escape, unrelated to the
    promotions but liable to confuse work in the sort area (**C3** / Finding 8);
  - `top32_rm`'s 32-bit unpack branch does not clear its tile, so a partially-filled chunk
    sorts against stale Dest (**C6** / Finding 19), pinned by an xfail.

All results measured on Blackhole p100a. Last updated 2026-08-20.
