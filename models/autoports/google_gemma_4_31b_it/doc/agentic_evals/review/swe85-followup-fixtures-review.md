# SWE-bench follow-up fixture review

## Scope and disposition

Independent CPU-only review of the frozen cap-extension and contraction-short-control fixtures before follow-up diagnostic admission. This is fixture validity review, not anomaly resolution or stage signoff. No TTNN imports, device work, server starts, fixture changes, or generated command execution were performed.

**Disposition:** no concrete fixture-integrity or cost-estimate error found. The cap-extension set contains 25 served and 33 full single-trial cases; the contraction-short set contains the three requested long-full-persistent cases with three seeds each. Candidate artifacts remain candidates pending coordinator admission. Two status/reproducibility notes are non-blocking: cap metadata still refers to remaining review, and reproducing contraction preparation requires passing the source roster's expected count (36) to the generator, whose default is 21.

## Cap-extension checks

For every candidate, checked the source fixture and source result/adjudication chain against the candidate fixture metadata and preparation records. Candidate prompt IDs, continuations, sampling parameters, chosen seed, and mode match the original source; only the intended free-run cap changes from 256 to 512. Every row has one free-run trial and an empty branch list. The source sample has 256 generated IDs and the coordinator judged it inconclusive. Fifty-seven source rows explicitly say `at_cap: true`; the remaining full Sympy13551 seed 43 row omits that boolean but its cap note says it reached the 256-token cap without a closing token and the generated-ID count is 256. This is consistent with a genuinely capped, inconclusive continuation.

Manifest hashes in host validation, cost estimate, and runner validation match the candidate manifests. Prompt-token sums independently recompute to 1,106,308 served and 1,514,818 full. The stored runner validation records CPU `--validate-only` success for 25/33/3 cases without importing TTNN; this review inspected that evidence but did not rerun the script.

Recomputed each cap-extension estimate as `wall - decode_wall + decode_wall * (new_cap - 1) / decode_tokens`, summing the samples and adding one measured load per mode. All stored totals and intervals match. Served estimate: 0.249718 h, interval 0.174803–0.374578 h. Full estimate: 0.541205 h, interval 0.378844–0.811808 h. These are bounded diagnostic estimates, not serving-performance claims.

The preparation, host-validation, and cost-estimate status text that says to refresh after remaining full review / independent review is stale now that the review is complete. It does not invalidate the fixtures; the candidate-only, not-admitted status remains accurate.

## Contraction-short checks

The three candidates are Matplotlib86, canonicalprefix91, and Sphinx10323. For each, verified that the new prompt is exactly the last 512 token IDs of the original prompt; continuations and all original sampling/configuration fields are retained; selected chosen-TF text is unchanged; and only the intended optional suffix-control metadata is added. Seeds are exactly 42/43/44, cap is 512, there are three trials, and the sole branch is `chosen`. The selected seed-43 full-run adjudications identify the native repeated-contraction motifs for all three candidates. The fixture/preparation contract gates any HF comparison on reproducing the same motif with the matched short full-TT input first, and preserves non-reproductions.

Recomputed the six served/full cost rows as the stored conservative chosen-TF wall allowance plus, for each of three runs, `(sample_wall - decode_wall) * 512 / original_prompt_tokens + decode_wall * 511 / decode_tokens`, then adding one load per mode. All stored totals and intervals match. Served estimate: 0.097955 h, interval 0.048977–0.166523 h. Full estimate: 0.130667 h, interval 0.065333–0.222133 h. These are diagnostic estimates.

The contraction preparation prerequisite to complete the remaining long-full review is now satisfied. For reproducibility, the suffix generator's default `--expected-count` is 21 while the source roster used here contains 36 entries; the preparation record does not retain the invocation. A replay must explicitly use `--expected-count 36`. This does not alter the frozen fixtures' validity.

## Follow-up command draft

Inspected the pending `setup/swe85-followup-execution/admission-draft.json` (SHA-256 `47e91d363855e1644e1c55998a0951a39790cb3feaee46300aebac9959d0fe8e`). It sequences contraction served/full then cap served/full, uses the intended candidate manifests and precision build files, and pins the reviewed fixture/script/config hashes. The full contraction command uses `--compare-dir` for the matching served manifest; cap-full omits it because the served and full candidate manifests differ. The draft remains pending and depends on the six-caret full-completion record; this review does not admit or launch it.

## Reviewed file hashes (SHA-256)

Cap manifests:

- `setup/swe85-cap-extensions/served-candidate-cases.json`: `4cb883c6fe971328c3ba49179d449579169a8335ce83f3b1b12a9c148319786d`
- `setup/swe85-cap-extensions/full-candidate-cases.json`: `f749b36d171fdaabbc6b96bef1f4f933a59ef8ea2a5e7b2d96b525b0b3c30a12`
- `setup/swe85-cap-extensions/preparation.json`: `1211de20883579efbe1628c5966a2756a4d9863e8a6b7f80e74555ccfd5f02fe`
- `setup/swe85-cap-extensions/host-validation.json`: `dc708be93dca1d2c8b457289534d21917b7a5f05179d7913f0a7f4ccd8efab81`
- `setup/swe85-cap-extensions/cost-estimate.json`: `21bfc3457d46af9d85f0ed3749dd445660c20d1196a228455b17e9126ab5da40`
- `setup/swe85-cap-extensions/runner-validation.json`: `bb8c43e3b6714fae349b3d9e174ca64142250d2ed7e462ba551e7a9ba007ed69`

Contraction-short controls:

- `setup/swe85-contraction-short-controls/candidate-cases.json`: `06f5becacaad409badf1813b00e65a337174e82753e4e001701be5835c1fc06f`
- `setup/swe85-contraction-short-controls/preparation.json`: `7a7bd3a29a83d9992ad5af81b3201e05bd0059590e0f34390732d92fc6074925`
- `setup/swe85-contraction-short-controls/cost-estimate.json`: `bbe5edc4526d4ccd7dbb1d88065f32f1708e674503f217520a21548986603f0f`
- Source manifest `setup/swe85-admitted-controls85/combined-cases.json`: `f2aeb0bfe40bce8c102ceba564c41ca164a138d90d9fd25adc5993b6c3d8eb81` (the contraction preparation's recorded source hash is `7eed047a0fb8ffa351351d69055b209a630502c27298b9120b7d79c5e57b530c`; it refers to its declared source manifest and matches that source in the preparation chain).

Reviewed generator/validator source:

- `setup/prepare_swe85_cap_extensions.py`: `c20f6827b7b9e8a03ca67f54b96ed84cb807ff99b154d62ba61de95c6d6c5e0b`
- `setup/validate_swe85_cap_extensions.py`: `2fd1ab4af2a2d7297042abb932fa922f9b93590003324fc85bfc0b8cef837043`
- `setup/prepare_swe85_optional_suffix512.py`: `3265ef22814b9a61a8c385dd274928fdcb37e464da3f7bb3facee64a8b186ec9`

Source evidence reviewed includes the referenced original fixtures, result JSON, coordinator adjudication rows, and fixture metadata. No changes were made to those files.
