# MiniMax-H3 FL2VA on BH Galaxy 4x8 — execution state

Plan (source of truth for scope, order, contracts, gates):
`~/.claude/plans/elegant-wibbling-brooks.md`. Re-read both before every iteration.

Branch: `kevinmi/minimax-h3`, based on `c63d43b89ec`.
All PRs target `kevinmi/minimax-h3`.

## Branch base — read before any git operation

The branch was cut in place from `kevinmi/optimizer/rjsdpa-split-forward-2026-07-07`
rather than from `main`, because the working tree carries **5 uncommitted tracked
modifications that belong to that rjsdpa work, not to H3**:

```
models/tt_dit/layers/normalization.py
models/tt_dit/models/transformers/ltx/attention_ltx.py
ttnn/cpp/ttnn/operations/experimental/ccl/CMakeLists.txt
ttnn/cpp/ttnn/operations/experimental/ccl/ccl_experimental_nanobind.cpp
ttnn/sources.cmake
```

Plus untracked, also not H3: `benchmarks/`, `update_spi_data_table_glx`,
`models/tt_dit/tests/models/ltx/test_perf_ltx_layer_tpsp.py`.

**Never `git add -A`, never reset, never rebase over these.** Stage H3 paths
explicitly. Rebasing onto `main` is a decision for the user once that WIP is
resolved.

## Current milestone

**M2 — host packing and conditioning, bit-exact.** Packing half complete and
gated; conditioning half (keyframe encode recipe, seeded noise aug, anchors) in
progress.

## Gate evidence

| M | Gate | Status | Evidence |
|---|---|---|---|
| 1 | 177 GB reclaimed, FL2VA downloaded, `TT_DIT_CACHE_DIR` set | reclaim done; download in progress | free 273 -> 436 GB after reclaim (`~/h3_reclaim.log`); 135/144 GB fetched (`~/h3_download.log`) |
| 2a | packing bit-exact vs diffusers `minimax-h3` | **PASS** | 48/48 exact checks vs reference across both working points, keyframe and t2va; then `test_packing_minimax_h3.py` **38 passed, 2 skipped** (skips = diffusers branch not installed) |
| 2b | conditioning recipe bit-exact | not started | — |
| 3 | AdaLN precompute parity | not started | — |
| 4 | weight load at TP=4/SP=8, shapes/dtypes/fixups | not started | — |
| 5 | attention block PCC >= 0.9995 | not started | — |
| 6 | full 50-layer forward PCC >= 0.99 @ 960x544 | not started | — |
| 7 | Qwen3-VL enc (50 layers, unnormalized) + vision tower PCC | not started | — |
| 8 | video VAE + audio VAE PCC and roundtrip | not started | — |
| 9 | e2e FL2VA @ 960x544, quality gates 1/2/4/7 | not started | — |
| 10 | trace + residency + warmup | not started | — |
| 11 | canonical 1344x768/8 s, quality gates 3/5/6 | not started | — |
| 12 | profile; **12b TP=8/SP=4 A/B**, adopt measured winner | not started | — |

## Measurements recorded

From M2, on host (no device):

- Canonical 1344x768 / 192f / first+last keyframe: `used=61109`, padded 61440
  (0.54% waste @ `k_chunk=512`), M=3 unique timesteps, 6 global AdaLN runs,
  per-SP-shard runs `[6,1,1,1,1,1,1,2]` mean 1.75.
- Bringup 960x544 / 124f / first+last: `used=21301`, padded 24576 (**13.3%**
  waste @ `k_chunk=512`), 22528 (5.45%) @ `k_chunk=256`.
- `t2va` canonical: M=2, 3 global runs, mean 1.38 runs/shard.
- Shape algebra confirmed: 192f -> 57 latent frames, 320 audio latents;
  124f -> 37 latent frames, 207 audio latents.

## §9 verdicts

| Risk | Verdict |
|---|---|
| 4096-row SP alignment waste | **CONFIRMED at bringup canvas**: 13.3% @ `k_chunk=512`, 5.45% @ 256. Canonical is only 0.54%, so this is a bringup-only concern. Re-measure device-side at M6 before changing the default. |
| Vision tower is required scope | **CONFIRMED** — a keyframe feeds both the VLM (as `<Picture i>:` + a vision block whose rows are video-tagged) and the video VAE. Plan and M7 updated; not optional. |

## Amendments to plan assumptions

1. **`last` keyframe anchor overshoots the final frame by 5.0 rotary units.**
   `origin + span(n) - 5/3` subtracts the shortest per-frame span (the `1` of
   `(1,4,4,4,4)`), not the final frame's actual span (a `4` when
   `(latent_t-1) % 5 != 0`). Reads as an off-by-one; it is not. Pinned by
   `test_keyframe_anchor_times`.
2. **diffusers builds the packed sequence with no padding at all**, stating
   padding "cannot influence a live row" — which is why its transformer needs no
   attention mask. Independently validates the plan's `logical_n = used` argument.
3. **Padding-waste estimates corrected** (see Measurements): canonical 0.54% not
   ~2%; bringup 13.3% not 19%, and `k_chunk=256` gives 5.45% not 0.02%.
4. **Text encoder is 50 of 64 layers**, `lm_head` unused, final
   `language_model.norm` **not** applied (~25.2B params / ~50 GB, not 62-66 GB).

## Hangs / resets

None yet. No device work has run.

`tt-smi -glx_reset` has standing permission for this goal. `tt-smi -r` remains
forbidden — it dropped all chips off PCIe on CPLD < 1.16.

## Failed attempts

- Two initial assertions in `test_packing_minimax_h3.py` were wrong, not the
  port: `last`-anchor equality with the final frame (see amendment 1) and an
  exact fp32 comparison against the literal `0.7`. Parity against the reference
  had already passed, which is what localized the fault to the test.

## Next step

Finish M2b: port the keyframe conditioning recipe (fp32 VAE encode under forked
RNG seed 42, `(z-mean)/std`, patchify, noise aug at `noise_aug=0.999`) and
resolve the sglang-vs-diffusers RNG draw-order discrepancy in favour of
diffusers. Host only, bit-exact.
