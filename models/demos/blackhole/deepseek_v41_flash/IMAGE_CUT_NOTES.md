# Image-cut / unaligned-piece recompute (DSV41_PIECE_LAG)

Status: implemented behind `DSV41_PIECE_LAG=1` (default off), CPU-tested, device verification pending (see "Verification").
Branches: `ssinghal/dsv4p1-image-cut` (this repo, off `ssinghal/dsv4p1-b8-final` 95a82b1cadf3); no plugin change needed (`ssinghal/dsv4p1-image-cut-plugin` = fd7ed0c).

## What the log says

`DSV4.1 chunk continuation at S without carried state: recomputing from position 0` (generator_vllm.py `_prefill_interleaved`, `_place_row_aware`): vLLM reports a continuation piece
`[S, E)` of a request, the model has no resumable state at `S`, so it prefills `[0, E)` again.

Two different cases hide behind the message (tools: `dsv4-logs/tau3_imagecut/waste.py`, wasted = S - restored positions):

1. vLLM block-cache hit of a NEW request (S = 512 / 2048 / 4096 ...): the model's snapshot store (APC) restores at R <= S and computes the rest. Benign for the message (logged at INFO with APC),
   the waste is S - R (APC granularity), not an image problem. In the tau3 candidate run (text only) this is 19.2% of computed positions (599 messages, 1123 prefill calls).
2. A real continuation of a prompt that is prefilled over several scheduler steps: the previous piece ended at an end the model cannot resume from. THIS is the image-cut waste.

## Root cause of case 2 (file:line in 95a82b1cadf3)

`Model._prefill_interleaved_one` (dsv41_model.py, docstring ~1797): a user can resume at `start` only if its previous piece ended on a multiple of the chunk (512): the last window of an
unaligned piece is padded, `pf_resume` is not a valid resume point. The scheduler's piece ends are not chunk multiples in two cases:

* vLLM rolls a piece back to the start of an image span (an image is never split, `max_num_batched_tokens` 32768 / the encoder budget ~4096 tokens per step): ends are span starts, i.e.
  arbitrary positions (e.g. `[(0, 32593)]`, then +4095 per step for 250 images of 184 tokens).
* a piece ends inside an image span of a request whose images are in the encoder cache (repeat of a request): `vision_state.cut_end` shortens the piece to the chunk boundary below the span
  (generator_vllm.py ~1135-1150), but `inprog.update(p, ends[i], toks)` records the SHORTENED end while vLLM's next piece starts at the UNCUT end.

In both, `PrefillTracker.find(start, prefix)` (vllm_state.py ~265, exact `end == start` match) or `pf_resume != start` fails, so the whole prompt from position 0 is recomputed at every piece.
Cost is quadratic in the number of pieces: a 250-image prompt (48779 tokens, cand_acc A4 many250) is prefilled as `(0,32593) (0,36688) (0,40783) (0,44878) (0,48779)` = 203 721 positions
for 48 779 prompt tokens (4.2x); the 100-image prompt 19529 tokens as 4 pieces of 4095 (3928, 8023, 12118, 16213 recomputed). Text prompts with long concurrent prefills suffer the same
(any piece end that is not a multiple of 512 and not the end of the prompt).

## Measured waste (server logs, `waste.py`)

| log | what | computed positions | wasted | note |
|---|---|---|---|---|
| cand_acc/srv/h2 (many250 window 17:57:57-18:02:21) | many250 (16/100/250 images) + background | 364 246 | 196 248 (53.9%) | 8 non-512 messages = 195 224 of it: image-cut |
| cand_acc/srv/h2 whole run | all vision + text + stress | 7 753 437 | 1 255 576 (16.2%) | 15.3% non-512 (image-cut + 524288-deep long prompts) |
| cand_rel/srv/r1 (tau3, text only) | agent chat | 3 653 920 | 699 904 (19.2%) | 4096-hit restored at 2048: APC snapshot granularity, not image-cut |
| apc_img2 baseline waste.json (112x112 images) | 250 images | 79 908 for 46 271 tokens | 33 125 (ratio 1.73) | one cut at 33125 |

## Fix (minimal, no model/kernel change)

"Piece lag": the model keeps its carried end `a` (a chunk multiple) while vLLM believes the piece reached `E > a`. The next piece `[E, E2)` resumes at `a` and computes `[a, E2)` (tokens are
all present: the adapter gets the whole prompt from position 0). Cost per piece: at most one chunk (+ one image span) recomputed instead of the whole prefix.

* `vllm_state.PrefillTracker`: records `reported` (vLLM's end) next to the computed end; `find(..., lag=True)` also matches a user whose computed end `< start == reported` and whose
  computed prefix equals `prefix[:end]`; `bump()` for skipped pieces.
* `generator_vllm._prefill_interleaved`: a piece that does not end its prompt (plugin `prefill_final` flag False; flags exist for text and image launches) and ends off a chunk boundary is cut to
  `e // chunk * chunk` (skipped when that is not past its start); then the existing image cut (`VisionBook.cut_end`, aligned to the chunk) runs on the shortened end; the tracker stores
  both ends; `_lag_start(p, st)` returns the resume position (st, or the carried end when the lag matches) for the plain and the row-aware placement.
* A final piece (ends its prompt) is never cut; the carried state check (`pf_resume == carried end`) still guards every resume (lost state -> recompute from 0 as before).
* Speculative per-window lookahead uploads (`_look_key`) are keyed by (user, start, end): aligned piece ends make them hit more often, a mismatch only falls back.
* Switch: `DSV41_PIECE_LAG=1` (default 0). Text-only behaviour with the switch off is byte-for-byte the old path (`reported` is recorded but never used).

Expected effect: the 250-image prompt 203 721 -> ~49-50k computed positions (about 4x less prefill work on that request; ~3 min -> ~1 min under load), the 100-image prompt 4 pieces ->
~1.03x. No effect on prompts whose pieces already end on chunk boundaries.

## Interplay with APC-with-images (`ssinghal/dsv4p1-apc-img2`, read only)

* Their generator_vllm.py hunks: `mm_cache_salt` (lines ~883, ~1052, the `update_many` call ~1185); vision_state.py `VisionBook` keys; prefix_cache.py/dsv41_model.py restore at continuation.
  Mine: `__init__` (+4 lines), `_lag_start`/`_busy_rows` neighbourhood, the two `find` call sites, the cut block before `eff = [...]`, two lines at `inprog.update` / skipped loop, plus
  vllm_state.PrefillTracker. No overlapping hunks; `git merge` of both onto 95a82b1 should be textual-conflict free (to be verified at integration, merge order: APC-img2 first, then this).
* Semantic: their restore at a continuation piece removes recompute only up to the deepest snapshot (4096 granularity); the lag fix needs no snapshot and also covers prompts without images
  and prefix-cache-off servers. Both can be on: a lagging continuation keeps its carried state, so the restore path is not entered for it (restore is for NEW prompts / lost state).

## Effort

Implemented: ~0.5 day (+ device verification). Remaining: verification runs (below), then default-on decision.

## Verification (device, pending a free host)

1. text-only unchanged (switch off and on): tool_checks / gsm-style short run, same outputs, no new log lines.
2. many250 (16/100/250 images): waste.py before/after (switch off vs on), answers 16/16 100/100 250/250 correct.
3. vision suite (client_img, many250, vis_apc), repeated-image test (`serving_apc_image_repeat.py`), image-cut regression `serving_image_recompute_waste.py` (APC-img2 tests dir).
