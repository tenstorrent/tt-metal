# Dormant-flag triage and deletion — 2026-07-28

Every default-OFF `DG_*` flag in the module was examined and forced into one of three outcomes:
**ENABLE** (it works and is better), **KEEP** (it earns its place, with the concrete situation that
justifies it named), or **DELETE**. 43 flags were triaged; **24 were deleted** across five commits,
`99c154f0df8..13bd1b34efc`, removing about **4,400 lines**. The python-only `DG_*` name count in the
module went **106 → 91**.

The bias was toward DELETE, for a specific reason: this module had already shipped three *silently
inert* switches — a `C == DEFAULT_CAPACITY` MoE gate that disabled the flagship lever for 12 days, an
unconditional `tt-smi` requirement, and `DG_TERMINAL_SHARDED`, whose producer had zero callers. The
triage found three more (`DG_CHUNKED_PREFILL`, `DG_COMMIT_WRITE_BATCH`, and a
`segment_rows == DEFAULT_CAPACITY` conjunct), bringing the count to six. Every dormant flag is a
place that can happen again.

## Deleted

| flag | why |
|---|---|
| `DG_CHUNKED_PREFILL` | `chunked_prefill_enabled()` had **zero dispatch sites**; two docs advertised it as a working offline switch |
| `DG_CHUNKED_PREFILL_CHUNK` | only reader of a knob nothing sets, in a module nothing dispatches to |
| `DG_PREFILL_RAGGED_M_BLOCKS` | passed a value equal to the callee's own default |
| `DG_COMMIT_WRITE_BATCH` | entire body was a `logger.warning` that the knob is obsolete |
| `DG_MOE_EXPERT_BFP8` | memory-**negative** as written (bf16 originals kept alive via `_orig`); never measured; the same lever measured properly under `DG_EXPERTS_BFP8` was rejected at clean-argmax 0.227 |
| `DG_MOE_L1` | its own doc: bit-identical, −0.6% @48 / a wash @12; mode `chain` a no-op by construction; mode `all` never measured |
| `DG_MOE_DISPATCH_FUSED` (v1) | measured perf-neutral by its own landing commit, superseded by FUSED2 27 minutes later |
| `DG_MOE_DISPATCH_ABLATE` | answered its one question (12.7%); its driver bench was already deleted, so it was un-runnable |
| `DG_DENOISE_CANVAS_TAIL` | its own doc: **+1.2% and +1.6% slower** over two runs, sha-identical, ~106 MB scratch, "leave it off" |
| `DG_ROPE_FUSED` | refuted the day it landed: −0.1% inside a 1.5% spread **and** the committed sha changes |
| `DG_ROPE_FULLCANVAS`, `DG_SDPA_FULLCANVAS` | default-ON flags whose OFF paths were kept alive only by host-only fake-ttnn tests that set the env themselves |
| `DG_TERMINAL_SHARDED` | silent no-op: `prepare_sharded_terminal` had zero callers in all of committed history, so the context was always `None`. 560 lines of apparatus that never executed |
| `DG_DEDUP_ARGMAX` | structurally unreachable under the shipped `DG_UPFRONT_CAPTURE=1`; not bit-identical by its own docstring; +0.5% measured |
| `DG_DENOISE_DEVICE_LOOP` | unreachable on the shipped traced path; strictly less capable (no trajectory records, no early halt); zero tests |
| `DG_DENOISE_COMPACT_RAGGED` + `DG_COMPACT_{COMBINE,PRIMARY_TUNED,SEGMENT_ROWS}` | its bit-compatibility claim is against a committed sha that `6b370bf1320` tables as the *replaced* path; pinned to TP=4 by a hard raise; ~800 Python lines + 3 kernels |
| `DG_PREFIX_CACHE` | only the exact-full-match tier could fire from serving; the proper-prefix tier measured 57/256 flipped tokens; `invalidate()` had zero callers; vLLM advertises `supports_prefix_caching: False` |
| `DG_GELU_TANH` | the OFF arm is wrong math against the checkpoint (`gelu_pytorch_tanh`) **and**, read in six places, simultaneously swapped the shared-MLP path and its CCL call — so it could not serve the bisect it was retained for |
| `DG_SPARSE_MOE_CAPACITY` | produced the retracted "~5× vs dense-128" figure by setting 32 |
| `DG_COMMIT_BATCHED` | the path it forces measures PCC 0.154 vs 0.994 and ~6.3× slower |
| `DG_VLLM_MAX_DENOISE_STEPS` | rejected by the up-front validator for every value but 48; the export in six runbooks was pure ritual |
| `DG_DEGENERACY_MAX_RUN` | the one member of its group with no writer anywhere in the repo |

Also removed: `return_sharded` from `models/demos/gemma4/tt/model.py`. That param does not exist on
`origin/main`; a DiffusionGemma commit added it. The shared-edits gate flags any touch of that tree,
so it flags this one — but the direction is inward: the DG delta in that file goes 107 → 94 changed
lines and `return_sharded` 3 → 0 occurrences.

## Two recommendations overruled

* **`DG_DENOISE_SLIDING_WINDOW_OVERRIDE` — kept.** The plan proposed deleting it together with the
  plumbing in `verify_denoise_sliding_window.sh`, on the grounds that the real-window arm still runs.
  But that harness exists to prove the #51080 retention plumbing is live, and its own comment
  explains why the real 1024 window cannot: at P=1056 it masks 2.5% of the attended span, too little
  to reliably flip an argmax. Deleting the override deletes the only arm that can fail.
* **`check_committed_block` — kept.** Zero production callers (generate.py inlines the same policy),
  but ten policy assertions in `test_degeneracy.py` exercise it. Deleting it would move tested pure
  logic onto an untested integration path. The duplication is real; the fix is to unify, and that is
  now stated in the docstring.

## Not done

`DG_MOE_FUSED_GATHER`'s scaffold spans the shared `ttnn/cpp` tree and needs a `_ttnncpp.so` rebuild.
It is out of scope under the no-shared-edits rule; the DG-side `raise NotImplementedError` remains as
the loud signal it already is.

## What the verification caught, and what only one kind of check could catch

Six symbols were swallowed by span cuts. Where they were caught matters more than that they happened.

**Caught by the host test suite (3):** `_temperature_at_step` (used by the self-conditioning feedback
on every traced step); the six sliding-window / pad helpers including the #51080 retention gate; and
the numba `_pack_ragged_assignments` — the plan located that block at 1577-1644, it is actually at
1368-1435, inside the cut.

**Caught by a DEVICE run only (2, and they are the important ones):** `_FUSED2_KERNEL` /
`_FUSED2_PLAN_CACHE`, and `build_capacity_dispatch` itself. Both are needed by the **default-ON**
sparse dispatch. The host tests mock the dispatch, and every device check up to that point used
`DG_MOE_CONCAT=1`, which bypasses it entirely — so **the default serving path was broken from
`e04d4490973` until `13bd1b34efc` and nothing said so.** What found it was adding a plain `default:`
arm alongside the `concat:` arm in the device sweep.

Two consequences, both adopted:

1. **Every device verification carries a `default:` arm.** Verifying only the configuration you are
   optimizing verifies only the configuration you are optimizing.
2. **A module-wide undefined-name AST scan** over all fifteen touched files, which now reports clean.
   A cut between two markers is fast and it does not know what lives in between.

## Final state

* 742 host tests pass, 124 skipped, 0 failures.
* On device, **both** arms are bit-identical to their pre-deletion baselines:
  `concat` (`DG_MOE_CONCAT=1 DG_NORM_FULLCANVAS=1`) `7b29837d637ec26b` at 9.449 s/block, and
  `default` `1c1934f6f781bb75` at 21.030 s/block.

## The convention that would stop the next one

Route every `DG_*` flag through one registry — `dg_flag(name, default, *, values=…, expires=…)` — and
make the registry the only way to read one. It (a) validates the value and fails loud outside
`values`, which kills `DG_SKIP=moe1`-style silent mis-measurement; (b) records every resolved
non-default value in the run-metadata JSON the sweep harness already emits, which kills the
silent-model-swap class; (c) requires an expiry naming the dated artifact that will decide it. One CI
test asserting that the set of `DG_*` literals in the tree equals the registry's keys, and that no
flag is past its expiry, would have failed on `DG_TERMINAL_SHARDED` the day its producer landed
without a caller.

A flag whose doc has reported a result gets exactly one more commit: the one that flips the default
or deletes the flag.
