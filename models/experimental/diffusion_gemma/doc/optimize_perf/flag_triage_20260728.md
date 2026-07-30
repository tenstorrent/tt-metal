# Dormant-flag triage and deletion — 2026-07-28

Status: current (2 lines over the 100-line cap — the 22-row deleted-flag table is the fact this file exists to carry, and the GPQA denominator traps are never cut for length).
Owns: **the only list of dead `DG_*` flag names in the tree**, why each died, and the flag-registry convention.
See also: [refuted list](../REFUTED.md) · [stage hub](README.md) · [campaign ledger](perf_progress.md)

Every default-OFF `DG_*` flag in the module was forced into one of three outcomes: ENABLE, KEEP (with the concrete situation that justifies it named), or DELETE. **43 flags triaged, 24 deleted** across `99c154f0df8..13bd1b34efc`, removing about **4,400 lines**; the python-only `DG_*` name count went **106 → 91**.

**The failure mode this was correcting.** The module had shipped **six silently inert switches**: a `C == DEFAULT_CAPACITY` MoE gate that disabled the flagship lever for 12 days, an unconditional `tt-smi` requirement, `DG_TERMINAL_SHARDED`, `DG_CHUNKED_PREFILL`, `DG_COMMIT_WRITE_BATCH`, and a `segment_rows == DEFAULT_CAPACITY` conjunct. Every dormant flag is a place that can happen again.

## Deleted, with reason

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
| `DG_DENOISE_CANVAS_TAIL` | its own doc: **+1.2% and +1.6% slower** over two runs, sha-identical, ~106 MB scratch |
| `DG_ROPE_FUSED` | refuted the day it landed: −0.1% inside a 1.5% spread **and** the committed sha changes |
| `DG_ROPE_FULLCANVAS`, `DG_SDPA_FULLCANVAS` | default-ON flags whose OFF paths were kept alive only by host-only fake-ttnn tests that set the env themselves |
| `DG_TERMINAL_SHARDED` | silent no-op: `prepare_sharded_terminal` had zero callers in all of committed history — 560 lines of apparatus that never executed |
| `DG_DEDUP_ARGMAX` | structurally unreachable under the shipped `DG_UPFRONT_CAPTURE=1`; not bit-identical by its own docstring; +0.5% measured |
| `DG_DENOISE_DEVICE_LOOP` | unreachable on the shipped traced path; strictly less capable (no trajectory records, no early halt); zero tests |
| `DG_DENOISE_COMPACT_RAGGED` + `DG_COMPACT_{COMBINE,PRIMARY_TUNED,SEGMENT_ROWS}` | its bit-compatibility claim is against a committed sha that `6b370bf1320` tables as the *replaced* path; pinned to TP=4 by a hard raise; ~800 Python lines + 3 kernels |
| `DG_PREFIX_CACHE` | only the exact-full-match tier could fire from serving; the proper-prefix tier measured 57/256 flipped tokens; `invalidate()` had zero callers; vLLM advertises `supports_prefix_caching: False` |
| `DG_GELU_TANH` | the OFF arm is wrong math against the checkpoint (`gelu_pytorch_tanh`) **and**, read in six places, simultaneously swapped the shared-MLP path and its CCL call — so it could not serve the bisect it was retained for |
| `DG_SPARSE_MOE_CAPACITY` | produced the retracted "~5× vs dense-128" figure by setting 32 |
| `DG_COMMIT_BATCHED` | the path it forces measures PCC 0.154 vs 0.994 and ~6.3× slower |
| `DG_VLLM_MAX_DENOISE_STEPS` | rejected by the up-front validator for every value but 48; the export in six runbooks was pure ritual |
| `DG_DEGENERACY_MAX_RUN` | the one member of its group with no writer anywhere in the repo |

Also removed: `return_sharded` from `models/demos/gemma4/tt/model.py`. That param does not exist on `origin/main`; a DiffusionGemma commit added it. The shared-edits gate flags any touch of that tree, so it flags this one — but the direction is inward: the DG delta in that file goes 107 → 94 changed lines and `return_sharded` 3 → 0 occurrences.

### Later deletions, after this triage

`DG_MOE_CONCAT`, `DG_SPARSE_MOE`, `DG_ALLOW_DENSE_MOE`, `DG_SPARSE_MOE_TUNED`, `DG_MOE_DISPATCH_FUSED2` and `DG_MOE_FUSED_GATHER` were deleted with the token-gather path in `7417bd7d69d` (2026-07-29); `DG_NORM_FULLCANVAS` was deleted with the full-canvas norm choice on 2026-07-30. Setting any of them now does nothing.

Legacy trace knobs that still appear in dated artifacts and select nothing: `DG_VLLM_TRACE`, `DG_DENOISE_TRACED`, `DG_DENOISE_TRACED_MULTISTEP`, `DG_DENOISE_MULTISTEP_GROUP`, `DG_DENOISE_EARLY_HALT`, `DG_DENOISE_EARLY_HALT_WINDOW`, `DG_DENOISE_FROZEN_PREFIX`, `DG_DENOISE_REVEAL_MASK`, `DG_DENOISE_LAZY_CAPTURE`.

## Two recommendations overruled

- **`DG_DENOISE_SLIDING_WINDOW_OVERRIDE` — kept.** It is the only arm of `verify_denoise_sliding_window.sh` that can FAIL: at P=1056 the real 1024 window masks only 2.5% of the attended span, too little to reliably flip an argmax.
- **`check_committed_block` — kept.** Zero production callers, but ten policy assertions in `test_degeneracy.py` exercise it; deleting it would move tested pure logic onto an untested integration path.

## Not done

*(Resolved 2026-07-30.)* `DG_MOE_FUSED_GATHER`'s scaffold spanned the shared `ttnn/cpp` tree, which is why it was left alone here — but the out-of-folder audit removed it at the source instead: the ttnn in0-gather hook and the `TTNN_SPARSE_MATMUL_IN0_GATHER` gate were reverted (`9f3f558319d`), and the DG-side flag had already gone with the token-gather path in `7417bd7d69d`. Nothing is left to gate.

## What only one kind of check could catch

Six symbols were swallowed by span cuts. Three were caught by the host suite (`_temperature_at_step`, the six sliding-window / pad helpers, and the numba `_pack_ragged_assignments` block the plan located at 1577-1644 when it is actually at 1368-1435). **Two were caught by a DEVICE run only, and they are the important ones:** `_FUSED2_KERNEL` / `_FUSED2_PLAN_CACHE` and `build_capacity_dispatch` itself — both needed by the then-default-ON sparse dispatch. Host tests mock the dispatch and every device check up to that point ran the concat arm, which bypassed it entirely, **so the default serving path was broken from `e04d4490973` until `13bd1b34efc` and nothing said so.** What found it was adding a plain `default:` arm alongside the concat arm in the device sweep.

Two rules adopted:

1. **Every device verification carries a `default:` arm.** Verifying only the configuration you are
   optimizing verifies only the configuration you are optimizing. (The specific `concat:` / `default:`
   arms of 2026-07-28 no longer exist — concat is now the only denoise MoE — but the rule stands.)
2. **A module-wide undefined-name AST scan** over every touched file. A cut between two markers is
   fast and it does not know what lives in between.

## Verification evidence

- 742 host tests pass, 124 skipped, 0 failures.
- On device both denoise arms were bit-identical to their pre-deletion baselines: concat
  `7b29837d637ec26b` at **9.449 s/block** and default `1c1934f6f781bb75` at **21.030 s/block**.
- The vLLM serving path was proven byte-identical by running the same 2-question
  `run_upfront_gpqa.sh smoke` at the pre-deletion tree (`25057cda4c4`) and at `afea6ce090e`:
  q0 3884 chars `6c35a0607c7acde1`, q1 6268 chars `429e93c2ad708381` on both.

## GPQA measurement traps from the post-deletion run

- The post-deletion 198-sample run scored `exact_match,none` = **6.57%** (stderr 1.76%) and **that is
  not evidence in either direction**: only **4 of 198** responses contain a `\boxed{}` at all, which
  is what the strict `none` filter extracts, and there is no same-metric pre-deletion baseline on this
  box (the 07-23 attempt aborted at 1/15).
- **Matched task and filter, or nothing.** The 70.71% / 70.20% reference numbers are
  `gpqa_diamond_cot_zeroshot` with `flexible-extract` — a different task AND a different filter from
  the strict `none` score above. See [decision fidelity](../decision_fidelity/README.md) for the
  three-denominator rule.
- What the run *is* evidence of: the integration holds after 4,400 lines were removed. It also
  reconfirms the open #48291 state — generations open correctly then collapse (one sample sets up the
  Heisenberg uncertainty relation correctly, then emits garbage), with **146 degeneracy-guard fires
  clustered at blocks 0-3 and 26 at block 0 exactly**. `DG_DENOISE_HIDE_PREFILL_PADS`, the correctness
  fix measured at 7 of 7 on precisely those block-0 collapses, is **now default `1`**
  (`tt/denoise_forward.py:277`, shipped via `205e87956cc`) — this file previously said it was still OFF.

## The convention that would stop the next one

Route every `DG_*` flag through one registry — `dg_flag(name, default, *, values=…, expires=…)` — and
make the registry the only way to read one. It validates the value and fails loud outside `values`;
records every resolved non-default value in the run-metadata JSON the sweep harness already emits; and
requires an expiry naming the dated artifact that will decide it. One CI test asserting that the set of
`DG_*` literals in the tree equals the registry's keys, and that no flag is past its expiry, would have
failed on `DG_TERMINAL_SHARDED` the day its producer landed without a caller.

**Policy:** a flag whose doc has reported a result gets exactly one more commit — the one that flips
the default or deletes the flag.
