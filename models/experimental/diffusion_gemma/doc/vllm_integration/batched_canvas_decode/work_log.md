# Batched canvas decode — work log (#47557, Agent C)

**Goal.** Model-side batched denoise: process B>1 canvases through the DiffusionGemma denoise loop
simultaneously. Deliver (1) a design note of every batch=1 assumption in the diffusion delta, (2) a
prototype behind `DG_BATCH_DECODE` (default OFF) running B=2 canvases, (3) a device correctness
check that B=2 committed argmax == two B=1 runs, bit-exact (independence / no cross-canvas leakage).
vLLM multi-request wiring (#47488) is out of scope.

Branch: `dg-vllm-batch`. Device: QB2 (shared 4-chip mesh; runs under `flock /tmp/dg-mesh.lock`).

## What was built

| Artifact | Path | Purpose |
|---|---|---|
| Design note | `doc/.../batched_canvas_decode/design_note.md` | every batch=1 op in the delta + generalization + independence argument |
| Batched driver | `tt/batched_decode.py` | `DG_BATCH_DECODE` flag, `BatchedDenoiseLogitsFn`, `run_batched_denoise_block` |
| Device demo | `demo/batched_decode_smoke.py` | B=2 vs 2×B=1 committed-argmax independence check on QB2 |
| Tests | `tests/test_batched_denoise.py` | CPU (flag + wrapper logic) + device (decision-kernel independence; full model smoke) |

## Design summary (what assumes batch=1, and how it generalizes)

Full detail in `design_note.md`. Key findings:

- **Already batch-generic (✓):** canvas init/re-noise host fns + `host_canvas_to_device`
  (`[B,C]`); all of `tt/sampling.py` (vocab-axis reductions, per-row); `denoise_step`,
  `entropy_budget_accept` (sort/cumsum/scatter on dim=-1), `renoise`, `make_denoise_constants`,
  `run_fixed_denoise_steps`. **The diffusion-delta decision kernels already handle B canvases.**
- **The one genuine batch-coupling (△):** `denoise_block` early-halt uses whole-tensor
  `torch.equal` + a scalar mean entropy → couples the batch. Removed by using the **fixed-step
  loop** (`run_fixed_denoise_steps`), which is per-row independent and is the shipping/trace-safe
  path anyway (RUN-first makes early-halt a no-op under #48291).
- **DG-local ops generalized (△, byte-identical at B=1):** `embed_canvas_tokens` (drop the
  `batch!=1` guard; reshape embed `[1,B,C,H] → [B,1,C,H]`); `denoise_attention` prefix-KV concat
  (broadcast the shared batch=1 prompt prefix to B rows before the concat — attention is
  per-batch-element, so this is the structural no-leakage guarantee); DG-local norms
  (`_rms_norm_dram`, `_chunked_norm_forward`), the FFN block (`_denoise_ffn_block_batched`), the LM
  head (`_apply_lm_head_batched`), and self-conditioning (`condition`) loop per-canvas for B>1
  because the shared **gemma4 backbone ops (MoE experts, LM-head program config, sharded RMSNorm)
  are hard `[1,1,seq,H]`** and must not be edited.
- **Out of scope (⊘ #47488):** `commit_canvas_tokens*` / `denoise_and_commit_block` (committing B
  canvases needs B KV caches with per-request page tables = the paged-cache half of #47488);
  distinct-prompt prefill. **The correctness check therefore validates the committed argmax *tokens*
  emitted by the denoise loop, not the KV append** — exactly the model-side boundary the task draws.

## Prototype design (behind `DG_BATCH_DECODE`, default OFF)

`run_batched_denoise_block` drives B canvases through **one** `run_fixed_denoise_steps` loop, so the
decision kernels run batched `[B,1,C,vocab] → [B,C]`. Two logits modes:

- **`loop`** (default / gate): `BatchedDenoiseLogitsFn` loops the proven single-canvas adapter over
  the B rows and stacks `[B,1,C,vocab]`, threading **per-row** self-conditioning state. Guarantees
  correctness/independence without editing the shared batch=1 backbone.
- **`dim0`** (opt-in probe): feeds `[B,1,C,…]` straight through the DG-local denoise forward (my
  generalized ops: batched attention on dim0, per-row loop for MoE/LM-head/norms). Non-gating; its
  viability depends on the shared op batch support and is reported by the demo's `--probe-dim0`.

Because the fixed-step loop has no early-halt, argmax sampling has no RNG, and the prefix is
broadcast (not shared in-place), canvas i's committed argmax is independent of canvas j.

## Evidence so far

- **CPU tests PASS** (no device): `test_flag_default_off`, `test_run_batched_requires_flag`,
  `test_batched_logits_fn_loops_and_threads_per_row_state` (verifies the wrapper loops per row,
  keeps per-row prev-logits, frees the previous step's, and `owns_logits` lets the loop free the
  stacked tensor). `py_compile` clean on all changed files.
- **gemma4 isolation gate held:** `git status --porcelain -- models/demos/gemma4/` empty.
- **Device correctness check:** PENDING (next; under flock on QB2).

## SHAs

- (this increment) design note + `tt/batched_decode.py` + demo + tests + DG-local batch
  generalizations; CPU-verified. SHA logged on push.

## OPEN QUESTIONS / next

- Run the device check under flock: `test_denoise_step_batch_is_row_independent` (decision-kernel
  independence, no checkpoint) and the full `batched_decode_smoke` (B=2 vs 2×B=1, reduced layers).
- `dim0` mode: confirm whether the shared `nlp_create_qkv_heads` / `concat_heads` accept a batch
  dim on the prefill-shaped denoise; if not, `dim0` stays a documented non-path and `loop` is the
  shipped prototype.
