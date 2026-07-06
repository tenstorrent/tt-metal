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

- `8345eb1e62d` — design note + `tt/batched_decode.py` + demo + tests + DG-local batch
  generalizations; CPU-verified. Pushed to `dg-vllm-batch`.

## DEVICE_PROBLEM — 2026-07-06 (device correctness check BLOCKED)

The device correctness check (`test_denoise_step_batch_is_row_independent` and the full
`batched_decode_smoke`) is **BLOCKED on a degraded mesh**, not on the batched-decode code.

- **First attempt** (`test_denoise_step_batch_is_row_independent` under
  `flock /tmp/dg-mesh.lock timeout 900`): device opened, then a JIT kernel build failed at link
  (`collect2: error: ld returned 1 exit status`, `JitBuildState::link` → `device.py:107`). At the
  time another agent (Agent B / APC, `prefix_cache_smoke`) was running a device job under the same
  lock — consistent with concurrent JIT-build contention, so I retried.
- **Retry** (same command, flock blocking for the mesh): the mesh is now **hardware-degraded** —
  `TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to become active
  again. Try resetting the board.` (`assert_active_ethernet_cores_to_reset` during teardown, core
  dump, exit 134). This is the **recurring recoverable eth core 29-25 fault** recorded in project
  memory (`dg-vllm-serving-env`: "eth core 29-25 recurring reset") — recovery is `tt-smi -r` +
  `(1,4)` mesh-smoke, which per the COMMON RULES the **orchestrator** must coordinate; I do **not**
  run `tt-smi -r` myself. No leftover python processes from me (`pkill` clean).

**Status:** STOPPED cleanly per the rules (do not thrash a degraded mesh). The batched-decode
prototype is CPU-verified and ready; the device independence gate is a single re-run once the mesh
is reset:

```bash
# after the orchestrator resets the mesh (tt-smi -r + (1,4) mesh-smoke):
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole PYTHONPATH=$PWD DG_RUN_DEVICE=1 TT_LOGGER_LEVEL=ERROR

# 1) cheapest gate: decision-kernel batch independence (no checkpoint, ~1 min)
flock /tmp/dg-mesh.lock timeout 900 python -m pytest \
  models/experimental/diffusion_gemma/tests/test_batched_denoise.py::test_denoise_step_batch_is_row_independent -q -s

# 2) full model-side batched denoise: B=2 == 2×B=1 committed argmax, reduced 2 layers
export DG_BATCH_DECODE=1 DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
flock /tmp/dg-mesh.lock timeout 900 python -m models.experimental.diffusion_gemma.demo.batched_decode_smoke \
  --mesh P150x4 --num-layers 2 --canvas-length 64 --batch 2 --max-denoising-steps 3 \
  --mode loop --probe-dim0 --local-files-only \
  --metrics-json models/experimental/diffusion_gemma/doc/vllm_integration/batched_canvas_decode/batched_decode_smoke.json
# expect: DG_BATCH_DECODE_SUCCESS per_row_match=[True, True] mismatch_count=[0, 0]
```

## OPEN QUESTIONS / next

- Re-run the two device commands above once the mesh is reset (blocked on orchestrator reset of the
  eth-core-29-25 fault).
- `dim0` mode: the `--probe-dim0` run will confirm whether the shared `nlp_create_qkv_heads` /
  `concat_heads` accept a batch dim on the prefill-shaped denoise; if not, `dim0` stays a documented
  non-path and `loop` is the shipped prototype (the gate uses `loop`).
