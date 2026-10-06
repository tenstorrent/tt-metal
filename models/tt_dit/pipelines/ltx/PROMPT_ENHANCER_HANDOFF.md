# LTX prompt enhancer: hand-off

Branch `rsalman-ltx-prompt-enhancing` (from `main` @ `f6f9cc14e40`). State as of 2026-10-06, written on
the Blackhole Galaxy `bh-glx-120` (32 chips, mesh 4x8). Scope decided by the owner: **Galaxy only**; the
2x4 LoudBox `dynamic_load` path is out of scope and refused in code. All experiments run with
`LTX_TRACED=0` unless stated; the serving path is traced.

## What this is

LTX-2.3 serving rewrites the raw user prompt into the caption style the model was trained on before the
Gemma-3 text encoder sees it (Lightricks: "a separate, smaller model, Gemma 4 E2B"). This branch adds that
stage to `models/tt_dit/pipelines/ltx/`:

- `prompt_enhancer.py`: `PromptEnhancer` contract; `HostPromptEnhancer` (CPU `transformers`, bring-up and
  A/B only); `DevicePromptEnhancer` (Gemma-4-E2B-it through `models/demos/gemma4` `Gemma4Generator`, run
  on the **pipeline's own (4,8) MeshDevice handle**, TP=8 along the row, 4 redundant replica rows);
  `build_prompt_enhancer(backend)`; `apply_prompt_enhancer`; the Lightricks T2V/I2V system prompts verbatim.
- `pipeline_ltx.py`: `prompt_enhancer=` and `enhancer_seed=` kwargs; the pipeline binds its handle and ring
  topology into an unbound device enhancer; `_enhance_prompt` runs before the prompt is hashed or encoded,
  memoizes one rewrite per (backend, mode, seed, prompt), falls back to the raw prompt when the prompt does
  not fit the rewriter budget, exposes `last_enhanced_prompt`, adds a "Prompt enhance" timing row;
  `_warmup_prompt_enhancer`. Raises `NotImplementedError` for a device enhancer under `dynamic_load`.
- `pipeline_ltx_distilled.py` / one-stage / two-stages: the hook on the positive prompt only; warmup order.
- Tests: `models/tt_dit/tests/unit/test_ltx_prompt_enhancer.py` (37 host-only tests, fake generator seam);
  `test_pipeline_ltx_distilled.py` reads `LTX_PROMPT_ENHANCER`.
- Experiments: `models/tt_dit/tests/models/ltx/prompt_enhancer_experiments/` (README with commands, drivers,
  measured `results/`).

Enable: `LTX_PROMPT_ENHANCER=host|device` (unset/off = raw prompt), `LTX_ENHANCER_PATH` (default
`google/gemma-4-E2B-it`), `LTX_ENHANCER_CACHE_DIR` (converted weight cache; default shared
`/mnt/models/huggingface/tt_cache/gemma-4-E2B-it` if writable, else `~/.cache/tt-gemma4-e2b`). Sampling:
temperature 0.7, top_k 64, top_p 0.95, seed = `enhancer_seed` (default 10); greedy at temperature 0.
Cap `max_new_tokens` = 512 by default (constructor argument; the A/B driver sets 300). Stop ids
`[1, 50, 106]` from the snapshot's `generation_config.json`.

## Weights and machine prerequisites

- `google/gemma-4-E2B-it` snapshot `3e22461f65e89153144f8adb70e3b8c2cc9845a7` and `google/gemma-4-E2B`
  (base, unused) in `/mnt/models/huggingface/hub`; the pipeline's text encoder stays
  `google/gemma-3-12b-it-qat-q4_0-unquantized`; LTX checkpoint `Lightricks/LTX-2.3:ltx-2.3-22b-distilled-1.1.safetensors`.
  Run with `HF_HOME=/mnt/models/huggingface HF_HUB_OFFLINE=1`.
- Converted E2B cache for the 4x8 mesh: 11 GB at `/mnt/models/huggingface/tt_cache/gemma-4-E2B-it/tensor_cache_bf16_mesh4x8`
  (group `tt-cache`). Another machine reconverts on first load (~50 s extra); a different mesh shape is a
  different cache.
- Launch form (no tt-device-mcp on this box): `mpirun -np 1 --bind-to none --tag-output --wdir $TT_METAL_HOME
  python3 -m pytest -s --timeout 2400 ...` with the env block in the experiments README. `-s` matters for
  drivers outside the tests tree, otherwise pytest swallows the pipeline log on success.
- `hf`/`huggingface-cli` in `python_env` exits 1 after a successful command (typer/click mismatch).

## Verified (numbers from `prompt_enhancer_experiments/results/`)

| Fact | Evidence |
|---|---|
| Overlapping submeshes are accepted but each MeshDevice handle owns its own allocator: a (1,1) sibling of the (4,8) got the same DRAM address, corrupted the DiT handle's tensor on the shared chip, and replay corruption crossed handles undetected by the trace tracker. Hence: same handle, never a sibling submesh. | `results/submesh_overlap.md` |
| E2B-it on the (4,8) handle builds as TP=8 / DP=4 with no gemma4 code change; rows 1–3 are replicas; logits argmax identical across rows. | `results/e2b_on_dit_handle.md` Phase 0 |
| Eager decode 3.6–3.9 tok/s (0.26 s/step), prefill 0.6 s for the 976-token templated prompt, load 17 s warm; 189-token rewrite = 53 s vs a 24 s untraced generate. The single-chip reference with decode trace is 22.8 tok/s. | same, Phases 1–3 |
| Parity vs HF bf16 greedy: teacher-forced top-1 agreement 61/64, HF argmax in device top-5 64/64; the three flips are HF near-ties (< 0.5 logit). Free-running greedy diverges at token 2 for that reason. | `parity_probe` section |
| Beekeeper A/B inside the real pipeline, 1088x1920x153f: arm A raw 29.5 s wall, arm B device-enhanced 82.7 s; LTX stage times unchanged with E2B resident; rewrite byte-identical across runs for a fixed seed. | `results/ab_beekeeper_device.md` |
| **Ordering invariant (bug found by the A/B, fixed):** the resident Gemma-3 encoder captures its encode trace on its first encode regardless of `LTX_TRACED`; E2B allocated after that capture was corrupted by the first replay (garbage rewrite, arm A video untouched). The enhancer now warms before the encoder's first encode in both distilled warmup branches. | same, "Attempt 1" |

## Open at hand-off, in order

1. **Serving-configuration check (step 1) is unfinished.** `LTX_TRACED=1` with the enhancer resident and
   eager, 3-gen e2e, trace allocation tracker on. Attempt with the strict tracker fails at gen 0's first
   replay on three buffers that are LTX's own per-step temporaries (`v_vel`/`a_vel` typecasts at
   `pipeline_ltx_distilled.py:553/567` and the timestep `StateTensor.update` copy); LTX alone trips the
   tracker the same way. With the report-only plugin, gen 0 and gen 1 completed with healthy latents and
   valid videos; the run was killed externally during gen 2 (the pure replay), and 24 extra copy-op
   buffers flagged at gen 1's first replay are **unclassified** (window contains both the rewrite and the
   encoder trace capture; the tracer clones its inputs). Rerun the command in the README; the plugin now
   flushes JSON per flag; classify with `classify_tracker_report.py`. Acceptance: gen 2 completes healthy,
   every flagged buffer attributed, any enhancer-owned one freed before replay or justified.
2. **Latency.** Eager decode is the floor. The lever is gemma4's decode trace (`DevicePromptEnhancer(enable_trace=True)`,
   plumbed, never exercised). Capture order on one handle must be: E2B weights+KV → encoder weights and
   connector workspace → E2B decode capture (lazy, on the warmup rewrite) → DiT captures → encoder capture
   last (`defer_trace_capture` gate). Validate with the tracker. Also measure on-device sampling
   (`GEMMA4_HOST_SAMPLE=0`; drops the 262144-wide logits all-gather per token) and async CCL.
3. Serving token cap (300 vs 512) and whether a per-request bypass flag on `generate` should exist (none today).
4. Quality gate: 10-prompt A/B with `RUN_VBENCH=1 RUN_CLIP=1`; only one clip has been eyeballed. The
   rewriter already violated one system-prompt rule once (invented camera motion).
5. System-prompt KV prefix reuse (976 tokens re-prefilled every request; matters once decode is traced).
6. Rows 1–3 compute duplicates (`generator.data_parallel=1`); acceptable for batch 1.
7. Serving integration (tt-media-server LTX runner): pass `prompt_enhancer=build_prompt_enhancer(...)`,
   gate health on enhancer warmup, return the enhanced prompt.
8. Host-backend speed varies 28–130 s for the same rewrite depending on the LTX conftest's pre-import core
   pinning; irrelevant for serving, relevant when quoting CPU numbers.

## Gotchas

- Device code must run on `LTXPipeline.mesh_device` (the pipeline's submesh), never the fixture parent.
- `TT_CACHE_PATH` and `GEMMA4_CCL_TOPOLOGY` are set only for the duration of generator construction.
- Trace allocation tracker: use the report-only plugin for anything involving LTX; strict mode cannot get
  past LTX's own temporaries.
- Untraced e2e drivers outside `models/tt_dit/tests` do not get the LTX conftest's pre-import re-exec
  pinning; harmless for correctness.
- A stale UMD mutex `CHIP_IN_USE_0_PCIe` from a dead PID cost 85 s at startup once; UMD recovered alone.
- Log monitors tailing a file written by a detached `setsid nohup` process missed events twice; poll the
  log directly.
