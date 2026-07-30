# DiffusionGemma 26B-A4B-it — plan, spec and execution contract

Status: current. The branch's authoritative launch/metric/quality contract, plus the canonical
model, mask-geometry and environment facts.
Owns: model config and the three-phase generation procedure, the reference-config faithfulness
check, reuse-vs-build, the canonical denoise mask geometry (+ the W2b long-context result), the
QB2/`bh-qbge-06` environment and hardware-recovery recipe, current shipped defaults and best numbers.
See also: [refuted list + open contradictions](doc/REFUTED.md) · [agent guide](AGENTS.md) · [decision fidelity](doc/decision_fidelity/README.md) · [perf hub](doc/optimize_perf/README.md) · [serving hub](doc/vllm_integration/README.md) · [QB2 memory budget](QB2_MEMORY_BUDGET.md)

*Over the 250-line target on purpose: §4 carries four measurement traps, §5 is the tree's only copy
of the environment/recovery recipe, and §6 holds three open contradictions. None of those may be cut
for length.*

Tracking issue [tenstorrent/tt-metal#47452](https://github.com/tenstorrent/tt-metal/issues/47452)
(label `DiffusionGemma`). Work branch `diffusion-gemma-function` (earlier work on
`zni/diffusion-gemma-bringup`). Module root `models/experimental/diffusion_gemma/`.

---

## 1. Model and generation procedure

**Text backbone (identical to the in-repo Gemma-4 26B-A4B MoE, `models/demos/gemma4/`):** 30 layers ·
hidden 2816 · 16 heads / 8 KV · head_dim 256 · MoE 128 experts top-8 + 1 shared MLP ·
`moe_intermediate` 704 · `intermediate_size` 2112 · `num_global_key_value_heads` 2 ·
`global_head_dim` 512 · dual RoPE (θ = 1e6 full / 1e4 sliding) · final logit softcap 30 ·
vocab 262144 · `canvas_length` 256. All verified against `config.json`.

**The layer split is 25 `sliding_attention` (window 1024) + 5 `full_attention` — 83% sliding, not the
1:1 interleave older text implies.** That ratio is what makes sliding-layer key retention decisive;
see [device Gumbel restored](doc/decision_fidelity/device_gumbel_restored.md).

Vision tower (Functional+, #47467): `gemma4_vision`, 27 layers · hidden 1152 · `patch_size` 16 ·
280 soft tokens/image ("SigLIP" is an author label; the config says `gemma4_vision`).

### Three-phase block-autoregressive multi-canvas diffusion

The same backbone and weights run in three phases per 256-token block, selected by attention mode:
**prefill** (encoder, causal — encode the prompt, write KV) → **denoise** (decoder, bidirectional —
iteratively denoise a 256-token *canvas*, cross-attending to the prompt by concatenating encoder K/V
in front of canvas K/V, prefix-style, no separate cross-attention module) → **commit** (encoder,
causal — re-encode the finished canvas, append its KV, emit 256 tokens). Denoise is **read-only** on
the prompt/committed KV; the canvas's own K/V is recomputed every step (a 256-token mini-prefill
against the frozen prefix) and is never written into the frozen cache until commit.

**Noise is RANDOM tokens, not a `[MASK]` token** — the canvas is initialized to random ids and
rejected positions are re-noised to random ids (uniform discrete diffusion, not absorbing-mask).

**Per denoise step (≤ 48):** temperature-scale linear 0.8 → 0.4 → **Gumbel-max**
`argmax(logits/T + gumbel)` → **entropy-budget acceptance** with the **EXCLUSIVE** prefix
`(cum - sorted_vals) <= budget` (HF `accept_canvas`), **not** inclusive `cum <= budget` → re-noise
the rejected positions → halt when the argmax canvas is stable AND mean entropy <
`confidence_threshold` 0.005 with `stability_threshold` 1, else cap at `max_denoising_steps` 48.
**Commit = clean argmax**, not the noisy sampled values.

**Self-conditioning** (the one net-new weight module): previous-step softmax → probability-weighted
average of token embeddings → small gated MLP → added to canvas embeddings. Active only in denoise;
**zeroed on encoder passes**.

**Reference-config faithfulness check.** Every generation knob was compared item by item against the
released `generation_config.json` and matches: `max_denoising_steps` 48, `confidence_threshold`
0.005, `stability_threshold` 1, `t_max` 0.8 / `t_min` 0.4, `sampler_config.entropy_bound` 0.1
(absolute, in nats). So a non-converging trajectory is not a settings bug.

Algorithm reference: transformers `modeling_diffusion_gemma.py` (`DiffusionGemmaForBlockDiffusion`);
vLLM blog <https://vllm-project.github.io/2026/06/10/diffusion-gemma.html>.

---

## 2. Reuse vs build

`models/demos/gemma4/` is a near-complete, trace-compatible on-device Gemma-4 26B-A4B MoE; its MoE,
softcap, dual RoPE and weight loading already match the target.

**Already present — do NOT rebuild:**

- **K=V tying for FULL-ATTENTION (global) layers ONLY** — flag `attention_k_eq_v`
  (`tt/model_config.py:45`), gated `… and not self.is_sliding` (`tt/attention/__init__.py:34`),
  implemented as `v_w = k_w` (`tt/attention/weights.py:73`). **Sliding/local layers keep a REAL
  separate V** — assuming K=V there gets the bidirectional local-window path wrong (#47462).
- Scaleless V-norm (`tt/attention/prefill.py:61` and `:214`, `decode.py:84`).
- The bounded-sliding hybrid KV cache; tokenizer and chat template.

Net-new: bidirectional canvas attention + mask geometry (#47462), the three-phase KV state machine
(#47474), the decode loop (#47463), the self-conditioning gated MLP (#47461/#47463), and on-device
canvas sampling over all 256 positions (#47472). `weight_mapping.py` remaps DiffusionGemma
`model.decoder.*` onto the unmodified gemma4 loader `model.language_model.*` — a pure prefix swap;
the remapped backbone keyset equals gemma4's exactly.

---

## 3. Canonical denoise mask geometry (read before touching a mask)

`reference/attention_mask.py` is the oracle.

- **Full-attention layers:** all-attend `[C, P+C]` (zeros / maskless fast path).
- **Sliding layers, short prompts** (`P + C - 1 <= sliding_window`): also all-attend.
- **Sliding layers, long prompts:** HF's bidirectional sliding visibility
  `abs(q_idx - kv_idx) <= sliding_window`, which drops prompt tokens before the window. Use
  `build_canvas_denoise_mask(..., layer_type="sliding_attention", sliding_window=...)`.
- **Never pass `sliding_window_size` on the denoise path**, even for sliding layers: `attn_mask` and
  `sliding_window_size` are mutually exclusive in the ttnn SDPA op
  (`sdpa_device_operation.cpp:67-68`), so any window must be baked into the dense mask.
  `sliding_window_size` stays on the causal prefill/commit paths only.
- Canvas absolute/RoPE positions are offset by `prompt_len` (`canvas_positions`).
- gemma4's prefill SDPA is hardcoded `is_causal=True` (`tt/attention/prefill.py:126,264`,
  `operations.py:333`); DG adds the explicit `attn_mask` / `is_causal=False` branch. gemma4's
  existing chunked-prefill long-context workaround is **causal-only** (`operations.py:25-29`,
  `prefill.py:106-130`) and silently returns wrong results non-causally.

**W2b (long prompts > 32768), resolved 2026-06-26 with no new kernel.** Plain non-causal SDPA passes
PCC ≥ 0.99 against an independent fp32 online-softmax oracle at `[256 × Sk]` for
Sk ∈ {8192, 32768, 33000, 65536, 131072, 262144} and head_dim ∈ {256, 512}; masked and maskless arms
both pass, RoPE caches reach 262144, and integrated tiny-model denoise attention passes at
`P+C = 33280` and `262144` for both layer types. The work collapsed to re-keying the `prefill.py`
`long_seq` guard against **K length** instead of Q `seq_len`. Repro (env: see §5):
`DG_W2B_SDPA_SWEEP=full pytest models/experimental/diffusion_gemma/tests/test_device_long_sdpa_w2b.py -q`
(29 passed), also wired into `tests/pipeline_reorg/blackhole_e2e_tests.yaml` as
`bh-diffusion-gemma-w2b-full-sweep`. Residual: the integrated case is a tiny config (hidden 128,
head_dim 32); real-26B integration is #47464. Four refuted framings from that spike are in the
[refuted list](doc/REFUTED.md#sampling-rng-and-decision-fidelity).

---

## 4. Current execution contract

### Exactly two denoise execution modes

1. **Metal trace (the default):** one model-lifetime, startup-captured path. `DG_UPFRONT_CAPTURE`
   defaults to `1`; reveal masking, on-device Gumbel, K=48 and one-step/window early halt are
   **intrinsic**, not separately selected. The trace/controller is rebound across requests and
   released only at teardown.
2. **Eager fallback:** set `DG_UPFRONT_CAPTURE=0` **explicitly** — leaving it unset no longer
   disables capture. Eager is the only path that emits per-step trajectory records (a replayed trace
   does not), but it is not optimized traced-serving evidence.

There is no supported fixed-budget, grouped/multistep, frozen-prefix, per-request or argmax trace
variant. The knobs that once selected them were deleted; those names do nothing — see
[flag triage](doc/optimize_perf/flag_triage_20260728.md).

### Required, fail-loud, and defaulted

- **Required and fail-loud** (neither can be derived from anything the wrapper knows):
  `DG_UPFRONT_PREFILL_WARMUP_LENS=<all admitted aligned prompt lengths>` and
  `DG_TRACE_REGION_SIZE=<validated positive reservation>` (mirroring
  `--additional-config tt.trace_region_size`). The trace region is deliberately not defaulted: Metal
  takes it as an open-time constructor argument with **no getter**, so this process cannot read the
  reservation back, defaulting it would silence the guard without reserving anything, and a
  trace-region overflow poisons the device (needs `tt-smi -r`).
- Every admitted aligned prefill length must be known and compiled before capture; an unseen runtime
  shape is rejected rather than compiled while traces are resident.
- **`DG_DENOISE_REVEAL_PMAX` is optional.** Unset, the fixed reveal span is derived as
  `--max-model-len` rounded **DOWN** to a tile and logged — the KV cache seq dim is `max_model_len`
  verbatim, so rounding **up** would exceed the allocated span and abort startup. An explicit
  tile-aligned value wins; both paths get identical validation (positive tile multiple, ≥ tile + one
  canvas, within the allocated KV span). With neither an explicit value nor `max_model_len` it raises.
- A max-denoise-steps env override was **deleted 2026-07-28**: the up-front validator rejected every
  value but 48, so the export was ritual and the model config is already 48.
- Gumbel mode defaults to `device`; its history and the deleted host arm live in the
  [perf hub](doc/optimize_perf/README.md).

### Launch requirements (vLLM)

Always pass `--generation-config vllm`; set `--max-num-batched-tokens` at least as large as the
largest whole prompt; keep `--max-num-seqs 1` and `--block-size 64`. HTTP `temperature`, `top_p`,
`top_k` and per-request seed are **not consumed** by the model-owned denoise sampler; `ignore_eos=true`
is a transport stress control only. Full serving recipe: [serving hub](doc/vllm_integration/README.md).

### Metric contract

One model step emits a physical 256-token block. Report `prefill_s`, prefill + block-0 TTFT,
`denoise_steps`, denoise/commit/block latency, and `256 / block_latency` output tok/s.

- **TRAP:** API-visible `completion_tokens / wall_time` depends on EOS trimming and queueing and is
  **not** a device throughput metric; with `max_num_seqs=1`, curl wall time may include another request.
- **TRAP:** `--upfront` **forces 48 denoise steps** (`--max-denoising-steps` is ignored there) while
  the shipped early halt fires at ~2–9 steps, so use `serving_smoke --entropy-stop-threshold -1` for a
  per-step A/B.
- **TRAP:** every device verification must include a plain `default:` arm beside the optimized arm.
  Two deleted symbols broke the then-default-ON sparse dispatch for four commits because every check
  set the optimized flag, which bypassed it, and the host tests mock it. Both of those paths are gone;
  the rule is not.
- GPQA has its own three-denominator / dead-engine / matched-budget traps — see
  [decision fidelity](doc/decision_fidelity/README.md#gpqa-measurement-traps).

### Prefill contract

Distinguish pure `prefill_prompt_tokens` timing from serving TTFT and from
`BlockDiffusionServingSession.prefill`, which also constructs generation state. For up-front vLLM,
exact startup prefill warmups are a correctness requirement, not a benchmark convenience: a
variable-context harness that discovers tokenized lengths after startup is invalid.
`tt/prefill_moe.py` defaults `DG_PREFILL_RAGGED_LONG=1` — every multi-token prefill uses ragged top-8
expert execution and sequences above 4096 are processed in 4096-token slices; the 4K→16K dense-MoE
cliff belongs to the pre-fix `ec5b64b4891` control only. Current pure-prefill evidence:
`doc/optimize_perf/context_window_prefill_only_chunkedlong_20260713_msl65536.json` at `233b88276ab`.

### Shipped defaults and current best numbers

| what | state |
|---|---|
| `DG_SDPA_GRID=device` | −8.8%, bit-exact — **default flipped** |
| concat MoE (`tt/concat_moe.py`) | −29.9%, fold verified at PCC 0.9999218 — **shipped 2026-07-29 as the ONLY denoise MoE**; `tt/sparse_moe.py` is prefill-only |
| full-canvas RMSNorm | −20.4%/block on the full 198, commit 2.04 s → 0.37 s — **shipped 2026-07-30** |
| sliding-layer key retention (`DG_DENOISE_SLIDING_WINDOW`) | **default ON** — repairs 52 of 64 collapsed questions, regresses 1 of 67 clean, and is 1.53× faster |
| hiding the prefill pad keys (`DG_DENOISE_HIDE_PREFILL_PADS`) | landed — fixes 7 of 7 block-0 collapses |

- Flag triage 2026-07-28: 24 default-OFF `DG_*` flags deleted, ~4,400 lines, python-only `DG_*`
  names 106 → 91 (`99c154f0df8..66166060146`). Six switches this module shipped were silently inert.
- **Any absolute quality number measured before the concat-MoE flip is void as a current result.**
  The concat MoE shipped as a correctness fix, not a trade: the token-gather MoE it replaced does not
  let the denoise trajectory converge (halted 0/9 vs 19/19; min halt entropy 0.44 vs 6e-4 against a
  0.005 threshold; ~2/3 of requests degenerate).
- The full-canvas norm's two reasons for staying opt-in were both measurement errors: "not
  bit-identical" rested on a ~2e-6 figure with no measurement behind it (the real cause was ttnn's
  rmsnorm defaulting to bf16 partial accumulation; with fp32 accumulation the two shapes disagree on
  0 of 69,206,016 elements), and "27% shorter answers" was a 10-question artifact that shrank to −10%
  at 71 questions and vanished at 198.
  > **OPEN CONTRADICTION (unexplained):** the per-norm delta is stated as ~2e-6 / PCC 0.999998 and as
  > 5.73 bf16 ULP / rel max 2.24e-2 — four orders of magnitude apart, never reconciled
  > ([l1_residency](doc/optimize_perf/l1_residency.md),
  > [perf hub](doc/optimize_perf/README.md)). Not explained.
- **CURRENT BEST GPQA:** the 198-question full-canvas-norm run scored **71.21%** (`max_gen_toks=13824`),
  with 0 empty replies and 0 responses over the 2% non-Latin threshold. **There is no budget-matched
  comparison for it** — no reference and no TT baseline was run at 13824. It was previously stated here
  as beating the 66.67% full run; that run used `max_gen_toks=5632`, 2.45x smaller, so the two are not
  comparable and the norm's effect on score is unmeasured. The only budget-matched TT-vs-reference
  reading is **66.67% vs 65.66% at 5632**.
- **CURRENT PERF:** at 238 ms/step the split is MoE 75.5, attention 29.6, shared MLP 9.1,
  self-conditioning 2.3, and **~120 ms outside the layer stack** — layer matmul is no longer the
  bottleneck. Commit is 0.27–0.37 s/block (7–10%).

---

## 5. QB2 environment and recovery recipe (`bh-qbge-06`) — the one copy

Every other command block in this tree starts after this preamble with "env: see plan.md".

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate     # Python 3.12, transformers 5.12.1
export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn
export TT_METAL_HOME=/home/zni/tt-metal                    # must be the BUILT tree, not a worktree
export TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal
export DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
export HF_HUB_OFFLINE=1 ARCH_NAME=blackhole
export MESH_DEVICE=P150x4                                  # QB2 = 4x Blackhole, (1,4) mesh, TP=4
export DG_RUN_DEVICE=1                                     # device tests skip unless this is set
```

- **Box identity:** QB2 = `bh-qbge-06` = TT-QuietBox 2 — 4 Blackhole processors on 2× `p300`
  dual-Blackhole PCIe cards (**not** `p150`), 480 Tensix cores, 720 MB SRAM, 128 GB DDR6 at
  16 GT/s (1024 GB/s), Ryzen 7 9700X + 256 GB DDR5 host. `MESH_DEVICE=P150x4` is a mesh-shape
  **launch label** for a 1×4 Blackhole mesh, not the card SKU. The model path is mesh-shape-agnostic:
  TP comes from `mesh_device.shape[1]` (`tt/common.py:59-65`); CCL links are arch-gated (2 on
  Blackhole). No mesh code edits are needed to target QB2.
- **Pre-run ownership check:** QB2 is shared. If `ttnn` cannot see 4 devices another job holds them
  (UMD reports `CHIP_IN_USE_0_PCIe`, typically someone's `VLLM::EngineCore`) — wait or coordinate;
  do not queue behind the lock.
- **Worktree trap:** worktrees have no firmware build artifacts. `TT_METAL_HOME` must point at the
  built `/home/zni/tt-metal` even when the source you are editing lives elsewhere.
- **Build-consistency trap:** if the prebuilt `.so` and the source kernels drift, JIT compile fails
  with a `tt_memmove` overload mismatch in the permute reader. Fix by building the source tree:
  `./build_metal.sh --disable-profiler`.
- **Hardware recovery:** erisc/eth core 29-25 re-hangs on teardown after a device run (board fw
  19.9.0 is ahead of tt-metal's tested 19.5.0 — an env quirk, not a DG bug). Reset between runs with
  `sudo /home/zni/.local/bin/tt-smi -r`, then re-run the cheap mesh smoke
  (`MESH_DEVICE=P150x4 pytest models/demos/gemma4/tests/unit/test_model.py::test_single_layer_model -k "1x4"`).
  Use `@pytest.mark.use_module_device` so a test opens/closes the mesh once and device churn stays low.
- **Checkpoints:** `/home/zni/dg_models/diffusiongemma-26B-A4B-it` (the #47461 target, 51.7 GB),
  `/home/zni/dg_models/gemma-4-26B-A4B-it` (stage-1 stepping stone, 51.6 GB — passing on it does NOT
  validate DiffusionGemma), `google/gemma-4-12B-it` (dense QB2 device-flow proof). The canonical HF
  source is vendored at `/home/zni/dg_ref_src/` and `reference/_upstream.py` is the bit-for-bit
  parity guard against it.
- Kernels are JIT: editing an LLK header needs no rebuild, but clear `~/.cache/tt-metal-cache`.

CPU-only reference/parity tests need none of this: `pytest models/experimental/diffusion_gemma/tests -q`
(device tests auto-skip).

---

## 6. Open items and contradictions

> **OPEN CONTRADICTION (unexplained):** #47462 records the shared backbone as "untouched", but
> R0.3/R0.4 made ungated, non-diffusion-gated edits to `models/demos/gemma4` **decode**
> (`apply_rope_decode_peruser` for all batches; 1×1 SDPA grid + `k_chunk_size=32` for all layers;
> weightless-router + per-head width-sharded RMSNorm; expert down-proj + Q L1→DRAM). No plain
> Gemma-4 26B decode PCC/throughput re-baseline has ever been run, so both readings stand. Not
> explained. See also the inherited hard-rule gate failure in
> [decision fidelity](doc/decision_fidelity/README.md).

> **OPEN CONTRADICTION (unexplained):** early halt is described here and in older serving notes as
> never firing under #48291 (full 48 model-faithful steps), and is measured firing at ~2–9 steps,
> at `[9,17,2]/48`, and at 100% of blocks under the retention mask
> ([early halt](doc/optimize_perf/early_halt.md),
> [device Gumbel restored](doc/decision_fidelity/device_gumbel_restored.md)). Not explained.

> **OPEN CONTRADICTION (unexplained):** host-vs-device Gumbel is quoted as 1.94× here, as ~1.48×
> (53.6 vs 36.3 tok/blk/s) in the launch notes and in
> [device Gumbel restored](doc/decision_fidelity/device_gumbel_restored.md), and as ~1.8×/denoise-step
> in [degenerate output](doc/decision_fidelity/degenerate_output_fix.md). The host arm was deleted, so
> this can no longer be re-measured. Not explained.

- **OPEN:** fp32 MoE backbone precision is blocked by `ttnn.topk` `TT_FATAL` on FLOAT32 and by fp32
  experts exceeding the QB2 DRAM budget. It is a separate owned effort on the shared backbone.
- **OPEN:** paged-cache ownership / concurrency (`--max-num-seqs 1` is still mandatory, #47557),
  the absolute served-context ceiling, near-limit prefill hardening, and multimodal serving.
- **HARD RULE:** DiffusionGemma must not edit `models/demos/gemma4/` or any other shared directory —
  see [AGENTS.md](AGENTS.md#the-no-shared-edits-rule).

## 7. Status

Foundation is closed: #47461's exit gate — causal backbone logits PCC vs HF **on the DiffusionGemma
checkpoint**, measured on QB2 2026-06-24 by `tests/test_device_backbone_pcc.py -k 1x4` — is **0.877**
(5-tok) / **0.847** (24-tok) against the 0.83 baseline, i.e. ≈ the plain-gemma 0.866, so the
fine-tuned weights add no extra error. The device pieces are validated — KV-phase machine (#47474), bidirectional masked SDPA at
both ≤32768 and >32768 (#47462), decode-loop control flow (#47463), on-device canvas sampling
(#47472). #47464 e2e, #47465 perf and #47466/#47488 serving are live: the module serves
block-granular generation through the tenstorrent/vllm TT plugin and is evaluated at GPQA-Diamond
scale. Remaining: #47557 batched decode, #47467 multimodal, #47475 quant dequant, #47489 CI, and the
open items above.

**Determinism trap for parity work:** token-for-token PCC vs torch requires injecting the torch run's
**exact** Gumbel noise **and** random-renoise token ids — on-device RNG will not bit-match. Reserve
regenerated noise for distributional checks, and validate the *decisions* (entropy, argmax, accept
mask), not just logits. top-k/top-p is not shipped in the reference (transformers defers it; vLLM PR
#45429 open/unmerged), so it is not a gate.
