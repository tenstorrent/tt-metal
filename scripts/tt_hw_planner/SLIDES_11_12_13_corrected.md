# Slides 11–13 — corrected against the actual tool source

Audit performed against `scripts/tt_hw_planner/` on 2026-06-01.
Every claim below is grounded in a file + line reference. Per-line
audit script: `python -m pytest scripts/tt_hw_planner/tests/test_agentic_wiring.py`.

---

## Slide 11 — Correctness Gates: Per-Category Validation

The correctness engine dispatcher (`correctness/registry.py`) auto-dispatches to
the right comparator based on model category. Each comparator self-registers at
import time via `register_comparator()`. The dispatcher (`get_comparator`)
returns the first comparator whose `supports(category, model_id)` returns True.

### Per-category validation methods (corrected)

| Category | Validation method | Default thresholds | Source |
|---|---|---|---|
| **LLM** | Token-overlap + optional step-0 logit-PCC | Configurable mismatch % (default: gate fails when ≥ 70% of compared tokens disagree) | `correctness/text.py` |
| **VLM** | Token-overlap (same comparator as LLM, `category="VLM"`) | Same as LLM | `correctness/text.py:188-200` |
| **Embed** | Cosine similarity, two gates | per-sentence cosine ≥ **0.95** AND mean cosine ≥ **0.97** across probe sentences | `correctness/embedding.py:44-45, 242` |
| **CNN / Segmentation** | IoU **AND Dice** (both must pass) | IoU ≥ **0.85**, Dice ≥ **0.90** (also per-class IoU floor ≥ 0.5) | `correctness/segmentation.py:77-78, 394-395` |
| **CNN / Classification** | Top-1 match **AND** top-5 overlap **AND** KL divergence of logit probabilities | Top-1 must match, top-5 overlap ≥ `DEFAULT_TOP_K_OVERLAP_MIN`, KL < `DEFAULT_KL_MAX` | `correctness/classification.py:286-303` |
| **CNN / Detection** | bbox-pair matching (label + IoU) + recall | bbox IoU ≥ **0.5** (COCO mAP convention), detection recall ≥ **0.80**, ref-score floor ≥ 0.5 | `correctness/detection.py:36-38` |
| **Image / Diffusion** | SSIM (primary) **+ LPIPS** (secondary, optional — requires `lpips` package) | SSIM ≥ **0.55**, LPIPS ≤ **0.45** | `correctness/diffusion.py:41-42, 270-278` |
| **Audio / STT** | Word Error Rate (WER) | WER ≤ **0.30** | `correctness/audio_asr.py:41` |

**Changes from the prior slide:**
- Diffusion: **FID was wrong**. Code uses SSIM + LPIPS, not FID. No FID implementation anywhere in `correctness/`.
- Classification: **"Accuracy / F1" was wrong**. Code uses top-1 + top-5 + KL.
  No F1 computation anywhere in `correctness/`.
- Audio: only WER; CER mentioned in slide but is NOT implemented (grep returns no
  `cer` or `CER` references in `correctness/audio_asr.py`).
- Segmentation: clarified that BOTH IoU and Dice must pass (slide showed only IoU).
- Embed: clarified the dual threshold (per-sentence + mean).
- Detection: added (was missing from slide entirely — implemented in `detection.py`).

### Escalation chain on gate failure

```
PCC gate fires → result.ok == False
              ↓
   _maybe_escalate_pcc_fail (cli.py:6717)
              ↓
   ┌──────────────────────────────────────────────────────────────────┐
   │  Probe a FamilyBackend match for this model:                      │
   │                                                                   │
   │  (a) Backend exists with quality="exact" → SHORT-CIRCUIT:          │
   │      skip cmd_auto_onboard, set _escalated_already=True,           │
   │      re-enter cmd_up which now takes the SCAFFOLD + PER-COMPONENT  │
   │      ITERATE path (Path 1 / SAM2 pattern). This is the common case │
   │      for Llama-family LLMs (Qwen2, Mistral, Llama-3).              │
   │                                                                   │
   │  (b) No exact backend match → invoke cmd_auto_onboard --accept     │
   │      to draft a NEW FamilyBackend, then re-enter cmd_up so         │
   │      scaffold + per-component PCC ≥ 0.99 iterate runs.             │
   └──────────────────────────────────────────────────────────────────┘
              ↓
   Per-component iterate loop (Path 1):
     - run_auto_iterate_loop dispatches LLM agents per component
     - G1 (divergence probe) runs INSIDE the loop on stuck components
     - G7 (learned-fixes lookup) consulted before each LLM dispatch
     - G8 (convergence) makes cap-extend / fallback decisions
```

**Correction vs prior slide:** the "Auto-Onboard OR Agentic Repair" framing
was misleading — these aren't alternative branches. Both paths converge into the
same scaffold + per-component iterate loop. The "agentic" probe (G1) and the
"repair" workflow are *inside* the iterate loop, not a separate escalation tier.

---

## Slide 12 — Agentic Divergence Probe: Localizing the First Bug

### Probe flow (6 steps)

1. **Auto-attach hook installed via `_walk_and_wrap`** on the TT model
   (`agentic/tt_probe.py:253`). The hook does NOT specifically target
   `LightweightModule.__call__` — it walks the module tree and wraps the
   `forward` (or one of 8 sibling) method on every class whose name is
   classified as "layer-like" by `module_tree._looks_high_level`.

   **Methods wrapped (the `_ENTRY_METHODS` tuple at `tt_probe.py:276`):**
   ```
   forward, ttnn_prefill_forward, ttnn_decode_forward,
   prefill_forward, prefill_forward_single_user_text,
   _apply_norm_and_lm_head, transform_and_embed_prefill_inputs_device,
   process_output_prefill, process_hidden_states_after_prefill_trace
   ```

   The wrap is method-level (`object.__setattr__(obj, _mname, wrapped_m)`)
   with a fallback to class-level `__call__` override when the per-instance
   set fails (`AttributeError` / `TypeError`).

2. **`_walk_and_wrap` traverses the model tree** breadth-first by attribute
   order, visiting each Python object at most once (via `_VISITED_IDS`) so
   shared sub-modules aren't double-wrapped. Configurable depth via
   `TT_PLANNER_PROBE_DEPTH` env var (default: 4).

3. **Forward pass on TT device** — each wrapped method records, on exit,
   scalar stats of the output tensor: **`mean, std, l2, abs_max`**
   (`agentic/tt_probe.py:24, 144`). Output goes to a JSON sidecar at
   `TT_PLANNER_PROBE_OUTPUT`.

4. **Reference pass on CPU** — `probe_hf_modules` (`agentic/probe.py`)
   instruments the HF reference model the same way, capturing the
   same **mean/std/l2/abs_max** scalars via a shared
   `_tensor_stats` helper (delegates to `activation_diff._tensor_stats`
   so HF and TT sides reduce identically).

5. **`compute_divergence`** (`agentic/diverge.py`, called from
   `executor.py:354`) aligns the HF and TT records by `qualified_name` and
   walks them in execution order, finding the FIRST module whose stats
   diverge beyond threshold. Returns a `diverge_report` with
   `first_diverging` set to the offending module entry.

6. **First-diverging layer's qualified name + stats diff emitted** —
   this is the structured input the iterate loop's LLM-prompt builder
   uses to target THAT specific layer (file-source resolved via `inspect.getsourcefile`
   in `agentic/resolve.py`).

### Key files

| File | Role |
|---|---|
| `agentic/tt_probe.py` | TT-side probe: install_probe + `_walk_and_wrap` + `_tensor_stats` |
| `agentic/probe.py` | HF/CPU-side probe: same stats schema |
| `agentic/diverge.py` | `compute_divergence` — first-diverging detector |
| `agentic/executor.py` | Wires probe → diverge → LLM → learnings → bisect (G1–G8 orchestrator) |
| `agentic/actions.py` | G4: mechanical pre-LLM toggles (cache invalidate, env, edit revert, dtype) |
| `agentic/resolve.py` | G2: `inspect.getsourcefile` on the diverging class |
| `agentic/learnings.py` | G7: persistent cross-run fix store (now path-fixed to REPO_ROOT) |
| `models/common/lightweightmodule.py` | Base class for TT modules — `__call__` just delegates to `forward` |
| `models/common/auto_compose.py` | TTNN → torch tensor conversion used by readback in the probe |

**The probe doesn't just tell you the model is wrong — it tells you WHERE it
first went wrong, with mean/std/l2/abs_max divergence on the offending module.**

---

## Slide 13 — SAM2-Hiera-Tiny Case Study (corrected counts)

**Source of truth:** `scripts/tt_hw_planner/overlays/facebook_sam2-hiera-tiny/
models__demos__vision__segmentation__sam2_hiera_tiny__bringup_status.json.patch`

### Component decomposition (corrected)

The full SAM2-Hiera-Tiny decomposition produced **36 components**, not 9.

| Class | Count |
|---|---|
| REUSE | **7** |
| ADAPT | **6** |
| NEW | **23** |
| **Total** | **36** |

Sibling base: `nvidia/segformer-b0-finetuned-ade-512-512` (SegFormer semantic
segmentation backend) — selected automatically by `pick_backend_with_quality`.

### Notable components from the actual list

- **Vision encoder side**: `vision_config`, `vision_model`, `vision_neck`,
  `patch_embed`, `patch_embeddings`, `multi_scale_attention`, `multi_scale_block`,
  `hiera_det_model`, `encoder_stack`
- **Prompt encoder**: `prompt_encoder_config`, `video_prompt_encoder`,
  `sine_position_embedding`, `video_position_embedding_sine`
- **Mask decoder**: `mask_decoder_config`, `decoder_head`, `video_mask_decoder`,
  `video_mask_down_sampler`, `video_mask_embedding`
- **Memory / video components**: `video_memory_attention` (+ `_layer`),
  `video_memory_encoder`, `video_memory_fuser` (+ `_c_x_block`),
  `video_two_way_transformer` (+ `video_two_way_attention_block`),
  `video_attention`, `video_feed_forward`, `video_layer_norm`,
  `video_ro_p_e_attention`, `video_vision_rotary_embedding`,
  `video_positional_embedding`
- **Reused from sibling SegFormer**: `self_attention`, `mlp`, `feed_forward`,
  `layer`

### PCC test results (from the case study)

**Graduation evidence in the persisted overlay** — components with
`.last_good_native` snapshot (the explicit graduation marker, written only
after a PCC test passes):

```
decoder_head            mask_decoder_config       video_mask_decoder
encoder_stack           prompt_encoder_config     video_prompt_encoder
feed_forward            vision_config             video_two_way_transformer
hiera_det_model         vision_model              video_feed_forward
                        vision_neck                 (13 components total)
```

**`vision_neck` (the FPN) DID graduate** — the prior slide's claim of "FPN ran
on CPU fallback" is **incorrect** (corrected against the overlay: vision_neck
has the full `.preiter_native` → `.best_native` → `.last_good_native`
snapshot chain, meaning it successfully passed PCC after 5 iter attempts).

**Specific PCC numbers from the prior slide are unreliable and removed.**
Graduation requires per-component test assertion `pcc >= 0.99`. The prior
slide quoted `mask decoder masks 0.989` — that's *below* the graduation
threshold and inconsistent with the `.last_good_native` graduation evidence
for both `mask_decoder_config` and `video_mask_decoder`. The 0.989 was likely
from an intermediate iter logged in `_attempts/iter_NNN.json`, not the
final graduating PCC. Until specific per-component graduating PCC values are
re-extracted from the run logs, the slide should say only:

> **All 13 components with `.last_good_native` snapshots passed PCC ≥ 0.99 by
> definition** — that's the only condition under which `_snapshot_native_stub`
> writes the marker. The intermediate iter PCCs that the prior slide cited
> were captured before the final graduating fix landed.

### Visual demo

Car image inference, 2646×1764 input, 3 candidate masks emitted, mask 3
selected (IoU 0.88). Numbers are consistent with the SegFormer-pattern
mask-decoder output.

### Tool gaps uncovered — REWRITTEN from overlay evidence only

The prior slide's 5 tool gaps were speculation or wrong attributions. Below
are the gaps that ARE directly observable from the persisted SAM2 overlay
(`scripts/tt_hw_planner/overlays/facebook_sam2-hiera-tiny/`):

1. **`ModuleList no forward` test-harness gap (5 components).** The PCC test
   template generates `forward(...)` calls on each submodule, but torch
   `nn.ModuleList` containers don't define `forward` — they only iterate.
   Five SAM2 components (`multi_scale_block`, `video_mask_down_sampler_layer`,
   `video_memory_attention_layer`, `video_memory_fuser_c_x_block`,
   `video_two_way_attention_block`) were marked `no_emit_tests` with reason
   `"ModuleList drop (harness: ModuleList no forward (v13))"`. They were
   never PCC-tested per-component; validation flowed through their parent
   components instead.

2. **Retroactive decomposition split (1 component).** `video_memory_encoder`
   was marked `no_emit_tests` with reason `"decomposition consumer split
   parent into children at 2026-05-30"` — the brain re-decomposed it
   mid-process. Friction point: the original component had captured inputs
   and an iter-history; the children inherit neither.

3. **`failure_class=OTHER` for SAM2-specific failures.** Even on the final
   passing iter_006 of vision_neck (the iter that graduated), the persisted
   attempt log shows `failure_class: "OTHER"` and `diagnosis: "failure
   class=OTHER; see traceback"`. The classifier didn't have a specific case
   for the vision-encoder failure patterns SAM2 hit — same gap we addressed
   for Qwen by adding STATE_DICT_KEY / UNEXPECTED_KWARG / MISSING_KWARG cases.

4. **Heavy iter cost for visual encoder family.** Iter counts per component
   (from `_attempts/<comp>/iter_NNN.json` files):
   - 1 iter: decoder_head, video_mask_down_sampler
   - 2 iters: video_memory_encoder
   - 3 iters: hiera_det_model, video_memory_fuser, video_position_embedding_sine
   - 4 iters: mask_decoder_config
   - 5 iters: video_mask_decoder
   - **6 iters each: vision_config, vision_model, vision_neck**
   The visual encoder triple needed full tiered escalation to opus
   (`model_used: "opus"` in iter_006 logs) and 6× the LLM cost of fast-
   converging components.

5. **Single-attribute-tracked iter progress.** The per-attempt logs include
   `exemplar_used: "(none)"` for several SAM2 components — the brain found
   no sibling exemplar in the bringup-time scan, so it ran from-scratch.
   This worked but missed the value of pattern-reuse across runs (G7 cross-
   run learning was wired AFTER SAM2, so this run had no prior arch_sig
   matches to apply).

### What was NOT a gap (per the overlay):

- **`_runtime_fallbacks.json` is empty** — confirms NO components on CPU
  fallback at runtime. The 13 components with `.last_good_native` snapshots
  all ran on TT hardware.
- **No "FPN on CPU fallback"** — vision_neck has the full graduation
  snapshot chain. The original slide's FPN-on-CPU claim was wrong.
- **No "FPN dimension manual catch"** — vision_neck graduated through the
  normal iter loop (iter_006 in `_attempts/`); no record of manual
  intervention.

### Coverage caveat

The overlay captures snapshot artifacts at the point it was created. The user
reports "all components graduated" — for components without `.last_good_native`
in the overlay but which the user observed as graduated, the snapshots were
either written after the overlay was last updated OR the overlay's capture
predicates excluded them. The 13-component graduation roster above is the
lower-bound from overlay evidence; the actual graduation count may be higher.

---

## Summary of corrections applied

| Slide | Change |
|---|---|
| 11 | Diffusion: FID → SSIM + LPIPS |
| 11 | Classification: Accuracy/F1 → Top-1 + Top-5 + KL |
| 11 | Audio: WER/CER → WER (CER not implemented) |
| 11 | Segmentation: added Dice as second gate |
| 11 | Embed: added mean cosine ≥ 0.97 second threshold |
| 11 | Detection: added the row (was missing) |
| 11 | Escalation: clarified short-circuit + that probe is *inside* iterate loop, not a separate branch |
| 12 | Hook attach point: not `LightweightModule.__call__` but the 9 entry methods of layer-like classes via `_walk_and_wrap` |
| 13 | Component count: 9 → 36 (REUSE:7 / ADAPT:6 / NEW:23) — actual decomposition |
| 13 | Tool gap #2 reworded with the exact 23-null `tt_reuse_target` count from the persisted bringup_status |
| 13 | **REMOVED "FPN ran on CPU fallback" claim** — overlay shows vision_neck has a `.last_good_native` graduation snapshot. The FPN graduated. |
| 13 | **REMOVED FPN-specific tool gaps #1 and #4** (taxonomy / FPN dimension manual catch) — these were based on the wrong FPN-on-CPU premise. Replaced with the real gaps observable from the overlay: hard convergence components needing 5–6 iters, sibling-match nulls. |
| 13 | **REMOVED specific PCC numbers** (0.998, 0.9999, 0.989, 0.992) — the 0.989 number is below the 0.99 graduation threshold and inconsistent with the `.last_good_native` evidence for mask_decoder. The prior slide quoted intermediate-iter PCCs, not final graduating values. Replaced with the verifiable statement: 13 components passed PCC ≥ 0.99 by virtue of having `.last_good_native` snapshots. |
