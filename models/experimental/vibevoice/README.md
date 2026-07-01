# VibeVoice-1.5B (TT-Metal experimental)

Reference PyTorch setup for porting [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) to TTNN. The backbone is **Qwen2.5-1.5B** (28 layers, hidden 1536, GQA); plan to reuse or wrap [`models/tt_transformers/`](../../tt_transformers/) for `language_model`.

Weights and demo assets are **not** vendored in this tree. On first run, demos and tests download:

- **Model weights:** [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B) into
  `models/experimental/vibevoice/weights/VibeVoice-1.5B` (requires `huggingface_hub`).
- **Demo text + voices:** [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice/tree/main/demo)
  (`demo/text_examples` and `demo/voices`) into `models/experimental/vibevoice/resources/` via
  `common/resource_utils.py`.

Override the checkpoint location with:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B
```

## Layout

```
vibevoice/
├── README.md
├── conftest.py              # pytest: reference/ on PYTHONPATH
├── common/
│   ├── config.py            # paths, HF repo id, transformers pin
│   ├── model_utils.py       # resolve path + auto-download weights
│   └── resource_utils.py    # download demo text/voices from upstream GitHub
├── reference/
│   ├── vibevoice/           # vendored 1.5B-only Python (from VibeVoice repo)
│   ├── model_print.py
│   └── run_inference.py
├── resources/               # auto-downloaded demo assets (gitignored content)
│   ├── voices/              # from github .../demo/voices
│   └── text/                # from github .../demo/text_examples
├── weights/                 # auto-downloaded HF checkpoint (gitignored content)
├── tests/pcc/
│   ├── lm_pcc_common.py       # shared LM PCC helpers, probes, diagnostics
│   ├── test_lm_prefill_pcc.py # prefill hidden-state PCC (+ ISL sweep)
│   ├── test_lm_decode_pcc.py  # full-LM decode after prefill (+ diagnostics)
│   ├── test_decoder_layer_pcc.py  # Devstral-style layer-0 decode (no prefill)
│   └── test_lm_pcc.py         # basic prefill smoke test
└── tt/                      # TTNN layers (empty initially)
```

## Dependencies

Pin `transformers` — **4.57+** breaks `generate()` KV-cache behavior for this model.

```bash
pip install 'transformers==4.51.3' torch accelerate diffusers tqdm librosa scipy huggingface_hub
```

The processor also pulls **Qwen/Qwen2.5-1.5B** tokenizer assets from the Hugging Face cache (`QWEN_TOKENIZER` in `common/config.py`).

## Quick start (from tt-metal root)

```bash
export PYTHONPATH=$(pwd)

# Print architecture (downloads weights on first run)
python models/experimental/vibevoice/reference/model_print.py

# End-to-end reference TTS (CPU default; use --device cuda if available)
python models/experimental/vibevoice/reference/run_inference.py \
  --output_dir /tmp/vibevoice_out

# PCC tests (auto-download weights; skipped if download fails)
pytest models/experimental/vibevoice/tests/pcc/ -v
```

## Language model PCC tests

Prefill and decode hidden-state PCC vs a **bf16 HuggingFace Qwen2** reference (`PCC >= 0.99`).
Shared helpers live in `tests/pcc/lm_pcc_common.py`; fixtures (`vv_config`, `lm_state`) are in
`conftest.py`.

**Regression vs diagnostic:** only tests marked **regression** assert `PCC >= 0.99` and fail CI
when drift appears. **Diagnostic** tests print probes/reports and always pass pytest unless setup
breaks — run them while investigating, not as merge gates.

### Prefill

Compares full-sequence `last_hidden_state` after TT prefill vs HF forward (seed=0).

**Reference dtype:** prefill uses a **bf16-only** HuggingFace Qwen2 reference (`model.to(bfloat16)` in
`reference_lm_forward`). TT prefill also runs in bf16; PCC is computed after promoting both sides to
float32. An fp32 HF reference was tried first and failed at longer ISLs (128+) because the dtype
mismatch inflated the error — switching the reference to bf16 resolved prefill across the full ISL
sweep.

```bash
# Single sequence (S=32)
pytest models/experimental/vibevoice/tests/pcc/test_lm_prefill_pcc.py::test_lm_prefill_hidden_state_pcc -v -s

# ISL sweep: 32, 64, 128, 256, 512, 1024
pytest models/experimental/vibevoice/tests/pcc/test_lm_prefill_pcc.py::test_lm_prefill_hidden_state_pcc_isl_sweep -v -s

# Extended ISL sweep: 32 … 65536 with HF/TT wall-time per length (may take hours)
pytest models/experimental/vibevoice/tests/pcc/test_lm_prefill_pcc.py::test_lm_prefill_hidden_state_pcc_isl_sweep_extended_with_timing -v -s
```

| Test | Type | Asserts PCC? | Status (Blackhole, seed=0) |
|------|------|--------------|----------------------------|
| `test_lm_prefill_hidden_state_pcc` | regression | yes (overall S=32) | **pass** (~0.997, bf16 HF ref) |
| `test_lm_prefill_hidden_state_pcc_isl_sweep` | regression | yes (overall per ISL) | **pass** (ISL 32–1024 overall ≥ 0.99, bf16 HF ref) |

Per-token minima can dip below 0.99 (e.g. ISL 1024 has a bad token at p956) while **overall**
sequence PCC still passes the threshold.

### Decode

Two decode PCC modes (mirroring Devstral's split between layer decode and full-model paths):

**Devstral-style (layer 0, recommended decode regression):** random hidden states `[1, 1, H]`,
empty KV cache, positions **0–9**, no prefill. Isolates decode SDPA at low cache depth.

```bash
pytest models/experimental/vibevoice/tests/pcc/test_decoder_layer_pcc.py::test_decoder_layer_decode_pcc -v -s
```

**Full-LM integration:** decode-step hidden states after a 32-token prefill (seed=0). Uses fused
`scaled_dot_product_attention_decode` on TT; HF reference uses `attn_implementation="sdpa"`.

```bash
# Single decode step (step 0, position 32)
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_hidden_state_pcc -v -s

# 10-step decode sweep after 32-token prefill (full-LM integration)
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_multi_step_hidden_state_pcc -v -s
```

| Test | Type | Scope | Asserts PCC? | Status (Blackhole, seed=0) |
|------|------|-------|--------------|----------------------------|
| `test_decoder_layer_decode_pcc` | regression | layer 0, pos 0–9, no prefill | yes (all 10 steps) | **pass** (min 0.99997) |
| `test_lm_decode_hidden_state_pcc` | regression | full LM, step 0 @ pos 32 | yes | **pass** |
| `test_lm_decode_multi_step_hidden_state_pcc` | integration | full LM, pos 32–41 | yes (all 10 steps) | **fail** at steps 7–8 (~0.988–0.990); steps 0–6, 9 pass |

**Known issue:** multi-step failure is traced to fused decode SDPA (manual fp32 SDPA monkeypatch
passes all 10 steps). Diagnostic tests below print localization data; they do **not** assert the
fused path meets 0.99:

```bash
# Layer-wise / L0 attention / SDPA stage probes at failing step 7
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_layerwise_pcc_at_failing_step -v -s
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_l0_attention_stage_pcc_at_failing_step -v -s
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_l0_sdpa_stage_pcc_at_failing_step -v -s

# Manual fp32 SDPA vs fused (monkeypatch); HF reference mode comparison
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_multi_step_pcc_manual_fp32_sdpa_diagnostic -v -s
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_hf_reference_attn_comparison_diagnostic -v -s

# Stage-wise fused vs manual fp32 SDPA report at step 7 (position 39)
pytest models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_fused_vs_manual_sdpa_report_at_step_7 -v -s
```

| Test | Type | Asserts PCC? | Status |
|------|------|--------------|--------|
| `test_lm_decode_layerwise_pcc_at_failing_step` | diagnostic | no (print only) | pass |
| `test_lm_decode_l0_attention_pcc_at_failing_step` | diagnostic | no | pass |
| `test_lm_decode_l0_attention_stage_pcc_at_failing_step` | diagnostic | no | pass |
| `test_lm_decode_l0_sdpa_stage_pcc_at_failing_step` | diagnostic | no | pass |
| `test_lm_decode_multi_step_pcc_manual_fp32_sdpa_diagnostic` | diagnostic | no (proves manual path passes) | pass |
| `test_lm_decode_hf_reference_attn_comparison_diagnostic` | diagnostic | no | pass |
| `test_lm_decode_fused_vs_manual_sdpa_report_at_step_7` | diagnostic | no (report only) | pass |

Run **regression** gates only (expect decode multi-step to fail until fused SDPA is fixed):

```bash
pytest models/experimental/vibevoice/tests/pcc/test_lm_prefill_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_decoder_layer_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py::test_lm_decode_hidden_state_pcc -v
```

Run all LM prefill + decode tests (including diagnostics):

```bash
pytest models/experimental/vibevoice/tests/pcc/test_lm_prefill_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_lm_decode_pcc.py -v
```

Basic smoke test (prefill only, legacy entry point):

```bash
pytest models/experimental/vibevoice/tests/pcc/test_lm_pcc.py -v
```

## Porting notes

| Submodule | Reference | TT target |
|-----------|-----------|-----------|
| Language model | Qwen2 in `modeling_vibevoice` | `tt_transformers` Qwen2.5-1.5B |
| Acoustic / semantic tokenizers | `modular_vibevoice_tokenizer.py` | `tt/` (later) |
| Diffusion head | `modular_vibevoice_diffusion_head.py` | `tt/` (later) |
| Pipeline | `modeling_vibevoice_inference.py` | `tt/` generate loop |

Closest template: [`models/experimental/speecht5_tts/`](../speecht5_tts/) (`reference/` = PyTorch gold, `tt/` = TTNN, `tests/pcc/` = PCC).
