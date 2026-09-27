# Dependencies, environments and advisories

Every advisory below is in the **reference venv**, the host-only environment that runs upstream's PyTorch
CosyVoice2, its frontend and the scorer. Nothing in it is part of the on-device model: the model, the demo and
every test run in tt-metal's `python_env`, which this package extends only with `inflect`.
- **torch and setuptools:** one advisory each. Neither is closed by any upgrade that keeps the pins compatible.
- **transformers:** 18 distinct CVEs, 30 OSV records. They come from pinning upstream's own transformers 4.51.3
  (see "Deltas").

None is on a code path the reference venv runs. The next section gives the evidence and asks for a disposition.
The rest of this file describes the environments, the pins, and how the audit was run.

## Disposition requested

### torch and setuptools

| Advisory | Package (pinned) | Affected function / scenario | Fixed in | Why it stays open | Reachable here? |
|---|---|---|---|---|---|
| [CVE-2025-3000](https://osv.dev/vulnerability/GHSA-rrmf-rvhw-rf47) (GHSA-rrmf-rvhw-rf47, LOW, local) | `torch` 2.11.0+cpu | memory corruption in `torch.jit.script` | 2.13.0 | The reference venv pins tt-metal's `python_env` torch (2.11.0+cpu), so the reference and the TTNN port share one torch. `python_env` itself carries the same torch. | **No.** See the first grep below. |
| [CVE-2026-59890](https://osv.dev/vulnerability/GHSA-h35f-9h28-mq5c) (GHSA-h35f-9h28-mq5c) | `setuptools` 81.0.0 | `MANIFEST.in` exclusion bypass when **building an sdist** on macOS APFS/HFS+ (Unicode normalization collision) | 83.0.0 | torch 2.11.0 requires `setuptools<82`; 81.0.0 is the newest release it accepts. | **No.** The flaw is in *creating* an sdist from a source tree on macOS. The reference venv runs on Linux, and nothing in it creates an sdist. |

```bash
# CVE-2025-3000: torch.jit.script is never called on the reference path or by this package.
grep -rnE "torch\.jit\.script" $COSYVOICE2_REPO/cosyvoice --include=*.py
#   cosyvoice/bin/export_jit.py:42  -- JIT export tool, not on the inference path
grep -rnE "torch\.jit" models/experimental/cosyvoice2 --include=*.py
#   (no matches; torch.jit.load in upstream's cli/model.py runs only with load_jit=True, and
#    scripts/reference_env.py loads with load_jit=False)
```

### transformers 4.51.3 (upstream's own pin)

Everything the reference venv does with transformers:
1. **Upstream's LLM backbone:** `Qwen2ForCausalLM.from_pretrained(<snapshot>/CosyVoice-BlankEN)`, from the
   `FunAudioLLM/CosyVoice2-0.5B` snapshot `eec1ae6c79877dbd9379285cf8789c9e0879293d`.
2. **Upstream's frontend tokenizer:** `AutoTokenizer.from_pretrained` on the same directory, plus
   `add_special_tokens`.
3. **The scorer's speaker model:** `WavLMForXVector` and `AutoFeatureExtractor.from_pretrained` on
   `microsoft/wavlm-base-plus-sv`, pinned to revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`
   (`scripts/eval_wer_sim.py`).

Nothing else: no `Trainer`, no `save_pretrained`, no checkpoint-conversion scripts, and no other model class.

| Advisory | Where the flaw is | Fixed in | Reachable here? |
|---|---|---|---|
| [CVE-2026-4372](https://osv.dev/vulnerability/GHSA-29pf-2h5f-8g72) (HIGH) | `from_pretrained` reading a crafted `config.json` whose `_attn_implementation_internal` names a Hub repo; that repo's code is fetched and run | 5.3.0 | **No, with these inputs.** `from_pretrained` reads only the two configs above, both at pinned revisions, and neither has `_attn_implementation_internal`, `auto_map` or `trust_remote_code` (grep below). |
| [CVE-2026-5241](https://osv.dev/vulnerability/GHSA-fgcw-684q-jj6r) (HIGH) | LightGlue model loading overrides `trust_remote_code` | 5.5.0 | No: LightGlue is never loaded. |
| [CVE-2026-9856](https://osv.dev/vulnerability/GHSA-xrqw-3rrv-vx5w) (HIGH) | `save_pretrained` path traversal through chat-template names | 5.10.0 | No: nothing calls `save_pretrained`. |
| [CVE-2026-1839](https://osv.dev/vulnerability/GHSA-69w3-r845-3855) (MODERATE) | `Trainer._load_rng_state` calls `torch.load` unguarded (torch < 2.6) | 5.0.0 | No: no `Trainer`, and torch is 2.11. |
| CVE-2025-14920, -14921, -14924, -14926, -14927, -14928, -14929, -14930 (PYSEC-2025-211…218; no fixed release listed) | Deserialization or code injection in Perceiver, Transformer-XL, megatron_gpt2, SEW / SEW-D / HuBERT `convert_config`, X-CLIP checkpoint conversion and GLM4 weight parsing | none listed | No: none of these models or conversion scripts is used. |
| CVE-2025-3933, CVE-2025-5197, CVE-2025-6051, CVE-2025-6638, CVE-2025-6921 (MODERATE, ReDoS) | `DonutProcessor`; `convert_tf_weight_name_to_pt_weight_name`; `normalize_numbers`; `MarianTokenizer`; `AdamWeightDecay` | 4.52.1–4.53.0 | No: the path uses the Qwen2 tokenizer and the WavLM feature extractor only. |
| CVE-2025-3777 (LOW) | URL validation in `image_utils` | 4.52.1 | No: no image processing. |

```bash
# CVE-2026-4372: neither config the reference venv loads names remote code or an attention implementation.
grep -l "_attn_implementation_internal\|auto_map\|trust_remote_code" \
    $HF_HOME/hub/models--FunAudioLLM--CosyVoice2-0.5B/snapshots/eec1ae6c79877dbd9379285cf8789c9e0879293d/CosyVoice-BlankEN/config.json \
    $HF_HOME/hub/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/config.json
#   (no matches, exit 1)
```

**The alternative the pin replaced.** transformers 5.12.1 (python_env's version) carries none of these advisories.
But upstream misbehaves under it: the LLM loads in bf16, and its decode mask attends to position 0 only. It needed
two behaviour shims in `scripts/reference_env.py` (commit `0d687d840e`). The pin runs upstream unmodified, at the
version it was written for.

## Two environments

| Environment | What runs there | What it adds |
|---|---|---|
| tt-metal `python_env` | The on-device model (`tt/`), the demo and every test under `tests/` | `requirements.txt`: `inflect` (plus `typeguard`); no advisories (OSV, 2026-09-27) |
| Reference venv (host-only, CPU) | `scripts/prepare_inputs.py` (the upstream frontend: ONNX speech tokenizer and CAM++, Whisper log-mel), `scripts/run_reference.py` (the upstream PyTorch CosyVoice2), `scripts/eval_wer_sim.py` (Whisper large-v3 and the WavLM x-vector scorer) | `requirements-reference-torch.txt`, then `requirements-reference.txt` |

The device side never imports `onnxruntime`, `whisper` or the upstream `cosyvoice` package. The two sides exchange
files only: `.npz` prompt inputs, wavs and `results.json`.

The upstream code is `FunAudioLLM/CosyVoice` at `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`, the commit CosyVoice1
also pinned, cloned with `--recursive` (for `third_party/Matcha-TTS`) and pointed to by `COSYVOICE2_REPO`.

## How the reference venv is built, and why in two steps

```bash
uv venv --python 3.10 $COSYVOICE2_REF_ENV
VIRTUAL_ENV=$COSYVOICE2_REF_ENV uv pip install -r requirements-reference-torch.txt   # CPU index only
VIRTUAL_ENV=$COSYVOICE2_REF_ENV uv pip install -r requirements-reference.txt         # PyPI only
uv pip show --python $COSYVOICE2_REF_ENV/bin/python torch torchaudio                 # both: 2.11.0+cpu
```

- **Step 1 takes only torch and torchaudio from the PyTorch CPU index.** With that index as an *extra* index,
  uv's first-index strategy (a dependency-confusion defence) takes every package the index hosts from it. It
  hosts old `requests` 2.28.1, `urllib3` 1.26.13, `certifi` 2022.12.7 and `idna` 3.4, with 32 advisories
  between them. Resolving everything else from PyPI gets current releases without weakening that defence.
- **Step 2 restates `torch==2.11.0` and `torchaudio==2.11.0`.** These match the installed `+cpu` builds under
  PEP 440. Without them, uv re-resolved torch to PyPI's default CUDA build (2.14.0 plus about 15 `nvidia-*`
  packages) when another pin allowed it; this was measured on 2026-09-27. After both steps the venv holds no
  `nvidia-*` or CUDA packages.

## Deltas from upstream's `requirements.txt`

- GPU- and serving-only packages dropped: `deepspeed`, `onnxruntime-gpu` (replaced by `onnxruntime`),
  `tensorrt-cu12*`, `fastapi*`, `uvicorn`, `gradio`, `grpcio*`, `tensorboard`.
- **Only packages the reference path imports are listed.** The import closure was established by loading
  `CosyVoice2` on the real checkpoint and adding each missing module in turn.
- **transformers is upstream's own pin, 4.51.3**, with `tokenizers` 0.21.4 and `huggingface-hub` 0.36.2, not
  python_env's 5.12.1. `scripts/reference_env.py` refuses any other version.
- **Two shims live in `scripts/reference_env.py`:**
  - `load_wav`: torchaudio ≥ 2.9 needs TorchCodec and system FFmpeg for `torchaudio.load`. The shim reads the
    same files with soundfile (the libsndfile decode upstream's old backend used), then applies upstream's own
    channel mean and `Resample`.
  - `pyworld`: imported only by the training data pipeline, which HyperPyYAML imports eagerly. It is replaced by
    a module that raises on any use, because every pyworld release that installs on Python 3.10 needs
    `pkg_resources`.
  - Under the 4.51.3 pin, the two transformers-5 shims that `0d687d840e` added (fp32 load, decode mask) are
    unnecessary. They were removed after these checks, with no shims: every Qwen2 parameter is fp32 and equals
    `llm.pt`; upstream's own decode matches a no-cache forward within 3.1e-5 over 40 greedy steps; and
    `prepare_inputs.py` writes arrays identical to the 5.12.1 run's.
- **`wetext` is deliberately absent.** Upstream then skips it, exactly as the device-side normalizer does, so
  text normalization matches on both sides.
- **`onnxruntime` is held at 1.18.0.** The speech-token sequence depends on the onnxruntime version (CosyVoice1
  measured a different sequence from a newer release), and tokens drive everything downstream.

## How the audit was run

The resolved set (`uv pip freeze`, 109 packages) was queried against the [OSV](https://osv.dev) database
(`https://api.osv.dev/v1/querybatch`, PyPI ecosystem, local version labels stripped), then each advisory's
record (`/v1/vulns/<id>`) was read. The query was re-run after each change on 2026-09-27; the tables above are
the final result, with transformers 4.51.3. Before the pin, with transformers 5.12.1, only the torch and
setuptools advisories were open.
