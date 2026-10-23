# Dependencies, environments and advisories

Two advisories are open against the pinned dependencies after every available upgrade. Neither is closed by a
version bump that keeps the pins compatible. The next section asks for a disposition and gives the evidence. Both
are in the **reference venv**, the host-only environment that runs upstream's PyTorch CosyVoice2, its frontend and
the scorer. Nothing in that venv is part of the on-device model: the model, the demo and every test run in
tt-metal's `python_env`, which this package extends only with `inflect`. A later section explains why the venv runs
transformers 5.12.1 with two shims rather than upstream's own 4.51.3. The rest of this file describes the
environments, the pins and how the audit was run.

## Disposition requested: two open advisories

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

## Why transformers 5.12.1 with two shims, not upstream's 4.51.3

Upstream CosyVoice2 pins transformers 4.51.3. The reference venv runs 5.12.1, python_env's version, and
`scripts/reference_env.py` refuses any other.
- **Why not 4.51.3:** it carries 18 distinct transformers CVEs (30 OSV records, 2026-09-27), 3 of them HIGH. The
  HIGH ones are fixed only in 5.3.0 (CVE-2026-4372, remote code via a crafted `config.json`), 5.5.0 (CVE-2026-5241)
  and 5.10.0 (CVE-2026-9856). None is on a code path the reference venv runs. Even so, carrying them would mean 18
  dispositions for a host-only tool, and a security bot flag on every scan. 5.12.1 carries none.
- **What 5.12.1 changes under upstream, and how it is shimmed:** the shims live in `scripts/reference_env.py`,
  `install_transformers_shims`.
  - `from_pretrained` loads the config's dtype (bf16) where 4.51.3 loaded fp32. The shim passes
    `dtype=torch.float32`.
  - Upstream's non-streaming decode loop passes a length-1 attention mask at each decode step. 4.51.3 dropped it;
    5.x right-pads it with zeros, so every step attends to position 0 alone and generation runs to `max_len`. The
    shim sizes the mask over cache plus input, as upstream's own `inference_bistream` does.
- **Evidence that the shims are exact:**
  - On 2026-09-27, upstream under 4.51.3 with no shims gave **bit-identical** reference output to upstream under
    5.12.1 with the shims: every speech token and every audio sample (max |diff| 0), on all seven corpus cases.
  - `prepare_inputs.py` wrote identical arrays under both.
  - `tests/reference/test_reference_env.py` keeps checking both shims on upstream's own `Qwen2Encoder` with the real
    BlankEN weights, each against a negative control. That file runs as a plain script in the reference venv, which
    has no pytest.
    - fp32 load: the unshimmed load is bf16.
    - The shimmed decode matches a no-cache forward within 1e-4 of the hidden-state scale (measured 2–8e-6); the
      unshimmed decode is off by 1.8.

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
VIRTUAL_ENV=$COSYVOICE2_REF_ENV uv pip install -r requirements-reference.txt \
    -c requirements-reference-lock.txt                                               # PyPI only, locked
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
- **Step 2 is locked.** `requirements-reference-lock.txt` pins all 109 resolved packages, indirect dependencies
  included, as constraints.
  - It was generated from the venv that reproduced the recorded reference on 2026-09-29.
  - Unlocked, three indirect dependencies had drifted since 2026-09-27: msgpack, platformdirs and regex.
  - A fresh venv built with the steps above resolves exactly the locked set.

## Deltas from upstream's `requirements.txt`

- GPU- and serving-only packages dropped: `deepspeed`, `onnxruntime-gpu` (replaced by `onnxruntime`),
  `tensorrt-cu12*`, `fastapi*`, `uvicorn`, `gradio`, `grpcio*`, `tensorboard`.
- **Only packages the reference path imports are listed.** The import closure was established by loading
  `CosyVoice2` on the real checkpoint and adding each missing module in turn.
- **transformers is python_env's 5.12.1** (with `tokenizers` 0.22.2 and `huggingface-hub` 1.33.0), not upstream's
  4.51.3; see the section above.
- **Four shims live in `scripts/reference_env.py`:**
  - `load_wav`: torchaudio ≥ 2.9 needs TorchCodec and system FFmpeg for `torchaudio.load`. The shim reads the
    same files with soundfile (the libsndfile decode upstream's old backend used), then applies upstream's own
    channel mean and `Resample`.
  - `pyworld`: imported only by the training data pipeline, which HyperPyYAML imports eagerly. It is replaced by
    a module that raises on any use, because every pyworld release that installs on Python 3.10 needs
    `pkg_resources`.
  - fp32 Qwen2 backbone and the decode-step attention mask: the two transformers-5 shims (see the section above).
- **`wetext` is deliberately absent.** Upstream then skips it, exactly as the device-side normalizer does, so
  text normalization matches on both sides.
- **`onnxruntime` is held at 1.18.0.** The speech-token sequence depends on the onnxruntime version (CosyVoice1
  measured a different sequence from a newer release), and tokens drive everything downstream.

## How the audit was run

The resolved set (`uv pip freeze`, 109 packages) was queried against the [OSV](https://osv.dev) database
(`https://api.osv.dev/v1/querybatch`, PyPI ecosystem, local version labels stripped), then each advisory's
record (`/v1/vulns/<id>`) was read. The query was re-run after each change on 2026-09-27; the table above is the
final result. Briefly pinning upstream's transformers 4.51.3 raised the count by 30 OSV records (18 CVEs), which is
why 5.12.1 stays.

On 2026-09-29 the same query ran on the locked set (`requirements-reference-lock.txt`). It found the same two
advisories as in the table above (three OSV records, two of them for CVE-2026-59890) and nothing else.
