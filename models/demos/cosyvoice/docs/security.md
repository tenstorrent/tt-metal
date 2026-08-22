# Dependency and input-handling review

`pip-audit` run 2026-08-05; re-worked 2026-08-12 against the Cycode scan on the PR, which
flagged 38 advisories across 10 pinned packages plus one SAST finding. A 39th arrived on
2026-08-22 against `hydra-core` and is closed the same way, by a bump. The conclusion depends
on a distinction the two-environment split already enforces, so it is stated first.

## The port adds no runtime dependencies

Everything that runs on device imports only what tt-metal already ships:

| directory | third-party imports | runs where |
|---|---|---|
| `tt/` | `torch`, `numpy`, `ttnn`, `loguru` | tt-metal `python_env` |
| `tests/` | `torch`, `numpy`, `ttnn`, `pytest` | tt-metal `python_env` |
| `demo/` | `torch`, `numpy`, `ttnn` | tt-metal `python_env` |
| `scripts/` | `cosyvoice`, `hyperpyyaml`, `onnxruntime`, `transformers`, `whisper`, `torchaudio`, `zhconv` | **`cosyvoice_env` only** |

All four of the first group are present in tt-metal's environment already. **Merging this demo
installs nothing.**

`scripts/` is the reference side — golden capture, weight export, front-end preparation, WER/SIM
scoring. It runs once, on a host, in its own venv, and never on device. That boundary is not a
convenience: installing whisper into tt-metal's `python_env` during early bring-up pulled a
`triton` that broke `import torch` outright, which is what established the rule.

## Audit findings, and where they live

**None of them ship.** Every advisory below is confined to the reference venv, which:

- is not installed by anything in `models/demos/cosyvoice/`,
- is not needed to run the model, the demo, or any test in `tests/pcc`, `tests/e2e` or `tests/perf`,
- exists to reproduce goldens and to score audio.

That containment was the whole of the original argument for leaving upstream's pins alone. It is
still true, but it is not sufficient on its own: `gen_golden.py` `torch.load`s three checkpoints
fetched from ModelScope, so the venv does process third-party bytes, and `torch==2.3.1` carried
CVE-2025-32434 — which is precisely the finding that `weights_only=True` was bypassable before
2.6.0. All four `torch.load` call sites already pass `weights_only=True`; on 2.3.1 that did not
help. So the pins moved.

| | advisories | disposition |
|---|---|---|
| removed with the package | 1 CRITICAL, 10 HIGH, 6 MODERATE | `gradio`, `onnx` — never imported |
| fixed by a version bump | 3 CRITICAL, 9 HIGH, 7 MODERATE | `torch`, `lightning`, `diffusers`, `pyarrow`, `protobuf`, `modelscope`, `gdown`, `transformers`, `hydra-core` |
| **outstanding** | **3 MODERATE** | `torch` ×3 |

**36 of 39 closed, including every CRITICAL and every HIGH.** The three that remain are all
`torch` MODERATE local-DoS: CVE-2025-3730 (fixed 2.8.0), CVE-2025-2999 (fixed 2.9.1), and
CVE-2025-2998, whose range ends at `<= 2.6.0` with no fixed version recorded — 2.6.0 is its
last affected release.

Re-measured 2026-08-22, because the reason recorded here was wrong. Neither the `triton` pin
nor the `torchaudio` decoder blocks a bump any more: `openai-whisper` 20250625 relaxes triton
to `>=2` and 3.7.1 resolves against torch 2.8/2.9, `torchaudio` 2.8 keeps its native decoder,
and `Qwen2ForCausalLM` imports cleanly on torch 2.9.1. The real blocker is narrower.
`torch.multinomial(probs, num_samples=1)` consumes the RNG stream differently in 2.8 than in
2.6 — the batched form, `topk` and `sort` are byte-identical, and so is the generator. Model
arithmetic survives the bump bit-exact, but the token drawn at decode step 2 changes, the
utterance ends at 147 semantic tokens instead of 164, and all 29 goldens shift.

That makes a torch bump a re-baseline rather than a numerical risk: the golden set would have
to be regenerated and every figure in PERF.md derived from it re-measured. Three MODERATE
local-DoS findings in a venv that ships nothing do not justify moving the reference the port
is measured against. `requirements-reference.txt` carries the per-package detail.

One correction worth recording, because it changed the outcome. An earlier pass here reported
`transformers` 5.x as incompatible — that `Qwen2ForCausalLM` was no longer importable from the
top-level namespace. That was measured in an environment that also carried torch 2.9.1, and
torch 2.9.1 breaks that same import by itself; `transformers` **4.53.0** fails identically
there. Re-tested against torch 2.6.0, `transformers` 5.5.0 imports cleanly and reproduces the
goldens bit-for-bit, which closed three advisories that had been written up as unfixable.

What made the bumps safe to take is that they are checked, not asserted: regenerating the full
golden set on the new pins reproduces all 29 files at **PCC ≥ 0.9999993**, e2e waveform
`max|diff|` 3.5e-04. Re-verified 2026-08-22 from a venv built clean from this file: worst PCC
`0.9999993220`, e2e `max|diff|` `3.457e-04`. The reference the TTNN port is measured against did not move.

The 39th, CVE-2026-68508 against `hydra-core` (HIGH, `hydra.utils.instantiate` running code
from an untrusted config), is closed by 1.3.2 -> 1.3.4. `hydra` is imported on the reference
path, but only as a side effect of `matcha/utils/__init__.py`; `instantiate` is called only
from `matcha/train.py`, which the reference never runs. The bump is a patch release holding
the same `omegaconf` range, so it costs nothing to take. Checked the same way as the others:
holding every other pin fixed and moving only `hydra-core`, the regenerated set is bit-exact
against the committed goldens, 139/139 arrays — the bump is numerically inert.

One pin is held *back* for the same reason. `onnxruntime` has no advisory, but 1.23.2 emits a
different token sequence from `speech_tokenizer_v1.onnx` than 1.18.0 does; that reroutes the LLM
and desynchronises every downstream RNG draw, giving an e2e waveform at PCC 0.01. Version-pinning
an ONNX runtime is a numerical decision here, not a security one.

A reproduction that needs the reference venv should create it in a container, keeping the eval
dependencies in a separate virtualenv for the same isolation reason.

`torch` and `torchaudio` are reported as un-auditable by `pip-audit` (`2.6.0+cpu` is not a PyPI
version string), so their advisories were tracked from the Cycode scan instead.

The SAST finding (unsanitized input in an OS command, `gen_golden.py`) is closed: the call site
was removed rather than sanitized — see Input handling below. Tracked separately from the
dependency advisories above.

## Input handling

No `subprocess`, no `shell=True`, no `os.system`, no `eval()`, no `exec()`, no `pickle.load` and
no bare `yaml.load` anywhere in the tree.

* **No `subprocess`, by rule.** `gen_golden.py` records the CosyVoice commit by reading
  `.git/HEAD` (and `packed-refs`, when the ref has been packed) rather than by calling `git`.
  The argv-list form is *not* sufficient grounds to reintroduce one: it rules out shell
  metacharacters, but the path arrives from `--cosyvoice-root`/`$COSYVOICE_REPO`, and `git`
  parses a value beginning with `-` as an option rather than a directory. No child process
  means no argument vector to inject into.
* `module.eval()` in `export_weights.py` and `eval_wer_sim.py` is `torch.nn.Module.eval()`, the
  training-mode switch, not Python's `eval`.
* Every script takes paths through `argparse` and joins them with `os.path.join`.
* Weight and golden files are read with `numpy.load` **without** `allow_pickle`, so a malformed
  `.npz` cannot execute code.

## Reproducing

Audit the pins:

```bash
VIRTUAL_ENV=$COSYVOICE_ENV uv pip install pip-audit
$COSYVOICE_ENV/bin/python -m pip_audit
```

Check that a pin change has not moved the reference — the step that makes a bump safe to take,
and the one worth repeating before any future bump:

```bash
export PYTHONPATH=$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS
$COSYVOICE_ENV/bin/python scripts/gen_golden.py --mode zero_shot --out /tmp/golden-check
```

Then PCC every array in `/tmp/golden-check/*.npz` against `tests/golden/`. A bump that reproduces
shows PCC ~1.0 everywhere; one that does not shows near-zero PCC on the noise tensors
(`hift.m_source[call0.out_noise]` is the most sensitive), which means the token sequence changed
and the RNG streams desynchronised rather than that any single module regressed.
