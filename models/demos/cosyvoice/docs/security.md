# Dependency and input-handling review (R6)

`pip-audit` run 2026-08-05; re-worked 2026-08-12 against the Cycode scan on the PR, which
flagged 38 advisories across 10 pinned packages plus one SAST finding. The conclusion depends
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
convenience: installing whisper into tt-metal's `python_env` during P0 pulled a `triton` that broke
`import torch` outright, which is what established the rule.

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
| removed with the package | 1 CRITICAL, 10 HIGH, 6 MEDIUM | `gradio`, `onnx` — never imported |
| fixed by a version bump | 3 CRITICAL, 8 HIGH, 7 MEDIUM | `torch`, `lightning`, `diffusers`, `pyarrow`, `protobuf`, `modelscope`, `gdown`, `transformers` |
| **outstanding** | **3 MEDIUM** | `torch` ×3 |

**35 of 38 closed, including every CRITICAL and every HIGH.** The three that remain are all
`torch` MEDIUM local-DoS: CVE-2025-2998 has no fixed version at any release, and CVE-2025-3730
and CVE-2025-2999 need torch ≥ 2.8, which breaks this venv twice over — torch ≥ 2.8's inductor
imports `triton.backends`, which the `triton` that `openai-whisper` pins does not have, and
`torchaudio` 2.9 removed its native decoder so `torchaudio.load` needs `torchcodec` + FFmpeg.
`requirements-reference.txt` records the reasoning per package; it is the file that has to be
right, so it is the file that carries it.

One correction worth recording, because it changed the outcome. An earlier pass here reported
`transformers` 5.x as incompatible — that `Qwen2ForCausalLM` was no longer importable from the
top-level namespace. That was measured in an environment that also carried torch 2.9.1, and
torch 2.9.1 breaks that same import by itself; `transformers` **4.53.0** fails identically
there. Re-tested against torch 2.6.0, `transformers` 5.5.0 imports cleanly and reproduces the
goldens bit-for-bit, which closed three advisories that had been written up as unfixable.

What made the bumps safe to take is that they are checked, not asserted: regenerating the full
golden set on the new pins reproduces all 29 files at **PCC ≥ 0.9999993**, e2e waveform
`max|diff|` 3.5e-04. The reference the TTNN port is measured against did not move.

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
