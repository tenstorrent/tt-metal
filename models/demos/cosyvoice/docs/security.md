# Dependency and input-handling review (R6)

`pip-audit`, run 2026-08-05. The conclusion depends on a distinction the two-environment split
already enforces, so it is stated first.

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

`pip-audit` on `cosyvoice_env` reports advisories in `transformers` 4.51.3 and `urllib3` 1.26.13,
both pulled in transitively by the upstream CosyVoice requirements.

**None of them ship.** They are confined to the reference venv, which:

- is not installed by anything in `models/demos/cosyvoice/`,
- is not needed to run the model, the demo, or any test in `tests/pcc`, `tests/e2e` or `tests/perf`,
- exists to reproduce goldens and to score audio.

The pinned versions come from CosyVoice's own `requirements.txt` and the reference's `torch==2.3.1`
pin; moving off them would change the numbers the goldens encode, which is the opposite of what a
reference environment is for. A reproduction that needs the reference venv should create it in a
container — `tt-oxmiq/Dockerfile.cosy-voice` does exactly that, with the eval dependencies in a
separate `/opt/evalenv` for the same isolation reason.

`torch` and `torchaudio` are reported as un-auditable (`2.3.1+cpu` is not a PyPI version string).

## Input handling

No `shell=True`, no `os.system`, no `eval()`, no `exec()` anywhere in the tree.

* `scripts/gen_golden.py` calls `subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"])`
  — argument-list form, so the path cannot be interpreted as shell syntax.
* `module.eval()` in `export_weights.py` and `eval_wer_sim.py` is `torch.nn.Module.eval()`, the
  training-mode switch, not Python's `eval`.
* Every script takes paths through `argparse` and joins them with `os.path.join`.
* Weight and golden files are read with `numpy.load` **without** `allow_pickle`, so a malformed
  `.npz` cannot execute code.

## Reproducing

```bash
VIRTUAL_ENV=/root/tt/cosyvoice_env uv pip install pip-audit
/root/tt/cosyvoice_env/bin/python -m pip_audit
```
