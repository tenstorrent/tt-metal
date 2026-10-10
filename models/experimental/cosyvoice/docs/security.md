# Dependency and input-handling review

Four advisories against the reference venv's pins are open and are not closed by a
version bump: three `torch` MEDIUM and one `transformers` HIGH. The next section asks
for a disposition and gives the evidence; the rest is the audit.

## Disposition requested: four open advisories

| advisory | affected function | impact | fixed in |
|---|---|---|---|
| [CVE-2025-3730](https://nvd.nist.gov/vuln/detail/CVE-2025-3730) | `torch.nn.functional.ctc_loss` | denial of service | `2.8.0` |
| [CVE-2025-2999](https://nvd.nist.gov/vuln/detail/CVE-2025-2999) | `torch.nn.utils.rnn.unpack_sequence` | memory corruption | `2.9.1` |
| [CVE-2025-2998](https://nvd.nist.gov/vuln/detail/CVE-2025-2998) | `torch.nn.utils.rnn.pad_packed_sequence` | memory corruption | none recorded; range ends at `<= 2.6.0` |
| [CVE-2026-9856](https://nvd.nist.gov/vuln/detail/CVE-2026-9856) | `PreTrainedTokenizerBase.save_pretrained` / `ProcessorMixin.save_pretrained` | arbitrary file write | `5.10.1` (advisory names `5.10.0`, yanked) |

The three `torch` advisories are CVSS 4.0 base 4.8 MEDIUM, `AV:L/AC:L/PR:L/UI:N`: local
access with existing privileges, no user interaction. Two are memory corruption and one
is denial of service; NVD's own text for CVE-2025-3730 adds "the real existence of this
vulnerability is still doubted at the moment".

CVE-2026-9856 (GHSA-xrqw-3rrv-vx5w, CVSS 7.1 HIGH) is a path traversal:
`PreTrainedTokenizerBase`/`ProcessorMixin.save_pretrained` write a checkpoint's
chat-template name as `<name>.jinja` without checking that it stays inside the target
directory. `5.10.0`, the version the advisory names, was yanked by its authors ("missing
a bunch of fixes"); `5.10.1` carries the fix. It is not reachable: this tree's only
`transformers` import (`scripts/eval_wer_sim.py`) never calls `save_pretrained`, and
upstream CosyVoice@074ca6d's three call sites are all off the reference path — a
CUDA-only Triton export script, a GRPO training example, and a `vllm`-only path that
saves a raw model, not a tokenizer. `transformers` sits in the AR decode loop that
produces every golden, so a bump needs a golden re-run first; none has been done, and
the pin stays at `5.5.0`.

What ships: nothing. `requirements-reference.txt` builds a host-only venv for golden
capture, weight export and WER/speaker scoring. `tt/`, `tests/` and `demo/` import only
`torch`, `numpy`, `ttnn` and `loguru`, all already in tt-metal's `python_env` (*The port
adds no runtime dependencies*).

Reachability. Each `torch` advisory names one function, so whether the reference path
calls it is decidable:

| where | `ctc_loss` | `unpack_sequence` | `pad_packed_sequence` |
|---|---|---|---|
| CosyVoice reference repo (all of it) | — | — | — |
| this port's `scripts/` | — | — | — |
| `openai-whisper` (WER scoring) | — | — | — |
| `transformers` | `WavLMForCTC.forward`, under `if labels is not None` | — | — |
| `torchaudio` | — | — | `models/tacotron2.py` |
| `modelscope` | `trainers/audio/kws_utils` | — | `models/multi_modal/{mmr,prost}` |

None of those call sites is on the reference path. The scoring script instantiates
`WavLMForXVector`, a different class in the same file as `WavLMForCTC`, and never passes
`labels`, so the `ctc_loss` branch does not run. From `torchaudio` it calls `load`,
`save` and `functional.resample`, not Tacotron-2. From `modelscope`, CosyVoice imports
only `snapshot_download`, which reaches neither the video-retrieval models nor the
keyword-spotting trainer. To repeat the check:

```bash
grep -rn "ctc_loss\|unpack_sequence\|pad_packed_sequence" --include="*.py" \
  $COSYVOICE_REPO models/experimental/cosyvoice/scripts
grep -rl "ctc_loss\|unpack_sequence\|pad_packed_sequence" --include="*.py" \
  $COSYVOICE_ENV/lib/python3.10/site-packages
```

Why the `torch` pin stays. Compatibility is not the blocker: `openai-whisper` 20250625
accepts `triton>=2`, `torchaudio` 2.8 keeps its native decoder, and `Qwen2ForCausalLM`
imports cleanly on torch 2.9.1. The blocker is that
`torch.multinomial(probs, num_samples=1)` consumes the RNG stream differently in 2.8
than in 2.6. The batched form, `topk`, `sort` and the generator are byte-identical, and
model arithmetic survives the bump bit-exact, but the token drawn at decode step 2
changes, the utterance ends at 147 semantic tokens instead of 164, and all 29 goldens
shift. Every accuracy figure in `PERF.md` is measured against those goldens, so a bump
is a re-baseline: regenerate the golden set, re-run the PCC suite on both architectures,
and re-measure every derived figure. It changes nothing in the reachability table above.

The disposition requested is one of:

1. Accept the risk and keep both pins — the recommendation. Every finding is in a
   function no reference-path code calls, in a venv the merge does not install.
2. Require a bump — golden set regenerated and every figure re-measured before merge;
   `5.10.1` for `transformers`, not the yanked `5.10.0`. *Reproducing* below is the
   procedure.
3. Require `requirements-reference.txt`'s removal from the PR, at the cost that the
   goldens and the WER/similarity scores stop being reproducible from this tree.

If (1) is granted, please record it on the PR; this document will then carry the
decision and its date, so a later scan meets a written disposition rather than an open
finding.

---

## The port adds no runtime dependencies

Everything that runs on device imports only what tt-metal already ships:

| directory | third-party imports | runs where |
|---|---|---|
| `tt/` | `torch`, `numpy`, `ttnn`, `loguru` | tt-metal `python_env` |
| `tests/` | `torch`, `numpy`, `ttnn`, `pytest` | tt-metal `python_env` |
| `demo/` | `torch`, `numpy`, `ttnn` | tt-metal `python_env` |
| `scripts/` | `cosyvoice`, `hyperpyyaml`, `onnxruntime`, `transformers`, `whisper`, `torchaudio`, `zhconv` | **`cosyvoice_env` only** |

All four of the first group are already in tt-metal's environment, so merging this demo
installs nothing.

`scripts/` is the reference side — golden capture, weight export, front-end
preparation, WER/SIM scoring. It runs on a host, in its own venv, never on device. The
two environments stay separate because installing whisper into tt-metal's `python_env`
pulls in a `triton` that breaks `import torch`.

## Audit findings

41 advisories against the reference venv's pins (from `pip-audit` and the PR's Cycode
scans), plus one SAST finding. None ships: every advisory is confined to the reference
venv, which

- is not installed by anything in `models/experimental/cosyvoice/`,
- is not needed to run the model, the demo, or any test in `tests/pcc`, `tests/e2e` or
  `tests/perf`,
- exists to reproduce goldens and to score audio.

Containment alone does not settle it. `gen_golden.py` `torch.load`s three checkpoints
fetched from ModelScope, so the venv processes third-party bytes, and `weights_only=True`
— passed at all four `torch.load` call sites — was bypassable before torch 2.6.0
(CVE-2025-32434). So the pins move wherever a bump is safe:

| | advisories | disposition |
|---|---|---|
| removed with the package | 1 CRITICAL, 10 HIGH, 6 MODERATE | `gradio`, `onnx` — never imported |
| fixed by a version bump | 3 CRITICAL, 10 HIGH, 7 MODERATE | `torch`, `lightning`, `diffusers`, `pyarrow`, `protobuf`, `modelscope`, `gdown`, `transformers`, `hydra-core` |
| **outstanding** | **1 HIGH, 3 MODERATE** | `transformers`, `torch` ×3 |

37 of 41 are closed; the four open ones are dispositioned at the top of this document.

Every bump taken is checked against the goldens: regenerating the full set from a venv
built clean from `requirements-reference.txt` reproduces all 29 files at worst PCC
`0.9999993220`, with an e2e waveform `max|diff|` of `3.457e-04`. The reference the port
is measured against does not move.

Per package:

- `transformers` 5.5.0 imports cleanly on torch 2.6.0 and reproduces the goldens
  bit-for-bit. torch 2.9.1 on its own breaks the top-level `Qwen2ForCausalLM` import,
  with `transformers` 4.53.0 as well as 5.x.
- `hydra-core` 1.3.2 → 1.3.4 closes CVE-2026-68508 (HIGH, `hydra.utils.instantiate`
  running code from an untrusted config). `hydra` loads on the reference path only as a
  side effect of `matcha/utils/__init__.py`, and `instantiate` is called only from
  `matcha/train.py`, which the reference never runs. With every other pin fixed, the
  regenerated set is bit-exact against the committed goldens, 139/139 arrays.
- `lightning` 2.3.3 → 2.6.6 closes CVE-2026-58659 (HIGH, `LightningModule.load_from_checkpoint`
  running an `_instantiator` import path from the checkpoint even under
  `weights_only=True`); 2.6.6 adds the `_ALLOWED_INSTANTIATORS` allowlist. `lightning` is
  on the reference path only because `matcha/utils/pylogger.py` imports `rank_zero_only`
  from `lightning.pytorch`, a logging decorator with no numeric surface; nothing calls
  `load_from_checkpoint`. Confirm with *Reproducing* before merge, as for the others.
- `onnxruntime` stays at 1.18.0 for numerical reasons, with no advisory against it:
  1.23.2 emits a different token sequence from `speech_tokenizer_v1.onnx`, which
  reroutes the LLM and desynchronises every downstream RNG draw, giving an e2e waveform
  at PCC 0.01.

`pip-audit` cannot audit `torch` and `torchaudio` (`2.6.0+cpu` is not a PyPI version
string), so their advisories come from the Cycode scan. `requirements-reference.txt`
carries the per-package detail.

The SAST finding (unsanitised input in an OS command, `gen_golden.py`) is closed by
removing the call site; see *Input handling*.

A reproduction that needs the reference venv should create it in a container, with the
eval dependencies in a separate virtualenv.

## Input handling

No `subprocess`, no `shell=True`, no `os.system`, no `eval()`, no `exec()`, no `pickle.load` and
no bare `yaml.load` anywhere in the tree.

* No `subprocess`, by rule. `gen_golden.py` records the CosyVoice commit by reading
  `.git/HEAD` (and `packed-refs`, when the ref has been packed) rather than by calling `git`.
  The argv-list form is *not* sufficient grounds to reintroduce one: it rules out shell
  metacharacters, but the path arrives from `--cosyvoice-root`/`$COSYVOICE_REPO`, and `git`
  parses a value beginning with `-` as an option rather than a directory. No child process
  means no argument vector to inject into.
* `module.eval()` in `export_weights.py` and `eval_wer_sim.py` is `torch.nn.Module.eval()`, the
  training-mode switch, not Python's `eval`.
* Every script takes paths through `argparse` and joins them with `os.path.join`.
* Weight and golden files are read with `numpy.load` without `allow_pickle`, so a malformed
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
