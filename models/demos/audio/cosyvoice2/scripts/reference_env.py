# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the scripts that run in the REFERENCE venv (see requirements-reference*.txt).

Never imported by the on-device model or its tests. It makes upstream CosyVoice (FunAudioLLM/CosyVoice @
074ca6dc9e80, the commit CosyVoice1 also pinned) importable and loadable in that venv:

* `COSYVOICE2_REPO` -- the upstream checkout (cloned with --recursive for third_party/Matcha-TTS); it and its
  Matcha-TTS are put on sys.path here.
* `COSYVOICE2_MODEL_DIR` -- the FunAudioLLM/CosyVoice2-0.5B snapshot; if unset, resolved from the local
  Hugging Face cache (populate it with huggingface_hub.snapshot_download first; nothing is downloaded here).
* `LIBRISPEECH_ROOT` -- the directory holding LibriSpeech/test-clean (see corpus.py).

Four compatibility shims (all four also listed in requirements-reference.txt and docs/security.md):

* `load_wav`: upstream reads audio with `torchaudio.load(..., backend='soundfile')`. torchaudio >= 2.9 ignores
  `backend` and requires TorchCodec (plus system FFmpeg). The replacement reads with soundfile -- the same
  libsndfile decode the old backend used, float32 in [-1, 1] -- then applies upstream's exact channel mean and
  `torchaudio.transforms.Resample`, so the waveform upstream sees is unchanged.
* `pyworld`: imported at module level by `cosyvoice.dataset.processor`, which HyperPyYAML imports eagerly while
  resolving cosyvoice2.yaml's training data-pipeline keys. Only `compute_f0` uses it, and only in that training
  pipeline. Every pyworld release that installs on Python 3.10 imports `pkg_resources`, which setuptools >= 82
  no longer ships, and the setuptools advisory CVE-2026-59890 is fixed only from 83.0.0. So pyworld is not
  installed; a stand-in module raises on any attribute access, so a call on the reference path would fail loudly
  rather than silently.
* fp32 Qwen2 backbone: transformers >= 5 loads `from_pretrained` in the checkpoint config's dtype (bfloat16 for
  CosyVoice-BlankEN) where upstream's pinned 4.51 loaded fp32; see `setup_upstream`.
* decode-step attention mask: under transformers >= 5, upstream's non-streaming LLM loop attends only to the first
  position at every decode step and never stops; see `setup_upstream` for the cause and the measurement.
"""
from __future__ import annotations

import os
import sys
import types

UPSTREAM_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
MODEL_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see {__file__})")
    return value


def upstream_repo() -> str:
    return _require_env("COSYVOICE2_REPO")


def librispeech_root() -> str:
    return _require_env("LIBRISPEECH_ROOT")


def model_dir() -> str:
    explicit = os.environ.get("COSYVOICE2_MODEL_DIR")
    if explicit:
        return explicit
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO_ID, local_files_only=True)


def _stub_module(name: str, why: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def __getattr__(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)  # introspection (inspect, hasattr(m, "__file__")) sees an ordinary module
        raise RuntimeError(f"{name}.{attr} used, but {name} is a stand-in in the reference venv: {why}")

    module.__getattr__ = __getattr__
    return module


def _load_wav(wav, target_sr, min_sr=16000):
    import soundfile
    import torch
    import torchaudio

    data, sample_rate = soundfile.read(wav, dtype="float32", always_2d=True)
    speech = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def setup_upstream() -> None:
    """Put upstream on sys.path and install the four shims. Idempotent; call before importing `cosyvoice`."""
    repo = upstream_repo()
    for p in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.modules.setdefault(
        "pyworld", _stub_module("pyworld", "only cosyvoice.dataset.processor.compute_f0 (training) uses it")
    )
    import cosyvoice.cli.frontend as frontend
    import cosyvoice.llm.llm as llm
    import cosyvoice.utils.file_utils as file_utils
    import torch

    file_utils.load_wav = _load_wav
    frontend.load_wav = _load_wav

    # Qwen2Encoder builds its backbone with `Qwen2ForCausalLM.from_pretrained(pretrain_path)` and no dtype.
    # Upstream pins transformers 4.51, which loads fp32; transformers >= 5 loads the checkpoint config's dtype
    # (CosyVoice-BlankEN says bfloat16), so llm.pt's fp32 weights would be copied into bf16 parameters and the
    # fp32 speech embedding would then fail the first matmul. Loading fp32 restores upstream's behaviour.
    if not getattr(llm.Qwen2ForCausalLM.from_pretrained, "_cosyvoice2_fp32", False):
        from_pretrained = llm.Qwen2ForCausalLM.from_pretrained

        def fp32_from_pretrained(*args, **kwargs):
            kwargs.setdefault("dtype", torch.float32)
            return from_pretrained(*args, **kwargs)

        fp32_from_pretrained._cosyvoice2_fp32 = True
        llm.Qwen2ForCausalLM.from_pretrained = staticmethod(fp32_from_pretrained)

    # Qwen2Encoder.forward_one_step passes `masks[:, -1, :]`, the last row of a causal (tril) mask, as the 2D
    # attention mask. That row is all ones, but `inference_wrapper` (the non-streaming loop) sizes it to the step's
    # new input only: a length-1 mask at every decode step. transformers 4.51 dropped an all-ones mask, so each
    # step attended to the whole cache; transformers >= 5 right-pads a short mask with zeros
    # (masking_utils.prepare_padding_mask), so each step attends to position 0 alone. Measured on corpus case 1:
    # over 40 greedy steps, the log-probs differ from a full no-cache forward by up to 16.4 as upstream stands and
    # by at most 3.1e-5 with this shim. Unshimmed, the first two corpus cases each generated until max_len
    # (20 x their text tokens: 180 and 440) without an end-of-speech token.
    # `inference_bistream` already sizes its mask over cache + input; this does the same for every call.
    if not getattr(llm.Qwen2Encoder.forward_one_step, "_cosyvoice2_mask", False):
        forward_one_step = llm.Qwen2Encoder.forward_one_step

        def full_mask_forward_one_step(self, xs, masks, cache=None):
            kv_len = (0 if cache is None else cache.get_seq_length()) + xs.shape[1]
            if masks.shape[-1] != kv_len:
                assert bool(masks[:, -1, :].all()), "upstream passes causal masks only; the last row is all ones"
                masks = torch.ones((1, 1, kv_len), dtype=torch.bool, device=xs.device)
            return forward_one_step(self, xs, masks, cache)

        full_mask_forward_one_step._cosyvoice2_mask = True
        llm.Qwen2Encoder.forward_one_step = full_mask_forward_one_step


def check_upstream_commit() -> str:
    """The checkout's HEAD; warns (does not fail) when it is not the pinned commit."""
    import subprocess

    head = subprocess.run(
        ["git", "-C", upstream_repo(), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        print(f"WARNING: {upstream_repo()} is at {head}, not the pinned {UPSTREAM_COMMIT}", file=sys.stderr)
    return head


def load_upstream():
    """The upstream PyTorch CosyVoice2 (CPU, fp32, no JIT/TensorRT/vLLM), with the shims installed."""
    setup_upstream()
    from cosyvoice.cli.cosyvoice import CosyVoice2

    return CosyVoice2(model_dir(), load_jit=False, load_trt=False, fp16=False)


def versions() -> dict:
    import importlib.metadata as md

    out = {"upstream_commit": check_upstream_commit(), "model_dir": model_dir()}
    for pkg in ("torch", "torchaudio", "onnxruntime", "openai-whisper", "transformers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out
