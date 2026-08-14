# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run a stock ``models.common.readiness_check`` runner against this port.

Two shims, both required, both narrow:

1. **``AutoModelForCausalLM`` registration.**  ``generate.py`` and
   ``run_autoregressive.py`` build the HF reference with
   ``AutoModelForCausalLM.from_pretrained``, and ``muse_glimmer`` is not in
   ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``: its checkpoint declares
   ``MuseGlimmerForConditionalGeneration`` (the multimodal wrapper, which *is* a
   ``GenerationMixin`` whose ``forward`` returns ``.logits``).  Registering the
   config -> class pair is what makes the stock runner work; without it it raises
   *"Unrecognized configuration class"* before touching a device.

2. **Reference dtype.**  Those two runners call ``from_pretrained`` with no
   ``dtype``, which on this checkpoint materialises FP32 -- 112 GB of host RAM and
   twice the memory traffic per token.  ``--hf-dtype`` (default ``bfloat16``, the
   checkpoint's own storage dtype) injects it.

Everything else -- CLI parsing, metrics, artifact paths, thresholds -- is the
stock runner, unmodified.

Usage::

    python doc/full_model/bench/readiness_cli.py <runner-module> [runner args...]

    # e.g.
    python doc/full_model/bench/readiness_cli.py models.common.readiness_check.generate \\
        --hf-model meta-models/Muse-Glimmer-30B --prompt-source aime24 --chat-template \\
        --gen-len 100 --top-k 100 --output <model_dir>/readiness_aime24_chat.refpt
"""

from __future__ import annotations

import runpy
import sys

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def register_muse_glimmer_causal_lm() -> str:
    """Make ``AutoModelForCausalLM`` resolve ``MuseGlimmerConfig``.

    ``AutoModelForCausalLM.register`` cannot do this: ``_LazyAutoMapping.register``
    silently returns for any config whose module starts with ``transformers.``
    (auto_factory.py:680, a guard against remote code hijacking a native config),
    so the public API is a no-op here.  Writing the pair into ``_extra_content``
    is exactly what ``register`` does for a non-native config, and
    ``_LazyAutoMapping.__getitem__`` checks ``_extra_content`` first.
    """
    from transformers import AutoModelForCausalLM
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

    AutoModelForCausalLM._model_mapping._extra_content[MuseGlimmerConfig] = MuseGlimmerForConditionalGeneration
    assert AutoModelForCausalLM._model_mapping[MuseGlimmerConfig] is MuseGlimmerForConditionalGeneration
    return MuseGlimmerForConditionalGeneration.__name__


def resolve_local_snapshot(model_id: str) -> str:
    """The cached snapshot that actually holds the safetensors shards.

    ``refs/main`` for this repo points at a **metadata-only** revision (config,
    tokenizer and the weight index, but no shards), so
    ``from_pretrained("meta-models/Muse-Glimmer-30B")`` resolves to a snapshot
    with no weights and fails deep inside ``safe_open`` with a missing-file error
    that reads like a broken cache.  A snapshot only counts when every shard its
    own index names is present.

    Deliberately duplicated from ``tt/model.py::weights_snapshot_dir`` rather than
    imported: importing that module pulls in ttnn, and the reference-generation
    runner is host-only and must not open or contend for a device.
    """
    import json
    from pathlib import Path

    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    for index_path in sorted(repo.glob("snapshots/*/model.safetensors.index.json")):
        snapshot = index_path.parent
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        if all((snapshot / shard).exists() for shard in shards):
            return str(snapshot)
    raise FileNotFoundError(f"no cached snapshot of {model_id} under {repo} holds all its safetensors shards")


def patch_hf_loaders(dtype: torch.dtype, model_id: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = resolve_local_snapshot(model_id)

    def redirect(args, kwargs):
        args = tuple(snapshot if arg == model_id else arg for arg in args)
        if kwargs.get("pretrained_model_name_or_path") == model_id:
            kwargs["pretrained_model_name_or_path"] = snapshot
        kwargs.setdefault("local_files_only", True)
        return args, kwargs

    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained

    def patched_model(*args, **kwargs):
        args, kwargs = redirect(args, kwargs)
        kwargs.setdefault("dtype", dtype)
        return original_model(*args, **kwargs)

    def patched_tokenizer(*args, **kwargs):
        args, kwargs = redirect(args, kwargs)
        return original_tokenizer(*args, **kwargs)

    AutoModelForCausalLM.from_pretrained = patched_model
    AutoTokenizer.from_pretrained = patched_tokenizer
    return snapshot


def main() -> int:
    argv = sys.argv[1:]
    hf_dtype = "bfloat16"
    model_id = "meta-models/Muse-Glimmer-30B"
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--hf-dtype":
            hf_dtype = argv[index + 1]
            index += 2
            continue
        if token.startswith("--hf-dtype="):
            hf_dtype = token.split("=", 1)[1]
            index += 1
            continue
        if token == "--hf-model" and index + 1 < len(argv):
            model_id = argv[index + 1]
        elif token.startswith("--hf-model="):
            model_id = token.split("=", 1)[1]
        filtered.append(token)
        index += 1
    if not filtered:
        print(__doc__)
        return 2
    module = filtered[0]
    if hf_dtype not in _DTYPES:
        print(f"--hf-dtype must be one of {sorted(_DTYPES)}, got {hf_dtype!r}", file=sys.stderr)
        return 2

    print(f"readiness_cli: registered {register_muse_glimmer_causal_lm()} for AutoModelForCausalLM", flush=True)
    snapshot = patch_hf_loaders(_DTYPES[hf_dtype], model_id)
    print(f"readiness_cli: HF reference dtype={hf_dtype} snapshot={snapshot}", flush=True)
    print(f"readiness_cli: running {module} {' '.join(filtered[1:])}", flush=True)

    sys.argv = [module] + filtered[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
