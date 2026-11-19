# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the installed readiness reference generator with exact mmap HF weights.

Only the HF model factory is replaced: the installed readiness CLI, prompt
rendering, generation, reference extraction and exact-match cache are unchanged.
The original HF text model executes every transformer/PLE operation. PLE lookup
reads exact BF16 checkpoint rows, avoiding a resident 95 GiB embedding. No TTNN
imports, quantization, model-forward substitutions, or shortened model are used.
"""

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"
MODEL_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = MODEL_DIR / "doc/full_model/hf_reference"


class ExactCheckpointRows(torch.nn.Module):
    """Equivalent embedding indexing over the checkpoint's row shards."""

    def __init__(self, snapshot, index, prefix):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        self.snapshot, self.index, self.prefix = snapshot, index, prefix
        self.rows = {}

    def forward(self, ids):
        flat = ids.flatten().tolist()
        by_shard = {}
        for row in sorted(set(flat) - self.rows.keys()):
            shard, local = divmod(row, 2500012)
            by_shard.setdefault(shard, []).append((row, local))
        for shard, locations in by_shard.items():
            key = f"{self.prefix}shard_{shard}.weight"
            with safe_open(self.snapshot / self.index[key], framework="pt") as handle:
                source = handle.get_slice(key)
                for row, local in locations:
                    self.rows[row] = source[local : local + 1, :].clone()[0]
        return torch.stack([self.rows[row] for row in flat]).reshape(*ids.shape, 160)


def load_exact_text_model(model_id=MODEL_ID, **kwargs):
    if model_id != MODEL_ID:
        raise ValueError(f"This pinned loader only supports {MODEL_ID}")
    started = time.monotonic()
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    required = {
        file for key, file in index.items() if key.startswith("model.language_model.") or key == "lm_head.weight"
    }
    missing = sorted(file for file in required if not (snapshot / file).exists())
    if missing:
        raise FileNotFoundError(f"Missing pinned checkpoint shards: {missing}")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    # Same eager attention implementation used by the established HF decoder controls.
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    state = {}
    checkpoint_bytes = 0
    for filename in sorted(required):
        with safe_open(snapshot / filename, framework="pt") as handle:
            for key in handle.keys():
                if ".ngram_embedding.shard_" in key:
                    continue
                if key.startswith("model.language_model."):
                    target = "model." + key.removeprefix("model.language_model.")
                elif key == "lm_head.weight":
                    target = key
                else:
                    continue
                state[target] = handle.get_tensor(key)
                checkpoint_bytes += state[target].numel() * state[target].element_size()
    result = model.load_state_dict(state, strict=False, assign=True)
    ple_layers = [one_based - 1 for one_based in config.ple_layer_ids]
    expected = [f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight" for layer in ple_layers]
    if sorted(result.missing_keys) != sorted(expected) or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint loading contract: {result}")
    for layer in ple_layers:
        prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding."
        model.model.layers[layer].ple.ple_embedding.ngram_embedding = ExactCheckpointRows(snapshot, index, prefix)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
    model.generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    meta = [
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if value.device.type == "meta"
    ]
    if meta:
        raise RuntimeError(f"Unmaterialized model tensors: {meta}")
    model.eval()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "loader.json").write_text(
        json.dumps(
            {
                "hf_model": MODEL_ID,
                "revision": REVISION,
                "snapshot": str(snapshot),
                "model_class": type(model).__name__,
                "layers": len(model.model.layers),
                "checkpoint_parameter_bytes_mmap": checkpoint_bytes,
                "ple_lookup": "exact original BF16 rows from checkpoint; original HF PLE forward",
                "attention_implementation": config._attn_implementation,
                "dtype_policy": "original per-tensor checkpoint dtypes; no casts or quantization",
                "scope": "original HF text autoregressive model; vision and MTP unused",
                "cpu_threads": torch.get_num_threads(),
                "load_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    # Lightweight forward timing makes CPU progress inspectable without changing semantics.
    count = [0]

    def before(module, inputs, call_kwargs):
        module._reference_started = time.monotonic()

    def after(module, inputs, output):
        count[0] += 1
        print(json.dumps({"hf_forward": count[0], "seconds": time.monotonic() - module._reference_started}), flush=True)

    model.register_forward_pre_hook(before, with_kwargs=True)
    model.register_forward_hook(after)
    return model


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    from readiness_check.reference_cache import model_revision

    if model_revision(MODEL_ID) != f"hub:{REVISION}":
        raise RuntimeError("HF cache main alias must resolve to the pinned revision before reference generation")
    # Preserve the package's exact CLI and --model-dir cache behavior.
    AutoModelForCausalLM.from_pretrained = load_exact_text_model
    runpy.run_module("readiness_check.generate", run_name="__main__")


if __name__ == "__main__":
    main()
