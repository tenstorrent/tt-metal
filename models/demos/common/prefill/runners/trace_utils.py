# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free reference reuse policy; generators and cache validation are model-specific."""

import json
from pathlib import Path


def ensure_trace(trace_dir, seq_len, generate, *, validate=None):
    """Reuse a complete prefix or call ``generate()`` for a missing/short reference.

    The generator returns its output directory. It must not overwrite a shared
    input trace. Optional validation checks the model's actual cache files;
    missing files trigger generation, while incompatible data raises an error.
    """
    if seq_len <= 0:
        raise ValueError("reference sequence length must be positive")

    def available(path):
        try:
            metadata = json.loads((path / "metadata.json").read_text())
            if len(metadata["token_ids"]) < seq_len:
                return False
            if validate is not None:
                validate(path, seq_len)
            return True
        except FileNotFoundError:
            return False

    trace = Path(trace_dir)
    if available(trace):
        print(f"Reusing golden prefix: {trace} ({seq_len} tokens)", flush=True)
        return trace
    print(f"Golden missing, incomplete or shorter than {seq_len} tokens: {trace}", flush=True)
    generated = Path(generate())
    if not available(generated):
        raise ValueError(f"generated reference does not cover {seq_len} tokens: {generated}")
    return generated


def golden_key_frame(trace_dir):
    """Normalize metadata used by HF and Meta GQA exporters; older traces default to HF."""
    metadata = json.loads((Path(trace_dir) / "metadata.json").read_text())
    aliases = {"hf": "hf", "hf_half_split": "hf", "meta": "meta", "interleaved": "meta"}
    frames = [metadata[key] for key in ("rope_frame", "key_rotary_frame") if key in metadata]
    if not frames:
        return "hf"
    if any(frame not in aliases for frame in frames) or len({aliases[frame] for frame in frames}) != 1:
        raise ValueError(f"unsupported or conflicting golden key frame: {frames}")
    return aliases[frames[0]]


def validate_gqa_trace(trace_dir, seq_len, *, num_layers, num_kv_heads, head_dim):
    """Check layer keys and saved capacity from safetensors headers without loading the KV tensors."""
    from safetensors import safe_open

    trace = Path(trace_dir)
    golden_key_frame(trace)
    for layer in range(num_layers):
        with safe_open(str(trace / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as handle:
            for kind in ("key", "value"):
                shape = handle.get_slice(f"{kind}_cache_layer_{layer}").get_shape()
                if len(shape) != 4 or shape[:2] != [1, num_kv_heads] or shape[3] != head_dim:
                    raise ValueError(f"layer {layer}: incompatible {kind} cache shape {shape}")
                if shape[2] < seq_len:
                    raise FileNotFoundError(f"layer {layer}: {kind} cache has {shape[2]} rows, needs {seq_len}")
