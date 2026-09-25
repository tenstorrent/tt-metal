# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host reader for the Mistral-Medium-3.5 HF checkpoint (torch + safetensors only).

The checkpoint stores the language model under ``model.language_model.`` (plus ``lm_head.weight``); the
vision tower and projector are skipped. Linear weights are ``float8_e4m3fn`` with one rank-0
``<name>_scale_inv`` each (``quantization_config``: ``quant_method=fp8``, ``weight_block_size=null``):

    w = w_fp8.float() * weight_scale_inv        (then bf16)

``*_activation_scale`` entries belong to the static activation-quantization scheme and are dropped.
A float8 tensor without its scale fails loudly. This matches the golden trace's dequantization
bit-exactly (``tests/test_golden_hf_checkpoint.py``).
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

LM_PREFIX = "model.language_model."


def dequantize_fp8_per_tensor(tensors: dict, dtype=torch.bfloat16) -> dict:
    """Sorted walk over ``tensors``: apply each weight's rank-0 ``_scale_inv``, drop scale entries."""
    out = {}
    for name in sorted(tensors):
        if name.endswith("_scale_inv") or name.endswith("activation_scale"):
            continue
        t = tensors[name]
        scale = tensors.get(f"{name}_scale_inv")
        if t.dtype == torch.float8_e4m3fn:
            if scale is None:
                raise ValueError(f"float8 tensor {name!r} has no {name}_scale_inv")
            if scale.ndim != 0:
                raise ValueError(f"{name}_scale_inv must be a per-tensor scalar, got shape {tuple(scale.shape)}")
            out[name] = (t.float() * scale.float()).to(dtype)
        else:
            assert scale is None, f"non-fp8 tensor {name!r} carries a scale"
            out[name] = t.to(dtype)
    return out


class CheckpointReader:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]

    def names(self, prefix: str):
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def read(self, names) -> dict:
        """Raw tensors (checkpoint dtype) for ``names``, one safetensors open per shard file."""
        by_file = defaultdict(list)
        for n in names:
            by_file[self.weight_map[n]].append(n)
        out = {}
        for fname, keys in by_file.items():
            with safe_open(str(self.path / fname), framework="pt") as f:
                for k in keys:
                    out[k] = f.get_tensor(k)
        return out

    def dequantized(self, prefix: str, dtype=torch.bfloat16) -> dict:
        """Every tensor under ``prefix``, dequantized, with ``prefix`` stripped from the names."""
        names = self.names(prefix)
        assert names, f"no checkpoint tensors under {prefix!r}"
        deq = dequantize_fp8_per_tensor(self.read(names), dtype)
        return {k[len(prefix) :]: v for k, v in deq.items()}

    def layer_state_dict(self, i: int, dtype=torch.bfloat16) -> dict:
        """Decoder layer ``i`` in ReferenceDecoderLayer naming (``self_attn.q_proj.weight``, ...)."""
        return self.dequantized(f"{LM_PREFIX}layers.{i}.", dtype)

    def embedding(self, dtype=torch.bfloat16):
        return self.dequantized(f"{LM_PREFIX}embed_tokens.", dtype)["weight"]

    def final_norm(self, dtype=torch.bfloat16):
        return self.dequantized(f"{LM_PREFIX}norm.", dtype)["weight"]

    def lm_head(self, dtype=torch.bfloat16):
        return self.dequantized("lm_head.", dtype)["weight"]

    def iter_layers(self, indices, dtype=torch.bfloat16, prefetch: int = 3):
        """Yield ``(i, layer_state_dict)`` in order, reading up to ``prefetch`` layers ahead on worker
        threads (the checkpoint read, not the dequant, dominates on NFS)."""
        indices = list(indices)
        with ThreadPoolExecutor(max_workers=max(1, prefetch)) as pool:
            pending = {}
            for n, i in enumerate(indices):
                for j in indices[n : n + prefetch + 1]:
                    if j not in pending:
                        pending[j] = pool.submit(self.layer_state_dict, j, dtype)
                yield i, pending.pop(i).result()
