# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load the reranker checkpoint and convert it to device tensors.

The checkpoint is a HuggingFace Llama state dict under ``model.`` plus a
``score.weight`` row ([1, 2048]). Q and K projection rows are permuted from the
HF rotate-half layout to the interleaved layout ``rotary_embedding_llama``
expects.
"""

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute


def checkpoint_dir(model_id: str) -> Path:
    if os.path.isdir(model_id):
        return Path(model_id)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id, allow_patterns=["*.json", "*.py", "model.safetensors"], local_files_only=os.getenv("CI") == "true"
        )
    )


def load_state_dict(model_id: str) -> dict:
    return load_file(str(checkpoint_dir(model_id) / "model.safetensors"))


def default_cache_dir(model_id: str) -> Path:
    base = os.getenv("TT_CACHE_PATH", os.path.expanduser("~/.cache/tt_nemotron_rerank"))
    return Path(base) / model_id.strip("/").split("/")[-1]


class WeightLoader:
    """Turns host weights into device tensors, caching each one as a tensorbin."""

    def __init__(self, device, cfg, state_dict, cache_dir: Path | None):
        self.device = device
        self.cfg = cfg
        self.sd = state_dict
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _as_tensor(self, name, host, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        cache = None
        if self.cache_dir is not None:
            cache = str(self.cache_dir / f"{name}_{str(dtype).split('.')[-1]}")
        return ttnn.as_tensor(
            host,
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=memory_config,
            cache_file_name=cache,
        )

    def _get(self, key):
        return self.sd[key] if self.sd is not None else None

    def norm(self, name, key):
        w = self._get(key)
        host = None if w is None else w.float().reshape(1, 1, self.cfg.dim // 32, 32)
        return self._as_tensor(name, host, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def linear(self, name, keys, dtype, permute_heads=None):
        """[out, in] HF rows (concatenated over ``keys``) -> [1, 1, in, out] device weight."""
        host = None
        if self.sd is not None:
            parts = []
            for k, heads in zip(keys, permute_heads or [None] * len(keys)):
                w = self._get(k).float()
                if heads is not None:
                    w = reverse_permute(w, heads, w.shape[0], w.shape[1])
                parts.append(w)
            host = torch.cat(parts, dim=0).T.contiguous().unsqueeze(0).unsqueeze(0)
        return self._as_tensor(name, host, dtype)

    def swiglu(self, name, gate_key, up_key, dtype):
        """FF1 (gate) and FF3 (up) as one [1, 1, dim, 2 * hidden] weight for
        ``minimal_matmul(fuse_swiglu=True)``: column tiles alternate gate, up, gate, up, ...
        so the kernel writes silu(gate_tile) * up_tile per output tile."""
        host = None
        if self.sd is not None:
            g = self._get(gate_key).float().T  # [dim, hidden]
            u = self._get(up_key).float().T
            k, n = g.shape
            host = torch.stack([g.reshape(k, n // 32, 32), u.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)
            host = host.unsqueeze(0).unsqueeze(0).contiguous()
        return self._as_tensor(name, host, dtype)

    def embedding(self):
        w = self._get("model.embed_tokens.weight")
        host = None if w is None else w.to(torch.bfloat16)
        return self._as_tensor("embed_tokens", host, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def score_head(self):
        """score.weight [1, dim] placed in column 0 of a [dim, 32] tile-padded matrix."""
        host = None
        w = self._get("score.weight")
        if w is not None:
            host = torch.zeros(1, 1, self.cfg.dim, 32)
            host[0, 0, :, 0] = w.float()[0]
        return self._as_tensor("score_head", host, ttnn.bfloat16)
