# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The MTP predictor's free-running CPU reference, and its stored copy.

Causal, so one run over the longest sequence serves every shorter one.
"""

from __future__ import annotations

import functools
import hashlib
import os
import socket
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_3.mtp import glm_mtp_predictor_reference


def predictor_inputs(hidden: int, vocab: int, seq_len: int, num_levels: int, seed: int):
    """``(table, h0, ids)``: the token table, every row's trunk hidden, and the token ids, ``p + 1`` at ``p``."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(vocab, hidden, generator=g).to(torch.bfloat16)
    h0 = torch.randn(seq_len, hidden, generator=g).to(torch.bfloat16)
    return table, h0, list(range(1, seq_len + num_levels + 1))


def compute_predictor_reference(
    config,
    mtp_state_dict,
    layer_state_dict,
    *,
    index_share: bool,
    num_levels: int,
    seed: int,
    vocab: int,
    seq_len: int,
    chunk: int,
    cache_len: int,
):
    """Levels 1..``num_levels`` over ``[0, seq_len)`` in ``chunk`` pieces, each fed the level below's output.

    ``x``, ``out``, ``normed`` are ``[K, L, H]``, ``kvpe`` ``[K, L, 576]``, and ``index_k`` level 1's ``[L, 128]``.
    """
    table, h0, ids = predictor_inputs(config.hidden_size, vocab, seq_len, num_levels, seed)
    token_ids = torch.tensor(ids)
    mla_weights = layer_state_dict["mla_weights"]
    moe_weights = {k: layer_state_dict[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}
    refs = [SparseMLAReference(config, mla_weights, seq_len=cache_len) for _ in range(num_levels)]
    chunks = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        chunks.append(
            glm_mtp_predictor_reference(
                config,
                mla_weights,
                mtp_state_dict,
                layer_state_dict["attn_norm_weight"],
                layer_state_dict["ffn_norm_weight"],
                [table[token_ids[start + k + 1 : end + k + 1]].unsqueeze(0) for k in range(num_levels)],
                h0[start:end].unsqueeze(0),
                cache_len,
                moe_weights=moe_weights,
                num_levels=num_levels,
                index_share=index_share,
                mla_refs=refs,
                actual_start=start,
                actual_end=end,
            )
        )
    x, out, normed = (
        torch.stack([torch.cat([c[part][k][0] for c in chunks]) for k in range(num_levels)]) for part in range(3)
    )
    return {
        "x": x,
        "out": out,
        "normed": normed,
        "kvpe": torch.stack([ref.kvpe_cache[0, 0] for ref in refs]),
        # A copy: the slice is a view of the whole cache_len-row cache, which torch.save would write out.
        "index_k": refs[0].index_cache[0].clone(),
    }


@functools.cache
def stored_reference_path(root: str, prefix: str, checkpoint: str, layer_idx: int, **args) -> Path:
    """The stored reference's path under ``root``: ``prefix``, then a key over ``checkpoint``'s real path, config.json
    and model.safetensors.index.json, ``layer_idx`` and ``args``, the values the reference is computed from."""
    checkpoint = Path(checkpoint).resolve()
    key = hashlib.sha256(str(checkpoint).encode())
    for name in ("config.json", "model.safetensors.index.json"):
        key.update((checkpoint / name).read_bytes())
    key.update(repr((layer_idx, sorted(args.items()))).encode())
    return Path(root) / f"{prefix}_{key.hexdigest()[:16]}.pt"


def load_reference(path: Path) -> dict:
    """The stored reference, memory-mapped: a case reads only the rows it slices."""
    return torch.load(path, mmap=True, weights_only=True)


def prepare_store(path: Path) -> bool:
    """Whether a new file can go at ``path``. A missing directory is created open to all: CI and people share it."""
    try:
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.chmod(0o777)
        return os.access(path.parent, os.W_OK | os.X_OK)
    except OSError:
        return False


def save_reference(ref: dict, path: Path) -> None:
    """Write under a temporary name, then hard-link: a reader never sees a partial file, and a stored copy, which
    other hosts may have mapped, is never replaced."""
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    try:
        torch.save(ref, tmp)
        tmp.chmod(0o644)  # readable by all, whatever the writer's umask
        os.link(tmp, path)
    except FileExistsError:
        pass  # another run stored it first
    finally:
        tmp.unlink(missing_ok=True)
