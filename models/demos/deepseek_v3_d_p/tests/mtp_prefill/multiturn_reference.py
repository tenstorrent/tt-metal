# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The free-running CPU reference ``test_mtp_multiturn_pcc`` checks, and its stored copy.

Causal, so one run over the longest conversation serves every schedule. A pretrained case that finds no stored copy
computes every level and stores it when ``$TT_GLM53_MTP_REF_CACHE`` is writable; to build it without a device, run
``python3 -m models.demos.deepseek_v3_d_p.tests.mtp_prefill.multiturn_reference``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import os
import socket
import time
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_3.mtp import glm_mtp_predictor_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config, glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.tests.mtp_prefill.conftest import HF_ENV, glm53_checkpoint_path, glm_layer_state_dict
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.mtp_config import MTPConfig
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import enable_mtp_indexer_slot, load_mtp_state_dict

CHUNK = 5 * 1024
CACHE_LEN = 3 * CHUNK
REFERENCE_LEN = 11520  # the longest schedule, multichunk
MAX_LEVELS = 7
INPUT_SEED = 11
# One tile past the cache. h0 is drawn after the table, so a change here changes every input.
VOCAB = CACHE_LEN + 32
REF_CACHE_ENV = "TT_GLM53_MTP_REF_CACHE"
# Bump in any commit that changes the reference's numerics, so a stored copy stops matching.
REFERENCE_VERSION = 1


def _glm53_config_for_mtp(config_only, seq_len: int, layer_idx: int):
    """A GLM-5.3 config with the MTP layer's indexer slot declared, safe to mutate.

    ``copy.copy`` because ``config_only`` is lru_cached; GLM-5.3's own map stops at the trunk.
    """
    config = copy.copy(config_only)
    config.max_seq_len = seq_len
    enable_mtp_indexer_slot(config, layer_idx)
    return config


def multiturn_inputs(hidden: int):
    """``(table, h0, ids)``: the token table, every row's trunk hidden, and the token ids, ``p + 1`` at ``p``."""
    g = torch.Generator().manual_seed(INPUT_SEED)
    table = torch.randn(VOCAB, hidden, generator=g).to(torch.bfloat16)
    h0 = torch.randn(REFERENCE_LEN, hidden, generator=g).to(torch.bfloat16)
    return table, h0, list(range(1, REFERENCE_LEN + MAX_LEVELS + 1))


def compute_multiturn_reference(config, mtp_state_dict, layer_state_dict, *, index_share: bool, num_levels: int):
    """Levels 1..``num_levels`` over ``[0, REFERENCE_LEN)`` in ``CHUNK`` pieces, each fed the level below's output.

    ``x``, ``out``, ``normed`` are ``[K, L, H]``, ``kvpe`` ``[K, L, 576]``, and ``index_k`` level 1's ``[L, 128]``.
    """
    table, h0, ids = multiturn_inputs(config.hidden_size)
    token_ids = torch.tensor(ids)
    mla_weights = layer_state_dict["mla_weights"]
    moe_weights = {k: layer_state_dict[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}
    refs = [SparseMLAReference(config, mla_weights, seq_len=CACHE_LEN) for _ in range(num_levels)]
    chunks = []
    for start in range(0, REFERENCE_LEN, CHUNK):
        end = min(start + CHUNK, REFERENCE_LEN)
        chunks.append(
            glm_mtp_predictor_reference(
                config,
                mla_weights,
                mtp_state_dict,
                layer_state_dict["attn_norm_weight"],
                layer_state_dict["ffn_norm_weight"],
                [table[token_ids[start + k + 1 : end + k + 1]].unsqueeze(0) for k in range(num_levels)],
                h0[start:end].unsqueeze(0),
                CACHE_LEN,
                moe_weights=moe_weights,
                num_levels=num_levels,
                index_share=index_share,
                mla_refs=refs,
                actual_start=start,
                actual_end=end,
            )
        )
    x, out, normed = (
        torch.stack([torch.cat([chunk[part][k][0] for chunk in chunks]) for k in range(num_levels)])
        for part in range(3)
    )
    return {
        "x": x,
        "out": out,
        "normed": normed,
        "kvpe": torch.stack([ref.kvpe_cache[0, 0] for ref in refs]),
        # A copy: the slice is a view of the whole CACHE_LEN-row cache, which torch.save would write out.
        "index_k": refs[0].index_cache[0].clone(),
    }


@functools.cache
def stored_reference_path(layer_idx: int) -> Path | None:
    """The stored pretrained reference's path under ``$TT_GLM53_MTP_REF_CACHE``, keyed by checkpoint and inputs.
    None when the variable or the checkpoint is not set."""
    root, checkpoint = os.environ.get(REF_CACHE_ENV), glm53_checkpoint_path()
    if not (root and checkpoint):
        return None
    checkpoint = Path(checkpoint).resolve()
    key = hashlib.sha256(str(checkpoint).encode())
    for name in ("config.json", "model.safetensors.index.json"):
        key.update((checkpoint / name).read_bytes())
    key.update(f"{layer_idx} {INPUT_SEED} {REFERENCE_LEN} {MAX_LEVELS} {REFERENCE_VERSION}".encode())
    return Path(root) / f"glm53_mtp_multiturn_ref_{key.hexdigest()[:16]}.pt"


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


def main():
    checkpoint = glm53_checkpoint_path()
    if checkpoint is None or not os.environ.get(REF_CACHE_ENV):
        raise SystemExit(f"set {HF_ENV} to the GLM-5.3 checkpoint and {REF_CACHE_ENV} to the reference's directory")
    mtp_cfg = MTPConfig.from_pretrained(checkpoint)
    layer_idx = mtp_cfg.mtp_layer_idx
    path = stored_reference_path(layer_idx)
    if path.exists():
        logger.info(f"{path} is already stored; delete it to rebuild")
        return
    if not prepare_store(path):
        raise SystemExit(f"cannot write to {path.parent}")
    t0 = time.monotonic()
    config_only = glm_5_3_hf_config()
    mtp_state_dict = {k: v.to(torch.bfloat16) for k, v in load_mtp_state_dict(checkpoint, layer_idx=layer_idx).items()}
    layer_state_dict = glm_layer_state_dict(config_only, checkpoint, layer_idx, GLM53Config.NUM_ROUTED_EXPERTS)
    logger.info(f"layer {layer_idx} of {checkpoint} loaded in {time.monotonic() - t0:.0f}s; {MAX_LEVELS} levels next")
    ref = compute_multiturn_reference(
        _glm53_config_for_mtp(config_only, CACHE_LEN, layer_idx),
        mtp_state_dict,
        layer_state_dict,
        index_share=mtp_cfg.index_share_for_mtp_iteration,
        num_levels=MAX_LEVELS,
    )
    save_reference(ref, path)
    logger.info(f"stored {path}, {path.stat().st_size / 2**30:.2f} GiB, in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
