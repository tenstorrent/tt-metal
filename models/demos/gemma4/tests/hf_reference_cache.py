# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A file cache for host HF reference logits keyed by (model, exact padded token ids, HF dtype).

The bf16 HF forward of the 31B on a CI runner's CPU is ~8 min per full-model test run while the TT side is ~20 s, so
the reference is computed once per prompt and reused from GEMMA4_HF_REF_CACHE (default <TT_CACHE_PATH>/hf_reference,
the shared weight cache the CI jobs mount). No cache dir means compute every time; a write failure keeps the computed
reference. Device-free: torch and the filesystem only.
"""
import hashlib
import os
import tempfile

import torch
from loguru import logger


def hf_reference_cache_dir():
    """GEMMA4_HF_REF_CACHE, else <TT_CACHE_PATH>/hf_reference, else None (compute every time)."""
    explicit = os.environ.get("GEMMA4_HF_REF_CACHE", "").strip()
    if explicit:
        return explicit
    tt_cache = os.environ.get("TT_CACHE_PATH", "").strip()
    return os.path.join(tt_cache, "hf_reference") if tt_cache else None


def hf_reference_key(model_path, input_ids_padded, dtype="bfloat16"):
    """One file per (model, exact padded token ids, HF dtype): the reference is a pure function of those."""
    ids = [int(t) for t in input_ids_padded.reshape(-1).tolist()]
    h = hashlib.sha256(f"{dtype}|{len(ids)}|{','.join(str(t) for t in ids)}".encode()).hexdigest()[:16]
    name = str(model_path).rstrip("/").replace("/", "--")
    return f"{name}__{dtype}__{len(ids)}__{h}.pt"


def hf_reference_logits(model_path, input_ids_padded, compute, dtype="bfloat16"):
    """The cached fp32 reference logits for ``input_ids_padded``, computing them with ``compute()`` on a miss."""
    cache_dir = hf_reference_cache_dir()
    if not cache_dir:
        return compute()
    path = os.path.join(cache_dir, hf_reference_key(model_path, input_ids_padded, dtype))
    if os.path.exists(path):
        saved = torch.load(path, map_location="cpu")
        if torch.equal(saved["input_ids"], input_ids_padded.cpu()):
            logger.info(f"HF reference loaded from the cache: {path}")
            return saved["logits"].float()
        logger.warning(f"HF reference cache entry {path} does not match the prompt; recomputing")
    logits = compute()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # atomic publish: parallel CI jobs on the shared cache must never read a half-written file
        fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".hf_ref_", suffix=".pt")
        os.close(fd)
        torch.save({"input_ids": input_ids_padded.cpu(), "logits": logits.cpu(), "model": str(model_path)}, tmp)
        os.replace(tmp, path)
        logger.info(f"HF reference computed and written to the cache: {path}")
    except OSError as e:  # a read-only or full cache mount keeps the test on the computed reference
        logger.warning(f"HF reference computed; cache write to {cache_dir} failed: {e}")
    return logits
