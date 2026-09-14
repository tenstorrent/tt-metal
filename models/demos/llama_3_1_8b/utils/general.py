# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small package-local helpers: state-dict slicing, weight-cache paths, mesh link count.

Deliberately tiny and dependency-free. The genuinely shared helpers named by the recipe
(``get_num_dram_banks``, the DeepSeek golden-cache helpers) are IMPORTED at their use sites, not
re-implemented here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import ttnn


def substate(state_dict: dict, prefix: str) -> dict:
    """The sub-dict under ``prefix.``, with the prefix stripped. Empty in cache-only mode."""
    if not state_dict:
        return {}
    p = prefix + "."
    return {k[len(p) :]: v for k, v in state_dict.items() if k.startswith(p)}


def cache_name(base: Optional[str], name: str) -> Optional[str]:
    """Join a weight-cache path fragment, or ``None`` when caching is off.

    ``ttnn.as_tensor(cache_file_name=None)`` means "do not cache", so threading ``None`` through the
    module tree is how a random-weight unit test avoids writing tilized junk to disk.
    """
    return None if base is None else str(Path(base) / name)


def default_num_links(mesh_device) -> int:
    """Fabric links per CCL call. ``LLAMA_NUM_LINKS`` overrides; 2 is what a BH galaxy exposes."""
    env = os.getenv("LLAMA_NUM_LINKS")
    if env:
        return int(env)
    try:
        return max(1, ttnn.get_num_links(mesh_device))
    except AttributeError:
        return 2


def weight_cache_dir(mesh_shape, dtype=ttnn.bfloat8_b) -> Path:
    """Where tilized weights are cached, keyed on mesh shape and dtype.

    Defaults under ``$TT_METAL_HOME/generated`` rather than beside the checkpoint: the shared
    ``/mnt/models`` store is other-user-owned NFS and a bring-up must not write into it (and must not
    modify the preparation inputs). ``TT_CACHE_PATH`` overrides.
    """
    base = os.getenv("TT_CACHE_PATH")
    root = Path(base) if base else Path(os.environ.get("TT_METAL_HOME", ".")) / "generated" / "llama_3_1_8b"
    tag = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
    path = root / f"tensor_cache_{tag}_{tuple(mesh_shape)}"
    path.mkdir(parents=True, exist_ok=True)
    return path
