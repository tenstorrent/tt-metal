# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Redirect Hugging Face downloads away from a full ``~`` (e.g. NFS ``/proj_sw`` with space).

Also centralizes ``HF_MODULES_CACHE`` (``transformers`` dynamic modules) under the same tree as
``HF_HOME`` so imports do not try to create ``~/.cache/huggingface/modules`` on a full disk.

Patches local sharded ``*.safetensors.index.json`` files that omit ``metadata`` (required by recent
``transformers`` when loading shard maps; some Hub repos only ship ``weight_map``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _parse_hf_home_from_argv(argv: list[str]) -> Path | None:
    for i, a in enumerate(argv):
        if a == "--hf-home" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if a.startswith("--hf-home="):
            return Path(a.split("=", 1)[1].strip()).expanduser().resolve()
    return None


def bootstrap_hf_env_before_transformers(argv: list[str], *, default_hf_home: Path) -> Path:
    """Set ``HF_HOME`` / ``HF_HUB_CACHE`` / ``HF_MODULES_CACHE`` before ``import transformers``.

    ``transformers`` dynamic modules use ``HF_MODULES_CACHE`` at import time; if it defaults
    under a full ``/home``, ``makedirs`` can fail with ENOSPC. Call this **before** importing
    any module that pulls in ``transformers`` (e.g. ``trace_replay_base`` → ``AutoTokenizer``).

    Uses :func:`os.environ.setdefault` so a user-exported ``HF_HOME`` still wins.
    """
    parsed = _parse_hf_home_from_argv(argv)
    root = (
        parsed
        if parsed is not None
        else Path(os.environ.get("HF_HOME", str(default_hf_home))).expanduser().resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    mod = root / "modules"
    hub.mkdir(parents=True, exist_ok=True)
    mod.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("HF_MODULES_CACHE", str(mod))
    return root


def set_hf_home(path: str | Path) -> Path:
    """Set ``HF_HOME`` / ``HF_HUB_CACHE`` / ``HF_MODULES_CACHE`` for Hub + ``transformers`` dynamic code.

    **Note:** ``huggingface_hub`` reads ``HF_HUB_CACHE`` when the package is first imported.
    Scripts that ``import huggingface_hub`` at module load time must pass explicit
    ``cache_dir=...`` to ``snapshot_download`` / ``hf_hub_download`` (see NextN run scripts).
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    hub = p / "hub"
    mod = p / "modules"
    hub.mkdir(parents=True, exist_ok=True)
    mod.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(p)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_MODULES_CACHE"] = str(mod)
    return p


def ensure_nextn_snapshot_has_modeling_deepseek(snapshot_dir: Path, *, hub_cache: Path) -> None:
    """If ``snapshot_dir`` is a NextN-style Hub layout missing ``modeling_deepseek.py``, fetch it.

    ``lmsys/*-NextN`` checkpoints reference ``modeling_deepseek`` in ``config.json`` ``auto_map`` but do
    not include that file on the Hub. ``SGLang/DeepSeek-V3-NextN`` provides the same
    ``configuration_deepseek.py`` blob and ships ``modeling_deepseek.py`` for ``trust_remote_code`` loads.

    Also patches ``model.safetensors.index.json`` if it lacks ``metadata`` (see
    :func:`ensure_sharded_safetensors_index_has_metadata`).
    """
    snap = snapshot_dir.expanduser().resolve()
    ensure_sharded_safetensors_index_has_metadata(snap)
    modeling = snap / "modeling_deepseek.py"
    if modeling.is_file() and modeling.stat().st_size > 128:
        return
    cfg_path = snap / "config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    auto_map = cfg.get("auto_map")
    if not isinstance(auto_map, dict):
        return
    if not any(isinstance(v, str) and "modeling_deepseek" in v for v in auto_map.values()):
        return

    from huggingface_hub import hf_hub_download

    from models.demos.speculative_deepseek_r1_broad.default_paths import NEXTN_MODELING_AUX_REPO_ID

    hub_cache = hub_cache.expanduser().resolve()
    try:
        hf_hub_download(
            repo_id=NEXTN_MODELING_AUX_REPO_ID,
            filename="modeling_deepseek.py",
            local_dir=str(snap),
            cache_dir=str(hub_cache),
        )
    except Exception as e:
        raise RuntimeError(
            f"NextN snapshot at {snap} needs modeling_deepseek.py (listed in config auto_map but "
            f"omitted from the weight Hub repo). Tried to download it from {NEXTN_MODELING_AUX_REPO_ID!r}. "
            f"Original error: {e}"
        ) from e


def ensure_sharded_safetensors_index_has_metadata(model_dir: str | Path) -> int:
    """Patch shard index JSON under ``model_dir`` if it has ``weight_map`` but no ``metadata``.

    Newer ``transformers`` ``get_checkpoint_shard_files`` expects ``index[\"metadata\"]`` and mutates
    it (adds ``all_checkpoint_keys``, etc.). Repos such as ``lmsys/DeepSeek-R1-NextN`` publish an index
    with only ``weight_map``, which causes ``KeyError: 'metadata'``.

    Returns the number of index files updated (0 if none needed).
    """
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        return 0
    names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.fp16.safetensors.index.json",
    )
    patched = 0
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "weight_map" not in data:
            continue
        if isinstance(data.get("metadata"), dict):
            continue
        data["metadata"] = {}
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        patched += 1
    return patched
