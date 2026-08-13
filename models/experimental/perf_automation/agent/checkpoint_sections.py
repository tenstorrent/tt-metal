# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How many block stacks a model has and how deep each is -- read off the weights, not off a config.

THE SECTION COUNT HAD ONE SOURCE AND IT WAS AN HF CONFIG. `_declared_sections` parses
num_hidden_layers out of a transformers config, which is an independent witness worth having and is
unavailable to every model that did not come from HF. A model trained in-house, exported from a
research repo, or shipped as a bare checkpoint has no config.json to read, so the only remaining
statement of "this model has three sections" was the walk itself -- and the walk is what is being
checked.

A CHECKPOINT DECLARES ITS OWN STRUCTURE. Weight keys are paths, and a repeated block prints its
index into every one of them:

    audio_tower.layers.0.self_attn.q_proj.weight     ->  section "audio_tower.layers", index 0
    audio_tower.layers.31.mlp.fc2.bias               ->  section "audio_tower.layers", index 31
    language_model.layers.29.mlp.down_proj.weight    ->  section "language_model.layers", index 29

Grouping keys by "the part before the integer" gives one entry per stack and the depth of each --
count and boundaries together, from a file, needing no config, no transformers, no torch and no
device. It works on any checkpoint that stores named tensors, which is all of them.

KEYS ONLY, NEVER VALUES. A safetensors file states its whole key set in a JSON header at byte 8, and
a torch .pt is a zip whose member names carry the keys, so both are read without materialising a
single weight. That matters: this runs during discovery, where loading a 3B model to count its
layers would cost more than the check saves.

WHAT IT IS FOR is disagreement. The walk says how many stacks it can see; this says how many the
weights describe. Neither is trusted over the other -- a mismatch means structure is hidden, which is
the same conclusion the HF reference gives for models that have one.
"""

from __future__ import annotations

import json
import os
import re
import struct
import zipfile
from pathlib import Path

_INDEXED = re.compile(r"^(.*?)\.(\d+)\.")
_WEIGHT_SUFFIX = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_MAX_FILES = 24


def _safetensors_keys(path: Path) -> list:
    """Keys from the JSON header, without reading a single tensor.

    Layout: 8 bytes of little-endian header length, then that many bytes of JSON.
    """
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return []
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 100_000_000:
            return []
        head = json.loads(fh.read(n).decode("utf-8"))
    return [k for k in head if k != "__metadata__"]


def _torch_zip_keys(path: Path) -> list:
    """Keys from a torch save file's zip directory. A .pt is a zip whose members are named for the
    tensors, so the archive listing carries the key set with nothing unpickled -- which also means no
    arbitrary code from the checkpoint is executed to answer a structural question."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    out = []
    for n in names:
        parts = n.split("/")
        if len(parts) > 2 and parts[1] == "data":
            continue  # storage blobs, not key paths
        out.append("/".join(parts[1:]) if len(parts) > 1 else n)
    return out


def hf_cache_dir(model_id: str):
    """Where the weights for `model_id` actually live.

    A TT-METAL DEMO SHIPS CODE, NOT WEIGHTS. Its directory holds the pipeline, the stubs and the
    tests; `from_pretrained(HF_REPO_ID)` pulls ~9 GB into a shared cache once and every demo reads it
    from there. Globbing the model root for checkpoints therefore finds NOTHING for every model in
    this repo -- which is exactly what it did, silently, returning "no sections declared".

    This resolves the cache path by name only. No transformers import, no download, no network: the
    layout is hub/models--<org>--<name>/snapshots/<sha>/.
    """
    if not model_id or "/" not in str(model_id):
        return None
    home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    roots = [Path(home) / "hub" if home else None, Path.home() / ".cache" / "huggingface" / "hub"]
    # MODELS AND DATASETS SHARE THE CACHE under different prefixes. A pipeline commonly loads BOTH --
    # weights from a model repo and sample inputs from a dataset repo (Voxtral pulls
    # hf-internal-testing/dummy-audio-samples for its test audio) -- and looking only under models--
    # reports a present dataset as missing, which turns a readiness gate into a false alarm.
    slugs = [pre + str(model_id).replace("/", "--") for pre in ("models--", "datasets--")]
    for base in [r for r in roots if r]:
        snaps = next((base / sl / "snapshots" for sl in slugs if (base / sl / "snapshots").is_dir()), None)
        if snaps is None:
            continue
        # Newest snapshot: a cache may hold several revisions of one repo.
        for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if snap.is_dir():
                return snap
    return None


def _index_keys(snapshot: Path) -> list:
    """Every tensor name from a shard index, without opening a single shard.

    A sharded checkpoint ships `*.index.json` mapping tensor name -> shard file. Reading that one
    small JSON answers the whole structural question; opening the shards is unnecessary, and reading
    only SOME of them reports a model with fewer sections than it has.
    """
    out = []
    for idx in sorted(snapshot.glob("*.index.json")):
        try:
            data = json.loads(idx.read_text())
        except (OSError, ValueError):
            continue
        wm = data.get("weight_map")
        if isinstance(wm, dict):
            out.extend(wm.keys())
    return out


def checkpoint_keys(root) -> list:
    """Every tensor key from the checkpoint files under `root`. [] when there are none to read."""
    root = Path(root)
    if not root.exists():
        return []
    files = []
    if root.is_file():
        files = [root]
    else:
        for suf in _WEIGHT_SUFFIX:
            files.extend(sorted(root.rglob("*" + suf))[:_MAX_FILES])
    keys = []
    for f in files[:_MAX_FILES]:
        try:
            if f.suffix == ".safetensors":
                keys.extend(_safetensors_keys(f))
            else:
                keys.extend(_torch_zip_keys(f))
        except Exception:  # noqa: BLE001 -- an unreadable checkpoint is simply not a witness
            continue
    return keys


def sections_from_keys(keys) -> dict:
    """{section prefix: depth} for every indexed run in the key set.

    A section is real when its indices start at 0 and there are at least two of them: `layers.0`
    through `layers.31` is a stack, while a lone `adapter.0` is a naming choice. Depth is
    max index + 1 rather than the count, so a sharded checkpoint holding only some of the layers
    still reports the model's true depth.
    """
    idx = {}
    for k in keys or []:
        m = _INDEXED.match(str(k))
        if m:
            idx.setdefault(m.group(1), set()).add(int(m.group(2)))
    return {p: max(v) + 1 for p, v in idx.items() if len(v) >= 2 and 0 in v}


def declared_sections(root, model_id: str = "") -> dict:
    """{section: depth} straight from a model's weights. {} when nothing readable is present.

    Looks in the model directory first (a self-contained checkpoint), then in the shared HF cache for
    `model_id` -- which is where every tt-metal demo's weights actually are.
    """
    got = sections_from_keys(checkpoint_keys(root))
    if got:
        return got
    snap = hf_cache_dir(model_id)
    if snap is None:
        return {}
    keys = _index_keys(snap)
    if keys:
        return sections_from_keys(keys)
    return sections_from_keys(checkpoint_keys(snap))


def section_count(root, model_id: str = "") -> int:
    """How many block stacks the weights describe. 0 means no evidence, never "none exist"."""
    return len(declared_sections(root, model_id))
