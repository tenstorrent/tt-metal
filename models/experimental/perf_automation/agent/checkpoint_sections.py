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


def _safetensors_section_bytes(path: Path) -> dict:
    """{top-level prefix: bytes} from one shard's JSON header, reading no tensor data.

    The header already states every tensor's byte span in `data_offsets`, so the size of a tower is
    a sum over names -- no device, no load, no dtype table to keep in step with the loader.

    WHY A TOWER AND NOT THE WHOLE FILE. A decode token reads the language backbone and none of the
    audio encoder, but the checkpoint's byte total covers both, so pricing any single stage from it
    is wrong for a multi-tower model -- weight_census says exactly this and declines to guess. The
    prefix IS the tower: `audio_tower.layers.31.mlp.fc2.bias` and `language_model.layers.29...` are
    the names the model itself ships, and the stack survey already reports stacks by that same path.
    """
    out: dict = {}
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return out
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 100_000_000:
                return out
            head = json.loads(fh.read(n).decode("utf-8"))
    except Exception:  # noqa: BLE001
        return out
    for k, v in head.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        off = v.get("data_offsets") or []
        if len(off) != 2:
            continue
        try:
            nbytes = int(off[1]) - int(off[0])
        except (TypeError, ValueError):
            continue
        if nbytes > 0:
            out[str(k).split(".", 1)[0]] = out.get(str(k).split(".", 1)[0], 0) + nbytes
    return out


def section_bytes(snapshot) -> dict:
    """{top-level prefix: bytes} across every shard of a checkpoint snapshot."""
    total: dict = {}
    snap = Path(snapshot)
    if not snap.is_dir():
        return total
    for f in sorted(snap.glob("*.safetensors")):
        for k, v in _safetensors_section_bytes(f).items():
            total[k] = total.get(k, 0) + v
    return total


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


def tower_geometry(snapshot) -> dict:
    """{depth: {layers, hidden_size, intermediate_size}} for every tower the config declares.

    A multi-tower checkpoint states each tower's shape in its own sub-config -- audio_config,
    vision_config, text_config -- and the ceiling needs them because a tower's activations are its
    OWN width, not the language model's. Voxtral: audio 32x1280x5120 beside text 30x3072x8192, so
    pricing the audio encoder with the text numbers is wrong by more than a factor of two.

    KEYED BY DEPTH, not by name, so it joins to the rest of the evidence chain: the probe reports a
    stack's block count, declared_sections reports a section's, and this reports a config's. Three
    vocabularies, one number they all agree on. A tower nobody has named before -- vocoder, denoiser,
    a second vision stack -- lands here with no code change, which is the whole point.

    Any sub-dict carrying a layer count and a hidden size is a tower; nothing is recognised by name.
    """
    out: dict = {}
    # A DICT IS ALSO A CONFIG. The single-tower path has no checkpoint to point at -- its geometry
    # arrives as the manifest's model_config or _hf_cache_dims' answer -- and requiring a path there
    # meant a flat model produced no blocks at all, which cost it the geometry it used to publish.
    if isinstance(snapshot, dict):
        doc = snapshot
    else:
        snap = Path(snapshot)
        cfg = snap / "config.json" if snap.is_dir() else snap
        try:
            doc = json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            return out
    if not isinstance(doc, dict):
        return out
    for sub in list(doc.values()) + [doc]:
        if not isinstance(sub, dict):
            continue
        try:
            n = int(sub.get("num_hidden_layers") or sub.get("layers") or 0)
            h = int(sub.get("hidden_size") or sub.get("d_model") or 0)
        except (TypeError, ValueError):
            continue
        if n <= 0 or h <= 0 or n in out:
            continue
        try:
            i = int(sub.get("intermediate_size") or sub.get("ffn_dim") or 0)
        except (TypeError, ValueError):
            i = 0
        # ATTENTION GEOMETRY COMES FROM THE SAME SUB-DICT, or not at all. The KV term needs kv_heads
        # and head_dim, and reading them from a different tower than hidden_size is the mistake this
        # function exists to end -- so they are taken from THIS tower's config, and left absent when
        # it does not declare them rather than borrowed from the model root.
        try:
            _kvh = int(sub.get("num_key_value_heads") or sub.get("num_attention_heads") or sub.get("num_heads") or 0)
        except (TypeError, ValueError):
            _kvh = 0
        try:
            _hd = int(sub.get("head_dim") or 0)
        except (TypeError, ValueError):
            _hd = 0
        try:
            _heads = int(sub.get("num_attention_heads") or sub.get("num_heads") or 0)
        except (TypeError, ValueError):
            _heads = 0
        if not _hd and h and _heads:
            _hd = h // _heads
        geo = {"layers": n, "hidden_size": h, "intermediate_size": i or 4 * h}
        if _kvh:
            geo["kv_heads"] = _kvh
        if _hd:
            geo["head_dim"] = _hd
        out[n] = geo
    return out


def layer_kinds_from_keys(keys) -> tuple:
    """(k, n_kinds): how many DISTINCT kinds of block the checkpoint contains, and how many layers
    from the front you must take to see one of each. (None, 0) when it declares no repeated blocks.

    COUNTED, NOT DECLARED. The config route reads a per-layer pattern out of one of four attribute
    names -- hybrid_override_pattern, layer_types, layers_block_type, block_types -- and a model
    using a fifth spelling, or one transformers cannot load at all, yields nothing. Voxtral is the
    second case: AutoConfig raises on its model type, so that path returns (None, 0) and the caller
    gives up with "no_window: probe_failed".

    A checkpoint says it outright. Two blocks are the same KIND when they hold the same set of
    parameter names, and different kinds when they do not -- an attention block has q_proj/k_proj,
    a Mamba block has in_proj/conv1d, and no vocabulary is needed to see that they differ. Indices
    come from the names too, so "the first k layers cover every kind" is a count.

    Per stack, then the deepest answer across stacks: a window has to be representative of every
    tower, not of the first one enumerated.
    """
    import re as _re

    stacks: dict = {}
    for name in keys or []:
        m = _re.match(r"^(.*\.layers)\.(\d+)\.(.+)$", str(name))
        if not m:
            continue
        stacks.setdefault(m.group(1), {}).setdefault(int(m.group(2)), set()).add(m.group(3))
    best_k, best_n = None, 0
    for _prefix, per_idx in stacks.items():
        if len(per_idx) < 2:
            continue
        sigs = {i: frozenset(v) for i, v in per_idx.items()}
        kinds = set(sigs.values())
        seen: set = set()
        k = None
        for i in sorted(sigs):
            seen.add(sigs[i])
            if seen == kinds:
                k = i + 1
                break
        if k is None:
            continue
        if best_k is None or k > best_k:
            best_k = k
        best_n = max(best_n, len(kinds))
    return best_k, best_n


def layer_kinds(root, model_id: str = "") -> tuple:
    """layer_kinds_from_keys against this model's checkpoint. (None, 0) when unreadable."""
    # THE RESULT DECIDES, NOT THE KEY COUNT. A model directory can hold keys that are not the model
    # -- voxtral's ships 24 from its captured references and stubs -- so "the list is non-empty" is
    # not "the weights are here", and testing it skipped the cache where the weights actually are.
    # declared_sections above resolves the same way, on whether the ANSWER came out.
    try:
        got = layer_kinds_from_keys(checkpoint_keys(root))
        if got[0]:
            return got
        snap = hf_cache_dir(model_id) if model_id else None
        if snap is None:
            return None, 0
        return layer_kinds_from_keys(_index_keys(snap) or checkpoint_keys(snap))
    except Exception:  # noqa: BLE001 -- an unreadable checkpoint declares nothing
        return None, 0


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
