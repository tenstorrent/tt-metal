# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint that ships its weights twice is read once, from the copy the loader reads.

Devstral-Small-2-24B-Instruct-2512 (2026-10-19) carries the six `model-*` shards transformers loads
beside two `consolidated-*` shards in Mistral's own layout, each set under its own index. Every
checkpoint reader walked both: 48.0B parameters for a 24.0B model, so prefill's compute floor sat 2x
too slow and the stage read "93% of peak" at 47%; two sections of every depth, so no stage could be
placed on a tower, no block got geometry, and the vision encoder was priced with the language model's
parameters. One rule, in model_bytes: the loader's copy is the model, the rest is the same weights
again -- neither read nor reported unread. A checkpoint with one copy reads exactly as before.
Tensor names, file names and depths here are this file's own.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import model_bytes as mb  # noqa: E402
from agent import weight_census as wc  # noqa: E402
from agent.checkpoint_sections import _index_keys, checkpoint_keys, sections_from_keys  # noqa: E402


def _st(path: Path, tensors: dict) -> None:
    hdr, off = {}, 0
    for k, s in tensors.items():
        n = 2
        for dim in s:
            n *= dim
        hdr[k] = {"dtype": "BF16", "shape": list(s), "data_offsets": [off, off + n]}
        off += n
    hdr["__metadata__"] = {"format": "pt"}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _index(path: Path, files: dict) -> None:
    path.write_text(json.dumps({"metadata": {}, "weight_map": {k: f for f, ks in files.items() for k in ks}}))


_LOADER = {
    "model-00001-of-00002.safetensors": {"lm.model.layers.0.w": (64, 32), "lm.model.layers.1.w": (64, 32)},
    "model-00002-of-00002.safetensors": {
        "lm.model.layers.2.w": (64, 32),
        "lm.model.layers.3.w": (64, 32),
        "vt.transformer.layers.0.w": (16, 16),
        "vt.transformer.layers.1.w": (16, 16),
        "lm.lm_head.weight": (10, 32),
    },
}
# The same weights again, in another layout and under other names: one shard, its own index.
_OTHER = {
    "consolidated-00001-of-00001.safetensors": {
        **{"layers.%d.w" % i: (64, 32) for i in range(4)},
        **{"vision_encoder.transformer.layers.%d.w" % i: (16, 16) for i in range(2)},
        "output.weight": (10, 32),
    }
}
_ONE_COPY_PARAMS = 4 * 64 * 32 + 2 * 16 * 16 + 10 * 32


def _twice(tmp_path, with_loader_index=True):
    d = tmp_path / "snap"
    for f, ts in _LOADER.items():
        _st(d / f, ts)
    if with_loader_index:
        _index(d / "model.safetensors.index.json", {f: list(ts) for f, ts in _LOADER.items()})
    for f, ts in _OTHER.items():
        _st(d / f, ts)
    _index(d / "consolidated.safetensors.index.json", {f: list(ts) for f, ts in _OTHER.items()})
    return d


def test_the_second_copy_is_named_and_the_loaders_copy_is_read(tmp_path):
    d = _twice(tmp_path)
    assert [p.name for p in mb.duplicate_weight_files(d)] == ["consolidated-00001-of-00001.safetensors"]
    assert [f.name for _, f in mb.weight_files(d)] == sorted(_LOADER)
    assert mb.unread_weight_files(d) == [], "a second copy is skipped by rule, not left unread"


def test_the_parameter_count_is_one_copy(tmp_path):
    d = _twice(tmp_path)
    assert mb.weight_bytes(d, unit="step")["params"] == _ONE_COPY_PARAMS
    assert sum(n for n, _s, _n in wc._checkpoint_tensor_sections(str(d))) == _ONE_COPY_PARAMS


def test_every_tower_is_declared_once(tmp_path):
    """Two sections of one depth place no stage; the second copy made every depth a tie."""
    d = _twice(tmp_path)
    assert sections_from_keys(_index_keys(d)) == {"lm.model.layers": 4, "vt.transformer.layers": 2}
    assert sections_from_keys(checkpoint_keys(d)) == {"lm.model.layers": 4, "vt.transformer.layers": 2}


def test_a_single_copy_reads_whole_whatever_it_is_called(tmp_path):
    """Without the loader's copy nothing is a duplicate: the other layout IS the checkpoint."""
    d = tmp_path / "only_other"
    for f, ts in _OTHER.items():
        _st(d / f, ts)
    _index(d / "consolidated.safetensors.index.json", {f: list(ts) for f, ts in _OTHER.items()})
    assert mb.duplicate_weight_files(d) == []
    assert mb.weight_bytes(d, unit="step")["params"] == _ONE_COPY_PARAMS
    assert sections_from_keys(_index_keys(d)) == {"layers": 4, "vision_encoder.transformer.layers": 2}


def test_the_loaders_copy_without_an_index_is_still_the_copy(tmp_path):
    """One unsharded `model.safetensors` beside a consolidated file: the first is the model."""
    d = tmp_path / "single"
    _st(d / "model.safetensors", {k: s for ts in _LOADER.values() for k, s in ts.items()})
    for f, ts in _OTHER.items():
        _st(d / f, ts)
    assert [p.name for p in mb.duplicate_weight_files(d)] == ["consolidated-00001-of-00001.safetensors"]
    assert mb.weight_bytes(d, unit="step")["params"] == _ONE_COPY_PARAMS
    assert mb.unread_weight_files(d) == []


def test_a_flat_checkpoint_with_one_copy_is_untouched(tmp_path):
    d = tmp_path / "flat"
    for f, ts in _LOADER.items():
        _st(d / f, ts)
    assert mb.duplicate_weight_files(d) == []
    assert [f.name for _, f in mb.weight_files(d)] == sorted(_LOADER)
    assert mb.weight_bytes(d, unit="step")["params"] == _ONE_COPY_PARAMS


def test_the_facts_say_when_a_copy_was_skipped():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _perf_target_inputs(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_mb.duplicate_weight_files(_snap)" in body and "ships its weights twice" in body
