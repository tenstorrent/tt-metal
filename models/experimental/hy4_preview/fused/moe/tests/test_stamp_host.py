# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the stamp of the expert container (fused/moe/prepare.py container_stamp, "Container format" in its
docstring) and the two places that refuse files of another container: the loaders (prepare.check_stamp, called by
prepare.load_streams and fused/layer_mlp/program.py load_streams) and the conversion's completeness check
(fused/model/convert.py state).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

test_stamp_fields            what the stamp says of tp32, tp8g4 and the dense FFN: the entries a chip holds, which of
                             them are routed, the shared expert's entries, terms and group, the share, the bytes of an
                             entry in a bank
test_loader_refuses          a directory stamped for one shared group is refused for another (equal file sizes: the
                             shared expert would be read from zero entries), and so are another split, another
                             format number, another source and files without a stamp
test_convert_state           a layer with complete files is "unstamped" without a stamp, "complete" with this
                             tree's, "broken: stamp" with another container's; a stamp from the conversion's record
                             (convert.stamp_from_manifest) is refused when the record says another shared group
test_files_on_disk           the converted layers under $HY4_MOE_STREAMS (skipped where a split is not converted): each
                             is stamped and its stamp is what a loader of that split asks for
"""

import json
import os

import pytest

from models.experimental.hy4_preview.fused.model import convert
from models.experimental.hy4_preview.fused.moe import prepare

STREAMS = prepare.DEFAULT_OUT


def test_stamp_fields():
    tp32, tp8, dense = (
        prepare.container_stamp("tp32"),
        prepare.container_stamp("tp8g4"),
        prepare.container_stamp("tp32", dense=True),
    )
    for stamp in (tp32, tp8, dense):
        assert stamp["format"] == prepare.FORMAT and stamp["dtype"] == "bfloat4_b" and stamp["source"] == "bf16"
        assert stamp["chips"] == 32
        json.dumps(stamp)  # plain values: it is written into manifest.json
    # tp32: every chip 1/32 of every routed expert (ids 0 .. 255) and of the shared expert's two terms
    assert (tp32["P"], tp32["S"], tp32["entries"], tp32["routed_entries"]) == (2, 4, 258, 256)
    assert tp32["shared_entries"] == [256, 257] and tp32["shared_terms"] == 2 and tp32["shared_group"] is None
    # tp8g4: a chip 1/8 of its group's 64 routed experts; the shared expert's two terms behind them, on one group
    assert (tp8["P"], tp8["S"], tp8["entries"], tp8["routed_entries"]) == (8, 1, 66, 64)
    assert tp8["shared_entries"] == [64, 65] and tp8["shared_terms"] == 2
    assert tp8["shared_group"] == prepare.GROUP_SHARED
    assert prepare.container_stamp("tp8g4", shared_group=1)["shared_group"] == 1
    # the dense FFN: 9 slices of two terms, no routed entry, no shared expert
    assert (dense["kind"], dense["entries"], dense["routed_entries"], dense["shared_entries"]) == ("dense", 18, 0, [])
    assert (dense["dense_slices"], dense["dense_terms"], dense["shared_terms"]) == (9, 2, 0)
    # an entry's bytes in a bank are the share's strides: 4 x as many for 4 x the channels
    assert tp32["entry_bytes_per_bank"] == {"gate_up": 55296, "down": 27648}
    assert tp8["entry_bytes_per_bank"] == {"gate_up": 221184, "down": 110592}


def _stamped(root, layer, split, stamp):
    path = prepare.manifest_path(root, layer, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layer": layer, "split": split, **({"stamp": stamp} if stamp else {})}))


def test_loader_refuses(tmp_path, expect_error):
    _stamped(tmp_path, 5, "tp8g4", prepare.container_stamp("tp8g4"))
    assert prepare.check_stamp(tmp_path, 5, "tp8g4")["shared_group"] == prepare.GROUP_SHARED
    # the shared group changed and the files were kept: same sizes, the shared expert on other chips
    with expect_error(AssertionError, "holds another expert container"):
        prepare.check_stamp(tmp_path, 5, "tp8g4", shared_group=1)
    # files of one split read as another, as a dense layer, from another source
    _stamped(tmp_path, 5, "tp32", prepare.container_stamp("tp8g4"))
    with expect_error(AssertionError, "holds another expert container"):
        prepare.check_stamp(tmp_path, 5, "tp32")
    _stamped(tmp_path, 5, "tp32", prepare.container_stamp("tp32"))
    prepare.check_stamp(tmp_path, 5, "tp32")
    with expect_error(AssertionError, "holds another expert container"):
        prepare.check_stamp(tmp_path, 5, "tp32", dense=True)
    with expect_error(AssertionError, "holds another expert container"):
        prepare.check_stamp(tmp_path, 5, "tp32", source="mxfp8")
    # another format number
    _stamped(tmp_path, 6, "tp8g4", {**prepare.container_stamp("tp8g4"), "format": prepare.FORMAT + 1})
    with expect_error(AssertionError, "holds another expert container"):
        prepare.check_stamp(tmp_path, 6, "tp8g4")
    # files from before the stamp, and no files at all
    _stamped(tmp_path, 7, "tp8g4", None)
    with expect_error(AssertionError, "has no container stamp"):
        prepare.check_stamp(tmp_path, 7, "tp8g4")
    with expect_error(AssertionError, "has no container stamp"):
        prepare.check_stamp(tmp_path, 8, "tp8g4")


def _layer_on_disk(root, layer, split, stamp=None, shared_group=prepare.GROUP_SHARED):
    """A converted layer as the conversion leaves it, with one-byte files: manifest.json and files.json."""
    d = convert.layer_dir(root, layer, split)
    d.mkdir(parents=True, exist_ok=True)
    for name in convert.expected_files():
        (d / name).write_bytes(b"\0")
    want = prepare.container_stamp(split, dense=layer == 0)
    share = prepare.split_share(split)
    info = {
        "layer": layer,
        "split": split,
        "entries": want["entries"],
        "chips": 32,
        "bytes_per_chip": want["entries"] * share.expert_bytes + 768,
    }
    if prepare.split_groups(split):
        info["shared_group"] = shared_group
    files = {"layer": layer, "split": split, "complete": True, "files": {n: [1, ""] for n in convert.expected_files()}}
    if stamp is not None:
        info["stamp"] = files["stamp"] = stamp
    prepare.manifest_path(root, layer, split).write_text(json.dumps(info))
    (d / "files.json").write_text(json.dumps(files))


def test_convert_state(tmp_path, expect_error):
    assert convert.state(tmp_path, 5, split="tp8g4")[0] == "missing"
    _layer_on_disk(tmp_path, 5, "tp8g4")
    assert convert.state(tmp_path, 5, split="tp8g4")[0] == "unstamped"
    assert convert.stamp_from_manifest(tmp_path, 5, "tp8g4") == prepare.container_stamp("tp8g4")
    _layer_on_disk(tmp_path, 5, "tp8g4", stamp=prepare.container_stamp("tp8g4"))
    assert convert.state(tmp_path, 5, split="tp8g4")[0] == "complete"
    # files of another shared group: the sizes are the same, the layer is not complete
    _layer_on_disk(tmp_path, 5, "tp8g4", stamp=prepare.container_stamp("tp8g4", shared_group=1))
    assert convert.state(tmp_path, 5, split="tp8g4")[0].startswith("broken: stamp")
    # ... and a record of another shared group does not get this tree's stamp
    _layer_on_disk(tmp_path, 6, "tp8g4", shared_group=1)
    with expect_error(AssertionError, "records another container"):
        convert.stamp_from_manifest(tmp_path, 6, "tp8g4")
    # layer 0 is the dense FFN: its stamp is the dense one
    _layer_on_disk(tmp_path, 0, "tp32", stamp=prepare.container_stamp("tp32"))
    assert convert.state(tmp_path, 0, split="tp32")[0].startswith("broken: stamp")
    _layer_on_disk(tmp_path, 0, "tp32", stamp=prepare.container_stamp("tp32", dense=True))
    assert convert.state(tmp_path, 0, split="tp32")[0] == "complete"


@pytest.mark.parametrize("split", ["tp32", "tp8g4"])
def test_files_on_disk(split):
    layers = [n for n in range(78) if convert.layer_dir(STREAMS, n, split).is_dir() and (split == "tp32" or n > 0)]
    if not layers:
        pytest.skip(f"no layer of split {split} under {STREAMS}")
    states = {n: convert.state(STREAMS, n, split=split)[0] for n in layers}
    assert all(v == "complete" for v in states.values()), {n: v for n, v in states.items() if v != "complete"}
    for n in layers:
        prepare.check_stamp(STREAMS, n, split, dense=n == 0)
    with_shared = [n for n in layers if n > 0]
    if split == "tp8g4" and with_shared:
        other = (prepare.GROUP_SHARED + 1) % 4
        assert prepare.stamp_difference(
            prepare.read_stamp(STREAMS, with_shared[0], split), prepare.container_stamp(split, shared_group=other)
        ) == {"shared_group": (prepare.GROUP_SHARED, other)}
    assert os.path.isdir(STREAMS)
