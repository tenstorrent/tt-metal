# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert weight streams of the streamed decode MoE (fused/moe/layout.py) from the checkpoint, on disk.

    python -m models.experimental.hy4_preview.fused.moe.prepare --layers 1 2 3 4 5 --split tp32 --out <dir>

writes, per layer and chip, ``<out>/layer<L>/<split>/chip<cc>.{gate_up,down}.tensorbin`` (host BFP4 tensors,
[1, 1, rows, 256]: one [rows, 32] shard per DRAM bank) and ``manifest.json`` with the time and the bytes.
Splits: tp<N> (chip c holds intermediate channels [c, c + 1) * 2048 / N of every expert), ep (chip c holds the
8 whole experts [8 c, 8 c + 8)) and tp<N>g<G> ("N chips per group": the N * G chips are G groups of N; group g owns
the routed experts [256 g / G, 256 (g + 1) / G) and nobody else holds them; a chip holds 1 / N of each expert of
its group: chip c is slice c // G of group c % G, i.e. on a mesh of N rows x G columns, chips row-major, a group is
a mesh column and the slice is the mesh row. tp8g4 is the 8x4 Galaxy's. The shared expert, 64 pairs like any
expert, lives on ONE group at the same 1 / N (GROUP_SHARED), as the two entries behind that group's experts; the
other groups' files carry two zero entries there, never read, so that all chips' streams have one shape (a mesh
tensor has one shape for all its shards)).

Host only. Run it with TT_METAL_MOCK_CLUSTER_DESC_PATH set (ttnn's host tensor conversion creates the device
context, which opens the board otherwise).

Container format 1 (``FORMAT``; what a reader other than fused/moe has to know). A layer of a split is the
directory ``<out>/layer<L>/<split>/``: per chip (row-major on the mesh) two host tensors, ``chip<cc>.gate_up`` and
``chip<cc>.down``, and ``manifest.json``, whose "stamp" (``container_stamp``) names everything below; a loader
compares it with what it is about to read (``check_stamp``) and refuses another container.
  values    BFP4 tiles (576 B) of the BF16 checkpoint's weights (``SOURCE``), rounded once, on the host
            (layout.bfp4_round); the tiles are tiles of W^T, the orientation ttnn.linear holds a weight in
            (gate | up: [hidden k, channel n]; down: [channel k, hidden n])
  shape     [1, 1, rows, 256]: 8 shards [rows, 32], one per DRAM bank; the same bytes of every entry in every bank
  entry     one expert's share on the chip, at byte offset ``entry * stride`` in each bank's shard, both files:
            P pairs of gate / up tile columns (32 intermediate channels each) and the same channels of down
  order     gate_up: bank > entry > item t > K tile > (gate tile, up tile), piece g = pair * S + q (K range q of S)
            in bank g % 8 as item g // 8;  down: bank > entry > output tile column (24 a bank) > K tile (layout.py)
  entries   MoE layer, tp<N> (every chip the channels [c, c + 1) * 2048 / N of every entry): 0 .. 255 the routed
            experts by their id; 256, 257 the shared expert.
            MoE layer, tp<N>g<G> (chip c: group c % G, slice c // G: the channels [s, s + 1) * 2048 / N):
            0 .. 256 / G - 1 the group's routed experts, local id e = routed expert g * 256 / G + e; the next two
            the shared expert on the chips of group ``shared_group`` and ZEROS on every other chip.
            Layer 0 (the dense FFN, tp32 only): 18 entries, slice j of 9 slices of 2048 channels as 2 j, 2 j + 1.
  shared    the shared expert (and a dense slice) is TWO entries, two BFP4 terms: q1 = BFP4(w) and q2 =
            BFP4(w - q1), the value the device computes with is q1 + q2 (layout.two_terms). They are not routed
            experts: fused/moe runs them as always-on plain slots (no clamp, weight 1). A reader that keeps its
            own shared expert reads the routed entries only and skips the entries from ``routed_entries`` on.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from . import layout as L

DEFAULT_OUT = os.environ.get("HY4_MOE_STREAMS", "generated/hy4_preview/moe_streams")


def checkpoint_tensors(names):
    """Tensors by name from the BF16 checkpoint in $HY4_HF_MODEL, as stored (BF16)."""
    import safetensors

    root = Path(os.environ["HY4_HF_MODEL"])
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for name in names:
        with safetensors.safe_open(str(root / weight_map[name]), framework="pt") as f:
            out.append(f.get_tensor(name))
    return out


def _terms(gate_up, down, terms):
    """Entries as `terms` BFP4 terms each (layout.two_terms), fp32."""
    if terms == 1:
        return gate_up, down
    return L.two_terms(gate_up.float()), L.two_terms(down.float())


def layer_experts(layer, shared=False, shared_terms=2):
    """(gate_up_proj [256, 4096, 6144], down_proj [256, 6144, 2048]) of a layer, BF16. shared: the shared expert
    appended from entry 256 on (the streamed program runs it as one more, always-on, plain union slot), as
    shared_terms BFP4 terms (2: entries 256 and 257, the accuracy of BFP8)."""
    prefix = f"model.layers.{layer}.mlp."
    gate_up, down = checkpoint_tensors([prefix + "experts.gate_up_proj", prefix + "experts.down_proj"])
    if shared:
        g, u, d = checkpoint_tensors([prefix + f"shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")])
        sg, sd = _terms(torch.cat((g, u), dim=0)[None], d[None], shared_terms)
        gate_up = torch.cat((gate_up, sg.to(gate_up.dtype)), dim=0)
        down = torch.cat((down, sd.to(down.dtype)), dim=0)
    return gate_up, down


def dense_experts(layer=0, terms=2):
    """The dense FFN of layer 0 as 9 slices of 2048 intermediate channels in the experts' form, each as `terms`
    BFP4 terms: (gate_up [9 terms, 4096, 6144], down [9 terms, 6144, 2048])."""
    prefix = f"model.layers.{layer}.mlp."
    g, u, d = checkpoint_tensors([prefix + f"{n}_proj.weight" for n in ("gate", "up", "down")])
    I = L.INTERMEDIATE
    n = g.shape[0] // I
    gate_up = torch.cat((g.reshape(n, I, -1), u.reshape(n, I, -1)), dim=1)
    down = d.reshape(d.shape[0], n, I).permute(1, 0, 2).contiguous()
    return _terms(gate_up, down, terms)


GROUP_SHARED = 3  # tp<N>g<G>: the group whose chips hold the shared expert (not the root chip's column 0)
FORMAT = 1  # the container's format number: any change of the files' bytes, or of what an entry is, is a new number
SOURCE = "bf16"  # what the values are converted from: the BF16 checkpoint in $HY4_HF_MODEL (the only source here)
SHARED_TERMS = 2  # BFP4 terms of the shared expert and of a dense slice (layout.two_terms)
ROUTED = 256


def split_groups(split):
    """tp<N>g<G> -> (N, G); None for the other splits."""
    if split.startswith("tp") and "g" in split:
        n, g = split[2:].split("g")
        return int(n), int(g)
    return None


def chip_group(split, chip):
    """tp<N>g<G>: (group, slice) of chip ``chip`` (row-major on a mesh of N rows x G columns: column, row)."""
    _, groups = split_groups(split)
    return chip % groups, chip // groups


def group_experts(split, group, routed=256):
    """tp<N>g<G>: (expert_base, expert_count) of a group among ``routed`` routed experts."""
    _, groups = split_groups(split)
    assert routed % groups == 0
    return group * (routed // groups), routed // groups


def chip_union(split, chip, indices, weights, routed=256, shared_group=GROUP_SHARED):
    """tp<N>g<G>: what chip ``chip`` runs for the rows' picks (indices, weights [T, K]): (local ids of its stream
    entries, weights [T, U], n_plain): its group's part of the union (layout.union_of, the hub's mirror), and on
    the shared expert's group one plain slot more (entry ``expert_count``, weight 1). An empty list is a chip with
    nothing to do: its partial is zero."""
    group, _ = chip_group(split, chip)
    base, count = group_experts(split, group, routed)
    ids, w = L.union_of(indices, weights, expert_base=base, expert_count=count)
    if group != shared_group:
        return ids, w, 0
    return ids + [count], torch.cat((w, torch.ones(w.shape[0], 1)), dim=1), 1


def split_share(split):
    if split == "ep":
        return L.ep_share()
    groups = split_groups(split)
    return L.tp_share(groups[0] if groups else int(split[2:]))


def split_chips(split):
    if split == "ep":
        return 32
    groups = split_groups(split)
    return groups[0] * groups[1] if groups else int(split[2:])


def chip_weights(gate_up, down, split, chip, experts=None, routed=256, shared_group=GROUP_SHARED):
    """The chip's share (gate, up [E, I_c, 6144], down [E, 6144, I_c]) as fp32. experts: only these (global ids,
    for a tensor-parallel split).
    tp<N>g<G>: the chip's slice of its group's ``routed`` / G experts (local ids 0 ..), then the entries behind the
    routed ones (the shared expert's terms): the chip's slice of them on group ``shared_group``, zeros elsewhere."""
    I = L.INTERMEDIATE
    if split == "ep":
        sel = slice(8 * chip, 8 * chip + 8)
        return gate_up[sel, :I].float(), gate_up[sel, I:].float(), down[sel].float()
    if split_groups(split):
        assert experts is None
        n, _ = split_groups(split)
        I = gate_up.shape[1] // 2
        group, part = chip_group(split, chip)
        base, count = group_experts(split, group, routed)
        c0, c1 = part * I // n, (part + 1) * I // n
        # (the slice first: the copies are of the chip's share only)
        own = lambda w, cut: torch.cat((cut(w[base : base + count]), cut(w[routed:])), dim=0).float()
        gate = own(gate_up, lambda w: w[:, c0:c1])
        up = own(gate_up, lambda w: w[:, I + c0 : I + c1])
        dn = own(down, lambda w: w[:, :, c0:c1])
        if group != shared_group:
            for w in (gate, up, dn):
                w[count:] = 0.0
        return gate, up, dn
    # (a tensor-parallel chip's channels: the same slice of every entry, the shared expert and the dense FFN's
    # entries included)
    n = int(split[2:])
    c0, c1 = chip * I // n, (chip + 1) * I // n
    e = slice(None) if experts is None else experts
    return gate_up[e, c0:c1].float(), gate_up[e, I + c0 : I + c1].float(), down[e][:, :, c0:c1].float()


def container_stamp(split, dense=False, shared_group=GROUP_SHARED, source=SOURCE) -> dict:
    """What a layer's files of ``split`` hold (the container format in this module's docstring), as plain values:
    prepare_layer writes it into manifest.json, check_stamp compares it before a layer's streams are loaded.
    dense: the layer is the dense FFN. shared_group: the group that holds the shared expert (a grouped split)."""
    share, groups = split_share(split), split_groups(split)
    assert not (dense and (groups or split == "ep")), "the dense FFN is stored as tp32"
    if dense:
        routed, plain, group = 0, 9 * SHARED_TERMS, None
    elif split == "ep":
        routed, plain, group = 8, 0, None
    elif groups:
        routed, plain, group = ROUTED // groups[1], SHARED_TERMS, int(shared_group)
    else:
        routed, plain, group = ROUTED, SHARED_TERMS, None
    return {
        "format": FORMAT,
        "split": split,
        "kind": "dense" if dense else "moe",
        "chips": split_chips(split),
        "P": share.P,
        "S": share.S,
        "entries": routed + plain,
        "routed_entries": routed,
        "shared_entries": [] if dense else list(range(routed, routed + plain)),
        "shared_terms": 0 if dense or split == "ep" else SHARED_TERMS,
        "shared_group": group,
        "dense_slices": 9 if dense else 0,
        "dense_terms": SHARED_TERMS if dense else 0,
        "dtype": "bfloat4_b",
        "source": source,
        "rounding": "host, once: BFP4 of the source value",
        "tiles": "of W^T",
        "order": {
            "gate_up": "bank > entry > item > K tile > (gate, up)",
            "down": "bank > entry > output tile column > K tile",
        },
        "entry_bytes_per_bank": {"gate_up": share.gate_up_stride, "down": share.down_stride},
    }


def manifest_path(out, layer, split) -> Path:
    return Path(out) / f"layer{layer}" / split / "manifest.json"


def read_stamp(out, layer, split):
    """The stamp a layer's files were written with; None for files from before the container had one."""
    path = manifest_path(out, layer, split)
    return json.loads(path.read_text()).get("stamp") if path.is_file() else None


def stamp_difference(have, want) -> dict:
    """{field: (on disk, wanted)} of every field that differs; {} when the files are what the reader expects."""
    if have is None:
        return {"stamp": (None, "format %d" % want["format"])}
    return {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}


def check_stamp(out, layer, split, dense=False, shared_group=GROUP_SHARED, source=SOURCE) -> dict:
    """Refuse a layer's files unless they are the container this run reads: the split, its share (P, S), the
    entries and which of them are routed, the shared expert's group and terms, the format, the dtype, the order
    and the source. Equal file sizes do not say so: with another shared group the shared expert's plain slot
    would read the zero entries and the shared expert would drop out of the layer's output without an error."""
    want = container_stamp(split, dense=dense, shared_group=shared_group, source=source)
    have = read_stamp(out, layer, split)
    where = manifest_path(out, layer, split)
    assert have is not None, (
        f"{where} has no container stamp: these files are from before the format had one. "
        f"fused/model/convert.py --stamp checks them against their manifest and writes it"
    )
    diff = stamp_difference(have, want)
    assert not diff, f"{where.parent} holds another expert container than this run reads, (on disk, wanted): {diff}"
    return have


def host_stream(stream):
    return ttnn.from_torch(stream[None, None], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)


def stream_paths(out, layer, split, chip):
    d = Path(out) / f"layer{layer}" / split
    return d / f"chip{chip:02d}.gate_up.tensorbin", d / f"chip{chip:02d}.down.tensorbin"


def load_streams(device, out, layer, split, chip, dense=False, shared_group=GROUP_SHARED):
    """The chip's two DRAM stream tensors from disk (the files' stamp checked first: check_stamp)."""
    check_stamp(out, layer, split, dense=dense, shared_group=shared_group)
    tensors = []
    for path in stream_paths(out, layer, split, chip):
        host = ttnn.load_tensor(str(path))
        tensors.append(ttnn.to_device(host, device, memory_config=L.stream_memory_config(host.shape[-2])))
    return tensors


def prepare_layer(out, layer, split, dense=False):
    """dense: the layer is a dense FFN (layer 0), stored as 9 entries. MoE layers of a tensor-parallel split get
    the shared expert as entry 256."""
    share = split_share(split)
    assert not (dense and split_groups(split)), "the dense FFN of layer 0 is stored as tp32 (layer_mlp/PLAN.md 15.3)"
    t0 = time.time()
    gate_up, down = dense_experts(layer) if dense else layer_experts(layer, shared=split != "ep")
    t_load = time.time() - t0
    nbytes = 0
    for chip in range(split_chips(split)):
        gate, up, dn = chip_weights(gate_up, down, split, chip)
        p_gu, p_dn = stream_paths(out, layer, split, chip)
        p_gu.parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(str(p_gu), host_stream(L.gate_up_stream(gate, up, share)))
        ttnn.dump_tensor(str(p_dn), host_stream(L.down_stream(dn, share)))
        nbytes += p_gu.stat().st_size + p_dn.stat().st_size
    info = {
        "layer": layer,
        "split": split,
        "entries": gate_up.shape[0],
        "chips": split_chips(split),
        "seconds_total": round(time.time() - t0, 1),
        "seconds_checkpoint_read": round(t_load, 1),
        "bytes": nbytes,
        "bytes_per_chip": nbytes // split_chips(split),
    }
    if split_groups(split):
        info["shared_group"] = GROUP_SHARED
        info["entries"] = int(gate.shape[0])  # per chip: the group's experts and the shared expert's two terms
    info["stamp"] = container_stamp(split, dense=dense)
    assert info["stamp"]["entries"] == int(gate.shape[0]), (info["stamp"]["entries"], gate.shape)
    manifest_path(out, layer, split).write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--split", default="tp32")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    assert os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"), "host only: set TT_METAL_MOCK_CLUSTER_DESC_PATH"
    for layer in args.layers:
        info = prepare_layer(args.out, layer, args.split, dense=layer == 0)
        print("HY4_MK5 prepared", json.dumps(info), flush=True)
