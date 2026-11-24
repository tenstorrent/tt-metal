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


def host_stream(stream):
    return ttnn.from_torch(stream[None, None], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)


def stream_paths(out, layer, split, chip):
    d = Path(out) / f"layer{layer}" / split
    return d / f"chip{chip:02d}.gate_up.tensorbin", d / f"chip{chip:02d}.down.tensorbin"


def load_streams(device, out, layer, split, chip):
    """The chip's two DRAM stream tensors from disk."""
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
    (Path(out) / f"layer{layer}" / split / "manifest.json").write_text(json.dumps(info, indent=1))
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
