# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert weight streams of the streamed decode MoE (fused/moe/layout.py) from the checkpoint, on disk.

    python -m models.experimental.hy4_preview.fused.moe.prepare --layers 1 2 3 4 5 --split tp32 --out <dir>

writes, per layer and chip, ``<out>/layer<L>/<split>/chip<cc>.{gate_up,down}.tensorbin`` (host BFP4 tensors,
[1, 1, rows, 256]: one [rows, 32] shard per DRAM bank) and ``manifest.json`` with the time and the bytes.
Splits: tp<N> (chip c holds intermediate channels [c, c + 1) * 2048 / N of every expert) and ep (chip c holds the
8 whole experts [8 c, 8 c + 8)).

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

DEFAULT_OUT = "generated/hy4_preview/moe_streams"


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


def split_share(split):
    return L.ep_share() if split == "ep" else L.tp_share(int(split[2:]))


def split_chips(split):
    return 32 if split == "ep" else int(split[2:])


def chip_weights(gate_up, down, split, chip, experts=None):
    """The chip's share (gate, up [E, I_c, 6144], down [E, 6144, I_c]) as fp32. experts: only these (global ids,
    for a tensor-parallel split)."""
    I = L.INTERMEDIATE
    if split == "ep":
        sel = slice(8 * chip, 8 * chip + 8)
        return gate_up[sel, :I].float(), gate_up[sel, I:].float(), down[sel].float()
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
