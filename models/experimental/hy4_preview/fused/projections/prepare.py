# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight streams of the decode projection programs (fused/projections/program.py) from the checkpoint, on disk.

    python -m models.experimental.hy4_preview.fused.projections.prepare --layers 0 1 2 3 4 5 --chips 0 1 ... \\
        --formats bfp8 --out generated/hy4_preview/weights/projections

writes ``<out>/layer<L>/<format>/chip<cc>.<group>*.tensorbin`` (group = pre, query, post: the host tensor
ttnn.as_tensor caches, [1, 1, rows, 256], one [rows, 32] shard per DRAM bank) and ``manifest.json``. The programs
load them with ``cache=stream_cache(...)``. Formats: bfp8, bfp4x2 (two BFP4 terms), bfp4.

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

from . import layout as L

DEFAULT_OUT = "generated/hy4_preview/weights/projections"
FORMATS = {"bfp8": ("bfp8", 1), "bfp4x2": ("bfp4", 2), "bfp4": ("bfp4", 1)}
ATTENTION_NAMES = (
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "linear_gate.weight",
    "learnable_sink_param",
)
INDEXER_NAMES = ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight")


def is_full(layer):
    return layer == 0 or (layer - 1) % 4 == 0


def checkpoint_tensors(names):
    """Tensors by name from the BF16 checkpoint in $HY4_HF_MODEL, as fp32."""
    import safetensors

    root = Path(os.environ["HY4_HF_MODEL"])
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for name in names:
        with safetensors.safe_open(str(root / weight_map[name]), framework="pt") as f:
            out.append(f.get_tensor(name).float())
    return out


def checkpoint_layer(layer):
    """(attention dict by reference name, input norm weight, indexer dict or None) of a layer."""
    prefix = f"model.layers.{layer}."
    att = checkpoint_tensors([prefix + "self_attn." + n for n in ATTENTION_NAMES])
    attention = {
        n.removesuffix(".weight") if n != "learnable_sink_param" else n: t for n, t in zip(ATTENTION_NAMES, att)
    }
    (gamma,) = checkpoint_tensors([prefix + "input_layernorm.weight"])
    indexer = None
    if is_full(layer):
        idx = checkpoint_tensors([prefix + "self_attn.indexer." + n for n in INDEXER_NAMES])
        indexer = dict(zip(INDEXER_NAMES, idx))
    return attention, gamma, indexer


def layer_weights(layer, gamma=True):
    """gamma=False: the input norm's weight is NOT folded into the x-fed weights (q_a, kv_a, gate, the index key and
    head weights): for a layer program whose root applies the whole norm and sends x normalised (scale = 1)."""
    attention, gamma_in, indexer = checkpoint_layer(layer)
    return L.layer_weights(attention, gamma_in if gamma else None, indexer)


def load_host_stream(out, layer, fmt_name, chip, group):
    """The prepared stream of one chip and group as a host tensor (a shard of program.mesh_stream)."""
    import ttnn

    p = Path(stream_cache(out, layer, fmt_name, chip, group))
    (path,) = sorted(p.parent.glob(p.name + "*.tensorbin"))
    return ttnn.load_tensor(str(path))


def stream_cache(out, layer, fmt_name, chip, group):
    """The cache file of one chip's group. The RoPE pairing of the process is part of the path: the streams hold q_b,
    kv_a and the index weights in the frame of tt/frames.py, which depends on it at equal shapes."""
    from models.experimental.hy4_preview.reference.attention import rope_pairing

    return str(Path(out) / f"rope-{rope_pairing()}" / f"layer{layer}" / fmt_name / f"chip{chip:02d}.{group}")


def placeholder_share(chip, full):
    """A chip's share with zero weights: the shapes a program needs when its stream comes from the cache."""
    z = torch.zeros
    w = {
        "q_a": z(L.HIDDEN, 64),
        "gate": z(L.HIDDEN, L.HEADS_PER_CHIP * L.V),
        "q_b": [z(L.Q_LORA, L.QK) for _ in range(L.HEADS_PER_CHIP)],
        "w_k": [z(L.NOPE, L.KV_LORA) for _ in range(L.HEADS_PER_CHIP)],
        "w_v": [z(L.KV_LORA, L.V) for _ in range(L.HEADS_PER_CHIP)],
        "o": z(L.HEADS_PER_CHIP * L.V, L.HIDDEN),
    }
    if L.pool_of(chip, full) is not None:
        w["pool"] = z(L.HIDDEN, L.TILE)
    if full:
        w["idx_q"] = z(L.HIDDEN, L.IDX_DIM)
    return w


def main():
    import ttnn
    from . import program as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--chips", type=int, nargs="+", default=list(range(L.CHIPS)))
    ap.add_argument("--formats", nargs="+", default=["bfp8"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--normalised",
        action="store_true",
        help="streams for x that arrives normalised: no input-norm weight folded in, every term pre-rounded (the "
        "stream holds layout.stored of the weight); written to <format>_xn",
    )
    args = ap.parse_args()
    assert os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"), "host only: set TT_METAL_MOCK_CLUSTER_DESC_PATH"
    device = ttnn.open_device(device_id=0)
    try:
        for layer in args.layers:
            t0 = time.time()
            lw = layer_weights(layer, gamma=not args.normalised)
            t_read = time.time() - t0
            for fmt_name in args.formats:
                fmt, terms = FORMATS[fmt_name]
                if args.normalised:
                    fmt_name += "_xn"
                Path(stream_cache(args.out, layer, fmt_name, 0, "pre")).parent.mkdir(parents=True, exist_ok=True)
                nbytes = 0
                for chip in args.chips:
                    w = L.chip_weights(lw, chip)
                    for group, build in (("pre", P.pre_program), ("query", P.query_program), ("post", P.post_program)):
                        cache = stream_cache(args.out, layer, fmt_name, chip, group)
                        op = build(device, w, fmt=fmt, terms=terms, cache=cache, pre_rounded=args.normalised)
                        nbytes += op.weight_bytes
                manifest = {
                    "layer": layer,
                    "full": lw.full,
                    "format": fmt_name,
                    "chips": args.chips,
                    "bytes": nbytes,
                    "bytes_per_chip": nbytes // len(args.chips),
                    "seconds_checkpoint_read": round(t_read, 1),
                    "seconds_total": round(time.time() - t0, 1),
                }
                path = Path(args.out) / f"layer{layer}" / fmt_name / "manifest.json"
                old = json.loads(path.read_text()) if path.exists() else {}
                manifest["chips"] = sorted(set(old.get("chips", [])) | set(args.chips))
                path.write_text(json.dumps(manifest, indent=1))
                print("HY4_MK7 prepared", json.dumps(manifest), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
