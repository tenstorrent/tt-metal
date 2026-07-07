# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-block variant of capture_qkv_plugin: dump post-RoPE attn1 Q/K/V for EVERY
Stage-2 block index in CAP_BLOCKS (comma list) in a single denoise run, one .pt per
block under CAP_OUT_DIR. Used to map the dense->windowable depth transition for L4."""
import os

import torch

import ttnn

CAP_BLOCKS = sorted({int(x) for x in os.environ.get("CAP_BLOCKS", "8,12,16,20,32,40,47").split(",") if x.strip()})
CAP_OUT_DIR = os.environ.get("CAP_OUT_DIR", "tmp/qkv_depth")
CAP_HEADS = int(os.environ.get("CAP_HEADS", "8"))
CAP_QNSHARD = int(os.environ.get("CAP_QNSHARD", "4864"))  # stage-2 per-device Q shard

_state = {"count": 0, "saved": set(), "orig": None}


def _capture(q, k, v, mesh_device, idx):
    md = mesh_device
    shp = tuple(md.shape)
    dims = [None, None]
    for ax, sz in enumerate(shp):
        dims[ax] = 2 if sz == 8 else 1
    comp = ttnn.ConcatMesh2dToTensor(md, dims=dims, mesh_shape=shp)
    out = {}
    for name, t in (("q", q), ("k", k), ("v", v)):
        full = ttnn.to_torch(t, mesh_composer=comp)
        out[name] = full[:, :CAP_HEADS, :, :].contiguous().to(torch.bfloat16)
        del full
    out["meta"] = {"cap_block": idx, "mesh_shape": shp, "heads_saved": CAP_HEADS, "full_shape": tuple(out["q"].shape)}
    os.makedirs(CAP_OUT_DIR, exist_ok=True)
    path = os.path.join(CAP_OUT_DIR, f"qkv_b{idx}.pt")
    torch.save(out, path)
    print(f"[capture_multi] SAVED block={idx} -> {path} q{tuple(out['q'].shape)}", flush=True)


def _install():
    orig = ttnn.transformer.ring_joint_scaled_dot_product_attention
    _state["orig"] = orig

    def wrapper(q, k, v, *args, **kwargs):
        if len(_state["saved"]) < len(CAP_BLOCKS):
            try:
                is_cross = bool(kwargs.get("is_cross", False))
                qshape = tuple(q.shape)
                is_target = (not is_cross) and len(qshape) == 4 and qshape[2] == CAP_QNSHARD
            except Exception:
                is_target = False
            if is_target:
                idx = _state["count"]
                _state["count"] += 1
                if idx in CAP_BLOCKS and idx not in _state["saved"]:
                    try:
                        _capture(q, k, v, kwargs.get("mesh_device"), idx)
                        _state["saved"].add(idx)
                    except Exception as e:  # noqa: BLE001
                        print(f"[capture_multi] CAPTURE FAILED block={idx}: {e!r}", flush=True)
                        _state["saved"].add(idx)
        return orig(q, k, v, *args, **kwargs)

    ttnn.transformer.ring_joint_scaled_dot_product_attention = wrapper
    print(f"[capture_multi] installed hook: CAP_BLOCKS={CAP_BLOCKS} CAP_OUT_DIR={CAP_OUT_DIR}", flush=True)


_install()
