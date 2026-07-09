# SPDX-License-Identifier: Apache-2.0
"""
pytest plugin (load via `-p capture_qkv_plugin`) that captures ONE real post-RoPE
attn1 (video self-attention) Q/K/V triple from a live LTX Stage-2 denoise step,
WITHOUT editing any shared source file.

Mechanism: monkeypatch ttnn.transformer.ring_joint_scaled_dot_product_attention.
attention_ltx.py calls it as `ttnn.transformer.ring_joint_scaled_dot_product_attention(...)`
(a live attribute lookup at call time), so replacing that module attribute is picked
up transparently.

Filter for the target op:
  - NOT is_cross         -> excludes the a2v/v2a cross ring-SDPA calls
  - q.shape[2] == 4864   -> Stage-2 video self-attn per-device Q shard (38912/8).
                            Stage-1 self-attn shard is 1216; audio self-attn is smaller.
The k-th such call in a step == block k (blocks run in order). We capture the call
whose running index == CAP_BLOCK (default 24, mid-depth of 48), once.

We gather the SP-sharded (per-chip) q/k/v back to full sequence via ConcatMesh2dToTensor:
  mesh axis of size 8 (SP) -> concat tensor dim 2 (N)  -> full 38912 tokens
  mesh axis of size 4 (TP) -> concat tensor dim 1 (H)  -> full 32 heads
q_BHNE/k_BHNE/v_BHNE passed to ring-SDPA are the per-chip Q shard and per-chip K/V
shard (the ring gathers K/V internally); concatenating the SP shards reconstructs the
full K/V, and the middle SP chip's slice is the local Q shard. Saved to CAP_OUT.
"""
import os

import torch

import ttnn

CAP_BLOCK = int(os.environ.get("CAP_BLOCK", "24"))
CAP_OUT = os.environ.get("CAP_OUT", "tmp/real_attn_qkv.pt")
CAP_HEADS = int(os.environ.get("CAP_HEADS", "8"))  # heads to save (from TP row 0)
CAP_QNSHARD = int(os.environ.get("CAP_QNSHARD", "4864"))  # stage-2 per-device Q shard

_state = {"count": 0, "done": False, "orig": None}


def _capture(q, k, v, mesh_device):
    md = mesh_device
    shp = tuple(md.shape)
    # axis size 8 -> SP -> N (dim2); axis size 4 -> TP -> H (dim1)
    dims = [None, None]
    for ax, sz in enumerate(shp):
        dims[ax] = 2 if sz == 8 else 1
    comp = ttnn.ConcatMesh2dToTensor(md, dims=dims, mesh_shape=shp)

    out = {}
    for name, t in (("q", q), ("k", k), ("v", v)):
        full = ttnn.to_torch(t, mesh_composer=comp)  # [1, 32, 38912, 128]
        # keep only the first CAP_HEADS heads (TP row 0's local heads), store bf16
        sl = full[:, :CAP_HEADS, :, :].contiguous().to(torch.bfloat16)
        out[name] = sl
        del full

    out["meta"] = {
        "cap_block": CAP_BLOCK,
        "mesh_shape": shp,
        "concat_dims": dims,
        "heads_saved": CAP_HEADS,
        "q_nshard": CAP_QNSHARD,
        "full_shape": tuple(out["q"].shape),  # [1, CAP_HEADS, 38912, 128]
        "note": "post-RoPE attn1 video self-attn, SP-gathered full seq, first CAP_HEADS heads",
    }
    torch.save(out, CAP_OUT)
    print(
        f"[capture_qkv] SAVED block={CAP_BLOCK} -> {CAP_OUT} "
        f"q{tuple(out['q'].shape)} k{tuple(out['k'].shape)} v{tuple(out['v'].shape)} "
        f"mesh={shp} dims={dims} dtype={out['q'].dtype}",
        flush=True,
    )


def _install():
    orig = ttnn.transformer.ring_joint_scaled_dot_product_attention
    _state["orig"] = orig

    def wrapper(q, k, v, *args, **kwargs):
        if not _state["done"]:
            try:
                is_cross = bool(kwargs.get("is_cross", False))
                qshape = tuple(q.shape)
                is_target = (not is_cross) and len(qshape) == 4 and qshape[2] == CAP_QNSHARD
            except Exception:
                is_target = False
            if is_target:
                idx = _state["count"]
                _state["count"] += 1
                if idx == CAP_BLOCK:
                    try:
                        _capture(q, k, v, kwargs.get("mesh_device"))
                    except Exception as e:  # noqa: BLE001
                        print(f"[capture_qkv] CAPTURE FAILED: {e!r}", flush=True)
                    _state["done"] = True
        return orig(q, k, v, *args, **kwargs)

    ttnn.transformer.ring_joint_scaled_dot_product_attention = wrapper
    print(
        f"[capture_qkv] installed hook: CAP_BLOCK={CAP_BLOCK} CAP_OUT={CAP_OUT} "
        f"CAP_HEADS={CAP_HEADS} CAP_QNSHARD={CAP_QNSHARD}",
        flush=True,
    )


def pytest_configure(config):  # noqa: ARG001
    _install()
