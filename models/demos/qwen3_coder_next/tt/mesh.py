# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chip placement for the `Qwen/Qwen3-Coder-Next` pipeline: TP=2 x DP=16 on 32 chips.

WHY TP=2, AND WHY IT IS CARVED OUT OF THE 32-CHIP MESH
------------------------------------------------------
The bring-up tool selected **TP=2 x DP=16** for 32 chips, and the per-component scheme check
against THIS model's config agrees that 2 is the largest degree every graduated stub can hold:

| stub               | scheme                                   | divisibility at TP=d           | max d |
|--------------------|------------------------------------------|--------------------------------|-------|
| `attention`        | head-wise TP, all_reduce on `o_proj`     | 16 heads AND **2 kv heads** % d| **2** |
| `gated_delta_net`  | head-wise TP, all_reduce on `out_proj`   | 16 k-heads and 32 v-heads % d  | 16    |
| `experts`          | EXPERT-parallel, all_reduce              | 512 experts % d                | 512   |
| `top_k_router`     | expert-axis column-parallel, all_gather  | 512 % d, and 512/d >= 32 (tile)| 16    |
| `m_l_p`            | column/row-parallel, all_reduce          | 512 inter % d, 512/d >= 32     | 16    |
| `sparse_moe_block` | composite of the three above             | -                              | 16    |
| `decoder_layer`    | composite; norms replicated at the seams | -                              | -     |

`num_key_value_heads = 2` is the binding constraint, so **d = 2**. That is not a compromise: each
stub's own build guard silently falls back to a REPLICATED placement when its divisibility fails,
so running the stack at d = 32 would quietly de-parallelise `attention` and `gated_delta_net` --
the pure-replication outcome this placement is required not to produce.

The graduated stubs shard with `ttnn.ShardTensorToMesh(device, dim=...)` and reduce with a
mesh-wide `ttnn.all_reduce`, i.e. they treat *every chip of the device handed to them* as one TP
group.  Composing them AS-IS at TP=2 therefore means handing them a **2-chip device**.  So the
caller's 32-chip mesh is carved into `MeshShape(1, 2)` SUBMESHES: 16 of them, one per DP replica,
each a TP=2 group.  Mesh axis 1 is the tensor-parallel axis and axis 0 is the data-parallel axis,
exactly as specified -- a (1, 2) tile spans the TP axis and steps along the DP axis.

THIS MODULE NEVER OPENS A DEVICE.  It only re-shapes the one it is handed.  The package's single
opener is `device_harness.open_mesh()`, one level up and outside `tt/`, so the pipeline always
runs on the device its caller opened -- see that module's docstring.

Placement changes only WHERE the pipeline runs.  The PCC gate is measured against the same HF
golden at every rung.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn

_MANIFEST = Path(__file__).resolve().parent.parent / "parallelism_manifest.json"


def _manifest():
    try:
        return json.loads(_MANIFEST.read_text())
    except Exception:
        return {"chips": 32, "tp": 8, "dp": 4, "mesh": [4, 8]}


MANIFEST = _manifest()
# The tensor-parallel degree bring-up proved and the stubs are sharded for.
TP_DEGREE = int(os.environ.get("TT_QWEN3_TP", MANIFEST.get("tp", 8)))
DP_DEGREE = int(MANIFEST.get("dp", 4))

def _tp_submesh_shapes(tp):
    """Submesh shapes worth trying for a TP group of `tp` chips, widest-along-the-TP-axis first.

    A submesh must divide the parent mesh ELEMENTWISE, so (1, tp) is impossible whenever the
    parent's rows are narrower than tp. On this box SystemMesh is (8, 4): a TP=8 group has to be
    (2, 4) -- asking for (1, 8) raises
        TT_FATAL: Shape MeshShape([8, 4]) is not divisible by submesh shape MeshShape([1, 8])
    and the old code then fell back to ONE 32-chip TP group (TP=32), where `gated_delta_net`
    (max 16), `m_l_p` and `top_k_router` all de-parallelise -- measured e2e PCC 0.60.
    """
    pairs = [(r, tp // r) for r in range(1, tp + 1) if tp % r == 0]
    pairs.sort(key=lambda rc: (-rc[1], rc[0]))   # prefer wide-on-axis-1, then fewest rows
    return pairs


def tp_groups(parent, tp=None):
    """Carve the open mesh into TP groups of `tp` chips -- one per data-parallel replica.

    Returns `(groups, note)`.  `groups[i]` is the device the pipeline is built on; the graduated
    stubs see exactly `tp` chips and apply the TP split bring-up proved.  Falls back to the parent
    device itself when the mesh is already that wide (or submeshes are unavailable), printing what
    it landed on -- never silently.
    """
    tp = int(tp or TP_DEGREE)
    total = parent.get_num_devices()
    if total <= tp:
        print(f"[placement] TP group = the whole {total}-chip mesh (TP={total}, DP=1)", flush=True)
        return [parent], f"TP={total} DP=1"
    groups, last_exc = None, None
    for rows, cols in _tp_submesh_shapes(tp):
        try:
            cand = list(parent.create_submeshes(ttnn.MeshShape(rows, cols)))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[placement] MeshShape({rows}, {cols}) rejected ({str(exc).splitlines()[0][:80]})", flush=True)
            continue
        if cand and cand[0].get_num_devices() == tp:
            print(f"[placement] MeshShape({rows}, {cols}) accepted -> {len(cand)} group(s) of {tp}", flush=True)
            groups = cand
            break
        print(f"[placement] MeshShape({rows}, {cols}) gave {cand[0].get_num_devices() if cand else 0} chips, not {tp}", flush=True)
    if groups is None:
        print(
            f"[placement] no submesh shape of {tp} chips divides this mesh "
            f"(last error: {str(last_exc).splitlines()[0][:90] if last_exc else 'none'}); "
            f"falling back to the whole {total}-chip mesh as ONE TP group",
            flush=True,
        )
        return [parent], f"TP={total} DP=1"
    print(
        f"[placement] carved {len(groups)} data-parallel replicas x TP={groups[0].get_num_devices()} "
        f"= {len(groups) * groups[0].get_num_devices()} chips  (TP axis = mesh axis 1)",
        flush=True,
    )
    return groups, f"TP={groups[0].get_num_devices()} DP={len(groups)}"


def rows_cols(device):
    shape = getattr(device, "shape", None)
    try:
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    except TypeError:
        pass
    n = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    return 1, int(n)


def replicate(device):
    """Mapper that puts an identical copy of a tensor on every chip of `device`."""
    return ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None


def shard_tp(device, dim):
    """Mapper that splits `dim` across the chips of a TP group -- the stubs' own convention."""
    return ttnn.ShardTensorToMesh(device, dim=dim) if device.get_num_devices() > 1 else None


def all_gather_tp(tensor, dim, device):
    """Re-assemble a column-parallel tensor across the TP group (no-op at TP=1)."""
    if device.get_num_devices() <= 1:
        return tensor
    return ttnn.all_gather(tensor, dim=dim)


def to_host(tensor):
    """Bring ONE chip's copy of a replicated device tensor back to torch.

    Composing all shards would materialise a copy per chip of a `(1, 1, C, 151936)` logits
    tensor -- gigabytes for a value that is identical on every chip by construction.
    """
    shards = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(shards[0] if shards else tensor)
