# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Consolidated PCC bring-up for the tt-lang-INDEPENDENT LEAF text ops.

Opens the bh_galaxy mesh ONCE and validates all five leaf blocks against the
reference golden tensors (PCC > 0.99 on the mesh):

  rms_norm, final_norm, qk_norm, embedding, rope

Each block loads the SAME input + weights the reference used (from
``reference/golden/<block>.input.pt`` / ``.weight.pt`` / ``.params.pt``) and
compares the TTNN output to ``reference/golden/<block>.pt``.

Run standalone (prints a JSON summary as the last line) or under pytest.
"""

from __future__ import annotations

import json
import os
import sys

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimaxai_minimax_m3.tt import model_config as mc
from models.demos.minimaxai_minimax_m3.tt.embedding import Embedding
from models.demos.minimaxai_minimax_m3.tt.final_norm import FinalNorm
from models.demos.minimaxai_minimax_m3.tt.qk_norm import QKNorm
from models.demos.minimaxai_minimax_m3.tt.rms_norm import RMSNorm
from models.demos.minimaxai_minimax_m3.tt.rope import RotaryEmbedding, build_cos_sin, cos_sin_to_mesh
from models.demos.minimaxai_minimax_m3.tt.weight_loader import to_mesh_tensor

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
PCC_TARGET = 0.99


def _g(name):
    return torch.load(os.path.join(GOLDEN, name), map_location="cpu", weights_only=False)


def _replicated_norm_weight(mesh, w):
    """gemma +1 folded, replicated, tile layout."""
    return to_mesh_tensor(w, mesh, shard="replicate", dtype=mc.NORM_WEIGHT_DTYPE, add_gemma_one=True)


def _act_to_mesh(mesh, t, dtype=mc.ACT_DTYPE, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mc.replicate_mapper(mesh))


def _from_dev0(t):
    """Read one device copy back to torch (replicated tensors)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


# --------------------------------------------------------------------------- #
def run_rms_norm(mesh):
    x = _g("rms_norm.input.pt")  # [1,128,6144] bf16
    w = _g("rms_norm.weight.pt")  # [6144]
    ref = _g("rms_norm.pt")
    tt_x = _act_to_mesh(mesh, x)
    tt_w = _replicated_norm_weight(mesh, w)
    block = RMSNorm(mesh, tt_w, eps=1e-6)
    out = _from_dev0(block(tt_x)).reshape(ref.shape)
    return comp_pcc(ref.float(), out.float(), PCC_TARGET)


def run_final_norm(mesh):
    x = _g("final_norm.input.pt")
    w = _g("final_norm.weight.pt")
    ref = _g("final_norm.pt")
    tt_x = _act_to_mesh(mesh, x)
    tt_w = _replicated_norm_weight(mesh, w)
    block = FinalNorm(mesh, tt_w, eps=1e-6)
    out = _from_dev0(block(tt_x)).reshape(ref.shape)
    return comp_pcc(ref.float(), out.float(), PCC_TARGET)


def run_qk_norm(mesh):
    inp = _g("qk_norm.input.pt")  # {q:[1,64,8,128], k:[1,4,8,128]}
    params = _g("qk_norm.params.pt")  # {q_weight, k_weight, eps}
    ref = _g("qk_norm.pt")
    tt_q = _act_to_mesh(mesh, inp["q"])
    tt_k = _act_to_mesh(mesh, inp["k"])
    tt_qw = _replicated_norm_weight(mesh, params["q_weight"])
    tt_kw = _replicated_norm_weight(mesh, params["k_weight"])
    block = QKNorm(mesh, tt_qw, tt_kw, eps=float(params["eps"]))
    q_out, k_out = block(tt_q, tt_k)
    q_t = _from_dev0(q_out).reshape(ref["q"].shape)
    k_t = _from_dev0(k_out).reshape(ref["k"].shape)
    _, pq = comp_pcc(ref["q"].float(), q_t.float(), PCC_TARGET)
    _, pk = comp_pcc(ref["k"].float(), k_t.float(), PCC_TARGET)
    pcc = min(_pcc_num(pq), _pcc_num(pk))
    return (pcc >= PCC_TARGET), f"q={pq} k={pk}"


def run_embedding(mesh):
    ids = _g("embedding.input.pt")  # [2,8] int64
    ref = _g("embedding.pt")  # [2,8,6144] bf16
    # embedding table: reference golden does not ship the table; use the same
    # weight the reference used by loading from the checkpoint embed_tokens.
    from models.demos.minimaxai_minimax_m3.tt import model_config as _mc
    from models.demos.minimaxai_minimax_m3.tt.weight_loader import load_torch_weight

    table = load_torch_weight(_mc.EMBED_TOKENS_KEY)
    tt_table = ttnn.from_torch(
        table, dtype=mc.ACT_DTYPE, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=mc.replicate_mapper(mesh)
    )
    tt_ids = ttnn.from_torch(
        ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=mc.replicate_mapper(mesh),
    )
    block = Embedding(mesh, tt_table)
    out = _from_dev0(block(tt_ids)).reshape(ref.shape)
    return comp_pcc(ref.float(), out.float(), PCC_TARGET)


def run_rope(mesh):
    inp = _g("rope.input.pt")
    ref = _g("rope.pt")
    q, k = inp["q"], inp["k"]  # [1,8,128,128], [1,4,128,128]
    meta = inp["meta"]
    seq_len = meta["seq_len"]
    rotary_dim = meta["rotary_dim"]
    theta = meta["theta"]
    pos = inp.get("position_ids")
    # build cos/sin host-side (must match the golden cos/sin in the input bundle)
    cos, sin = build_cos_sin(seq_len, rotary_dim=rotary_dim, theta=theta, position_ids=pos)
    tt_cos, tt_sin = cos_sin_to_mesh(cos, sin, mesh)
    tt_q = _act_to_mesh(mesh, q)
    tt_k = _act_to_mesh(mesh, k)
    block = RotaryEmbedding(mesh, tt_cos, tt_sin, rotary_dim=rotary_dim)
    q_out, k_out = block(tt_q, tt_k)
    q_t = _from_dev0(q_out).reshape(ref["q_embed"].shape)
    k_t = _from_dev0(k_out).reshape(ref["k_embed"].shape)
    _, pq = comp_pcc(ref["q_embed"].float(), q_t.float(), PCC_TARGET)
    _, pk = comp_pcc(ref["k_embed"].float(), k_t.float(), PCC_TARGET)
    pcc = min(_pcc_num(pq), _pcc_num(pk))
    return (pcc >= PCC_TARGET), f"q={pq} k={pk}"


def _pcc_num(pcc_str):
    """comp_pcc returns a message string like 'PCC: 0.999...'; extract the float."""
    if isinstance(pcc_str, (int, float)):
        return float(pcc_str)
    try:
        return float(str(pcc_str).split(":")[-1].strip().split()[0])
    except Exception:
        return 0.0


BLOCKS = {
    "rms_norm": run_rms_norm,
    "final_norm": run_final_norm,
    "qk_norm": run_qk_norm,
    "embedding": run_embedding,
    "rope": run_rope,
}


def main():
    results = {}
    mesh = None
    last_error = None
    try:
        mesh = mc.open_mesh()
    except Exception as e:  # noqa: BLE001
        # fabric/topology degraded: fall back to the largest available mesh so
        # the (replicated / per-head) leaf math can still be validated.
        last_error = f"open (1,32) failed: {e}"
        try:
            ttnn.set_fabric_config(mc.FABRIC_CONFIG)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        except Exception as e2:  # noqa: BLE001
            print(json.dumps({"error": f"{last_error} | fallback failed: {e2}"}))
            sys.exit(2)

    for name, fn in BLOCKS.items():
        try:
            res = fn(mesh)
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
                passed, detail = res
                results[name] = {"pass": bool(passed), "detail": str(detail)}
            else:
                passed, detail = res
                results[name] = {"pass": bool(passed), "detail": str(detail)}
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            results[name] = {"pass": False, "detail": f"EXC: {e}"}

    mc.close_mesh(mesh)
    print(json.dumps({"results": results, "fallback": last_error}))


if __name__ == "__main__":
    main()
