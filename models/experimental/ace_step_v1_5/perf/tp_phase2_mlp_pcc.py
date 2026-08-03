# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP Phase 2 gate: ``TtQwen3MLP`` tensor-parallel (column gate/up + row-parallel down + all-reduce)
must match the replicate baseline AND the torch reference on the real BH_QB 2×2 mesh.

Run (device free):
    python models/experimental/ace_step_v1_5/perf/tp_phase2_mlp_pcc.py

Builds the tiny DiT fixture, runs the MLP three ways in one process (torch ref; TT replicate with
ACE_STEP_TP=off; TT tensor-parallel with ACE_STEP_TP=on) and prints PCC. Gate: TP-on vs replicate
and TP-on vs torch both ≥ 0.99.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import tiny_dit_decoder_fixture
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import TtQwen3MLP
from models.experimental.ace_step_v1_5.utils.tt_device import (
    ace_step_replicate_mesh_mapper,
    open_dit_device,
)

_GATE = 0.99


def _torch_mlp_ref(sd: dict, base: str, x: torch.Tensor) -> torch.Tensor:
    wg = torch.from_numpy(sd[f"{base}.gate_proj.weight"]).to(torch.float32)
    wu = torch.from_numpy(sd[f"{base}.up_proj.weight"]).to(torch.float32)
    wd = torch.from_numpy(sd[f"{base}.down_proj.weight"]).to(torch.float32)
    xf = x.to(torch.float32)
    h = F.silu(F.linear(xf, wg)) * F.linear(xf, wu)
    return F.linear(h, wd)


def _run_mlp(mesh, sd, base, d_model, intermediate, x) -> torch.Tensor:
    mlp = TtQwen3MLP(
        state_dict=sd,
        base_address=base,
        mesh_device=mesh,
        hidden_size=d_model,
        intermediate_size=intermediate,
        dtype=ttnn.bfloat16,
    )
    # Activations are replicated across the mesh (column-parallel expects a full x on each chip).
    x_tt = ttnn.from_torch(
        x,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ace_step_replicate_mesh_mapper(mesh),
    )
    y_tt = mlp(x_tt)
    # Output is replicated across the mesh (post all-reduce for TP; plain replicate for off).
    # Read device-0's shard directly: auto_compose mis-infers the post-CCL topology
    # ("dims must be unique") for the all-reduced tensor.
    return ttnn.to_torch(ttnn.get_device_tensors(y_tt)[0]).to(torch.float32)


def main() -> int:
    cfg, sd, d_model, seq_len, _enc = tiny_dit_decoder_fixture()
    base = "layers.0.mlp"
    intermediate = int(sd[f"{base}.gate_proj.weight"].shape[0])
    torch.manual_seed(0)
    x = torch.randn(1, 1, seq_len, d_model, dtype=torch.bfloat16)

    ref = _torch_mlp_ref(sd, base, x)

    print(f"[phase2] dims: d_model={d_model} intermediate={intermediate} seq_len={seq_len}", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = open_dit_device(ttnn, mesh_sku="BH_QB", num_command_queues=1)
    try:
        os.environ["ACE_STEP_TP"] = "off"
        y_off = _run_mlp(mesh, sd, base, d_model, intermediate, x)

        os.environ["ACE_STEP_TP"] = "on"
        y_on = _run_mlp(mesh, sd, base, d_model, intermediate, x)

        _, pcc_off_ref = comp_pcc(ref, y_off, pcc=_GATE)
        _, pcc_on_ref = comp_pcc(ref, y_on, pcc=_GATE)
        _, pcc_on_off = comp_pcc(y_off, y_on, pcc=_GATE)
        print(f"[phase2][PCC] replicate(off) vs torch : {float(pcc_off_ref):.6f}", flush=True)
        print(f"[phase2][PCC] TP(on)       vs torch : {float(pcc_on_ref):.6f}", flush=True)
        print(f"[phase2][PCC] TP(on)       vs replicate : {float(pcc_on_off):.6f}", flush=True)
        ok = float(pcc_on_ref) >= _GATE and float(pcc_on_off) >= _GATE
        print(f"[phase2] GATE {'PASS' if ok else 'FAIL'} (threshold {_GATE})", flush=True)
        return 0 if ok else 1
    finally:
        try:
            from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device

            close_ace_step_device(ttnn, mesh)
        except Exception:
            pass
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
