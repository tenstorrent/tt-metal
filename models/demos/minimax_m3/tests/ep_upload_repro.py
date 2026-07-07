# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal repro of the EP expert-weight UPLOAD crash (no 870GB model load).

Builds the real-sized M3 MoE expert weights (128 experts, emb=6144, hidden=3072, bf4) onto the (8,4)
mesh, ONE "layer" at a time, KEEPING them resident (like the real 60-layer model), under faulthandler.
This isolates the exact op that faults and tells us WHICH hypothesis is right:
  - crashes at the SAME layer every run  -> deterministic (code/shape bug OR cumulative DRAM ceiling)
  - crashes at a RANDOM layer            -> intermittent device/UMD fault
  - never crashes resident for 60 layers -> upload is fine; real crash is elsewhere (other resident tensors)

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m3/tests/ep_upload_repro.py            # bf4 (faithful), EP_SEQ=640
"""

import faulthandler
import os
import sys

import torch

import ttnn

faulthandler.enable()

EMB, HIDDEN, E, TOPK, NLAYERS = 6144, 3072, 128, 8, 60


def dram_used_gb(mesh):
    """Best-effort per-chip DRAM-used readout; returns None if the API isn't available."""
    try:
        dev = mesh.get_devices()[0]
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        for attr in ("total_bytes_allocated_per_bank", "total_bytes_allocated"):
            v = getattr(mv, attr, None)
            if v is not None:
                nb = getattr(mv, "num_banks", 1) or 1
                return (v * (nb if "per_bank" in attr else 1)) / 1e9
    except Exception:
        return None
    return None


def main():
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
    from models.demos.minimax_m3.tt.experts_throughput.tt_minimax_moe import TtMiniMaxMoE

    ep_seq = int(os.getenv("EP_SEQ_PER_CHIP", "640"))
    dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    keep_resident = os.getenv("KEEP_RESIDENT", "1") == "1"
    rows, cols = 8, 4
    print(
        f"[repro] mesh=({rows},{cols}) ep_seq={ep_seq} dtype={dtype} keep_resident={keep_resident} "
        f"E={E} emb={EMB} hidden={HIDDEN}",
        flush=True,
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        mc = extract_mesh_config(mesh)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            ep_seq, E, TOPK, mesh.get_num_devices(), dgs, 2
        )
        print(
            f"[repro] dgs={dgs} ndg={ndg} experts_per_chip={experts_per_chip} max_tok={max_tok} max_buf={max_buf}",
            flush=True,
        )

        # Random expert weights — shapes match mlp.py / TtRoutedExpert dummy layout.
        gate_w = {"weight": torch.randn(E, EMB) * 0.02, "e_score_correction_bias": torch.randn(E) * 0.1}
        routed_w = [
            {
                "gate_proj": torch.randn(HIDDEN, EMB) * 0.02,
                "up_proj": torch.randn(HIDDEN, EMB) * 0.02,
                "down_proj": torch.randn(EMB, HIDDEN) * 0.02,
            }
            for _ in range(E)
        ]

        resident = []
        for layer in range(NLAYERS):
            moe = TtMiniMaxMoE(
                mesh_device=mesh,
                dispatch_group_size=dgs,
                num_dispatch_groups=ndg,
                experts_per_chip=experts_per_chip,
                num_routed_experts=E,
                num_experts_per_tok=TOPK,
                metadata_len=metadata_len,
                max_dispatched_tokens_per_expert=max_tok,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=ep_seq,
                emb_dim=EMB,
                hidden_dim=HIDDEN,
                gate_weights=gate_w,
                routed_expert_weights=routed_w,
                num_links=1,
                routed_expert_weights_dtype=dtype,
                weight_cache_path=None,
                layer_idx=layer,
            )
            if keep_resident:
                resident.append(moe)
            used = dram_used_gb(mesh)
            print(
                f"[repro] built layer {layer} OK | DRAM/chip used: " f"{used:.2f} GB"
                if used is not None
                else f"[repro] built layer {layer} OK",
                flush=True,
            )
        print(f"[repro] ALL {NLAYERS} layers built resident with NO crash -> upload path is fine.", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
