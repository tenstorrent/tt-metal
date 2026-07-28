# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
E3 — MiniMax-M2 EXPERT-PARALLEL MoE block vs torch reference, on a real Galaxy.

Reuses the DeepSeek EP machinery (it's generic + already validated on this box):
  gate -> deepseek_prefill.dispatch -> routed_expert_ffn -> combine -> reduce
but with MiniMax-M2 specifics:
  - emb_dim=3072, hidden_dim=1536, 256 experts / top-8
  - NO expert groups (n_expert_groups=1, n_limited_groups=1) -> plain top-8
  - NO shared expert (shared_expert_weights=None)
  - route_scale=1.0

Validates the TTNN EP MoE (TtMoe) against the TorchMoe reference (both DeepSeek's,
configured identically) -> proves the EP mechanics are correct at MiniMax dims.
Random weights; isolated from attention/SP. Standalone (no pytest fixtures).

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m2/tests/test_ep_moe_vs_ref.py --rows 8 --cols 4 --seq 128
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)  # matches stock single_bh_galaxy [8,4] MGD
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128, help="seq_len_per_chip (small = fast iteration)")
    ap.add_argument("--capacity", type=int, default=8, help="dispatch_buffer_capacity_factor")
    args = ap.parse_args()

    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
    from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        ExpertMapping,
        compute_constants,
        create_gate_weights,
        create_shared_expert_weights,
        create_torch_expert_weights,
        extract_mesh_config,
        get_tp_mesh_composer,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import open_mesh_device
    from models.demos.minimax_m2.tt.experts_throughput.tt_minimax_moe import TtMiniMaxMoE

    torch.manual_seed(42)

    # MiniMax-M2 MoE dims
    EMB, HID, E, K = 3072, 1536, 256, 8
    N_GROUP, TOPK_GROUP, ROUTE_SCALE = 1, 1, 1.0  # no groups, no scaling
    shape = (args.rows, args.cols)
    print(
        f"[ep-moe] mesh={shape} seq/chip={args.seq} experts={E}/top{K} emb={EMB} hid={HID} (no groups, no shared)",
        flush=True,
    )

    mesh = open_mesh_device(shape, MiniMaxM27Config)  # FABRIC_1D (sp<=8) + MiniMax payload
    print(f"[ep-moe] mesh opened: {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)

    try:
        mc = extract_mesh_config(mesh)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
            args.seq, E, K, mesh.get_num_devices(), dgs, args.capacity
        )
        print(
            f"[ep-moe] dispatch_group_size={dgs} num_dispatch_groups={ndg} experts/chip={experts_per_chip}", flush=True
        )

        routed_w = create_torch_expert_weights(E, EMB, HID, seed=1234)
        gate_w = create_gate_weights(E, EMB, seed=9012)
        # MiniMax has NO shared expert. TtMoe/TorchMoe always build one, and None ->
        # identity-init (needs emb==hidden). Use a ZERO-weight shared expert so its
        # contribution is 0 for both ref and TT (silu(0)*0 @ 0 == 0).
        shared_w = {k: torch.zeros_like(v) for k, v in create_shared_expert_weights(EMB, HID, seed=5678).items()}
        dispatch_table = ExpertMapping.create_dispatch_table(E, dgs, ndg)

        x = torch.randn(dgs, args.seq, EMB, dtype=torch.bfloat16)

        # --- torch reference (no shared expert, no groups) ---
        torch_moe = TorchMoe(
            dispatch_group_size=dgs,
            experts_per_chip=experts_per_chip,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_tok,
            max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=args.seq,
            emb_dim=EMB,
            hidden_dim=HID,
            expert_dispatch_table=dispatch_table,
            num_dispatch_groups=ndg,
            routed_expert_weights=routed_w,
            shared_expert_weights=shared_w,
            gate_weights=gate_w,
            n_expert_groups=N_GROUP,
            n_limited_groups=TOPK_GROUP,
            route_scale=ROUTE_SCALE,
        )
        torch_out, _ = torch_moe(x, return_intermediates=True)
        print(f"[ep-moe] torch reference done, out shape {tuple(torch_out.shape)}", flush=True)

        # --- TTNN EP MoE (MiniMax-owned module: no shared expert, host gate) ---
        tt_moe = TtMiniMaxMoE(
            mesh_device=mesh,
            dispatch_group_size=dgs,
            num_dispatch_groups=ndg,
            experts_per_chip=experts_per_chip,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_tok,
            max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=args.seq,
            emb_dim=EMB,
            hidden_dim=HID,
            num_links=2,
            topology=ttnn.Topology.Linear,
            routed_expert_weights=routed_w,
            routed_expert_activations_dtype=ttnn.bfloat8_b,
            routed_expert_weights_dtype=ttnn.bfloat4_b,
            gate_weights=gate_w,
            gate_fallback_mode=GateComputeMode.HOST_ALL,
        )
        tt_x = ttnn.from_torch(
            x,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, -1)),
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            dtype=ttnn.bfloat16,
        )
        tt_out = tt_moe(tt_x)
        ttnn.synchronize_device(mesh)
        print("[ep-moe] tt forward done", flush=True)

        tt_host = ttnn.to_torch(tt_out, mesh_composer=get_tp_mesh_composer(mesh), dtype=torch.bfloat16)
        passing, pcc = comp_pcc(torch_out.float(), tt_host.float(), 0.96)
        print(f"[ep-moe] final_output PCC (TtMoe vs TorchMoe) = {pcc}", flush=True)
        assert passing, f"EP MoE PCC fail: {pcc}"
        print("[ep-moe] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
