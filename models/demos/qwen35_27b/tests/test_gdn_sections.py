# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-section timing breakdown within a single GDN layer.

Updated to match the current _forward_decode_fused path:
- PROJ: qkvz matmul + ab matmul + slicing
- CONV: 4-tap conv1d + silu
- PREP: unshard a/b/conv_out (scalars extracted by reader kernel)
- KERNEL: full fused kernel (L2 norm + gates + recurrence)
- POST: rms_norm + silu gate + output projection + all-reduce
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.gdn import _shard_linear, _unshard
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
def test_gdn_section_breakdown(mesh_device, reset_seeds, ensure_gc):
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
    )

    # Warmup / compile
    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    for pos_idx in range(len(prompt_tokens)):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

    logger.info("Compile done, now profiling GDN layer 0 sections...")

    # Profile layer 0 (a GDN layer)
    gdn = model.layers[0].attention
    tw = gdn.tw
    B = gdn.batch_size
    Nk_TP, Nv_TP, Dk, Dv = gdn.Nk_TP, gdn.Nv_TP, gdn.Dk, gdn.Dv
    qkv_dim_tp, qkvz_dim_tp = gdn.qkv_dim_tp, gdn.qkvz_dim_tp
    key_dim_tp = gdn.key_dim_tp
    act_shard = gdn.args.act_shard_hidden
    compute_cfg = gdn.compute_cfg
    num_pairs = B * Nv_TP
    repeat_factor = Nv_TP // Nk_TP

    num_steps = 5
    results = []

    for step in range(num_steps):
        # Prepare input
        x = ttnn.from_torch(
            torch.randn(1, 1, B, 5120, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x3d = ttnn.reshape(x, (1, B, 5120))

        # === SECTION 1: PROJECTIONS ===
        t0 = _sync(mesh_device)
        qkvz_tt = _unshard(_shard_linear(x3d, tw["qkvz"], act_shard, gdn.args.gdn_qkvz_progcfg, compute_cfg))
        qkv_tt = ttnn.slice(qkvz_tt, (0, 0, 0), (1, B, qkv_dim_tp))
        z_tt = ttnn.slice(qkvz_tt, (0, 0, qkv_dim_tp), (1, B, qkvz_dim_tp))
        ttnn.deallocate(qkvz_tt)
        ab_tt = ttnn.linear(x3d, tw["ab"])
        if len(ab_tt.shape) == 4:
            ab_tt = ttnn.reshape(ab_tt, (1, B, Nv_TP * 2))
        a_tt = ttnn.slice(ab_tt, (0, 0, 0), (1, B, Nv_TP))
        b_tt = ttnn.slice(ab_tt, (0, 0, Nv_TP), (1, B, Nv_TP * 2))
        ttnn.deallocate(ab_tt)
        t1 = _sync(mesh_device)

        # === SECTION 2: CONV1D ===
        if len(qkv_tt.shape) == 4:
            qkv_tt = ttnn.reshape(qkv_tt, (1, B, qkv_dim_tp))
        states = gdn.conv_states
        ttnn.copy(states[1], states[0])
        ttnn.copy(states[2], states[1])
        ttnn.copy(states[3], states[2])
        ttnn.copy(qkv_tt, states[3])
        conv_acc = ttnn.multiply(states[0], tw["conv_taps"][0])
        for j in range(1, gdn.conv_kernel_size):
            prod = ttnn.multiply(states[j], tw["conv_taps"][j])
            conv_acc = ttnn.add(conv_acc, prod)
            ttnn.deallocate(prod)
        conv_out = ttnn.silu(conv_acc)
        ttnn.deallocate(conv_acc)
        if len(conv_out.shape) == 4:
            conv_out = ttnn.reshape(conv_out, (1, B, qkv_dim_tp))
        t2 = _sync(mesh_device)

        # === SECTION 3: PREP (unshard only — scalars extracted by kernel reader) ===
        a_tt = _unshard(a_tt)
        b_tt = _unshard(b_tt)
        conv_out = _unshard(conv_out)
        t3 = _sync(mesh_device)

        # === SECTION 4: FUSED KERNEL (L2 norm + gates + recurrence) ===
        if gdn.fused_output is None:
            gdn.fused_output = ttnn.from_torch(
                torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        gdn_full_fused_inplace(
            conv_out,
            a_tt,
            b_tt,
            gdn.neg_exp_A,
            tw["dt_bias"],
            tw["norm_w"],
            gdn.scale_tt,
            gdn.rms_scale_tt,
            gdn.rms_eps_tt,
            gdn.rec_states,
            gdn.fused_output,
            num_pairs=num_pairs,
            num_cores=min(96, num_pairs),
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )
        t4 = _sync(mesh_device)

        ttnn.deallocate(conv_out)
        ttnn.deallocate(a_tt)
        ttnn.deallocate(b_tt)

        # === SECTION 5: POST (rms_norm + silu gate + output proj + all-reduce) ===
        out_r = ttnn.reshape(gdn.fused_output, (B, Nv_TP, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, gdn.value_dim_tp))
        ttnn.deallocate(out_n)
        z_act = ttnn.silu(z_tt)
        ttnn.deallocate(z_tt)
        out_f = _unshard(out_f)
        gated = ttnn.multiply(out_f, z_act)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z_act)
        act_shard_out = gdn.args.act_shard_gdn_value
        out_partial = _unshard(_shard_linear(gated, tw["out"], act_shard_out, gdn.args.gdn_out_progcfg, compute_cfg))
        ttnn.deallocate(gated)
        out_partial = ttnn.reshape(out_partial, (1, 1, B, out_partial.shape[-1]))
        t5 = _sync(mesh_device)

        ttnn.deallocate(out_partial)
        ttnn.deallocate(x)

        proj_ms = (t1 - t0) * 1000
        conv_ms = (t2 - t1) * 1000
        prep_ms = (t3 - t2) * 1000
        kern_ms = (t4 - t3) * 1000
        post_ms = (t5 - t4) * 1000
        total = proj_ms + conv_ms + prep_ms + kern_ms + post_ms
        results.append((proj_ms, conv_ms, prep_ms, kern_ms, post_ms, total))

    # Print results
    print("\n" + "=" * 70)
    print("GDN LAYER 0 — PER-SECTION BREAKDOWN (fused kernel path)")
    print("=" * 70)
    for i, (proj, conv, prep, kern, post, total) in enumerate(results):
        print(
            f"  Step {i}: PROJ={proj:.2f}  CONV={conv:.2f}  PREP={prep:.2f}  KERNEL={kern:.2f}  POST={post:.2f}  TOTAL={total:.2f} ms"
        )

    avg = [sum(r[i] for r in results) / len(results) for i in range(6)]
    print(f"  ---")
    print(
        f"  Avg:    PROJ={avg[0]:.2f}  CONV={avg[1]:.2f}  PREP={avg[2]:.2f}  KERNEL={avg[3]:.2f}  POST={avg[4]:.2f}  TOTAL={avg[5]:.2f} ms"
    )
    print(f"  ---")
    print(f"  PROJ  (qkvz+ab matmuls):   {avg[0]:.2f} ms  ({100*avg[0]/avg[5]:.0f}%)")
    print(f"  CONV  (4-tap conv1d+silu):  {avg[1]:.2f} ms  ({100*avg[1]/avg[5]:.0f}%)")
    print(f"  PREP  (unshard only):       {avg[2]:.2f} ms  ({100*avg[2]/avg[5]:.0f}%)")
    print(f"  KERNEL (fused recurrence):  {avg[3]:.2f} ms  ({100*avg[3]/avg[5]:.0f}%)")
    print(f"  POST  (norm+gate+out_proj): {avg[4]:.2f} ms  ({100*avg[4]/avg[5]:.0f}%)")
