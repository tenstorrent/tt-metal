#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Single DIT forward pass — no tracing, for perf tuning.

Loads the text encoder to produce real caption features, then runs the DIT
forward pass repeatedly, printing per-iteration latency.

Usage (from z_image_turbo directory):
    python dit_single_pass.py
    python dit_single_pass.py --runs 10
    python dit_single_pass.py --prompt "a robot painting"
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from dit.model_ttnn import ZImageTransformerTTNN
from text_encoder.model_ttnn import TextEncoderTTNN
from transformers import AutoTokenizer

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
REF_OUTPUT_PATH = os.path.join(HERE, "reference_output.pt")

# ── Performance optimizations (applied before model init) ─────────────────────

# Matmul blocking configs for 13x10 core grid (not in default registry).
_GRID_13x10_CONFIGS = {
    # Default 8x8x8 underutilizes 130 cores (~50-75 work items).
    # Use (4, K, 4) to create 180-270 work items for better utilization.
    # Joint blocks (30 layers, M=1152, 36 M-tiles): dominant cost
    (1152, 3840, 3072): (4, 8, 4),  # QKV fused, 9*24=216 items
    (1152, 1024, 3840): (4, 4, 4),  # to_out, 9*30=270 items
    (1152, 3840, 2560): (4, 8, 4),  # MLP w1/w3, 9*20=180 items
    (1152, 2560, 3840): (4, 8, 4),  # MLP w2, 9*30=270 items
    # Noise refiner (2 layers, M=1024, 32 M-tiles)
    (1024, 3840, 3072): (4, 8, 4),  # QKV fused, 8*24=192 items
    (1024, 1024, 3840): (4, 4, 4),  # to_out, 8*30=240 items
    (1024, 3840, 2560): (4, 8, 4),  # MLP w1/w3, 8*20=160 items
    (1024, 2560, 3840): (4, 8, 4),  # MLP w2, 8*30=240 items
    # Context refiner (2 layers, M=128, 4 M-tiles)
    (128, 3840, 3072): (2, 8, 4),  # QKV fused, 2*24=48 items
    (128, 1024, 3840): (2, 4, 4),  # to_out, 2*30=60 items
    (128, 3840, 2560): (2, 8, 4),  # MLP w1/w3, 2*20=40 items
    (128, 2560, 3840): (2, 8, 4),  # MLP w2, 2*30=60 items
}


def _apply_matmul_config_patch():
    """Monkey-patch get_matmul_config in dit.model_ttnn to handle 13x10 grid."""
    import dit.model_ttnn as dit_mod

    _orig = dit_mod._get_matmul_config

    def _patched(M, K, N, core_grid, default_block_size=None):
        gx = getattr(core_grid, "x", None)
        gy = getattr(core_grid, "y", None)
        if (gx, gy) == (13, 10):
            cfg = _GRID_13x10_CONFIGS.get((M, K, N))
            if cfg is not None:
                sb_h, sb_w = 2, 2
                if len(cfg) == 4:
                    sb_h, sb_w = cfg[3]
                    cfg = cfg[:3]
                return ttnn.MinimalMatmulConfig(
                    M_block_size=cfg[0],
                    K_block_size=cfg[1],
                    N_block_size=cfg[2],
                    subblock_h=sb_h,
                    subblock_w=sb_w,
                    compute_with_storage_grid_size=core_grid,
                )
        return _orig(M, K, N, core_grid, default_block_size)

    dit_mod._get_matmul_config = _patched


def _apply_compute_config_patch():
    """Use HiFi2 + packer_l1_acc for matmuls (faster), keep HiFi4 for norms."""
    import dit.model_ttnn as dit_mod

    FAST_MATMUL_KERNEL = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    _orig_mm = dit_mod.ZImageTransformerTTNN._mm

    def _fast_mm(self, x, weight, M, K, N, dtype=ttnn.DataType.BFLOAT16):
        config = dit_mod._get_matmul_config(M, K, N, self._core_grid)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            config=config,
            compute_kernel_config=FAST_MATMUL_KERNEL,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    dit_mod.ZImageTransformerTTNN._mm = _fast_mm


def _apply_fast_activations_patch():
    """HiFi2 + approx + packer_l1_acc for REDUCE_KERNEL (norms, activations)."""
    import dit.model_ttnn as dit_mod

    dit_mod.REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _apply_sdpa_patch():
    """Add compute_kernel_config to SDPA for faster attention."""
    import dit.model_ttnn as dit_mod

    SDPA_KERNEL = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    _orig_attn = dit_mod.ZImageTransformerTTNN._attention

    def _fast_attention(self, x, seq_len, block_prefix, is_caption=False):
        import dit.model_ttnn as dm

        x_2d = ttnn.reshape(x, [seq_len, dm.HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        N = dm.HEADS_PER_DEV * dm.HEAD_DIM

        fused_qkv = self.weights[f"{block_prefix}.attention.qkv_fused_mmT"]
        config = dm._get_matmul_config(seq_len, dm.HIDDEN_DIM, 3 * N, self._core_grid)
        q_2d, k_2d, v_2d = ttnn.experimental.minimal_matmul_split(
            x_2d,
            fused_qkv,
            chunks=3,
            dim=-1,
            config=config,
            compute_kernel_config=dm.REDUCE_KERNEL,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_2d, False)

        q = ttnn.reshape(q_2d, [1, seq_len, dm.HEADS_PER_DEV, dm.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_2d, False)
        old_q = q
        q = self._qk_norm(old_q, self.weights[f"{block_prefix}.attention.norm_q.weight"], seq_len, dm.HEADS_PER_DEV)
        ttnn.deallocate(old_q, False)
        old_q = q
        q = self._apply_rope(old_q, seq_len, dm.HEADS_PER_DEV, is_caption=is_caption)
        ttnn.deallocate(old_q, False)

        k = ttnn.reshape(k_2d, [1, seq_len, dm.HEADS_PER_DEV, dm.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_2d, False)
        old_k = k
        k = self._qk_norm(old_k, self.weights[f"{block_prefix}.attention.norm_k.weight"], seq_len, dm.HEADS_PER_DEV)
        ttnn.deallocate(old_k, False)
        old_k = k
        k = self._apply_rope(old_k, seq_len, dm.HEADS_PER_DEV, is_caption=is_caption)
        ttnn.deallocate(old_k, False)

        v_4d = ttnn.reshape(v_2d, [1, 1, seq_len, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_2d, False)
        old_v_4d = v_4d
        v_4d = self._ensure_tile(old_v_4d)
        if v_4d is not old_v_4d:
            ttnn.deallocate(old_v_4d, False)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_4d,
            num_heads=dm.HEADS_PER_DEV,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_4d, False)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=dm.ATTN_SCALE,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=SDPA_KERNEL,
        )
        ttnn.deallocate(q, False)
        ttnn.deallocate(k, False)
        ttnn.deallocate(v, False)
        old_attn = attn_out
        attn_out = ttnn.transformer.concatenate_heads(old_attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(old_attn, False)
        old_attn = attn_out
        attn_out = ttnn.reshape(old_attn, [seq_len, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(old_attn, False)

        out_wT = self.weights[f"{block_prefix}.attention.to_out.0.weight_mmT"]
        old_attn = attn_out
        attn_out = self._mm(old_attn, out_wT, seq_len, N, dm.HIDDEN_DIM)
        ttnn.deallocate(old_attn, False)
        return self._all_reduce(attn_out, seq_len)

    dit_mod.ZImageTransformerTTNN._attention = _fast_attention


def _convert_mlp_weights_to_bfp8(dit):
    """Convert MLP matmul weights from BF16 to BFLOAT8_B to halve DRAM bandwidth."""
    converted = 0
    for key in list(dit.weights.keys()):
        is_mlp_mm = "feed_forward" in key and key.endswith("_mmT")
        if is_mlp_mm:
            w = dit.weights[key]
            if w.dtype == ttnn.DataType.BFLOAT16:
                new_w = ttnn.typecast(w, ttnn.DataType.BFLOAT8_B, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(w, force=True)
                dit.weights[key] = new_w
                converted += 1
    print(f"  Converted {converted} MLP weights to BFLOAT8_B")


def compute_pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def compute_metrics(a, b):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    rel = diff / (b_f.abs() + 1e-10)
    return {
        "pcc": compute_pcc(a, b),
        "max_abs_diff": diff.max().item(),
        "max_rel_diff": rel.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def open_mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 4)),
        l1_small_size=1 << 15,
        trace_region_size=70_000_000,
    )
    device.enable_program_cache()
    return device


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def encode_prompt(te, tokenizer, mesh_device, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, padding="max_length", truncation=True, max_length=CAP_TOKENS, return_tensors="pt")[
        "input_ids"
    ]
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = te(tt_ids)
    cap_cpu = _tt_to_torch(tt_out, mesh_device)[:CAP_TOKENS].bfloat16()
    ttnn.deallocate(tt_ids, force=True)
    ttnn.deallocate(tt_out, force=True)
    return cap_cpu.unsqueeze(0)  # [1, CAP_TOKENS, 2560]


def main():
    parser = argparse.ArgumentParser(description="DIT single-pass perf benchmark (no trace)")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed iterations")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over the ocean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Opening mesh device ...")
    mesh_device = open_mesh_device()

    print("Loading text encoder ...")
    te = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    print("Applying perf patches ...")
    _apply_matmul_config_patch()
    _apply_compute_config_patch()
    _apply_fast_activations_patch()
    _apply_sdpa_patch()

    print("Loading DIT ...")
    dit = ZImageTransformerTTNN(mesh_device)

    print("Converting MLP weights to BFP8 ...")
    _convert_mlp_weights_to_bfp8(dit)

    # Encode a real prompt for caption features
    print("Encoding prompt ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    cap_cpu = encode_prompt(te, tokenizer, mesh_device, args.prompt)
    dit.set_cap_feats(cap_cpu)

    # Prepare latent + timestep inputs
    torch.manual_seed(args.seed)
    latent_pt = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
    lat_pt = latent_pt.squeeze(0).unsqueeze(1).bfloat16()  # [16, 1, 64, 64]
    t_norm = 0.5
    ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)  # [1]

    # ── Compile run ───────────────────────────────────────────────────────────
    print("\n[1/3] Compile run ...")
    tt_lat = _to_device_bf16(lat_pt, mesh_device)
    tt_ts = _to_device_bf16(ts_pt, mesh_device)
    t0 = time.time()
    out = dit([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    print(f"  Compile: {(time.time() - t0) * 1000:.0f} ms")
    for t in out:
        ttnn.deallocate(t, force=True)
    ttnn.deallocate(tt_lat, force=True)
    ttnn.deallocate(tt_ts, force=True)

    # ── Warm run (programs cached) ────────────────────────────────────────────
    print("[2/4] Warm run ...")
    tt_lat = _to_device_bf16(lat_pt, mesh_device)
    tt_ts = _to_device_bf16(ts_pt, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out = dit([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  Warm: {warm_ms:.1f} ms")
    for t in out:
        ttnn.deallocate(t, force=True)
    ttnn.deallocate(tt_lat, force=True)
    ttnn.deallocate(tt_ts, force=True)

    # ── Trace capture ─────────────────────────────────────────────────────────
    print("[3/4] Capturing metal trace ...")
    lat_buf = _to_device_bf16(lat_pt, mesh_device)
    ts_buf = _to_device_bf16(ts_pt, mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = dit._forward_impl([lat_buf], ts_buf)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    dit_output_ref = trace_out[0]
    print("  Trace captured.")

    # ── Timed runs (trace replay) ─────────────────────────────────────────────
    print(f"[4/4] Running {args.runs} timed iterations (traced) ...")
    timings = []
    for i in range(args.runs):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        timings.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")

    # ── PCC check (read from persistent trace output) ─────────────────────────
    out_torch = _tt_to_torch(dit_output_ref, mesh_device)

    if not os.path.exists(REF_OUTPUT_PATH):
        torch.save(out_torch, REF_OUTPUT_PATH)
        print(f"\nSaved reference output to {REF_OUTPUT_PATH}")
        print("PCC: 1.0000 (reference)")
    else:
        ref = torch.load(REF_OUTPUT_PATH, weights_only=True)
        m = compute_metrics(out_torch, ref)
        print(f"\nPCC: {m['pcc']:.6f}")
        print(f"Max abs diff: {m['max_abs_diff']:.6f}")
        print(f"Max rel diff: {m['max_rel_diff']:.6f}")
        print(f"Mean abs diff: {m['mean_abs_diff']:.6f}")

    # Summary
    avg = sum(timings) / len(timings)
    best = min(timings)
    worst = max(timings)
    print(f"\nSummary ({args.runs} runs):")
    print(f"  Warm:  {warm_ms:.1f} ms")
    print(f"  Avg:   {avg:.1f} ms")
    print(f"  Best:  {best:.1f} ms")
    print(f"  Worst: {worst:.1f} ms")

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
