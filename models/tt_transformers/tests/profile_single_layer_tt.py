# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Profile a single tt_transformers TransformerBlock (prefill + decode).

Mirrors models/demos/qwen3_tts/tests/profile_single_layer.py so the resulting
tracy CSVs can be diffed sub-block-for-sub-block. Intended for any HF model
loaded via the standard HF_MODEL env var.

Usage with tracy (one process, prefill BEFORE decode so the
first-SdpaDecode-op heuristic separates phases in post-processing):

    HF_MODEL=Qwen/Qwen3-1.7B \
    python -m tracy -p -v -r --op-support-count 2600 --dump-device-data-mid-run \
      models/tt_transformers/tests/profile_single_layer_tt.py --prefill-seq-len 128
"""

import argparse
import os
import time

import torch

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, get_rot_transformation_mat
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup, get_rot_mats, get_rot_mats_hf


def build_model_args(mesh_device, max_seq_len: int, batch_size: int = 1):
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        cache_hf=True,
        prefetcher=None,
        use_hf_rope=False,
    )
    model_args.n_layers = 1
    return model_args


def make_page_table(model_args, mesh_device, page_block_size: int = 32, page_max_num_blocks: int = 1024):
    cfg = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=page_max_num_blocks)
    permutation = torch.randperm(cfg.max_num_blocks)
    reverse = torch.argsort(permutation)
    page_table = reverse.reshape(model_args.max_batch_size, cfg.max_num_blocks // model_args.max_batch_size)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    return cfg, page_table_tt


def profile_prefill(mesh_device, model_args, state_dict, seq_len: int, warmup: int, measure: int):
    print(f"\n=== Profiling Prefill (seq_len={seq_len}, warmup={warmup}, measure={measure}) ===")

    rot_mats_fn = get_rot_mats_hf if model_args.use_hf_rope else get_rot_mats
    rot_mats = rot_mats_fn(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    rot_mats_local = None
    if model_args.rope_theta_local is not None:
        rot_mats_local = get_rot_mats(
            head_dim=model_args.head_dim,
            device=mesh_device,
            seq_len=seq_len,
            theta=model_args.rope_theta_local,
            rope_scaling=None,
        )

    # Build both prefill and decode transformation mats so the same TransformerBlock
    # can run either mode without rebuilding.
    DefaultRopeSetup = HfRotarySetup if model_args.use_hf_rope else RotarySetup
    rope_setup = DefaultRopeSetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
        prefetcher=None,
    )
    transformation_mats = rope_setup.get_both_trans_mats()
    if not model_args.use_hf_rope and "prefill" not in transformation_mats:
        trans_t = get_rot_transformation_mat(model_args.head_dim)
        transformation_mats["prefill"] = ttnn.as_tensor(
            trans_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    paged_cfg, page_table_tt = make_page_table(model_args, mesh_device)
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TransformerBlock(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        transformation_mats=transformation_mats,
        args=model_args,
        paged_attention_config=paged_cfg,
        prefetcher=None,
    )

    pt_in = (torch.rand(1, seq_len, model_args.dim, dtype=torch.bfloat16) * 2) - 1
    decode_in = model_args.prepare_residual_tensor_prefill(pt_in)

    def run():
        out = tt_model(
            decode_in,
            None,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            user_id=0,
            mode=Mode.PREFILL,
            page_table=page_table_tt,
        )
        ttnn.deallocate(out)

    print(f"  Warmup ({warmup} iter)…")
    for _ in range(warmup):
        run()
    ttnn.synchronize_device(mesh_device)

    print(f"  Measuring {measure} iter…")
    t0 = time.perf_counter()
    for _ in range(measure):
        run()
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    print(f"  Total {dt*1000:.2f} ms; per iter {dt*1000/measure:.2f} ms")
    return tt_model, paged_cfg, page_table_tt


def profile_decode(
    mesh_device, model_args, tt_model, paged_cfg, page_table_tt, cur_pos: int, warmup: int, measure: int
):
    print(f"\n=== Profiling Decode (cur_pos={cur_pos}, warmup={warmup}, measure={measure}) ===")

    DefaultRopeSetup = HfRotarySetup if model_args.use_hf_rope else RotarySetup
    rope_setup = DefaultRopeSetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
        prefetcher=None,
    )
    rope_setup_local = None
    if model_args.rope_theta_local is not None:
        rope_setup_local = RotarySetup(
            mesh_device,
            model_args.max_batch_size,
            model_args.head_dim,
            model_args.max_seq_len,
            model_args.rope_theta_local,
            None,
        )

    cur_pos_t = torch.tensor([cur_pos] * model_args.max_batch_size)
    rot_mats = rope_setup.get_rot_mats(cur_pos_t)
    rot_mats_local = None if rope_setup_local is None else rope_setup_local.get_rot_mats(cur_pos_t)

    cur_pos_tensor = ttnn.from_torch(
        cur_pos_t,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    pt_in = (torch.rand(model_args.max_batch_size, 1, model_args.dim, dtype=torch.bfloat16) * 2) - 1
    decode_in = model_args.prepare_residual_tensor_decode(
        pt_in,
        model_args.get_residual_mem_config(Mode.DECODE, None),
    )

    def run():
        out = tt_model(
            decode_in,
            cur_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=Mode.DECODE,
            page_table=page_table_tt,
        )
        ttnn.deallocate(out)

    print(f"  Warmup ({warmup} iter)…")
    for _ in range(warmup):
        run()
    ttnn.synchronize_device(mesh_device)

    print(f"  Measuring {measure} iter…")
    t0 = time.perf_counter()
    for _ in range(measure):
        run()
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    print(f"  Total {dt*1000:.2f} ms; per iter {dt*1000/measure:.2f} ms")


def main():
    p = argparse.ArgumentParser(description="tt_transformers single-layer prefill+decode profiling")
    p.add_argument("--prefill-seq-len", type=int, default=128)
    p.add_argument("--decode-cur-pos", type=int, default=100)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--measure", type=int, default=1)
    p.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    args = p.parse_args()

    print("=" * 70)
    print(f"tt_transformers single-layer profiling [HF_MODEL={os.getenv('HF_MODEL')}]")
    print("=" * 70)

    # Mesh device matches the demo's batch-1 P150 path
    mesh_shape = (1, 1)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    try:
        model_args = build_model_args(mesh_device, max_seq_len=max(args.prefill_seq_len, args.decode_cur_pos + 64))
        state_dict = model_args.load_state_dict()
        print(
            f"Model: {model_args.model_name}  dim={model_args.dim}  "
            f"heads={model_args.n_heads}/{model_args.n_kv_heads}  "
            f"head_dim={model_args.head_dim}  intermediate={getattr(model_args, 'hidden_dim', '?')}  "
            f"n_layers={model_args.n_layers}"
        )

        tt_model = paged_cfg = page_table_tt = None
        if args.mode in ("prefill", "both"):
            tt_model, paged_cfg, page_table_tt = profile_prefill(
                mesh_device, model_args, state_dict, args.prefill_seq_len, args.warmup, args.measure
            )
        if args.mode in ("decode", "both"):
            if tt_model is None:
                # Decode-only run still needs a model; build it but skip prefill calls.
                tt_model, paged_cfg, page_table_tt = profile_prefill(
                    mesh_device, model_args, state_dict, args.prefill_seq_len, 0, 0
                )
            profile_decode(
                mesh_device,
                model_args,
                tt_model,
                paged_cfg,
                page_table_tt,
                args.decode_cur_pos,
                args.warmup,
                args.measure,
            )
        print("\nDone.")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
