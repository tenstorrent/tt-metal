# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI A/B only, not for merge: matmuls that take the block pack through pack_block at SyncHalf with a math-bound loop,
at Blackhole model configurations with random weights, N_CALLS calls each under the device profiler."""

import pytest
import torch
import ttnn

N_CALLS = 7


# ---- tt_transformers Llama 3.1 8B prefill at 128 tokens on a P150, the model's own weights tensors, memory and program
# configs (attention.py:1069-1077 QKV, 1309-1316 WO; performance precision), HF_MODEL=meta-llama/Llama-3.1-8B-Instruct ----
@pytest.mark.parametrize("mesh_device", [len(ttnn.get_device_ids())], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_ra_llama8b_prefill128_attn(mesh_device):
    from models.tt_transformers.tt.attention import Attention
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import Mode
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import RotarySetup

    args = ModelArgs(mesh_device, dummy_weights=True, max_batch_size=1, max_seq_len=256, cache_hf=True)
    args.n_layers = 1
    sd = args.load_state_dict()
    rope = RotarySetup(mesh_device, 1, args.head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling, args.use_qk_fused)
    att = Attention(mesh_device, TT_CCL(mesh_device), args, sd, weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
                    layer_num=0, dtype=ttnn.bfloat8_b, transformation_mats=rope.get_both_trans_mats(), configuration=args)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(torch.randn(1, 1, 128, args.dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    for _ in range(N_CALLS):
        y = ttnn.linear(x, att.wqkv, dtype=att.activation_dtype or ttnn.bfloat16,
                        memory_config=args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                        compute_kernel_config=att.li_qkv_prefill_compute_kernel_cfg,
                        program_config=args.get_attn_qkv_program_config(Mode.PREFILL, 128, None))
        ttnn.deallocate(y)
    a = ttnn.from_torch(torch.randn(1, 1, 128, args.n_heads * args.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    for _ in range(N_CALLS):
        y = ttnn.linear(a, att.wo, compute_kernel_config=att.li_o_prefill_compute_kernel_cfg,
                        dtype=att.activation_dtype or ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=args.get_attn_wo_program_config(Mode.PREFILL, 128, None))
        ttnn.deallocate(y)


# ---- ViT-base on Blackhole, the model's attention and feed-forward with random weights at batch 10
# (vit/blackhole/tt/ttnn_optimized_sharded_vit_bh.py: QKV, Q K^T, P V, self output, FF1, FF2) ----
def _vit(device):
    from types import SimpleNamespace as NS
    import transformers
    from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
    from models.demos.vision.classification.vit.blackhole.tt import ttnn_optimized_sharded_vit_bh as vit

    torch.manual_seed(0)
    config = vit.update_model_config(transformers.ViTConfig(), 10)
    lin = lambda o, i: NS(weight=ttnn.to_device(preprocess_linear_weight(torch.randn(o, i), dtype=ttnn.bfloat8_b), device),
                          bias=ttnn.to_device(preprocess_linear_bias(torch.randn(o), dtype=ttnn.bfloat8_b), device))
    h = ttnn.from_torch(torch.rand(10, 224, config.hidden_size) * 2 - 1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    mc = ttnn.create_sharded_memory_config(h.shape, core_grid=config.core_grid, strategy=ttnn.ShardStrategy.BLOCK,
                                           orientation=ttnn.ShardOrientation.ROW_MAJOR)
    return vit, config, lin, NS, h, mc


def test_ra_vit_attention(device):
    vit, config, lin, NS, h, mc = _vit(device)
    p = NS(attention=NS(query_key_value=lin(3 * config.hidden_size, config.hidden_size)),
           output=NS(dense=lin(config.hidden_size, config.hidden_size)))
    for _ in range(N_CALLS):
        x = ttnn.to_memory_config(h, memory_config=mc, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(vit.vit_attention(config, x, parameters=p))


def test_ra_vit_feedforward(device):
    vit, config, lin, NS, h, mc = _vit(device)
    p = NS(intermediate=NS(dense=lin(config.intermediate_size, config.hidden_size)),
           output=NS(dense=lin(config.hidden_size, config.intermediate_size)))
    for _ in range(N_CALLS):
        x = ttnn.to_memory_config(h, memory_config=mc, dtype=ttnn.bfloat8_b)
        r = ttnn.to_memory_config(h, memory_config=mc, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(vit.vit_feedforward(config, x, r, parameters=p))
