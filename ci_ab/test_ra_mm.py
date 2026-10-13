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
