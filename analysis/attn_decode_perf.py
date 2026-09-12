# SPDX-License-Identifier: Apache-2.0
"""Model-level decode attention run for the SDPA decode ops CSV (revamp T2.5).

Env-driven copy of models/tt_transformers/tests/test_attention.py (decode, hf_rope, paged attention as in the
demo): real Llama weights via HF_MODEL, one Attention layer, batch ATTN_BATCH (default 32), one decode step at
each cache position in ATTN_POS (default "128,1024,4096"), ATTN_ITERS invocations per position (default 3).
The KV cache is not filled (timing does not depend on cache content). No torch reference.
Run under `python -m tracy -r -m pytest analysis/attn_decode_perf.py -s`.
"""
import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import HfRotarySetup


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attn_decode_perf(mesh_device, reset_seeds):
    batch_size = int(os.environ.get("ATTN_BATCH", "32"))
    positions = [int(x) for x in os.environ.get("ATTN_POS", "128,1024,4096").split(",")]
    iters = int(os.environ.get("ATTN_ITERS", "3"))
    max_seq_len = int(os.environ.get("ATTN_MAXSEQ", str(max(8192, max(positions) * 2))))
    paged = os.environ.get("ATTN_PAGED", "1") == "1"
    dtype = ttnn.bfloat8_b
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True, use_hf_rope=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    rope_setup = HfRotarySetup(mesh_device, batch_size, model_args.head_dim, model_args.max_seq_len, model_args.rope_theta,
                               model_args.rope_scaling, model_args.use_qk_fused, prefetcher=None)
    transformation_mats = rope_setup.get_both_trans_mats()
    page_table_tt = None
    paged_attention_config = None
    if paged:
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=max(1024, batch_size * max_seq_len // 32))
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size)
        page_table_tt = ttnn.from_torch(page_table, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape))
    tt_ccl = TT_CCL(mesh_device)
    tt_model = Attention(mesh_device, tt_ccl, model_args, state_dict, weight_cache_path=model_args.weight_cache_path(dtype),
                         layer_num=0, dtype=dtype, transformation_mats=transformation_mats, configuration=model_args,
                         paged_attention_config=paged_attention_config, prefetcher=None)
    print(f"\n[attn_decode_perf] model={model_args.model_name} batch={batch_size} max_seq_len={max_seq_len} paged={paged} "
          f"positions={positions} iters={iters} sdpa_decode_prog={model_args.get_attn_sdpa_decode_program_config(None)}", flush=True)
    for pos in positions:
        current_pos = torch.tensor([pos for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(current_pos, device=mesh_device, dtype=ttnn.int32,
                                             mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape))
        for it in range(iters):
            pt_attention_input = torch.randn(batch_size, 1, model_args.dim, dtype=torch.bfloat16)
            attention_input = model_args.prepare_residual_tensor_decode(pt_attention_input, model_args.get_attn_input_mem_config(Mode.DECODE, None), force_replicated=True)
            rot_mats = rope_setup.get_rot_mats(current_pos)
            tt_out = tt_model(attention_input, current_pos_tensor, rot_mats=rot_mats, mode=Mode.DECODE, page_table=page_table_tt)
            ttnn.synchronize_device(mesh_device)
            tt_out.deallocate()
            print(f"[attn_decode_perf] pos={pos} iter {it} done", flush=True)
