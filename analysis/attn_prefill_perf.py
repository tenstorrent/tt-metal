# SPDX-License-Identifier: Apache-2.0
"""Model-level prefill attention run for the SDPA ops CSV (revamp T2.5).

Env-driven copy of models/tt_transformers/tests/test_attention_prefill.py (hf_rope, paged attention as in the
demo's --paged_attention 1): real Llama weights via HF_MODEL, one Attention layer, one prefill of ATTN_SEQ
tokens (default 4096), ATTN_ITERS invocations (default 3). The torch reference and PCC are skipped when
ATTN_SKIP_REF=1 (device timing only). Run under `python -m tracy -r -m pytest analysis/attn_prefill_perf.py -s`
to get ops_perf_results*.csv with the ATTRIBUTES column of the production SDPA call.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, precompute_freqs
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import get_rot_mats_hf


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attn_prefill_perf(mesh_device, reset_seeds):
    max_seq_len = int(os.environ.get("ATTN_SEQ", "4096"))
    iters = int(os.environ.get("ATTN_ITERS", "3"))
    skip_ref = os.environ.get("ATTN_SKIP_REF", "1") == "1"
    paged = os.environ.get("ATTN_PAGED", "1") == "1"
    dtype = ttnn.bfloat8_b
    batch_size = 1
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True, use_hf_rope=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    rot_mats = get_rot_mats_hf(head_dim=model_args.head_dim, device=mesh_device, seq_len=max_seq_len,
                               theta=model_args.rope_theta, rope_scaling=model_args.rope_scaling)
    page_table_tt = None
    paged_attention_config = None
    if paged:
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=max(1024, max_seq_len // 32))
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size)
        page_table_tt = ttnn.from_torch(page_table, device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    tt_ccl = TT_CCL(mesh_device)
    tt_model = Attention(mesh_device, tt_ccl, model_args, state_dict, weight_cache_path=model_args.weight_cache_path(dtype),
                         layer_num=0, dtype=dtype, transformation_mats={}, configuration=model_args,
                         paged_attention_config=paged_attention_config, prefetcher=None)
    print(f"\n[attn_prefill_perf] model={model_args.model_name} S={max_seq_len} paged={paged} iters={iters} "
          f"sdpa_prog={model_args.get_attn_sdpa_prefill_program_config(max_seq_len)} ck={model_args.compute_kernel_config_sdpa}", flush=True)
    reference_model = None if skip_ref else model_args.reference_attention(load_checkpoint=True)
    for it in range(iters):
        pt_attention_input = (torch.rand(batch_size, max_seq_len, model_args.dim, dtype=torch.bfloat16) * 2) - 1
        attention_input = model_args.prepare_residual_tensor_prefill(pt_attention_input.clone(), force_replicated=True)
        tt_out = tt_model(attention_input, current_pos=None, rot_mats=rot_mats, user_id=0, mode=Mode.PREFILL, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)
        if not skip_ref and it == 0:
            tt_out_t = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape))
            tt_output_torch = tt_out_t[:, 0:1, :, : model_args.dim].view(batch_size, max_seq_len, -1)
            positions = torch.LongTensor(range(max_seq_len))
            cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2, model_args.rope_theta,
                                        model_args.rope_scaling.factor if model_args.rope_scaling else None,
                                        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
                                        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3")
            freqs_cis_i = torch.complex(cos, sin)[positions]
            attn_mask = torch.triu(torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min), diagonal=1)
            ref = reference_model(pt_attention_input.to(get_ref_model_dype(reference_model, model_args.model_name)), positions[0], freqs_cis_i, mask=attn_mask)
            passing, pcc_message = comp_pcc(ref, tt_output_torch, 0.99)
            logger.info(f"PCC: {pcc_message} passing={passing}")
        tt_out.deallocate()
        print(f"[attn_prefill_perf] iter {it} done", flush=True)
