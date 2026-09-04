# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the WHOLE model at target SP x TP vs a composed torch reference.

Embedding -> N decoder layers -> final norm -> lm_head, sequence sharded across the SP rows, the
residual stream SP-sharded through every layer. This catches the two classes of bug a single-layer
test cannot:

  * **per-layer weight slicing** — layer ``i`` loading layer ``j``'s weights. Each layer gets
    distinct random weights, so a mis-sliced layer changes the output.
  * **KV slot packing across layers** — ``slot = user*num_layers + layer``. Every layer writes its
    own K/V rows; a layer-indexing bug makes one layer read another's cache.

Runs at ``num_layers=2`` and ``num_layers=4`` at REAL dims rather than all 32: the failure modes above
are all visible at 2+ layers, and a 32-layer random-weight build costs ~4.5 GB of host weights per
run for no extra coverage. The full 32-layer path is exercised with real weights at P1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention import allocate_kv_cache
from models.demos.llama3_1_8b_d_p.tt.model import Model

from ..test_factory import (
    concat_sp,
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
)
from .test_kv_cache_write_vs_ref import gather_kv_cache

PCC = 0.99


def model_state_dict(ref_model, head_dim):
    """Reference whole-model weights in the ``model.*`` / ``lm_head.*`` layout ``Model`` expects."""
    sd = {"model.embed_tokens.weight": ref_model.embed_tokens.weight.data}
    for i, layer in enumerate(ref_model.layers):
        a = layer.self_attn
        attn = hf_to_meta_qk(
            {
                "q_proj.weight": a.q_proj.weight.data,
                "k_proj.weight": a.k_proj.weight.data,
                "v_proj.weight": a.v_proj.weight.data,
                "o_proj.weight": a.o_proj.weight.data,
            },
            head_dim,
        )
        for k, v in attn.items():
            sd[f"model.layers.{i}.self_attn.{k}"] = v
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight.data
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = layer.mlp.up_proj.weight.data
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = layer.mlp.down_proj.weight.data
        sd[f"model.layers.{i}.input_layernorm.weight"] = layer.input_layernorm.weight.data
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.data
    sd["model.norm.weight"] = ref_model.norm.weight.data
    sd["lm_head.weight"] = ref_model.lm_head.weight.data
    return sd


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("num_layers", [2, 4], ids=["L2", "L4"])
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])
def test_model_sp_vs_ref(mesh_device, device_params, num_layers, seq_len, reset_seeds):
    cfg_full = llama_config()
    hd = cfg_full.head_dim
    # Real dims everywhere except the layer count and a reduced vocab (a 128256-row random embedding
    # table is 1 GiB of host memory per run and adds nothing to what this test checks — the lm_head's
    # real vocab layout is pinned by test_lm_head_vs_ref).
    cfg = cfg_full.reduced(num_hidden_layers=num_layers, intermediate_size=cfg_full.intermediate_size, vocab_size=2048)
    mesh_config = make_mesh_config(mesh_device)
    n_kv_local = cfg.num_key_value_heads // mesh_config.tp
    rows, cols = tuple(mesh_device.shape)

    torch.manual_seed(0)
    ref_model = ref.LlamaModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with torch.no_grad():
        ref_logits, ref_kvs, ref_hidden = ref_model(ids, return_hidden_states=True)

    model = Model(
        mesh_device,
        cfg,
        model_state_dict(ref_model, hd),
        ccl_manager=make_ccl(mesh_device),
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        attn_weight_dtype=ttnn.bfloat16,
        mlp_weight_dtype=ttnn.bfloat16,
        sequence_parallel=True,
    )

    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=seq_len, chunk_size=seq_len, sp_axis=mesh_config.sp_axis
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=seq_len,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tokens_tt = ttnn.from_torch(
        ids.reshape(1, 1, seq_len).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
    )

    out_tt = model.prefill_forward(
        tokens_tt, rope_mats, kv_cache=kv_cache, user_id=0, cached_len=0, indexed_rope=True, skip_lm_head=True
    )
    ttnn.synchronize_device(mesh_device)

    # 1) Residual stream after the last layer.
    out = concat_sp(mesh_device, out_tt, mesh_config).reshape(1, seq_len, cfg.hidden_size)
    ok, pcc = comp_pcc(ref_hidden[-1], out, PCC)
    logger.info(f"model L{num_layers} s={seq_len} final hidden: {pcc}")
    assert ok, f"final hidden PCC fail: {pcc}"

    # 2) Per-layer KV — the real product of a prefill run, and where slot packing shows up.
    for i in range(num_layers):
        ref_k = ref.hf_to_meta_head_perm(ref_kvs[i][0], hd)
        ref_v = ref_kvs[i][1]
        host_k = gather_kv_cache(mesh_device, kv_cache.k, n_kv_local, slot_row=i)
        host_v = gather_kv_cache(mesh_device, kv_cache.v, n_kv_local, slot_row=i)
        ok_k, pcc_k = comp_pcc(ref_k, host_k, PCC)
        ok_v, pcc_v = comp_pcc(ref_v, host_v, PCC)
        logger.info(f"model L{num_layers} layer {i} KV: K={pcc_k} V={pcc_v}")
        assert ok_k, f"layer {i} K PCC fail: {pcc_k}"
        assert ok_v, f"layer {i} V PCC fail: {pcc_v}"
