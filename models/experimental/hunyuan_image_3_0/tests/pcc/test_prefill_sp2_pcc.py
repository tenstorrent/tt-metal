# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PCC gate for the SP=2 / TP=2 backbone forward — the config the L1-residency seq
# gate (parallel_utils.resid_mem_config / moe_full_seq_mem_config) was measured and
# built for. The KV-cache prefill/decode tests run sp_factor=1 (KV cache requires
# every device see the full K/V), so they never exercise the sp=2 gated path; this
# test does, via a plain use_cache=False full-sequence forward on the 2x2 mesh —
# exactly the recaption/perf execution mode — checked against an fp32 host reference.
#
# Run:
#   HY_NUM_LAYERS=32 HY_MAX_ISL=2560 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_prefill_sp2_pcc.py -s
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tests.pcc.kv_cache_pcc_common import HF_MAX_ISL, _pad_ids_to
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
MAX_ISL = min(int(os.environ.get("HY_MAX_ISL", "512")), HF_MAX_ISL)
SEQ_LENGTHS = sorted({128, MAX_ISL})
PCC_REQUIRED = 0.95 if NUM_LAYERS <= 2 else 0.85
STREAM_EXPERTS = os.environ.get("HY_STREAM_EXPERTS", "1" if NUM_LAYERS > 8 else "0") != "0"
SP_FACTOR = 2


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_prefill_sp2_pcc(mesh_device):
    mesh_device.enable_program_cache()
    c = h.model_cfg()
    H = c["H"]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"].float()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    wte_f = wte.float()

    backbone = HunyuanTtModel(
        mesh_device,
        num_layers=NUM_LAYERS,
        hidden_size=H,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV_HEADS"],
        head_dim=c["HEAD_DIM"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        use_qk_norm=c["USE_QK_NORM"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=STREAM_EXPERTS,
        layer_loader=layer_loader,
        embed_state_dict={"model.wte.weight": wte},
        norm_state_dict={"model.ln_f.weight": ln_f_w},
        apply_final_norm=True,
        weight_dtype=ttnn.bfloat8_b,
        bf16_layers=[],
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=SP_FACTOR,
    )

    ln_f = HunyuanRMSNorm(H, eps=c["EPS"])
    ln_f.weight.data = ln_f_w.float()
    attn_slices = [[]]

    # real recaption prompt ids (same as kv_cache_pcc_common), padded per ISL.
    tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    prompt_ids = prepare_recaption_inputs(
        tok, "a cat on a mat", bot_task="recaption", sequence_template="instruct"
    ).input_ids

    @torch.no_grad()
    def reference(ids):
        S = int(ids.shape[1])
        hidden = F.embedding(ids.long(), wte_f)
        mask = to_additive(build_attention_mask(S, attn_slices, bsz=1)).reshape(1, 1, S, S)
        cos, sin = build_batch_2d_rope(image_infos=None, seq_len=S, n_elem=c["HEAD_DIM"], device=hidden.device)
        for i in range(NUM_LAYERS):
            sd = h.load_prefix(f"model.layers.{i}")
            layer = RefLayer(
                hidden_size=H,
                num_attention_heads=c["HEADS"],
                num_key_value_heads=c["KV_HEADS"],
                attention_head_dim=c["HEAD_DIM"],
                num_experts=c["NUM_EXPERTS"],
                moe_topk=c["MOE_TOPK"],
                moe_intermediate_size=c["MOE_INTER"],
                num_shared_expert=c["NUM_SHARED"],
                use_mixed_mlp_moe=c["USE_MIXED"],
                norm_topk_prob=c["NORM_TOPK"],
                use_qk_norm=c["USE_QK_NORM"],
                rms_norm_eps=c["EPS"],
                layer_idx=i,
            )
            layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            layer.eval()
            hidden = layer(hidden, attention_mask=mask, custom_pos_emb=(cos, sin))
            del layer, sd
        hidden = ln_f(hidden)
        logits = lm_head_logits(hidden, lm_w)
        return hidden[:, -1, :], logits[:, -1, :]

    logger.info(f"SP=2 prefill PCC: layers={NUM_LAYERS} ISLs={SEQ_LENGTHS} pcc>={PCC_REQUIRED}")
    failing = []
    for S in SEQ_LENGTHS:
        ids = _pad_ids_to(prompt_ids, S)
        emb = F.embedding(ids.long(), wte_f)
        x = ttnn.from_torch(
            emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        m = to_additive(build_attention_mask(S, attn_slices, bsz=1), dtype=torch.bfloat16).reshape(1, 1, S, S)
        mask_tt = ttnn.from_torch(
            m,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        hidden = backbone.forward(inputs_embeds=x, seq_len=S, image_infos=None, attention_mask=mask_tt, use_cache=False)
        hid_full = ttnn.to_torch(hidden, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
        tt_hidden_last = hid_full[:, -1, :]
        tt_logits_last = lm_head_logits(hid_full, lm_w)[:, -1, :]
        ttnn.deallocate(hidden)
        ttnn.deallocate(x)
        ttnn.deallocate(mask_tt)

        ref_hidden, ref_logits = reference(ids)
        hp = h.pcc(ref_hidden, tt_hidden_last)
        lp = h.pcc(ref_logits, tt_logits_last)
        logger.info(f"  ISL={S:5d}: hidden_pcc={hp:.6f}  logits_pcc={lp:.6f} (>= {PCC_REQUIRED})")
        if hp < PCC_REQUIRED or lp < PCC_REQUIRED:
            failing.append((S, hp, lp))

    assert not failing, "SP=2 prefill PCC below threshold: " + ", ".join(
        f"ISL={s} hidden={hp:.4f} logits={lp:.4f}" for s, hp, lp in failing
    )
