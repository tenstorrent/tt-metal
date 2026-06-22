# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end VLM PCC for Mistral-Small-4 (vision -> projector -> scatter -> text -> logits).

Composes the individually-verified TT stages on a golden image+text input (reduced 2-layer text
core for a tractable reference): TtMistralVisionTower -> TtMistral4Projector -> host masked_scatter
of image embeds into the text embeddings at image-token positions -> TtMistral4TextModel -> logits.
Verified in two PCCs vs the HF Mistral3ForConditionalGeneration golden: the merged embeddings
(vision+projector+scatter) and the final logits (text). Stage hand-offs go via host here (data
movement only; a fused on-device pipeline is the serving/perf follow-up). The golden is built and
cached by get_cached_vlm_golden (self-contained).
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_vlm_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4Projector, TtMistral4TextModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import convert_vision_hf_to_meta, load_hf_state_dict_filtered
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.mistral_24b.mistral_vision_tower import MistralVisionTower

N_LAYERS = 2


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_vlm(mesh_device, reset_seeds):
    pcc_required = 0.98
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt)
    g = get_cached_vlm_golden(ckpt, n_layers=N_LAYERS)
    ids, px, H, W = g["ids"], g["px"], g["H"], g["W"]
    B, S = ids.shape[0], ids.shape[1]
    itok = cfg.image_token_index

    def _repl(t, shape=None):
        return ttnn.from_torch(
            (t if shape is None else t.view(shape)).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # --- stage 1: TT vision tower (weights via convert_vision_hf_to_meta, like test_m4_vision_tower) ---
    args = ModelArgs(mesh_device)
    vsd = load_hf_state_dict_filtered(ckpt, ["vision_tower."])
    tt_vsd = convert_vision_hf_to_meta(dict(vsd), args.head_dim)
    tt_ccl = TT_CCL(mesh_device)
    vm = MistralVisionTower(mesh_device, tt_ccl, tt_vsd, "vision_tower.", ttnn.bfloat16, args)
    v_out = vm(px, image_sizes=[(H, W)])
    vh = cfg.vision_config.hidden_size
    feats = ttnn.to_torch(v_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[..., :vh].reshape(-1, vh)

    # --- stage 2: TT projector ---
    psd = {
        k[len("multi_modal_projector.") :]: v
        for k, v in load_hf_state_dict_filtered(ckpt, ["multi_modal_projector."]).items()
    }
    proj = TtMistral4Projector(mesh_device, psd, cfg)
    img_embeds = ttnn.to_torch(
        proj(_repl(feats), torch.tensor([[H, W]])), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[: g["n_img"]].float()

    # --- stage 3: host masked_scatter image embeds into text embeddings ---
    tsd = load_m4_weights(ckpt, N_LAYERS)
    emb = tsd["model.embed_tokens.weight"][ids].float().clone()  # [B,S,hidden]
    emb[ids == itok] = img_embeds.to(emb.dtype)
    p_m, m_m = comp_pcc(g["merged"], emb, pcc_required)
    logger.info(f"VLM merged-embeds (vision+projector+scatter) PCC: {m_m}")

    # --- stage 4: TT text core on the merged embeddings (feed golden rope) ---
    tt_model = TtMistral4TextModel(
        mesh_device,
        tsd,
        cfg.text_config,
        N_LAYERS,
        cfg.text_config.rms_norm_eps,
        shard_experts=True,
        expert_dtype=ttnn.bfloat8_b,
    )
    rope = cfg.text_config.qk_rope_head_dim
    logits = tt_model(_repl(emb), _repl(g["cos"], (B, 1, S, rope)), _repl(g["sin"], (B, 1, S, rope)))
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]
    p_l, m_l = comp_pcc(g["logits"], lt, pcc_required)
    logger.info(f"VLM end-to-end logits PCC: {m_l}")

    assert p_m and p_l, "VLM e2e PCC below threshold"
