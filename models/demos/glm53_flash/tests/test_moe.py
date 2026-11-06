# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE FFN (router + sparse experts + shared expert) vs HF Glm5NextTextMoE, real layer weights.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_moe.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import (
    cast_like_from_pretrained,
    convert_to_hf_module_keys,
    load_config,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.moe import TtMoE

LAYER = int(os.environ.get("GLM53_MOE_LAYER", "4"))


def _hf_moe(text_config, raw_mlp_prefixed):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    sd = {k[len("mlp.") :]: v for k, v in convert_to_hf_module_keys(raw_mlp_prefixed).items()}
    with torch.device("meta"):
        hf = Glm5NextTextMoE(text_config)
    hf = hf.to_empty(device="cpu")
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    cast_like_from_pretrained(hf)
    return hf, sd


def realistic_mlp_input(snapshot, text_config, raw_layer, seq_len: int, layer: int | None = None) -> torch.Tensor:
    """The MoE input distribution: post_attention_layernorm(ffn_hc(residual streams)). A random
    input makes every expert score ~sigmoid(0) and the top-8 degenerates into tie-breaks, so the
    streams come from the host reference dump of this layer (``layer``, default the module's
    ``LAYER``) when present, else from the embedding of random tokens (layer-0-like scale)."""
    layer = LAYER if layer is None else layer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection, Glm5NextTextRMSNorm

    hc = Glm5NextTextHyperConnection(text_config)
    hc.fn.data = raw_layer["hc_ffn_fn"].float()
    hc.base.data = raw_layer["hc_ffn_base"].float()
    hc.scale.data = raw_layer["hc_ffn_scale"].float()
    norm = Glm5NextTextRMSNorm(text_config.hidden_size, text_config.rms_norm_eps)
    norm.weight.data = raw_layer["post_attention_layernorm.weight"]
    dump = None
    for d in sorted(Path("bringup/references/glm53_flash").glob("T*")):
        f = d / f"layer{layer}_prefill.pt"
        if f.exists():
            dump = torch.load(f)
            break
    if dump is not None and dump["x_out"].shape[1] >= seq_len:
        streams = dump["x_out"][:, :seq_len]
    else:
        from models.demos.glm53_flash.reference.checkpoint import load_embedding

        emb = load_embedding(snapshot)
        g = torch.Generator().manual_seed(3)
        ids = torch.randint(0, emb.shape[0], (1, seq_len), generator=g)
        streams = emb[ids].unsqueeze(2).expand(-1, -1, text_config.hc_mult, -1).contiguous()
    with torch.no_grad():
        _, _, collapsed = hc(streams)
        return norm(collapsed).to(torch.bfloat16)


@pytest.fixture(scope="module")
def weights():
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(
        snapshot,
        LAYER,
        shard_headers(snapshot),
        name_filter=lambda n: n.startswith(("mlp.", "hc_ffn", "post_attention_layernorm")),
    )
    hf, sd = _hf_moe(text, {k: v for k, v in raw.items() if k.startswith("mlp.")})
    return snapshot, text, raw, hf, sd


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 256])
def test_moe_vs_hf(mesh_device, weights, seq_len):
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    # stacked experts already converted (saves a second 14 GB stack); shared/gate from raw
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    tt = TtMoE(mesh_device, cfg, mlp)

    x = realistic_mlp_input(snapshot, text, raw, seq_len)
    T = seq_len
    with torch.no_grad():
        y_hf = hf(x)
        _, w_hf, idx_hf = hf.gate(x)
    x_tt = ttnn.from_torch(x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    # router alone
    w_tt, idx_tt, dense_tt = tt.router(x_tt)
    idx_t = ttnn.to_torch(idx_tt).reshape(T, -1).long()
    w_t = ttnn.to_torch(w_tt).reshape(T, -1).float()
    same_sets = sum(set(idx_t[t].tolist()) == set(idx_hf[t].tolist()) for t in range(T))
    dense_t = ttnn.to_torch(dense_tt).reshape(T, -1).float()
    dense_hf = torch.zeros(T, cfg.n_routed_experts).scatter_(1, idx_hf.reshape(T, -1), w_hf.reshape(T, -1).float())
    logger.info(f"router: identical top-8 sets {same_sets}/{T}; dense weights pcc {pcc(dense_hf, dense_t):.6f}")

    y_tt = tt(x_tt)
    y = ttnn.to_torch(y_tt).reshape(1, T, -1)
    p = pcc(y_hf, y)
    logger.info(
        f"MoE layer {LAYER} T={T}: pcc {p:.6f}  (hf std {y_hf.float().std():.4f}, tt std {y.float().std():.4f})"
    )
    assert same_sets >= T - max(1, T // 64)  # tie-breaks near the top-8 boundary are allowed
    assert p > 0.995
