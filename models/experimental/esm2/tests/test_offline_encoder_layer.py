# SPDX-License-Identifier: MIT
"""Offline correctness check: one encoder layer + rotary tables, CPU only.

Compares tt/esm2/reference_layers (the TTNN op-map twin) against a
random-weight transformers.EsmLayer / EsmRotaryEmbedding (the oracle code)
with exact weight transfer through the canonical key map. No device, no
network. Run from the model root: python tests/test_offline_encoder_layer.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from tt.esm2.config import Esm2TTConfig  # noqa: E402
from tt.esm2.loader import layer_key_map  # noqa: E402
from tt.esm2.reference_layers import (  # noqa: E402
    Esm2Layer,
    RotaryTables,
    additive_attention_mask,
    position_ids_from_input_ids,
)

from tests.util import nrmse  # noqa: E402

LAYER_PREFIX = "esm.encoder.layer.0."


def build_hf_layer(cfg: Esm2TTConfig, seed: int):
    from transformers import EsmConfig
    from transformers.models.esm import modeling_esm

    hf_cfg = EsmConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        layer_norm_eps=cfg.layer_norm_eps,
        pad_token_id=cfg.pad_token_id,
        mask_token_id=cfg.mask_token_id,
        max_position_embeddings=cfg.max_position_embeddings,
        position_embedding_type="rotary",
        token_dropout=cfg.token_dropout,
        hidden_act="gelu",
    )
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    layer = modeling_esm.EsmLayer(hf_cfg).eval()
    rotary = modeling_esm.EsmRotaryEmbedding(config=hf_cfg).eval()
    return layer, rotary


def transfer_weights(hf_layer, twin: Esm2Layer):
    sd = hf_layer.state_dict()
    for hf_key, canon_key in layer_key_map(0):
        local_key = hf_key[len(LAYER_PREFIX) :]
        path = canon_key.replace("layers.0.", "")
        parts = path.split(".")
        obj = twin
        for p in parts[:-1]:
            obj = getattr(obj, p)
        src = sd[local_key]
        if parts[-1] == "weight":
            obj.weight = torch.nn.Parameter(src.clone())
        else:
            obj.bias = torch.nn.Parameter(src.clone())


def main() -> int:
    cfg = Esm2TTConfig.from_json_file("/weights/config.json")
    torch.manual_seed(0)
    B, L = 2, 97
    # random ids with pads and masks, cls first / eos last per policy
    ids = torch.randint(4, 24, (B, L))
    ids[:, 0] = 0
    ids[:, -1] = 2
    ids[0, 40:44] = cfg.pad_token_id
    ids[1, 5:9] = cfg.pad_token_id
    ids[:, [7, 21, 23]] = cfg.mask_token_id
    am = ids.ne(cfg.pad_token_id).long()

    hf_layer, hf_rotary = build_hf_layer(cfg, seed=1234)
    twin = Esm2Layer(cfg)
    transfer_weights(hf_layer, twin)
    twin.eval()

    # rotary tables: twin vs oracle module
    pos = position_ids_from_input_ids(ids, cfg.pad_token_id)
    cos_t, sin_t = RotaryTables(cfg).cos_sin(pos)
    emb_t = torch.zeros(B, L, cfg.hidden_size)
    cos_h, sin_h = hf_rotary(emb_t, pos)
    rot_err = max((cos_t - cos_h).abs().max().item(), (sin_t - sin_h).abs().max().item())

    attn_bias = additive_attention_mask(am)
    with torch.no_grad():
        x0 = torch.randn(B, L, cfg.hidden_size)
        out = hf_layer(x0.clone(), attention_mask=attn_bias, position_embeddings=(cos_h, sin_h))
        out_hf = out[0] if isinstance(out, tuple) else out
        out_twin = twin(x0.clone(), attn_bias, cos_t, sin_t)

    err = nrmse(out_hf, out_twin)
    print(f"[rotary] max |cos/sin| diff = {rot_err:.3e}")
    print(f"[layer ] output NRMSE = {err:.3e}  shape={tuple(out_twin.shape)}")
    ok_rot = rot_err < 1e-6
    ok_layer = err < 1e-6
    print("PASS" if (ok_rot and ok_layer) else "FAIL")
    return 0 if (ok_rot and ok_layer) else 1


if __name__ == "__main__":
    raise SystemExit(main())
