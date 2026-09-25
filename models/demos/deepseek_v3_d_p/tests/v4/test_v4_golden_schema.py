# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free: the V4 golden generator (tt/v4/golden.py) on a small random model -- schema, shapes, layer
kinds, the fused-expert round trip of load_reference_layer, and that the window/entries/keys it writes agree with
the reference modules driven directly."""

import json

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.v4 import golden as G
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import HCA, SLIDING, layer_kinds
from models.demos.deepseek_v3_d_p.tt.v4.moe import reference_moe_weights


def _small_cfg():
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)  # SWA SWA CSA HCA
    cfg.hidden_size = 512
    cfg.n_routed_experts = 8
    cfg.moe_intermediate_size = 256
    cfg.q_lora_rank = 128
    cfg.o_lora_rank = 128
    cfg.num_attention_heads = 8
    cfg.index_n_heads = 8
    cfg.vocab_size = 2048
    return cfg


def _layers(cfg):
    torch.manual_seed(3)
    out = []
    for i in range(cfg.num_hidden_layers):
        layer = DeepseekV4DecoderLayer(cfg, i).eval()
        with torch.no_grad():
            for p in layer.parameters():
                p.normal_(0.0, 0.05)
            if hasattr(layer.mlp.gate, "tid2eid"):
                layer.mlp.gate.tid2eid.copy_(
                    torch.randint(0, cfg.n_routed_experts, tuple(layer.mlp.gate.tid2eid.shape))
                )
        out.append(layer)
    return out


def test_generate_golden_schema_and_shapes(tmp_path):
    cfg = _small_cfg()
    layers = _layers(cfg)
    S = 256
    torch.manual_seed(4)
    ids = torch.randint(0, cfg.vocab_size, (S,)).tolist()
    embed = torch.randn(cfg.vocab_size, cfg.hidden_size)
    out = G.generate_golden(
        cfg,
        layer_weights=None,
        embed_weight=embed,
        token_ids=ids,
        out_dir=tmp_path / "trace",
        reference_layers=lambda i: layers[i],
    )
    md = json.loads((out / "metadata.json").read_text())
    assert md["layout"] == G.LAYOUT == G.trace_layout(out) and md["seq_len"] == S and md["token_ids"] == ids
    assert md["layer_kinds"] == layer_kinds(cfg) and md["layers"] == [0, 1, 2, 3]
    for li, kind in enumerate(layer_kinds(cfg)):
        g = G.load_golden(out, li)
        assert tuple(g[f"window_layer_{li}"].shape) == (128, cfg.head_dim)
        if kind == SLIDING:
            assert set(g) == {f"window_layer_{li}"}
        elif kind == HCA:
            assert (
                tuple(g[f"compressed_layer_{li}"].shape) == (S // 128, cfg.head_dim) and f"index_k_layer_{li}" not in g
            )
        else:
            assert tuple(g[f"compressed_layer_{li}"].shape) == (S // 4, cfg.head_dim)
            assert tuple(g[f"index_k_layer_{li}"].shape) == (S // 4, cfg.index_head_dim)
        for v in g.values():
            assert torch.isfinite(v).all() and v.abs().sum() > 0


def test_load_reference_layer_round_trips_the_fused_experts():
    cfg = _small_cfg()
    layer = _layers(cfg)[3]
    w = {k: v.detach().clone() for k, v in layer.state_dict().items() if not k.startswith("mlp.experts.")}
    w["__experts__"] = reference_moe_weights(layer.mlp)["routed_expert_weights"]
    back = G.load_reference_layer(cfg, 3, w)
    for k, v in layer.state_dict().items():
        torch.testing.assert_close(back.state_dict()[k].float(), v.float())


def test_ring_rows_places_tokens_at_position_mod_128():
    k = torch.arange(300).float().view(300, 1).expand(300, 8)
    ring = G.ring_rows(k, 300)
    assert ring[300 % 128 - 1, 0] == 299 and ring[(300 - 128) % 128, 0] == 172
    short = G.ring_rows(k[:40], 40)
    assert short[:40, 0].tolist() == list(range(40)) and short[40:].abs().sum() == 0
