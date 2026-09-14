# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer with residuals — the composition, after every piece passes alone.

This is D3's closing test. It is also the first place a residual-stream layout error can show:
every block below it is correct in isolation and still adds up wrong if the residual is added at the
wrong point or in the wrong layout.

The chunked variant runs the same layer over two chunks and compares against the one-shot reference,
which is the layer-level statement of P2's property.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.tests.common import (
    assert_pcc,
    assert_tp_replicas_agree,
    cfg_full,
    galaxy_mesh,
    make_ccl,
    pcc,
    random_layer_weights,
    reference_layer,
    spec_mesh_config,
    sp_shard_activation,
)
from models.demos.llama_3_1_8b.tt.attention import allocate_kv_cache
from models.demos.llama_3_1_8b.tt.layer import DecoderLayer
from models.demos.llama_3_1_8b.tt.rope import RopeSetup

CHUNK = 5120
NUM_LAYERS = 2
LAYER = 1


def _build(mesh_device, cfg, mc, weights, max_seq_len):
    ccl = make_ccl(mesh_device)
    rope = RopeSetup(mesh_device, cfg, mc)
    layer = DecoderLayer(
        mesh_device, cfg, mc, ccl, rope, layer_idx=LAYER, state_dict=weights, max_seq_len=max_seq_len
    )
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_key_value_heads,
        tp=mc.tp,
        sp_axis=mc.sp_axis,
        head_dim=cfg.head_dim,
    )
    return layer, cache, rope


@galaxy_mesh()
def test_decoder_layer_vs_ref(mesh_device, device_params, topology_name):
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    weights = random_layer_weights(cfg, seed=31)
    layer, cache, rope = _build(mesh_device, cfg, mc, weights, CHUNK)

    torch.manual_seed(6)
    x = torch.randn(1, 1, CHUNK, cfg.hidden_size) * 0.5
    out = layer(
        sp_shard_activation(x, mesh_device, mc),
        rope.build_indexed_rope(CHUNK, CHUNK),
        kv_cache=cache,
        cached_len=0,
        indexed_rope=True,
    )
    got = assert_tp_replicas_agree(out, mesh_device, mc, name="decoder_layer", tol=0.1)

    cos, sin = ref.rope_cos_sin(cfg, CHUNK, dtype=torch.float32)
    ref_out, _, _ = reference_layer(cfg, weights)(x[0].to(torch.float16), cos, sin)
    assert_pcc(f"decoder_layer[{topology_name}]", pcc(ref_out.unsqueeze(0).float(), got))


@galaxy_mesh()
def test_decoder_layer_chunked_vs_ref(mesh_device, device_params, topology_name):
    """Two chunks through one layer; chunk 1's hidden state must match the one-shot reference."""
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    total = 2 * CHUNK
    weights = random_layer_weights(cfg, seed=32)
    layer, cache, rope = _build(mesh_device, cfg, mc, weights, total)
    indexed = rope.build_indexed_rope(total, CHUNK)

    torch.manual_seed(7)
    x = torch.randn(1, 1, total, cfg.hidden_size) * 0.5
    outs = []
    for start in (0, CHUNK):
        outs.append(
            assert_tp_replicas_agree(
                layer(
                    sp_shard_activation(x[:, :, start : start + CHUNK], mesh_device, mc),
                    indexed,
                    kv_cache=cache,
                    cached_len=start,
                    indexed_rope=True,
                ),
                mesh_device,
                mc,
                name=f"decoder_layer chunk@{start}",
                tol=0.1,
            )
        )

    cos, sin = ref.rope_cos_sin(cfg, total, dtype=torch.float32)
    ref_out, _, _ = reference_layer(cfg, weights)(x[0].to(torch.float16), cos, sin)
    p = pcc(ref_out[:, CHUNK:].unsqueeze(0).float(), outs[1])
    logger.info(f"decoder layer chunk 1 vs one-shot reference: PCC {p:.6f}")
    assert_pcc(f"decoder_layer_chunked[{topology_name}]", p)


@galaxy_mesh()
def test_residual_path_is_present(mesh_device, device_params):
    """Control: zeroing the attention and MLP outputs must leave the input exactly.

    With all projection weights zero the layer reduces to ``x -> x + 0 -> x + 0``, so any deviation
    is a residual that is missing, doubled, or added in the wrong layout — none of which a PCC
    against a full reference would isolate.
    """
    cfg = cfg_full()
    mc = spec_mesh_config(mesh_device)
    weights = random_layer_weights(cfg, seed=33)
    for key in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        weights[key] = torch.zeros_like(weights[key])
    layer, cache, rope = _build(mesh_device, cfg, mc, weights, CHUNK)

    torch.manual_seed(8)
    x = torch.randn(1, 1, CHUNK, cfg.hidden_size) * 0.5
    out = layer(
        sp_shard_activation(x, mesh_device, mc),
        rope.build_indexed_rope(CHUNK, CHUNK),
        kv_cache=cache,
        cached_len=0,
        indexed_rope=True,
    )
    got = assert_tp_replicas_agree(out, mesh_device, mc, name="residual_only", tol=0.05)
    assert_pcc("residual_only", pcc(x, got), target=0.9999)
