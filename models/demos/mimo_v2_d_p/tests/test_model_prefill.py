# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stitched MiMo-V2 layers 0..5 (GA+dense, 4x SWA+MoE, GA+MoE) on a real prompt, chunked prefill vs the HF chain.

Checks every layer's output hidden and the K/V written to the block-cyclic caches.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.golden import golden
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel
from models.demos.mimo_v2_d_p.tt.rope import rope_perm
from models.demos.mimo_v2_d_p.tt.tt_prefill_runtime import MiMoKvCaches, MiMoPrefillRuntime


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("seq,chunk", [(8192, 4096)], ids=["2x4k"])
def test_model_prefill(mesh_device, device_params, n_layers, seq, chunk):
    g = golden(n_layers, seq)
    cfg = MiMoTextConfig.from_json()
    model = TtMiMoModel(mesh_device, cfg, lambda i: layer_state(i, cfg), fabric_config=device_params["fabric_config"], max_seq_len=seq,
                        chunk_size=chunk, layers=list(range(n_layers)), global_state=global_state)
    got = torch.zeros(n_layers, seq, cfg.hidden_size)

    def capture(layer_idx, x, kv_actual):
        got[layer_idx, kv_actual : kv_actual + chunk] = model.gather_hidden(x, kv_actual)

    for c in range(seq // chunk):
        out = model.prefill_chunk(g["ids"][c * chunk : (c + 1) * chunk], c * chunk, capture=capture)
        out.deallocate(True)

    worst = 1.0
    reader = MiMoPrefillRuntime.__new__(MiMoPrefillRuntime)
    reader.mesh_device, reader.model, reader.cfg = mesh_device, model, cfg
    reader.config = type("C", (), {"chunk_size": chunk, "max_seq_len": seq})
    for i in range(n_layers):
        ref = g["hidden"][i + 1].float()
        _, p = comp_pcc(ref, got[i])
        _, pd = comp_pcc(ref - g["hidden"][i].float(), got[i] - g["hidden"][i].float())
        spec = cfg.layer_attn(i)
        k_dev, v_dev = reader.read_layer_kv(MiMoKvCaches(model.kv), 0, i, seq)
        k_ref, v_ref = (t.float() for t in g["kv"][i])
        pk = comp_pcc(k_ref[..., rope_perm(spec.head_dim, spec.rope_dim)], k_dev)[1]
        pv = comp_pcc(v_ref, v_dev)[1]
        worst = min(worst, p, pk, pv)
        logger.info(f"L{i} ({spec.kind[:4]}, {'moe' if cfg.is_moe(i) else 'dense'}): hidden PCC {p:.5f} delta {pd:.5f} | K {pk:.5f} V {pv:.5f}")
    logger.info(f"mesh={mesh_id(mesh_device)} {n_layers} layers {seq // chunk}x{chunk}: worst PCC {worst:.5f}")
    assert worst > 0.98, worst
