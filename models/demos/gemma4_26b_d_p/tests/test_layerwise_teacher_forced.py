# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced per-layer error: each TT decoder layer gets the fp32 golden input of that layer
(both chunks, so its KV cache is golden-driven too) and is compared to the golden output.

Separates per-layer numerical error from error accumulated through depth.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_d_p.reference.config import FULL, SLIDING, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.gemma4_26b_d_p.tests.test_model_prefill import golden
from models.demos.gemma4_26b_d_p.tt.model import TtGemma4Model, block_cyclic_index


@MESH_PARAMS
@pytest.mark.parametrize("seq,chunk", [(8192, 4096)])
def test_layerwise_teacher_forced(mesh_device, device_params, seq, chunk):
    g = golden(30, seq)
    cfg = Gemma4TextConfig.from_json()
    reader = CheckpointReader()
    sp, tp = tuple(mesh_device.shape)
    C = chunk // sp
    x_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None))
    results = []
    for li in range(cfg.num_hidden_layers):
        m = TtGemma4Model(mesh_device, cfg, reader, fabric_config=device_params["fabric_config"], max_seq_len=seq, chunk_size=chunk, layers=[li])
        layer = m.layers[0]
        t = SLIDING if layer.is_sliding else FULL
        pccs = []
        for c in range(seq // chunk):
            kv_actual = c * chunk
            if li == 0:
                x_in = m.embed(g["ids"][kv_actual : kv_actual + chunk][block_cyclic_index(kv_actual, sp, C) - kv_actual])
            else:
                src = g["hidden"][li - 1, kv_actual : kv_actual + chunk].float()
                x_in = ttnn.from_torch(src[block_cyclic_index(kv_actual, sp, C) - kv_actual][None, None], device=mesh_device,
                                       layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=x_map)
            y = layer(x_in, m.rope[t], m.trans_mat, m.kv[t], cache_layer=0, kv_actual=kv_actual)
            got = m.gather_hidden(y, kv_actual)
            ref = g["hidden"][li, kv_actual : kv_actual + chunk].float()
            inp = g["hidden"][li - 1, kv_actual : kv_actual + chunk].float() if li else None
            d_pcc = comp_pcc(ref - inp, got - inp)[1] if inp is not None else float("nan")
            pccs.append((comp_pcc(ref, got)[1], d_pcc))
            y.deallocate(True)
        results.append((li, cfg.layer_types[li][:4], pccs))
        logger.info(f"L{li:2d} {cfg.layer_types[li][:4]}: out/delta PCC per chunk {[(round(a, 5), round(b, 5)) for a, b in pccs]}")
        del m
    worst = min(min(p[0] for p in r[2]) for r in results)
    logger.info(f"mesh={mesh_id(mesh_device)} teacher-forced worst single-layer PCC {worst}")
