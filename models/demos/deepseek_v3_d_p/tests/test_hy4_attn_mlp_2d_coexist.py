# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttMLA2DTP's sparse path and TtMlp2D in one process on the prefill mesh (L1_SMALL 1536), alternating per chunk as
the block does: attention (with its all-to-alls / persistent gathers) then the 2D FFN, layer after layer. Checks the
FFN against torch on the attention output it was fed, so a CCL that runs out of L1_SMALL or corrupts state shows.

ttMLA2DTP lives on the 2D-attention branch (#66248); until both are on one branch this skips. Measured on a merge of
the two (2026-11-19): with ttnn.all_gather / all_reduce / reduce_scatter (TtMlp2D prefill_ccl="generic") the first
FFN after attention fails, "Out of Memory: Not enough space to allocate 1920 B L1_SMALL buffer ... bank size is
1536 B"; with the persistent path (the default) 8K x 3 rounds and 32K x 2 rounds pass.

Env: ``HY4_COEXIST_SEQ`` (default 8192), ``HY4_COEXIST_ROUNDS`` (default 3)."""

from __future__ import annotations

import copy
import os
import time

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference import hy4_moe as ref
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4TP32Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params

pytest.importorskip("models.demos.deepseek_v3_d_p.tt.mla.mla_2d", reason="needs ttMLA2DTP (#66248)")
from models.demos.deepseek_v3_d_p.tests.test_hy4_attention import (  # noqa: E402
    CHECKPOINT,
    CHUNK,
    L1_SMALL,
    SHARED_LAYER,
    host_indices_to_device,
    layer0_hidden,
    sliding_topk,
    ttmla_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG, ttMLA2DTP  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.moe.tt_mlp_2d import TtMlp2D  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology  # noqa: E402
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache  # noqa: E402
from tt_py_test_utils_common.utils_for_testing import comp_pcc  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
]
SP_AXIS, TP_AXIS = 0, 1
SEQ = int(os.environ.get("HY4_COEXIST_SEQ", "8192"))
ROUNDS = int(os.environ.get("HY4_COEXIST_ROUNDS", "3"))


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=L1_SMALL), id="tp32_2d")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_attention_then_mlp_2d_same_process(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    config = hy4_hf_config(max_seq=SEQ)
    config.max_seq_len = SEQ
    setattr(config, MLA_2D_TP_FLAG, True)
    config.indexer_types = ["shared"] * len(config.indexer_types)
    topology = per_axis_topology(device_params["fabric_config"])
    ckpt = Hy4Checkpoint(CHECKPOINT)
    hn, _ = layer0_hidden(ckpt, SEQ, config.rms_norm_eps)
    topk = sliding_topk(SEQ, config.index_topk)
    chunk = min(SEQ, CHUNK)
    kvpe = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=SEQ,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        tp_axis=TP_AXIS,
    )
    mla = ttMLA2DTP(
        config,
        ttmla_weights(ckpt, SHARED_LAYER, copy.copy(config), False),
        mesh_device,
        layer_idx=SHARED_LAYER,
        seq_len=SEQ,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        sparse_kv_cache_format=kvpe.format,
    )
    w = {p: ckpt._read(f"model.layers.0.mlp.{p}.weight") for p in ("gate_proj", "up_proj", "down_proj")}
    mlp = TtMlp2D(
        mesh_device,
        emb_dim=Hy4TP32Config.EMB_SIZE,
        hidden_dim=Hy4TP32Config.INTERMEDIATE_SIZE,
        hidden_axis=TP_AXIS,
        torch_weights=w,
        num_links=2,
        topology=topology,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=SEQ, chunk_size_global=chunk)
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)
    pccs = []
    t0 = time.time()
    for rnd in range(ROUNDS):
        for start in range(0, SEQ, chunk):
            x = ttnn.from_torch(
                hn[start : start + chunk].view(1, 1, chunk, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            a = mla.forward(
                hidden_states=x,
                rope_tensors=rope_tensors,
                kvpe_cache=kvpe,
                actual_start=start,
                indexer_indices=host_indices_to_device(topk[start : start + chunk], mesh_device, SP_AXIS),
            )
            y = mlp(a)
            a_host = ttnn.to_torch(a, mesh_composer=composer).float()[0, 0]
            y_host = ttnn.to_torch(y, mesh_composer=composer).float()[0, 0]
            want = ref.swiglu_mlp(a_host.to(torch.bfloat16), w["gate_proj"], w["up_proj"], w["down_proj"])
            pccs.append(float(comp_pcc(want, y_host)[1]))
            for t in (x, a, y):
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"round {rnd}: {SEQ // chunk} chunk(s) of attention + 2D FFN, min FFN PCC {min(pccs):.6f}")
    logger.info(f"COEXIST_RESULT rounds={ROUNDS} seq={SEQ} min_pcc={min(pccs):.6f} seconds={time.time() - t0:.1f}")
    assert min(pccs) >= 0.99
