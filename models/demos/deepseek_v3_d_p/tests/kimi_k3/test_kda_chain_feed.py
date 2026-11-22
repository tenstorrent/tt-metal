# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A Kimi-K3.1 KDA layer (1 or 2) teacher-fed from the FP64 chain on one 4x4 stage (SP on axis 1), chunk by chunk.

The device half of ``kda_chain_feed.py`` (its docstring has the three steps; the CPU half is tested in
``test_teacher_feed_contract.py``). The production KDA attention of the input's layer (``build_attention`` from the
stage's TTNN weight cache, as ``TtKimiK3Transformer`` builds it) runs every 2560-token chunk of the prefix in order,
eager, with its recurrent and convolution carry in the engine's state slabs (``KdaStates``, one slot), the real
``actual_start`` / ``actual_end`` of each chunk: the carry passes from chunk to chunk exactly as in chunked prefill.
Its input is the chain's FP64 KDA input rounded once to bf16, so only the KDA layer's own error accumulates, in its
state. Each chunk's output is saved to ``$TT_KDA_TF_OUT``, and the carry after every chunk ending on a multiple of
``$TT_KDA_TF_STATE_EVERY`` rows (default 20480: where a 2560-row chunk and a 4096-row chain segment end together) is
read back (recurrent fp32 ``[1, 96, 128, 128]``, convolution q / k / v ``[1, 3, 12288]``). Nothing is scored while the
device is held (``kda_chain_feed.py score``). Skips unless ``$TT_KDA_TF_INPUT`` and ``$TT_KDA_TF_OUT`` are set.

    TT_KDA_TF_INPUT=/mnt/.../k31-1m-kda-l1-input TT_KDA_TF_OUT=/mnt/.../k31-1m-kda-l1-dev \\
        scripts/run_safe_pytest.sh \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_chain_feed.py::test_k31_kda_layer_teacher_fed[blackhole-4x4-sp1]"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, provenance
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    read_slab_layer,
    reconstruct_convolution_at_sp_rank,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import run_kda_chunk
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path, finish_run, load_input, start_run
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, deallocate_state
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]
SOURCE, OUT = os.environ.get("TT_KDA_TF_INPUT"), os.environ.get("TT_KDA_TF_OUT")


def read_carry(mesh_device, slabs: KdaStates, layer: int, sp_axis: int, tp_axis: int) -> dict[str, torch.Tensor]:
    """The layer's carry in slot 0, in ``KDAReferenceState``'s layout, as the last SP rank holds it."""
    config = kimi_k3_kda_config()
    state = read_slab_layer(mesh_device, slabs, 0, layer)
    try:
        sp_rank = tuple(mesh_device.shape)[sp_axis] - 1
        recurrent = reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tp_axis, sp_rank)
        convolution = reconstruct_convolution_at_sp_rank(
            state.convolution, mesh_device, sp_axis, tp_axis, sp_rank, config, KDA_CONVOLUTION_PAGE_CHANNELS
        )
    finally:
        deallocate_state(state)
    width = config.num_heads * config.head_k_dim
    carry = {"recurrent": recurrent.float()}
    for i, name in enumerate(("q", "k", "v")):
        carry[f"{name}_convolution"] = convolution[..., i * width : (i + 1) * width].float()
    return carry


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.skipif(
    not (SOURCE and OUT),
    reason="set TT_KDA_TF_INPUT (a kda_chain_feed input) and TT_KDA_TF_OUT (the run's output dir)",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
def test_k31_kda_layer_teacher_fed(mesh_device, device_params):
    source, out = Path(SOURCE), Path(OUT)
    built = json.loads((source / MANIFEST).read_text())
    layer = built["layer"]
    assert built.get("kind") == "kda_chain_feed", f"{source} is not a kda_chain_feed input"
    assert layer + 1 not in KimiK31Config.DSA_LAYERS_1BASED, f"layer {layer} is not a KDA layer"
    count = int(os.environ.get("TT_KDA_TF_COUNT", built["segments"][-1]["end"]))
    state_every = int(os.environ.get("TT_KDA_TF_STATE_EVERY", 20480))
    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    assert state_every % chunk == 0, f"state reads every {state_every} rows must end on {chunk}-row chunks"
    max_seq_len = -(-count // chunk) * chunk
    hf_config, params = k31_stage.window_params(
        mesh_shape, layer, 1, max_seq_len, is_first_rank=False, is_last_rank=True
    )
    manifest = {
        "layer": layer,
        "input": str(source),
        "input_manifest": {k: built[k] for k in ("trace", "trace_commit", "input", "checkpoint", "rebuild")},
        "count": count,
        "chunk": chunk,
        "mesh": list(mesh_shape),
        "sp_axis": layout.sp_axis,
        "state_every": state_every,
        "weight_cache": str(params.weight_cache_path),
        **provenance(),
    }
    start_run(out, manifest)

    schedule = KimiK3LayerSchedule.build(KimiK31Config, layer, 1)
    topology = per_axis_topology(mesh_device=mesh_device)
    attention = build_attention(
        mesh_device,
        hf_config,
        KimiK31Config,
        {},
        layer_idx=layer,
        schedule=schedule,
        seq_len=chunk,
        sp_axis=layout.sp_axis,
        tp_axis=layout.tp_axis,
        num_links=params.num_links,
        topology=topology,
        weight_cache_path=params.weight_cache_path,
        max_seq_len=max_seq_len,
        is_chunked=True,
        first_layer_idx=layer,
    )
    geometry = kimi_k3_kda_geometry(mesh_shape, layout.sp_axis, layout.tp_axis)
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(layer,), num_slots=1)
    dims = [None, None]
    dims[layout.sp_axis], dims[layout.tp_axis] = -2, -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(dims))
    num_chunks = -(-count // chunk)
    states = []
    began = time.perf_counter()
    try:
        for c in range(num_chunks):
            lo, hi = c * chunk, min((c + 1) * chunk, count)
            rows = torch.zeros(chunk, hf_config.hidden_size, dtype=torch.bfloat16)
            rows[: hi - lo] = load_input(source, lo, hi)
            hidden = ttnn.from_torch(
                rows.reshape(1, 1, chunk, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            output = run_kda_chunk(attention, hidden, slabs, None, cache_user_id=0, actual_start=lo, actual_end=hi)
            ttnn.deallocate(hidden)
            torch.save(
                {"start": lo, "end": hi, "attn_out": compose(mesh_device, output, hi - lo).to(torch.bfloat16)},
                chunk_path(out, c),
            )
            ttnn.deallocate(output)
            if hi % state_every == 0:
                name = f"state_{hi:08d}.pt"
                torch.save(read_carry(mesh_device, slabs, layer, layout.sp_axis, layout.tp_axis), out / name)
                states.append({"row": hi, "file": name})
            logger.info(
                f"KDA_TF chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) done, {time.perf_counter() - began:.1f} s"
            )
        final = read_carry(mesh_device, slabs, layer, layout.sp_axis, layout.tp_axis)
    finally:
        ttnn.deallocate(slabs.recurrent)
        ttnn.deallocate(slabs.convolution)
    for name, values in final.items():
        assert torch.isfinite(values).all(), f"non-finite values in the final {name} carry"
    seconds = time.perf_counter() - began
    finish_run(out, {**manifest, "states": states}, seconds)
    logger.info(f"KDA_TF done: layer {layer}, {num_chunks} chunks, {count} rows, {seconds:.1f} s -> {out}")
