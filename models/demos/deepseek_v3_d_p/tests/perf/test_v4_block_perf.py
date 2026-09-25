# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-kind perf probe for the DeepSeek-V4-Flash prefill block (DS4F-0246): one 5120-token chunk per layer kind on 2x4
with random weights, timed three ways -- host ISSUE time (forward returns), host TOTAL time (after synchronize_device)
and, when the realtime profiler is active, DEVICE time per program (max across chips, summed). issue ~= total ~= a
multiple of the device time means the host dispatch is the bottleneck; total ~= device time means the ops are.

Chunk 0 is a fresh slot (compile excluded by a warm-up pass); chunk 1 follows it, so it also attends chunk 0's entries.
Run: TT_VISIBLE_DEVICES=0..7 + the 2x4 single-host MGD; V4_PERF_ITERS overrides the sample count."""

import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_block import (
    _EXPERTS,
    _MESH_CONFIGS,
    _cfg,
    init_reference_layer,
    reference_layer_weights,
    to_device_streams,
)
from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4PrefillBlock
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

_CHUNK = 5120
_ITERS = int(os.environ.get("V4_PERF_ITERS", "3"))


@pytest.mark.timeout(0)
@pytest.mark.parametrize("layer_idx", [0, 2, 3], ids=["swa-layer0-hash", "csa-layer2-hash", "hca-layer3-topk"])
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v4_block_perf(mesh_device, device_params, layer_idx):
    cfg = _cfg()
    layer = init_reference_layer(cfg, layer_idx)
    rot = DeepseekV4RotaryEmbedding(cfg)
    sp = mesh_device.shape[0]
    max_seq = 2 * _CHUNK
    params = SimpleNamespace(
        max_seq_len=max_seq,
        sp_factor=sp,
        first_layer_idx=0,
        num_layers=4,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=0,
        num_users=1,
    )
    caches = allocate_v4_flash_kv_caches(mesh_device=mesh_device, hf_config=cfg, params=params)
    block = TtV4PrefillBlock(
        mesh_device,
        cfg,
        layer_idx,
        reference_layer_weights(layer),
        rotary_emb=rot,
        seq_len_per_chip=_CHUNK // sp,
        num_routed_experts=_EXPERTS,
    )
    block.alloc_states(1, max_seq, _CHUNK)
    torch.manual_seed(1)
    streams = to_device_streams(
        mesh_device, (torch.randn(1, _CHUNK, 4, cfg.hidden_size) * 1.5).to(torch.bfloat16).float()
    )
    input_ids = torch.randint(0, cfg.vocab_size, (_CHUNK,))

    def run(start):
        out = block(streams, slot=0, caches=caches, actual_start=start, actual_end=start + _CHUNK, input_ids=input_ids)
        return out

    def timed(start):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = run(start)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        for t in out:
            ttnn.deallocate(t)
        return (t1 - t0) * 1e3, (t2 - t0) * 1e3

    # warm-up: compile both chunk positions
    block.reset_slot(0)
    for t in run(0):
        ttnn.deallocate(t)
    for t in run(_CHUNK):
        ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)

    rows = []
    for it in range(_ITERS):
        block.reset_slot(0)
        i0, w0 = timed(0)
        i1, w1 = timed(_CHUNK)
        rows.append((i0, w0, i1, w1))
        logger.info(
            f"[v4 perf] layer {layer_idx} ({block.kind}) iter {it}: chunk0 issue {i0:.1f} ms total {w0:.1f} ms | "
            f"chunk1 issue {i1:.1f} ms total {w1:.1f} ms"
        )
    med = [sorted(c)[len(c) // 2] for c in zip(*rows)]
    logger.info(
        f"[v4 perf] layer {layer_idx} ({block.kind}) MEDIAN of {_ITERS}: chunk0 issue {med[0]:.1f} / total {med[1]:.1f} ms; "
        f"chunk1 issue {med[2]:.1f} / total {med[3]:.1f} ms  (5120 tokens, 2x4, {_EXPERTS} experts)"
    )

    if not ttnn.device.IsProgramRealtimeProfilerActive():
        logger.warning("[v4 perf] realtime profiler inactive -- device time not measured")
        return
    block.reset_slot(0)
    for start, tag in ((0, "chunk0"), (_CHUNK, "chunk1")):
        out, per_program = profile_realtime_program_merged(mesh_device, lambda: run(start))
        for t in out:
            ttnn.deallocate(t)
        dev_ms = sum(e["duration_ns"] for e in per_program.values()) / 1e6
        logger.info(
            f"[v4 perf] layer {layer_idx} ({block.kind}) {tag}: DEVICE {dev_ms:.2f} ms over {len(per_program)} programs"
        )
        top = sorted(per_program.items(), key=lambda kv: -kv[1]["duration_ns"])[:12]
        for rid, e in top:
            names = sorted({s.rsplit("/", 1)[-1] for s in e["kernel_sources"]})
            logger.info(f"    {e['duration_ns'] / 1e6:7.3f} ms  {names}")
