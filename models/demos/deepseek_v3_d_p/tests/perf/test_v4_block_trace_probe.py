# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace-capture probe for the V4 prefill block (DS4F-0246 lever 1): can one 5120-token chunk of each layer kind be
captured as a ttnn trace at a FIXED chunk index, and what does the replay cost? The eager block is host-dispatch
bound (issue == wall, 571-780 programs per layer), so the replay time is the ceiling of the trace lever. The MoE's
sub-device swaps go through SubDeviceTraceController (segmented capture, as the V3 engine does). Whatever op rejects
the capture is the finding; the replay is compared with the eager output (same chunk, same state) for PCC."""

import time
import traceback
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from models.demos.deepseek_v3_d_p.tests.pcc.test_v4_block import (
    _EXPERTS,
    _cfg,
    init_reference_layer,
    reference_layer_weights,
    to_device_streams,
    to_host_streams,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4PrefillBlock
from models.demos.deepseek_v3_d_p.tt.v4.kv_cache import allocate_v4_flash_kv_caches
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

_CHUNK = 5120
_MESH_CONFIGS = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "trace_region_size": 256 * 1024 * 1024,
        },
        id="fabric2d-mesh-2x4-trace",
    ),
]


@pytest.mark.timeout(0)
@pytest.mark.xfail(
    strict=False,
    reason="DS4F-0247: the forward pushes per-chunk scalars / allocates on reset / uploads input_ids -- capture rejected "
    "until prepare_chunk lands; passes (and reports replay time) once it does",
)
@pytest.mark.parametrize("layer_idx", [0, 2, 3], ids=["swa-layer0-hash", "csa-layer2-hash", "hca-layer3-topk"])
@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v4_block_trace_probe(mesh_device, device_params, layer_idx):
    cfg = _cfg()
    layer = init_reference_layer(cfg, layer_idx)
    rot = DeepseekV4RotaryEmbedding(cfg)
    sp = mesh_device.shape[0]
    params = SimpleNamespace(
        max_seq_len=2 * _CHUNK,
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
    block.alloc_states(1, 2 * _CHUNK, _CHUNK)
    torch.manual_seed(1)
    streams = to_device_streams(
        mesh_device, (torch.randn(1, _CHUNK, 4, cfg.hidden_size) * 1.5).to(torch.bfloat16).float()
    )
    input_ids = torch.randint(0, cfg.vocab_size, (_CHUNK,))

    def run():
        block.reset_slot(0)
        return block(streams, slot=0, caches=caches, actual_start=0, actual_end=_CHUNK, input_ids=input_ids)

    # eager warm-up (compile) + reference output
    out = run()
    ttnn.synchronize_device(mesh_device)
    ref = to_host_streams(mesh_device, out)
    for t in out:
        ttnn.deallocate(t)

    controller = SubDeviceTraceController(mesh_device)
    block.moe.set_trace_controller(controller)
    try:
        controller.begin_capture()
        out = run()
        controller.end_capture()
    except Exception as e:  # the finding: which op rejects the capture
        frames = [
            ln.strip()
            for ln in traceback.format_exc().splitlines()
            if 'File "' in ln and "/models/demos/" in ln and "test_v4_block_trace_probe" not in ln
        ]
        logger.error(
            f"[v4 trace] layer {layer_idx} ({block.kind}) CAPTURE FAILED: {type(e).__name__}: "
            f"{str(e).splitlines()[0][:300]}\n    V4 frames:\n    " + "\n    ".join(frames[-8:])
        )
        # close the open trace so the fixture teardown (device reads) does not abort the process
        try:
            controller.end_capture()
        except Exception:
            pass
        try:
            controller.release()
        except Exception:
            pass
        block.moe.set_trace_controller(None)
        pytest.fail(f"trace capture rejected in {block.kind}: {frames[-1] if frames else e}")
    logger.info(
        f"[v4 trace] layer {layer_idx} ({block.kind}) captured: {controller.num_segments()} segments, "
        f"{controller.trace_bytes() / 1e6:.1f} MB trace"
    )
    times = []
    for _ in range(5):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        controller.replay()
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    got = to_host_streams(mesh_device, out)
    worst = min(comp_pcc(ref[0, :, h, :].float(), got[0, :, h, :].float())[1] for h in range(4))
    logger.info(
        f"[v4 trace] layer {layer_idx} ({block.kind}) REPLAY median {sorted(times)[2]:.1f} ms "
        f"(all {[round(t, 1) for t in times]}), replay vs eager worst-stream PCC {worst:.6f}"
    )
    controller.release()
    block.moe.set_trace_controller(None)
    assert worst >= 0.999, worst
