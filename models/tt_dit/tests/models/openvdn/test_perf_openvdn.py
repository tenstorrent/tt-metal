# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN block performance at the production shape on the Blackhole 4x8 (TP 4 x SP 8): the t2va 1344x768, 5 s
request the pipeline test runs (37 latent frames of 24 x 42, 39 prompt and 414 audio rows, packed length 37749 padded
to 37888). Random weights in the released checkpoint's shapes; a step is 50 of these blocks.

Measures the warm block time (program cache populated, weights on device; mean of OPENVDN_PERF_ITERS forwards, each
followed by a device sync). With OPENVDN_SYNC_LOG=1 the hybrid attention logs every section's time instead (the syncs
add overhead; use for ranking, not the headline). Under Tracy one more forward runs between the `start` / `stop`
signposts.

Regression gate at the production shape: the block output is saved to OPENVDN_ARTIFACTS/block_out.pt; with
OPENVDN_PERF_BASELINE pointing at an earlier run's file, the output is compared to it (PCC and RMSE/std against the
bf16 bars) so a lossless optimization can be checked at the real length, which the CPU reference cannot reach.

Run: pytest models/tt_dit/tests/models/openvdn/test_perf_openvdn.py -s --tb=short
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
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from models.tt_dit.models.transformers.openvdn import reference_hybrid_openvdn as ref_h
from models.tt_dit.models.transformers.openvdn.hybrid_attention_openvdn import HybridLayout
from models.tt_dit.models.transformers.openvdn.transformer_block_openvdn import OpenVDNTransformerBlock
from models.tt_dit.tests.models.minimax_h3.common import (
    MESH_4X8_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    randomize_norm_weights,
)
from models.tt_dit.tests.models.openvdn.test_mesh_openvdn import SP_AXIS, TP_AXIS, Packing, _parallel
from models.tt_dit.tests.models.openvdn.test_modules_openvdn import (
    BF16_PCC,
    BF16_RMSE,
    _bf16,
    _metrics,
    _randomize_hybrid,
    _round_params,
)
from models.tt_dit.utils import tensor

try:
    from tracy import signpost
except ImportError:  # not a profiler build

    def signpost(_name: str) -> None:
        pass


HIDDEN = REAL_BLOCK_CONFIG["hidden_size"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
NUM_TEXT, NUM_AUDIO = 39, 414
NUM_FRAMES = 37
GRID = (24, 42)
NUM_VIDEO = NUM_FRAMES * GRID[0] * GRID[1]
SEQ_LEN = NUM_TEXT + NUM_AUDIO + NUM_VIDEO
CHUNK, RADIUS = 5, 1
ITERS = int(os.environ.get("OPENVDN_PERF_ITERS", 2))
#: OPENVDN_PERF_TRACE=1: capture the block forward in a trace (persistent CCL buffers) and time trace replays.
TRACE = os.environ.get("OPENVDN_PERF_TRACE") == "1"
#: OPENVDN_ALIGNED=1: the block's rows in the frame-aligned order (`HybridLayout.aligned`), as the pipeline packs them.
ALIGNED = os.environ.get("OPENVDN_ALIGNED", "1") == "1"
MESH = (
    pytest.param((4, 8), {**MESH_4X8_RING.values[1], "trace_region_size": 1_175_000_000}, id="4x8_traced")
    if TRACE
    else MESH_4X8_RING
)


def _artifacts() -> Path:
    path = Path(os.environ.get("OPENVDN_ARTIFACTS", Path.home() / "openvdn_artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH], indirect=True)
def test_block_perf_openvdn(mesh_device: ttnn.MeshDevice) -> None:
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block

    gen = torch.Generator().manual_seed(8)
    torch.manual_seed(7)
    block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG)
    randomize_norm_weights(block)
    block.attn = ref_h.HybridAttention(block.attn, HIDDEN, radius=RADIUS, chunk=CHUNK)
    _randomize_hybrid(block.attn, gen)
    block = _round_params(block).eval()

    layout = HybridLayout(
        seq_len=SEQ_LEN,
        video_start=NUM_TEXT + NUM_AUDIO,
        num_frames=NUM_FRAMES,
        frame_height=GRID[0],
        frame_width=GRID[1],
        text_start=0,
        text_len=NUM_TEXT,
        aligned=ALIGNED,
    )
    packing = Packing(mesh_device, layout)
    padded_len = packing.padded_len
    position_ids, token_tags, timestep_indices = packed_layout(NUM_TEXT, NUM_AUDIO, NUM_VIDEO, grid_hw=GRID)
    cos, sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)(position_ids)
    tt_cos, tt_sin = (
        tensor.from_torch(
            packing.pack(t).reshape(1, 1, padded_len, -1),
            device=mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., SP_AXIS, None],
        )
        for t in prepare_rope_tables(cos, sin, HEAD_DIM)
    )
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    x = _bf16(torch.randn(SEQ_LEN, HIDDEN, generator=gen))
    temb = torch.randn(num_timesteps, REAL_BLOCK_CONFIG["time_embed_dim"], generator=gen)

    ccl_manager, parallel_config = _parallel(mesh_device)
    tt_block = OpenVDNTransformerBlock(
        radius=RADIUS,
        chunk=CHUNK,
        **TT_BLOCK_CONFIG,
        rotary_dim=cos.shape[-1],
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
        use_persistent_ccl_buffers=TRACE,
    )
    tt_block.load_torch_state_dict(block.state_dict())
    del block
    tt_block.attn.layout = layout

    spatial = tensor.from_torch(
        packing.pack(x).reshape(1, 1, padded_len, HIDDEN),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_axes=[None, None, SP_AXIS, TP_AXIS],
    )
    kwargs = dict(
        logical_n=tensor.logical_length_tensor(SEQ_LEN, device=mesh_device),
        temb=tensor.from_torch(temb.reshape(1, 1, *temb.shape), device=mesh_device, dtype=ttnn.float32),
        adaln_indices=tensor.from_torch(
            packing.pack(adaln_indices.to(torch.int32)).reshape(1, 1, 1, padded_len),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, SP_AXIS],
        ),
        rope_cos=tt_cos,
        rope_sin=tt_sin,
    )

    def forward() -> ttnn.Tensor:
        out = tt_block(spatial_1BND=spatial, **kwargs)
        ttnn.synchronize_device(mesh_device)
        return out

    t0 = time.perf_counter()
    forward()
    cold_ms = (time.perf_counter() - t0) * 1e3
    logger.info(f"block forward, cold (compile): {cold_ms:.0f} ms")

    if TRACE:
        eager = forward()  # warm, and every cached constant made before the capture
        eager_out = tensor.to_torch(eager, mesh_axes=[None, None, SP_AXIS, TP_AXIS]).float()
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced = tt_block(spatial_1BND=spatial, **kwargs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        def forward() -> ttnn.Tensor:  # noqa: F811
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            return traced

        forward()
        same = torch.equal(tensor.to_torch(traced, mesh_axes=[None, None, SP_AXIS, TP_AXIS]).float(), eager_out)
        pcc, rmse = _metrics(tensor.to_torch(traced, mesh_axes=[None, None, SP_AXIS, TP_AXIS]).float(), eager_out)
        logger.info(f"block forward, traced replay identical to eager: {same} (PCC {pcc:.7f}, RMSE/std {rmse:.2e})")

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = forward()
        times.append((time.perf_counter() - t0) * 1e3)
    warm_ms = sum(times) / len(times)
    sync_log = os.environ.get("OPENVDN_SYNC_LOG") == "1"
    logger.info(
        f"block forward, warm: {warm_ms:.1f} ms (runs {[round(t, 1) for t in times]}; x50 layers = "
        f"{warm_ms * 50 / 1e3:.2f} s/step){' [section syncs on]' if sync_log else ''}"
    )

    signpost("start")
    forward()
    signpost("stop")

    result = tensor.to_torch(out, mesh_axes=[None, None, SP_AXIS, TP_AXIS]).float()
    result = packing.unpack(result.reshape(result.shape[-2], result.shape[-1]))
    artifacts = _artifacts()
    torch.save(result.bfloat16(), artifacts / "block_out.pt")
    summary = dict(
        seq_len=SEQ_LEN,
        padded_len=padded_len,
        cold_ms=cold_ms,
        warm_ms=warm_ms,
        runs_ms=times,
        sync_log=sync_log,
        traced=TRACE,
        aligned=ALIGNED,
    )

    baseline = os.environ.get("OPENVDN_PERF_BASELINE")
    if baseline:
        expected = torch.load(baseline).float()
        pcc, rmse = _metrics(result, expected)
        summary.update(baseline=baseline, baseline_pcc=pcc, baseline_rmse=rmse)
        logger.info(f"block output vs baseline {baseline}: PCC {pcc:.7f} (>= {BF16_PCC}), RMSE/std {rmse:.2e}")
        assert pcc >= BF16_PCC and rmse <= BF16_RMSE, f"block output moved: PCC {pcc:.7f}, RMSE/std {rmse:.2e}"
    (artifacts / "block_perf.json").write_text(json.dumps(summary, indent=2))
