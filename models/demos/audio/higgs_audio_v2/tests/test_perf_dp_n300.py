# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""N300 data-parallel audio decode: one independent speaker stream per chip.

Single-stream decode is DRAM-bandwidth-bound on one Wormhole chip (single chip):
63.5 tok/s, RTF 0.394, ~15.75ms/step at the device floor. The N300's second
chip doubles aggregate DRAM bandwidth, but tensor-parallel is slower for batch-1
(per-layer CCL > bandwidth gain). Data-parallel sidesteps that: split the (1,2)
mesh into two 1x1 submeshes, put a FULL replicated model on each (num_devices=1
-> zero CCL, identical to the single-chip path), and decode a different stream on each.
Both traces are dispatched non-blocking per step so the chips run concurrently.

Expected: per-stream rate ~unchanged (~63.5 tok/s, RTF ~0.39 each), aggregate
~2x (~127 tok/s), amortized/serving RTF = 25 / aggregate ~= 0.197 -> crosses the
RTF<0.2 stretch goal in throughput terms (the Stage-3 multi-speaker reading).

Token values are irrelevant here (no delay masking) — only op shapes/timing.
"""
import json
import os
import pathlib
import time

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config, load_higgs_v2_state_dict
from models.demos.audio.higgs_audio_v2.tt.model_args import HiggsModelArgs, BASE_TEXT_MODEL
from models.demos.audio.higgs_audio_v2.tt.model import HiggsAudioTTModel
from models.demos.audio.higgs_audio_v2.tt.precision_presets import build_precision
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import create_submeshes


HIGGS_MODEL_DIR = "/data/hf_cache/higgs"
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "baseline_tts_short.json"
CODEC_FRAME_RATE_HZ = 25.0
WARMUP = 4
STEPS = 64


@pytest.fixture(scope="module")
def mesh_device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=200000000)
    yield dev
    ttnn.close_mesh_device(dev)


def _build_stream(submesh, higgs_cfg, state_dict, prompt_ids, precision):
    """Build a full replicated model + traced decode step on one 1x1 submesh."""
    opt = build_precision(precision, higgs_cfg.num_hidden_layers, BASE_TEXT_MODEL)
    args = HiggsModelArgs(mesh_device=submesh, higgs_config=higgs_cfg, max_batch_size=1, max_seq_len=1024,
                          optimizations=opt)
    assert args.num_devices == 1, f"submesh must be 1 device (got {args.num_devices})"
    K, cb_size, dim = args.audio_num_codebooks, args.audio_codebook_size, args.dim
    tt_ccl = TT_CCL(submesh)
    RopeCls = HfRotarySetup if args.use_hf_rope else RotarySetup
    rope_setup = RopeCls(
        device=submesh, batch_size=args.max_batch_size, head_dim=args.head_dim,
        max_seq_len=args.max_seq_len, rope_theta=args.rope_theta, rope_scaling=args.rope_scaling,
        use_qk_fused=args.use_qk_fused, prefetcher=None,
    )
    model = HiggsAudioTTModel(
        args=args, mesh_device=submesh, tt_ccl=tt_ccl,
        state_dict=state_dict, transformation_mats=rope_setup.get_both_trans_mats(), dtype=ttnn.bfloat8_b,
    )
    S = prompt_ids.shape[0]
    _ = model.prefill_text(prompt_ids, rope_setup)
    skip_mem_cfg = args.get_residual_mem_config(Mode.DECODE, None)
    nc = args.get_norm_config("lm_head", Mode.DECODE, None)

    cur_tokens = ttnn.from_torch(
        torch.zeros(1, K, dtype=torch.int32), device=submesh, dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    offsets_dev = ttnn.from_torch(
        (torch.arange(K, dtype=torch.int32) * cb_size).view(1, K), device=submesh, dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    cp_dev = ttnn.from_torch(
        torch.tensor([S], dtype=torch.int32), device=submesh, dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    rope_idxs = rope_setup.get_rot_idxs(torch.tensor([S], dtype=torch.int32), on_host=False)

    def step_body():
        ids = ttnn.add(cur_tokens, offsets_dev)
        emb = model.audio_embedding(ids)
        h = ttnn.sum(emb, dim=1, keepdim=True)
        h = ttnn.reshape(h, (1, 1, 1, h.shape[-1]))
        ttnn.deallocate(emb)
        h = ttnn.to_memory_config(h, skip_mem_cfg)
        rot_mats = rope_setup.get_rot_mats(rope_idxs)
        for blk in model.layers:
            h = blk(h, cp_dev, rot_mats, mode=Mode.DECODE, is_audio_token=True)
        h = model.norm(h, Mode.DECODE, norm_config=nc)
        logits = model.audio_lm_head(h)
        logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, 1, K * cb_size))
        logits = ttnn.reshape(logits, (1, 1, K, cb_size))
        nxt = ttnn.argmax(logits, dim=-1, keepdim=False)
        nxt = ttnn.reshape(nxt, (1, K))
        nxt = ttnn.to_layout(nxt, ttnn.ROW_MAJOR_LAYOUT)
        nxt = ttnn.typecast(nxt, ttnn.uint32)
        ttnn.copy(nxt, cur_tokens)
        return logits

    def advance_pos(pos):
        cp_host = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32,
                                  mesh_mapper=ttnn.ShardTensor2dMesh(submesh, dims=(None, None),
                                                                     mesh_shape=args.cluster_shape))
        ttnn.copy_host_to_device_tensor(cp_host, cp_dev)
        ridx_host = rope_setup.get_rot_idxs(torch.tensor([pos], dtype=torch.int32), on_host=True)
        ttnn.copy_host_to_device_tensor(ridx_host, rope_idxs)

    # compile + capture trace
    advance_pos(S)
    _ = step_body()
    ttnn.synchronize_device(submesh)
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    _ = step_body()
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)
    return {"submesh": submesh, "trace_id": trace_id, "advance_pos": advance_pos, "S": S}


def test_decode_perf_dp_n300(mesh_device):
    assert mesh_device.get_num_devices() == 2, "this test needs a 2-chip (N300) mesh"
    with open(FIXTURE) as _fh:
        fixture = json.load(_fh)
    prompt_ids = torch.tensor(fixture["prompt_text_tokens"], dtype=torch.int64)
    higgs_cfg = HiggsAudioV2Config.from_json(pathlib.Path(HIGGS_MODEL_DIR) / "config.json")
    precision = os.environ.get("HIGGS_PRECISION", "performance")
    _, state_dict = load_higgs_v2_state_dict(HIGGS_MODEL_DIR)

    submeshes = create_submeshes(mesh_device, 2)
    assert len(submeshes) == 2, f"expected 2 submeshes, got {len(submeshes)}"
    logger.info(f"DP: {len(submeshes)} submeshes (one full model per chip), precision={precision}")

    streams = [_build_stream(sm, higgs_cfg, state_dict, prompt_ids, precision) for sm in submeshes]
    logger.info("both streams compiled + traced")

    def run_step(st, pos):
        st["advance_pos"](pos)
        ttnn.execute_trace(st["submesh"], st["trace_id"], cq_id=0, blocking=False)

    for i in range(WARMUP):
        for st in streams:
            run_step(st, st["S"] + i)
    for st in streams:
        ttnn.synchronize_device(st["submesh"])

    t0 = time.perf_counter()
    for i in range(STEPS):
        for st in streams:
            run_step(st, st["S"] + WARMUP + i)
    for st in streams:
        ttnn.synchronize_device(st["submesh"])
    wall = time.perf_counter() - t0

    per_stream_tokps = STEPS / wall
    aggregate_tokps = 2 * STEPS / wall
    per_stream_rtf = CODEC_FRAME_RATE_HZ / per_stream_tokps
    amortized_rtf = CODEC_FRAME_RATE_HZ / aggregate_tokps
    logger.info("==== N300 DATA-PARALLEL DECODE (2 streams, one per chip) ====")
    logger.info(f"  per-stream: {per_stream_tokps:.2f} tok/s  RTF={per_stream_rtf:.4f}")
    logger.info(f"  AGGREGATE : {aggregate_tokps:.2f} tok/s  amortized RTF={amortized_rtf:.4f}  "
                f"(single-chip baseline 63.5 tok/s / 0.394)")
    print(f"PERF_DP_N300 per_stream_tokps={per_stream_tokps:.2f} aggregate_tokps={aggregate_tokps:.2f} "
          f"per_stream_rtf={per_stream_rtf:.4f} amortized_rtf={amortized_rtf:.4f} per_step_ms={1e3*wall/STEPS:.2f}")

    for st in streams:
        ttnn.release_trace(st["submesh"], st["trace_id"])
