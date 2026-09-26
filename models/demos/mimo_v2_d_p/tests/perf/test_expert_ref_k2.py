# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Baseline for the streamed-expert prototypes at Kimi K2 / DeepSeek expert shapes (H 7168, I 2048, bf4 weights, bf8
activations) on one chip: TtRoutedExpert's unified (unified_routed_expert_moe) and fused (moe_fused_swiglu) paths,
with EXPERTS experts of TOKENS rows each (synthetic dispatch result, random weights). Same stats format as the
streamed tests (tag ``kref_{path}_M{tokens}_E{experts}``, per-expert time = op time / E) so se_run's analyzer applies.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


PATHS = _env_list("MIMO_KREF_PATHS", "unified,fused")
TOKENS = _env_list("MIMO_KREF_TOKENS", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_KREF_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_KREF_ITERS", "3"))
H, I = int(os.environ.get("MIMO_KREF_H", "7168")), 2048
RM = int(os.environ.get("MIMO_KREF_RM", "0"))
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("tokens", TOKENS, ids=lambda t: f"M{t}")
@pytest.mark.parametrize("path", PATHS)
def test_expert_ref_k2(mesh_device, path, tokens):
    E = EXPERTS
    mc = extract_mesh_config(mesh_device)
    gidx_host = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=E, dispatch_group_size=mc.dispatch_group_size, num_dispatch_groups=mc.num_dispatch_groups
    )
    gidx = ttnn.from_torch(
        gidx_host,
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
    torch.manual_seed(0)
    w = {
        "gate_proj": torch.randn(I, H) * 0.02,
        "up_proj": torch.randn(I, H) * 0.02,
        "down_proj": torch.randn(H, I) * 0.02,
    }
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=E,
        global_expert_idx_table=gidx,
        emb_dim=H,
        hidden_dim=I,
        max_tokens=tokens,
        torch_weights=[w] * E,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
        hybrid_token_threshold={"unified": None, "fused": tokens}[path],
    )
    tok_pad = (tokens + 31) // 32 * 32
    ids = gidx_host[0, 0].long()
    counts = torch.zeros(1, E, dtype=torch.int32)
    regions = torch.zeros(1, E, dtype=torch.int32)
    counts[0, ids] = tokens
    if os.environ.get("MIMO_KREF_COUNTS"):  # uneven per-expert counts (each at most `tokens`, the op's capacity)
        counts[0, ids] = torch.tensor([int(c) for c in os.environ["MIMO_KREF_COUNTS"].split(",")], dtype=torch.int32)
    regions[0, ids] = torch.arange(E, dtype=torch.int32) * tok_pad
    tt_counts = ttnn.from_torch(counts, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_regions = ttnn.from_torch(regions, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if RM:  # the model's input: the row-major bf16 dispatch buffer (the op tilizes / packs x itself)
        x = ttnn.from_torch(
            torch.randn(E * tok_pad, H) * 0.1,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        x = ttnn.from_torch(
            torch.randn(E * tok_pad, H) * 0.1,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tag = f"kref_{path}{'_rm' if RM else ''}_M{tokens}_E{E}" + (
        "_c" + os.environ["MIMO_KREF_COUNTS"].replace(",", "-") if os.environ.get("MIMO_KREF_COUNTS") else ""
    )
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "M": tokens,
                    "E": E,
                    "weight_bytes": E * 3 * H * I * 0.5625,
                    "flops": 6 * E * tokens * H * I,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        y = expert(x, tt_counts, tt_regions)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        if y.buffer_address() != x.buffer_address():
            y.deallocate(True)
    logger.info(f"ran {tag}")
