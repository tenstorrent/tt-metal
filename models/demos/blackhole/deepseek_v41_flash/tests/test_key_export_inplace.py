# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill -> decode index-key export must update the decode key slab IN PLACE: the captured decode traces read ``k_cache`` by address, so a re-allocated slab (the slice_write export,
DSV41_KEY_EXPORT_FILL=0) leaves them scoring stale keys (B=16 / 262,144 wrong-answer bug, evals_gpqa_tau3/b16_isolation). Cheap variant (3 layers, no vLLM): production sizes (B=16, max_ctx 262,144,
one prefill slot per mesh row, chunk 512), a 4,934-token prompt prefilled into several decode users; ``buffer_address()`` of every key slab must be unchanged by the exports.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_key_export_inplace.py
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "probe_t22_k3_ids.json")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 800000000},
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
def test_key_slab_export_is_in_place(mesh_device, device_params, monkeypatch):
    for k, v in dict(
        DSV41_PREFILL_UP="1",
        DSV41_VLLM_CHUNK="512",
        DSV41_PF_UMASK="1",
        DSV41_MO_OVERLAP="0",
        DSV41_POOL_DTYPE="fp8",
        DSV41_ENGRAM_RAM="1",
        MOE_COMPUTE_FP32_ACC="1",
        MOE_COMPUTE_BFP8_WEIGHTS="1",
        DSV41_IDX_BACKEND="matmul",
    ).items():
        monkeypatch.setenv(k, os.environ.get(k, v))
    ids = json.load(open(IDS))["ids"]
    L, B, ctx = len(ids), 16, 262144
    pp = default_page_params(ctx, B // 4)
    paged = PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"])
    args, m, _, _ = create_tt_model(mesh_device, B, ctx, paged, layer_ids=[0, 1, 2, 3])
    assert (
        m.Up == 1 and m.U == 4 and m.index_owner
    ), "production-like build expected (one prefill slot per row, index key owners present)"
    before = {lid: ix.k_cache.buffer_address() for lid, ix in m.index_owner.items()}
    toks = torch.as_tensor(ids).long()
    for user in (0, 5, 15, 1):  # different mesh rows / in-row positions, repeated exports
        m.prefill_interleaved([(user, toks, 0, L)], 512, s_pad_max=-(-ctx // 512) * 512)
        after = {lid: ix.k_cache.buffer_address() for lid, ix in m.index_owner.items()}
        assert (
            after == before
        ), f"decode key slab re-allocated by the export of user {user}: {before} -> {after} (DSV41_KEY_EXPORT_FILL={os.environ.get('DSV41_KEY_EXPORT_FILL', 'default')})"
