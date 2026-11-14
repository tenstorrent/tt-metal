# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_ATTN_GLUE_CUTS: the cut data-movement ops leave every output bit-identical (device).

apply_partial_rope_decode with the flag on writes the pass-through slice to DRAM directly instead of slicing to L1 and
copying; the rotated output must equal the flag-off output byte for byte, for the verify shape (32 rows = 8 users x 4
candidates, q with the local heads, k with one KV head), the drafter / batched-decode shape (8 rows) and an L1 input
(the attention's q / k are L1 after the q/k norm), with distinct per-row positions.

Run: MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_attn_glue_cuts.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device
from models.demos.blackhole.qwen36.tests.test_rope_tp import _rope_params
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, rot_mats_decode


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_ATTN_GLUE_CUTS", raising=False)
    assert tpc.attn_glue_cuts_enabled() is False
    monkeypatch.setenv("QWEN36_ATTN_GLUE_CUTS", "1")
    assert tpc.attn_glue_cuts_enabled() is True


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("rows", [32, 8], ids=["verify32", "decode8"])
def test_partial_rope_decode_glue_cut_bit_identical(mesh_device, rows, monkeypatch, reset_seeds, ensure_gc):
    _, HD, rope_dim, theta, NH, NKV = _rope_params(mesh_device, B=rows)
    positions = torch.arange(rows, dtype=torch.int32) * 37 + 32761  # long-context, distinct per row
    cos_tt, sin_tt = rot_mats_decode(mesh_device, rope_dim, 40960, theta, positions)
    cos_tt = ttnn.reshape(cos_tt, (1, 1, rows, rope_dim))  # the verify hands the rope [1,1,R,rd] (once per trace)
    sin_tt = ttnn.reshape(sin_tt, (1, 1, rows, rope_dim))
    for heads in (NH, NKV):
        x = torch.randn(1, rows, heads, HD, dtype=torch.bfloat16) * 3
        outs = []
        for flag in ("0", "1"):
            monkeypatch.setenv("QWEN36_ATTN_GLUE_CUTS", flag)
            x_tt = ttnn.to_memory_config(replicate_to_device(mesh_device, x), ttnn.L1_MEMORY_CONFIG)
            y = apply_partial_rope_decode(x_tt, cos_tt, sin_tt, heads, rows, rope_dim)
            outs.append(ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)))
            ttnn.deallocate(y)
            ttnn.deallocate(x_tt)
        same = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
        logger.info(f"rows={rows} heads={heads}: glue cut output bit-identical={same} shape={tuple(outs[0].shape)}")
        assert same, f"rows={rows} heads={heads}: QWEN36_ATTN_GLUE_CUTS changed the partial-RoPE output"
