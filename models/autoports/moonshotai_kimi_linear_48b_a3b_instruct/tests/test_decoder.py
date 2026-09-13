# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decoder layers 0 (KDA+dense), 1 (KDA+MoE), 3 (MLA+MoE), 26 (MLA+MoE) vs transformers-5.17 goldens (stage 02)."""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import assert_pcc, first_shard, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.layer import KimiDecoderLayer

BLOCK = 64


def _tensor(h):
    return h if torch.is_tensor(h) else h[0]


@pytest.mark.parametrize("layer_idx", [0, 1, 3, 26])
@pytest.mark.parametrize("T", [32, 128])
def test_layer_prefill_vs_golden(mesh_device, ccl, hf_config, checkpoint, goldens, cache_path, layer_idx, T):
    run = goldens["runs"][f"prefill{T}"]["hooks"]
    x_in = _tensor(run[f"layer{layer_idx}"]["in"]).float()  # [1,T,H]
    x_ref = _tensor(run[f"layer{layer_idx}"]["out"]).float()
    attn_ref = _tensor(run[f"layer{layer_idx}.attn"]["out"]).float()
    layer = KimiDecoderLayer(
        mesh_device,
        hf_config,
        checkpoint.layer_state_dict(layer_idx),
        layer_idx=layer_idx,
        ccl=ccl,
        cache_path=cache_path,
    )
    kw = {}
    if layer.is_kda:
        kw["kda_state"] = layer.attn.allocate_prefill_state()
    else:
        kw["cache"] = layer.attn.allocate_cache(num_blocks=8)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
        kw["page_table"] = ttnn.from_torch(
            torch.arange(8, dtype=torch.int32).reshape(1, 8),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )
    out, _ = layer.forward_prefill(replicated(mesh_device, x_in.reshape(1, 1, T, -1).bfloat16()), valid_len=T, **kw)
    out_t = first_shard(out).float()[0, 0]
    # residual-stream PCC is dominated by the input; also check the layer's own contribution (out - in)
    assert_pcc(x_ref[0], out_t, 0.995, f"layer {layer_idx} T={T} residual out")
    assert_pcc(x_ref[0] - x_in[0], out_t - x_in[0], 0.99, f"layer {layer_idx} T={T} delta (attn+mlp)")


@pytest.mark.parametrize("layer_idx", [0, 1, 3, 26])
def test_layer_decode_vs_golden(mesh_device, ccl, hf_config, checkpoint, goldens, cache_path, layer_idx):
    pre = goldens["runs"]["prefill128"]["hooks"]
    dec = goldens["runs"]["decode128"]["hooks"]
    x_pre = _tensor(pre[f"layer{layer_idx}"]["in"]).float()  # [1,128,H]
    x_dec = _tensor(dec[f"layer{layer_idx}"]["in"]).float()  # [1,1,H]
    x_ref = _tensor(dec[f"layer{layer_idx}"]["out"]).float()
    layer = KimiDecoderLayer(
        mesh_device,
        hf_config,
        checkpoint.layer_state_dict(layer_idx),
        layer_idx=layer_idx,
        ccl=ccl,
        cache_path=cache_path,
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    if layer.is_kda:
        st = layer.attn.allocate_prefill_state()
        _, st = layer.forward_prefill(
            replicated(mesh_device, x_pre.reshape(1, 1, 128, -1).bfloat16()), kda_state=st, valid_len=128
        )
        ds = layer.attn.allocate_decode_state(batch=1)
        layer.attn.prefill_state_to_decode(st, ds)
        out = layer.forward_decode(replicated(mesh_device, x_dec.reshape(1, 1, 1, -1).bfloat16()), kda_state=ds)
    else:
        cache = layer.attn.allocate_cache(num_blocks=8)
        pt = ttnn.from_torch(
            torch.arange(8, dtype=torch.int32).reshape(1, 8),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )
        layer.forward_prefill(
            replicated(mesh_device, x_pre.reshape(1, 1, 128, -1).bfloat16()), cache=cache, page_table=pt, valid_len=128
        )
        pos = ttnn.from_torch(
            torch.tensor([128], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )
        out = layer.forward_decode(
            replicated(mesh_device, x_dec.reshape(1, 1, 1, -1).bfloat16()), cache=cache, page_table=pt, cur_pos=pos
        )
    out_t = first_shard(out).float().reshape(-1)
    assert_pcc(x_ref.reshape(-1), out_t, 0.995, f"layer {layer_idx} decode@128 residual out")
    assert_pcc(
        x_ref.reshape(-1) - x_dec.reshape(-1), out_t - x_dec.reshape(-1), 0.99, f"layer {layer_idx} decode@128 delta"
    )
