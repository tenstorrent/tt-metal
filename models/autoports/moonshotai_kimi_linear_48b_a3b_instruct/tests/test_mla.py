# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NoPE MLA layer (absorbed, paged latent cache) vs the fp32 torch oracle with real layer-3 weights."""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.mla_ref import mla_forward_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import assert_pcc, first_shard, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.mla.layer import KimiMLA

MLA_LAYER = 3
BLOCK = 64


@pytest.fixture(scope="module")
def mla_sd(checkpoint):
    return checkpoint.attention_state_dict(MLA_LAYER)


def _page_table(mesh_device, batch, blocks_per_user):
    pt = torch.arange(batch * blocks_per_user, dtype=torch.int32).reshape(batch, blocks_per_user)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return pt, ttnn.from_torch(
        pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper
    )


def test_absorbed_equals_expanded_reference(hf_config, mla_sd):
    torch.manual_seed(20)
    x = (torch.randn(1, 40, hf_config.hidden_size) * 0.5).bfloat16()
    o1, _ = mla_forward_reference(x, mla_sd, hf_config, absorbed=False)
    o2, _ = mla_forward_reference(x, mla_sd, hf_config, absorbed=True)
    assert_pcc(o1, o2, 0.9999, "absorbed vs expanded torch MLA")


@pytest.mark.parametrize("T,valid", [(32, 32), (64, 61), (256, 256)])
def test_mla_prefill(mesh_device, ccl, hf_config, mla_sd, cache_path, T, valid):
    torch.manual_seed(21)
    x = (torch.randn(1, valid, hf_config.hidden_size) * 0.5).bfloat16()
    ref, _ = mla_forward_reference(x, mla_sd, hf_config)
    mla = KimiMLA(mesh_device, hf_config, mla_sd, layer_idx=MLA_LAYER, ccl=ccl, cache_path=cache_path, block_size=BLOCK)
    cache = mla.allocate_cache(num_blocks=64)
    _, pt = _page_table(mesh_device, 1, 16)
    xp = torch.zeros(1, 1, T, hf_config.hidden_size, dtype=torch.bfloat16)
    xp[0, 0, :valid] = x[0]
    out = mla.forward_prefill(replicated(mesh_device, xp), cache, pt, valid_len=valid)
    assert_pcc(ref[0], first_shard(out).float()[0, 0, :valid], 0.99, f"mla prefill T={T} valid={valid}")


def test_mla_prefill_then_decode(mesh_device, ccl, hf_config, mla_sd, cache_path):
    torch.manual_seed(22)
    P, D = 61, 5
    x = (torch.randn(1, P + D, hf_config.hidden_size) * 0.5).bfloat16()
    ref, _ = mla_forward_reference(x, mla_sd, hf_config)
    mla = KimiMLA(mesh_device, hf_config, mla_sd, layer_idx=MLA_LAYER, ccl=ccl, cache_path=cache_path, block_size=BLOCK)
    cache = mla.allocate_cache(num_blocks=64)
    _, pt = _page_table(mesh_device, 1, 16)
    xp = torch.zeros(1, 1, 64, hf_config.hidden_size, dtype=torch.bfloat16)
    xp[0, 0, :P] = x[0, :P]
    mla.forward_prefill(replicated(mesh_device, xp), cache, pt, valid_len=P)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    for i in range(D):
        pos = ttnn.from_torch(
            torch.tensor([P + i], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )
        xi = x[0, P + i].reshape(1, 1, 1, -1)
        out = mla.forward_decode(replicated(mesh_device, xi), cache, pt, pos)
        assert_pcc(ref[0, P + i], first_shard(out).float().reshape(-1), 0.99, f"mla decode token {i}")


def test_mla_decode_batch(mesh_device, ccl, hf_config, mla_sd, cache_path):
    """Batch of 8 users at different positions after individual prefills (slot page tables)."""
    torch.manual_seed(23)
    B, Ptot = 8, 40
    mla = KimiMLA(mesh_device, hf_config, mla_sd, layer_idx=MLA_LAYER, ccl=ccl, cache_path=cache_path, block_size=BLOCK)
    cache = mla.allocate_cache(num_blocks=B * 4)
    pt_torch, pt = _page_table(mesh_device, B, 4)
    xs, refs, lens = [], [], []
    for u in range(B):
        L = 8 + 4 * u
        x = (torch.randn(1, L + 1, hf_config.hidden_size) * 0.5).bfloat16()
        ref, _ = mla_forward_reference(x, mla_sd, hf_config)
        xp = torch.zeros(1, 1, 32, hf_config.hidden_size, dtype=torch.bfloat16)
        xp[0, 0, :L] = x[0, :L]
        mla.forward_prefill(replicated(mesh_device, xp), cache, pt, user_id=u, valid_len=L)
        xs.append(x[0, L])
        refs.append(ref[0, L])
        lens.append(L)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    pos = ttnn.from_torch(
        torch.tensor(lens, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
    )
    xb = torch.stack(xs).reshape(1, 1, B, -1)
    out = first_shard(mla.forward_decode(replicated(mesh_device, xb), cache, pt, pos)).float()[0, 0]
    for u in range(B):
        assert_pcc(refs[u], out[u], 0.99, f"batched decode user {u} pos {lens[u]}")
