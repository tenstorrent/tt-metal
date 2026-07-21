# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for SP sliding-window prefill SDPA (all_gather composition).

Isolates ``tt/attention/sp_sliding_sdpa.py``: all_gather(K/V) + local SDPA with
global causal+sliding mask (+ sinks).  Golden is full-sequence ``_causal_sdpa``.
"""

import pytest
import torch

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.config import ProgramConfig
from models.demos.gpt_oss_d_p.tt.attention.sp_sliding_sdpa import sp_sliding_window_sdpa
from models.demos.gpt_oss_d_p.tt.ccl import CCLManager
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links

from ..test_factory import compare_tensors, parametrize_mesh_with_fabric

PCC_THRESHOLD = 0.99

# GPT-OSS 120B defaults
_HEAD_DIM = 64
_NUM_Q_HEADS = 64
_NUM_KV_HEADS = 8
_SCALE = _HEAD_DIM**-0.5
_SLIDING_WINDOW = 128


def _build_causal_mask(S: int, sliding_window: int | None, device=None) -> torch.Tensor:
    i = torch.arange(S, device=device).unsqueeze(1)
    j = torch.arange(S, device=device).unsqueeze(0)
    causal = j > i
    if sliding_window is not None:
        masked = causal | (j < (i - sliding_window + 1))
    else:
        masked = causal
    mask = torch.zeros(S, S, device=device)
    mask[masked] = float("-inf")
    return mask


def _causal_sdpa(q, k, v, sinks=None, scale=_SCALE, sliding_window=_SLIDING_WINDOW):
    """GQA causal SDPA with optional sliding window and per-head sinks."""
    _, Hq, S, _ = q.shape
    group = Hq // k.shape[1]
    k_exp = k.repeat_interleave(group, dim=1)
    v_exp = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k_exp) * scale
    if sinks is not None:
        scores = scores + sinks
    scores = scores + _build_causal_mask(S, sliding_window, device=q.device)
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.einsum("bhst,bhtd->bhsd", probs, v_exp)


def _make_sdpa_inputs(S, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    q = torch.randn(1, _NUM_Q_HEADS, S, _HEAD_DIM, generator=g)
    k = torch.randn(1, _NUM_KV_HEADS, S, _HEAD_DIM, generator=g)
    v = torch.randn(1, _NUM_KV_HEADS, S, _HEAD_DIM, generator=g)
    sinks = torch.randn(1, _NUM_Q_HEADS, 1, 1, generator=g)
    return q, k, v, sinks


def _gather_sharded_sdpa_output(tt_out, mesh_device, sp_axis=0, tp_axis=1):
    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 1
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(tt_out, mesh_composer=mesh_composer)


def _upload_sharded_qkv(q, k, v, mesh_device):
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(2, 1))
    kwargs = dict(device=mesh_device, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    return ttnn.from_torch(q, **kwargs), ttnn.from_torch(k, **kwargs), ttnn.from_torch(v, **kwargs)


def _upload_sinks(sinks, mesh_device, mesh_config):
    return ttnn.from_torch(
        sinks / _SCALE,
        device=mesh_device,
        mesh_mapper=mesh_config.sequence_parallel(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )


@parametrize_mesh_with_fabric([(2, 4), (4, 8)])
@pytest.mark.parametrize(
    "seq_len_local",
    [32, 48],
    ids=["seq_local_32", "seq_local_48"],
)
def test_sp_sliding_window_sdpa(mesh_device, device_params, reset_seeds, seq_len_local):
    """
    Compositional ring-style path: all_gather(K/V) + local SDPA with sliding mask.

    Each SP row keeps its local Q shard and attends against the full gathered K/V
    under a global causal + sliding-window mask.
    """
    mesh_shape = tuple(mesh_device.shape)
    sp, tp = mesh_shape[0], mesh_shape[1]
    if sp <= 1:
        pytest.skip("SP sliding-window SDPA requires mesh rows > 1")

    from models.common.utility_functions import is_blackhole
    from models.demos.gpt_oss_d_p.config import MeshConfig, ModeConfig

    mesh_config = MeshConfig(
        mesh_shape,
        decode=ModeConfig(tp=tp, ep=sp),
        prefill=ModeConfig(tp=tp, sp=sp, ep=1),
    )
    topology = ttnn.Topology.Ring if not is_blackhole() and mesh_shape != (1, 1) else ttnn.Topology.Linear
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=topology)

    seq_total = seq_len_local * sp
    q, k, v, sinks = _make_sdpa_inputs(seq_total, seed=seq_len_local)
    ref_out = _causal_sdpa(q, k, v, sinks=sinks, sliding_window=_SLIDING_WINDOW)

    tt_q, tt_k, tt_v = _upload_sharded_qkv(q, k, v, mesh_device)
    tt_sinks = _upload_sinks(sinks, mesh_device, mesh_config)

    tt_out = sp_sliding_window_sdpa(
        tt_q,
        tt_k,
        tt_v,
        tt_sinks,
        seq_len=seq_len_local,
        sliding_window=_SLIDING_WINDOW,
        mesh_config=mesh_config,
        mesh_device=mesh_device,
        program_config=ProgramConfig(),
        ccl_manager=ccl_manager,
    )

    tt_out_torch = _gather_sharded_sdpa_output(tt_out, mesh_device)
    passing, output = compare_tensors(tt_out_torch, ref_out, mesh_device, pcc_threshold=PCC_THRESHOLD)
    assert passing, f"SP sliding-window SDPA failed on {mesh_shape} seq_local={seq_len_local}: {output}"
