# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does a layer stack still place its kernels once the AttnRes sealed set is deep? (#54876)

At four or more sealed snapshots, a full-attention layer fails to place its circular buffers:

    TT_THROW: Statically allocated circular buffers in program N clash with L1 buffers ...
    L1 buffer allocated at 1563072 and static circular buffer region ends at 1563264

192 bytes, at the top of a ~1.5 MB per-core L1. It is a placement failure, not a capacity one: the
sealed set lives in DRAM and is 18 MB/chip even at depth 8, and `attn_res_gather_softmax`'s own
per-core L1 is a function of `Wt` and `ring_size`, not of the candidate count.

`tests/attn_res/model/test_attn_res.py` already runs the read at `num_sealed=8` and passes, because
it runs AttnRes and nothing else. What fails is the INTERACTION — AttnRes's L1 tenancy plus MLA's
CB placement in one process — so the reproducer has to be a real layer stack, not an op.

The trick that makes this cheap: sealed depth is a function of GLOBAL position, not of how many
layers a rank holds. A rank of twelve layers starting at layer `F` inherits `F // 12` snapshots and
seals one more, so `first_layer_idx=84` reaches depth 8 — the full 93-layer model's maximum — with
twelve layers on one Galaxy, instead of an unbuildable 96-layer run.

    first_layer_idx    depth   equivalent to
                  0        1   a first rank
                 36        4   2xGLX rank 1 at its first chunk
                 72        7   4xGLX rank 3
                 84        8   the tail of the full model

This is the gate a fix has to clear. Passing at depth 4 is not evidence: `l1_small_size=2048` does
that and still dies at depth 6.
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import resolve_checkpoint
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120
SLICE_LAYERS = 12
_L1_SMALL = int(os.environ.get("PREFILL_L1_SMALL_SIZE", 4096))

# first_layer_idx -> the sealed depth the rank reaches. 84 is the deepest the 93-layer model has.
DEPTH_CASES = [0, 36, 60, 72, 84]

PLACEMENTS = [
    pytest.param(
        (8, 4),
        # 4096 is what the adapter ships; overridable because the value that satisfies both
        # AttnRes and MLA moves with sealed depth (#54876) and finding it is a sweep.
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": _L1_SMALL},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _packed_input(mesh_device, planes):
    """A stand-in for what a rank receives: `[1, planes, N, d]`, seq on SP and emb on TP.

    Content is irrelevant — this measures whether the kernels can be PLACED, not what they compute
    (`test_transformer_pipeline_split.py` owns the numbers). Random rather than zeros so no op takes
    a degenerate shortcut on an all-zero tensor.
    """
    torch.manual_seed(0)
    host = torch.randn(1, planes, SEQ_LEN, KimiK3Config.EMB_SIZE, dtype=torch.float32).to(torch.bfloat16)
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("first_layer_idx", DEPTH_CASES, ids=[f"F{f}" for f in DEPTH_CASES])
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_layer_stack_places_kernels_at_sealed_depth(mesh_device, device_params, first_layer_idx):
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("needs KIMI_K3_HF_MODEL")

    checkpoint = Path(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)
    is_first = first_layer_idx == 0
    inherited = first_layer_idx // KimiK3Config.ATTN_RES_BLOCK_SIZE

    # The deepest rungs run off the end of the stack: 93 layers means F=84 has only nine left,
    # not twelve. Clamp rather than skip -- what this measures is the sealed depth a rank OPENS
    # at, which F fixes on its own, so a short tail is still the case under test.
    slice_layers = min(SLICE_LAYERS, KimiK3Config.NUM_LAYERS - first_layer_idx)

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        {},
        num_layers=slice_layers,
        seq_len=SEQ_LEN,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first,
        is_last_rank=True,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )
    assert model.inbound_planes == inherited + 1

    kvpe = (
        allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=SEQ_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )
        if model.schedule.num_mla_layers
        else None
    )

    # A first rank embeds token ids; every other rank receives the packed [live | sealed] tensor.
    if is_first:
        from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor

        model_input = prepare_prefill_input_tensor(
            [0] * SEQ_LEN, mesh_device, tuple(mesh_device.shape)[SP_AXIS], False, tuple(mesh_device.shape), SP_AXIS
        )
    else:
        model_input = _packed_input(mesh_device, model.inbound_planes)

    try:
        # The assertion IS that this returns. A placement failure raises RuntimeError out of
        # program.cpp, which pytest reports verbatim — more useful than anything asserted after.
        out = model.forward(model_input, kvpe_cache=kvpe)
        assert out is not None
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()
