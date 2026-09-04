# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's layer 0 against the model itself: real AttnRes, real weights, the vLLM trace.

`test_block_layer0.py` scores the same layer against a torch reference with the residual swapped for
a plain sum. That isolates KDA, the norms and the FFN, and it deliberately cannot see AttnRes. This
one closes the loop: the real `TtAttnResWalk`, the checkpoint's own folded queries, and the golden
`decoder_output_layer_0` as the target — so what is being compared is Kimi-K3 on this box against
Kimi-K3 on a GPU, with nothing of ours in between.

Layer 0 is the whole architecture in miniature. Its pre-attention read is skipped because nothing is
sealed yet; the seal then fires and moves the embedding into `block_residual`, clearing the live
stream; attention accumulates into the cleared stream; the pre-MLP read mixes those two candidates;
and the MLP accumulates. So the layer's output is `attn + mlp` with **no embedding term**, which is
exactly what `test_golden_contract.py` pinned on host and what makes this a real test of the seal
rather than of addition.

**Fabric2D**, the fabric Kimi-K3 runs on. The torus hang that once forced this was fixed
upstream in #53318, so the restriction is no longer load-bearing.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120
NUM_LAYERS = 1

# Measured 0.9998514 at 5120 tokens. Against the real model rather than a reference, so this carries
# the whole device error budget: bf16 activations through a 5120-step recurrence, a SiTU FFN, and an
# AttnRes read whose softmax runs in fp32 on both sides. The package's chunked per-layer bar is 0.88
# and its single-shot transformer bar is 0.99; one layer against the model beats both, so the bar
# sits where the measurement is rather than where the convention is.
GOLDEN_PCC = 0.999

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 1152},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _shard(mesh_device, hidden):
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.from_torch(
        hidden.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )


def _compose(mesh_device, tensor):
    """The inverse of `_shard`. Dims go in mesh-axis order — swapping them is silent."""
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_layer0_attn_res_matches_golden(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None:
        pytest.skip("no Kimi-K3 checkpoint on this host; set KIMI_K3_HF_MODEL")
    if trace is None:
        pytest.skip("no Kimi-K3 100k golden trace on this host; set KIMI_K3_GOLDEN_TRACE")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    state_dict = load_layer_state_dict(checkpoint, 0)
    hidden = trace.decoder_input(0, SEQ_LEN)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, NUM_LAYERS)

    states = None
    try:
        # One TtAttnRes serves a whole stack; here the stack is one layer, so it holds three
        # queries — q_pre[0] (never issued), q_post[0], and the model-level q_out.
        attn_res = TtAttnRes(
            mesh_device,
            hidden_size=KimiK3Config.EMB_SIZE,
            eps=KimiK3Config.RMS_NORM_EPS,
            tp_axis=TP_AXIS,
            weights=load_attn_res_weights(
                mesh_device,
                load_attn_res_state_dict(checkpoint, NUM_LAYERS, root),
                None,
                num_layers=NUM_LAYERS,
                tensor_parallel_axis=TP_AXIS,
                prefix=root,
            ),
        )

        attention = build_attention(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            schedule=schedule,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )
        states = KdaStateCache({0: attention.kda})
        attention.bind_state_cache(states)
        block = TtKimiK3Block(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            local_idx=0,
            attention=attention,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )

        walk = TtAttnResWalk(
            attn_res,
            _shard(mesh_device, hidden),
            list(attn_res.weights.pre),
            list(attn_res.weights.post),
            attn_res.weights.output,
            NUM_LAYERS,
        )
        residual = TtAttnResResidual(walk)
        block.forward(residual, K3AttnContext())

        # The LIVE stream after the layer, not the model-level read: that is what the trace records
        # as decoder_output_layer_i, and `finish()` would consume a further read site.
        got = _compose(mesh_device, residual.current())
    finally:
        if states is not None:
            states.deallocate()

    want = trace.decoder_output(0, 0, SEQ_LEN)
    passed, message = comp_pcc(want, got, GOLDEN_PCC)
    logger.info(f"K3 layer 0 with AttnRes vs golden decoder_output_layer_0, T={SEQ_LEN}: {message}")
    assert passed, f"K3 layer 0 != the model's own layer 0: {message}"
