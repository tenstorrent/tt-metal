# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's KDA at layer 1, alone, fed the input the model gives it.

The 2-layer ladder localizes cleanly. Every layer-0 stage is >= 0.9998 against the trace, layer 1's
AttnRes read is 0.99988 against a host-derived oracle — so the walk, the seal, the site indexing and
the deferred write are all correct — and then layer 1's KDA output is 0.001. One module, one layer,
and the input to it is known good.

Two hypotheses remain, and this file separates them:

  * layer 1's KDA is wrong ON ITS OWN — a weight, a config, or the layer index somewhere;
  * layer 1's KDA is wrong only AFTER layer 0's has run — which would make it the state cache, the
    `ttnn.copy` write-back, or an allocator interaction between two ttKDA instances.

So the layer is built and run in isolation here. If it matches, the fault is in the sequence, not
the layer, and `KdaStateCache` is the first suspect.

Both the input and the expected output are derived from the trace rather than recorded in it — the
100k trace instruments KDA for layer 0 only:

    read_1   = attn_res(running_sum=out_0, block_residual=[embed], q_pre[1])
    kda_in_1 = input_layernorm_1(read_1)
    attn_1   = out_1 - out_0 - moe_output_layer_1      (layer 1 does not seal: 1 % 12 != 0)
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import attn_res, fold_query
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import (
    TRACE_100K,
    load_checkpoint_tensors,
    resolve_checkpoint,
    resolve_trace,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120

# Layer 0's KDA scores 0.99990 against its recorded output, so the same bar applies. The failing
# path scores 0.001, so there is no grey zone to calibrate.
KDA_PCC = 0.99

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 1152},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _shard(mesh_device, hidden, seq_len=SEQ_LEN):
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


def _compose(mesh_device, tensor, seq_len=SEQ_LEN):
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:seq_len]


def _derived(checkpoint: Path, trace, root: str):
    """Layer 1's KDA input and its expected output, both from the trace plus the checkpoint."""
    names = [
        f"{root}layers.1.{k}"
        for k in ("self_attention_res_norm.weight", "self_attention_res_proj.weight", "input_layernorm.weight")
    ]
    w = {k: v.float() for k, v in load_checkpoint_tensors(checkpoint, names).items()}

    out0 = trace.decoder_output(0, 0, SEQ_LEN)
    read1 = attn_res(
        out0,
        trace.decoder_input(0, SEQ_LEN).unsqueeze(1),
        fold_query(w[names[0]], w[names[1]]),
        eps=KimiK3Config.RMS_NORM_EPS,
    )
    kda_in = read1 * torch.rsqrt(read1.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS) * w[names[2]]
    attn1 = trace.decoder_output(1, 0, SEQ_LEN) - out0 - trace.rows("moe_io", "moe_output_layer_1", 0, SEQ_LEN)
    return kda_in, attn1


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize(
    "layer_idx, input_layer, preceded_by_layer0",
    [(0, 0, False), (1, 1, False), (1, 1, True), (1, 0, False), (0, 1, False)],
    ids=[
        "layer0_control",
        "layer1_alone",
        "layer1_after_layer0",
        "layer1_weights_on_recorded_input",
        "layer0_weights_on_derived_input",
    ],
)
def test_kda_layer_matches_golden(mesh_device, device_params, layer_idx, input_layer, preceded_by_layer0):
    """One KDA layer, fed the input the model gives it.

    `layer0_control` is the discriminator. It runs through exactly the same harness as the other
    two but compares against tensors the trace RECORDED rather than any host derivation, so it
    separates "layer 1 is broken" from "this harness is broken" — the block-level layer-0 test that
    scores 0.9998 goes through `TtKimiK3Block`, not through here, so it cannot make that call.

    `layer1_after_layer0` distinguishes the layer from the sequence: if `layer1_alone` passes and it
    fails, two ttKDA instances sharing a `KdaStateCache` interfere and the layer itself is fine.
    """
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    # Layer 0 is the only KDA layer the 100k trace instruments, so its input and output are the only
    # recorded pair; layer 1's are derived (and the derivation is checked on host by the torch
    # reference, which reproduces the derived output at 0.99996).
    recorded_in = trace.rows("kda", "kda_input_layer_0", 0, SEQ_LEN)
    if input_layer == 0:
        kda_in = recorded_in
    else:
        kda_in, derived_out = _derived(checkpoint, trace, root)

    if (layer_idx, input_layer) == (0, 0):
        want = trace.rows("kda", "kda_output_layer_0", 0, SEQ_LEN)
    elif (layer_idx, input_layer) == (1, 1):
        want = derived_out
    else:
        # A crossed pair the model never computes, so the oracle is the torch reference — which the
        # two uncrossed cases above show is faithful for both layers. Crossing separates the two
        # remaining suspects: the same weights on a known-good input tensor, and known-good weights
        # on the derived one.
        want, _ = kda_forward_reference(
            kda_in.unsqueeze(0),
            load_layer_state_dict(checkpoint, layer_idx)["kda_weights"],
            kimi_k3_kda_config(),
        )
        want = want.squeeze(0)

    layers = [layer_idx] if not preceded_by_layer0 else [0, 1]
    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, 2)

    attentions, states = {}, None
    try:
        for layer_idx in layers:
            attentions[layer_idx] = build_attention(
                mesh_device,
                config,
                KimiK3Config,
                load_layer_state_dict(checkpoint, layer_idx),
                layer_idx=layer_idx,
                schedule=schedule,
                seq_len=SEQ_LEN,
                sp_axis=SP_AXIS,
                tp_axis=TP_AXIS,
            )
        states = KdaStateCache({idx: a.kda for idx, a in attentions.items()})
        for attention in attentions.values():
            attention.bind_state_cache(states)

        if preceded_by_layer0:
            # Run layer 0 exactly as the stack does, on its own real input, and throw the result
            # away. Only its effect on the cache matters here.
            layer0_in = trace.rows("kda", "kda_input_layer_0", 0, SEQ_LEN)
            ttnn.deallocate(attentions[0].forward(_shard(mesh_device, layer0_in), K3AttnContext()))

        got = _compose(mesh_device, attentions[layer_idx].forward(_shard(mesh_device, kda_in), K3AttnContext()))
    finally:
        if states is not None:
            states.deallocate()

    # Per-position PCC separates the two ways a recurrence goes wrong. Token 0's output depends on
    # token 0 and the (zero) conv history alone, so if the first segment is already broken the fault
    # is in the weights or the layout and has nothing to do with the scan; if the first segment is
    # clean and later ones decay, it is accumulation over the 5120 steps.
    segment = SEQ_LEN // 8
    logger.info(
        "  per-position PCC: "
        + "  ".join(
            f"[{i * segment}] {float(str(comp_pcc(want[i * segment:(i + 1) * segment], got[i * segment:(i + 1) * segment], 0.99)[1]).split()[-1]):.5f}"
            for i in range(8)
        )
    )
    passed, message = comp_pcc(want, got, KDA_PCC)
    label = (
        f"layer {layer_idx} weights on layer {input_layer} input" f"{' after layer 0' if preceded_by_layer0 else ''}"
    )
    logger.info(f"K3 KDA {label}: {message}")
    assert passed, f"K3 KDA {label} != the model's own: {message}"
