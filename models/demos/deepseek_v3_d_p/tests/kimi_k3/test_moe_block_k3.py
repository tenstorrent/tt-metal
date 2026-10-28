# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's first N layers with the MoE on the MoE block (K3_MOE_IMPL=block, random MoE weights).

The integration rung, not an accuracy test: every non-MoE weight (embedding, norms, KDA, MLA, AttnRes, layer 0's
dense FFN) is the checkpoint's, read per run (caching off - nothing is written to a shared cache); the MoE layers run
on the MoE block with seeded random weights (tt/kimi_k3/moe_block_ffn.py), so the decoder outputs are not the model's.
What it checks: the block builds and runs next to KDA / MLA / AttnRes on one Galaxy, the first K3_BLOCK_CHECK MoE
calls match the block's host model on the device's own routing, every layer's output is finite; it reports the eager
wall a forward.

    KIMI_K3_CKPT=/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-dequantized K3_MOE_IMPL=block K3_BLOCK_CHECK=2 \
        pytest models/demos/deepseek_v3_d_p/tests/kimi_k3/test_moe_block_k3.py -k L2
"""

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import resolve_checkpoint
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict, load_tensors
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120  # (one chunk: 640 rows a mesh row - the block's K3 Galaxy geometry)
DEPTHS = [2, 5, 12, 24]
ITERS = int(os.environ.get("K3_BLOCK_ITERS", "1"))  # (more: KDA carries across repeated forwards untested here)

PLACEMENTS = [
    pytest.param(
        (8, 4),
        # (the combine's packet needs 2 x 3584 + 32 B: K3's own FABRIC_PAYLOAD_SIZE, 7168, is short of it)
        # (K3_FABRIC=torus_xy: the wrap links on both axes - the fused head's routing exchange needs fewer forwarding
        # slots a port there; on FABRIC_2D lines K3's at 640 rows a mesh row does not fit the head's port arena)
        fabric2d_device_params(
            fabric_payload_size=fabric_payload(KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE),
            l1_small_size=KimiK3Config.L1_SMALL_SIZE,
            **(
                {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}
                if os.environ.get("K3_FABRIC") == "torus_xy"
                else {}
            ),
        ),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _state_dict(checkpoint: Path, num_layers: int, root: str) -> dict:
    """Embedding, final norm, every layer's non-routed weights (no routed experts: the block has its own)."""
    model = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    layers = [load_layer_state_dict(checkpoint, idx) for idx in range(num_layers)]
    return {"embed_weight": model["embed_weight"].float(), "norm_weight": model["norm_weight"], "layers": layers}


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
def test_k3_moe_block(mesh_device, device_params, num_layers):
    assert os.environ.get("K3_MOE_IMPL") == "block", "set K3_MOE_IMPL=block"
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("needs KIMI_K3_CKPT (the dequantized checkpoint)")
    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    t0 = time.time()
    state_dict = _state_dict(checkpoint, num_layers, root)
    logger.info(f"K3 L{num_layers}: checkpoint read in {time.time() - t0:.0f} s")

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        tt_ccl=get_tt_ccl(mesh_device),
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        walk = TtAttnResWalk(
            attn_res,
            hidden,
            list(attn_res.weights.pre),
            list(attn_res.weights.post),
            attn_res.weights.output,
            num_layers,
        )
        return TtAttnResResidual(walk)

    t0 = time.time()
    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        state_dict,
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=None,
    )
    logger.info(f"K3 L{num_layers}: model built in {time.time() - t0:.0f} s")
    del state_dict

    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=SEQ_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )
    tokens = torch.randint(0, config.vocab_size, (SEQ_LEN,), generator=torch.Generator().manual_seed(7)).tolist()
    tokens_tt = prepare_prefill_input_tensor(
        tokens, mesh_device, tuple(mesh_device.shape)[SP_AXIS], False, tuple(mesh_device.shape), SP_AXIS
    )

    stats = {}

    def tap(local_idx, hidden):
        dims = [0, 0]
        dims[SP_AXIS], dims[TP_AXIS] = 2, 3
        h = ttnn.to_torch(
            hidden,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
        ).float()
        stats[local_idx] = (bool(torch.isfinite(h).all()), float(h.abs().max()), float(h.std()))

    walls = []
    try:
        for it in range(ITERS):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            model.forward(tokens_tt, kvpe_cache=kvpe, layer_tap=tap if it == 0 else None)
            ttnn.synchronize_device(mesh_device)
            walls.append(time.perf_counter() - t0)
            logger.info(f"K3 L{num_layers}: forward {it} {walls[-1] * 1e3:.1f} ms")
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    for i in sorted(stats):
        finite, mx, sd = stats[i]
        kind = "MLA" if model.schedule.local_is_mla(i) else "KDA"
        logger.info(f"K3 L{num_layers} layer {i} ({kind}): finite {finite}, |max| {mx:.3g}, std {sd:.3g}")
    logger.info(
        f"K3_BLOCK_RESULT L{num_layers}: eager forward first {walls[0] * 1e3:.1f} ms, then "
        f"{', '.join(f'{w * 1e3:.1f}' for w in walls[1:])} ms (untraced, synced; tap off after the first)"
    )
    assert all(s[0] for s in stats.values()), "a layer's output is not finite"
