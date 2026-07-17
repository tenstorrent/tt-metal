# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Tenstorrent DFlash drafter (``TtDFlashDrafter``) vs the REAL z-lab HF ``DFlashDraftModel``.

This is the sign-off validation of the *validator* — it compares the device's context-KV build to the
ground truth produced by the actual HF drafter's forward (NOT our torch golden). It jointly validates
the compute graph, the weights, and the RoPE (the device's deepseek-yarn vs the trained model's own
rope), against the production model rather than a re-derivation.

How the ground truth is obtained: the real ``Qwen3DFlashAttention`` builds K/V as
``[k_proj(target_hidden) | k_proj(noise)]`` (context ++ noise), applies k_norm + RoPE to the whole
thing, and stores it in ``past_key_values``. The CONTEXT part is the first ``ctx_len`` positions — and
because k_norm/RoPE are per-position, that slice equals exactly what prefill builds. So we run the real
forward (via the ``hf_context_kv`` fixture), slice ``key_cache[:, :, :ctx_len, :]`` per layer, and PCC
the device K/V against it.

K vs V isolates the failure mode: V is matmul-only (v_proj, no norm/rope) so it should be ~1.0; K adds
k_norm + RoPE, so if V passes but K drops, the culprit is the rope (device deepseek-yarn ≠ trained rope)
or k_norm — not the weights/matmul.

The HF reference drafter, its config, and its weights (shared with the device, gated on ``use_pretrained``)
are built by the fixtures in ``conftest.py``. This file only feeds a SYNTHETIC context feature and PCCs.
Requires (host): torch + transformers; ``$DFLASH_HF_MODEL`` = a dir with ``config.json`` (+ safetensors for
the pretrained axis); and a Blackhole mesh for the device side. Skips cleanly if the model can't be built.

    DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/speculative_decoding/dflash/test_dflash.py -svv
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.dflash.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.999

_FABRIC_1D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
}


@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.parametrize("ctx_len", [5120], ids=["ctx5k"])
@pytest.mark.parametrize("fc_mode", ["sliced", "concat"], ids=["sliced", "concat"])
@pytest.mark.parametrize("seq_parallel", [False, True], ids=["seq_repl", "seq_par"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            _FABRIC_1D,
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_device_vs_hf_pcc(
    mesh_device,
    device_params,
    num_links,
    topology,
    ctx_len,
    fc_mode,
    seq_parallel,
    use_pretrained,
    drafter_cfg,
    drafter_state_dict,
    hf_context_kv,
):
    # random / pretrained weights come from the conftest ``use_pretrained`` fixture ([random, pretrained]).
    # The HF reference model, cfg, and weights come from the conftest fixtures — the SAME weights feed both
    # the HF reference (via hf_context_kv) and the device drafter (drafter_state_dict). Here we just feed a
    # synthetic context feature and PCC the device K/V.
    logger.info(f"fc_mode={fc_mode}  weights={'pretrained' if use_pretrained else 'random'}  ctx_len={ctx_len}")
    cfg = drafter_cfg
    sd = drafter_state_dict

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    tp = mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size

    # One synthetic context feature, fed identically to both sides.
    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, ctx_len, cfg.target_feature_size, generator=gen, dtype=torch.float32)

    # ---- ground truth: the REAL HF drafter forward (context slice of its KV cache) ----
    real = hf_context_kv(ctx)

    # ---- device ----
    drafter = TtDFlashDrafter(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=ctx_len,
        num_links=num_links,
        topology=topology,
        fc_mode=fc_mode,
        seq_parallel=seq_parallel,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim
    if seq_parallel:
        hidden_shard[sp_axis] = 2  # ALSO SP-shard the tap on seq → each chip taps its own [seq/sp] slice
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    drafter.reset()
    for j, tid in enumerate(cfg.target_layer_ids):
        h_j = ctx[:, :, j * H : (j + 1) * H].to(torch.bfloat16).reshape(1, 1, ctx_len, H)
        h_tt = ttnn.from_torch(
            h_j,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        drafter.tap(h_tt, tid)
    drafter.write_kv_cache()
    ttnn.synchronize_device(mesh_device)

    # seq_parallel: cache SP-sharded on seq → concat SP along seq(dim2), TP along kv-head(dim1) → full
    # [num_layers, kv_heads, ctx_len, head_dim] directly (the host[:num_layers] slice is then a no-op).
    # Phase-1: SP-replicated → concat stacks 8 copies on dim0, take the first replica.
    read_dims = (2, 1) if seq_parallel else (0, 1)

    def _read(cache):
        host = ttnn.to_torch(
            cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=read_dims, mesh_shape=mesh_shape)
        )
        return host[: cfg.num_hidden_layers][:, :, :ctx_len, :].float()  # [num_layers, kv_heads, ctx_len, head_dim]

    dk = _read(drafter.k_cache)
    dv = _read(drafter.v_cache)

    for i in range(cfg.num_hidden_layers):
        rk, rv = real[i]
        ok_k, pcc_k = comp_pcc(rk, dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(rv, dv[i], PCC_THRESHOLD)
        logger.info(f"layer {i}: K pcc={pcc_k} (ok={ok_k})  V pcc={pcc_v} (ok={ok_v})")
        # V (matmul-only) should be ~1.0; if V passes but K fails, suspect the RoPE (deepseek-yarn vs the
        # trained model's rope) or k_norm, not the weights.
        assert ok_v, f"V layer {i}: device vs HF PCC {pcc_v} < {PCC_THRESHOLD} (matmul/weights mismatch)"
        assert ok_k, f"K layer {i}: device vs HF PCC {pcc_k} < {PCC_THRESHOLD} (norm/rope mismatch if V passed)"
