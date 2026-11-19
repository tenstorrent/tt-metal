# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill blocks with iHC on the TP32 layout (#65884, #65891), layers 0..N-1 vs the fp32 reference.

The TP32 pieces do not compose in ``TtPrefillTransformer`` yet (``tt/hy4/layout.py`` TP32_BLOCKERS): the DSA indexer has
no 2D layout (#65887), so a full-indexer layer cannot build ``ttMLA2DTP``, and the embedding is not vocab-sharded
(#65892). This test runs the blocks themselves on the TP32 placement instead, with those two pieces from the host:

  * mesh: the physical 8x4 torus with #65884's split: the residual's hidden over axis 1 (4 chips), heads (8 per chip)
    and the prefill sequence over axis 0 (8 chips); residual ``[S / 8, 6144 / 4]`` per chip. NOT RUN YET: it needs
    ``ttMLA2DTP`` / ``TtMlp2D`` restacked onto this axis mapping (#66248, #66244);
  * ``TtPrefillBlock`` per layer, built through ``mla_class(config)`` with ``config.mla_2d_tp`` (``ttMLA2DTP``) and
    ``Hy4TP32Config`` (``TtMlp2D`` dense layer 0 / 2D-TP shared expert, EP32 routed experts), iHC with its hidden on
    axis 1 (``hidden_tp`` 4);
  * every layer is built as a ``shared`` indexer layer and fed the golden top-k (layer 0's for layer 0, layer 1's for
    layers 1..3, as Hy4's indexer map reuses them);
  * the embedding is gathered on the host (BF16 table -> fp32), expanded to the 4 streams and uploaded.

Scored per iHC site against the fp32 CPU reference's streams (``HY4_IHC_REF_STREAMS``; see
``test_hy4_prefill_ihc_layers.py``): streams in, sublayer input y, streams out, and the update out - in.

Weights: what the cache lacks is read from the checkpoint once and cached under ``HY4_TP32_TTNN_CACHE`` (default: the
dev layout's cache dir; the 2D shards are named by their layout, everything else is shared). Env: ``HY4_HF_MODEL``, ``HY4_IHC_REF_STREAMS``, ``HY4_TP32_LAYERS`` (default 4), ``HY4_IHC_REPORT``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4TP32Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_hy4_prefill_ihc_layers import MOE_UPDATE_FLOOR, REF_DIR, _SiteRecorder
from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc

CHECKPOINT = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
GOLDEN = Path(os.environ.get("PREFILL_TRACE_DIR", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-v3-row2"))
# The 2D shards carry their layout in their names (mla2d_<layout>, *_8x4_h4ax1_i8ax0), and on this mesh mapping the
# norms, router and EP32 routed experts are the dev layout's own files, so the TP32 blocks share the dev cache dir.
CACHE = Path(
    os.environ.get(
        "HY4_TP32_TTNN_CACHE", "/mnt/tt-data/dwadhwani/Hy4-Preview-Cache/prefill-v1/hy4_preview_bh_32dev/8x4"
    )
)
NUM_LAYERS = int(os.environ.get("HY4_TP32_LAYERS", "4"))
S = 4096
SP_AXIS, TP_AXIS = 0, 1  # sequence + heads on the 8-chip axis, hidden on the 4-chip axis (#65884)
L1_SMALL = 1536  # the prefill's
PCC_FLOOR = 0.99

pytestmark = [
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
    pytest.mark.skipif(not (REF_DIR / "streams_embed.pt").is_file(), reason=f"no reference streams at {REF_DIR}"),
    pytest.mark.skipif(not (GOLDEN / "metadata.json").is_file(), reason=f"no Hy4 golden at {GOLDEN}"),
]


def _golden_topk(layer: int) -> torch.Tensor:
    from safetensors import safe_open

    src = layer if layer in (0, 1) else 1  # Hy4 indexer map: 0, 1 full; 2, 3, 4 reuse layer 1
    f = GOLDEN / "dsa" / f"dsa_topk_indices_layer_{src}" / f"rows_00000000_{S:08d}.safetensors"
    with safe_open(str(f), framework="pt") as h:
        return h.get_tensor(f"dsa_topk_indices_layer_{src}").long()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=moe_fabric_payload(Hy4TP32Config), l1_small_size=L1_SMALL),
            id="tp32-8x4-sp0-hidden1",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["hy4_preview"], indirect=True, ids=["hy4"])
@pytest.mark.timeout(0)
def test_hy4_prefill_ihc_tp32_blocks(variant, config_only, mesh_device, device_params, monkeypatch):
    from models.demos.deepseek_v3_d_p.tests.test_hy4_attention import host_indices_to_device
    from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG, ttMLA2DTP
    from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

    model_cfg = Hy4TP32Config
    config = copy.deepcopy(config_only)
    setattr(config, MLA_2D_TP_FLAG, True)
    config.indexer_types = ["shared"] * len(config.indexer_types)  # top-k from the golden (#65887)
    config.max_seq_len = S
    sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    assert (sp, tp) == (8, 4)
    topology = per_axis_topology(device_params["fabric_config"])
    experts_per_chip = model_cfg.NUM_ROUTED_EXPERTS // (sp * tp)

    ckpt = hc.Hy4Checkpoint(CHECKPOINT)
    CACHE.mkdir(parents=True, exist_ok=True)
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE

    # Only what this cache lacks for the TP32 blocks (the runner's own per-category check), from the checkpoint.
    missing = wc.missing_trunk_categories(
        CACHE,
        num_layers=NUM_LAYERS,
        first_layer_idx=0,
        experts_per_chip=experts_per_chip,
        model_cfg=model_cfg,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        is_first_rank=True,
        is_last_rank=False,
        mtp_levels=0,
        kv_only_last_layer=False,
        hf_config=config,
        tp_factor=tp,
    )
    logger.info(f"[tp32] cache {CACHE}: building {wc.format_missing(missing) if missing else 'nothing'}")
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=S,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=NUM_LAYERS,
        tp_axis=TP_AXIS,
    )
    blocks = []
    for i in range(NUM_LAYERS):
        cats = {c for c, layers in missing.items() if i in layers}
        logger.info(f"[tp32] layer {i}: {sorted(cats) if cats else 'cache'}")
        sd = hc.hy4_layer_state_dict(ckpt, config, model_cfg, i, cats) if cats else {}
        blocks.append(
            TtPrefillBlock(
                mesh_device,
                config,
                model_cfg,
                sd,
                layer_idx=i,
                seq_len=S,
                dispatch_buffer_capacity_factor=8,
                num_links=2,
                topology=topology,
                sp_axis=SP_AXIS,
                tp_axis=TP_AXIS,
                gate_fallback_mode=GateComputeMode.DEVICE_FP32,
                weight_cache_path=CACHE,
                is_chunked=True,
                layer_num=NUM_LAYERS,
                max_seq_len=S,
                routing_use_l1_small_for_semaphores=True,
                sparse_kv_cache_format=kvpe_cache.format,
            )
        )
        del sd
        assert isinstance(blocks[-1].mla, ttMLA2DTP) and blocks[-1].ihc and blocks[-1].ihc_attn.plan.hidden_tp == 4

    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=S, chunk_size_global=S)

    # Embedding on the host (#65892), expanded to the streams and packed per chip: [S/sp, n * hidden/tp].
    ids = torch.tensor(json.loads((GOLDEN / "metadata.json").read_text())["token_ids"][:S])
    emb = ckpt.get("model.embed_tokens.weight")[ids].to(torch.bfloat16).float()  # [S, C]
    n, C = model_cfg.HC_MULT, config.hidden_size
    streams = emb.unsqueeze(1).expand(S, n, C)
    packed = streams.reshape(S, n, tp, C // tp).permute(0, 2, 1, 3).reshape(1, 1, S, n * C)
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    x = ttnn.from_torch(
        packed,
        device=mesh_device,
        dtype=hy4_ihc.STREAM_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )

    ihc_weights = {i: hy4_ihc.read_layer_ihc(ckpt, i) for i in range(NUM_LAYERS)}
    sites = _SiteRecorder(monkeypatch, mesh_device, SP_AXIS, ihc_weights)
    for i, block in enumerate(blocks):
        out, _ = block(
            x,
            rope_tensors,
            kvpe_cache,
            cache_layer_idx=i,
            actual_start=0,
            actual_end=S,
            actual_isl=S,
            indexer_indices=host_indices_to_device(_golden_topk(i), mesh_device, SP_AXIS),
        )
        ttnn.deallocate(x)
        x = out
    ttnn.synchronize_device(mesh_device)

    logger.info(
        "Hy4 prefill blocks with iHC, TP32 layout (8x4 torus: sequence + heads on axis 0, hidden on axis 1), golden top-k:"
    )
    logger.info("site      streams_in  y          streams_out  update")
    for name, r in sites.rows.items():
        logger.info(
            f"{name:8s}  {r.get('in', float('nan')):.6f}  {r.get('y', float('nan')):.6f}  "
            f"{r.get('out', float('nan')):.6f}     {r.get('update', float('nan')):.6f}"
        )
    if os.environ.get("HY4_IHC_REPORT"):
        Path(os.environ["HY4_IHC_REPORT"]).write_text(
            json.dumps({"layout": "tp32-blocks", "sites": sites.rows}, indent=1)
        )
    for b in blocks:
        b.release_sub_device_managers()

    # The update of an MoE site is held to the MoE bar (#65923), as in test_hy4_prefill_ihc_layers.py.
    def floor(name, m):
        moe = name.endswith(".ffn") and int(name[1 : name.index(".")]) >= model_cfg.NUM_DENSE_LAYERS
        return MOE_UPDATE_FLOOR if (moe and m == "update") else PCC_FLOOR

    bad = [
        f"{k} {m}={v:.4f}"
        for k, r in sites.rows.items()
        for m, v in r.items()
        if m != "max_abs_out" and v < floor(k, m)
    ]
    assert not bad, f"below {PCC_FLOOR} (MoE update: {MOE_UPDATE_FLOOR}): {bad}"
