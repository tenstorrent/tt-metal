# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf worker for combine_fabric2d: one warm-up launch, then ITERATIONS more.

No correctness check here; `op_unit_tests/test_combine_fabric2d.py` covers that.

Environment:
    TT_DS_INPUT_LAYOUT     row_major (default) or tile. The model passes tile.
    TT_DS_CAPTURED_LAYER   an integer: use one recorded MoE layer's real routing instead of the
                           synthetic draw.
    TT_DS_CAPTURED_PATH    where that recording lives; defaults to the golden prefill cache.
    TT_DS_INPUT_DTYPE      bf16 (default) or bf8: the dispatched buffer's dtype. bf8 needs tile; the routed expert
                           hands combine bf8 tiles.
    TT_DS_COMBINE_WIRE     bf16 (default) or fp8: the tokens combine sends. fp8 has the untilizers quantize each token
                           in 1x128 blocks (fp8_scale_block=128) and needs tile. fp8-pre quantizes them before combine
                           instead, with per_token_cast_to_fp8 on the full grid, and combine moves the ROW_MAJOR fp8
                           rows as bytes; every launch runs both ops.
    TT_DS_MODEL            glm53 or kimi27: that model's emb_dim, routed experts and top-k, and an extra 8x4 mesh
                           opened at the payload the model's TtMoe opens it with (fabric2d-torus-xy-8x4-2link-model).

A harness that launches this worker must pass them through `extra_env`, not as a prefix on the
command: tracy misreads a leading KEY=VAL.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_combine_fabric2d import (
    PRODUCTION_ROUTING,
    _Fixture,
    quantize_before_combine,
)
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import extract_mesh_config, moe_fabric_payload_size

# Production prefill geometry: one 5120-token chunk over an 8-chip dispatch group.
CHUNK = 5 * 1024
DISPATCH_GROUP_SIZE = 8
SEQ_LEN_PER_CHIP = CHUNK // DISPATCH_GROUP_SIZE
EMB_DIM = 7 * 1024
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
_MODELS = {"glm53": GLM53Config, "kimi27": KimiK27Config}
_MODEL = os.environ.get("TT_DS_MODEL")
assert _MODEL in (None, *_MODELS), f"TT_DS_MODEL: expected one of {sorted(_MODELS)}, got {_MODEL!r}"
if _MODEL is not None:
    EMB_DIM = _MODELS[_MODEL].EMB_SIZE
    NUM_ROUTED_EXPERTS = _MODELS[_MODEL].NUM_ROUTED_EXPERTS
    NUM_EXPERTS_PER_TOK = _MODELS[_MODEL].NUM_EXPERTS_PER_TOKEN

# Launches after the warm-up one.
ITERATIONS = 10

_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-xy-8x4-2link"]
assert len(_MESH_CONFIGS) == 1, "Galaxy TorusXY config missing from ALL_MESH_CONFIGS"
if _MODEL is not None:
    _MESH_CONFIGS.append(
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload_size(_MODELS[_MODEL]),
                reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-torus-xy-8x4-2link-model",
        )
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# Kept finite: a hang leaves the eth links down, and bringing them back needs admin access.
@pytest.mark.timeout(900)
def test_combine_fabric2d_perf_worker(mesh_device, device_params, num_links):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"

    routing = "production"
    label = "synthetic"
    # Optionally, real routing from one recorded MoE layer: 5120 tokens split as 8 chips x 640. Every
    # dispatch group gets the same picks and keeps only the ones for its own experts.
    captured_layer = os.environ.get("TT_DS_CAPTURED_LAYER")
    if captured_layer is not None:
        from safetensors import safe_open

        path = os.environ.get(
            "TT_DS_CAPTURED_PATH",
            "/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_5120_nopad/expert_routing.safetensors",
        )
        with safe_open(path, "pt") as f:
            ids = f.get_tensor(f"expert_ids_layer_{int(captured_layer)}").to(torch.int64)
        routing = ids.view(H, SEQ_LEN_PER_CHIP, NUM_EXPERTS_PER_TOK).unsqueeze(0).expand(G, -1, -1, -1).clone()
        label = f"captured-L{int(captured_layer)}"

    # Sized to the draw: combine reads only the pages its runs cover, so spare capacity costs host time,
    # not device time. A recorded layer's hot experts would outgrow the fixed default.
    fx = _Fixture(
        mesh_device,
        H,
        G,
        emb_dim=EMB_DIM,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        topk=NUM_EXPERTS_PER_TOK,
        seed=42,
        routing=routing,
        capacity="exact",
    )
    realized = float(fx.kept.to(torch.float64).mean())

    layout = os.environ.get("TT_DS_INPUT_LAYOUT", "row_major")
    assert layout in ("row_major", "tile"), f"TT_DS_INPUT_LAYOUT: expected row_major or tile, got {layout!r}"
    tt_layout = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    input_dtype = os.environ.get("TT_DS_INPUT_DTYPE", "bf16")
    assert input_dtype in ("bf16", "bf8"), f"TT_DS_INPUT_DTYPE: expected bf16 or bf8, got {input_dtype!r}"
    wire = os.environ.get("TT_DS_COMBINE_WIRE", "bf16")
    assert wire in ("bf16", "fp8", "fp8-pre"), f"TT_DS_COMBINE_WIRE: expected bf16, fp8 or fp8-pre, got {wire!r}"
    assert layout == "tile" or (input_dtype == "bf16" and wire == "bf16"), "bf8 input and the fp8 wire need tile"
    tt_dtype = ttnn.bfloat8_b if input_dtype == "bf8" else ttnn.bfloat16
    fp8_scale_block = 128 if wire == "fp8" else 0

    def launch():
        if wire == "fp8-pre":
            rows = quantize_before_combine(fx.tt_payload(tt_layout, tt_dtype))
            out = fx.run(sp_axis, num_links, payload=rows)
            ttnn.deallocate(rows)
            return out
        return fx.run(sp_axis, num_links, layout=tt_layout, dtype=tt_dtype, fp8_scale_block=fp8_scale_block)

    logger.info(
        f"perf worker: mesh={tuple(mesh_device.shape)} seq={SEQ_LEN_PER_CHIP} emb={EMB_DIM} "
        f"experts={NUM_ROUTED_EXPERTS} topk={NUM_EXPERTS_PER_TOK} epc={fx.experts_per_chip} "
        f"capacity={fx.capacity} links={num_links} iters={ITERATIONS}"
    )
    logger.info(f"routing {label}: picks landing in this dispatch group {100 * realized:.1f}%")
    logger.info(f"input layout: {layout}, input dtype: {input_dtype}, wire: {wire}, model: {_MODEL or 'default'}")
    if captured_layer is None:
        # The draw must hit its target share, or the measured traffic silently isn't what we think.
        in_group_share = PRODUCTION_ROUTING[0]
        assert abs(realized - in_group_share) < 0.02, (
            f"the routing draw asked for {100 * in_group_share:.1f}% of picks in this dispatch group "
            f"and drew {100 * realized:.1f}%"
        )

    launch()  # warm-up; the gate sums it with the rest
    ttnn.synchronize_device(mesh_device)

    # Synced after every launch so launches never overlap; overlap is test_combine_fabric2d_back_to_back's
    # subject, not this one's.
    signpost("combine_fabric2d")
    for _ in range(ITERATIONS):
        launch()
        ttnn.synchronize_device(mesh_device)
    signpost("done")
