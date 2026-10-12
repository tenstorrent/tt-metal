# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill runner's in-process ViT (``V41PrefillRuntime._init_vit``, ``V41_PREFILL_VIT=1``): the ttnn ViT + aligner on a
(1, 1) SUBMESH of the runner's FABRIC_2D 8 x 4 mesh must give the same span rows as the stand-alone one-chip front end
(``scripts/v41_vit_rows.py`` + tt-blaze ``prof/vit_prompt.py``). ``V41_VIT_SUB_REF``: that front end's ``<prefix>.rows.pt``."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v41.config import DEFAULT_MODEL_DIR
from models.demos.deepseek_v3_d_p.tt.v41.vision import V41Vision, load_vision_state
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

IMAGE = os.environ.get("V41_VIT_IMAGE", "/mnt/tt-data/sdawle/dsv41_vit/images/T5_shapes_wide.png")
REF = os.environ.get("V41_VIT_SUB_REF", "/mnt/tt-data/sdawle/dsv41_vit/qa/T5_dev.rows.pt")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            id="fabric2d-mesh-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_v41_vision_on_submesh(mesh_device, device_params):
    sys.path.insert(0, os.path.join(DEFAULT_MODEL_DIR, "inference"))
    import image_processor as ip  # noqa: E402

    args = types.SimpleNamespace(**json.load(open(os.path.join(DEFAULT_MODEL_DIR, "inference", "config.json"))))
    state = load_vision_state(checkpoint(DEFAULT_MODEL_DIR))
    if (
        os.environ.get("V41_VIT_ON_SUBMESH", "0") == "1"
    ):  # the (1, 1) submesh: bit-exact alone, but hung the runner (DS41F-0049)
        sub = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        vit = V41Vision(sub, state)
    else:  # replicated on the whole mesh (the runner's mode)
        vit = V41Vision(mesh_device, state)
    patches, n_h, n_w, n_llm_h, n_llm_w = ip.load_image({"url": IMAGE}, args)
    vit.aligner(vit.vit(patches, n_h, n_w), n_h, n_w)  # compile
    rows = vit.aligner(vit.vit(patches, n_h, n_w), n_h, n_w)
    d = {k: state[k].to(torch.bfloat16).reshape(-1) for k in ("image_start", "image_end", "image_newline")}
    out, r = [], 0
    for t in ip.image_token_types(n_llm_h, n_llm_w).tolist():
        if t == ip.IMAGE_START:
            out.append(d["image_start"])
        elif t == ip.IMAGE_END:
            out.append(d["image_end"])
        elif t == ip.IMAGE_NEW_LINE:
            out.append(d["image_newline"])
        else:
            out.append(rows[r])
            r += 1
    span = torch.stack(out)
    ref = torch.load(REF)["rows"]
    eq = torch.equal(span, ref.to(span.dtype))
    pcc = float(torch.corrcoef(torch.stack([span.float().flatten(), ref.float().flatten()]))[0, 1])
    logger.info(
        f"[vit submesh] {IMAGE}: span {tuple(span.shape)} vs one-chip front end {tuple(ref.shape)}: bit-equal {eq}, PCC {pcc:.6f}"
    )
    assert span.shape == ref.shape and pcc > 0.9999, (span.shape, ref.shape, pcc)
    # the mesh must still run a CCL-free mesh op after the ViT (the runner's next chunk hung after a submesh ViT)
    t = ttnn.from_torch(
        torch.ones(1, 1, 32, 32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)
    assert float(ttnn.to_torch(ttnn.get_device_tensors(ttnn.add(t, t))[31]).max()) == 2.0
