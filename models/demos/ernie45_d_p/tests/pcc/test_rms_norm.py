# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.3: RMSNorm (input, post-attention, final) vs golden, chunk 0 of 2k->2k."""

import pytest

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.ops import TtRMSNorm

TASK = "P2.3"


@mesh_1x4
@pytest.mark.parametrize("layer", [0, 13, 27])
def test_layer_norms(mesh_device, cfg, loader, golden_2k, record, layer):
    g = golden_2k.layer(0, layer)
    p = f"model.layers.{layer}."
    for which, inp, out, wname in [
        ("attn_norm", "in", "attn_norm", "input_layernorm.weight"),
        ("ffn_norm", "h_mid", "ffn_norm", "post_attention_layernorm.weight"),
    ]:
        norm = TtRMSNorm(mesh_device, loader.get(p + wname).float(), cfg.rms_norm_eps, f"L{layer}.{which}")
        y = replicated_to_torch(norm(to_mesh_activation(mesh_device, g[inp].float())))[0, 0]
        record(f"pcc_{which}_L{layer:02d}", y, g[out], 0.999)
    record.check()


@mesh_1x4
def test_final_norm(mesh_device, cfg, loader, golden_2k, record):
    x = golden_2k.layer(0, cfg.num_hidden_layers - 1)["out"]
    norm = TtRMSNorm(mesh_device, loader.get("model.norm.weight").float(), cfg.rms_norm_eps, "final_norm")
    y = replicated_to_torch(norm(to_mesh_activation(mesh_device, x.float())))[0, 0]
    record("pcc_final_norm", y, golden_2k.model(0)["final_norm"], 0.999)
    record.check()
