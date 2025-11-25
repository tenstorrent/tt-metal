# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

torch = pytest.importorskip("torch")

from models.experimental.dpt_large.config import DPTLargeConfig
from models.experimental.dpt_large.reassembly import DPTReassembly
from models.experimental.dpt_large.vit_backbone import ViTBackboneOutputs


def test_reassembly_shapes():
    cfg = DPTLargeConfig(image_size=128, hidden_size=64, output_layers=[1, 2, 3, 4])
    reassembly = DPTReassembly(config=cfg, proj_channels=32)
    dummy_feats = {}
    for idx in cfg.output_layers:
        dummy_feats[idx + 1] = torch.randn(
            2, cfg.hidden_size, cfg.image_size // cfg.patch_size, cfg.image_size // cfg.patch_size
        )
    outputs = reassembly(ViTBackboneOutputs(features=dummy_feats))
    assert len(outputs) == len(cfg.output_layers)
    for out in outputs:
        assert out.shape[1] == 32
