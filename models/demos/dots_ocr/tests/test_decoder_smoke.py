# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.mark.skipif(os.environ.get("MESH_DEVICE") is None, reason="Requires TT device (set MESH_DEVICE)")
def test_dots_decoder_import_smoke():
    """
    Smoke test: decoder modules import and basic classes construct.

    Full e2e PCC will be added once weight mapping is validated on WH LB.
    """
    from transformers import AutoConfig

    import ttnn
    from models.demos.dots_ocr.tt.model import DotsTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs

    cfg = AutoConfig.from_pretrained("rednote-hilab/dots.mocr", trust_remote_code=True)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = DotsModelArgs(device, hf_config=cfg, dummy_weights=True, max_batch_size=1, max_seq_len=2048)
        model = DotsTransformer(args, dtype=ttnn.bfloat16, mesh_device=device, state_dict={}, weight_cache_path=None)
        assert model.args.dim == cfg.hidden_size
    finally:
        ttnn.close_mesh_device(device)
