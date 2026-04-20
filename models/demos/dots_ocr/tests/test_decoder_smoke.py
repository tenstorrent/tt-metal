# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


@pytest.mark.skipif(os.environ.get("MESH_DEVICE") is None, reason="Requires TT device (set MESH_DEVICE)")
def test_dots_decoder_import_smoke():
    """
    Smoke test: decoder modules import and basic classes construct.

    Full e2e PCC will be added once weight mapping is validated end-to-end on
    single-chip (N150) and 2-chip (N300 / T3K 1x2 submesh) configurations.
    """
    from transformers import AutoConfig

    import ttnn
    from models.demos.dots_ocr.tt.mesh import open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs

    try:
        cfg = AutoConfig.from_pretrained(
            "rednote-hilab/dots.mocr",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError:
        cfg = AutoConfig.from_pretrained("rednote-hilab/dots.mocr", trust_remote_code=True)
    device = open_mesh_device()
    try:
        args = DotsModelArgs(device, hf_config=cfg, dummy_weights=True, max_batch_size=1, max_seq_len=2048)
        # ``Embedding`` reads ``tok_embeddings.weight`` from ``state_dict``; an empty dict raises KeyError.
        state_dict = args.load_state_dict()
        model = DotsTransformer(
            args,
            dtype=ttnn.bfloat16,
            mesh_device=device,
            state_dict=state_dict,
            weight_cache_path=None,
        )
        assert model.args.dim == cfg.hidden_size
    finally:
        ttnn.close_mesh_device(device)
