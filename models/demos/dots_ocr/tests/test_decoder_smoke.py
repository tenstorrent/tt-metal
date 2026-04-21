# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_dots_decoder_import_smoke():
    """
    Smoke test: decoder modules import and ``DotsTransformer`` constructs with real checkpoint tensors.

    For **prefill PCC** against HF logits on the same ``DotsTransformer`` stack, use
    ``test_dots_decoder_prefill_pcc`` or ``test_text_prefill_pcc.py::test_text_only_prefill_pcc_gt_0_99``.
    """
    from transformers import AutoConfig

    import ttnn
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
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
    try:
        device = open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    except Exception as e:
        pytest.skip(f"Requires TT device runtime (could not open mesh device): {e!r}")
    try:
        args = DotsModelArgs(device, hf_config=cfg, max_batch_size=1, max_seq_len=2048)
        try:
            state_dict = args.load_real_state_dict(qkv_permute=True)
        except Exception as exc:
            pytest.skip(f"Real Dots weights required for smoke test: {exc}")
        model = DotsTransformer(
            args,
            dtype=ttnn.bfloat16,
            mesh_device=device,
            state_dict=state_dict,
            weight_cache_path=None,
        )
        assert model.args.dim == cfg.hidden_size
    finally:
        close_dots_mesh_device(device)


def test_dots_decoder_prefill_pcc(tmp_path):
    """
    PCC: ``DotsTransformer`` text-only prefill vs HF reference last-token logits (``comp_pcc``).

    Shared implementation with ``test_text_only_prefill_pcc_gt_0_99`` in ``test_text_prefill_pcc.py``.
    """
    from models.demos.dots_ocr.tests.test_text_prefill_pcc import run_text_decoder_prefill_pcc_check

    run_text_decoder_prefill_pcc_check(tmp_path)
