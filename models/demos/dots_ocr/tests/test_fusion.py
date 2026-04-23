# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.dots_ocr.reference.fusion import merge_vision_tokens
from models.demos.dots_ocr.reference.pcc import comp_pcc
from models.demos.dots_ocr.tt.fusion import merge_vision_tokens_host


def test_merge_vision_tokens_replaces_positions():
    torch.manual_seed(0)
    B, S, D = 2, 8, 16
    image_token_id = 99
    input_ids = torch.randint(0, 200, (B, S))
    # Force exactly 3 image tokens in known order
    input_ids[0, 1] = image_token_id
    input_ids[0, 5] = image_token_id
    input_ids[1, 0] = image_token_id

    input_embeds = torch.randn(B, S, D)
    image_embeds = torch.randn(3, D)

    out = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)

    # Non-image positions unchanged
    mask = input_ids == image_token_id
    assert torch.allclose(out[~mask], input_embeds[~mask])

    # Image positions match image_embeds in row-major order
    gathered = out[mask]
    assert torch.allclose(gathered, image_embeds)


def test_merge_vision_tokens_mismatch_raises():
    B, S, D = 1, 4, 8
    image_token_id = 7
    input_ids = torch.tensor([[image_token_id, 1, 2, 3]])
    input_embeds = torch.randn(B, S, D)
    image_embeds = torch.randn(2, D)
    with pytest.raises(ValueError):
        _ = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)


def test_fusion_layer_pcc_matches_reference():
    """
    PCC check for the fusion layer (vision-token scatter into text embeddings).

    Today the TT pipeline uses a host-side fusion wrapper (`merge_vision_tokens_host`) that delegates to the
    same reference implementation; this test protects against future divergence.
    """
    torch.manual_seed(0)
    B, S, D = 2, 64, 128
    image_token_id = 151643

    input_ids = torch.randint(0, 200000, (B, S))
    # Plant a deterministic set of image-token positions.
    input_ids[0, 3] = image_token_id
    input_ids[0, 17] = image_token_id
    input_ids[1, 5] = image_token_id
    input_ids[1, 63] = image_token_id

    input_embeds = torch.randn(B, S, D, dtype=torch.bfloat16)
    image_embeds = torch.randn(int((input_ids == image_token_id).sum().item()), D, dtype=torch.bfloat16)

    ref = merge_vision_tokens(input_ids, input_embeds, image_embeds, image_token_id=image_token_id)
    tt = merge_vision_tokens_host(
        input_ids=input_ids,
        input_embeds=input_embeds,
        image_embeds=image_embeds,
        image_token_id=image_token_id,
    )

    assert ref.shape == tt.shape
    pcc = comp_pcc(ref, tt)
    print(f"Fusion PCC: {pcc:.6f}")
    assert pcc > 0.9999


def test_ttnn_merge_and_prefill_matches_host_torch_path(tmp_path):
    """
    With hardware: ttnn ``merge_vision_tokens_ttnn`` + ``preprocess_inputs_prefill_ttnn`` (then read back)
    should match the host torch :func:`merge_vision_tokens` + :func:`preprocess_inputs_prefill` result.
    """
    try:
        import ttnn
    except Exception:
        pytest.skip("ttnn not available")

    from models.demos.dots_ocr.demo.demo import _load_dots_ttnn_state_dict
    from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
    from models.demos.dots_ocr.reference.model import DotsOCRReference
    from models.demos.dots_ocr.tt.common import (
        fused_ttnn_embeddings_to_torch,
        merge_vision_tokens,
        merge_vision_tokens_ttnn,
        pad_embedding_ttnn,
        pad_embedding_ttnn_tensor,
        preprocess_inputs_prefill,
        preprocess_inputs_prefill_ttnn,
        text_embeds_from_ttnn_embedding,
        text_embeds_from_ttnn_embedding_ttnn,
        ttnn_fused_batch_to_user_list,
    )
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, get_max_seq_len_cap, open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs

    mesh = None
    try:
        mesh = open_mesh_device()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"TT device: {exc}")

    try:
        ref = DotsOCRReference(HFLoadSpec(model_id="rednote-hilab/dots.mocr", dtype=torch.bfloat16))
        model_args = DotsModelArgs(
            mesh_device=mesh, hf_config=ref.model.config, max_batch_size=1, max_seq_len=get_max_seq_len_cap() or 4096
        )
        model_args.lm_head_dtype = ttnn.bfloat16
        sd = _load_dots_ttnn_state_dict(model_args, text_qkv_permute=True)
        tt = DotsTransformer(
            args=model_args,
            dtype=ttnn.bfloat16,
            mesh_device=mesh,
            state_dict=sd,
            weight_cache_path=tmp_path / "w",
            paged_attention_config=None,
        )
        B, S, D = 1, 32, int(model_args.dim)
        input_ids = torch.randint(1, 500, (B, S))
        n_img = 5
        input_ids[0, 3:8] = int(ref.model.config.image_token_id)
        im = torch.randn(n_img, D, dtype=torch.bfloat16)

        text_t = text_embeds_from_ttnn_embedding(tt, input_ids)
        emb_h = merge_vision_tokens(input_ids, text_t, im, ref.model.config)
        pad = pad_embedding_ttnn(tt, int(ref.tokenizer.pad_token_id or 0))
        pre_h, _, _ = preprocess_inputs_prefill([emb_h[0]], model_args, torch.ones(B, S, dtype=torch.int32), pad)

        text_u = text_embeds_from_ttnn_embedding_ttnn(tt, input_ids)
        fused_u = merge_vision_tokens_ttnn(input_ids, text_u, im, ref.model.config, mesh_device=mesh)
        pre_u, _, _ = preprocess_inputs_prefill_ttnn(
            ttnn_fused_batch_to_user_list(fused_u),
            model_args,
            torch.ones(B, S, dtype=torch.int32),
            pad_embedding_ttnn_tensor(tt, int(ref.tokenizer.pad_token_id or 0)),
        )
        pre_u_torch = fused_ttnn_embeddings_to_torch(pre_u, mesh)
        assert pre_h.shape == pre_u_torch.shape
        pcc = comp_pcc(pre_h, pre_u_torch)
        assert pcc > 0.98, f"device fusion prefill mismatch pcc={pcc}"
    finally:
        try:
            close_dots_mesh_device(mesh)
        except Exception:
            pass
