# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-shot PCC test for the SeamlessM4Tv2 T2U module at its maximum sequence length.

T2U design max = ``t2u_max_position_embeddings = 4096`` (HF). The single test below runs the full
encoder + decoder + ``lm_head`` at ``encoder_seq_len = 4096`` (one character per encoder frame so
upsampled char / unit sequences also sit at 4096), exercising the chunked DRAM matmul
(``_T2U_LONG_SDPA_TP_THRESHOLD`` = 128), the chunked SDPA decode-side attention, and the
``_hard_upsample_matmul`` chunked path on both char and unit upsamplers.

Real weights only — if ``huggingface_hub`` is missing or the snapshot download fails the test is
skipped.

PCC threshold: we *target* ≥ 0.99 across the chain, but at the full 4096-position extent the
combined chunked-matmul / SDPA / conv halo error accumulates across encoder + decoder; with the
current Blackhole kernels logits PCC stabilises just under 0.99 (~0.973). We keep the assertion at
0.99 — if it drops below, a regression has occurred; if it remains slightly under in steady state
this comment is the place to relax it. The end-to-end demo (``test_seamless_m4t_v2_model``)
verifies HF-matching outputs at realistic sequence lengths.
"""

import pytest
import torch
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_logits_and_padding,
    hf_discrete_duration_counts_batch1,
    load_pretrained_text_to_unit,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_to_unit_condgen_parameters
from models.experimental.seamless_m4t_v2_large.tests.pcc.prof_capture_limits import TEXT_TO_UNIT_ENCODER_SEQ
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)

PCC_THRESHOLD = 0.99
MAX_ENCODER_SEQ = 4096  # HF ``t2u_max_position_embeddings``
PROF_CAPTURE_ENCODER_SEQ = TEXT_TO_UNIT_ENCODER_SEQ


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_to_unit_max_seq_pcc(mesh_device, device_params, reset_seeds):
    """T2U PCC at HF ``t2u_max_position_embeddings`` = 4096 (encoder + decoder + ``lm_head``).

    Uses ``chars_per_encoder_step=1`` so the upsampled char / unit lengths also sit at the maximum
    extent, exercising every long-seq T2U code path in one go.
    """
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    with mesh_default_device(mesh_device):
        torch.manual_seed(1)
        t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

        batch = 1
        encoder_seq_len = MAX_ENCODER_SEQ
        inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
            cfg,
            batch=batch,
            encoder_seq_len=encoder_seq_len,
            chars_per_encoder_step=1,
            seed=1,
            dtype=torch.bfloat16,
        )
        dev = next(t2u.parameters()).device
        char_count_per_id = char_count_per_id.to(dev)
        mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        ref_logits, ref_padding_mask = forward_t2u_logits_and_padding(
            t2u, inputs_embeds, attention_mask, char_input_ids, char_count_per_id
        )
        ref_logits = ref_logits.to(torch.bfloat16).cpu()
        ref_padding_mask = ref_padding_mask.to(torch.float32).cpu()
        ref_durs = hf_discrete_duration_counts_batch1(
            t2u, inputs_embeds.to(dev), attention_mask.to(dev), char_input_ids.to(dev), char_count_per_id
        )
        assert sum(ref_durs) == int(ref_logits.shape[1]), "HF duration sum must match HF logits sequence length."

        params = create_text_to_unit_condgen_parameters(t2u, device=mesh_device)
        tt_full = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
            mesh_device,
            params,
            layer_norm_eps=cfg.layer_norm_eps,
            encoder_layers=cfg.encoder_layers,
            encoder_attention_heads=cfg.encoder_attention_heads,
            decoder_layers=cfg.decoder_layers,
            decoder_attention_heads=cfg.decoder_attention_heads,
            hidden_size=cfg.hidden_size,
            pad_token_id=cfg.pad_token_id,
            variance_predictor_embed_dim=cfg.variance_predictor_embed_dim,
            variance_predictor_hidden_dim=cfg.variance_predictor_hidden_dim,
            variance_predictor_kernel_size=cfg.variance_predictor_kernel_size,
        )

        inputs_embeds_tt = from_torch_bfloat16_tile(mesh_device, inputs_embeds)
        attn_tt = from_torch_bfloat16_tile(mesh_device, mask_4d)
        char_ids_tt = from_torch_uint32_rm(mesh_device, char_input_ids)

        logits_tt, pad_tt = tt_full.forward(
            inputs_embeds_tt,
            attn_tt,
            char_ids_tt,
            char_count_per_id.squeeze(0).tolist(),
            reference_discrete_durations=ref_durs,
        )

        Vr = int(ref_logits.shape[2])
        Sr = int(ref_logits.shape[1])
        flat_logits = to_torch_replicated_first_shard(logits_tt).to(torch.bfloat16).contiguous().reshape(-1)
        Sp = flat_logits.numel() // Vr
        tt_logits_cpu = flat_logits.reshape(1, Sp, Vr)[:, :Sr, :Vr].contiguous()

        flat_pad = to_torch_replicated_first_shard(pad_tt).to(torch.float32).contiguous().reshape(-1)
        Spr = int(ref_padding_mask.shape[1])
        tt_pad_cpu = flat_pad.reshape(1, -1)[:, :Spr].contiguous()

        assert tt_logits_cpu.shape == ref_logits.shape
        assert tt_pad_cpu.shape == ref_padding_mask.shape

        ok_logits, msg_logits = check_with_pcc(ref_logits, tt_logits_cpu, pcc=PCC_THRESHOLD)
        ok_pad, msg_pad = check_with_pcc(ref_padding_mask, tt_pad_cpu, pcc=PCC_THRESHOLD)
        logger.info(
            f"SeamlessM4Tv2 T2U PCC @ encoder_seq={encoder_seq_len}: "
            f"logits={msg_logits} padding={msg_pad} (threshold {PCC_THRESHOLD})"
        )
        assert ok_logits, msg_logits
        assert ok_pad, msg_pad


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_to_unit_prof_capture_seq_pcc(mesh_device, device_params, reset_seeds):
    """T2U PCC ≥ 0.99 at the tracy-safe encoder seq (``PROF_CAPTURE_ENCODER_SEQ`` = 512).

    Highest power-of-two encoder length where ``python3 -m tracy -r -v`` and PCC both pass on
    BH 1×4. ``encoder_seq=1024`` hits L1; full HF extent (4096) breaks PCC and capture.
    """
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    with mesh_default_device(mesh_device):
        torch.manual_seed(1)
        t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

        batch = 1
        encoder_seq_len = PROF_CAPTURE_ENCODER_SEQ
        inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
            cfg,
            batch=batch,
            encoder_seq_len=encoder_seq_len,
            chars_per_encoder_step=1,
            seed=1,
            dtype=torch.bfloat16,
        )
        dev = next(t2u.parameters()).device
        char_count_per_id = char_count_per_id.to(dev)
        mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        ref_logits, ref_padding_mask = forward_t2u_logits_and_padding(
            t2u, inputs_embeds, attention_mask, char_input_ids, char_count_per_id
        )
        ref_logits = ref_logits.to(torch.bfloat16).cpu()
        ref_padding_mask = ref_padding_mask.to(torch.float32).cpu()
        ref_durs = hf_discrete_duration_counts_batch1(
            t2u, inputs_embeds.to(dev), attention_mask.to(dev), char_input_ids.to(dev), char_count_per_id
        )
        assert sum(ref_durs) == int(ref_logits.shape[1]), "HF duration sum must match HF logits sequence length."

        params = create_text_to_unit_condgen_parameters(t2u, device=mesh_device)
        tt_full = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
            mesh_device,
            params,
            layer_norm_eps=cfg.layer_norm_eps,
            encoder_layers=cfg.encoder_layers,
            encoder_attention_heads=cfg.encoder_attention_heads,
            decoder_layers=cfg.decoder_layers,
            decoder_attention_heads=cfg.decoder_attention_heads,
            hidden_size=cfg.hidden_size,
            pad_token_id=cfg.pad_token_id,
            variance_predictor_embed_dim=cfg.variance_predictor_embed_dim,
            variance_predictor_hidden_dim=cfg.variance_predictor_hidden_dim,
            variance_predictor_kernel_size=cfg.variance_predictor_kernel_size,
        )

        inputs_embeds_tt = from_torch_bfloat16_tile(mesh_device, inputs_embeds)
        attn_tt = from_torch_bfloat16_tile(mesh_device, mask_4d)
        char_ids_tt = from_torch_uint32_rm(mesh_device, char_input_ids)

        logits_tt, pad_tt = tt_full.forward(
            inputs_embeds_tt,
            attn_tt,
            char_ids_tt,
            char_count_per_id.squeeze(0).tolist(),
            reference_discrete_durations=ref_durs,
        )

        Vr = int(ref_logits.shape[2])
        Sr = int(ref_logits.shape[1])
        flat_logits = to_torch_replicated_first_shard(logits_tt).to(torch.bfloat16).contiguous().reshape(-1)
        Sp = flat_logits.numel() // Vr
        tt_logits_cpu = flat_logits.reshape(1, Sp, Vr)[:, :Sr, :Vr].contiguous()

        flat_pad = to_torch_replicated_first_shard(pad_tt).to(torch.float32).contiguous().reshape(-1)
        Spr = int(ref_padding_mask.shape[1])
        tt_pad_cpu = flat_pad.reshape(1, -1)[:, :Spr].contiguous()

        assert tt_logits_cpu.shape == ref_logits.shape
        assert tt_pad_cpu.shape == ref_padding_mask.shape

        ok_logits, msg_logits = check_with_pcc(ref_logits, tt_logits_cpu, pcc=PCC_THRESHOLD)
        ok_pad, msg_pad = check_with_pcc(ref_padding_mask, tt_pad_cpu, pcc=PCC_THRESHOLD)
        logger.info(
            f"SeamlessM4Tv2 T2U prof-capture PCC @ encoder_seq={encoder_seq_len}: "
            f"logits={msg_logits} padding={msg_pad} (threshold {PCC_THRESHOLD})"
        )
        assert ok_logits, msg_logits
        assert ok_pad, msg_pad
