# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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
    MESH_DEVICE_PARAMETRIZE_T2U_TRACE,
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_to_unit_condgen_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
    make_t2u_trace_prealloc_tensors,
)

PCC_THRESHOLD = 0.99
# Full T2U at ``encoder_seq=4096`` accumulates chunked-matmul / SDPA / conv halo error across
# encoder + decoder; logits PCC stays ~0.973 with current Blackhole kernels (see max test).
PCC_THRESHOLD_MAX_ENCODER = 0.97
SHORT_ENCODER_SEQ = 32
MAX_ENCODER_SEQ = 4096  # HF ``t2u_max_position_embeddings``


def _run_text_to_unit_pcc(
    device,
    *,
    encoder_seq_len: int,
    chars_per_encoder_step: int = 2,
    pcc_threshold: float = PCC_THRESHOLD,
) -> None:
    """
    Shared PCC body. Compares TT vs HF for the full T2U module (encoder + decoder + ``lm_head``).
    Mesh-safe readbacks go through ``to_torch_replicated_first_shard``.
    """
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

    batch = 1
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        cfg,
        batch=batch,
        encoder_seq_len=encoder_seq_len,
        chars_per_encoder_step=chars_per_encoder_step,
        seed=1,
        dtype=torch.bfloat16,
    )
    dev = next(t2u.parameters()).device
    char_count_per_id = char_count_per_id.to(dev)

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

    ref_logits, ref_padding_mask = forward_t2u_logits_and_padding(
        t2u,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id,
    )
    ref_logits = ref_logits.to(torch.bfloat16).cpu()
    ref_padding_mask = ref_padding_mask.to(torch.float32).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        t2u,
        inputs_embeds.to(dev),
        attention_mask.to(dev),
        char_input_ids.to(dev),
        char_count_per_id,
    )
    assert sum(ref_durs) == int(ref_logits.shape[1]), "HF duration sum must match HF logits sequence length."

    params = create_text_to_unit_condgen_parameters(t2u, device=device)
    tt_full = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
        device,
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

    inputs_embeds_tt = from_torch_bfloat16_tile(device, inputs_embeds)
    attn_tt = from_torch_bfloat16_tile(device, mask_4d)
    char_ids_tt = from_torch_uint32_rm(device, char_input_ids)

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

    assert (
        tt_logits_cpu.shape == ref_logits.shape
    ), f"TT logits {tuple(tt_logits_cpu.shape)} vs HF {tuple(ref_logits.shape)} — unit length must match."
    assert (
        tt_pad_cpu.shape == ref_padding_mask.shape
    ), f"TT padding_mask {tuple(tt_pad_cpu.shape)} vs HF {tuple(ref_padding_mask.shape)}."

    ok_logits, msg_logits = check_with_pcc(ref_logits, tt_logits_cpu, pcc=pcc_threshold)
    ok_pad, msg_pad = check_with_pcc(ref_padding_mask, tt_pad_cpu, pcc=pcc_threshold)
    logger.info(
        f"SeamlessM4Tv2 text-to-unit PCC (encoder_seq={encoder_seq_len}): "
        f"logits={msg_logits} padding={msg_pad} (threshold {pcc_threshold})"
    )
    assert ok_logits, msg_logits
    assert ok_pad, msg_pad


def _run_text_to_unit_trace_pcc(
    device,
    *,
    encoder_seq_len: int,
    chars_per_encoder_step: int = 2,
    pcc_threshold: float = PCC_THRESHOLD,
) -> None:
    """PCC for T2U forward via Metal trace replay (2CQ device params, execute on CQ0)."""
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

    batch = 1
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        cfg,
        batch=batch,
        encoder_seq_len=encoder_seq_len,
        chars_per_encoder_step=chars_per_encoder_step,
        seed=1,
        dtype=torch.bfloat16,
    )
    dev = next(t2u.parameters()).device
    char_count_per_id = char_count_per_id.to(dev)

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

    ref_logits, ref_padding_mask = forward_t2u_logits_and_padding(
        t2u,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id,
    )
    ref_logits = ref_logits.to(torch.bfloat16).cpu()
    ref_padding_mask = ref_padding_mask.to(torch.float32).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        t2u,
        inputs_embeds.to(dev),
        attention_mask.to(dev),
        char_input_ids.to(dev),
        char_count_per_id,
    )
    assert sum(ref_durs) == int(ref_logits.shape[1]), "HF duration sum must match HF logits sequence length."

    params = create_text_to_unit_condgen_parameters(t2u, device=device)
    tt_full = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
        device,
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

    inputs_embeds_tt = from_torch_bfloat16_tile(device, inputs_embeds)
    attn_tt = from_torch_bfloat16_tile(device, mask_4d)
    char_ids_tt = from_torch_uint32_rm(device, char_input_ids)

    cc_list = char_count_per_id.squeeze(0).tolist()
    char_w = int(char_input_ids.shape[1])
    char_inc, char_prev = tt_full._cached_repeat_cumsums(cc_list)
    unit_inc, unit_prev = tt_full._cached_repeat_cumsums(ref_durs)
    hard_cums = make_t2u_trace_prealloc_tensors(
        device,
        pad_token_id=int(cfg.pad_token_id),
        hidden_size=int(cfg.hidden_size),
        char_w=char_w,
        cc_list=cc_list,
        ref_durs=ref_durs,
        char_inc=char_inc,
        char_prev=char_prev,
        unit_inc=unit_inc,
        unit_prev=unit_prev,
    )

    try:
        tt_full.capture_forward_trace(
            inputs_embeds_tt,
            attn_tt,
            char_ids_tt,
            cc_list,
            reference_discrete_durations=ref_durs,
            hard_upsample_cums=hard_cums,
        )
        logits_tt, pad_tt = tt_full.execute_forward_trace()
    finally:
        tt_full.release_forward_trace()

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

    ok_logits, msg_logits = check_with_pcc(ref_logits, tt_logits_cpu, pcc=pcc_threshold)
    ok_pad, msg_pad = check_with_pcc(ref_padding_mask, tt_pad_cpu, pcc=pcc_threshold)
    logger.info(
        f"SeamlessM4Tv2 text-to-unit trace PCC (encoder_seq={encoder_seq_len}): "
        f"logits={msg_logits} padding={msg_pad} (threshold {pcc_threshold})"
    )
    assert ok_logits, msg_logits
    assert ok_pad, msg_pad


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_to_unit_short_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ``encoder_seq_len=32`` — default 1D matmul / short hard-upsampling path."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_text_to_unit_pcc(mesh_device, encoder_seq_len=SHORT_ENCODER_SEQ)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_to_unit_max_encoder_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC at HF ``t2u_max_position_embeddings = 4096``.

    Exercises the chunked ``_hard_upsample_matmul`` path (``sum_r > MATMUL_1D_SEQ_THRESHOLD``) on both
    char and unit upsamplers. Uses one character per encoder frame so upsampled lengths stay at 4096.
    """
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_text_to_unit_pcc(
            mesh_device,
            encoder_seq_len=MAX_ENCODER_SEQ,
            chars_per_encoder_step=1,
            pcc_threshold=PCC_THRESHOLD_MAX_ENCODER,
        )


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_T2U_TRACE, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_text_to_unit_short_seq_trace_pcc(mesh_device, device_params, reset_seeds):
    """T2U Metal trace replay PCC at ``encoder_seq_len=32`` (2CQ + trace region device params)."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_text_to_unit_trace_pcc(mesh_device, encoder_seq_len=SHORT_ENCODER_SEQ)
