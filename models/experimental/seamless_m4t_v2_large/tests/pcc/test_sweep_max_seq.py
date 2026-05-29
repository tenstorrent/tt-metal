# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Empirical sweep to find the actual maximum supported seq length per submodule.

Runs the speech encoder / text decoder at a range of seq lengths in one pytest
invocation each, logs (seq, PCC, status) tuples. Used to set ``MAX_SEQ`` in
``test_speech_encoder.py`` and ``test_text_decoder.py`` to the *measured* ceiling
rather than a guess. Not a regression test — meant to be run manually when
implementation/L1 behavior changes.

Run:

    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_sweep_max_seq.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import create_position_ids_from_input_ids

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_speech_encoder import (
    forward_torch_speech_encoder,
    load_pretrained_speech_encoder,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import (
    forward_torch_reference,
    load_pretrained_text_decoder,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import (
    create_speech_encoder_parameters,
    create_text_decoder_parameters,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import TTSeamlessM4Tv2Decoder

PCC_THRESHOLD = 0.99


def _weights_dir_or_skip():
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"weights unavailable: {e}")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_sweep_speech_encoder_max_seq(mesh_device, device_params, reset_seeds):
    """Sweep speech encoder seq lengths to find the L1 ceiling and PCC ceiling."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    # Cover the gap between known-working (2125) and the previous test target (4096).
    seqs = [2125, 2400, 2700, 3000, 3300, 3600, 4096]
    results = []

    with mesh_default_device(mesh_device):
        torch.manual_seed(0)
        speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)
        params = create_speech_encoder_parameters(speech_enc, device=mesh_device)
        tt_model = TTSeamlessM4Tv2SpeechEncoder(
            mesh_device,
            params,
            hidden_size=cfg.hidden_size,
            feature_projection_input_dim=cfg.feature_projection_input_dim,
            speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
            speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
            speech_encoder_layers=cfg.speech_encoder_layers,
            layer_norm_eps=cfg.layer_norm_eps,
            speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
            speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
            matmul_token_rows=64,
        )

        for seq in seqs:
            try:
                batch = 1
                n_mels = cfg.feature_projection_input_dim
                input_features = torch.randn(batch, seq, n_mels, dtype=torch.bfloat16)
                attention_mask = torch.ones(batch, seq, dtype=torch.long)
                ref = forward_torch_speech_encoder(speech_enc, input_features, attention_mask).to(torch.bfloat16)
                tt_x = from_torch_bfloat16_tile(mesh_device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
                m1 = from_torch_bfloat16_tile(mesh_device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
                tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
                tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)
                ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
                results.append((seq, "PCC", msg, ok))
                logger.info(f"[speech_encoder sweep] seq={seq}: {msg} -> {'PASS' if ok else 'FAIL'}")
                ttnn.deallocate(tt_out)
            except Exception as e:
                results.append((seq, "EXC", str(e)[:100], False))
                logger.warning(f"[speech_encoder sweep] seq={seq}: EXCEPTION {str(e)[:100]} -> FAIL")
                # Clear program cache after an exception so following seq doesn't inherit bad state.
                try:
                    mesh_device.clear_program_cache()
                except Exception:
                    pass

    print("\n=== Speech encoder sweep results ===")
    print(f"{'seq':>6}  {'status':<6}  {'PCC / error':<60}")
    for seq, kind, msg, ok in results:
        print(f"{seq:>6}  {'PASS' if ok else 'FAIL':<6}  {msg}")
    print("=" * 80)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_sweep_text_decoder_max_seq(mesh_device, device_params, reset_seeds):
    """Sweep text decoder seq lengths + several seeds to find the joint L1 + PCC ceiling."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    # seq levels to probe — small to large to catch bf16 numerical drift and L1 overflow
    # separately. seed search: each seed gives a different activation geometry through the 24
    # decoder layers — some are stable to longer seq than others.
    sweep = [
        (32, [0, 1, 2, 3]),
        (64, [0, 1, 2, 3]),
        (128, [0, 1, 2, 3]),
        (256, [0, 1, 2, 3]),
        (512, [0, 1]),
        (768, [0, 1]),
    ]
    results = []

    with mesh_default_device(mesh_device):
        decoder, cfg = load_pretrained_text_decoder(weights_dir, dtype=torch.bfloat16)
        params = create_text_decoder_parameters(decoder, device=mesh_device)
        tt_dec = TTSeamlessM4Tv2Decoder(
            mesh_device,
            params,
            layer_norm_eps=cfg.layer_norm_eps,
            num_hidden_layers=cfg.decoder_layers,
            num_attention_heads=cfg.decoder_attention_heads,
            hidden_size=cfg.hidden_size,
        )

        for seq, seeds in sweep:
            for seed in seeds:
                try:
                    torch.manual_seed(seed)
                    batch = 1
                    enc_seq = seq
                    input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, seq), dtype=torch.int64)
                    encoder_hidden = torch.randn(batch, enc_seq, cfg.hidden_size, dtype=torch.bfloat16)
                    attn_mask = torch.ones(batch, seq, dtype=torch.long)
                    enc_mask = torch.ones(batch, enc_seq, dtype=torch.long)
                    inputs_embeds = decoder.embed_tokens(input_ids)
                    causal_mask = _prepare_4d_causal_attention_mask(
                        attn_mask, (batch, seq), inputs_embeds, past_key_values_length=0
                    )
                    cross_mask = _prepare_4d_attention_mask(enc_mask, inputs_embeds.dtype, tgt_len=seq)
                    position_ids = create_position_ids_from_input_ids(
                        input_ids, cfg.pad_token_id, past_key_values_length=0
                    )
                    ref = forward_torch_reference(decoder, input_ids, encoder_hidden, attn_mask, enc_mask).to(
                        torch.bfloat16
                    )

                    input_ids_tt = from_torch_uint32_rm(mesh_device, input_ids)
                    position_ids_tt = from_torch_uint32_rm(mesh_device, position_ids)
                    encoder_tt = from_torch_bfloat16_tile(mesh_device, encoder_hidden)
                    causal_tt = from_torch_bfloat16_tile(mesh_device, causal_mask)
                    cross_tt = from_torch_bfloat16_tile(mesh_device, cross_mask)
                    out_tt = tt_dec.forward(input_ids_tt, position_ids_tt, encoder_tt, causal_tt, cross_tt)
                    tt_cpu = (
                        to_torch_replicated_first_shard(out_tt)
                        .to(torch.bfloat16)
                        .reshape(batch, seq, cfg.hidden_size)
                        .contiguous()
                    )
                    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
                    results.append((seq, seed, "PCC", msg, ok))
                    logger.info(f"[text_decoder sweep] seq={seq} seed={seed}: {msg} -> {'PASS' if ok else 'FAIL'}")
                    ttnn.deallocate(out_tt)
                except Exception as e:
                    results.append((seq, seed, "EXC", str(e)[:100], False))
                    logger.warning(f"[text_decoder sweep] seq={seq} seed={seed}: EXCEPTION {str(e)[:100]} -> FAIL")
                    try:
                        mesh_device.clear_program_cache()
                    except Exception:
                        pass

    print("\n=== Text decoder sweep results ===")
    print(f"{'seq':>6}  {'seed':>4}  {'status':<6}  {'PCC / error':<60}")
    for seq, seed, kind, msg, ok in results:
        print(f"{seq:>6}  {seed:>4}  {'PASS' if ok else 'FAIL':<6}  {msg}")
    print("=" * 80)
