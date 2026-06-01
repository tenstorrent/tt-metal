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

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_speech_encoder import (
    forward_torch_speech_encoder,
    load_pretrained_speech_encoder,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_text_decoder import forward_torch_reference
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import (
    align_case_for_tt_prefill,
    load_hf_model_and_processor,
    make_t2tt_decoder_pcc_inputs,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    to_torch_replicated_first_shard,
    tt_position_ids,
)
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
def test_sweep_text_decoder_max_enc_seq(mesh_device, device_params, reset_seeds):
    """Sweep T2TT encoder timeline lengths (powers of two) with the production two-token decoder seed."""
    _ = reset_seeds
    _ = device_params
    weights_dir = _weights_dir_or_skip()

    enc_seqs = [32, 64, 128, 256, 512, 768, 1024]
    results = []

    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        decoder = hf_model.text_decoder
        cfg = hf_model.config
        params = create_text_decoder_parameters(decoder, device=mesh_device)
        tt_dec = TTSeamlessM4Tv2Decoder(
            mesh_device,
            params,
            layer_norm_eps=cfg.layer_norm_eps,
            num_hidden_layers=cfg.decoder_layers,
            num_attention_heads=cfg.decoder_attention_heads,
            hidden_size=cfg.hidden_size,
        )

        for enc_seq in enc_seqs:
            try:
                case = make_t2tt_decoder_pcc_inputs(hf_model, processor, enc_seq_len=enc_seq)
                aligned = align_case_for_tt_prefill(case, int(cfg.pad_token_id))
                batch = int(aligned.input_ids.shape[0])
                logical_dec = aligned.logical_dec_seq
                padded_dec = aligned.padded_dec_seq

                ref = forward_torch_reference(
                    decoder,
                    aligned.input_ids,
                    aligned.encoder_hidden_states,
                    aligned.attention_mask,
                    aligned.encoder_attention_mask,
                ).to(torch.bfloat16)
                ref = ref[:, :logical_dec, :].contiguous()

                input_ids_tt = from_torch_uint32_rm(mesh_device, aligned.input_ids)
                encoder_tt = from_torch_bfloat16_tile(mesh_device, aligned.encoder_hidden_states)
                enc_mask_tt = from_torch_uint32_rm(mesh_device, aligned.encoder_attention_mask)
                pos_tt = tt_position_ids(input_ids_tt, int(cfg.pad_token_id))
                causal_tt = build_causal_with_padding_4d(None, batch, padded_dec, mesh_device)
                cross_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=padded_dec, device=mesh_device)

                out_tt = tt_dec.forward(input_ids_tt, pos_tt, encoder_tt, causal_tt, cross_tt)
                tt_cpu = (
                    to_torch_replicated_first_shard(out_tt)
                    .to(torch.bfloat16)
                    .reshape(batch, padded_dec, cfg.hidden_size)[:, :logical_dec, :]
                    .contiguous()
                )
                ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
                results.append((enc_seq, "PCC", msg, ok))
                logger.info(f"[text_decoder sweep] enc_seq={enc_seq}: {msg} -> {'PASS' if ok else 'FAIL'}")
                ttnn.deallocate(out_tt)
                for t in (input_ids_tt, encoder_tt, enc_mask_tt, pos_tt, causal_tt, cross_tt):
                    ttnn.deallocate(t)
            except Exception as e:
                results.append((enc_seq, "EXC", str(e)[:100], False))
                logger.warning(f"[text_decoder sweep] enc_seq={enc_seq}: EXCEPTION {str(e)[:100]} -> FAIL")
                try:
                    mesh_device.clear_program_cache()
                except Exception:
                    pass

    print("\n=== Text decoder sweep results (T2TT enc timeline, dec seed=2) ===")
    print(f"{'enc_seq':>8}  {'status':<6}  {'PCC / error':<60}")
    for enc_seq, kind, msg, ok in results:
        print(f"{enc_seq:>8}  {'PASS' if ok else 'FAIL':<6}  {msg}")
    print("=" * 80)
