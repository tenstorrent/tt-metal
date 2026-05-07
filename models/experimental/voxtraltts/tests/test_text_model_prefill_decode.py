# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies as reference_compute_rope_frequencies,
    extract_layer_weights,
    rms_norm as reference_rms_norm,
    text_decoder_layer as reference_text_decoder_layer,
)
from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_prefill_pcc(device, reset_seeds):
    # Main parity test:
    # compares final last-token logits from TT text_model prefill vs PyTorch reference backbone.
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)
    args = model.inner.args
    state_dict = args.load_state_dict()

    seq_len = 128
    tokens = torch.randint(0, model.inner.vocab_size, (1, seq_len), dtype=torch.int64)

    # TT prefill path (final logits at last token)
    tt_x, rot_mats_global, rot_mats_local, _, _ = model.prepare_inputs_prefill(tokens, start_pos=0)
    tt_logits = model.inner.ttnn_prefill_forward(
        tt_x,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        get_last_token=((seq_len - 1) // 32) * 32,
    )
    tt_last_logits = model.inner.process_output_prefill(
        tt_logits.cpu(),
        last_token_idx=((seq_len - 1) % 32),
    ).float()

    # Reference full-text backbone forward using the same loaded checkpoint.
    ref_cfg = VoxtralTextConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.norm_eps,
    )
    ref_hidden = F.embedding(tokens, state_dict["tok_embeddings.weight"])
    ref_cos, ref_sin = reference_compute_rope_frequencies(
        head_dim=ref_cfg.head_dim,
        max_seq_len=seq_len,
        theta=ref_cfg.rope_theta,
        device=ref_hidden.device,
    )
    ref_attn_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32)
    ref_attn_mask = torch.triu(ref_attn_mask, diagonal=1)

    for layer_idx in range(ref_cfg.num_hidden_layers):
        layer_weights = extract_layer_weights(state_dict, layer_idx, prefix="layers.")
        ref_hidden = reference_text_decoder_layer(
            hidden_states=ref_hidden,
            layer_weights=layer_weights,
            cos=ref_cos,
            sin=ref_sin,
            config=ref_cfg,
            attention_mask=ref_attn_mask,
        )
    ref_hidden = reference_rms_norm(ref_hidden, state_dict["norm.weight"], eps=ref_cfg.rms_norm_eps)
    ref_last_logits = F.linear(ref_hidden[:, -1, :], state_dict["output.weight"]).squeeze(0).float()

    passing, pcc_message = comp_pcc(ref_last_logits, tt_last_logits, pcc=0.90)
    assert passing, f"Text model prefill logits mismatch vs reference: {pcc_message}"


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_decode_single_step_pcc(device, reset_seeds):
    # Consistency test (TT-only):
    # compares one decode-step output vs equivalent-position prefill output.
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    prompt_len = 128
    extended_seq_len = 256
    vocab_size = model.inner.vocab_size

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.int64)
    decode_input_token = torch.randint(0, vocab_size, (1,), dtype=torch.int64)

    # 1) Prefill prompt to populate KV cache, then run one decode step.
    tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _ = model.prepare_inputs_prefill(prompt_tokens, start_pos=0)
    _ = model.inner.ttnn_prefill_forward(
        tt_prompt_x,
        rot_mats_global=prompt_rot_global,
        rot_mats_local=prompt_rot_local,
        get_last_token=-1,
    )
    tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
        decode_input_token, torch.tensor([prompt_len], dtype=torch.int64)
    )
    tt_decode_logits, _ = model.inner.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rope_idxs,
        page_table=tt_page_table,
        kv_cache=None,
        sampling_on_device=False,
    )
    decode_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

    # 2) Compute the same position via pure prefill in a 256-token sequence.
    # Tokens after index prompt_len are masked by causal attention and do not affect
    # logits at prompt_len, but keep prefill length tile-compatible.
    future_tokens = torch.randint(0, vocab_size, (1, extended_seq_len - prompt_len - 1), dtype=torch.int64)
    extended_tokens = torch.cat([prompt_tokens, decode_input_token.view(1, 1), future_tokens], dim=1)
    tt_full_x, full_rot_global, full_rot_local, _, _ = model.prepare_inputs_prefill(extended_tokens, start_pos=0)
    tt_prefill_logits = model.inner.ttnn_prefill_forward(
        tt_full_x,
        rot_mats_global=full_rot_global,
        rot_mats_local=full_rot_local,
        get_last_token=prompt_len,
    )
    prefill_logits = model.inner.process_output_prefill(
        tt_prefill_logits.cpu(),
        last_token_idx=(prompt_len % 32),
    ).float()

    passing, pcc_message = comp_pcc(prefill_logits, decode_logits, pcc=0.95)
    assert passing, f"Decode vs prefill consistency failed on real Voxtral weights: {pcc_message}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("decode_steps", [2, 4], ids=["2_steps", "4_steps"])
def test_text_model_decode_multistep_pcc(device, reset_seeds, decode_steps):
    # Consistency test (TT-only):
    # repeats decode-vs-prefill comparison across multiple decode steps.
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    prompt_len = 128
    extended_seq_len = 256
    vocab_size = model.inner.vocab_size

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.int64)
    decode_tokens = torch.randint(0, vocab_size, (1, decode_steps), dtype=torch.int64)
    tt_prompt_x, prompt_rot_global, prompt_rot_local, _, _ = model.prepare_inputs_prefill(prompt_tokens, start_pos=0)
    _ = model.inner.ttnn_prefill_forward(
        tt_prompt_x,
        rot_mats_global=prompt_rot_global,
        rot_mats_local=prompt_rot_local,
        get_last_token=-1,
    )

    for step in range(decode_steps):
        current_pos = prompt_len + step
        step_token = decode_tokens[:, step]
        tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
            step_token, torch.tensor([current_pos], dtype=torch.int64)
        )
        tt_decode_logits, _ = model.inner.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=tt_page_table,
            kv_cache=None,
            sampling_on_device=False,
        )
        decode_logits = model.inner.process_output_decode(tt_decode_logits, B=1, S=1, is_tokens=False)[0, 0].float()

        # Compare step decode against equivalent full-sequence prefill at same position.
        prefix_tokens = torch.cat([prompt_tokens, decode_tokens[:, : step + 1]], dim=1)
        remaining_tail = torch.randint(
            0,
            vocab_size,
            (1, extended_seq_len - prefix_tokens.shape[1]),
            dtype=torch.int64,
        )
        extended_tokens = torch.cat([prefix_tokens, remaining_tail], dim=1)
        tt_full_x, full_rot_global, full_rot_local, _, _ = model.prepare_inputs_prefill(extended_tokens, start_pos=0)
        tt_prefill_logits = model.inner.ttnn_prefill_forward(
            tt_full_x,
            rot_mats_global=full_rot_global,
            rot_mats_local=full_rot_local,
            get_last_token=current_pos,
        )
        prefill_logits = model.inner.process_output_prefill(
            tt_prefill_logits.cpu(),
            last_token_idx=(current_pos % 32),
        ).float()

        passing, pcc_message = comp_pcc(prefill_logits, decode_logits, pcc=0.93)
        assert passing, f"Step {step} decode/prefill consistency failed " f"(pos={current_pos}): {pcc_message}"
