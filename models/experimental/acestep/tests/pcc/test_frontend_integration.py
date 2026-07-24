# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: full prompt front-end — TT text encoder -> ConditionEncoder -> DiT cross-attn context.

Validates the complete prompt->context chain the ACE-Step pipeline runs, all on genuine weights:

    prompt tokens ─► [TT Qwen3 text encoder] ─► text_hidden_states ─┐
    lyric tokens  ─► embed_tokens (lookup)   ─► lyric_hidden_states ┤─► [TT ConditionEncoder] ─► context
    timbre latents ────────────────────────────────────────────────┘   (text_projector + lyric + timbre)

This is the seam between Module 30 (text encoder) and the previously-validated ConditionEncoder:
the text encoder's real 28-layer output feeds the ConditionEncoder's text path (via text_projector),
alongside the genuine lyric-embed and timbre paths. Compared against the identical reference chain
(HF Qwen3 text encoder + HF AceStepConditionEncoder), so any divergence is device numerics only.

Mirrors the reference exactly (conditioning_embed.py + AceStepConditionEncoder.forward):
  - text_hidden_states = text_encoder(text_ids).last_hidden_state   (full 28-layer forward)
  - lyric_hidden_states = text_encoder.embed_tokens(lyric_ids)      (EMBED LOOKUP ONLY, no encoder)
  - context = pack(pack(lyric_core, timbre_core), text_proj)        (all-valid pack == concat)

Skipped if the pipeline bundle (text encoder + DiT/encoder checkpoint) isn't downloaded.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import have_pipeline, load_module_weights
from models.experimental.acestep.tt.model_config import build_condition_encoder, build_text_encoder
from models.experimental.acestep.tests.test_utils import require_single_device, to_torch

HEAD_DIM = 128
HIDDEN = 2048
TEXT_HIDDEN = 1024
SLIDING_WINDOW = 128
TEXT_LEN = 32
LYRIC_LEN = 64
TIMBRE_LEN = 96


def _encode_core_ref(ref_enc, m, x, seq_len):
    """Reference encoder core: embed(Linear) -> layers -> norm. Matches the TT lyric/timbre core."""
    e = ref_enc.embed_tokens(x)
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = ref_enc.rotary_emb(e, pos)
    h = e
    for rl in ref_enc.layers:
        mask = (
            None
            if rl.attention_type == "full_attention"
            else m.create_4d_mask(
                seq_len=seq_len,
                dtype=torch.float32,
                device=h.device,
                attention_mask=None,
                sliding_window=SLIDING_WINDOW,
                is_sliding_window=True,
                is_causal=False,
            )
        )
        (h,) = rl(hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask)
    return ref_enc.norm(h)


def _rope(cfg, seq_len, device):
    rope = Qwen3RotaryEmbedding(cfg)
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, seq_len, HEAD_DIM), pos)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return cos_tt, sin_tt


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle not downloaded")
def test_frontend_integration(device):
    require_single_device(device)

    # --- Build TT front-end (real weights): text encoder + condition encoder ---
    tt_text, hf_text = build_text_encoder(device)
    tt_ce = build_condition_encoder(_condition_args(), device)

    # --- Reference (genuine HF) text encoder + condition encoder, same checkpoint ---
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    ref_ce = m.AceStepConditionEncoder(hf).eval()
    load_module_weights(ref_ce, "encoder.")

    torch.manual_seed(0)
    text_ids = torch.randint(0, hf_text.config.vocab_size, (1, TEXT_LEN))
    lyric_ids = torch.randint(0, hf_text.config.vocab_size, (1, LYRIC_LEN))
    timbre = torch.randn(1, TIMBRE_LEN, hf.audio_acoustic_hidden_dim)

    # === Reference chain ===
    with torch.no_grad():
        text_hs = hf_text(input_ids=text_ids).last_hidden_state  # [1, Lt, 1024]
        text_proj_ref = text_hs @ ref_ce.text_projector.weight.t()  # [1, Lt, 2048], no bias
        lyric_hs = hf_text.embed_tokens(lyric_ids)  # embed lookup only -> [1, Ll, 1024]
        # lyric/timbre paths = the AceStepLyricEncoder core (embed Linear -> layers -> norm) run
        # directly on the embeddings (matches the validated condition-encoder test decomposition).
        lyric_core_ref = _encode_core_ref(ref_ce.lyric_encoder, m, lyric_hs, LYRIC_LEN)
        timbre_core_ref = _encode_core_ref(ref_ce.timbre_encoder, m, timbre, TIMBRE_LEN)
        ctx_ref = torch.cat([lyric_core_ref, timbre_core_ref, text_proj_ref], dim=1)  # all-valid pack==concat

    # === TT chain ===
    # 1. TT text encoder -> text_hidden_states.
    tcos, tsin = _rope(hf_text.config, TEXT_LEN, device)
    text_hs_tt = tt_text.forward(text_ids, tcos, tsin)  # [1,1,Lt,1024]
    text_hs_torch = to_torch(text_hs_tt, expected_shape=(1, 1, TEXT_LEN, TEXT_HIDDEN)).reshape(1, TEXT_LEN, TEXT_HIDDEN)

    # 2. TT ConditionEncoder: text = real TT text_hidden_states; lyric/timbre = embeds.
    from models.experimental.acestep.tests.test_utils import to_ttnn_tensor

    lyric_hs_tt = to_ttnn_tensor(lyric_hs.reshape(1, 1, LYRIC_LEN, TEXT_HIDDEN), device)
    timbre_tt = to_ttnn_tensor(timbre.reshape(1, 1, TIMBRE_LEN, hf.audio_acoustic_hidden_dim), device)
    text_tt = to_ttnn_tensor(text_hs_torch.reshape(1, 1, TEXT_LEN, TEXT_HIDDEN), device)

    lcos, lsin = _rope(hf, LYRIC_LEN, device)
    tbcos, tbsin = _rope(hf, TIMBRE_LEN, device)

    def _mask(seq):
        mk = m.create_4d_mask(
            seq_len=seq,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attention_mask=None,
            sliding_window=SLIDING_WINDOW,
            is_sliding_window=True,
            is_causal=False,
        )
        return ttnn.from_torch(mk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    ctx_tt = tt_ce.forward(
        text_tt,
        lyric_hs_tt,
        timbre_tt,
        lcos,
        lsin,
        tbcos,
        tbsin,
        lyric_sliding=_mask(LYRIC_LEN),
        timbre_sliding=_mask(TIMBRE_LEN),
    )
    total_len = LYRIC_LEN + TIMBRE_LEN + TEXT_LEN
    ctx = to_torch(ctx_tt, expected_shape=(1, 1, total_len, HIDDEN)).reshape(1, total_len, HIDDEN)

    passing, msg = comp_pcc(ctx_ref, ctx, 0.93)
    print(f"FRONTEND_PCC (text-enc -> condition-enc -> context): {msg}")
    assert passing, f"front-end integration PCC {msg} < 0.93"


def _condition_args():
    from models.experimental.acestep.tt.model_config import AceStepModelConfig

    return AceStepModelConfig.from_hf()
