# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t` of coqui/XTTS-v2 (`Xtts.gpt`, the training forward).

Reference submodule: `gpt`, a `TTS.tts.layers.xtts.gpt.GPT`. This port covers the
`return_latent=True` inference-conditioning path that the pipeline uses (and that
the capture recorded): given text token ids, audio (mel) code ids, and a
precomputed conditioning latent, it returns the mel latents.

Forward (captured kwargs -> output `[1, 12, 1024]`):

    text_inputs   [1, 8]   int64     # text token ids
    text_lengths  [1]      int64
    audio_codes   [1, 12]  int64     # mel code ids
    wav_lengths   [1]      int64
    cond_latents  [1, 32, 1024] f32  # precomputed conditioning prefix
    return_latent True

Reference algorithm (`GPT.forward` + `GPT.get_logits`, return_latent branch):

    1. Host token bookkeeping (pure integer indexing, identical to the reference):
       pad / truncate text & audio codes, set mel padding, prepend start tokens
       and append stop tokens  ->  text_ids [1, Lt], audio_ids [1, Lm].
    2. text_emb = text_embedding(text_ids) + text_pos_embedding(0..Lt)
       mel_emb  = mel_embedding(audio_ids) + mel_pos_embedding(0..Lm)
    3. emb = cat([cond_latents, text_emb, mel_emb], dim=1)          # [1, C+Lt+Lm, D]
    4. hidden = gpt(inputs_embeds=emb)   (causal GPT2, attn_mask None -> causal-only)
    5. enc = final_norm(hidden[:, C:])
    6. mel_latent = enc[:, -Lm:];  return mel_latent[:, :-5]         # [1, Lm-5, D]

Everything numeric (embeddings, the 30-layer GPT2 stack, the final LayerNorm, and
all slicing/concat) runs in ttnn. The GPT2 transformer reuses the graduated native
`g_p_t2_model` port. Only the integer token bookkeeping (padding / start-stop
tokens) runs on host — it is index arithmetic on token ids, not neural compute,
and mirrors the reference line for line.

Harness note: the captured forward takes only kwargs, so the PCC harness supplies a
throwaway synthetic positional `primary` arg (first param below) which we ignore;
all real inputs arrive as host torch tensors in **kwargs.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.g_p_t2_model import build as _build_gpt2_model
from models.demos.xtts_v2._stubs.learned_position_embeddings import build as _build_lpe

_LN_EPS = 1e-5
_SUB = 5  # reference: mel_logits[:, :-5] ("don't ask me why 😄")


def _emb_weight(device, w):
    import torch

    return ttnn.as_tensor(
        w.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tile_weight(device, w):
    import torch

    return ttnn.as_tensor(
        w.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def build(device, torch_module):
    """Bind the trained GPT weights and return a native ttnn forward closure."""
    import torch

    m = torch_module.float()

    # The GPT2 transformer stack (native, graduated port).
    gpt2_forward = _build_gpt2_model(device, m.gpt)

    # Embedding tables (weight lookup) live in ROW_MAJOR for ttnn.embedding.
    text_emb_w = _emb_weight(device, m.text_embedding.weight)
    mel_emb_w = _emb_weight(device, m.mel_embedding.weight)
    # Absolute position-prefix lookup runs through the graduated
    # learned_position_embeddings leaf stub (returns float32 [sl, D]).
    lpe_text = _build_lpe(device, m.text_pos_embedding)
    lpe_mel = _build_lpe(device, m.mel_pos_embedding)

    lnf_w = _tile_weight(device, m.final_norm.weight)
    lnf_b = _tile_weight(device, m.final_norm.bias)

    model_dim = int(m.model_dim)
    start_text_token = int(m.start_text_token)
    stop_text_token = int(m.stop_text_token)
    start_audio_token = int(m.start_audio_token)
    stop_audio_token = int(m.stop_audio_token)
    code_stride_len = int(m.code_stride_len)

    def _embed(id_row, tok_w, lpe):
        # id_row: host torch int [1, L], OR a device uint32 [1, L] (host-free path).
        L = int(id_row.shape[1])
        if isinstance(id_row, ttnn.Tensor):
            tok_tt = id_row                              # already device uint32 ROW_MAJOR
        else:
            tok_tt = ttnn.as_tensor(
                id_row.to(torch.int32).contiguous(), dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tok = ttnn.embedding(tok_tt, tok_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # lpe reads only id_row.shape[1]=L and returns float32 [L, D]; cast to the
        # bf16 token-embedding dtype so ttnn.add operand dtypes match.
        pos = lpe(id_row)                                # float32 [L, D]
        pos = ttnn.reshape(pos, [1, L, model_dim])
        pos = ttnn.typecast(pos, ttnn.bfloat16)
        return ttnn.add(tok, pos)

    def forward(_primary=None, *args, **kwargs):
        text_inputs = kwargs["text_inputs"]
        text_lengths = kwargs["text_lengths"]
        wav_lengths = kwargs["wav_lengths"]
        # Host-free path: caller supplies the audio token ids and conditioning
        # prefix as device tensors (built on-device from the AR decode), so this
        # forward never round-trips to host for the audio side.
        audio_ids_tt = kwargs.get("audio_ids_tt")
        cond_latents_tt = kwargs.get("cond_latents_tt")

        # --- host token bookkeeping (mirrors GPT.forward, integer indexing) ---
        import torch.nn.functional as F

        max_text_len = int(text_lengths.max())

        # text side is a fixed host input — bookkeeping is pure integer indexing.
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=stop_text_token)
        text_ids = F.pad(text_inputs, (1, 0), value=start_text_token)
        text_emb = _embed(text_ids, text_emb_w, lpe_text)     # [1, Lt, D]

        if audio_ids_tt is not None:
            mel_emb = _embed(audio_ids_tt, mel_emb_w, lpe_mel)     # device ids -> [1, Lm, D]
        else:
            audio_codes = kwargs["audio_codes"]
            code_lengths = torch.ceil(wav_lengths.float() / code_stride_len).long() + 3
            max_mel_len = int(code_lengths.max())
            if max_mel_len > audio_codes.shape[-1]:
                audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))
            audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=stop_audio_token)
            # set_mel_padding(audio_codes, code_lengths - 3)
            audio_codes = audio_codes.clone()
            for b in range(code_lengths.shape[0]):
                actual_end = int(code_lengths[b] - 3)
                if actual_end < audio_codes.shape[-1]:
                    audio_codes[b, actual_end:] = stop_audio_token
            audio_ids = F.pad(audio_codes, (1, 0), value=start_audio_token)
            mel_emb = _embed(audio_ids, mel_emb_w, lpe_mel)       # [1, Lm, D]

        import torch as _torch

        if cond_latents_tt is not None:
            cond_tt = cond_latents_tt
            if cond_tt.get_dtype() != ttnn.bfloat16:
                cond_tt = ttnn.typecast(cond_tt, ttnn.bfloat16)
        else:
            cond_latents = kwargs["cond_latents"]
            cond_tt = ttnn.as_tensor(
                cond_latents.to(_torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        offset = int(cond_tt.shape[1])
        Lt = int(text_emb.shape[1])
        Lm = int(mel_emb.shape[1])

        emb = ttnn.concat([cond_tt, text_emb, mel_emb], dim=1)  # [1, offset+Lt+Lm, D]

        # --- GPT2 transformer + final norm (ttnn) ---
        hidden = gpt2_forward(emb)
        normed = ttnn.layer_norm(hidden, epsilon=_LN_EPS, weight=lnf_w, bias=lnf_b)

        # enc = normed[:, offset:]; mel_latent = enc[:, -Lm:]; return mel_latent[:, :-5]
        mel_start = offset + Lt
        out_len = Lm - _SUB
        return ttnn.slice(normed, [0, mel_start, 0], [1, mel_start + out_len, model_dim])

    return forward


def g_p_t(*args, **kwargs):
    raise RuntimeError(
        "g_p_t requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
