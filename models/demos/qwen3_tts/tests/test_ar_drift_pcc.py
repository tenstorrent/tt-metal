# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
AR-loop drift PCC harness — Talker decode.

Extends the chained 28-layer prefill PCC test with N teacher-forced AR steps.
Both reference and TT consume the same next_embed sequence (synthesized by the
reference AR loop using its own greedy CP). At every step we report:
  - PCC of last-position Talker hidden state (post-norm, [1, 1, hidden])
  - PCC of codec_head logits at the last position
  - top-K agreement (codebook 0)

Goal: pinpoint at which AR step the perceptual divergence enters and whether
the decode-mode KV-cache path drifts faster than prefill PCC suggests.

Usage:
    pytest models/demos/qwen3_tts/tests/test_ar_drift_pcc.py -s -v
"""


import pytest
import torch
import torch.nn.functional as F

import ttnn

# Shared helpers from the prefill harness.
from models.demos.qwen3_tts.tests.test_chain_pcc import device  # fixture
from models.demos.qwen3_tts.tests.test_chain_pcc import state_dict  # fixture
from models.demos.qwen3_tts.tests.test_chain_pcc import compute_pcc

# Number of AR steps to probe. Keep small — non-trace decode is slow.
NUM_AR_STEPS = 6


def _allocate_kv_cache(device, num_layers, num_kv_heads, max_seq_len, head_dim):
    import os as _os

    cache_dtype = ttnn.float32 if _os.environ.get("TT_KV_CACHE_FP32", "0") == "1" else ttnn.bfloat16
    caches = []
    for _ in range(num_layers):
        k = ttnn.zeros(
            [1, num_kv_heads, max_seq_len, head_dim],
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.zeros(
            [1, num_kv_heads, max_seq_len, head_dim],
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        caches.append((k, v))
    return caches


def _deallocate_kv_cache(caches):
    for k, v in caches:
        ttnn.deallocate(k)
        ttnn.deallocate(v)


def run_reference_ar(
    state_dict,
    inputs_embeds: torch.Tensor,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    num_steps: int,
):
    """
    Reference AR loop (greedy, no KV cache — full attention every step).

    Returns list of dicts (one per step) with:
      step:         AR step index (0..num_steps-1)
      last_hidden:  [1, 1, hidden]  (post-final-norm, last position)
      logits:       [vocab]         (codec_head @ last_hidden)
      next_embed:   [1, 1, hidden]  (sum of 16 codebook embeds + text/pad)
      tokens:       List[int] of length 16 (codebook 0..15 sampled by reference)
    Plus the prefill output (step=-1) for sanity.
    """
    from models.demos.qwen3_tts.demo.demo_pure_reference_tts import TTSConfig
    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSCodePredictorConfig,
        Qwen3TTSConfig,
        code_predictor_forward,
        compute_mrope_frequencies,
        decoder_layer,
        extract_code_predictor_weights,
        extract_talker_weights,
        rms_norm,
    )

    config = TTSConfig()
    talker_weights = extract_talker_weights(state_dict)
    talker_config = Qwen3TTSConfig()

    codec_head = state_dict["talker.codec_head.weight"].float()
    codec_embed_weight = state_dict["talker.model.codec_embedding.weight"].float()

    cp_weights = extract_code_predictor_weights(state_dict)
    cp_weights = {k.replace("model.", ""): v.float() for k, v in cp_weights.items()}
    cp_config = Qwen3TTSCodePredictorConfig()

    mtp_w = cp_weights.get("small_to_mtp_projection.weight")
    mtp_b = cp_weights.get("small_to_mtp_projection.bias")

    def proj_cp(x):
        if mtp_w is not None:
            return F.linear(x, mtp_w, mtp_b)
        return x

    code_pred_embeds = [
        cp_weights[f"codec_embedding.{i}.weight"]
        for i in range(config.num_code_groups - 1)
        if f"codec_embedding.{i}.weight" in cp_weights
    ]
    lm_heads = [
        cp_weights[f"lm_head.{i}.weight"]
        for i in range(config.num_code_groups - 1)
        if f"lm_head.{i}.weight" in cp_weights
    ]

    hidden_states = inputs_embeds.float().clone()
    trailing_text_hidden = trailing_text_hidden.float()
    tts_pad_embed = tts_pad_embed.float()

    results = []

    for step in range(-1, num_steps):
        seq_len = hidden_states.shape[1]
        cos, sin = compute_mrope_frequencies(talker_config.head_dim, seq_len, talker_config.rope_theta)
        cos = cos.float()
        sin = sin.float()
        attn_mask = (
            torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0).float()
        )

        x = hidden_states
        for layer_idx in range(talker_config.num_hidden_layers):
            lp = f"layers.{layer_idx}."
            lw = {k.replace(lp, ""): v.float() for k, v in talker_weights.items() if k.startswith(lp)}
            x = decoder_layer(x, lw, cos, sin, talker_config, attention_mask=attn_mask, use_mrope=True)
        x = rms_norm(x, talker_weights["norm.weight"].float(), config.rms_norm_eps)
        last_hidden = x[:, -1:, :]
        logits = (last_hidden.squeeze(1) @ codec_head.T)[0]

        # Greedy codebook 0
        token_0 = int(logits.argmax().item())

        # CP loop for codebooks 1..15 (greedy, ignoring repetition penalty / temp).
        all_cb_embeds = [F.embedding(torch.tensor([[token_0]]), codec_embed_weight)]
        cp_input = torch.cat(
            [
                proj_cp(last_hidden),
                proj_cp(all_cb_embeds[0]),
            ],
            dim=1,
        )
        tokens = [token_0]
        cb_hiddens = []  # post-CP hidden for cb 1..15
        cb_logits_list = []  # LM-head logits for cb 1..15
        for cb_idx in range(config.num_code_groups - 1):
            cp_out = code_predictor_forward(cp_input, cp_weights, cp_config)
            cb_hidden = cp_out[:, -1, :].clone()  # [1, 1024]
            cb_logits = F.linear(cb_hidden, lm_heads[cb_idx])
            cb_token = int(cb_logits[0].argmax().item())
            tokens.append(cb_token)
            cb_hiddens.append(cb_hidden)
            cb_logits_list.append(cb_logits[0].clone())
            cb_embed = F.embedding(torch.tensor([[cb_token]]), code_pred_embeds[cb_idx])
            all_cb_embeds.append(cb_embed)
            if cb_idx < len(code_pred_embeds) - 1:
                cp_input = torch.cat([cp_input, proj_cp(cb_embed)], dim=1)

        cb_stack = torch.cat(all_cb_embeds, dim=1)  # [1, 16, hidden]
        next_embed = cb_stack.sum(dim=1, keepdim=True)
        # The reference advances trailing_text_hidden index by step starting at 0
        # for the FIRST AR step (step=0 in this loop). step=-1 is prefill.
        text_idx = step if step >= 0 else None
        if text_idx is not None:
            if text_idx < trailing_text_hidden.shape[1]:
                next_embed = next_embed + trailing_text_hidden[:, text_idx : text_idx + 1, :]
            else:
                next_embed = next_embed + tts_pad_embed

        results.append(
            {
                "step": step,
                "last_hidden": last_hidden.detach().clone(),
                "logits": logits.detach().clone(),
                "next_embed": next_embed.detach().clone(),
                "tokens": tokens,
                "cb_hiddens": [h.detach().clone() for h in cb_hiddens],  # 15 entries (cb 1..15)
                "cb_logits": [l.detach().clone() for l in cb_logits_list],  # 15 entries
                "talker_hidden": last_hidden.detach().clone(),  # alias for clarity
            }
        )

        # For step >= 0 advance hidden_states. For step = -1 (prefill), append next_embed
        # so the next iteration becomes step 0.
        hidden_states = torch.cat([hidden_states, next_embed], dim=1)

    return results


def run_ttnn_ar(
    device, state_dict, inputs_embeds: torch.Tensor, ref_steps: list, num_steps: int, mask_padding: bool = False
):
    """
    TT AR loop, teacher-forced from ref_steps.

    Args:
      ref_steps: output of run_reference_ar — used for next_embed at each step.

    Returns list of dicts per step (matches run_reference_ar shape):
      step, last_hidden ([1,1,hidden] torch float), logits ([vocab] torch float)
    """
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.talker import Talker

    config = Qwen3TTSTalkerConfig()
    talker = Talker(device=device, config=config, state_dict=state_dict)

    seq_len = inputs_embeds.shape[1]
    pad_seq = ((seq_len + 31) // 32) * 32
    pad = pad_seq - seq_len
    head_dim = config.head_dim
    max_seq_len = pad_seq + num_steps + 32  # generous

    # Prefill RoPE
    pf_pos = torch.arange(pad_seq)
    cos_pf, sin_pf = get_rope_tensors(device, head_dim, pad_seq, pf_pos, config.rope_theta)
    trans_mat = get_transformation_mat(head_dim, device)

    # KV caches
    kv_caches = _allocate_kv_cache(
        device,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
    )

    # Prefill input
    inp_padded = F.pad(inputs_embeds, (0, 0, 0, pad)).unsqueeze(1).to(torch.bfloat16)
    pf_input = ttnn.from_torch(
        inp_padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pf_hidden, kv_caches = talker.forward_from_hidden(
        pf_input,
        cos_pf,
        sin_pf,
        trans_mat,
        kv_caches=kv_caches,
        start_pos=0,
        mode="prefill",
    )
    pf_torch = ttnn.to_torch(pf_hidden).squeeze(1).float()[:, :seq_len, :]
    codec_head = state_dict["talker.codec_head.weight"].float()
    pf_last = pf_torch[:, -1:, :]
    pf_logits = (pf_last.squeeze(1) @ codec_head.T)[0]

    results = [
        {
            "step": -1,
            "last_hidden": pf_last.clone(),
            "logits": pf_logits.clone(),
        }
    ]

    # Decode mask buffer (we rebuild per-step; non-trace path)
    num_heads = config.num_attention_heads
    for k in range(num_steps):
        # Place the new token at real seq_len + k (matches production demo, where
        # padding slots [seq_len, pad_seq) are masked out and decode writes start
        # at seq_len). Using pad_seq + k instead would offset RoPE by
        # (pad_seq - seq_len) positions, breaking attention scores.
        cur_pos = seq_len + k
        # Use ref's next_embed (teacher-forced)
        # ref_steps[k] is step k-1 (results[0] = prefill, results[1] = step 0, ...).
        # AR step k consumes the next_embed produced by step k-1; for k=0 that is
        # the prefill's next_embed, i.e. ref_steps[0]["next_embed"].
        next_embed_torch = ref_steps[k]["next_embed"].to(torch.bfloat16)
        # Shape it to [1, 1, 1, hidden]
        dec_input_torch = next_embed_torch.unsqueeze(1)
        dec_input = ttnn.from_torch(
            dec_input_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # RoPE for cur_pos
        pos = torch.tensor([cur_pos])
        cos_dec, sin_dec = get_rope_tensors(device, head_dim, 1, pos, config.rope_theta)
        cur_pos_tensor = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Decode attn mask: 0.0 at valid positions [0, cur_pos], -inf beyond.
        # Same convention as the demo: positions seq_len..pad_seq-1 are also "valid"
        # (they contain prefill-time garbage K/V, but we let attention see them so
        # behavior matches the production path).
        # Mask: prefill columns [0, seq_len) + decode columns [seq_len, cur_pos+1)
        # are valid. Padding slots [seq_len, pad_seq) that happen to fall inside
        # the decode window are owned by us (we wrote them at this iteration or
        # earlier ones) — they're real K/V at proper RoPE phases.
        mask_host = torch.full((1, num_heads, 1, max_seq_len), float("-inf"))
        mask_host[0, :, 0, :seq_len] = 0.0
        mask_host[0, :, 0, seq_len : cur_pos + 1] = 0.0
        dec_mask = ttnn.from_torch(
            mask_host,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        dec_hidden, kv_caches = talker.forward_from_hidden(
            dec_input,
            cos_dec,
            sin_dec,
            trans_mat,
            kv_caches=kv_caches,
            start_pos=cur_pos,
            mode="decode",
            cur_pos_tensor=cur_pos_tensor,
            decode_attn_mask=dec_mask,
        )
        dec_torch = ttnn.to_torch(dec_hidden).squeeze(1).float()  # [1, 1, hidden]
        dec_logits = (dec_torch.squeeze(1) @ codec_head.T)[0]
        results.append(
            {
                "step": k,
                "last_hidden": dec_torch.clone(),
                "logits": dec_logits.clone(),
            }
        )
        ttnn.deallocate(dec_input)
        ttnn.deallocate(cur_pos_tensor)
        ttnn.deallocate(dec_mask)

    _deallocate_kv_cache(kv_caches)
    return results


@pytest.mark.parametrize(
    "voice,refcache_path,ref_text",
    [
        (
            "ashley",
            "/tmp/ashley_ref.refcache.pt",
            "Keeping my goals visible every day to stay focused on what matters most.",
        ),
        ("jim", "/tmp/jim_ref.refcache.pt", "Jason, can you put up the high level overview slides?"),
    ],
)
def test_ar_drift(device, state_dict, voice, refcache_path, ref_text):
    """Run prefill + N AR steps on both reference and TT (teacher-forced)."""
    import os

    if not os.path.exists(refcache_path):
        pytest.skip(f"Refcache missing: {refcache_path}")

    target_text = "Hello, welcome to Tenstorrent!!"
    cached = torch.load(refcache_path, weights_only=True)
    ref_codes = cached["ref_codes"].long()

    print(f"\n{'='*70}\n[{voice}] Building ICL embedding (ref_text='{ref_text[:40]}...')")
    # build_icl_embedding hardcodes a Jim ref_text — patch via monkeypatch of the
    # local reference. Easier: replicate inline here.
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_pure_reference_tts import TTSConfig, create_icl_embedding

    tcfg = TTSConfig()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)
    speaker_embedding = torch.zeros(1, 1, 2048)
    inputs_embeds, trailing_text, tts_pad = create_icl_embedding(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes,
        tokenizer=tokenizer,
        weights=state_dict,
        config=tcfg,
        speaker_embedding=speaker_embedding,
        language="english",
    )
    print(f"  ICL embedding shape: {inputs_embeds.shape}")

    print(f"\n[{voice}] Running reference AR (prefill + {NUM_AR_STEPS} steps)...")
    ref_steps = run_reference_ar(state_dict, inputs_embeds, trailing_text, tts_pad, NUM_AR_STEPS)

    mask_padding = os.environ.get("TT_AR_MASK_PADDING", "0") == "1"
    print(f"\n[{voice}] Running TT AR (prefill + {NUM_AR_STEPS} steps, teacher-forced, mask_padding={mask_padding})...")
    tt_steps = run_ttnn_ar(device, state_dict, inputs_embeds, ref_steps, NUM_AR_STEPS, mask_padding=mask_padding)

    print(f"\n{'='*70}")
    print(f"[{voice}] AR drift report")
    print(f"{'Step':>5} | {'Hidden PCC':>10} | {'Logit PCC':>10} | {'top-1':>6} | {'top-5∩':>6} | ref_top1 / tt_top1")
    print("-" * 80)
    for i, (rs, ts) in enumerate(zip(ref_steps, tt_steps)):
        h_pcc = compute_pcc(rs["last_hidden"], ts["last_hidden"])
        l_pcc = compute_pcc(rs["logits"].unsqueeze(0), ts["logits"].unsqueeze(0))
        ref_top5 = rs["logits"].topk(5).indices.tolist()
        tt_top5 = ts["logits"].topk(5).indices.tolist()
        top1 = "Y" if ref_top5[0] == tt_top5[0] else "N"
        top5_overlap = len(set(ref_top5) & set(tt_top5))
        label = "PF" if rs["step"] == -1 else f"AR{rs['step']}"
        print(
            f"{label:>5} | {h_pcc:>10.4f} | {l_pcc:>10.4f} | {top1:>6} | {top5_overlap:>6} | "
            f"{ref_top5[0]} / {tt_top5[0]}"
        )

    print(f"\n[{voice}] Reference greedy tokens (codebook 0):", [s["tokens"][0] for s in ref_steps if s["step"] >= 0])

    # Sanity assertion — prefill must already pass.
    pf_h_pcc = compute_pcc(ref_steps[0]["last_hidden"], tt_steps[0]["last_hidden"])
    assert pf_h_pcc > 0.95, f"Prefill last-hidden PCC {pf_h_pcc:.4f} too low (regression?)"


# ============================================================================
# Full AR-loop harness — Talker + Code Predictor (all 16 codebooks per frame).
# ============================================================================

NUM_AR_FRAMES_FULL = 3  # keep small — full CP loop is heavy on the non-trace path


def _alloc_cp_kv(device, num_layers, num_kv_heads, max_seq, head_dim):
    return _allocate_kv_cache(device, num_layers, num_kv_heads, max_seq, head_dim)


def run_ttnn_ar_full(device, state_dict, inputs_embeds, ref_steps, num_frames):
    """
    Full AR loop on TT: Talker prefill + (Talker decode + CP prefill + 14 CP decodes) per frame.
    Teacher-forced: at every codebook, we feed the reference's sampled token's embedding
    into TT (not the TT-sampled token). Returns per-frame, per-codebook diagnostics.

    Output schema: list of dicts (one per frame), each with:
      talker_hidden: torch [1, 1, 2048]
      talker_logits: torch [vocab]
      cb_hiddens:    list of 15 torch [1, 1024]   (cb 1..15 hidden after CP)
      cb_logits:     list of 15 torch [vocab]     (cb 1..15 logits)
    """
    import os as _os

    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.code_predictor_fp32 import CodePredictorFp32
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat
    from models.demos.qwen3_tts.tt.talker import Talker

    talker_cfg = Qwen3TTSTalkerConfig()
    cp_cfg = Qwen3TTSCodePredictorConfig()
    talker = Talker(device=device, config=talker_cfg, state_dict=state_dict)
    use_fp32_cp = _os.environ.get("TT_QWEN3_CP_FP32", "0") == "1"
    if use_fp32_cp:
        print("  [CP] using CodePredictorFp32 (fp32 activation path)")
        cp = CodePredictorFp32(
            device=device, config=cp_cfg, talker_hidden_size=talker_cfg.hidden_size, state_dict=state_dict
        )
    else:
        cp = CodePredictor(
            device=device, config=cp_cfg, talker_hidden_size=talker_cfg.hidden_size, state_dict=state_dict
        )

    seq_len = inputs_embeds.shape[1]
    pad_seq = ((seq_len + 31) // 32) * 32
    pad = pad_seq - seq_len
    head_dim = talker_cfg.head_dim
    max_talker_seq = pad_seq + num_frames + 32
    max_cp_seq = 32  # >= 16

    # Talker prefill
    pf_pos = torch.arange(pad_seq)
    cos_pf, sin_pf = get_rope_tensors(device, head_dim, pad_seq, pf_pos, talker_cfg.rope_theta)
    talker_trans_mat = get_transformation_mat(head_dim, device)
    talker_kv = _allocate_kv_cache(
        device,
        num_layers=talker_cfg.num_hidden_layers,
        num_kv_heads=talker_cfg.num_key_value_heads,
        max_seq_len=max_talker_seq,
        head_dim=head_dim,
    )
    inp_padded = F.pad(inputs_embeds, (0, 0, 0, pad)).unsqueeze(1).to(torch.bfloat16)
    pf_input = ttnn.from_torch(
        inp_padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pf_hidden_tt, talker_kv = talker.forward_from_hidden(
        pf_input, cos_pf, sin_pf, talker_trans_mat, kv_caches=talker_kv, start_pos=0, mode="prefill"
    )
    ttnn.deallocate(pf_input)

    # CP setup
    cp_head_dim = cp_cfg.head_dim
    cp_trans_mat = get_transformation_mat(cp_head_dim, device)

    # Codec embedding tables (ref-side torch) — used to embed teacher-forced tokens.
    codec_embed_w = state_dict["talker.model.codec_embedding.weight"].float()  # cb 0
    code_pred_embeds = []
    for i in range(cp_cfg.num_code_groups - 1):
        k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if k in state_dict:
            code_pred_embeds.append(state_dict[k].float())

    num_talker_heads = talker_cfg.num_attention_heads
    num_cp_heads = cp_cfg.num_attention_heads

    frames = []

    for k in range(num_frames):
        # ─── Talker decode at position seq_len + k ─────────────────────────
        cur_pos = seq_len + k
        # Use ref's next_embed from prior step (k=0 → ref_steps[0] = prefill's next_embed)
        next_embed_torch = ref_steps[k]["next_embed"].to(torch.bfloat16).unsqueeze(1)  # [1,1,1,2048]
        dec_in = ttnn.from_torch(
            next_embed_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos_d, sin_d = get_rope_tensors(device, head_dim, 1, torch.tensor([cur_pos]), talker_cfg.rope_theta)
        cur_pos_t = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_host = torch.full((1, num_talker_heads, 1, max_talker_seq), float("-inf"))
        mask_host[0, :, 0, :seq_len] = 0.0
        mask_host[0, :, 0, seq_len : cur_pos + 1] = 0.0
        dec_mask = ttnn.from_torch(
            mask_host,
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        talker_hidden_tt, talker_kv = talker.forward_from_hidden(
            dec_in,
            cos_d,
            sin_d,
            talker_trans_mat,
            kv_caches=talker_kv,
            start_pos=cur_pos,
            mode="decode",
            cur_pos_tensor=cur_pos_t,
            decode_attn_mask=dec_mask,
        )
        talker_hidden_torch = ttnn.to_torch(talker_hidden_tt).squeeze(1).float()  # [1,1,2048]
        codec_head = state_dict["talker.codec_head.weight"].float()
        talker_logits_torch = (talker_hidden_torch.squeeze(1) @ codec_head.T)[0]
        ttnn.deallocate(dec_in)
        ttnn.deallocate(cur_pos_t)
        ttnn.deallocate(dec_mask)

        # ─── CP forward (15 codebooks per frame) ──────────────────────────
        ref_tokens = ref_steps[k + 1]["tokens"]  # results[k+1] = AR step k
        token_0 = ref_tokens[0]

        if True:
            # CP forward_single_step path — same API for bf16 and fp32 CP.
            cp_dtype = ttnn.float32 if use_fp32_cp else ttnn.bfloat16
            cp_torch_dtype = torch.float32 if use_fp32_cp else torch.bfloat16
            token_0_embed = F.embedding(torch.tensor([[token_0]]), codec_embed_w).to(cp_torch_dtype)
            cp_pf_input_torch = torch.cat([talker_hidden_torch.to(cp_torch_dtype), token_0_embed], dim=1).unsqueeze(1)
            cp_pf_input = ttnn.from_torch(
                cp_pf_input_torch,
                device=device,
                dtype=cp_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Allocate CP KV cache in matching dtype.
            cp_kv = []
            for _ in range(cp_cfg.num_hidden_layers):
                k = ttnn.zeros(
                    [1, cp_cfg.num_key_value_heads, max_cp_seq, cp_head_dim],
                    dtype=cp_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                v = ttnn.zeros(
                    [1, cp_cfg.num_key_value_heads, max_cp_seq, cp_head_dim],
                    dtype=cp_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                cp_kv.append((k, v))
            cp_pf_pos = torch.arange(2)
            cp_pf_cos, cp_pf_sin = get_rope_tensors(device, cp_head_dim, 2, cp_pf_pos, cp_cfg.rope_theta)
            cp_logits_tt, cp_kv = cp.forward_single_step(
                cp_pf_input,
                cp_pf_cos,
                cp_pf_sin,
                cp_trans_mat,
                generation_step=1,
                kv_caches=cp_kv,
                start_pos=0,
                mode="prefill",
            )
            cb_logits_torch = ttnn.to_torch(cp_logits_tt).squeeze(1).float()[:, -1, :][0]
            ttnn.deallocate(cp_pf_input)
            cb_logits_tts = [cb_logits_torch]
            for cb_idx in range(1, cp_cfg.num_code_groups - 1):
                prev_token = ref_tokens[cb_idx]
                prev_embed = F.embedding(torch.tensor([[prev_token]]), code_pred_embeds[cb_idx - 1]).to(cp_torch_dtype)
                cp_dec_input = ttnn.from_torch(
                    prev_embed.unsqueeze(1),
                    device=device,
                    dtype=cp_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                dec_pos = cb_idx + 1
                cp_d_cos, cp_d_sin = get_rope_tensors(
                    device, cp_head_dim, 1, torch.tensor([dec_pos]), cp_cfg.rope_theta
                )
                cp_logits_tt, cp_kv = cp.forward_single_step(
                    cp_dec_input,
                    cp_d_cos,
                    cp_d_sin,
                    cp_trans_mat,
                    generation_step=dec_pos,
                    kv_caches=cp_kv,
                    start_pos=dec_pos,
                    mode="decode",
                )
                cb_logits_one = ttnn.to_torch(cp_logits_tt).squeeze(1).squeeze(1).float()[0]
                cb_logits_tts.append(cb_logits_one)
                ttnn.deallocate(cp_dec_input)
            _deallocate_kv_cache(cp_kv)

        frames.append(
            {
                "frame": k,
                "talker_hidden": talker_hidden_torch,
                "talker_logits": talker_logits_torch,
                "cb_logits": cb_logits_tts,  # 15 entries (cb 1..15)
            }
        )

    _deallocate_kv_cache(talker_kv)
    return frames


@pytest.mark.parametrize(
    "voice,refcache_path,ref_text",
    [
        (
            "ashley",
            "/tmp/ashley_ref.refcache.pt",
            "Keeping my goals visible every day to stay focused on what matters most.",
        ),
        ("jim", "/tmp/jim_ref.refcache.pt", "Jason, can you put up the high level overview slides?"),
    ],
)
def test_ar_full_loop(device, state_dict, voice, refcache_path, ref_text):
    """Full AR loop: Talker decode + CP prefill + 14 CP decodes per frame, teacher-forced."""
    import os

    if not os.path.exists(refcache_path):
        pytest.skip(f"Refcache missing: {refcache_path}")

    target_text = "Hello, welcome to Tenstorrent!!"
    cached = torch.load(refcache_path, weights_only=True)
    ref_codes = cached["ref_codes"].long()

    print(f"\n{'='*70}\n[{voice}] Building ICL embedding")
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_pure_reference_tts import TTSConfig, create_icl_embedding

    tcfg = TTSConfig()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)
    speaker_embedding = torch.zeros(1, 1, 2048)
    inputs_embeds, trailing_text, tts_pad = create_icl_embedding(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes,
        tokenizer=tokenizer,
        weights=state_dict,
        config=tcfg,
        speaker_embedding=speaker_embedding,
        language="english",
    )

    print(f"\n[{voice}] Reference AR (prefill + {NUM_AR_FRAMES_FULL} frames)...")
    ref_steps = run_reference_ar(state_dict, inputs_embeds, trailing_text, tts_pad, NUM_AR_FRAMES_FULL)

    print(f"\n[{voice}] TT full AR (prefill + {NUM_AR_FRAMES_FULL} frames)...")
    tt_frames = run_ttnn_ar_full(device, state_dict, inputs_embeds, ref_steps, NUM_AR_FRAMES_FULL)

    print(f"\n{'='*70}\n[{voice}] Full AR-loop drift report")
    print(f"{'Frame':>5} | {'CB':>3} | {'logit PCC':>10} | {'top-1':>6} | {'top-5∩':>6} | ref / tt top-1")
    print("-" * 70)
    for f_idx, tt_f in enumerate(tt_frames):
        rs = ref_steps[f_idx + 1]  # ref_steps[0] is prefill; AR step k is at index k+1
        # cb 0: talker logits
        ref_l0 = rs["logits"]
        tt_l0 = tt_f["talker_logits"]
        l_pcc = compute_pcc(ref_l0.unsqueeze(0), tt_l0.unsqueeze(0))
        ref_top5 = ref_l0.topk(5).indices.tolist()
        tt_top5 = tt_l0.topk(5).indices.tolist()
        top1 = "Y" if ref_top5[0] == tt_top5[0] else "N"
        ovl = len(set(ref_top5) & set(tt_top5))
        print(f"{f_idx:>5} | {0:>3} | {l_pcc:>10.4f} | {top1:>6} | {ovl:>6} | {ref_top5[0]} / {tt_top5[0]}")
        # cb 1..15
        for cb_i in range(15):
            ref_l = rs["cb_logits"][cb_i]
            tt_l = tt_f["cb_logits"][cb_i]
            l_pcc = compute_pcc(ref_l.unsqueeze(0), tt_l.unsqueeze(0))
            ref_top5 = ref_l.topk(5).indices.tolist()
            tt_top5 = tt_l.topk(5).indices.tolist()
            top1 = "Y" if ref_top5[0] == tt_top5[0] else "N"
            ovl = len(set(ref_top5) & set(tt_top5))
            print(f"{f_idx:>5} | {cb_i+1:>3} | {l_pcc:>10.4f} | {top1:>6} | {ovl:>6} | {ref_top5[0]} / {tt_top5[0]}")
