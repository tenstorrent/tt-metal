# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generator last-token prefill vs ``ttnn_prefill_forward`` on the same 128 tokens.

``test_full_model`` 0.98 PCC is a short prompt. This test asserts the generator
path agrees with that full-model prefill on a 128-token sequence, so a
generator-only regression cannot hide behind the short-prompt number.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.generator import Gemma4Generator

from ..test_factory import (
    _get_model_path,
    hf_reference_model_device,
    load_hf_reference_model,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)
from .test_teacher_forcing_e2e import _build_tokens

_SEQ = 128


def _to_host_logits(tt_model, tt_out):
    if tt_model.mesh_config is not None and tt_model.mesh_config.tp > 1:
        out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    else:
        out = ttnn.to_torch(tt_out).float()
    return out


def _last_row(t, vocab, seq_idx):
    if t.dim() == 4:
        t = t.squeeze(1)
    return t.reshape(-1, t.shape[-1])[seq_idx, :vocab].contiguous()


@pytest.mark.gemma4_hf_direct_parity
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric()
def test_generator_vs_full_model_prefill_parity(mesh_device, reset_seeds, request):
    skip_if_config_only_checkpoint()
    max_prefill = request.config.getoption("--max-prefill")
    if _SEQ > max_prefill:
        pytest.skip(f"seq={_SEQ} > --max-prefill={max_prefill}")

    model_path = _get_model_path()
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=4096,
        paged_attention_config=None,
    )
    tt_model = generator.model[0]
    model_kv = tt_kv_cache[0]
    vocab = int(tt_model.vocab_size)
    tokens = _build_tokens(tokenizer, _SEQ + 2)
    prompt = tokens[:, :_SEQ]
    padded_len = ((_SEQ + 31) // 32) * 32
    input_ids_padded = F.pad(prompt, (0, padded_len - _SEQ), value=0) if padded_len > _SEQ else prompt

    hf_model = load_hf_reference_model(model_path)
    try:
        device = hf_reference_model_device(hf_model)
        with torch.no_grad():
            hf_out = hf_model(tokens[:, : _SEQ + 1].long().to(device))
        hf_all = hf_out.logits[0, :_SEQ, :vocab].float().cpu()
        hf_prefill = hf_all[_SEQ - 1]
        hf_decode = hf_out.logits[0, _SEQ, :vocab].float().cpu()
    finally:
        del hf_model

    is_mesh = mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    def _embed_prompt():
        tokens_tt = ttnn.from_torch(
            input_ids_padded.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = tt_model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, tt_model.hidden_size))
        return ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    full_logits = tt_model.ttnn_prefill_forward(
        _embed_prompt(),
        page_table=None,
        kv_cache=model_kv,
        input_ids_torch=input_ids_padded,
        get_last_token=-1,
    )
    full_host = _to_host_logits(tt_model, full_logits)
    full_logits.deallocate(True)
    full_row = _last_row(full_host, vocab, _SEQ - 1)

    last_tile = tt_model.ttnn_prefill_forward(
        _embed_prompt(),
        page_table=None,
        kv_cache=model_kv,
        input_ids_torch=input_ids_padded,
        get_last_token=_SEQ - 1,
    )
    last_host = _to_host_logits(tt_model, last_tile)
    last_tile.deallocate(True)
    last_row = _last_row(last_host, vocab, (_SEQ - 1) % 32)

    page_table = None
    gen_out = generator.prefill_forward_text(
        prompt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor([_SEQ], dtype=torch.long),
        enable_trace=False,
        warmup_prefill=False,
        sampling_params=None,
    )
    gen_row = gen_out.float().cpu().reshape(-1, gen_out.shape[-1])[-1, :vocab].contiguous()

    decode_out, _ = generator.decode_forward(
        tokens[:, _SEQ].reshape(1, 1).long(),
        torch.tensor([_SEQ], dtype=torch.long),
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        sampling_params=None,
    )
    dec_row = decode_out.float().cpu().reshape(-1, decode_out.shape[-1])[-1, :vocab].contiguous()

    def _log(name, a, b):
        _, pcc = comp_pcc(b, a, pcc=0.0)
        a_id = int(a.argmax())
        b_id = int(b.argmax())
        logger.info(
            "{} PCC={:.6f} argmax TT={} ({!r}) ref={} ({!r}) match={}",
            name,
            float(pcc),
            a_id,
            tokenizer.decode([a_id]),
            b_id,
            tokenizer.decode([b_id]),
            a_id == b_id,
        )
        return float(pcc)

    pcc_full_hf = _log("full_model_prefill vs HF", full_row, hf_prefill)
    pcc_last_hf = _log("last_token_slice vs HF", last_row, hf_prefill)
    pcc_gen_hf = _log("generator_prefill vs HF", gen_row, hf_prefill)
    pcc_full_last = _log("last_token_slice vs full_model", last_row, full_row)
    pcc_gen_full = _log("generator_prefill vs full_model", gen_row, full_row)
    pcc_dec_hf = _log("generator_decode[0] vs HF", dec_row, hf_decode)

    if full_host.dim() == 4:
        tt_seq = full_host.squeeze(1)[0, :_SEQ, :vocab]
    else:
        tt_seq = full_host[0, :_SEQ, :vocab]
    logger.info("per-token full_model vs HF PCC (seq={}):", _SEQ)
    pccs = []
    n_match = 0
    for t in range(_SEQ):
        _, pcc_t = comp_pcc(hf_all[t], tt_seq[t], pcc=0.0)
        pccs.append(float(pcc_t))
        hf_tok = int(hf_all[t].argmax())
        tt_tok = int(tt_seq[t].argmax())
        match = hf_tok == tt_tok
        n_match += int(match)
        if t in (0, 1, 5, 31, 63, 127):
            logger.info(
                "  token[{}] pcc={:.6f} HF={!r} TT={!r} {}",
                t,
                float(pcc_t),
                tokenizer.decode([hf_tok]),
                tokenizer.decode([tt_tok]),
                "ok" if match else "MISMATCH",
            )
    logger.info(
        "  mean PCC={:.6f} min={:.6f} argmax match={}/{}",
        sum(pccs) / len(pccs),
        min(pccs),
        n_match,
        _SEQ,
    )

    logger.info(
        "shapes full={} last_tile={} gen={} vocab={}",
        tuple(full_host.shape),
        tuple(last_host.shape),
        tuple(gen_out.shape),
        vocab,
    )

    assert pcc_full_last > 0.99, f"last-token lm_head slice diverges from full-seq lm_head: PCC={pcc_full_last:.6f}"
    assert pcc_gen_full > 0.99, (
        f"generator prefill last row diverges from test_full_model path: PCC={pcc_gen_full:.6f} "
        f"(vs HF: generator={pcc_gen_hf:.6f} full_model={pcc_full_hf:.6f} slice={pcc_last_hf:.6f}; "
        f"decode0 vs HF={pcc_dec_hf:.6f})"
    )
