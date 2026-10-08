# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: does a long thinking trace stay faithful to HF along its whole length?

A real server-generated reasoning trace (thousands of tokens) is teacher-forced
through the TT generator exactly as serving does it: the prompt is prefilled,
then one traced decode step per generated token with the TRUE next token.
At every position TT's top-1 / top-5 are scored against the HF model's
predictions stored by ``generate_trace_reference.py``, and reported per
512-token window. Three gates:

* overall top-1 agreement with HF >= MIN_TOP1
* no window below MIN_WINDOW_TOP1
* no decay: the last quarter of windows is not more than MAX_DECAY below the first

Measured 2026-10-08 on the 12B QB2 1x4 mesh (bfp8 weights + KV): HF-side
teacher forcing of the same trace gives 0.81-0.94 top-1 per window with no
trend; a drifting device would show a falling curve or a cliff at the sliding
window (1024) or prefill chunk (4096) boundaries.

    HF_MODEL=google/gemma-4-12B-it MESH_DEVICE=P150x4 pytest test_long_trace_fidelity.py -k 1x4
    GEMMA4_TRACE_STEPS=2048  # quicker partial run
"""

import os

import pytest
import torch
from loguru import logger

from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _install_hybrid_page_tables,
    _resolve_bounded_sliding,
)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

WINDOW = 512
MIN_TOP1 = float(os.environ.get("GEMMA4_TRACE_MIN_TOP1", "0.80"))
MIN_WINDOW_TOP1 = float(os.environ.get("GEMMA4_TRACE_MIN_WINDOW_TOP1", "0.70"))
MAX_DECAY = float(os.environ.get("GEMMA4_TRACE_MAX_DECAY", "0.08"))
STEPS = int(os.environ.get("GEMMA4_TRACE_STEPS", "0"))  # 0 = the whole trace


def _reference_path(model_path):
    name = os.path.basename(os.path.normpath(model_path))
    return os.path.join(os.path.dirname(__file__), "reference_outputs", f"{name}_trace.refpt")


def _logits_of(out):
    logits = out[0] if isinstance(out, tuple) else out
    return logits.reshape(logits.shape[0], -1).float()  # [B, vocab]


@pytest.fixture
def model_path():
    return os.environ.get("HF_MODEL", "google/gemma-4-12B-it")


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_long_trace_fidelity(mesh_device, model_path):
    skip_if_config_only_checkpoint()
    ref_path = _reference_path(model_path)
    if not os.path.exists(ref_path):
        pytest.skip(f"no HF trace reference at {ref_path}; run generate_trace_reference.py")
    ref = torch.load(ref_path)
    ids = ref["reference_tokens"]  # [1, N + 1]
    hf_top1 = ref["top1_tokens"]  # [N]
    prefill_len = int(ref["score_start"])  # the prompt; every later token is generated
    n_positions = ids.shape[1] - 1
    if STEPS:
        n_positions = min(n_positions, prefill_len + STEPS)
    max_seq_len = 1 << (n_positions + 1).bit_length()  # next power of two above the sequence
    max_seq_len = max(max_seq_len, 4096)
    batch_size = 1
    page_params = _batch_page_params(batch_size, prefill_len, n_positions - prefill_len + 2)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=int(page_params["page_max_num_blocks"]),
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)
    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    generator, tt_kv_cache, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            generator.model_args[0],
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )
    prefill_out = generator.prefill_forward_text(
        ids[:, :prefill_len],
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[prefill_len],
        warmup_prefill=False,
        enable_trace=False,
    )
    tt_top5 = [torch.topk(_logits_of(prefill_out)[0], k=5).indices]
    positions = [prefill_len - 1]
    current_pos = torch.full((batch_size,), prefill_len, dtype=torch.int32)
    for pos in range(prefill_len, n_positions):
        out = generator.decode_forward(
            ids[:, pos],
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        tt_top5.append(torch.topk(_logits_of(out)[0], k=5).indices)
        positions.append(pos)
        current_pos += 1
    tt_top5 = torch.stack(tt_top5)  # [S, 5]
    positions = torch.tensor(positions)
    hf1 = hf_top1[positions]
    top1 = (tt_top5[:, 0] == hf1).float()
    top5 = (tt_top5 == hf1[:, None]).any(-1).float()
    gen_index = positions - (prefill_len - 1)  # 0 = first generated token
    rows = []
    for a in range(0, int(gen_index.max()) + 1, WINDOW):
        m = (gen_index >= a) & (gen_index < a + WINDOW)
        if m.any():
            rows.append((a, min(a + WINDOW, int(gen_index.max()) + 1), top1[m].mean().item(), top5[m].mean().item()))
    table = "\n".join(f"  {a:>6}-{b:<6} top1 {t1:.3f}  top5 {t5:.3f}" for a, b, t1, t5 in rows)
    overall1, overall5 = top1.mean().item(), top5.mean().item()
    q = max(1, len(rows) // 4)
    first_q, last_q = sum(r[2] for r in rows[:q]) / q, sum(r[2] for r in rows[-q:]) / q
    logger.info(
        f"{os.path.basename(model_path)} trace fidelity over {len(positions)} generated positions "
        f"({ref.get('meta', {}).get('source', '?')}): top1 vs HF {overall1:.4f}, top5 {overall5:.4f}, "
        f"first-quarter {first_q:.3f} -> last-quarter {last_q:.3f}\n{table}"
    )
    worst = min(rows, key=lambda r: r[2])
    assert overall1 >= MIN_TOP1, f"top-1 agreement with HF {overall1:.4f} < {MIN_TOP1}"
    assert worst[2] >= MIN_WINDOW_TOP1, f"window {worst[0]}-{worst[1]} top-1 {worst[2]:.3f} < {MIN_WINDOW_TOP1}"
    assert last_q >= first_q - MAX_DECAY, f"top-1 decays along the trace: {first_q:.3f} -> {last_q:.3f}"
