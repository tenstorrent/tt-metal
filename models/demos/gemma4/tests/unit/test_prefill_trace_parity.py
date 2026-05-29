# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill trace audit: eager and traced ``prefill_forward_text`` match HF last-token logits.

Both TT prefill paths (eager and traced) must PCC-match HF at 0.99. Traced must
also match eager. Closes the gap left by ``test_vllm_parity`` (no trace) and
``test_full_model`` (does not exercise traced ``prefill_forward_text``).
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import GEMMA4_TRACE_PREFILL_SEQ_LENS
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import (
    TestFactory,
    _get_model_path,
    compare_tensors,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
)

# Trace ISL buckets × SUPPORTED_PREFILL_BATCH_SIZES, minus:
#   - batch×kernel ≥ 128k (e.g. 32×4096 skips trace warmup)
#   - 31B 1×4 DRAM xfail: batch-32 × {2048, 4096} (2048 valid through batch-16 on QB2)
_PREFILL_TRACE_BUCKETS = list(GEMMA4_TRACE_PREFILL_SEQ_LENS)
_PREFILL_TRACE_BATCH_SIZES = list(SUPPORTED_PREFILL_BATCH_SIZES)

_DEFAULT_PCC = 0.99


@pytest.fixture(scope="module")
def hf_causal_lm():
    """Load HF reference once per module — 31B reload dominates runtime."""
    model_path = _get_model_path()
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    yield model
    del model


def _page_params(batch_size, prefill_len, max_new_tokens=32, page_block_size=64):
    blocks_per_user = (prefill_len + max_new_tokens + page_block_size - 1) // page_block_size
    return PagedAttentionConfig(block_size=page_block_size, max_num_blocks=batch_size * blocks_per_user)


def _create_page_table(batch_size, paged_attention_config, *, seed=0):
    max_num_blocks = paged_attention_config.max_num_blocks
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(max_num_blocks, generator=generator)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(batch_size, max_num_blocks // batch_size).to(torch.int32)


def _allocate_fresh_kv_cache(tt_model, *, max_batch_size, max_seq_len, paged_attention_config):
    from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache

    built = {}
    caches = []
    for i, layer in enumerate(tt_model.layers):
        if i in tt_model.kv_shared_layer_map:
            caches.append(built[tt_model.kv_shared_layer_map[i]])
            continue
        attn_cfg = Gemma4AttentionConfig(tt_model.hf_config, i)
        kv = init_kv_cache(
            mesh_device=tt_model.mesh_device,
            config=attn_cfg,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
        )
        built[i] = kv
        caches.append(kv)
    return caches


def _hf_last_token_logits(hf_model, tokens, prompt_len):
    with torch.no_grad():
        hf_out = hf_model(tokens[:, :prompt_len].long())
        hf_logits = hf_out.logits.float()
    last_idx = prompt_len - 1
    return hf_logits[:, last_idx, :]


def _build_tokens(batch_size, prefill_len, vocab_size, *, seed=0):
    kernel_len = get_padded_prefill_len(prefill_len)
    gen = torch.Generator()
    gen.manual_seed(seed)
    payload = torch.randint(1, min(vocab_size, 8192), (batch_size, prefill_len), generator=gen, dtype=torch.int32)
    tokens = torch.zeros(batch_size, kernel_len, dtype=torch.int32)
    tokens[:, :prefill_len] = payload
    prompt_lens = torch.tensor([prefill_len] * batch_size, dtype=torch.long)
    return tokens, prompt_lens, kernel_len


def _run_prefill_logits(generator, kv_cache, tokens, page_table, prompt_lens, *, enable_trace):
    """Warm up once for the given trace mode, then return measured last-token logits."""
    generator.already_warmed_up_prefill = False
    generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=enable_trace,
        warmup_prefill=True,
    )
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=enable_trace,
        warmup_prefill=False,
    )
    if out.dim() == 3:
        return out.squeeze(1).float()
    return out.float()


def _assert_parity(name, reference, candidate, request):
    threshold = get_pcc_threshold(request, default=_DEFAULT_PCC)
    passing, pcc = compare_tensors(candidate, reference, pcc_threshold=threshold)
    assert passing, f"{name}: {pcc} (threshold={threshold})"
    return pcc


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("batch_size", _PREFILL_TRACE_BATCH_SIZES, ids=lambda b: f"batch{b}")
def test_prefill_trace_eager_hf_parity(batch_size, prefill_len, mesh_device, reset_seeds, request, hf_causal_lm):
    """Eager and traced prefill last-token logits must match HF (and each other).

    Full trace matrix: ISL buckets {128,512,1024,2048,4096} × batch {1..32}.
    On QB2 (31B 1×4), prefill 2048 is expected through batch 16; batch-32×2048/4096
    are xfails (DRAM). Batch-32×4096 also skips the 128k batched-prefill ceiling.
    """
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models disable prefill trace")

    kernel_len = get_padded_prefill_len(prefill_len)
    if _batch_prefill_hits_ceiling(batch_size, prefill_len):
        pytest.skip(
            f"batch {batch_size} x kernel {kernel_len} meets 128k batched-prefill ceiling "
            f"(trace warmup skips this combo)"
        )

    if kernel_len not in GEMMA4_TRACE_PREFILL_SEQ_LENS:
        pytest.skip(f"kernel_len={kernel_len} not in trace ISL buckets {GEMMA4_TRACE_PREFILL_SEQ_LENS}")

    model_path = _get_model_path()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    tokens, prompt_lens, _kernel_len = _build_tokens(batch_size, prefill_len, hf_config.vocab_size)
    hf_last = _hf_last_token_logits(hf_causal_lm, tokens, prefill_len)

    max_new_tokens = 32
    max_seq_len = max(prefill_len + max_new_tokens, 4096)
    paged_cfg = _page_params(batch_size, prefill_len, max_new_tokens)
    page_table = _create_page_table(batch_size, paged_cfg)

    mesh_key = "x".join(str(d) for d in mesh_device.shape)
    logger.info(
        "Prefill trace audit: model={} mesh={} batch={} prompt_len={} kernel_len={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        _kernel_len,
    )

    max_batch_size = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)

    generator, kv_caches, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
    )
    model_args = generator.model_args[0]
    assert model_args.can_enable_trace(kernel_len), f"Trace not enabled for kernel_len={kernel_len}"

    kv_eager = kv_caches
    kv_trace = [
        _allocate_fresh_kv_cache(
            generator.model[0],
            max_batch_size=model_args.max_batch_size,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_cfg,
        )
    ]

    out_eager = _run_prefill_logits(generator, kv_eager, tokens, page_table, prompt_lens, enable_trace=False)
    out_trace = _run_prefill_logits(generator, kv_trace, tokens, page_table, prompt_lens, enable_trace=True)

    _assert_parity("eager vs HF", hf_last, out_eager, request)
    _assert_parity("traced vs HF", hf_last, out_trace, request)
    _assert_parity("traced vs eager", out_eager, out_trace, request)
