# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill trace audit: traced ``prefill_forward_text`` matches Generator eager and HF.

The direct-model reference path uses ``ttnn_prefill_forward(page_table=None)`` with a
minimal 32-tile-aligned kernel (same as ``test_full_model``), per batch slot.
Traced/batched Generator prefill uses the ISL bucket kernel and is checked against
Generator eager at 1.0 PCC — trace must not degrade numerics. When the prompt fills
the ISL bucket, traced vs HF is also asserted at 0.99.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from models.demos.gemma4.demo.text_demo import _batch_prefill_hits_ceiling, _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import (
    GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
    GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
    GEMMA4_TRACE_PREFILL_SEQ_LENS,
    can_gemma4_enable_prefill_trace,
    warmup_gemma4_prefill_bucket,
)
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import (
    TestFactory,
    _get_model_path,
    compare_tensors,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)

# Trace ISL buckets × SUPPORTED_PREFILL_BATCH_SIZES, minus:
#   - batch×kernel ≥ 128k (e.g. 32×4096 skips trace warmup)
#   - batch×kernel ≥ 32k or ISL > 4k (prefill trace disabled by policy)
#   - 31B 1×4 DRAM xfail: batch-32 × {2048, 4096} (2048 valid through batch-16 on 1×4)
_PREFILL_TRACE_BUCKETS = list(GEMMA4_TRACE_PREFILL_SEQ_LENS)
_PREFILL_TRACE_BATCH_SIZES = list(SUPPORTED_PREFILL_BATCH_SIZES)

_DEFAULT_PCC = 0.99


@pytest.fixture(scope="module")
def hf_causal_lm():
    """Load HF reference once per module — 31B reload dominates runtime."""
    skip_if_config_only_checkpoint()
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


def _hf_last_token_logits(hf_model, tokens, prompt_lens, *, kernel_len=None):
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens_list = prompt_lens.tolist()
    else:
        prompt_lens_list = list(prompt_lens)
    last_rows = []
    with torch.no_grad():
        for batch_idx, prompt_len in enumerate(prompt_lens_list):
            prompt_len = int(prompt_len)
            end = kernel_len if kernel_len is not None else prompt_len
            hf_out = hf_model(tokens[batch_idx : batch_idx + 1, :end].long())
            last_rows.append(hf_out.logits[0, prompt_len - 1, :].float())
    return torch.stack(last_rows, dim=0)


def _build_tokens(batch_size, prefill_len, vocab_size, *, seed=0):
    """Build ``test_full_model`` prompt tokens (short text, zero-padded to the ISL bucket)."""
    from transformers import AutoTokenizer

    kernel_len = get_padded_prefill_len(prefill_len)
    tokenizer = AutoTokenizer.from_pretrained(_get_model_path(), trust_remote_code=True)
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")
    seq_len = min(int(input_ids.shape[1]), prefill_len)
    if input_ids.shape[1] > prefill_len:
        input_ids = input_ids[:, :prefill_len]
        seq_len = prefill_len

    tokens = torch.zeros(batch_size, kernel_len, dtype=torch.int32)
    for batch_idx in range(batch_size):
        tokens[batch_idx, :seq_len] = input_ids[0, :seq_len].to(torch.int32)
    prompt_lens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
    return tokens, prompt_lens, kernel_len


def _minimal_tile_aligned_kernel_len(prompt_len):
    """Minimal 32-tile-aligned prefill length used by ``test_full_model``."""
    return ((prompt_len + 31) // 32) * 32


def _run_direct_model_prefill_last_logit(generator, tokens, prompt_lens):
    """Direct ``ttnn_prefill_forward(page_table=None)`` — same path as ``test_full_model``."""
    import torch.nn.functional as F

    import ttnn

    model = generator.model[0]
    model_args = generator.model_args[0]
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens_list = prompt_lens.tolist()
    else:
        prompt_lens_list = list(prompt_lens)
    is_mesh = hasattr(model.mesh_device, "shape") and model.mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(model.mesh_device) if is_mesh else None

    outputs = []
    for batch_idx, prompt_len in enumerate(prompt_lens_list):
        prompt_len = int(prompt_len)
        kernel_len = _minimal_tile_aligned_kernel_len(prompt_len)
        kv = _allocate_fresh_kv_cache(
            model,
            max_batch_size=1,
            max_seq_len=model_args.max_seq_len,
            paged_attention_config=None,
        )
        tokens_slice = tokens[batch_idx : batch_idx + 1, :kernel_len]
        if tokens_slice.shape[1] < kernel_len:
            tokens_slice = F.pad(tokens_slice, (0, kernel_len - tokens_slice.shape[1]), value=0)
        tokens_tt = ttnn.from_torch(
            tokens_slice.to(torch.int32),
            device=model.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, kernel_len, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
        embeds_torch = None
        if model._embed_weight_cpu is not None:
            embeds_torch = F.embedding(tokens_slice.long(), model._embed_weight_cpu).float() * model.embed_scale

        last_token_idx = prompt_len - 1
        tt_logits = model.ttnn_prefill_forward(
            embeds,
            page_table=None,
            kv_cache=kv,
            input_ids_torch=tokens_slice,
            embeds_torch=embeds_torch,
            get_last_token=(last_token_idx // 32) * 32,
        )
        tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0] if is_mesh else tt_logits).float()
        tt_logits.deallocate(True)
        if tt_logits_torch.dim() == 4:
            tt_logits_torch = tt_logits_torch.squeeze(1)
        outputs.append(tt_logits_torch[0, last_token_idx % 32, : model_args.vocab_size])
    return torch.stack(outputs, dim=0)


def _ensure_prefill_warmup(generator, kv_cache, tokens, page_table, prompt_lens, *, enable_trace):
    """Compile/capture prefill once for this bucket; skip the full batch×ISL warmup matrix."""
    warmed_attr = f"_parity_prefill_warmed_{enable_trace}"
    if getattr(generator, warmed_attr, False):
        return
    warmup_gemma4_prefill_bucket(
        generator,
        kv_cache,
        enable_trace=enable_trace,
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
    )
    setattr(generator, warmed_attr, True)


def _run_prefill_logits(generator, kv_cache, tokens, page_table, prompt_lens, *, enable_trace, paged_cfg):
    """Warm up once for the given trace mode, then return measured last-token logits on clean KV."""
    _ensure_prefill_warmup(generator, kv_cache, tokens, page_table, prompt_lens, enable_trace=enable_trace)

    model_args = generator.model_args[0]
    measure_kv = [
        _allocate_fresh_kv_cache(
            generator.model[0],
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            paged_attention_config=paged_cfg,
        )
    ]

    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=measure_kv,
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


@pytest.mark.gemma4_prefill_trace
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("batch_size", _PREFILL_TRACE_BATCH_SIZES, ids=lambda b: f"batch{b}")
def test_prefill_trace_eager_hf_parity(batch_size, prefill_len, mesh_device, reset_seeds, request, hf_causal_lm):
    """Eager and traced prefill last-token logits must match HF (and each other).

    Param matrix: ISL buckets {128,512,1024,2048,4096} × batch {1..32}. Each case warms
    up only its own bucket (no full batch×ISL trace capture sweep).
    Prefill trace is disabled when ISL > 4096 or padded_batch×kernel ≥ 32k.
    On 31B blackhole 1×4, prefill 2048 is expected through batch 8; batch-16×2048+
    are skipped (32k trace ceiling). batch-32×2048/4096 are xfails (DRAM).
    Batch-32×4096 also skips the 128k batched-prefill ceiling.
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

    max_padded_batch = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)
    if not can_gemma4_enable_prefill_trace(kernel_len, batch_size=max_padded_batch):
        pytest.skip(
            f"prefill trace disabled for padded_batch={max_padded_batch} x kernel={kernel_len} "
            f"(ISL>{GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN} or "
            f"batch×kernel>={GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS})"
        )

    model_path = _get_model_path()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    tokens, prompt_lens, kernel_len = _build_tokens(batch_size, prefill_len, hf_config.vocab_size)
    prompt_len = int(prompt_lens[0])
    minimal_kernel_len = _minimal_tile_aligned_kernel_len(prompt_len)
    hf_last_eager = _hf_last_token_logits(hf_causal_lm, tokens, prompt_lens, kernel_len=minimal_kernel_len)
    hf_last_trace = _hf_last_token_logits(hf_causal_lm, tokens, prompt_lens, kernel_len=kernel_len)

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
        kernel_len,
    )

    max_batch_size = max_padded_batch

    generator, kv_caches, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
    )
    model_args = generator.model_args[0]
    assert model_args.can_enable_trace(
        kernel_len, batch_size=max_batch_size
    ), f"Trace not enabled for kernel_len={kernel_len} batch_size={max_batch_size}"

    kv_eager = kv_caches
    kv_trace = [
        _allocate_fresh_kv_cache(
            generator.model[0],
            max_batch_size=model_args.max_batch_size,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_cfg,
        )
    ]

    out_eager = _run_prefill_logits(
        generator, kv_eager, tokens, page_table, prompt_lens, enable_trace=False, paged_cfg=paged_cfg
    )
    out_trace = _run_prefill_logits(
        generator, kv_trace, tokens, page_table, prompt_lens, enable_trace=True, paged_cfg=paged_cfg
    )

    out_direct = _run_direct_model_prefill_last_logit(generator, tokens, prompt_lens).float()
    _assert_parity("direct model prefill vs HF", hf_last_eager, out_direct, request)
    _assert_parity("traced vs eager (Generator)", out_eager, out_trace, request)
    if prompt_len >= prefill_len:
        _assert_parity("traced vs HF", hf_last_trace, out_trace, request)
        _assert_parity("traced vs direct model prefill", out_direct, out_trace, request)
