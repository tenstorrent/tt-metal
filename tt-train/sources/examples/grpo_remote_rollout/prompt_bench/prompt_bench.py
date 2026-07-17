#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-P150 prompt-length benchmark: opens a [1, 1] mesh, loads
Llama-3.2-1B-Instruct with real weights, and runs one prefill + on-device
decode loop per prompt length (16, 32, 64, 128, 256, 512).

The prefill and decode paths are inlined here (adapted from
``utils.ttt_generation_worker.TttGenerationWorker``) instead of instantiating
the worker class, so the flow is visible end-to-end.

Prompts are raw ``BOS + instruction + filler`` token slices (no chat-template
block) so every target length is hit exactly. Sampling is greedy (temperature
0, baked into the decode trace). ``HF_TOKEN`` must be set to pull the gated
Llama-3.2 weights.

Run:
    cd tt-metal
    python3 tt-train/sources/examples/grpo_remote_rollout/prompt_bench/prompt_bench.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, FrozenSet, List, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MESH_SHAPE = (1, 1)  # single P150 board
BATCH_SIZE = 1  # one prompt per prefill+decode pass
MAX_SEQ_LEN = 2048  # must fit max target prompt + MAX_NEW_TOKENS
MAX_NEW_TOKENS = 512  # bumped from 128 to expose late decode corruption
TARGET_PROMPT_LENS: Sequence[int] = (16, 32, 64, 128, 256, 512)

# Paged-KV sizing (per user); worst-case prompt+decode must fit in the cache.
_PAGED_BLOCK_SIZE = 32
_MIN_NUM_BLOCKS = 1024

# Trace-region size that tt-transformers' decode trace needs (matches demo default).
_TRACE_REGION_SIZE = 50_000_000

# Async-read cadence for the decode loop (drain read events every N steps).
_READ_EVERY = 4

# Filler prompt: BOS + instruction + filler passage. We slice raw tokens (no
# chat-template block) so we can hit small targets like 16/32 exactly. Instruct
# models still produce coherent output for this text-completion style prompt.
_INSTRUCTION = "Please continue the following passage in one short paragraph.\n\nPassage: "
_FILLER_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
    "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum. "
)


def _prepare_prompts(tokenizer, targets: Sequence[int]) -> List[List[int]]:
    """Build filler prompts of exactly the requested token lengths.

    Tokenize a big filler passage once (BOS + instruction + repeated paragraph),
    then slice the front to each target length. O(1) tokenize calls total, no
    growing loop, exact length for every target.
    """
    filler_pool = tokenizer.encode(_INSTRUCTION + _FILLER_PARAGRAPH * 40, add_special_tokens=True)
    max_target = max(targets)
    assert max_target <= len(filler_pool), (
        f"max target_len={max_target} exceeds filler pool ({len(filler_pool)} tokens); "
        "increase the _FILLER_PARAGRAPH multiplier"
    )
    return [[int(t) for t in filler_pool[:target]] for target in targets]


def _build_model(mesh_device: Any) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Build ``(model_args, model, generator, tt_kv_cache, page_table,
    paged_attention_config)`` for a single-submesh, batch-1 configuration.
    Mirrors the relevant init path of :class:`TttGenerationWorker` (DP=1)."""
    import torch
    import ttnn

    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model import Transformer
    from models.tt_transformers.tt.model_config import ModelArgs

    from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations

    dtype = ttnn.bfloat16

    required_blocks_per_user = (MAX_SEQ_LEN + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE
    max_num_blocks = max(_MIN_NUM_BLOCKS, BATCH_SIZE * required_blocks_per_user)
    blocks_per_user = max_num_blocks // BATCH_SIZE
    max_num_blocks = blocks_per_user * BATCH_SIZE
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=max_num_blocks)

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).repeat(1).reshape(BATCH_SIZE, blocks_per_user)

    os.environ["HF_MODEL"] = MODEL_ID  # ModelArgs reads HF_MODEL from env

    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=BATCH_SIZE,
        optimizations=lambda ma: bf16_attn_bfp8_mlp_optimizations(ma.n_layers, ma.model_name),
        max_seq_len=MAX_SEQ_LEN,
        cache_hf=True,
        dummy_weights=False,
    )
    model_args.lm_head_dtype = ttnn.bfloat16
    model_args.ccl_dtype = ttnn.bfloat16

    state_dict = model_args.load_state_dict()
    weight_cache_path = model_args.weight_cache_path(dtype)

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [[layer.attention.layer_past for layer in model.layers]]

    generator = Generator(model=[model], model_args=[model_args], mesh_device=mesh_device, tokenizer=None)

    assert model.sampling is not None, (
        "On-device sampling unavailable for this vocab_size / mesh shape; the decode "
        "loop bakes sampling into the trace and requires model.sampling."
    )
    return model_args, model, generator, tt_kv_cache, page_table, paged_attention_config


def _reset_kv_cache(model: Any, generator: Any) -> None:
    """Zero the KV cache in-place and clear the generator's cached page table."""
    import ttnn

    for layer in model.layers:
        k_cache, v_cache = layer.attention.layer_past
        ttnn.mul(k_cache, 0, output_tensor=k_cache)
        ttnn.mul(v_cache, 0, output_tensor=v_cache)
    generator.prev_page_table = None


def _generate_one(
    *,
    model: Any,
    generator: Any,
    tt_kv_cache: Any,
    page_table: Any,
    sampling_params: Any,
    prompt_ids: List[int],
    pad_token_id: int,
    stop_token_ids: FrozenSet[int],
    max_new_tokens: int,
) -> Tuple[List[int], float, float, int]:
    """Prefill + on-device decode for a single prompt (batch=1).

    Returns ``(completion_ids, prefill_seconds, decode_seconds, decode_steps)``.
    """
    import torch
    import ttnn

    _reset_kv_cache(model, generator)

    prompt_len = len(prompt_ids)
    input_tokens_prefill_pt = torch.full((1, prompt_len), pad_token_id, dtype=torch.int32)
    input_tokens_prefill_pt[0, :] = torch.tensor(prompt_ids, dtype=torch.int32)

    t_prefill = time.perf_counter()
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[prompt_len],
        sampling_params=sampling_params,
        warmup_prefill=False,
        enable_trace=True,
    )
    prefilled_token = (prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out).reshape(-1)
    prefill_s = time.perf_counter() - t_prefill

    completion: List[int] = []
    done = False
    first_tok = int(prefilled_token[0])
    if first_tok in stop_token_ids:
        done = True
    else:
        completion.append(first_tok)

    if done or max_new_tokens <= 1:
        return completion, prefill_s, 0.0, 0

    current_pos = torch.tensor([prompt_len], dtype=torch.int32)
    out_tok = prefilled_token.unsqueeze(1)  # stays on device

    buffered_reads: List[Any] = []
    read_events: Any = None

    def _consume(step_reads_list: List[Any]) -> None:
        nonlocal done
        for step_reads in step_reads_list:
            gathered = generator.process_decode_output_host(step_reads, is_tokens=True)
            tokens = gathered[0] if isinstance(gathered, tuple) else gathered
            for raw in tokens.reshape(-1).tolist():
                tok = int(raw)
                if done:
                    continue
                if tok in stop_token_ids:
                    done = True
                else:
                    completion.append(tok)

    t_decode = time.perf_counter()
    steps_executed = 0
    for step in range(max_new_tokens - 1):
        decoded = generator.decode_forward(
            out_tok,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            sampling_params=sampling_params,
            reset_batch=(step == 0),
            prompt_tokens=input_tokens_prefill_pt,
            output_tokens=out_tok,
            read_from_device=False,
        )
        step_reads, read_events = generator.read_decode_output(decoded, async_read=True)
        buffered_reads.append(step_reads)
        current_pos = current_pos + 1
        steps_executed += 1

        if (step + 1) % _READ_EVERY == 0:
            for ev in read_events:
                ttnn.event_synchronize(mesh_event=ev)
            _consume(buffered_reads)
            buffered_reads = []
            if done:
                break

    if buffered_reads:
        for ev in read_events:
            ttnn.event_synchronize(mesh_event=ev)
        _consume(buffered_reads)

    decode_s = time.perf_counter() - t_decode
    return completion, prefill_s, decode_s, steps_executed


def main() -> int:
    import ttnn
    from transformers import AutoTokenizer

    from models.common.sampling import SamplingParams

    from utils.llama_ttt_presets import llama_stop_and_pad

    required_chips = MESH_SHAPE[0] * MESH_SHAPE[1]
    available_chips = len(ttnn.get_device_ids())
    if available_chips < required_chips:
        print(f"[prompt_bench] need >= {required_chips} chip(s); host exposes {available_chips}")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    stop_token_ids_seq, pad_token_id = llama_stop_and_pad(MODEL_ID)
    stop_token_ids: FrozenSet[int] = frozenset(int(t) for t in stop_token_ids_seq)

    prompts_ids = _prepare_prompts(tokenizer, TARGET_PROMPT_LENS)
    for target, ids in zip(TARGET_PROMPT_LENS, prompts_ids):
        print(f"[prompt_bench] target_len={target:>4}  actual_len={len(ids):>4}")

    # ETH dispatch is only valid on N300/T3K/N300_2x2 (Wormhole). Single-P150
    # (Blackhole) must use WORKER dispatch; ``DispatchCoreConfig()`` picks the
    # cluster-appropriate type/axis automatically (WORKER + COL on Blackhole).
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        trace_region_size=_TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )

    model_args = model = generator = tt_kv_cache = page_table = None
    try:
        _t_setup = time.perf_counter()
        model_args, model, generator, tt_kv_cache, page_table, _ = _build_model(mesh_device)
        print(f"[prompt_bench] model built in {time.perf_counter() - _t_setup:.2f}s")

        sampling_params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, seed=0)

        for target, ids in zip(TARGET_PROMPT_LENS, prompts_ids):
            print(f"\n[prompt_bench] ===== prompt_len={target} =====")
            t0 = time.perf_counter()
            completion, prefill_s, decode_s, steps = _generate_one(
                model=model,
                generator=generator,
                tt_kv_cache=tt_kv_cache,
                page_table=page_table,
                sampling_params=sampling_params,
                prompt_ids=ids,
                pad_token_id=pad_token_id,
                stop_token_ids=stop_token_ids,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            total_s = time.perf_counter() - t0
            reply = tokenizer.decode(completion, skip_special_tokens=True)
            new_tokens = len(completion)
            decode_tok_s = (steps / decode_s) if decode_s > 0 else 0.0
            print(
                f"[prompt_bench] prompt_len={len(ids):>4}  new_tokens={new_tokens:>4}  "
                f"prefill={prefill_s:.2f}s  decode={decode_s:.2f}s ({steps} steps, {decode_tok_s:.1f} tok/s)  "
                f"total={total_s:.2f}s"
            )
            print(f"[prompt_bench] reply: {reply!r}")
    finally:
        # Order matters: drop model-side refs, GC, then close the mesh.
        model = None
        generator = None
        tt_kv_cache = None
        model_args = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
