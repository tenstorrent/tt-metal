# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Demo: DFlash speculative-decoding generation via ``DFlashFusedDecoder``
(tt/dflash_drafter.py) -- the same B=1 fused-trace loop that backs
``Gemma4DFlashForCausalLM`` in vLLM serving (tt/generator_vllm.py).

Replicates ``Gemma4DFlashForCausalLM``'s exact prefill_forward /
_spec_get_drafter / _spec_bootstrap / decode_forward sequencing directly
against a real ``Gemma4Generator`` target model, so it can be run and watched
without standing up vLLM. Prints the generated text plus a few headline
numbers (tok/s, mean accepted-drafts/iteration).

Requires HF_MODEL (target, e.g. google/gemma-4-31B-it) and the z-lab dFlash
drafter snapshot in the HF cache (auto-discovered) or GEMMA4_DFLASH_DRAFTER.

GEMMA4_DFLASH_SHARD_ARGMAX=1 is required: the default sampling path reuses
the target model's shared TTSampling module, whose route_full_row=False
branch does a broadcast-add (models/common/sampling/tt_sampling.py) that
TT_THROWs ("Invalid subtile broadcast type") on the drafter's multi-row block
logits -- a pre-existing, documented TTSampling limitation (see
spec_decode.py's own negative-results notes on scalar-wise broadcast), not
specific to this demo. This env var routes dFlash through its own
purpose-built ``_shard_argmax`` on-device argmax instead.

Run:
    HF_MODEL=google/gemma-4-31B-it GEMMA4_DFLASH_SHARD_ARGMAX=1 pytest \
        models/demos/gemma4/demo/dflash_fused_decoder_demo.py -k 1x8 -s

    # Different prompt / generation length:
    GEMMA4_DFLASH_PROMPT="..." GEMMA4_DFLASH_MAX_NEW=128 pytest \
        models/demos/gemma4/demo/dflash_fused_decoder_demo.py -k 1x8 -s
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

from ..tests.test_factory import parametrize_mesh_with_fabric

DEFAULT_PROMPT = (
    "Write a Python function that checks whether a given string is a "
    "palindrome, ignoring spaces, punctuation and case. Include a short "
    "docstring and two example calls."
)
MAX_SEQ_LEN = 2048
PAGE_BLOCK_SIZE = 64


def _dflash_default_snapshot():
    """Locate the z-lab drafter snapshot in the HF cache (mirrors
    generator_vllm.py's own helper -- reimplemented here rather than imported,
    since importing generator_vllm.py pulls in tt_transformers'
    generator_vllm.py -> vllm, which some dev environments don't have."""
    import glob

    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/*/"))
    return hits[0] if hits else None


def _unwrap_kv_layers(kv_cache):
    """Same defensive unwrap as Gemma4DFlashForCausalLM._spec_bootstrap."""
    kv_layers = kv_cache
    if (
        isinstance(kv_layers, (list, tuple))
        and kv_layers
        and isinstance(kv_layers[0], (list, tuple))
        and kv_layers[0]
        and isinstance(kv_layers[0][0], (list, tuple))
    ):
        kv_layers = kv_layers[0]
    return kv_layers


@parametrize_mesh_with_fabric(
    [(1, 8)],
    device_params_extra={
        "trace_region_size": 256_000_000,
        # CCL all_gather allocates semaphores in L1_SMALL when this is > 0 --
        # without it, they fragment the main L1 pool. At the 1024-token
        # prefill bucket (anything >128 real tokens rounds up to it) this was
        # observed to TT_THROW "Statically allocated circular buffers...
        # clash with L1 buffers" during prefill warmup -- a general
        # Gemma4-31B TP=8 issue, not specific to dFlash's own tap-capture
        # (confirmed: the plain, non-dFlash text_demo.py::test_demo hit the
        # identical TT_THROW at the identical L1 addresses at this bucket
        # before the same fix was applied there too). Same fix/value as
        # text_demo_v2.py's ``_device_params``.
        "l1_small_size": int(os.environ.get("GEMMA4_L1_SMALL_SIZE", 24576)),
    },
)
def test_demo_dflash_fused_decoder(mesh_device, device_params, reset_seeds):
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter, DFlashFusedDecoder
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    weights_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if not weights_dir:
        # _spec_get_drafter needs this to load the target's own embed_tokens
        # weight for the drafter's tied lm_head -- point it at the target
        # snapshot dir when the caller hasn't set it explicitly.
        import glob

        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*--{model_path.split('/')[-1]}/snapshots/*/")
        )
        weights_dir = hits[0].rstrip("/") if hits else None
    if not weights_dir or not os.path.isdir(weights_dir):
        pytest.skip(f"MODEL_WEIGHTS_DIR not found/set (tried {weights_dir!r}); set it explicitly")

    snap = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
    if not snap:
        pytest.skip("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")

    if os.environ.get("GEMMA4_DFLASH_SHARD_ARGMAX") != "1":
        pytest.skip(
            "set GEMMA4_DFLASH_SHARD_ARGMAX=1 -- required to route around a known TTSampling "
            "multi-row broadcast limitation (see module docstring)"
        )

    prompt = os.environ.get("GEMMA4_DFLASH_PROMPT", DEFAULT_PROMPT)
    max_new = int(os.environ.get("GEMMA4_DFLASH_MAX_NEW", 256))

    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    model0 = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)

    def _embed_loader():
        import json

        from safetensors import safe_open

        idx = json.load(open(f"{weights_dir}/model.safetensors.index.json"))
        key = next(
            k
            for k in idx["weight_map"]
            if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
        )
        with safe_open(f"{weights_dir}/{idx['weight_map'][key]}", framework="pt") as f:
            return f.get_tensor(key)

    drafter = DFlashDrafter(
        mesh_device=mesh_device,
        drafter_path=snap,
        target_embed_weight_loader=_embed_loader,
        mesh_config=model0.mesh_config,
        ccl_manager=model0.ccl_manager,
        tensor_cache_path=None,
    )

    # Chat/instruct-formatted prompt: what the dFlash drafter was validated
    # against (dflash_drafter.py module docstring: "mean greedy acceptance
    # ~3-4.3 of a 16 block") and what real vLLM serving always feeds it. A raw
    # continuation prompt is off-distribution for this it-tuned pair and
    # degenerates to near-zero acceptance.
    in_pt, encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, max_new + 32, max_prefill_len=MAX_SEQ_LEN
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    n = int(decoding_pos[0])
    anchor_token = int(encoded[0][n - 1])

    logger.info("=" * 70)
    logger.info(f"DFlash fused-decoder demo -- prompt: {prompt!r}")
    logger.info(f"prompt tokens: {n}  |  max_new: {max_new}  |  model: {model_path}")
    logger.info("=" * 70)

    # Real prefill with dFlash tap capture (Gemma4DFlashForCausalLM.prefill_forward):
    # enable_trace=False is required -- the residual taps are captured by a
    # python hook in the eager forward, which a traced replay skips.
    model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
    try:
        generator.prefill_forward_text(
            in_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=False,
            warmup_prefill=False,
        )
    finally:
        taps = model0.pop_dflash_taps()
        model0.dflash_capture_taps(None)

    # Fused decoder bootstrap (Gemma4DFlashForCausalLM._spec_bootstrap): one-time
    # drafter ctx ingest + fused-trace compile/capture.
    kv_layers = _unwrap_kv_layers(tt_kv_cache)
    dec = DFlashFusedDecoder(model0, drafter, kv_layers, page_table[:1])
    dec.prefill_ingest(taps, n)
    t_cap0 = time.perf_counter()
    dec.capture(anchor_token, n, max_new=max_new)
    logger.info(f"trace capture (one-time compile): {time.perf_counter() - t_cap0:.1f}s")

    # Steady-state decode loop (Gemma4DFlashForCausalLM.decode_forward's
    # internal block loop): call .step() back to back until max_new tokens
    # have been committed.
    eos = getattr(model0.hf_config, "eos_token_id", 1)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
    vocab = drafter.vocab

    committed = []
    iters = 0
    first = True
    hit_eos = False
    t_dec0 = time.perf_counter()
    while len(committed) < max_new:
        accepted, bonus, _produced = dec.step(first=first)
        first = False
        iters += 1
        toks = list(accepted) + [bonus]
        toks = [t if 0 <= t < vocab else int(bonus if 0 <= bonus < vocab else 1) for t in toks]
        committed.extend(toks)
        if eos_set & set(toks):
            hit_eos = True
            break
    wall = time.perf_counter() - t_dec0

    tps = len(committed) / wall if wall > 0 else float("nan")
    avg_accept = (len(committed) - iters) / iters if iters else float("nan")  # bonus token excluded per iter
    text = tokenizer.decode(committed)

    logger.info("=" * 70)
    logger.info("Generated text:")
    logger.info(text)
    logger.info("=" * 70)
    logger.info(
        f"{len(committed)} tokens in {iters} dFlash iterations, {wall:.2f}s "
        f"-> {tps:.1f} tok/s  |  mean accepted-drafts/iter: {avg_accept:.2f}  |  hit_eos={hit_eos}"
    )
    logger.info("=" * 70)
