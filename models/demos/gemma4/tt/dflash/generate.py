# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-iteration DFlash generation loop, real hardware, no static torch reference
required at runtime -- everything a live caller needs (the drafter's noise-block
embeddings, RoPE cos/sin, and the sliding-window "context" input) is now built from
whatever tokens the loop itself produces, entirely ON DEVICE.

Validated mechanisms this composes (see docs/dflash_design.md and
tests/dflash/test_dflash_*.py for the individual PCC/exact-match checks):

- Steps 1-5: weight loading, context extraction, drafter forward, logits/softcap/argmax.
- Step 6 (verify.py): verify/accept/commit against the real target, one block.
- Multi-iteration sliding context: the drafter's "context" input is NOT a growing
  accumulator -- after each verify call, it is REPLACED by that verify call's own
  hidden-state taps for just the newly-committed positions (reference
  dflash/dflash.py:305). Confirmed on real hardware across two iterations, including a
  verify call landing at a non-tile-aligned position inside an already-touched KV-cache
  tile (see verify.py's module docstring for the one-time tile-alignment requirement on
  the very FIRST verify call only).
- Noise-block embedding: ``model.raw_embed`` (the target's own on-device embedding
  table, tied to the drafter's), undoing the target's baked-in sqrt(hidden) scale --
  the reference's ``_raw_input_embeddings`` is unscaled, unlike ``model.embed_tokens``.
- RoPE cos/sin: ``rope_cache.py``'s on-device gather, confirmed bit-identical (bf16,
  PCC 1.0) against the torch reference's own ``Qwen3RotaryEmbedding`` output for known
  positions -- replacing what was previously a host-torch trig computation every
  iteration (build_noise_inputs, removed).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4.tt.dflash.context import compute_context
from models.demos.gemma4.tt.dflash.drafter import dflash_drafter_forward
from models.demos.gemma4.tt.dflash.lm_head import compute_dflash_logits
from models.demos.gemma4.tt.dflash.rope_cache import build_dflash_rope_cache_2d, gather_rope_on_device
from models.demos.gemma4.tt.dflash.verify import dflash_verify, greedy_accept_from_posterior


def _to_tt(mesh_device, x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _slice_seq(t, n):
    return ttnn.slice(t, [0, 0, 0, 0], [1, 1, n, t.shape[-1]])


def _tap_context(model, weights, config, forward_fn):
    """Run ``forward_fn`` (a prefill or verify call) with ``model.layer_probe`` attached,
    returning (forward_fn's own return value, context tensor built from the taps)."""
    tapped = {}

    def _probe(layer_idx, hidden_states):
        if layer_idx in config.target_layer_ids:
            tapped[layer_idx] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

    model.layer_probe = _probe
    try:
        result = forward_fn()
    finally:
        model.layer_probe = None
    context = compute_context(weights, [tapped[i] for i in config.target_layer_ids])
    return result, context


def dflash_generate(
    model,
    mesh_device,
    weights,
    lm_head_weight,
    config,
    mesh_config,
    ccl_manager,
    tt_kv_cache,
    page_table_torch: torch.Tensor,
    input_ids_padded: torch.Tensor,
    ctx_len: int,
    max_new_tokens: int,
    stop_token_ids: list[int] | None = None,
    layer_configs=None,
):
    """Full DFlash draft->verify->accept loop starting from a real prefill.

    ``ctx_len`` (the number of real, non-padding tokens in ``input_ids_padded``) MUST be
    a multiple of 32 -- see verify.py's module docstring for the known Gemma4 kernel bug
    this avoids. ``input_ids_padded`` is the full padded-to-bucket sequence as consumed
    by ``model.prepare_inputs_prefill``/``ttnn_prefill_forward``.

    Returns (output_ids: list[int] (generated tokens only, not the prompt), acceptance_lengths: list[int]).
    """
    if ctx_len % 32 != 0:
        raise ValueError(f"dflash_generate requires ctx_len to be a multiple of 32 (got {ctx_len})")

    layer_configs = config.layer_configs if layer_configs is None else layer_configs
    block_size = config.block_size
    mask_token_id = config.mask_token_id
    head_dim = config.head_dim
    num_local_heads = config.num_attention_heads // mesh_config.tp
    num_local_kv_heads = config.num_key_value_heads // mesh_config.tp
    is_mesh = hasattr(mesh_device, "shape")
    stop_tokens = set(stop_token_ids or [])

    max_seq_len = input_ids_padded.shape[-1]
    cos_2d, sin_2d = build_dflash_rope_cache_2d(mesh_device, head_dim, config.rope_theta, max_seq_len)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    page_table_tt = ttnn.from_torch(
        page_table_torch, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )
    embeds, _, _, _, _, _ = model.prepare_inputs_prefill(input_ids_padded.unsqueeze(0), page_table=page_table_tt)

    def _prefill():
        logits = model.ttnn_prefill_forward(
            embeds,
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            get_last_token=ctx_len - 1,
            input_ids_torch=input_ids_padded.unsqueeze(0),
            embeds_torch=None,
        )
        return logits

    prefill_logits, context_padded = _tap_context(model, weights, config, _prefill)
    logits_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(prefill_logits)[0]) if is_mesh else ttnn.to_torch(prefill_logits)
    )
    ttnn.deallocate(prefill_logits)
    tile_start = ((ctx_len - 1) // 32) * 32
    real_first_token = int(
        torch.argmax(logits_torch.float().reshape(1, -1, logits_torch.shape[-1])[0, ctx_len - 1 - tile_start]).item()
    )

    context_tt = _slice_seq(context_padded, ctx_len)
    ttnn.deallocate(context_padded)
    context_len = ctx_len  # length of the sliding "context" window feeding RoPE position math

    # output_ids[0] (real_first_token) is a genuine generated token (position ctx_len,
    # sampled from prefill's own last-token logits) -- it counts toward max_new_tokens
    # like every other element, and is included in the returned sequence.
    output_ids = [real_first_token]
    acceptance_lengths = []
    start = ctx_len
    stopped = real_first_token in stop_tokens

    while len(output_ids) < max_new_tokens and not stopped:
        verify_size = block_size  # always request a full block; excess is truncated below

        if verify_size > 1:
            pre_draft_ids = torch.tensor([[output_ids[-1]] + [mask_token_id] * (verify_size - 1)], dtype=torch.long)
            pre_draft_ids_tt = ttnn.from_torch(
                pre_draft_ids,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                device=mesh_device,
                mesh_mapper=replicate,
            )
            noise_tt = model.raw_embed(pre_draft_ids_tt)
            ttnn.deallocate(pre_draft_ids_tt)
            if len(noise_tt.shape) != 4:
                noise_tt = ttnn.unsqueeze_to_4D(noise_tt)
            if noise_tt.layout != ttnn.TILE_LAYOUT:
                noise_tt = ttnn.to_layout(noise_tt, ttnn.TILE_LAYOUT)

            positions = list(range(start - context_len, start + verify_size))
            cos_tt, sin_tt = gather_rope_on_device(mesh_device, positions, cos_2d, sin_2d, head_dim)

            drafter_out = dflash_drafter_forward(
                context_tt,
                noise_tt,
                weights,
                cos_tt,
                sin_tt,
                mesh_device,
                mesh_config,
                ccl_manager,
                num_local_heads,
                num_local_kv_heads,
                head_dim,
                config.rms_norm_eps,
                layer_configs,
            )
            final_out = weights.norm(drafter_out)
            drafter_logits = compute_dflash_logits(
                final_out, lm_head_weight, mesh_device, config.final_logit_softcapping
            )
            drafter_logits = drafter_logits.reshape(1, verify_size, -1)
            draft_tokens = torch.argmax(drafter_logits, dim=-1)[0, 1:].tolist()
        else:
            draft_tokens = []

        candidate_ids = [output_ids[-1]] + draft_tokens

        def _verify():
            return dflash_verify(model, mesh_device, tt_kv_cache, page_table_torch, candidate_ids, start_pos=start)

        (posterior, _), next_context_padded = _tap_context(model, weights, config, _verify)
        accept, bonus, committed = greedy_accept_from_posterior(candidate_ids, posterior)
        produced = min(accept + 1, verify_size)

        new_tokens = committed[1:]  # committed[0] == output_ids[-1], already recorded
        remaining_budget = max_new_tokens - len(output_ids)
        new_tokens = new_tokens[:remaining_budget]
        for tok in new_tokens:
            output_ids.append(tok)
            if tok in stop_tokens:
                stopped = True
                break
        acceptance_lengths.append(accept)

        context_tt = _slice_seq(next_context_padded, produced)
        ttnn.deallocate(next_context_padded)
        context_len = produced
        start += produced

        if stopped or len(output_ids) >= max_new_tokens:
            break

    return output_ids, acceptance_lengths
