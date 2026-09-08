# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-iteration DFlash generation loop, real hardware, no static torch reference
required at runtime -- everything a live caller needs (the drafter's noise-block
embeddings, RoPE cos/sin, and the drafter's context input) is now built from whatever
tokens the loop itself produces, entirely ON DEVICE.

Validated mechanisms this composes (see docs/dflash_design.md and
tests/dflash/test_dflash_*.py for the individual PCC/exact-match checks):

- Steps 1-5: weight loading, context extraction, drafter forward, logits/softcap/argmax.
- Step 6 (verify.py): verify/accept/commit against the real target, one block.
- Multi-iteration GROWING context via a genuinely INCREMENTAL per-layer K/V cache: the
  drafter attends to the full history back to prefill (never a bounded recent window),
  matching the reference's own growing ``past_key_values_draft`` -- see
  attention.py's module docstring and ``project_and_cache_context_delta`` for the
  mechanism, and drafter.py's ``dflash_drafter_update_kv_caches`` for how this file
  drives it once per iteration. Each drafter layer holds its OWN persistent
  ``[1,num_local_kv_heads,max_seq_len,head_dim]`` (k_cache, v_cache) pair -- real content
  in the first ``context_len`` rows (growing every iteration), masked-out padding in the
  rest via ``context_valid_len_tt`` (a fixed-size buffer with a dynamic valid-length
  mask, rather than a tensor that changes shape every iteration, is what keeps this
  compatible with Metal trace capture -- see ``_traced_steady_state``). Only the NEW
  iteration's own delta (produced-length rows) is ever projected/normed/RoPE'd each
  call -- older rows are read straight from cache, never recomputed -- matching the
  reference's actual O(new tokens) per-iteration compute profile rather than
  reprojecting the WHOLE history every call (an earlier version of this file did that;
  correct but ~2.7x slower in eager mode, see docs/dflash_design.md).
- Noise-block embedding: ``model.raw_embed`` (the target's own on-device embedding
  table, tied to the drafter's), undoing the target's baked-in sqrt(hidden) scale --
  the reference's ``_raw_input_embeddings`` is unscaled, unlike ``model.embed_tokens``.
- RoPE cos/sin: ``rope_cache.py``'s on-device gather, confirmed bit-identical (bf16,
  PCC 1.0) against the torch reference's own ``Qwen3RotaryEmbedding`` output for known
  positions -- replacing what was previously a host-torch trig computation every
  iteration (build_noise_inputs, removed).

KNOWN LATENT LIMITATION (not fixed here, believed pre-existing and currently
unobservable at any tested scale): the attention mask's causal/sliding-window position
grid is built relative to EACH CALL's own [context, noise] layout (``arange(0, ctx_len)``
for context, ``arange(ctx_len, ctx_len+q_len)`` for noise), not the sequence's true
absolute positions. Since context rows here always equal true absolute position (writes
only ever append starting at row 0), causal-among-noise and the valid-length check are
unaffected (the mismatch cancels out or doesn't apply), but the SLIDING-WINDOW distance
check for a noise query attending a context key is off by a constant
``max_seq_len - start`` that grows as generation proceeds -- benign while
``sliding_window`` (2048) comfortably exceeds ``max_seq_len`` (as in every config tested
so far), but would incorrectly under-mask or over-mask context once ``max_seq_len``
approaches or exceeds ``sliding_window`` in a longer-context production deployment.
Fixing it needs the mask's noise-side query position to depend on a per-iteration
DYNAMIC ``start`` tensor rather than a static arange -- straightforward given the same
dynamic-tensor-input pattern ``context_valid_len_tt`` already uses, just not done in this
change to keep this fix scoped to the missing-context-accumulation bug.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.gemma4.tt.dflash.attention import build_attention_mask_static_parts
from models.demos.gemma4.tt.dflash.context import ContextAccumulator, split_fc_slices
from models.demos.gemma4.tt.dflash.drafter import dflash_drafter_forward, dflash_drafter_update_kv_caches
from models.demos.gemma4.tt.dflash.lm_head import argmax_last_dim, compute_dflash_argmax
from models.demos.gemma4.tt.dflash.rope_cache import (
    build_dflash_rope_cache_2d,
    gather_rope_from_buffer,
    gather_rope_on_device,
    gather_rope_on_device_buffered,
    make_rope_gather_index_buffer,
    refresh_rope_gather_buffer,
)
from models.demos.gemma4.tt.dflash.verify import (
    greedy_accept_from_posterior,
    make_verify_buffers,
    refresh_verify_positions,
    run_verify_forward,
)


def _to_tt(mesh_device, x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def make_drafter_kv_caches(mesh_device, num_layers: int, num_local_kv_heads: int, max_seq_len: int, head_dim: int):
    """One persistent, zero-initialized ``(k_cache, v_cache)`` pair per drafter layer --
    each ``[1,num_local_kv_heads,max_seq_len,head_dim]``, real content in the first
    ``context_len`` rows (see attention.py's ``project_and_cache_context_delta``), the
    rest masked-out padding via ``context_valid_len_tt``."""
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None

    def _buf():
        return ttnn.from_torch(
            torch.zeros((1, num_local_kv_heads, max_seq_len, head_dim), dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

    return [(_buf(), _buf()) for _ in range(num_layers)]


def _tap_context(model, weights, fc_slices, forward_fn):
    """Run ``forward_fn`` (a prefill or verify call) with ``model.layer_probe`` attached,
    returning (forward_fn's own return value, context tensor built from the taps).

    Uses ContextAccumulator (FC-decomposed, accumulate-as-you-go) instead of collecting
    full hidden-state tensors into a Python dict -- confirmed PCC 0.9996 against the
    dict-based compute_context for a real prefill (bf16 summation-order noise, not a
    correctness difference; see context.py). A dict that grows across calls is exactly
    the shape a Metal trace can't replay; an accumulator that always holds at most one
    running total is the shape one can -- this doesn't itself enable tracing yet (the
    loop below still runs eager), but it's the prerequisite piece for the fused trace a
    later step will build."""
    accumulator = ContextAccumulator(fc_slices, weights.hidden_norm)

    def _probe(layer_idx, hidden_states):
        accumulator.tap(hidden_states, layer_idx)

    model.layer_probe = _probe
    try:
        result = forward_fn()
    finally:
        model.layer_probe = None
    context = accumulator.finalize()
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
    use_trace: bool = False,
):
    """Full DFlash draft->verify->accept loop starting from a real prefill.

    ``ctx_len`` (the number of real, non-padding tokens in ``input_ids_padded``) MUST be
    a multiple of 32 -- see verify.py's module docstring for the known Gemma4 kernel bug
    this avoids. ``input_ids_padded`` is the full padded-to-bucket sequence as consumed
    by ``model.prepare_inputs_prefill``/``ttnn_prefill_forward``.

    ``use_trace``: after the first block (always eager, to get a real set of steady-state
    inputs to compile/capture against), capture the steady-state iteration ONCE as a Metal trace
    (following spec_decode.py's ``_capture_fused_trace``/``_generate_fused_traced``
    pattern) and replay it for every subsequent block instead of re-issuing the whole op
    sequence eagerly each time. Requires the fixed-size dynamically-masked K/V caches
    (see attention.py/drafter.py) -- a trace's tensor shapes must stay fixed across every
    replay, which the steady state now guarantees.

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
    # The drafter's OWN K/V cache and attention mask never need to be wider than the
    # largest absolute position this session's REAL (committed) content could ever reach
    # (ctx_len + max_new_tokens) -- often meaningfully smaller than max_seq_len, which is
    # sized for the TARGET's own paged-attention KV cache, a session-wide bound unrelated
    # to how many tokens THIS call actually asked for. SDPA cost scales with cache width
    # regardless of how much of it is real content vs masked-out padding, so tightening
    # this is a genuine, safe speedup: it changes no result (the extra columns were
    # always masked out and unattended either way), only how much oversized padding gets
    # computed at all. Capped at max_seq_len since real content can never exceed what the
    # target's own KV cache supports.
    drafter_max_seq_len = min(max_seq_len, ctx_len + max_new_tokens)
    # The RoPE table, by contrast, must NOT be capped at max_seq_len: the noise block's
    # own gather reaches up to start+block_size-1, i.e. up to block_size-1 positions
    # PAST the last real committed position -- these are speculative draft positions
    # that never get written into the target's own KV cache (so max_seq_len's hardware
    # constraint doesn't apply to them), but the drafter's forward pass still computes
    # RoPE for the full noise block every call regardless of how many of those
    # predictions end up used. Building this table only max_seq_len rows wide (as an
    # earlier version of this file did) let that gather read past the table's end
    # whenever ctx_len+max_new_tokens landed close to max_seq_len -- confirmed as a real
    # bug via repeated runs: undefined/uninitialized memory read there produced
    # correct-looking output most of the time but occasionally flipped a token deep into
    # a long generation, with no reliable reproduction (consistent with reading
    # whatever happened to be at an address, rather than a deterministic logic error).
    rope_table_len = max(max_seq_len, ctx_len + max_new_tokens + block_size)
    cos_2d, sin_2d = build_dflash_rope_cache_2d(mesh_device, head_dim, config.rope_theta, rope_table_len)

    # Persistent, reused-every-iteration device buffers (see verify.py's module
    # docstring): allocated once here, refreshed in place via
    # copy_host_to_device_tensor inside the loop below instead of a fresh
    # ttnn.from_torch(..., device=...) allocation each call.
    verify_buffers = make_verify_buffers(mesh_device, page_table_torch, block_size)
    # anchor_buf holds ONLY the current iteration's anchor token id -- refreshed from
    # host each iteration (the anchor is the previous iteration's bonus token, only
    # known after that iteration's host-side accept decision; this one small host round
    # trip per iteration is inherent, not avoidable -- see verify.py's
    # greedy_accept_from_posterior docstring). mask_tail_buf is genuinely CONSTANT
    # (always mask_token_id, block_size-1 wide) and never refreshed after this. Both the
    # noise-block embedding input and verify's candidate-ids input are built by
    # concatenating anchor_buf with something already on device (mask_tail_buf, or the
    # drafter's own argmax output) -- never by rebuilding a token-id list on host and
    # re-uploading it, unlike the previous pre_draft_buf/candidate_ids-list approach.
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    anchor_buf = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )
    mask_tail_buf = ttnn.from_torch(
        torch.full((1, block_size - 1), mask_token_id, dtype=torch.int64),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mesh_mapper,
    )

    def _write_anchor(token_id: int) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[token_id]], dtype=torch.int64),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=mesh_mapper,
            ),
            anchor_buf,
        )

    # context_valid_len_buf: how many of each layer's max_seq_len-row K/V cache are real
    # so far (the rest is masked-out padding, see attention.py's
    # build_attention_mask_additive_device_dynamic) -- refreshed from host every
    # iteration including the first, since this cumulative count only ever grows. This is
    # the SECOND (and last) per-iteration host->device write, alongside anchor_buf.
    context_valid_len_buf = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mesh_mapper,
    )

    def _write_context_valid_len(n: int) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[n]], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=mesh_mapper,
            ),
            context_valid_len_buf,
        )

    # One persistent (k_cache, v_cache) pair per drafter layer -- see module docstring.
    # RoPE for the steady state only ever needs the NOISE block's own block_size-wide
    # position range: context's own K is already RoPE'd once, at cache-write time (see
    # project_and_cache_context_delta), and never re-rotated when read back later.
    kv_caches = make_drafter_kv_caches(
        mesh_device, len(layer_configs), num_local_kv_heads, drafter_max_seq_len, head_dim
    )
    rope_idx_buf = make_rope_gather_index_buffer(mesh_device, block_size)
    # weights.fc sliced into one [hidden,hidden] block per tapped layer -- constant for
    # the whole session, built once (see context.py::ContextAccumulator).
    fc_slices = split_fc_slices(weights, config.target_layer_ids, config.hidden_size)

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

    prefill_logits, context_padded = _tap_context(model, weights, fc_slices, _prefill)
    # prefill_logits is a 32-row tile (get_last_token slices to the tile containing
    # ctx_len-1, see model.py) -- slice down to just that one real row on device before
    # arguing, so the (262144-wide) vocab is never read to host.
    tile_start = ((ctx_len - 1) // 32) * 32
    row = ctx_len - 1 - tile_start
    vocab = prefill_logits.shape[-1]
    row_logits = ttnn.slice(prefill_logits, [0, 0, row, 0], [1, 1, row + 1, vocab])
    ttnn.deallocate(prefill_logits)
    real_first_id_tt = argmax_last_dim(row_logits, 1)
    ttnn.deallocate(row_logits)
    real_first_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(real_first_id_tt)[0]) if is_mesh else ttnn.to_torch(real_first_id_tt)
    )
    ttnn.deallocate(real_first_id_tt)
    real_first_token = int(real_first_torch.reshape(-1)[0].item())
    _write_anchor(real_first_token)

    # Seed every layer's K/V cache with the real prefill's own context (a ONE-OFF, wider
    # write -- ctx_len can exceed block_size, unlike every later iteration's delta).
    cos_prefill, sin_prefill = gather_rope_on_device(mesh_device, list(range(0, ctx_len)), cos_2d, sin_2d, head_dim)
    dflash_drafter_update_kv_caches(
        context_padded,
        ctx_len,
        weights,
        cos_prefill,
        sin_prefill,
        kv_caches,
        offset=0,
        num_local_heads=num_local_heads,
        num_local_kv_heads=num_local_kv_heads,
        head_dim=head_dim,
        eps=config.rms_norm_eps,
    )
    ttnn.deallocate(context_padded)
    ttnn.deallocate(cos_prefill)
    ttnn.deallocate(sin_prefill)
    # context_len: cumulative count of real (non-padding) rows in every layer's cache --
    # NOT a sliding window's width. Every iteration appends its newly-committed-token tap
    # here (see module docstring); context_len only ever grows, in lockstep with `start`.
    context_len = ctx_len

    # output_ids[0] (real_first_token) is a genuine generated token (position ctx_len,
    # sampled from prefill's own last-token logits) -- it counts toward max_new_tokens
    # like every other element, and is included in the returned sequence.
    output_ids = [real_first_token]
    acceptance_lengths = []
    start = ctx_len
    stopped = real_first_token in stop_tokens
    first_iteration = True

    # When use_trace, this loop runs ONLY the first (always-eager) block, then exits --
    # first_iteration becomes False at the end of that pass, making the condition below
    # false. When not use_trace, it's unrestricted, exactly as before this parameter
    # existed.
    while len(output_ids) < max_new_tokens and not stopped and (first_iteration or not use_trace):
        # ---- everything below builds each iteration's full op sequence using ONLY
        # persistent buffers and on-device concatenation -- anchor_buf (every iteration)
        # and context_valid_len_buf (every iteration after the first) are the ONLY
        # per-iteration host->device writes; the drafter's own draft-token predictions
        # never round-trip to host to get fed back into verify's input, only read back
        # once at the very end for the accept decision itself. ----
        noise_tt = model.raw_embed(ttnn.concat([anchor_buf, mask_tail_buf], dim=-1))
        if len(noise_tt.shape) != 4:
            noise_tt = ttnn.unsqueeze_to_4D(noise_tt)
        if noise_tt.layout != ttnn.TILE_LAYOUT:
            noise_tt = ttnn.to_layout(noise_tt, ttnn.TILE_LAYOUT)

        # Only the noise block's own RoPE positions are needed here -- context's K is
        # already RoPE'd once, at cache-write time, and read as-is (see module docstring).
        positions_noise = list(range(start, start + block_size))
        cos_tt, sin_tt = gather_rope_on_device_buffered(
            mesh_device, rope_idx_buf, positions_noise, cos_2d, sin_2d, head_dim
        )
        _write_context_valid_len(context_len)

        drafter_out = dflash_drafter_forward(
            kv_caches,
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
            drafter_max_seq_len,
            context_valid_len_tt=context_valid_len_buf,
        )
        final_out = weights.norm(drafter_out)
        draft_ids_tt = compute_dflash_argmax(
            final_out, lm_head_weight, mesh_device, mesh_config, ccl_manager, config.final_logit_softcapping
        )  # [1,1,block_size] uint32, stays on device

        # verify's candidate ids, built ON DEVICE: [anchor, draft_1, ..., draft_{block_size-1}]
        # -- draft_ids_tt's own row 0 (its prediction for the anchor slot) is dropped,
        # matching the original [0, 1:] slice, just done as a device op instead of a
        # host list slice.
        draft_ids_2d = ttnn.reshape(draft_ids_tt, [1, block_size])
        draft_tail_tt = ttnn.slice(draft_ids_2d, [0, 1], [1, block_size])
        candidate_ids_tt = ttnn.concat([anchor_buf, draft_tail_tt], dim=-1)

        refresh_verify_positions(mesh_device, verify_buffers, start)

        def _verify():
            logits, hidden = run_verify_forward(model, mesh_device, tt_kv_cache, verify_buffers, candidate_ids_tt)
            ttnn.deallocate(hidden)
            vocab = logits.shape[-1]
            if logits.shape[2] != block_size:
                sliced = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, block_size, vocab])
                ttnn.deallocate(logits)
                logits = sliced
            posterior_tt = argmax_last_dim(logits, block_size)
            ttnn.deallocate(logits)
            return posterior_tt

        posterior_tt, next_context_padded = _tap_context(model, weights, fc_slices, _verify)

        # ---- the one place per iteration a host readback is unavoidable: the accept
        # decision (how many tokens to keep) has to be visible to this host loop. Both
        # small results are read back together here, not interleaved earlier. ----
        draft_ids_torch = (
            ttnn.to_torch(ttnn.get_device_tensors(draft_ids_tt)[0]) if is_mesh else ttnn.to_torch(draft_ids_tt)
        )
        ttnn.deallocate(draft_ids_tt)
        draft_tokens = draft_ids_torch.reshape(-1).tolist()[1:]

        posterior_torch = (
            ttnn.to_torch(ttnn.get_device_tensors(posterior_tt)[0]) if is_mesh else ttnn.to_torch(posterior_tt)
        )
        ttnn.deallocate(posterior_tt)
        posterior = posterior_torch.reshape(1, -1)[:, :block_size].long()

        candidate_ids = [output_ids[-1]] + draft_tokens
        accept, bonus, committed = greedy_accept_from_posterior(candidate_ids, posterior)
        produced = min(accept + 1, block_size)

        new_tokens = committed[1:]  # committed[0] == output_ids[-1], already recorded
        remaining_budget = max_new_tokens - len(output_ids)
        new_tokens = new_tokens[:remaining_budget]
        for tok in new_tokens:
            output_ids.append(tok)
            if tok in stop_tokens:
                stopped = True
                break
        acceptance_lengths.append(accept)

        # next_context_padded is ALREADY exactly block_size-wide (ttnn_verify_forward
        # always processes exactly block_size candidates); its first `produced` rows are
        # real. Project THROUGH EVERY LAYER's own k_proj/v_proj/k_norm + RoPE (reusing
        # this iteration's own cos_tt/sin_tt -- the delta's positions are exactly the
        # first `produced` rows of noise's own range, see module docstring) and APPEND
        # into each layer's cache at the current context_len offset -- growing the
        # accumulator, never replacing it -- then advance context_len/start together
        # (they stay equal: context_len is always the count of real rows accumulated so
        # far, which is exactly the next absolute position).
        dflash_drafter_update_kv_caches(
            next_context_padded,
            produced,
            weights,
            cos_tt,
            sin_tt,
            kv_caches,
            offset=context_len,
            num_local_heads=num_local_heads,
            num_local_kv_heads=num_local_kv_heads,
            head_dim=head_dim,
            eps=config.rms_norm_eps,
        )
        ttnn.deallocate(next_context_padded)
        context_len += produced
        start += produced
        first_iteration = False

        if stopped or len(output_ids) >= max_new_tokens:
            break

        _write_anchor(bonus)  # next iteration's anchor -- skipped on the final iteration

    if use_trace and not stopped and len(output_ids) < max_new_tokens:
        output_ids, acceptance_lengths = _traced_steady_state(
            model=model,
            mesh_device=mesh_device,
            weights=weights,
            lm_head_weight=lm_head_weight,
            config=config,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            tt_kv_cache=tt_kv_cache,
            layer_configs=layer_configs,
            verify_buffers=verify_buffers,
            anchor_buf=anchor_buf,
            mask_tail_buf=mask_tail_buf,
            context_valid_len_buf=context_valid_len_buf,
            rope_idx_buf=rope_idx_buf,
            cos_2d=cos_2d,
            sin_2d=sin_2d,
            fc_slices=fc_slices,
            kv_caches=kv_caches,
            context_len=context_len,
            start=start,
            output_ids=output_ids,
            acceptance_lengths=acceptance_lengths,
            stop_tokens=stop_tokens,
            max_new_tokens=max_new_tokens,
            max_seq_len=drafter_max_seq_len,
            block_size=block_size,
            head_dim=head_dim,
            num_local_heads=num_local_heads,
            num_local_kv_heads=num_local_kv_heads,
            is_mesh=is_mesh,
            write_anchor=_write_anchor,
            write_context_valid_len=_write_context_valid_len,
        )

    return output_ids, acceptance_lengths


def _traced_steady_state(
    *,
    model,
    mesh_device,
    weights,
    lm_head_weight,
    config,
    mesh_config,
    ccl_manager,
    tt_kv_cache,
    layer_configs,
    verify_buffers,
    anchor_buf,
    mask_tail_buf,
    context_valid_len_buf,
    rope_idx_buf,
    cos_2d,
    sin_2d,
    fc_slices,
    kv_caches,
    context_len,
    start,
    output_ids,
    acceptance_lengths,
    stop_tokens,
    max_new_tokens,
    max_seq_len,
    block_size,
    head_dim,
    num_local_heads,
    num_local_kv_heads,
    is_mesh,
    write_anchor,
    write_context_valid_len,
):
    """Capture the steady-state iteration as ONE Metal trace, replay it for every
    subsequent block -- mirrors spec_decode.py's ``_capture_fused_trace``/
    ``_generate_fused_traced`` exactly: a compile pass (eager, warms the program cache)
    using the REAL first steady-state inputs (``kv_caches`` already hold everything up
    through iteration 0's own commit at this point), then
    ``begin_trace_capture``/``end_trace_capture`` binds that same real call's result to
    persistent output buffers, then a replay loop refreshes only the small per-iteration
    inputs between ``execute_trace`` calls -- each layer's cache is never reassigned
    after this point; its CONTENTS grow in place every replay
    (``dflash_drafter_update_kv_caches``, appending that replay's own newly-tapped rows
    at the current ``context_len`` offset -- never replacing what's already there),
    exactly matching the main loop's own accumulation (see module docstring)."""
    from loguru import logger as _lg

    # Static mask parts (position grids, causal/sliding base visibility, noise-column
    # exemption, zero/neg constants) built ONCE here, outside capture -- ctx_len==max_seq_len,
    # q_len==block_size, and every (is_causal, sliding_window) pair are fixed for the whole
    # steady-state loop. Only the context_valid_len_tt-dependent recombine (inside
    # _fused_body, via combine_attention_mask_dynamic) needs to re-run every replay. Building
    # these inside _fused_body instead (the original approach) uses ttnn.arange/ones/zeros/full
    # every call -- each does a host->device write, which begin_trace_capture rejects
    # (TT_FATAL: Writes are not supported during trace capture) -- confirmed as the actual
    # cause of this trace's first capture failure.
    distinct_configs = set(layer_configs)
    mask_static_parts = {
        cfg: build_attention_mask_static_parts(mesh_device, max_seq_len, block_size, cfg[0], cfg[1])
        for cfg in distinct_configs
    }

    def _fused_body():
        noise_tt = model.raw_embed(ttnn.concat([anchor_buf, mask_tail_buf], dim=-1))
        if len(noise_tt.shape) != 4:
            noise_tt = ttnn.unsqueeze_to_4D(noise_tt)
        if noise_tt.layout != ttnn.TILE_LAYOUT:
            noise_tt = ttnn.to_layout(noise_tt, ttnn.TILE_LAYOUT)

        cos_tt, sin_tt = gather_rope_from_buffer(rope_idx_buf, cos_2d, sin_2d, head_dim)

        drafter_out = dflash_drafter_forward(
            kv_caches,
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
            max_seq_len,
            context_valid_len_tt=context_valid_len_buf,
            mask_static_parts=mask_static_parts,
        )
        final_out = weights.norm(drafter_out)
        draft_ids = compute_dflash_argmax(
            final_out, lm_head_weight, mesh_device, mesh_config, ccl_manager, config.final_logit_softcapping
        )
        draft_ids_2d = ttnn.reshape(draft_ids, [1, block_size])
        draft_tail = ttnn.slice(draft_ids_2d, [0, 1], [1, block_size])
        candidate_ids_tt = ttnn.concat([anchor_buf, draft_tail], dim=-1)

        accumulator = ContextAccumulator(fc_slices, weights.hidden_norm)

        def _probe(layer_idx, hidden_states):
            accumulator.tap(hidden_states, layer_idx)

        model.layer_probe = _probe
        try:
            logits, hidden = run_verify_forward(model, mesh_device, tt_kv_cache, verify_buffers, candidate_ids_tt)
        finally:
            model.layer_probe = None
        ttnn.deallocate(hidden)
        vocab = logits.shape[-1]
        if logits.shape[2] != block_size:
            sliced = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, block_size, vocab])
            ttnn.deallocate(logits)
            logits = sliced
        posterior_tt = argmax_last_dim(logits, block_size)
        ttnn.deallocate(logits)
        next_context = accumulator.finalize()

        # cos_tt/sin_tt are returned too: the eager cache-update step between replays
        # (below) needs THIS SAME replay's noise-position RoPE values (the delta's
        # positions are a prefix of noise's own range, see module docstring), and can't
        # recompute them itself without re-reading rope_idx_buf's contents at a point
        # where they may have already been refreshed for the NEXT iteration.
        return draft_ids, posterior_tt, next_context, cos_tt, sin_tt

    def _refresh_inputs():
        # Only the noise block's own RoPE positions are needed -- context's K is already
        # RoPE'd once, at cache-write time, and read as-is (see module docstring).
        positions_noise = list(range(start, start + block_size))
        refresh_rope_gather_buffer(mesh_device, rope_idx_buf, positions_noise)
        refresh_verify_positions(mesh_device, verify_buffers, start)
        write_context_valid_len(context_len)

    _refresh_inputs()

    _lg.info("[dflash-trace] compile run")
    d0, p0, c0, cos0, sin0 = _fused_body()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(d0)
    ttnn.deallocate(p0)
    ttnn.deallocate(c0)
    ttnn.deallocate(cos0)
    ttnn.deallocate(sin0)

    _lg.info("[dflash-trace] begin_trace_capture")
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    draft_ids_out, posterior_out, next_context_out, cos_out, sin_out = _fused_body()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    _lg.info("[dflash-trace] capture done")

    stopped = False
    first_replay = True
    while len(output_ids) < max_new_tokens and not stopped:
        if not first_replay:
            _refresh_inputs()
        first_replay = False

        _lg.debug(f"[dflash-trace] replay start={start} context_len={context_len}")
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)

        draft_ids_torch = (
            ttnn.to_torch(ttnn.get_device_tensors(draft_ids_out)[0]) if is_mesh else ttnn.to_torch(draft_ids_out)
        )
        draft_tokens = draft_ids_torch.reshape(-1).tolist()[1:]

        posterior_torch = (
            ttnn.to_torch(ttnn.get_device_tensors(posterior_out)[0]) if is_mesh else ttnn.to_torch(posterior_out)
        )
        posterior = posterior_torch.reshape(1, -1)[:, :block_size].long()

        candidate_ids = [output_ids[-1]] + draft_tokens
        accept, bonus, committed = greedy_accept_from_posterior(candidate_ids, posterior)
        produced = min(accept + 1, block_size)

        new_tokens = committed[1:]
        remaining_budget = max_new_tokens - len(output_ids)
        new_tokens = new_tokens[:remaining_budget]
        for tok in new_tokens:
            output_ids.append(tok)
            if tok in stop_tokens:
                stopped = True
                break
        acceptance_lengths.append(accept)

        # Project this replay's newly-tapped rows through every layer's own
        # k_proj/v_proj/k_norm + RoPE (reusing THIS replay's own cos_out/sin_out --
        # trace-bound persistent output tensors, safe to read now: the blocking
        # to_torch() reads above already guarantee this replay's execute_trace has
        # completed) and append into each layer's cache -- BEFORE advancing
        # context_len -- context_len here is still the offset this tap belongs at
        # (mirrors the main loop's identical pattern).
        dflash_drafter_update_kv_caches(
            next_context_out,
            produced,
            weights,
            cos_out,
            sin_out,
            kv_caches,
            offset=context_len,
            num_local_heads=num_local_heads,
            num_local_kv_heads=num_local_kv_heads,
            head_dim=head_dim,
            eps=config.rms_norm_eps,
        )
        context_len += produced
        start += produced

        if stopped or len(output_ids) >= max_new_tokens:
            break

        write_anchor(bonus)

    return output_ids, acceptance_lengths
