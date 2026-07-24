# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TTNN text generation for the HunyuanImage-3.0 Instruct path (recaption / think /
# img_ratio sub-stages of `generate_image`). The base port is diffusion-only; this
# adds the missing token-sampling loop.
#
# Host golden (stage transitions, repetition penalty, decode orchestration) lives in
# `ref/generate.py`. On-device topk + sampling (``ttnn.topk`` / ``ttnn.sampling``) is
# opt-in via ``HY_DEVICE_SAMPLING=1`` when those host-only processors are inactive —
# see ``tt/device_sampling.py``.

from models.experimental.hunyuan_image_3_0.ref.generate import (  # noqa: F401
    ConditionalSliceVocabLogitsProcessor,
    SamplingConfig,
    StageTransitionLogitsProcessor,
    apply_repetition_penalty,
    generate_text as generate_text_host,
    sample_next_token,
    top_k_top_p_filter,
)

import ttnn

from models.experimental.hunyuan_image_3_0.tt.device_sampling import (  # noqa: E402
    StageForceController,
    append_token_ids_tt,
    can_use_device_sampling,
    device_sampling_enabled,
    materialize_generated_ids,
    sample_logits_ttnn,
    token_hits_stop_tt,
    upload_stop_ids_tt,
    upload_token_ids_tt,
)

__all__ = [
    "ConditionalSliceVocabLogitsProcessor",
    "SamplingConfig",
    "StageTransitionLogitsProcessor",
    "apply_repetition_penalty",
    "top_k_top_p_filter",
    "sample_next_token",
    "generate_text",
    "generate_text_host",
    "make_backbone_logits_fn",
    "make_recaption_logits_fn",
    "ArDualCQCoordinator",
    "recaption_2cq_enabled",
    "recaption_trace_enabled",
    "RecaptionDecodeTracer",
    "can_use_device_sampling",
    "device_sampling_enabled",
]

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (  # noqa: E402
    ArDualCQCoordinator,
    recaption_2cq_enabled,
)
from models.experimental.hunyuan_image_3_0.tt.ar_trace import (  # noqa: E402
    RecaptionDecodeTracer,
    recaption_trace_enabled,
)


def _logits_to_torch(logits_tt, device, batch_size: int, *, vocab_parallel: bool = False):
    """Device logits [B, 1, V] -> host [B, V].

    When ``vocab_parallel`` the vocab is sharded across the mesh: concat the per-device
    slices on the last dim. Otherwise the logits are replicated: gather the batch-dim
    replicas and keep one copy.
    """
    import ttnn

    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        if vocab_parallel:
            logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
        else:
            logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            logits = logits[:batch_size]
    else:
        logits = ttnn.to_torch(logits_tt)
    return logits.float().squeeze(1)


def _finish_logits_read(
    logits_tt, device, batch_size: int, dual_cq: ArDualCQCoordinator | None, *, vocab_parallel: bool = False
):
    """D2H logits on CQ1 (2CQ) or blocking read (1CQ). Caller deallocates ``logits_tt`` after."""
    if dual_cq is not None:
        dual_cq.launch_logits_d2h(logits_tt)
        logits = dual_cq.consume_logits(batch_size)
    else:
        logits = _logits_to_torch(logits_tt, device, batch_size, vocab_parallel=vocab_parallel)
    ttnn.deallocate(logits_tt)
    return logits


def make_backbone_logits_fn(
    model,
    lm_head,
    device,
    *,
    attention_mask_fn=None,
    image_infos=None,
    dual_cq: ArDualCQCoordinator | None = None,
    replicate_to_mesh=None,
    return_device_logits: bool = False,
):
    """Adapter: wrap the resident backbone + LM head as a `forward_logits_fn`.

    Text-only path (T2I recaption): pass ``input_ids`` via the loop; optional
    ``attention_mask_fn`` and ``image_infos`` for 2D RoPE on cond spans.

    For I2I recaption with scattered cond ViT/VAE tokens, use
    ``make_recaption_logits_fn`` instead.

    When ``dual_cq`` is set, forward runs on CQ0 and logits D2H uses CQ1.
    When ``return_device_logits=True``, skip D2H and return the ttnn logits tensor
    (for ``HY_DEVICE_SAMPLING``); dual_cq async read is not used in that mode.
    """
    import ttnn

    def _upload_ids(ids):
        kwargs = dict(
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if replicate_to_mesh is None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        elif replicate_to_mesh is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(ids, **kwargs)

    def forward_logits_fn(ids):
        if dual_cq is not None and not return_device_logits:
            dual_cq.fence_compute_before_forward()
        S = ids.shape[1]
        B = ids.shape[0]
        ids_tt = _upload_ids(ids)
        mask = attention_mask_fn(S) if attention_mask_fn is not None else None
        hidden = model.forward(
            input_ids=ids_tt,
            seq_len=S,
            image_infos=image_infos,
            attention_mask=mask,
        )
        logits_tt = lm_head(hidden, last_token_only=True)
        ttnn.deallocate(hidden)
        ttnn.deallocate(ids_tt)
        if mask is not None:
            ttnn.deallocate(mask)
        if return_device_logits:
            return logits_tt
        logits = _finish_logits_read(
            logits_tt, device, B, dual_cq, vocab_parallel=getattr(lm_head, "vocab_parallel", False)
        )
        return logits

    forward_logits_fn.return_device_logits = return_device_logits
    forward_logits_fn.vocab_size = getattr(lm_head, "vocab_size", None)
    forward_logits_fn.vocab_parallel = getattr(lm_head, "vocab_parallel", False)
    forward_logits_fn.device = device
    return forward_logits_fn


def make_recaption_logits_fn(
    model,
    lm_head,
    device,
    *,
    wte_tt=None,
    wte_weight=None,
    prefix_embeds=None,
    prefix_input_ids=None,
    image_infos,
    attn_slices,
    replicate_to_mesh=None,
    use_kv_cache: bool = False,
    max_new_tokens: int = 512,
    dual_cq: ArDualCQCoordinator | None = None,
    sp_factor: int = 1,
    use_trace: bool | None = None,
    return_device_logits: bool = False,
):
    """Recaption adapter: prefix (host embeds or on-device id embed) + wte for new tokens.

    Text-only: pass ``prefix_input_ids`` so prefix embedding stays on device
    (``wte_tt.embed``) with no host ``F.embedding`` + full-prefix H2D.
    I2I / cond: pass ``prefix_embeds`` (already mixed with VAE/ViT tokens).

    When ``use_kv_cache=True``, runs one prefix prefill then single-token decode
    steps with per-layer K/V cache (requires ``sp_factor=1`` on the backbone).

    When ``HY_RECAPTION_TRACE=1`` (default; set ``HY_RECAPTION_TRACE=0`` to disable) and
    KV + ``sp_factor=1``, captures one decode step on CQ0 and replays it for subsequent
    AR tokens; prefix prefill uses chunked KV and optional trace capture
    (``tt/ar_prefill.py``).

    ``return_device_logits=True`` keeps logits on device for ``HY_DEVICE_SAMPLING``
    (incompatible with trace replay returning host logits — disable trace or fall back).
    """
    import torch
    import torch.nn.functional as F

    import ttnn

    from models.experimental.hunyuan_image_3_0.ref.attention.mask import (
        build_attention_mask,
        build_attention_mask_query_row,
        to_additive,
    )
    from models.experimental.hunyuan_image_3_0.tt.kv_cache import HunyuanTtKvCache

    def _mask_has_image_spans(slices) -> bool:
        """True if any batch item has a bidirectional (image) span."""
        if not slices:
            return False
        if isinstance(slices[0], list):  # per-batch list-of-lists
            return any(len(spans) > 0 for spans in slices)
        return len(slices) > 0  # flat list of spans

    if prefix_embeds is None and prefix_input_ids is None:
        raise ValueError("make_recaption_logits_fn requires prefix_embeds or prefix_input_ids")
    if wte_tt is None and wte_weight is None:
        raise ValueError("make_recaption_logits_fn requires wte_tt or wte_weight")
    if wte_tt is not None and isinstance(wte_tt, torch.Tensor):
        wte_weight = wte_tt
        wte_tt = None

    if use_trace is None:
        use_trace = recaption_trace_enabled(device, sp_factor=sp_factor, use_kv_cache=use_kv_cache)
    # Device sampling needs chunked KV prefill (one-shot S≈793 L1-clashes with lm_head).
    # Use the tracer for chunked prefill + eager decode returning device logits; skip
    # decode-trace capture (incompatible with keeping logits on device for sampling).
    use_device_logits_tracer = return_device_logits and use_kv_cache
    if use_device_logits_tracer:
        use_trace = False  # don't double-create; we build the device-logits tracer below
        print(
            "[recaption] HY_DEVICE_SAMPLING: chunked KV prefill + eager decode " "(device logits; decode trace off)",
            flush=True,
        )
    elif return_device_logits and use_trace:
        use_trace = False
        print("[recaption] HY_DEVICE_SAMPLING: disabling decode trace (device logits)", flush=True)

    # On-device prefix embed and/or decode always need a device wte.
    need_wte = use_trace or use_device_logits_tracer or prefix_input_ids is not None
    if need_wte and wte_tt is None:
        if getattr(model, "embed_weight", None) is not None:
            from models.experimental.hunyuan_image_3_0.tt.wte import BackboneWteAdapter

            wte_tt = BackboneWteAdapter(model)
            print("[recaption] trace: reusing backbone embed_weight (no duplicate wte upload)", flush=True)
        elif wte_weight is not None:
            from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte

            mesh_mapper = None
            if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                mesh_mapper = ttnn.ReplicateTensorToMesh(device)
            wte_tt = HunyuanTtWte(device, wte_weight, mesh_mapper=mesh_mapper)
            print("[recaption] trace: created on-device wte from wte_weight", flush=True)
        else:
            raise ValueError(
                "device-sampling / prefix_input_ids / HY_RECAPTION_TRACE requires backbone embed_weight or wte_weight"
            )

    if prefix_input_ids is not None:
        ids = prefix_input_ids if prefix_input_ids.ndim == 2 else prefix_input_ids.unsqueeze(0)
        prefix_len = int(ids.shape[1])
    else:
        prefix_len = int(prefix_embeds.shape[1])
    wte_host = wte_weight.detach().float() if wte_weight is not None else None
    kv_cache = HunyuanTtKvCache(len(model.layers)) if use_kv_cache else None
    max_cache_len = prefix_len + max_new_tokens
    cos_sin_holder: list = [None]

    tracer: RecaptionDecodeTracer | None = None
    if use_trace or use_device_logits_tracer or (use_kv_cache and prefix_input_ids is not None):
        # Always use tracer when prefix is id-based so prefill can TT-embed.
        if not (use_trace or use_device_logits_tracer):
            use_device_logits_tracer = False  # keep host logits; still use tracer for prefill
        tracer = RecaptionDecodeTracer(
            device,
            model,
            lm_head,
            wte_tt=wte_tt,
            prefix_embeds=prefix_embeds,
            prefix_input_ids=prefix_input_ids,
            image_infos=image_infos,
            attn_slices=attn_slices,
            kv_cache=kv_cache,
            max_cache_len=max_cache_len,
            prefix_len=prefix_len,
            replicate_to_mesh=replicate_to_mesh,
            dual_cq=None if return_device_logits else dual_cq,
            return_device_logits=return_device_logits,
            enable_decode_trace=use_trace and not return_device_logits,
        )
        if use_trace:
            print(f"[recaption] trace decode path active prefix_len={prefix_len}", flush=True)
        elif return_device_logits:
            print(f"[recaption] device-sampling tracer active prefix_len={prefix_len}", flush=True)
        else:
            print(f"[recaption] on-device prefix-embed tracer active prefix_len={prefix_len}", flush=True)

    def _upload_hidden(hidden_host):
        if replicate_to_mesh is not None:
            return replicate_to_mesh(hidden_host)
        return ttnn.from_torch(
            hidden_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_mask_full(S: int, B: int):
        if not _mask_has_image_spans(attn_slices):
            # Pure-causal prefill (text-only recaption, incl. the max-ISL context):
            # supply NO mask so SDPA uses its built-in causal path — no S×S host build
            # + PCIe upload (~1 GB at S=22784, hundreds of ms of device stall) and the
            # causal SDPA itself is ~3x faster at large ISL than a materialized mask.
            return None
        # Image spans present (i2i): build the host mask — SDPA's causal path can't
        # express the bidirectional image block, and the device span builder is not yet
        # bitwise-equivalent (test_mask_image_span_bitwise, pre-existing). i2i prefills
        # run at much smaller S where the upload cost is negligible.
        mask_bool = build_attention_mask(S, attn_slices, bsz=B)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(B, 1, S, S)
        if replicate_to_mesh is not None:
            return replicate_to_mesh(mask_add)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _upload_mask_row(query_pos: int, total_len: int, B: int):
        mask_bool = build_attention_mask_query_row(total_len, query_pos, attn_slices, bsz=B)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(B, 1, 1, total_len)
        if replicate_to_mesh is not None:
            return replicate_to_mesh(mask_add)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _embed_ids(ids_slice: torch.Tensor) -> ttnn.Tensor:
        if wte_tt is not None:
            return wte_tt.embed(ids_slice.long())
        new_emb = F.embedding(ids_slice.long(), wte_host)
        return _upload_hidden(new_emb)

    def forward_logits_fn(ids):
        if tracer is not None:
            return tracer.forward(ids)

        B, S = ids.shape
        if S < prefix_len:
            raise ValueError(f"sequence length {S} < prefix length {prefix_len}")
        if dual_cq is not None and not return_device_logits:
            dual_cq.fence_compute_before_forward()

        decode_step = use_kv_cache and S > prefix_len
        if decode_step:
            hidden_tt = _embed_ids(ids[:, -1:])
            query_pos = S - 1
            if cos_sin_holder[0] is None:
                raise RuntimeError("KV cache prefill must run before decode steps")
            cos_full, sin_full = cos_sin_holder[0]
            cos_tt, sin_tt = model.layers[0].self_attn.rope.slice_cos_sin(cos_full, sin_full, query_pos)
            mask_tt = _upload_mask_row(query_pos, S, B)
            hidden = model.forward(
                inputs_embeds=hidden_tt,
                seq_len=S,
                image_infos=image_infos,
                attention_mask=mask_tt,
                kv_cache=kv_cache,
                use_cache=True,
                decode_step=True,
                cos_sin=(cos_tt, sin_tt),
            )
            ttnn.deallocate(mask_tt)
            ttnn.deallocate(hidden_tt)
        else:
            if S == prefix_len:
                if prefix_input_ids is not None:
                    hidden_tt = _embed_ids(ids[:, :prefix_len])
                else:
                    hidden_tt = _upload_hidden(prefix_embeds.float())
            elif prefix_input_ids is not None or wte_tt is not None:
                if prefix_input_ids is not None:
                    prefix_tt = _embed_ids(ids[:, :prefix_len])
                else:
                    prefix_tt = _upload_hidden(prefix_embeds.float())
                suffix_tt = _embed_ids(ids[:, prefix_len:])
                hidden_tt = ttnn.concat([prefix_tt, suffix_tt], dim=1)
                ttnn.deallocate(prefix_tt)
                ttnn.deallocate(suffix_tt)
            else:
                hidden_host = torch.cat(
                    [prefix_embeds.float(), F.embedding(ids[:, prefix_len:].long(), wte_host)],
                    dim=1,
                )
                hidden_tt = _upload_hidden(hidden_host)
            if use_kv_cache:
                cos_full, sin_full = model.layers[0].self_attn.rope.prepare_cos_sin(
                    max_cache_len, image_infos=image_infos
                )
                cos_sin_holder[0] = (cos_full, sin_full)
                cos_tt = ttnn.slice(cos_full, [0, 0, 0, 0], [1, 1, S, cos_full.shape[-1]])
                sin_tt = ttnn.slice(sin_full, [0, 0, 0, 0], [1, 1, S, sin_full.shape[-1]])
                cos_sin = (cos_tt, sin_tt)
            else:
                cos_sin = None
            mask_tt = _upload_mask_full(S, B)
            hidden = model.forward(
                inputs_embeds=hidden_tt,
                seq_len=S,
                image_infos=image_infos,
                attention_mask=mask_tt,
                kv_cache=kv_cache,
                use_cache=use_kv_cache,
                decode_step=False,
                cos_sin=cos_sin,
            )
            if mask_tt is not None:  # None on the pure-causal path (is_causal SDPA)
                ttnn.deallocate(mask_tt)
            ttnn.deallocate(hidden_tt)
            if kv_cache is not None:
                kv_cache.seq_len = S

        logits_tt = lm_head(hidden, last_token_only=True)
        ttnn.deallocate(hidden)
        if return_device_logits:
            return logits_tt
        logits = _finish_logits_read(
            logits_tt, device, B, dual_cq, vocab_parallel=getattr(lm_head, "vocab_parallel", False)
        )
        return logits

    forward_logits_fn.kv_cache = kv_cache
    forward_logits_fn.tracer = tracer
    forward_logits_fn.return_device_logits = return_device_logits
    forward_logits_fn.vocab_size = getattr(lm_head, "vocab_size", None)
    forward_logits_fn.vocab_parallel = getattr(lm_head, "vocab_parallel", False)
    forward_logits_fn.device = device
    forward_logits_fn.prefix_len = prefix_len
    if tracer is not None:

        def _decode_device_token(token_tt, *, seq_len: int):
            return tracer.forward_device_token(token_tt, seq_len=seq_len)

        forward_logits_fn.decode_device_token = _decode_device_token
    return forward_logits_fn


def generate_text(
    forward_logits_fn,
    input_ids,
    *,
    config: SamplingConfig = None,
    stage_transitions=None,
    final_stop_tokens=None,
    logits_processors: list | None = None,
    generator=None,
):
    """AR decode loop — host golden by default; device-logits sample when eligible.

    Default sampling: D2H full-V logits → host topk → host shortlist multinomial
    → host token ids → ``forward_logits_fn(ids)`` each step.

    On-device top-k when ``HY_TOP_K=32`` / ``HY_TOPK=32`` (or ``HY_TTNN_SAMPLING_OP=1``):
    ``ttnn.topk`` + ``ttnn.sampling`` with on-device id concat / stop checks.
    """
    import torch
    import ttnn

    from models.experimental.hunyuan_image_3_0.tt.device_sampling import ttnn_sampling_op_enabled

    config = config or SamplingConfig()
    use_device = can_use_device_sampling(
        config, stage_transitions=stage_transitions, logits_processors=logits_processors
    )
    wants_device_logits = getattr(forward_logits_fn, "return_device_logits", False)

    if not use_device:
        if wants_device_logits:
            # Adapter was built for device logits but processors require host — wrap.
            device = getattr(forward_logits_fn, "device", None)
            vocab_parallel = getattr(forward_logits_fn, "vocab_parallel", False)

            def _host_logits_fn(ids):
                logits_tt = forward_logits_fn(ids)
                B = ids.shape[0]
                return _logits_to_torch(logits_tt, device, B, vocab_parallel=vocab_parallel)

            return generate_text_host(
                _host_logits_fn,
                input_ids,
                config=config,
                stage_transitions=stage_transitions,
                final_stop_tokens=final_stop_tokens,
                logits_processors=logits_processors,
                generator=generator,
            )
        return generate_text_host(
            forward_logits_fn,
            input_ids,
            config=config,
            stage_transitions=stage_transitions,
            final_stop_tokens=final_stop_tokens,
            logits_processors=logits_processors,
            generator=generator,
        )

    if not wants_device_logits:
        raise ValueError("device sampling requires forward_logits_fn built with return_device_logits=True")

    device = forward_logits_fn.device
    vocab_size = forward_logits_fn.vocab_size
    vocab_parallel = getattr(forward_logits_fn, "vocab_parallel", False)
    decode_device_token = getattr(forward_logits_fn, "decode_device_token", None)
    if vocab_size is None:
        raise ValueError("forward_logits_fn.vocab_size is required for device sampling")

    if stage_transitions and final_stop_tokens is None:
        raise ValueError("final_stop_tokens must be provided when stage_transitions is set")
    prefix_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
    if prefix_ids.ndim == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    prefix_ids = prefix_ids.long()
    B = prefix_ids.shape[0]
    stop_set = set(final_stop_tokens or [])
    prefix_len = int(getattr(forward_logits_fn, "prefix_len", prefix_ids.shape[1]))

    seed = None
    if generator is not None:
        seed = int(generator.initial_seed()) % 1_000_000 + 1

    use_ttnn_op = ttnn_sampling_op_enabled()
    stage_force = StageForceController(stage_transitions, B) if stage_transitions else None
    if use_ttnn_op:
        print(
            f"[generate] on-device sampling + device bookkeeping "
            f"(ttnn.topk≤32 + ttnn.sampling → concat/stop on device → embed) "
            f"temp={config.temperature} top_p={config.top_p} top_k={config.top_k} "
            f"V={vocab_size} vocab_parallel={vocab_parallel} "
            f"stage_force={bool(stage_force)}",
            flush=True,
        )
    else:
        print(
            f"[generate] device-logits sampling (D2H + host torch topk/shortlist; "
            f"set HY_TOP_K=32 for ttnn.topk+sampling) "
            f"temp={config.temperature} top_p={config.top_p} top_k={config.top_k} "
            f"V={vocab_size} vocab_parallel={vocab_parallel} "
            f"stage_force={bool(stage_force)}",
            flush=True,
        )

    def _next_ids_host(logits_tt, ids_so_far, step_i: int):
        """Sample or stage-force the next token ids on host ``[B]``; frees ``logits_tt``."""
        forced = stage_force.forced_ids(ids_so_far[:, -1]) if stage_force is not None else [None] * B
        if all(f is not None for f in forced):
            ttnn.deallocate(logits_tt)
            return torch.tensor(forced, dtype=torch.long)
        step_seed = None if seed is None else (seed + step_i) % 1_000_000 + 1
        next_ids = sample_logits_ttnn(
            logits_tt,
            device,
            vocab_size=vocab_size,
            batch_size=B,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=step_seed,
            vocab_parallel=vocab_parallel,
            deallocate_input=True,
            return_device_ids=False,
        )
        for i, f in enumerate(forced):
            if f is not None:
                next_ids[i] = int(f)
        return next_ids

    # ---- Default: host-id AR loop over device logits ----
    if not use_ttnn_op:
        ids = prefix_ids
        new_tokens: list[list[int]] = [[] for _ in range(B)]
        finished = [False] * B
        logits_tt = forward_logits_fn(ids)
        for step_i in range(config.max_new_tokens):
            next_ids = _next_ids_host(logits_tt, ids, step_i)
            ids = torch.cat([ids, next_ids.view(B, 1)], dim=1)
            for i in range(B):
                tid = int(next_ids[i].item())
                new_tokens[i].append(tid)
                if tid in stop_set:
                    finished[i] = True
            if all(finished):
                break
            logits_tt = forward_logits_fn(ids)
        return {"sequences": ids, "new_tokens": new_tokens}

    # ---- Opt-in: pure ttnn.sampling + on-device bookkeeping ----
    stop_ids_tt = upload_stop_ids_tt(device, list(stop_set), B) if stop_set else None
    generated_tt = None
    finished = [False] * B
    ids_host = prefix_ids.clone()

    logits_tt = forward_logits_fn(prefix_ids)
    token_tt = None
    for step_i in range(config.max_new_tokens):
        forced = stage_force.forced_ids(ids_host[:, -1]) if stage_force is not None else [None] * B
        if all(f is not None for f in forced):
            ttnn.deallocate(logits_tt)
            next_ids = torch.tensor(forced, dtype=torch.long)
            token_tt = upload_token_ids_tt(device, next_ids)
        else:
            step_seed = None if seed is None else (seed + step_i) % 1_000_000 + 1
            token_tt = sample_logits_ttnn(
                logits_tt,
                device,
                vocab_size=vocab_size,
                batch_size=B,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=step_seed,
                vocab_parallel=vocab_parallel,
                deallocate_input=True,
                return_device_ids=True,
            )
            if any(f is not None for f in forced):
                # Mixed batch: override forced rows on host then re-upload.
                host_ids = ttnn.to_torch(token_tt).view(B).long()
                if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                    host_ids = ttnn.to_torch(token_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:B]
                    host_ids = host_ids.view(B).long()
                for i, f in enumerate(forced):
                    if f is not None:
                        host_ids[i] = int(f)
                ttnn.deallocate(token_tt)
                token_tt = upload_token_ids_tt(device, host_ids)
                next_ids = host_ids
            else:
                next_ids = None  # filled below from device for ids_host only when needed

        generated_tt = append_token_ids_tt(generated_tt, token_tt)

        if next_ids is None:
            if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
                next_ids = ttnn.to_torch(token_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:B]
            else:
                next_ids = ttnn.to_torch(token_tt)
            next_ids = next_ids.view(B).long()
        ids_host = torch.cat([ids_host, next_ids.view(B, 1)], dim=1)

        if stop_ids_tt is not None:
            hits = token_hits_stop_tt(token_tt, stop_ids_tt, device, B)
            for i, hit in enumerate(hits):
                if hit:
                    finished[i] = True
            if all(finished):
                ttnn.deallocate(token_tt)
                token_tt = None
                break

        seq_len = prefix_len + step_i + 1
        if decode_device_token is not None:
            logits_tt = decode_device_token(token_tt, seq_len=seq_len)
        else:
            ids_so_far, _ = materialize_generated_ids(generated_tt, device, B, prefix_ids=prefix_ids, stop_set=None)
            logits_tt = forward_logits_fn(ids_so_far)
        ttnn.deallocate(token_tt)
        token_tt = None

    if token_tt is not None:
        ttnn.deallocate(token_tt)
    if stop_ids_tt is not None:
        ttnn.deallocate(stop_ids_tt)

    sequences, new_tokens = materialize_generated_ids(
        generated_tt, device, B, prefix_ids=prefix_ids, stop_set=stop_set or None
    )
    if generated_tt is not None:
        ttnn.deallocate(generated_tt)

    return {"sequences": sequences, "new_tokens": new_tokens}
