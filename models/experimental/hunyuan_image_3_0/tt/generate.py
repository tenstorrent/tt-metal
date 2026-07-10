# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TTNN text generation for the HunyuanImage-3.0 Instruct path (recaption / think /
# img_ratio sub-stages of `generate_image`). The base port is diffusion-only; this
# adds the missing token-sampling loop.
#
# The sampling math (stage transitions, temperature / top-k / top-p / repetition
# penalty, the decode loop) is provider-standard and inherently host-side, so it lives
# in the `ref/generate.py` golden (PCC/bit-exact-gated against upstream + HF). This
# module RE-EXPORTS that golden unchanged — so the device path and the reference share
# one source of truth — and contributes device adapters that wire the resident TTNN
# backbone + LM head as the loop's `forward_logits_fn`.

from models.experimental.hunyuan_image_3_0.ref.generate import (  # noqa: F401
    ConditionalSliceVocabLogitsProcessor,
    SamplingConfig,
    StageTransitionLogitsProcessor,
    apply_repetition_penalty,
    generate_text,
    sample_next_token,
    top_k_top_p_filter,
)

_apply_repetition_penalty = apply_repetition_penalty
_top_k_top_p_filter = top_k_top_p_filter

import ttnn

__all__ = [
    "ConditionalSliceVocabLogitsProcessor",
    "SamplingConfig",
    "StageTransitionLogitsProcessor",
    "apply_repetition_penalty",
    "top_k_top_p_filter",
    "sample_next_token",
    "generate_text",
    "make_backbone_logits_fn",
    "make_recaption_logits_fn",
    "ArDualCQCoordinator",
    "recaption_2cq_enabled",
    "recaption_trace_enabled",
    "RecaptionDecodeTracer",
]

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (  # noqa: E402
    ArDualCQCoordinator,
    recaption_2cq_enabled,
)
from models.experimental.hunyuan_image_3_0.tt.ar_trace import (  # noqa: E402
    RecaptionDecodeTracer,
    recaption_trace_enabled,
)


def _logits_to_torch(logits_tt, device, batch_size: int):
    """Device logits [B, 1, V] -> host [B, V] (gather mesh replicas when needed)."""
    import ttnn

    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        logits = logits[:batch_size]
    else:
        logits = ttnn.to_torch(logits_tt)
    return logits.float().squeeze(1)


def _finish_logits_read(logits_tt, device, batch_size: int, dual_cq: ArDualCQCoordinator | None):
    """D2H logits on CQ1 (2CQ) or blocking read (1CQ). Caller deallocates ``logits_tt`` after."""
    if dual_cq is not None:
        dual_cq.launch_logits_d2h(logits_tt)
        logits = dual_cq.consume_logits(batch_size)
    else:
        logits = _logits_to_torch(logits_tt, device, batch_size)
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
):
    """Adapter: wrap the resident backbone + LM head as a `forward_logits_fn`.

    Text-only path (T2I recaption): pass ``input_ids`` via the loop; optional
    ``attention_mask_fn`` and ``image_infos`` for 2D RoPE on cond spans.

    For I2I recaption with scattered cond ViT/VAE tokens, use
    ``make_recaption_logits_fn`` instead.

    When ``dual_cq`` is set, forward runs on CQ0 and logits D2H uses CQ1.
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
        if dual_cq is not None:
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
        logits = _finish_logits_read(logits_tt, device, B, dual_cq)
        ttnn.deallocate(hidden)
        ttnn.deallocate(ids_tt)
        if mask is not None:
            ttnn.deallocate(mask)
        return logits

    return forward_logits_fn


def make_recaption_logits_fn(
    model,
    lm_head,
    device,
    *,
    wte_tt=None,
    wte_weight=None,
    prefix_embeds,
    image_infos,
    attn_slices,
    replicate_to_mesh=None,
    use_kv_cache: bool = False,
    max_new_tokens: int = 512,
    dual_cq: ArDualCQCoordinator | None = None,
    sp_factor: int = 1,
    use_trace: bool | None = None,
):
    """I2I recaption adapter: fixed cond ``prefix_embeds`` + on-device wte for new AR tokens.

    When ``use_kv_cache=True``, runs one prefix prefill then single-token decode
    steps with per-layer K/V cache (requires ``sp_factor=1`` on the backbone).

    When ``HY_RECAPTION_TRACE=1`` (default; set ``HY_RECAPTION_TRACE=0`` to disable) and
    KV + ``sp_factor=1``, captures one decode step on CQ0 and replays it for subsequent
    AR tokens; prefix prefill uses chunked KV and optional trace capture
    (``tt/ar_prefill.py``).
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

    if wte_tt is None and wte_weight is None:
        raise ValueError("make_recaption_logits_fn requires wte_tt or wte_weight")
    if wte_tt is not None and isinstance(wte_tt, torch.Tensor):
        wte_weight = wte_tt
        wte_tt = None

    if use_trace is None:
        use_trace = recaption_trace_enabled(device, sp_factor=sp_factor, use_kv_cache=use_kv_cache)
    if use_trace and wte_tt is None:
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
            raise ValueError("HY_RECAPTION_TRACE=1 requires backbone embed_weight or wte_weight")

    prefix_len = int(prefix_embeds.shape[1])
    wte_host = wte_weight.detach().float() if wte_weight is not None else None
    kv_cache = HunyuanTtKvCache(len(model.layers)) if use_kv_cache else None
    max_cache_len = prefix_len + max_new_tokens
    cos_sin_holder: list = [None]

    tracer: RecaptionDecodeTracer | None = None
    if use_trace:
        tracer = RecaptionDecodeTracer(
            device,
            model,
            lm_head,
            wte_tt=wte_tt,
            prefix_embeds=prefix_embeds,
            image_infos=image_infos,
            attn_slices=attn_slices,
            kv_cache=kv_cache,
            max_cache_len=max_cache_len,
            prefix_len=prefix_len,
            replicate_to_mesh=replicate_to_mesh,
            dual_cq=dual_cq,
        )
        print(f"[recaption] trace decode path active prefix_len={prefix_len}", flush=True)

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
        if dual_cq is not None:
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
                hidden_tt = _upload_hidden(prefix_embeds.float())
            elif wte_tt is not None:
                prefix_tt = _upload_hidden(prefix_embeds.float())
                suffix_tt = wte_tt.embed(ids[:, prefix_len:].long())
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
            ttnn.deallocate(mask_tt)
            ttnn.deallocate(hidden_tt)
            if kv_cache is not None:
                kv_cache.seq_len = S

        logits_tt = lm_head(hidden, last_token_only=True)
        logits = _finish_logits_read(logits_tt, device, B, dual_cq)
        ttnn.deallocate(hidden)
        return logits

    forward_logits_fn.kv_cache = kv_cache
    forward_logits_fn.tracer = tracer
    return forward_logits_fn
