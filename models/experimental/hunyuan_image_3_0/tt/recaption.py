# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device recaption / think AR for HunyuanImage-3.0 Instruct (I2I and text-only).

from __future__ import annotations

import os
import time

import torch

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.model_config import IMAGE_BASE_SIZE
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    RecaptionResult,
    build_recaption_stage_params,
    build_ratio_logits_processor,
    decode_cot_text,
    default_recaption_sampling_config,
    resolve_image_size_from_ratio,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import GenImageHostInputs
from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import ArDualCQCoordinator, recaption_2cq_enabled
from models.experimental.hunyuan_image_3_0.tt.ar_trace import recaption_trace_enabled
from models.experimental.hunyuan_image_3_0.tt.device_sampling import (
    can_use_device_sampling,
    device_sampling_enabled,
)
from models.experimental.hunyuan_image_3_0.tt.generate import (
    generate_text,
    make_backbone_logits_fn,
    make_recaption_logits_fn,
)


def run_recaption_on_device(
    model,
    lm_head,
    device,
    bundle: GenImageHostInputs,
    tok: HunyuanTokenizer,
    bot_task: str,
    processor,
    wte_tt=None,
    *,
    wte_weight: torch.Tensor | None = None,
    image_size: str | int = IMAGE_BASE_SIZE,
    config: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
    drop_think: bool = False,
    replicate_to_mesh=None,
) -> RecaptionResult:
    """Run AR recaption/think on TT using a resident backbone + LM head.

    ``model`` must be built with ``apply_final_norm=True`` and ``norm_state_dict`` so
    the LM head receives ln_f-normalized hidden states. For I2I, pass a bundle from
    ``prepare_recaption_ar_bundle`` (with ``inputs_embeds`` and ``full_attn_slices``).

    Device-logits sampling is **on by default** (``HY_DEVICE_SAMPLING`` /
    ``HY_SAMPLE_DEVICE``; set ``=0`` to disable). Falls back to the host sampler
    when ratio / rep-penalty processors are active. Simple stage force
    (``think_recaption`` ``</think>``→``<recaption>``) stays on the device path.

    Default sample step: D2H logits → host torch topk (Instruct ``top_k``, e.g. 1024)
    → host shortlist multinomial. Set ``HY_TOP_K=32`` / ``HY_TOPK=32`` (or
    ``HY_TTNN_SAMPLING_OP=1``) for on-device ``ttnn.topk`` + ``ttnn.sampling``.
    """
    if wte_tt is None and wte_weight is None:
        raise ValueError("run_recaption_on_device requires wte_tt or wte_weight")
    if wte_tt is not None and isinstance(wte_tt, torch.Tensor):
        wte_weight = wte_tt
        wte_tt = None
    params = build_recaption_stage_params(
        tok,
        bot_task,
        image_size=image_size,
        sequence_template=tok.sequence_template,
    )
    config = config or default_recaption_sampling_config()
    input_length = int(bundle.input_ids.shape[1])
    print(
        f"[recaption] starting AR decode input_len={input_length} "
        f"max_new_tokens={config.max_new_tokens} bot_task={bot_task}",
        flush=True,
    )
    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]

    logits_processors = None
    if params.need_ratio:
        logits_processors = [build_ratio_logits_processor(tok, force_greedy=True)]

    use_device_sampling = can_use_device_sampling(
        config,
        stage_transitions=params.stage_transitions or None,
        logits_processors=logits_processors,
    )
    if device_sampling_enabled() and not use_device_sampling:
        print(
            "[recaption] device sampling requested but host processors active "
            "(ratio / rep-penalty / greedy) — using host sampler",
            flush=True,
        )
    elif use_device_sampling:
        from models.experimental.hunyuan_image_3_0.tt.device_sampling import ttnn_sampling_op_enabled

        if ttnn_sampling_op_enabled():
            print(
                "[recaption] on-device sampling (ttnn.topk + ttnn.sampling"
                + ("; stage-force on device" if params.stage_transitions else "")
                + ")",
                flush=True,
            )
        else:
            print(
                "[recaption] on-device sampling (D2H + host torch topk/shortlist; "
                "set HY_TOP_K=32 / HY_TOPK=32 for ttnn.topk+sampling"
                + ("; stage-force on device" if params.stage_transitions else "")
                + ")",
                flush=True,
            )

    has_prefix_embeds = bundle.inputs_embeds is not None
    has_cond_images = bundle.batch_cond_images is not None and len(bundle.batch_cond_images[0]) > 0
    use_kv_cache = os.environ.get("HY_RECAPTION_KV", "1") != "0"
    sp_factor = int(getattr(model, "sp_factor", 1))
    # Text-only: embed prefix ids on device (no host F.embedding H2D). I2I keeps host/TT mixed embeds.
    use_device_prefix = (not has_cond_images) and (bundle.input_ids is not None)
    use_prefix_kv = use_kv_cache and (has_prefix_embeds or use_device_prefix)
    # Keep chunked KV prefill available under device sampling (via device-logits tracer).
    use_trace = (
        False
        if use_device_sampling
        else recaption_trace_enabled(device, sp_factor=sp_factor, use_kv_cache=use_kv_cache)
    )
    use_2cq = False if use_device_sampling else recaption_2cq_enabled(device)
    dual_cq = ArDualCQCoordinator(device) if use_2cq else None
    if dual_cq is not None:
        # Tell the async logits reader whether the lm_head sharded V across the mesh.
        dual_cq.vocab_parallel = getattr(lm_head, "vocab_parallel", False)
    if use_trace:
        print(
            f"[recaption] trace AR decode on CQ0 (kv_cache={use_kv_cache}, 2cq={use_2cq}, "
            f"prefix_kv={use_prefix_kv}, cond={has_cond_images})",
            flush=True,
        )
    elif use_2cq:
        print(
            f"[recaption] 2CQ AR path active (kv_cache={use_kv_cache})",
            flush=True,
        )

    # I2I uses mixed ``inputs_embeds``; text-only prefers on-device prefix embed from ids.
    if use_prefix_kv:
        prefix_kw = {}
        if use_device_prefix:
            prefix_kw["prefix_input_ids"] = bundle.input_ids[:1].contiguous()
            print("[recaption] text-only prefix: on-device wte embed (skip host F.embedding)", flush=True)
        else:
            prefix_kw["prefix_embeds"] = bundle.inputs_embeds[:1]
        forward_logits_fn = make_recaption_logits_fn(
            model,
            lm_head,
            device,
            wte_tt=wte_tt,
            wte_weight=wte_weight,
            image_infos=image_infos,
            attn_slices=attn_slices,
            replicate_to_mesh=replicate_to_mesh,
            use_kv_cache=use_kv_cache,
            max_new_tokens=config.max_new_tokens,
            dual_cq=dual_cq,
            sp_factor=sp_factor,
            use_trace=use_trace,
            return_device_logits=use_device_sampling,
            **prefix_kw,
        )
    else:

        def attention_mask_fn(S: int):
            import ttnn

            mask_bool = build_attention_mask(S, attn_slices, bsz=1)
            mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, S, S)
            if replicate_to_mesh is not None:
                return replicate_to_mesh(mask_add)
            return ttnn.from_torch(
                mask_add,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        forward_logits_fn = make_backbone_logits_fn(
            model,
            lm_head,
            device,
            attention_mask_fn=attention_mask_fn if bundle.full_attn_slices else None,
            image_infos=image_infos,
            dual_cq=dual_cq,
            replicate_to_mesh=replicate_to_mesh,
            return_device_logits=use_device_sampling,
        )

    tracer = getattr(forward_logits_fn, "tracer", None)

    # TTFT/TPS: track wall-clock per AR forward step regardless of verbosity.
    _step_times: list[float] = []
    _ar_t0 = time.time()
    _timed = forward_logits_fn

    def forward_logits_fn(ids):
        t_fwd = time.time()
        logits = _timed(ids)
        _step_times.append(time.time() - t_fwd)
        return logits

    if os.environ.get("HY_RECAPTION_VERBOSE", "1") != "0":
        _step = [0]
        _t0 = _ar_t0
        _orig = forward_logits_fn

        def forward_logits_fn(ids):
            _step[0] += 1
            print(
                f"[recaption] AR token {_step[0]}/{config.max_new_tokens} "
                f"seq_len={ids.shape[1]} starting forward ...",
                flush=True,
            )
            t_fwd = time.time()
            logits = _orig(ids)
            elapsed = time.time() - t_fwd
            print(
                f"[recaption] AR token {_step[0]}/{config.max_new_tokens} "
                f"forward done ({elapsed:.1f}s, total {time.time() - _t0:.0f}s)",
                flush=True,
            )
            return logits

        # Preserve metadata used by generate_text device path.
        for attr in (
            "return_device_logits",
            "decode_device_token",
            "prefix_len",
            "vocab_size",
            "vocab_parallel",
            "device",
            "kv_cache",
            "tracer",
        ):
            if hasattr(_orig, attr):
                setattr(forward_logits_fn, attr, getattr(_orig, attr))

    out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=config,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
        logits_processors=logits_processors,
        generator=generator,
    )

    if dual_cq is not None:
        print(f"[recaption] 2CQ completed {dual_cq.steps} logits D2H transfers on CQ1", flush=True)

    if tracer is not None:
        print(f"[recaption] trace replay steps={tracer.replay_steps}", flush=True)
        tracer.release()

    ttft = _step_times[0] if _step_times else None
    decode_steps = _step_times[1:]
    tps = (len(decode_steps) / sum(decode_steps)) if decode_steps and sum(decode_steps) > 0 else None
    num_tokens = len(_step_times)
    total_seconds = time.time() - _ar_t0
    if ttft is not None:
        tps_str = f"{tps:.2f} tok/s" if tps is not None else "n/a (only 1 token generated)"
        print(
            f"[recaption] AR decode: tokens={num_tokens} ttft={ttft:.2f}s tps={tps_str} "
            f"end_of_cot={total_seconds:.2f}s",
            flush=True,
        )

    cot_text = decode_cot_text(tok, out["sequences"], input_length, bot_task, drop_think=drop_think)
    resolved_size: str | int = image_size
    if params.need_ratio:
        resolved_size = resolve_image_size_from_ratio(tok, processor, int(out["sequences"][0, -1].item()))

    return RecaptionResult(
        cot_text=cot_text,
        sequences=out["sequences"],
        new_tokens=out["new_tokens"],
        input_length=input_length,
        image_size=resolved_size,
        stage_params=params,
        ttft=ttft,
        tps=tps,
        total_seconds=total_seconds,
    )
