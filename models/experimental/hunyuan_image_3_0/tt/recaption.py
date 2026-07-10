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
from models.experimental.hunyuan_image_3_0.tt.generate import (
    generate_text,
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
    image_size: str | int = 1024,
    config: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
    drop_think: bool = False,
    replicate_to_mesh=None,
) -> RecaptionResult:
    """Run AR recaption/think on TT using a resident backbone + LM head.

    ``model`` must be built with ``apply_final_norm=True`` and ``norm_state_dict`` so
    the LM head receives ln_f-normalized hidden states. For I2I, pass a bundle from
    ``prepare_recaption_ar_bundle`` (with ``inputs_embeds`` and ``full_attn_slices``).
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

    has_prefix_embeds = bundle.inputs_embeds is not None
    has_cond_images = bundle.batch_cond_images is not None and len(bundle.batch_cond_images[0]) > 0
    use_kv_cache = os.environ.get("HY_RECAPTION_KV", "1") != "0"
    sp_factor = int(getattr(model, "sp_factor", 1))
    use_prefix_kv = has_prefix_embeds and use_kv_cache
    use_trace = recaption_trace_enabled(device, sp_factor=sp_factor, use_kv_cache=use_kv_cache)
    use_2cq = recaption_2cq_enabled(device)
    dual_cq = ArDualCQCoordinator(device) if use_2cq else None
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

    # I2I and text-only T2I both use fixed prefix_embeds + KV/trace decode when
    # ``inputs_embeds`` is set (``prepare_recaption_ar_bundle``). Full-seq forward
    # per token is only used when KV is disabled or prefix embeds are unavailable.
    if use_prefix_kv:
        forward_logits_fn = make_recaption_logits_fn(
            model,
            lm_head,
            device,
            wte_tt=wte_tt,
            wte_weight=wte_weight,
            prefix_embeds=bundle.inputs_embeds[:1],
            image_infos=image_infos,
            attn_slices=attn_slices,
            replicate_to_mesh=replicate_to_mesh,
            use_kv_cache=use_kv_cache,
            max_new_tokens=config.max_new_tokens,
            dual_cq=dual_cq,
            sp_factor=sp_factor,
            use_trace=use_trace,
        )
    else:
        from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn

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
        )

    logits_processors = None
    if params.need_ratio:
        logits_processors = [build_ratio_logits_processor(tok, force_greedy=True)]

    tracer = getattr(forward_logits_fn, "tracer", None)

    if os.environ.get("HY_RECAPTION_VERBOSE", "1") != "0":
        _step = [0]
        _t0 = time.time()
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
    )
