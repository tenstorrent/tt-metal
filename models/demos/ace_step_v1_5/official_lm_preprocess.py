"""
Official ACE-Step LM + handler conditioning for TTNN demos.

Mirrors ``acestep.inference.generate_music`` Phase 1 (5 Hz LM / CoT) and the kwargs
passed into ``AceStepHandler.generate_music``, then runs only:

  ``_normalize_service_generate_inputs`` → ``_prepare_batch`` → ``preprocess_batch``
  → ``model.prepare_condition``

so TTNN can own diffusion.  Does not call ``service_generate`` / PyTorch DiT.
"""

from __future__ import annotations

import inspect
import math
import sys
from typing import Any

import torch
from loguru import logger


def configure_acestep_logging(*, level: str = "DEBUG") -> None:
    """Match CLI-style loguru lines (time | level | module:function:line - message)."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"),
    )


def build_filtered_dit_kwargs_for_handler(
    dit_handler: Any,
    llm_handler: Any,
    params: Any,
    config: Any,
    progress: Any = None,
) -> dict[str, Any]:
    """
    Same LM + ``dit_generate_kwargs`` filtering as ``acestep.inference.generate_music``,
    without invoking ``dit_handler.generate_music``.
    """
    # --- Begin: adapted from acestep.inference.generate_music (Phase 1 + kwargs) ---
    if params.task_type == "text2music" and params.flow_edit_morph:
        audio_code_string_to_use = ""
    else:
        audio_code_string_to_use = params.audio_codes
    lm_generated_metadata = None
    lm_generated_audio_codes_list: list[Any] = []
    lm_total_time_costs = {"phase1_time": 0.0, "phase2_time": 0.0, "total_time": 0.0}

    bpm = params.bpm
    key_scale = params.keyscale
    time_signature = params.timesignature
    audio_duration = params.duration
    dit_input_caption = params.caption
    dit_input_vocal_language = params.vocal_language
    dit_input_lyrics = params.lyrics

    from models.demos.ace_step_v1_5.acestep_preprocess_shim import (
        _load_cached_repaint_source,
        _resample_matching_source_seeds,
        _update_metadata_from_lm,
    )

    cached_repaint_source = _load_cached_repaint_source(params.src_audio) if params.task_type == "repaint" else None
    source_repaint_latents = cached_repaint_source.latents if cached_repaint_source is not None else None

    user_provided_audio_codes = bool(params.audio_codes and str(params.audio_codes).strip())
    need_audio_codes = not user_provided_audio_codes
    actual_batch_size = config.batch_size if config.batch_size is not None else 1

    seed_for_generation = ""
    if config.seeds is not None:
        if isinstance(config.seeds, list) and len(config.seeds) > 0:
            seed_for_generation = ",".join(str(s) for s in config.seeds)
        elif isinstance(config.seeds, int):
            seed_for_generation = str(config.seeds)

    actual_seed_list, _ = dit_handler.prepare_seeds(actual_batch_size, seed_for_generation, config.use_random_seed)
    use_random_seed_for_dit = config.use_random_seed
    if cached_repaint_source is not None:
        actual_seed_list = _resample_matching_source_seeds(actual_seed_list, cached_repaint_source.source_seed)
        seed_for_generation = ",".join(str(seed) for seed in actual_seed_list)
        use_random_seed_for_dit = False

    skip_lm_tasks = {"cover", "cover-nofsq", "repaint", "extract"}
    morph_on_text2music = params.task_type == "text2music" and params.flow_edit_morph
    need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas
    skip_lm = params.task_type in skip_lm_tasks or morph_on_text2music
    use_lm = (
        (params.thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized and not skip_lm
    )

    if skip_lm:
        reason = params.task_type if params.task_type in skip_lm_tasks else f"{params.task_type}+flow_edit_morph"
        logger.info(f"Skipping LM for task_type='{reason}' - using DiT directly")

    logger.info(
        f"[generate_music] LLM usage decision: thinking={params.thinking}, "
        f"use_cot_caption={params.use_cot_caption}, use_cot_language={params.use_cot_language}, "
        f"use_cot_metas={params.use_cot_metas}, need_lm_for_cot={need_lm_for_cot}, "
        f"llm_initialized={llm_handler.llm_initialized if llm_handler else False}, use_lm={use_lm}"
    )

    if use_lm:
        top_k_value = None if not params.lm_top_k or params.lm_top_k == 0 else int(params.lm_top_k)
        top_p_value = None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p

        user_metadata = {}
        if bpm is not None:
            try:
                bpm_value = float(bpm)
                if bpm_value > 0:
                    user_metadata["bpm"] = int(bpm_value)
            except (ValueError, TypeError):
                pass
        if key_scale and key_scale.strip():
            key_scale_clean = key_scale.strip()
            if key_scale_clean.lower() not in ["n/a", ""]:
                user_metadata["keyscale"] = key_scale_clean
        if time_signature and time_signature.strip():
            time_sig_clean = time_signature.strip()
            if time_sig_clean.lower() not in ["n/a", ""]:
                user_metadata["timesignature"] = time_sig_clean
        if audio_duration is not None:
            try:
                duration_value = float(audio_duration)
                if duration_value > 0:
                    user_metadata["duration"] = int(duration_value)
            except (ValueError, TypeError):
                pass
        user_metadata_to_pass = user_metadata if user_metadata else None

        infer_type = "llm_dit" if need_audio_codes and params.thinking else "dit"
        max_inference_batch_size = (
            int(config.lm_batch_chunk_size) if config.lm_batch_chunk_size > 0 else actual_batch_size
        )
        num_chunks = math.ceil(actual_batch_size / max_inference_batch_size)

        all_metadata_list: list[Any] = []
        all_audio_codes_list: list[Any] = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * max_inference_batch_size
            chunk_end = min(chunk_start + max_inference_batch_size, actual_batch_size)
            chunk_size = chunk_end - chunk_start
            chunk_seeds = actual_seed_list[chunk_start:chunk_end] if chunk_start < len(actual_seed_list) else None

            logger.info(
                f"LM chunk {chunk_idx+1}/{num_chunks} (infer_type={infer_type}) "
                f"(size: {chunk_size}, seeds: {chunk_seeds})"
            )

            result = llm_handler.generate_with_stop_condition(
                caption=params.caption or "",
                lyrics=params.lyrics or "",
                infer_type=infer_type,
                temperature=params.lm_temperature,
                cfg_scale=params.lm_cfg_scale,
                negative_prompt=params.lm_negative_prompt,
                top_k=top_k_value,
                top_p=top_p_value,
                target_duration=audio_duration,
                user_metadata=user_metadata_to_pass,
                use_cot_caption=params.use_cot_caption,
                use_cot_language=params.use_cot_language,
                use_cot_metas=params.use_cot_metas,
                use_constrained_decoding=params.use_constrained_decoding,
                constrained_decoding_debug=config.constrained_decoding_debug,
                batch_size=chunk_size,
                seeds=chunk_seeds,
                progress=progress,
            )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown LM error")
                raise RuntimeError(f"LM generation failed: {error_msg}")

            if chunk_size > 1:
                metadata_list = result.get("metadata", [])
                audio_codes_list = result.get("audio_codes", [])
                all_metadata_list.extend(metadata_list)
                all_audio_codes_list.extend(audio_codes_list)
            else:
                metadata = result.get("metadata", {})
                audio_codes = result.get("audio_codes", "")
                all_metadata_list.append(metadata)
                all_audio_codes_list.append(audio_codes)

            lm_extra = result.get("extra_outputs", {})
            lm_chunk_time_costs = lm_extra.get("time_costs", {})
            if lm_chunk_time_costs:
                for key in ["phase1_time", "phase2_time", "total_time"]:
                    if key in lm_chunk_time_costs:
                        lm_total_time_costs[key] += lm_chunk_time_costs[key]

        lm_generated_metadata = all_metadata_list[0] if all_metadata_list else None
        lm_generated_audio_codes_list = all_audio_codes_list

        if infer_type == "llm_dit":
            if actual_batch_size > 1:
                audio_code_string_to_use = all_audio_codes_list
            else:
                audio_code_string_to_use = all_audio_codes_list[0] if all_audio_codes_list else ""
        else:
            audio_code_string_to_use = params.audio_codes

        if lm_generated_metadata:
            bpm, key_scale, time_signature, audio_duration, vocal_language, caption, lyrics = _update_metadata_from_lm(
                metadata=lm_generated_metadata,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                audio_duration=audio_duration,
                vocal_language=dit_input_vocal_language,
                caption=dit_input_caption,
                lyrics=dit_input_lyrics,
            )
            if (not params.bpm or params.bpm <= 0) and bpm and int(bpm) > 0:
                params.cot_bpm = bpm
            if not params.keyscale:
                params.cot_keyscale = key_scale
            if not params.timesignature:
                params.cot_timesignature = time_signature
            if (not params.duration or params.duration <= 0) and audio_duration and float(audio_duration) > 0:
                params.cot_duration = audio_duration
            if not params.vocal_language:
                params.cot_vocal_language = vocal_language
            if not params.caption:
                params.cot_caption = caption
            if not params.lyrics:
                params.cot_lyrics = lyrics

        if lm_generated_metadata is not None:
            if params.use_cot_caption:
                dit_input_caption = lm_generated_metadata.get("caption", dit_input_caption)
            if params.use_cot_language:
                dit_input_vocal_language = lm_generated_metadata.get("vocal_language", dit_input_vocal_language)

    if params.task_type in ("repaint", "cover", "cover-nofsq", "extract"):
        dit_input_caption = params.caption or dit_input_caption
        dit_input_lyrics = params.lyrics if params.lyrics is not None else dit_input_lyrics
        logger.info(
            f"[generate_music] {params.task_type} task: using params.caption='{params.caption}', params.lyrics='{params.lyrics}'"
        )
        logger.info(
            f"[generate_music] Final inputs: dit_input_caption='{dit_input_caption}', dit_input_lyrics='{dit_input_lyrics}'"
        )

    if params.task_type in ("cover", "cover-nofsq", "repaint", "lego", "extract"):
        audio_duration = None

    dit_generate_kwargs = {
        "captions": dit_input_caption,
        "global_caption": params.global_caption,
        "lyrics": dit_input_lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": dit_input_vocal_language,
        "inference_steps": params.inference_steps,
        "guidance_scale": params.guidance_scale,
        "use_random_seed": use_random_seed_for_dit,
        "seed": seed_for_generation,
        "reference_audio": params.reference_audio,
        "audio_duration": audio_duration,
        "batch_size": config.batch_size if config.batch_size is not None else 1,
        "src_audio": (params.src_audio if params.task_type != "text2music" or params.flow_edit_morph else None),
        "audio_code_string": audio_code_string_to_use,
        "repainting_start": params.repainting_start,
        "repainting_end": params.repainting_end,
        "chunk_mask_mode": params.chunk_mask_mode,
        "repaint_latent_crossfade_frames": params.repaint_latent_crossfade_frames,
        "repaint_wav_crossfade_sec": params.repaint_wav_crossfade_sec,
        "repaint_mode": params.repaint_mode,
        "repaint_strength": params.repaint_strength,
        "source_repaint_latents": source_repaint_latents,
        "retake_seed": params.retake_seed,
        "retake_variance": params.retake_variance,
        "flow_edit_morph": params.flow_edit_morph,
        "flow_edit_source_caption": params.flow_edit_source_caption,
        "flow_edit_source_lyrics": params.flow_edit_source_lyrics,
        "flow_edit_n_min": params.flow_edit_n_min,
        "flow_edit_n_max": params.flow_edit_n_max,
        "flow_edit_n_avg": params.flow_edit_n_avg,
        "instruction": params.instruction,
        "audio_cover_strength": params.audio_cover_strength,
        "cover_noise_strength": params.cover_noise_strength,
        "task_type": params.task_type,
        "use_adg": params.use_adg,
        "cfg_interval_start": params.cfg_interval_start,
        "cfg_interval_end": params.cfg_interval_end,
        "shift": params.shift,
        "infer_method": params.infer_method,
        "sampler_mode": params.sampler_mode,
        "velocity_norm_threshold": params.velocity_norm_threshold,
        "velocity_ema_factor": params.velocity_ema_factor,
        "dcw_enabled": params.dcw_enabled,
        "dcw_mode": params.dcw_mode,
        "dcw_scaler": params.dcw_scaler,
        "dcw_high_scaler": params.dcw_high_scaler,
        "dcw_wavelet": params.dcw_wavelet,
        "timesteps": params.timesteps,
        "latent_shift": params.latent_shift,
        "latent_rescale": params.latent_rescale,
        "progress": progress,
    }
    supported_generate_keys = set(inspect.signature(dit_handler.generate_music).parameters.keys())
    filtered = {k: v for k, v in dit_generate_kwargs.items() if k in supported_generate_keys}
    dropped = sorted(set(dit_generate_kwargs.keys()) - supported_generate_keys)
    if dropped:
        logger.warning(f"[generate_music] Skipping unsupported generate_music kwargs: {dropped}")
    return filtered


def handler_prepare_condition_tensors(
    dit_handler: Any,
    filtered_generate_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Run the same early path as ``AceStepHandler.generate_music`` through
    ``preprocess_batch`` and ``model.prepare_condition`` (no diffusion).
    """
    progress = dit_handler._resolve_generate_music_progress(filtered_generate_kwargs.get("progress"))
    readiness = dit_handler._validate_generate_music_readiness()
    if readiness is not None:
        raise RuntimeError(readiness.get("error", "handler not ready"))

    task_type, instruction = dit_handler._resolve_generate_music_task(
        task_type=filtered_generate_kwargs["task_type"],
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        instruction=filtered_generate_kwargs["instruction"],
    )
    guidance_scale = float(filtered_generate_kwargs["guidance_scale"])
    if dit_handler.is_turbo_model() and guidance_scale != 1.0:
        logger.info(
            "[generate_music] Turbo model detected: overriding "
            "guidance_scale {:.1f} -> 1.0 (turbo does not use CFG).",
            guidance_scale,
        )
        guidance_scale = 1.0

    if getattr(dit_handler, "lora_loaded", False) and getattr(dit_handler, "use_lora", False):
        dit_handler._verify_decoder_device_dtype()

    logger.info("[generate_music] Starting generation...")
    if progress:
        progress(0.51, desc="Preparing inputs...")
    logger.info("[generate_music] Preparing inputs...")

    runtime = dit_handler._prepare_generate_music_runtime(
        batch_size=filtered_generate_kwargs.get("batch_size"),
        audio_duration=filtered_generate_kwargs.get("audio_duration"),
        repainting_end=filtered_generate_kwargs.get("repainting_end"),
        seed=filtered_generate_kwargs.get("seed"),
        use_random_seed=filtered_generate_kwargs.get("use_random_seed", True),
        retake_seed=filtered_generate_kwargs.get("retake_seed"),
        retake_variance=filtered_generate_kwargs.get("retake_variance", 0.0),
    )
    actual_batch_size = runtime["actual_batch_size"]
    actual_seed_list = runtime["actual_seed_list"]
    audio_duration = runtime["audio_duration"]
    repainting_end = runtime["repainting_end"]

    refer_audios, processed_src_audio, audio_error = dit_handler._prepare_reference_and_source_audio(
        reference_audio=filtered_generate_kwargs.get("reference_audio"),
        src_audio=filtered_generate_kwargs.get("src_audio"),
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        actual_batch_size=actual_batch_size,
        task_type=task_type,
        flow_edit_morph=filtered_generate_kwargs.get("flow_edit_morph", False),
    )
    if audio_error is not None:
        raise RuntimeError(audio_error.get("error", str(audio_error)))

    if processed_src_audio is not None and (
        task_type in ("cover", "cover-nofsq", "repaint", "lego", "extract")
        or (task_type == "text2music" and filtered_generate_kwargs.get("flow_edit_morph"))
    ):
        audio_duration = processed_src_audio.shape[-1] / dit_handler.sample_rate

    if filtered_generate_kwargs.get("flow_edit_morph") and task_type not in ("text2music", "cover", "cover-nofsq"):
        logger.warning(
            "[generate_music] flow_edit_morph=True but task_type={!r}; "
            "v1 overlay only applies to text2music / cover / cover-nofsq, ignoring.",
            task_type,
        )

    service_inputs = dit_handler._prepare_generate_music_service_inputs(
        actual_batch_size=actual_batch_size,
        processed_src_audio=processed_src_audio,
        audio_duration=audio_duration,
        captions=filtered_generate_kwargs["captions"],
        global_caption=filtered_generate_kwargs.get("global_caption") or "",
        lyrics=filtered_generate_kwargs["lyrics"],
        vocal_language=filtered_generate_kwargs["vocal_language"],
        instruction=instruction,
        bpm=filtered_generate_kwargs.get("bpm"),
        key_scale=filtered_generate_kwargs.get("key_scale") or "",
        time_signature=filtered_generate_kwargs.get("time_signature") or "",
        task_type=task_type,
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        repainting_start=filtered_generate_kwargs.get("repainting_start", 0.0),
        repainting_end=repainting_end,
        chunk_mask_mode=filtered_generate_kwargs.get("chunk_mask_mode", "auto"),
    )

    vram_error = dit_handler._vram_preflight_check(
        actual_batch_size=actual_batch_size,
        audio_duration=audio_duration,
        guidance_scale=guidance_scale,
    )
    if vram_error is not None:
        raise RuntimeError(vram_error.get("error", "VRAM preflight failed"))

    normalized = dit_handler._normalize_service_generate_inputs(
        captions=service_inputs["captions_batch"],
        lyrics=service_inputs["lyrics_batch"],
        keys=None,
        metas=service_inputs["metas_batch"],
        vocal_languages=service_inputs["vocal_languages_batch"],
        repainting_start=service_inputs["repainting_start_batch"],
        repainting_end=service_inputs["repainting_end_batch"],
        instructions=service_inputs["instructions_batch"],
        audio_code_hints=service_inputs["audio_code_hints_batch"],
        infer_steps=int(filtered_generate_kwargs["inference_steps"]),
        seed=actual_seed_list,
        return_intermediate=service_inputs["should_return_intermediate"],
    )
    batch = dit_handler._prepare_batch(
        captions=normalized["captions"],
        global_captions=service_inputs.get("global_captions_batch"),
        lyrics=normalized["lyrics"],
        keys=normalized["keys"],
        target_wavs=service_inputs["target_wavs_tensor"],
        refer_audios=refer_audios,
        metas=normalized["metas"],
        vocal_languages=normalized["vocal_languages"],
        repainting_start=normalized["repainting_start"],
        repainting_end=normalized["repainting_end"],
        instructions=normalized["instructions"],
        audio_code_hints=normalized["audio_code_hints"],
        audio_cover_strength=filtered_generate_kwargs.get("audio_cover_strength", 1.0),
        cover_noise_strength=filtered_generate_kwargs.get("cover_noise_strength", 0.0),
        chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
        task_type=task_type,
        source_repaint_latents=filtered_generate_kwargs.get("source_repaint_latents"),
    )
    processed = dit_handler.preprocess_batch(batch)
    payload = dit_handler._unpack_service_processed_data(processed)

    dit_backend = (
        "MLX (native)"
        if (getattr(dit_handler, "use_mlx_dit", False) and getattr(dit_handler, "mlx_decoder", None) is not None)
        else f"PyTorch ({dit_handler.device})"
    )
    logger.info(f"[service_generate] Generating audio... (DiT backend: {dit_backend})")
    with torch.inference_mode():
        enc_hs, enc_mask, ctx = dit_handler.model.prepare_condition(
            text_hidden_states=payload["text_hidden_states"],
            text_attention_mask=payload["text_attention_mask"],
            lyric_hidden_states=payload["lyric_hidden_states"],
            lyric_attention_mask=payload["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=payload["refer_audio_order_mask"],
            hidden_states=payload["src_latents"],
            attention_mask=torch.ones(
                payload["src_latents"].shape[0],
                payload["src_latents"].shape[1],
                device=payload["src_latents"].device,
                dtype=payload["src_latents"].dtype,
            ),
            silence_latent=dit_handler.silence_latent,
            src_latents=payload["src_latents"],
            chunk_masks=payload["chunk_mask"],
            is_covers=payload["is_covers"],
            precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
        )

    frames = int(payload["src_latents"].shape[1])
    null_emb = getattr(dit_handler.model, "null_condition_emb", None)
    if null_emb is None:
        raise RuntimeError("null_condition_emb missing on handler.model")

    enc_hs = enc_hs.detach().float().cpu()
    enc_mask = enc_mask.detach().float().cpu()
    ctx = ctx.detach().float().cpu()
    null_emb = null_emb.detach().float().cpu()

    return enc_hs, enc_mask, ctx, frames, null_emb
