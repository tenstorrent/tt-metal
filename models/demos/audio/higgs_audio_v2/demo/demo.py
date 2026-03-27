# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import soundfile as sf
import torch
from loguru import logger

import ttnn
from models.demos.audio.higgs_audio_v2.demo._audio_decode import (
    apply_delay_pattern_to_greedy_audio_tokens,
    initialize_delay_pattern_state,
)
from models.demos.audio.higgs_audio_v2.demo._prompts import (
    DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    build_demo_sample,
    load_chatml_sample_from_json,
)
from models.demos.audio.higgs_audio_v2.tt.model import create_higgs_tt_model
from models.demos.audio.higgs_audio_v2.tt.reference import (
    load_audio_tokenizer,
    load_higgs_config,
    load_higgs_tokenizer,
    prepare_inputs_for_generation,
    revert_delay_pattern,
)


def _build_chatml_sample(
    transcript: str | None,
    system_prompt: str,
    ref_audio: str | None,
    ref_transcript: str | None,
    messages_json: str | None,
    reference_audio_manifest: str | None,
    reference_audio_assets_root: str | None,
):
    if messages_json:
        return load_chatml_sample_from_json(
            messages_json,
            reference_audio_manifest_path=reference_audio_manifest or str(DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH),
            reference_audio_assets_root=reference_audio_assets_root,
        )
    if transcript is None:
        raise ValueError("`--transcript` is required when `--messages-json` is not provided.")
    return build_demo_sample(
        transcript=transcript,
        system_prompt=system_prompt,
        ref_audio=ref_audio,
        ref_transcript=ref_transcript,
    )


def run_demo(
    model_path: str,
    audio_tokenizer_path: str,
    out_path: str,
    transcript: str | None = None,
    system_prompt: str = "Generate audio following instruction.",
    ref_audio: str | None = None,
    ref_transcript: str | None = None,
    messages_json: str | None = None,
    max_new_tokens: int = 256,
    optimizations: str = "accuracy",
    use_hf_rope: bool = True,
    reference_audio_manifest: str | None = None,
    reference_audio_assets_root: str | None = None,
) -> str:
    config = load_higgs_config(model_path)
    tokenizer = load_higgs_tokenizer(model_path)
    audio_tokenizer = load_audio_tokenizer(audio_tokenizer_path, device="cpu")
    chat_ml_sample = _build_chatml_sample(
        transcript=transcript,
        system_prompt=system_prompt,
        ref_audio=ref_audio,
        ref_transcript=ref_transcript,
        messages_json=messages_json,
        reference_audio_manifest=reference_audio_manifest,
        reference_audio_assets_root=reference_audio_assets_root,
    )
    model_inputs = prepare_inputs_for_generation(
        chat_ml_sample=chat_ml_sample,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        config=config,
        force_audio_gen=True,
        device="cpu",
    )
    if int(model_inputs["input_ids"][0, -1].item()) != config.audio_out_bos_token_id:
        raise RuntimeError("Prepared prompt did not end at the audio generation boundary.")

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    current_pos_tt = None
    try:
        _, tt_model, _ = create_higgs_tt_model(
            mesh_device,
            model_path,
            optimizations=optimizations,
            use_hf_rope=use_hf_rope,
        )
        prompt_embeddings, prompt_audio_mask = tt_model.embed_prompt_inputs(
            input_ids=model_inputs["input_ids"],
            audio_in_ids=model_inputs.get("audio_in_ids"),
            audio_in_ids_start=model_inputs.get("audio_in_ids_start"),
            audio_out_ids=model_inputs.get("audio_out_ids"),
            audio_out_ids_start=model_inputs.get("audio_out_ids_start"),
        )
        tt_model.prefill(prompt_embeddings, prompt_audio_mask, return_logits=False)

        audio_sequence = torch.full(
            (config.audio_num_codebooks, 1),
            config.audio_stream_bos_id,
            dtype=torch.long,
        )
        num_delay, num_remaining_delays = initialize_delay_pattern_state(audio_sequence, config)
        current_pos = prompt_embeddings.shape[1]
        current_pos_tt = tt_model.create_current_pos_tensor(current_pos)
        finished = False

        for _ in range(max_new_tokens):
            current_embedding = tt_model.embed_audio_tokens(audio_sequence[:, -1:])[0]
            _, audio_logits_flat = tt_model.decode_step(
                current_embedding=current_embedding,
                current_pos=current_pos,
                is_audio_token=True,
                return_text_logits=False,
                current_pos_tt=current_pos_tt,
            )
            current_pos += 1
            tt_model.increment_current_pos_tensor(current_pos_tt)
            if audio_logits_flat is None:
                raise RuntimeError("Audio decode step did not return audio logits.")
            (
                next_audio_tokens,
                _,
                num_delay,
                num_remaining_delays,
                finished,
            ) = apply_delay_pattern_to_greedy_audio_tokens(
                audio_logits_flat.view(config.audio_num_codebooks, -1),
                config,
                num_delay,
                num_remaining_delays,
            )
            audio_sequence = torch.cat([audio_sequence, next_audio_tokens.unsqueeze(1)], dim=1)
            if finished:
                break

        if not finished:
            logger.warning(
                "Audio generation did not reach EOS within {} steps; decoding the truncated sequence.", max_new_tokens
            )

        vq_tokens = revert_delay_pattern(audio_sequence)
        if vq_tokens.shape[1] <= 1:
            raise RuntimeError("The model generated no decodable audio frames.")
        vq_tokens = vq_tokens[:, 1:]
        while vq_tokens.shape[1] > 0 and torch.all(vq_tokens[:, -1] == config.audio_stream_eos_id):
            vq_tokens = vq_tokens[:, :-1]
        vq_code = vq_tokens.clip(0, config.audio_codebook_size - 1)
        if vq_code.shape[1] == 0:
            raise RuntimeError("The model generated no decodable audio frames.")
        waveform = audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
        sf.write(out_path, waveform, audio_tokenizer.sampling_rate)
        logger.info("Saved waveform to {}", out_path)
        return out_path
    finally:
        if current_pos_tt is not None:
            ttnn.deallocate(current_pos_tt)
        ttnn.close_mesh_device(mesh_device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal greedy Higgs Audio v2 TT demo.")
    parser.add_argument("--model-path", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--audio-tokenizer-path", default="bosonai/higgs-audio-v2-tokenizer")
    parser.add_argument("--transcript")
    parser.add_argument("--messages-json")
    parser.add_argument("--out-path", default="higgs_audio_tt.wav")
    parser.add_argument("--system-prompt", default="Generate audio following instruction.")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-transcript")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--optimizations", choices=("accuracy", "performance"), default="accuracy")
    parser.add_argument("--use-hf-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-audio-manifest")
    parser.add_argument("--reference-audio-assets-root")
    args = parser.parse_args()
    run_demo(
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        out_path=args.out_path,
        transcript=args.transcript,
        system_prompt=args.system_prompt,
        ref_audio=args.ref_audio,
        ref_transcript=args.ref_transcript,
        messages_json=args.messages_json,
        max_new_tokens=args.max_new_tokens,
        optimizations=args.optimizations,
        use_hf_rope=args.use_hf_rope,
        reference_audio_manifest=args.reference_audio_manifest,
        reference_audio_assets_root=args.reference_audio_assets_root,
    )


if __name__ == "__main__":
    main()
