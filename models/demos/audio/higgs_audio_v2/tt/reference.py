# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoTokenizer

from models.demos.audio.higgs_audio_v2.demo._prompts import AudioContent, ChatMLSample, TextContent


def _ensure_higgs_audio_importable() -> None:
    try:
        import boson_multimodal.audio_processing  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import `boson_multimodal`. Export "
            "`PYTHONPATH=$HIGGS_AUDIO_REPO:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}` "
            "before running the Higgs Audio demo or tests."
        ) from exc


def load_higgs_config(model_name_or_path: str):
    checkpoint_path = snapshot_download(model_name_or_path, allow_patterns=["config.json"])
    with open(os.path.join(checkpoint_path, "config.json"), "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    text_config_dict = raw_config["text_config"]
    text_model_type = text_config_dict["model_type"]
    text_config = AutoConfig.for_model(
        text_model_type,
        **{key: value for key, value in text_config_dict.items() if key != "model_type"},
    )
    raw_config["text_config"] = text_config
    return SimpleNamespace(**raw_config)


def load_higgs_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_audio_tokenizer(tokenizer_name_or_path: str, device: str = "cpu"):
    _ensure_higgs_audio_importable()
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

    return load_higgs_audio_tokenizer(tokenizer_name_or_path, device=device)


def load_higgs_model_state_dict(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    checkpoint_path = snapshot_download(model_name_or_path)
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(set(index_data["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]

    state_dict = {}
    for shard_name in shard_names:
        shard_path = os.path.join(checkpoint_path, shard_name)
        shard_state = load_safetensors_file(shard_path, device="cpu")
        if dtype is not None:
            shard_state = {
                key: value.to(dtype=dtype) if value.is_floating_point() and value.dtype != dtype else value
                for key, value in shard_state.items()
            }
        state_dict.update(shard_state)
    return state_dict


def remap_higgs_state_dict_to_tt(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    audio_vocab_size: int,
) -> dict[str, torch.Tensor]:
    remapped = {
        "audio_codebook_embeddings.weight": state_dict["audio_codebook_embeddings.weight"],
        "norm.weight": state_dict["norm.weight"],
        "output.weight": state_dict["audio_decoder_proj.text_lm_head.weight"],
        "audio_output.weight": state_dict["audio_decoder_proj.audio_lm_head.weight"][:audio_vocab_size],
        "tok_embeddings.weight": state_dict["embed_tokens.weight"],
    }

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"
        self_attn_prefix = f"{prefix}.self_attn"
        remapped[f"{prefix}.attention.wq.weight"] = state_dict[f"{self_attn_prefix}.q_proj.weight"]
        remapped[f"{prefix}.attention.wk.weight"] = state_dict[f"{self_attn_prefix}.k_proj.weight"]
        remapped[f"{prefix}.attention.wv.weight"] = state_dict[f"{self_attn_prefix}.v_proj.weight"]
        remapped[f"{prefix}.attention.wo.weight"] = state_dict[f"{self_attn_prefix}.o_proj.weight"]

        remapped[f"{prefix}.attention_norm.weight"] = state_dict[f"{prefix}.input_layernorm.weight"]
        remapped[f"{prefix}.ffn_norm.weight"] = state_dict[f"{prefix}.post_attention_layernorm.weight"]
        remapped[f"{prefix}.audio_attention_norm.weight"] = state_dict[f"{prefix}.audio_input_layernorm.weight"]
        remapped[f"{prefix}.audio_ffn_norm.weight"] = state_dict[f"{prefix}.audio_post_attention_layernorm.weight"]

        remapped[f"{prefix}.feed_forward.w1.weight"] = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        remapped[f"{prefix}.feed_forward.w2.weight"] = state_dict[f"{prefix}.mlp.down_proj.weight"]
        remapped[f"{prefix}.feed_forward.w3.weight"] = state_dict[f"{prefix}.mlp.up_proj.weight"]

        remapped[f"{prefix}.audio_feed_forward.w1.weight"] = state_dict[f"{prefix}.audio_mlp.gate_proj.weight"]
        remapped[f"{prefix}.audio_feed_forward.w2.weight"] = state_dict[f"{prefix}.audio_mlp.down_proj.weight"]
        remapped[f"{prefix}.audio_feed_forward.w3.weight"] = state_dict[f"{prefix}.audio_mlp.up_proj.weight"]

    return remapped


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    bsz, num_codebooks, seq_len = input_ids.shape
    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    masked_input_ids = input_ids_with_gen_mask.clone()
    masked_input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return masked_input_ids, input_ids_with_gen_mask


def revert_delay_pattern(data: torch.Tensor) -> torch.Tensor:
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


def _normalize_audio_content(content) -> AudioContent:
    if isinstance(content, AudioContent):
        return content
    if hasattr(content, "type") and getattr(content, "type") == "audio":
        return AudioContent(
            audio_url=getattr(content, "audio_url"),
            raw_audio=getattr(content, "raw_audio", None),
            offset=getattr(content, "offset", None),
            duration=getattr(content, "duration", None),
            row_id=getattr(content, "row_id", None),
        )
    raise TypeError(f"Unsupported audio content type: {type(content)!r}")


def _normalize_text_content(content) -> TextContent:
    if isinstance(content, TextContent):
        return content
    if hasattr(content, "type") and getattr(content, "type") == "text":
        return TextContent(text=getattr(content, "text"))
    raise TypeError(f"Unsupported text content type: {type(content)!r}")


def _iter_message_contents(content) -> list[TextContent | AudioContent]:
    if isinstance(content, str):
        return [TextContent(text=content)]
    if isinstance(content, (TextContent, AudioContent)):
        return [content]
    if isinstance(content, list):
        normalized = []
        for element in content:
            if isinstance(element, str):
                normalized.append(TextContent(text=element))
            elif getattr(element, "type", None) == "audio":
                normalized.append(_normalize_audio_content(element))
            else:
                normalized.append(_normalize_text_content(element))
        return normalized
    if getattr(content, "type", None) == "audio":
        return [_normalize_audio_content(content)]
    return [_normalize_text_content(content)]


def prepare_chatml_sample(sample: ChatMLSample, tokenizer):
    input_tokens = []
    label_tokens = []
    audio_contents = []

    speaker_id = getattr(sample, "speaker", None)
    if speaker_id is None and getattr(sample, "misc", None) is not None:
        speaker_id = sample.misc.get("speaker")

    total_messages = len(sample.messages)
    for turn_id, message in enumerate(sample.messages):
        role = message.role
        recipient = getattr(message, "recipient", None)
        content_l = _iter_message_contents(message.content)

        if turn_id == 0:
            prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
        else:
            prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        input_tokens.extend(prefix_tokens)
        label_tokens.extend([-100 for _ in prefix_tokens])

        if recipient:
            recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
            input_tokens.extend(recipient_tokens)
            label_tokens.extend(recipient_tokens)

        for content in content_l:
            if content.type == "text":
                text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                input_tokens.extend(text_tokens)
                if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                    label_tokens.extend(text_tokens)
                else:
                    label_tokens.extend([-100 for _ in text_tokens])
                continue

            normalized_audio = _normalize_audio_content(content)
            if role in ("user", "system"):
                audio_tokens = tokenizer.encode("<|audio_bos|><|AUDIO|><|audio_eos|>", add_special_tokens=False)
                placeholder_type = "audio_in"
            else:
                audio_tokens = tokenizer.encode("<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>", add_special_tokens=False)
                placeholder_type = "audio_out"
            input_tokens.extend(audio_tokens)
            if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                label_tokens.extend(audio_tokens)
            else:
                label_tokens.extend([-100 for _ in audio_tokens])
            audio_contents.append((placeholder_type, normalized_audio))

        postfix = "<|eot_id|>"
        next_id = turn_id + 1
        if role == "assistant" and next_id != total_messages and sample.messages[next_id].role == "assistant":
            postfix = "<|eom_id|>"
        postfix_tokens = tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix_tokens)
        if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
            label_tokens.extend(postfix_tokens)
        else:
            label_tokens.extend([-100 for _ in postfix_tokens])

    return input_tokens, label_tokens, audio_contents, speaker_id


def _transform_audio_codes(audio_codes: torch.Tensor, config) -> torch.Tensor:
    audio_codes = audio_codes[: config.audio_num_codebooks]
    audio_codes = torch.cat(
        [
            torch.full((audio_codes.shape[0], 1), config.audio_stream_bos_id, dtype=torch.long),
            audio_codes,
            torch.full((audio_codes.shape[0], 1), config.audio_stream_eos_id, dtype=torch.long),
        ],
        dim=1,
    )
    if config.use_delay_pattern:
        audio_codes = build_delay_pattern_mask(
            audio_codes.unsqueeze(0),
            bos_token_id=config.audio_stream_bos_id,
            pad_token_id=config.audio_stream_eos_id,
        )[0].squeeze(0)
    return audio_codes.long()


def _concat_audio_segments(audio_segments: list[torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not audio_segments:
        return None, None
    starts = torch.tensor([0] + [segment.shape[1] for segment in audio_segments[:-1]], dtype=torch.long)
    starts = torch.cumsum(starts, dim=0)
    return torch.cat(audio_segments, dim=1).long(), starts


def _load_audio_for_prompt(audio_content: AudioContent, sampling_rate: int):
    import librosa

    load_kwargs = {
        "sr": sampling_rate,
        "offset": audio_content.offset or 0.0,
        "duration": audio_content.duration,
    }
    if audio_content.audio_url not in ("", "placeholder"):
        raw_audio, _ = librosa.load(audio_content.audio_url, **load_kwargs)
        return raw_audio
    if audio_content.raw_audio is not None:
        raw_audio, _ = librosa.load(BytesIO(base64.b64decode(audio_content.raw_audio)), **load_kwargs)
        return raw_audio
    return None


def _encode_prompt_audio(audio_content: AudioContent, audio_tokenizer, config) -> torch.Tensor:
    raw_audio = _load_audio_for_prompt(audio_content, audio_tokenizer.sampling_rate)
    if raw_audio is None:
        raise ValueError("Audio placeholders in the prompt require `audio_url` or `raw_audio`.")
    audio_ids = torch.as_tensor(audio_tokenizer.encode(raw_audio, audio_tokenizer.sampling_rate))
    if audio_ids.dim() == 3 and audio_ids.shape[0] == 1:
        audio_ids = audio_ids.squeeze(0)
    elif audio_ids.dim() == 3 and audio_ids.shape[1] == 1:
        audio_ids = audio_ids.squeeze(1)
    if audio_ids.dim() != 2:
        raise ValueError(f"Unexpected encoded audio shape: {tuple(audio_ids.shape)}")
    return _transform_audio_codes(audio_ids.cpu(), config)


def prepare_inputs_for_generation(
    chat_ml_sample,
    tokenizer,
    audio_tokenizer,
    config,
    force_audio_gen: bool = True,
    device: str = "cpu",
):
    input_tokens, _, audio_contents, _ = prepare_chatml_sample(chat_ml_sample, tokenizer)
    postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if force_audio_gen:
        postfix += "<|audio_out_bos|>"
    input_tokens.extend(tokenizer.encode(postfix, add_special_tokens=False))

    audio_in_segments = []
    audio_out_segments = []
    for placeholder_type, audio_content in audio_contents:
        encoded_audio = _encode_prompt_audio(audio_content, audio_tokenizer, config)
        if placeholder_type == "audio_in":
            if not config.encode_audio_in_tokens:
                raise NotImplementedError(
                    "This Higgs bring-up only supports checkpoints with `encode_audio_in_tokens=True`."
                )
            audio_in_segments.append(encoded_audio)
        else:
            audio_out_segments.append(encoded_audio)

    audio_in_ids, audio_in_ids_start = _concat_audio_segments(audio_in_segments)
    audio_out_ids, audio_out_ids_start = _concat_audio_segments(audio_out_segments)

    inputs = {
        "input_ids": torch.LongTensor(input_tokens).unsqueeze(0),
        "attention_mask": torch.ones((1, len(input_tokens)), dtype=torch.long),
        "audio_in_ids": audio_in_ids,
        "audio_in_ids_start": audio_in_ids_start,
        "audio_out_ids": audio_out_ids,
        "audio_out_ids_start": audio_out_ids_start,
    }
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs


def embed_audio_ids(
    audio_ids: torch.Tensor,
    audio_embedding_weights: torch.Tensor,
    audio_num_codebooks: int,
    audio_codebook_size: int,
    audio_embed_avg: bool = False,
) -> torch.Tensor:
    codebook_shift = torch.arange(audio_num_codebooks, device=audio_ids.device) * audio_codebook_size
    embeddings = torch.nn.functional.embedding(audio_ids + codebook_shift.unsqueeze(-1), audio_embedding_weights)
    if audio_embed_avg:
        return embeddings.mean(dim=0)
    return embeddings.sum(dim=0)
