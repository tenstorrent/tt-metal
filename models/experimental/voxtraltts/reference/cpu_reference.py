# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import safe_open
from transformers import MistralConfig, MistralForCausalLM

from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import (
    AudioSpecialTokens,
    FlowMatchingAudioTransformerRef,
    build_audio_model_args_from_voxtral_config,
)
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.reference.voxtral_request import (
    compose_speech_request,
    get_instruct_tokenizer,
    load_mistral_tokenizer,
)

_ACOUSTIC_CFG_ALPHA = 1.2


def _resolve_model_file(model_name_or_path: str, filename: str) -> Path:
    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        return model_path / filename

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download Voxtral reference files.") from exc

    return Path(hf_hub_download(model_name_or_path, filename=filename))


def _torch_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def _build_text_config(model_name_or_path: str) -> MistralConfig:
    config = load_voxtral_config(model_name_or_path)
    return MistralConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.hidden_dim,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        hidden_act="silu",
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_theta,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )


def _map_text_weight(name: str) -> str | None:
    if name == "mm_audio_embeddings.tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name.startswith("layers."):
        replacements = (
            (".attention.wq.", ".self_attn.q_proj."),
            (".attention.wk.", ".self_attn.k_proj."),
            (".attention.wv.", ".self_attn.v_proj."),
            (".attention.wo.", ".self_attn.o_proj."),
            (".feed_forward.w1.", ".mlp.gate_proj."),
            (".feed_forward.w2.", ".mlp.down_proj."),
            (".feed_forward.w3.", ".mlp.up_proj."),
            (".attention_norm.", ".input_layernorm."),
            (".ffn_norm.", ".post_attention_layernorm."),
        )
        hf_name = f"model.{name}"
        for old, new in replacements:
            hf_name = hf_name.replace(old, new)
        return hf_name
    return None


def _permute_mistral_attention_weight(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
) -> torch.Tensor:
    attn_in = head_dim * num_heads
    return (
        weight.view(num_heads, attn_in // num_heads // 2, 2, hidden_size).transpose(1, 2).reshape(attn_in, hidden_size)
    )


def _load_text_state_dict(checkpoint_path: Path, config: Any) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            mapped_name = _map_text_weight(name)
            if mapped_name is not None:
                tensor = f.get_tensor(name)
                if name.endswith(".attention.wq.weight"):
                    tensor = _permute_mistral_attention_weight(
                        tensor,
                        num_heads=config.n_heads,
                        head_dim=config.head_dim,
                        hidden_size=config.dim,
                    )
                elif name.endswith(".attention.wk.weight"):
                    tensor = _permute_mistral_attention_weight(
                        tensor,
                        num_heads=config.n_kv_heads,
                        head_dim=config.head_dim,
                        hidden_size=config.dim,
                    )
                state_dict[mapped_name] = tensor
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    return state_dict


def _audio_config_namespace(model_name_or_path: str) -> SimpleNamespace:
    config = load_voxtral_config(model_name_or_path)
    text_config = _build_text_config(model_name_or_path)
    audio_config = {
        "sampling_rate": config.audio_model_args.audio_encoding_args.sampling_rate,
        "codec_args": config.audio_tokenizer_args.__dict__,
        "audio_model_args": {
            **config.audio_model_args.__dict__,
            "audio_encoding_args": config.audio_model_args.audio_encoding_args.__dict__,
            "acoustic_transformer_args": config.audio_model_args.acoustic_transformer_args.__dict__,
        },
        "speaker_id": config.audio_tokenizer_args.voice,
    }
    return SimpleNamespace(text_config=text_config, audio_config=audio_config)


def _fake_vllm_config(hf_config: Any) -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))


class VoxtralCPUReference:
    """Direct PyTorch CPU reference for Voxtral TTS bring-up.

    This intentionally bypasses vLLM-Omni's CUDA workers. It is slow, but it gives
    TTNN bring-up a host-side golden path for the text backbone, acoustic
    transformer, and audio tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        dtype: str = "bfloat16",
        device: str = "cpu",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.dtype = _torch_dtype(dtype)
        self.config = load_voxtral_config(model_name_or_path)
        self.tokenizer = load_mistral_tokenizer(model_name_or_path)
        self.checkpoint_path = _resolve_model_file(model_name_or_path, "consolidated.safetensors")

        self.text_model = MistralForCausalLM(_build_text_config(model_name_or_path))
        missing, unexpected = self.text_model.load_state_dict(
            _load_text_state_dict(self.checkpoint_path, self.config), strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected text-model weights: {unexpected}")
        missing = [name for name in missing if name != "lm_head.weight"]
        if missing:
            raise RuntimeError(f"Missing text-model weights: {missing[:10]} ... total={len(missing)}")
        self.text_model.to(device=self.device, dtype=self.dtype).eval()

        from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_tokenizer import (
            VoxtralTTSAudioTokenizer,
        )

        self.audio_special_tokens = AudioSpecialTokens
        self.hf_audio_config = _audio_config_namespace(model_name_or_path)
        fake_config = _fake_vllm_config(self.hf_audio_config)
        audio_model_args = build_audio_model_args_from_voxtral_config(self.config)
        self.acoustic_transformer = FlowMatchingAudioTransformerRef(audio_model_args)
        self.audio_tokenizer = VoxtralTTSAudioTokenizer(vllm_config=fake_config)
        self._acoustic_cfg_alpha = _ACOUSTIC_CFG_ALPHA
        self._load_audio_weights()
        self.acoustic_transformer.to(device=self.device, dtype=self.dtype).eval()
        self.audio_tokenizer.to(device=self.device, dtype=self.dtype).eval()

        instruct_tokenizer = get_instruct_tokenizer(self.tokenizer)
        audio_encoder = getattr(instruct_tokenizer, "audio_encoder", None)
        self.audio_token_id = getattr(audio_encoder, "audio_token", None) if audio_encoder is not None else None
        if self.audio_token_id is None:
            self.audio_token_id = self.config.audio_model_args.audio_token_id
        self.end_audio_id = self.audio_special_tokens.id(self.audio_special_tokens.end_audio)

    def _load_audio_weights(self) -> None:
        with safe_open(self.checkpoint_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.startswith("acoustic_transformer."):
                    self.acoustic_transformer.load_weight((name.removeprefix("acoustic_transformer."), tensor))
                elif name.startswith("audio_tokenizer."):
                    self.audio_tokenizer.load_weight((name.removeprefix("audio_tokenizer."), tensor))
                elif name.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
                    remapped = name.removeprefix("mm_audio_embeddings.audio_codebook_embeddings.")
                    self.audio_tokenizer.load_weight((f"audio_token_embedding.{remapped}", tensor))

    def _load_voice_embedding(self, voice: str) -> torch.Tensor:
        path = _resolve_model_file(self.model_name_or_path, f"voice_embedding/{voice}.pt")
        return torch.load(path, map_location="cpu").to(device=self.device, dtype=self.dtype)

    def _prompt_embeddings(self, prompt_token_ids: list[int], voice: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
        inputs_embeds = self.text_model.model.embed_tokens(input_ids)
        if voice is not None:
            voice_embedding = self._load_voice_embedding(voice)
            audio_mask = input_ids == self.audio_token_id
            if int(audio_mask.sum().item()) != voice_embedding.shape[0]:
                raise RuntimeError(
                    "Voice embedding length does not match audio placeholder tokens: "
                    f"{voice_embedding.shape[0]} vs {int(audio_mask.sum().item())}"
                )
            inputs_embeds[audio_mask] = voice_embedding
        return input_ids, inputs_embeds

    def _decode_audio_codes(self, codes: torch.Tensor) -> torch.Tensor:
        eoa = (codes[:, 0] == self.end_audio_id).nonzero(as_tuple=False)
        cutting_point = int(eoa[0].item()) if len(eoa) else codes.shape[0]
        audio_tokens = codes[:cutting_point] - 2
        if audio_tokens.numel() == 0:
            return torch.tensor([], dtype=torch.float32)
        audio_values = self.audio_tokenizer.decode(audio_tokens.T.unsqueeze(0), dtype=self.dtype)
        audio_values = audio_values.detach().cpu().float().squeeze(0).squeeze(0)
        expected_samples = audio_tokens.shape[0] * self.audio_tokenizer.downsample_factor
        return audio_values[:expected_samples]

    def _audio_codes_to_input_embeds(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # vLLM-Omni feeds generated audio codes back as multimodal embeddings,
        # not as the plain text embedding of the audio placeholder token.
        audio_embeddings = self.audio_tokenizer.encode_tokens([audio_codes.unsqueeze(-1)])[0]
        return audio_embeddings.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        voice: str | None = "casual_male",
        ref_audio: str | None = None,
        max_tokens: int = 2500,
        seed: int = 0,
    ) -> torch.Tensor:
        if ref_audio is not None:
            raise NotImplementedError(
                "CPU reference currently supports preset voice embeddings, not ref-audio cloning."
            )
        if voice is None:
            raise ValueError("voice must be provided for CPU reference generation.")

        torch.manual_seed(seed)
        request = compose_speech_request(text, self.model_name_or_path, voice=voice, ref_audio=None)
        _, inputs_embeds = self._prompt_embeddings(request["prompt_token_ids"], voice)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1][:, -1, :]
        past_key_values = outputs.past_key_values
        generated_codes = []

        cfg_alpha = torch.tensor(self._acoustic_cfg_alpha, device=hidden.device, dtype=hidden.dtype)
        for _ in range(max_tokens):
            audio_codes = self.acoustic_transformer(hidden, cfg_alpha).to(torch.long)
            generated_codes.append(audio_codes[0].detach().cpu())
            if int(audio_codes[0, 0].item()) == self.end_audio_id:
                break

            next_input_embeds = self._audio_codes_to_input_embeds(audio_codes)
            outputs = self.text_model(
                inputs_embeds=next_input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values

        if not generated_codes:
            return torch.tensor([], dtype=torch.float32)
        return self._decode_audio_codes(torch.stack(generated_codes, dim=0).to(self.device))
