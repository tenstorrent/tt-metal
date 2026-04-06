from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CosyVoiceLLMConfig:
    text_encoder_input_size: int = 512
    llm_input_size: int = 1024
    llm_output_size: int = 1024
    text_token_size: int = 51866
    speech_token_size: int = 4096
    text_encoder_blocks: int = 6
    transformer_blocks: int = 14
    num_heads: int = 16


@dataclass(frozen=True)
class CosyVoiceSemanticInputs:
    text: torch.Tensor
    text_len: torch.Tensor
    prompt_text: torch.Tensor
    prompt_text_len: torch.Tensor
    prompt_speech_token: torch.Tensor
    prompt_speech_token_len: torch.Tensor
    llm_embedding: torch.Tensor
    merged_text: torch.Tensor
    merged_text_len: torch.Tensor
    encoded_text: torch.Tensor
    encoded_text_len: torch.Tensor
    speaker_projection: torch.Tensor
    sos_embedding: torch.Tensor
    task_embedding: torch.Tensor
    prompt_speech_embeddings: torch.Tensor
    lm_input: torch.Tensor
    lm_input_embed: torch.Tensor
    lm_input_positional_embedding: torch.Tensor
    lm_input_mask: torch.Tensor
    min_decode_length: int
    max_decode_length: int

    @property
    def prompt_speech_length(self) -> int:
        return int(self.prompt_speech_token.shape[1])

    @property
    def prefill_seq_len(self) -> int:
        return int(self.lm_input.shape[1])


@dataclass(frozen=True)
class CosyVoiceLegacyEmbedParameters:
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    epsilon: float = 1e-5


@dataclass(frozen=True)
class CosyVoiceSemanticLayerParameters:
    q_weight: torch.Tensor
    q_bias: torch.Tensor
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor
    out_weight: torch.Tensor
    out_bias: torch.Tensor
    pos_weight: torch.Tensor
    pos_bias_u: torch.Tensor
    pos_bias_v: torch.Tensor
    ffn_w1_weight: torch.Tensor
    ffn_w1_bias: torch.Tensor
    ffn_w2_weight: torch.Tensor
    ffn_w2_bias: torch.Tensor
    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    norm2_weight: torch.Tensor
    norm2_bias: torch.Tensor


@dataclass(frozen=True)
class CosyVoiceSemanticOutputParameters:
    after_norm_weight: torch.Tensor
    after_norm_bias: torch.Tensor
    decoder_weight: torch.Tensor
    decoder_bias: torch.Tensor
    speech_embedding_weight: torch.Tensor
    after_norm_epsilon: float = 1e-5


def compute_decode_length_bounds(
    text_token_len: int,
    prompt_text_token_len: int,
    min_token_text_ratio: float = 2.0,
    max_token_text_ratio: float = 20.0,
) -> tuple[int, int]:
    decode_text_tokens = max(text_token_len - prompt_text_token_len, 0)
    min_len = int(decode_text_tokens * min_token_text_ratio)
    max_len = int(decode_text_tokens * max_token_text_ratio)
    return min_len, max_len


def build_autoregressive_attention_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.bool))


def build_next_decode_embedding(speech_embedding_weight: torch.Tensor, token_id: int) -> torch.Tensor:
    return speech_embedding_weight[token_id].reshape(1, 1, -1)


def extract_legacy_embed_parameters(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "llm.embed.out",
) -> CosyVoiceLegacyEmbedParameters:
    prefix = prefix.rstrip(".")
    return CosyVoiceLegacyEmbedParameters(
        linear_weight=state_dict[f"{prefix}.0.weight"],
        linear_bias=state_dict[f"{prefix}.0.bias"],
        norm_weight=state_dict[f"{prefix}.1.weight"],
        norm_bias=state_dict[f"{prefix}.1.bias"],
    )


def extract_semantic_layer_parameters(
    state_dict: dict[str, torch.Tensor],
    layer_num: int,
    prefix: str = "llm.encoders",
) -> CosyVoiceSemanticLayerParameters:
    layer_prefix = f"{prefix.rstrip('.')}.{layer_num}"
    return CosyVoiceSemanticLayerParameters(
        q_weight=state_dict[f"{layer_prefix}.self_attn.linear_q.weight"],
        q_bias=state_dict[f"{layer_prefix}.self_attn.linear_q.bias"],
        k_weight=state_dict[f"{layer_prefix}.self_attn.linear_k.weight"],
        k_bias=state_dict[f"{layer_prefix}.self_attn.linear_k.bias"],
        v_weight=state_dict[f"{layer_prefix}.self_attn.linear_v.weight"],
        v_bias=state_dict[f"{layer_prefix}.self_attn.linear_v.bias"],
        out_weight=state_dict[f"{layer_prefix}.self_attn.linear_out.weight"],
        out_bias=state_dict[f"{layer_prefix}.self_attn.linear_out.bias"],
        pos_weight=state_dict[f"{layer_prefix}.self_attn.linear_pos.weight"],
        pos_bias_u=state_dict[f"{layer_prefix}.self_attn.pos_bias_u"],
        pos_bias_v=state_dict[f"{layer_prefix}.self_attn.pos_bias_v"],
        ffn_w1_weight=state_dict[f"{layer_prefix}.feed_forward.w_1.weight"],
        ffn_w1_bias=state_dict[f"{layer_prefix}.feed_forward.w_1.bias"],
        ffn_w2_weight=state_dict[f"{layer_prefix}.feed_forward.w_2.weight"],
        ffn_w2_bias=state_dict[f"{layer_prefix}.feed_forward.w_2.bias"],
        norm1_weight=state_dict[f"{layer_prefix}.norm1.weight"],
        norm1_bias=state_dict[f"{layer_prefix}.norm1.bias"],
        norm2_weight=state_dict[f"{layer_prefix}.norm2.weight"],
        norm2_bias=state_dict[f"{layer_prefix}.norm2.bias"],
    )


def extract_semantic_output_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    llm_prefix: str = "llm",
    decoder_prefix: str = "llm_decoder",
    speech_embedding_prefix: str = "speech_embedding",
) -> CosyVoiceSemanticOutputParameters:
    llm_prefix = llm_prefix.rstrip(".")
    decoder_prefix = decoder_prefix.rstrip(".")
    speech_embedding_prefix = speech_embedding_prefix.rstrip(".")
    return CosyVoiceSemanticOutputParameters(
        after_norm_weight=state_dict[f"{llm_prefix}.after_norm.weight"],
        after_norm_bias=state_dict[f"{llm_prefix}.after_norm.bias"],
        decoder_weight=state_dict[f"{decoder_prefix}.weight"],
        decoder_bias=state_dict[f"{decoder_prefix}.bias"],
        speech_embedding_weight=state_dict[f"{speech_embedding_prefix}.weight"],
    )


def apply_legacy_embed_torch(
    hidden_states: torch.Tensor,
    parameters: CosyVoiceLegacyEmbedParameters,
) -> torch.Tensor:
    hidden_states = F.linear(hidden_states, parameters.linear_weight, parameters.linear_bias)
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm_weight,
        parameters.norm_bias,
        parameters.epsilon,
    )
    return torch.relu(hidden_states)


def preprocess_tt_legacy_embed_parameters(parameters: CosyVoiceLegacyEmbedParameters, *, dtype):
    from ttnn.model_preprocessing import (  # noqa: PLC0415
        preprocess_layernorm_parameter,
        preprocess_linear_bias,
        preprocess_linear_weight,
    )

    import ttnn  # noqa: PLC0415

    return {
        "linear_weight": preprocess_linear_weight(parameters.linear_weight, dtype=dtype),
        "linear_bias": preprocess_linear_bias(parameters.linear_bias, dtype=dtype),
        "norm_weight": preprocess_layernorm_parameter(parameters.norm_weight, dtype=dtype),
        "norm_bias": preprocess_layernorm_parameter(parameters.norm_bias, dtype=dtype),
        "epsilon": parameters.epsilon,
    }


def preprocess_tt_semantic_layer_parameters(parameters: CosyVoiceSemanticLayerParameters, *, dtype):
    from ttnn.model_preprocessing import (  # noqa: PLC0415
        preprocess_layernorm_parameter,
        preprocess_linear_bias,
        preprocess_linear_weight,
    )

    import ttnn  # noqa: PLC0415

    return {
        "q_weight": preprocess_linear_weight(parameters.q_weight, dtype=dtype),
        "q_bias": preprocess_linear_bias(parameters.q_bias, dtype=dtype),
        "k_weight": preprocess_linear_weight(parameters.k_weight, dtype=dtype),
        "k_bias": preprocess_linear_bias(parameters.k_bias, dtype=dtype),
        "v_weight": preprocess_linear_weight(parameters.v_weight, dtype=dtype),
        "v_bias": preprocess_linear_bias(parameters.v_bias, dtype=dtype),
        "out_weight": preprocess_linear_weight(parameters.out_weight, dtype=dtype),
        "out_bias": preprocess_linear_bias(parameters.out_bias, dtype=dtype),
        "pos_weight": preprocess_linear_weight(parameters.pos_weight, dtype=dtype),
        "pos_bias_u": ttnn.from_torch(
            parameters.pos_bias_u.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        ),
        "pos_bias_v": ttnn.from_torch(
            parameters.pos_bias_v.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        ),
        "ffn_w1_weight": preprocess_linear_weight(parameters.ffn_w1_weight, dtype=dtype),
        "ffn_w1_bias": preprocess_linear_bias(parameters.ffn_w1_bias, dtype=dtype),
        "ffn_w2_weight": preprocess_linear_weight(parameters.ffn_w2_weight, dtype=dtype),
        "ffn_w2_bias": preprocess_linear_bias(parameters.ffn_w2_bias, dtype=dtype),
        "norm1_weight": preprocess_layernorm_parameter(parameters.norm1_weight, dtype=dtype),
        "norm1_bias": preprocess_layernorm_parameter(parameters.norm1_bias, dtype=dtype),
        "norm2_weight": preprocess_layernorm_parameter(parameters.norm2_weight, dtype=dtype),
        "norm2_bias": preprocess_layernorm_parameter(parameters.norm2_bias, dtype=dtype),
        "norm_epsilon": 1e-12,
    }


def preprocess_tt_semantic_output_parameters(parameters: CosyVoiceSemanticOutputParameters, *, dtype):
    from ttnn.model_preprocessing import (  # noqa: PLC0415
        preprocess_layernorm_parameter,
        preprocess_linear_bias,
        preprocess_linear_weight,
    )

    import ttnn  # noqa: PLC0415

    return {
        "after_norm_weight": preprocess_layernorm_parameter(parameters.after_norm_weight, dtype=dtype),
        "after_norm_bias": preprocess_layernorm_parameter(parameters.after_norm_bias, dtype=dtype),
        "decoder_weight": preprocess_linear_weight(parameters.decoder_weight, dtype=dtype),
        "decoder_bias": preprocess_linear_bias(parameters.decoder_bias, dtype=dtype),
        "speech_embedding_weight": parameters.speech_embedding_weight,
        "after_norm_epsilon": parameters.after_norm_epsilon,
    }


def apply_legacy_embed_ttnn(hidden_states, parameters, *, memory_config=None, dtype=None, core_grid=None):
    import ttnn  # noqa: PLC0415

    hidden_states = ttnn.linear(
        hidden_states,
        parameters["linear_weight"],
        bias=parameters["linear_bias"],
        memory_config=memory_config or ttnn.L1_MEMORY_CONFIG,
        dtype=dtype or ttnn.bfloat16,
        core_grid=core_grid,
    )
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters["norm_weight"],
        bias=parameters["norm_bias"],
        epsilon=parameters["epsilon"],
        memory_config=memory_config or ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    return ttnn.relu(hidden_states, memory_config=memory_config or ttnn.L1_MEMORY_CONFIG)


def apply_semantic_layer_torch(
    hidden_states: torch.Tensor,
    positional_embedding: torch.Tensor,
    parameters: CosyVoiceSemanticLayerParameters,
    *,
    num_heads: int = 16,
    cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    head_dim = hidden_states.shape[-1] // num_heads

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm1_weight,
        parameters.norm1_bias,
        1e-12,
    )
    q = F.linear(hidden_states, parameters.q_weight, parameters.q_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, parameters.k_weight, parameters.k_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, parameters.v_weight, parameters.v_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)

    pos = F.linear(positional_embedding, parameters.pos_weight, None).view(1, -1, num_heads, head_dim).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_with_bias_u = (q_t + parameters.pos_bias_u).transpose(1, 2)
    q_with_bias_v = (q_t + parameters.pos_bias_v).transpose(1, 2)

    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_with_bias_v, pos.transpose(-2, -1))
    if matrix_ac.shape != matrix_bd.shape:
        matrix_bd = _rel_shift_torch(matrix_bd)
    scores = (matrix_ac + matrix_bd) / math.sqrt(head_dim)
    if causal:
        causal_mask = torch.triu(torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1.0e4)
    attention = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(1, -1, num_heads * head_dim)
    attention_output = F.linear(attention_output, parameters.out_weight, parameters.out_bias)
    hidden_states = residual + attention_output

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm2_weight,
        parameters.norm2_bias,
        1e-12,
    )
    hidden_states = F.linear(hidden_states, parameters.ffn_w1_weight, parameters.ffn_w1_bias)
    hidden_states = torch.relu(hidden_states)
    hidden_states = F.linear(hidden_states, parameters.ffn_w2_weight, parameters.ffn_w2_bias)
    return residual + hidden_states, new_cache


def _rel_shift_torch(x: torch.Tensor) -> torch.Tensor:
    zero_pad = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
    shifted = x_padded[:, :, 1:].view_as(x)
    return shifted[:, :, :, : x.size(-1) // 2 + 1]


def assemble_prefill_embeddings(
    sos_embedding: torch.Tensor,
    speaker_embedding: torch.Tensor,
    encoded_text: torch.Tensor,
    task_embedding: torch.Tensor,
    prompt_speech_embeddings: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [sos_embedding, speaker_embedding, encoded_text, task_embedding, prompt_speech_embeddings],
        dim=1,
    )


def _empty_token_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 0), dtype=torch.int32, device=reference.device)


def _empty_length_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1,), dtype=torch.int32, device=reference.device)


def _project_optional_speaker_embedding(
    llm_module: Any, embedding: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if embedding.numel() == 0 or embedding.shape[0] == 0:
        return torch.zeros((1, 0, llm_module.llm_input_size), dtype=reference.dtype, device=reference.device)
    projected = llm_module.spk_embed_affine_layer(F.normalize(embedding, dim=1))
    return projected.unsqueeze(dim=1).to(dtype=reference.dtype)


def _embed_optional_prompt_speech(
    llm_module: Any, prompt_speech_token: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if prompt_speech_token.numel() == 0 or prompt_speech_token.shape[1] == 0:
        return torch.zeros((1, 0, llm_module.llm_input_size), dtype=reference.dtype, device=reference.device)
    return llm_module.speech_embedding(prompt_speech_token).to(dtype=reference.dtype)


def build_semantic_inputs(
    *,
    llm_module: Any,
    text: torch.Tensor,
    text_len: torch.Tensor,
    prompt_text: torch.Tensor,
    prompt_text_len: torch.Tensor,
    prompt_speech_token: torch.Tensor,
    prompt_speech_token_len: torch.Tensor,
    embedding: torch.Tensor,
    min_token_text_ratio: float = 2.0,
    max_token_text_ratio: float = 20.0,
) -> CosyVoiceSemanticInputs:
    merged_text = torch.concat([prompt_text, text], dim=1)
    merged_text_len = text_len + prompt_text_len
    text_embeddings = llm_module.text_embedding(merged_text)
    encoded_text, encoded_text_len = llm_module.encode(text_embeddings, merged_text_len)

    speaker_projection = _project_optional_speaker_embedding(llm_module, embedding, encoded_text)
    sos_embedding = llm_module.llm_embedding.weight[llm_module.sos].reshape(1, 1, -1).to(dtype=encoded_text.dtype)
    task_embedding = llm_module.llm_embedding.weight[llm_module.task_id].reshape(1, 1, -1).to(dtype=encoded_text.dtype)
    prompt_speech_embeddings = _embed_optional_prompt_speech(llm_module, prompt_speech_token, encoded_text)

    lm_input = assemble_prefill_embeddings(
        sos_embedding=sos_embedding,
        speaker_embedding=speaker_projection,
        encoded_text=encoded_text,
        task_embedding=task_embedding,
        prompt_speech_embeddings=prompt_speech_embeddings,
    )
    lm_input_mask = torch.ones((1, 1, lm_input.shape[1]), dtype=torch.bool, device=lm_input.device)
    lm_input_embed, lm_input_positional_embedding, _ = llm_module.llm.embed(lm_input, lm_input_mask)
    min_decode_length, max_decode_length = compute_decode_length_bounds(
        text_token_len=int(encoded_text_len.item()),
        prompt_text_token_len=int(prompt_text_len.item()),
        min_token_text_ratio=min_token_text_ratio,
        max_token_text_ratio=max_token_text_ratio,
    )
    return CosyVoiceSemanticInputs(
        text=text,
        text_len=text_len,
        prompt_text=prompt_text,
        prompt_text_len=prompt_text_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        llm_embedding=embedding,
        merged_text=merged_text,
        merged_text_len=merged_text_len,
        encoded_text=encoded_text,
        encoded_text_len=encoded_text_len,
        speaker_projection=speaker_projection,
        sos_embedding=sos_embedding,
        task_embedding=task_embedding,
        prompt_speech_embeddings=prompt_speech_embeddings,
        lm_input=lm_input,
        lm_input_embed=lm_input_embed,
        lm_input_positional_embedding=lm_input_positional_embedding,
        lm_input_mask=lm_input_mask,
        min_decode_length=min_decode_length,
        max_decode_length=max_decode_length,
    )


def build_semantic_inputs_from_model_input(
    *,
    llm_module: Any,
    model_input: dict[str, Any],
    min_token_text_ratio: float = 2.0,
    max_token_text_ratio: float = 20.0,
) -> CosyVoiceSemanticInputs:
    text = model_input["text"]
    prompt_text = model_input.get("prompt_text")
    if prompt_text is None:
        prompt_text = _empty_token_tensor(text)
    prompt_text_len = model_input.get("prompt_text_len")
    if prompt_text_len is None:
        prompt_text_len = _empty_length_tensor(text)
    prompt_speech_token = model_input.get("llm_prompt_speech_token")
    if prompt_speech_token is None:
        prompt_speech_token = _empty_token_tensor(text)
    prompt_speech_token_len = model_input.get("llm_prompt_speech_token_len")
    if prompt_speech_token_len is None:
        prompt_speech_token_len = _empty_length_tensor(text)
    embedding = model_input.get("llm_embedding")
    if embedding is None:
        embedding = torch.zeros(
            (0, llm_module.spk_embed_affine_layer.in_features),
            dtype=torch.float32,
            device=text.device,
        )
    return build_semantic_inputs(
        llm_module=llm_module,
        text=text,
        text_len=model_input["text_len"],
        prompt_text=prompt_text,
        prompt_text_len=prompt_text_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        embedding=embedding,
        min_token_text_ratio=min_token_text_ratio,
        max_token_text_ratio=max_token_text_ratio,
    )


def prepare_tt_prefill_input(
    lm_input: torch.Tensor,
    mesh_device,
    *,
    dtype=None,
    memory_config=None,
):
    import ttnn  # noqa: PLC0415

    return ttnn.from_torch(
        lm_input,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def prepare_next_decode_input(
    speech_embedding_weight: torch.Tensor,
    token_id: int,
    mesh_device,
    *,
    dtype=None,
    memory_config=None,
):
    import ttnn  # noqa: PLC0415

    token_embedding = build_next_decode_embedding(speech_embedding_weight, token_id)
    return ttnn.from_torch(
        token_embedding,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


class CosyVoiceTTSemanticGenerator:
    def __init__(self, llm_module: Any, mesh_device, *, dtype=None, memory_config=None):
        import ttnn  # noqa: PLC0415

        self.ttnn = ttnn
        self.llm_module = llm_module
        self.mesh_device = mesh_device
        self.dtype = dtype or ttnn.bfloat16
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.num_heads = int(llm_module.llm.encoders[0].self_attn.h)
        self.head_dim = int(llm_module.llm.encoders[0].self_attn.d_k)
        self.hidden_dim = self.num_heads * self.head_dim
        self.xscale = math.sqrt(self.hidden_dim)

        state_dict = llm_module.state_dict()
        num_layers = len(llm_module.llm.encoders)
        self.layer_parameters = [
            self._move_parameters_to_device(
                preprocess_tt_semantic_layer_parameters(
                    extract_semantic_layer_parameters(state_dict, layer_num=layer_idx, prefix="llm.encoders"),
                    dtype=self.dtype,
                )
            )
            for layer_idx in range(num_layers)
        ]
        self.output_parameters = self._move_parameters_to_device(
            preprocess_tt_semantic_output_parameters(
                extract_semantic_output_parameters(state_dict),
                dtype=self.dtype,
            )
        )
        self._causal_bias_cache: dict[int, Any] = {}

    def _move_parameters_to_device(self, parameters: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for name, value in parameters.items():
            if name.endswith("epsilon"):
                moved[name] = value
            elif isinstance(value, self.ttnn.Tensor):
                moved[name] = self.ttnn.to_device(value, self.mesh_device, memory_config=self.memory_config)
            else:
                moved[name] = value
        return moved

    def _torch_hidden_to_tt(self, hidden_states: torch.Tensor):
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.unsqueeze(1)
        return self.ttnn.from_torch(
            hidden_states,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=self.ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

    def _get_causal_bias(self, seq_len: int):
        if seq_len not in self._causal_bias_cache:
            bias = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
            bias.masked_fill_(torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1), -1.0e4)
            self._causal_bias_cache[seq_len] = self._torch_hidden_to_tt(bias.squeeze(1))
        return self._causal_bias_cache[seq_len]

    def _reshape_positional_heads(self, positional_embedding, parameters):
        position_seq_len = positional_embedding.shape[2]
        positional_embedding = self.ttnn.linear(
            positional_embedding,
            parameters["pos_weight"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        positional_embedding = self.ttnn.reshape(
            positional_embedding, (1, position_seq_len, self.num_heads, self.head_dim)
        )
        return self.ttnn.transpose(positional_embedding, 1, 2)

    def _rel_shift(self, x):
        zero_pad = self.ttnn.zeros(
            (x.shape[0], x.shape[1], x.shape[2], 1),
            device=self.mesh_device,
            dtype=x.dtype,
            layout=self.ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        x_padded = self.ttnn.concat([zero_pad, x], dim=-1, memory_config=self.memory_config)
        x_padded = self.ttnn.reshape(x_padded, (x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2]))
        shifted = self.ttnn.slice(x_padded, (0, 0, 1, 0), (x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2]))
        shifted = self.ttnn.reshape(shifted, tuple(x.shape))
        return self.ttnn.slice(shifted, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2 + 1))

    def _apply_layer(self, hidden_states, positional_embedding, parameters, *, causal_bias=None, cache=None):
        residual = hidden_states
        hidden_states = self.ttnn.layer_norm(
            hidden_states,
            weight=parameters["norm1_weight"],
            bias=parameters["norm1_bias"],
            epsilon=parameters["norm_epsilon"],
            memory_config=self.memory_config,
        )
        q = self.ttnn.linear(
            hidden_states,
            parameters["q_weight"],
            bias=parameters["q_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        k = self.ttnn.linear(
            hidden_states,
            parameters["k_weight"],
            bias=parameters["k_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        v = self.ttnn.linear(
            hidden_states,
            parameters["v_weight"],
            bias=parameters["v_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        qkv = self.ttnn.concat([q, k, v], dim=-1, memory_config=self.memory_config)
        q_heads, k_heads, v_heads = self.ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            memory_config=self.memory_config,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )
        if cache is not None:
            k_heads = self.ttnn.concat([cache[0], k_heads], dim=2, memory_config=self.memory_config)
            v_heads = self.ttnn.concat([cache[1], v_heads], dim=2, memory_config=self.memory_config)
        positional_heads = self._reshape_positional_heads(positional_embedding, parameters)
        q_transposed = self.ttnn.transpose(q_heads, 1, 2)
        q_with_bias_u = self.ttnn.add(q_transposed, parameters["pos_bias_u"], memory_config=self.memory_config)
        q_with_bias_v = self.ttnn.add(q_transposed, parameters["pos_bias_v"], memory_config=self.memory_config)
        q_with_bias_u = self.ttnn.transpose(q_with_bias_u, 1, 2)
        q_with_bias_v = self.ttnn.transpose(q_with_bias_v, 1, 2)
        matrix_ac = self.ttnn.matmul(
            q_with_bias_u, self.ttnn.transpose(k_heads, -2, -1), memory_config=self.memory_config
        )
        matrix_bd = self.ttnn.matmul(
            q_with_bias_v, self.ttnn.transpose(positional_heads, -2, -1), memory_config=self.memory_config
        )
        if tuple(matrix_ac.shape) != tuple(matrix_bd.shape):
            matrix_bd = self._rel_shift(matrix_bd)
        scores = self.ttnn.mul(
            self.ttnn.add(matrix_ac, matrix_bd, memory_config=self.memory_config), 1.0 / math.sqrt(self.head_dim)
        )
        if causal_bias is not None:
            scores = self.ttnn.add(scores, causal_bias, memory_config=self.memory_config)
        attention = self.ttnn.softmax_in_place(scores)
        attention_output = self.ttnn.matmul(attention, v_heads, memory_config=self.memory_config)
        attention_output = self.ttnn.transpose(attention_output, 1, 2)
        attention_output = self.ttnn.reshape(attention_output, (1, 1, attention_output.shape[1], self.hidden_dim))
        attention_output = self.ttnn.linear(
            attention_output,
            parameters["out_weight"],
            bias=parameters["out_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        hidden_states = self.ttnn.add(residual, attention_output, memory_config=self.memory_config)

        residual = hidden_states
        hidden_states = self.ttnn.layer_norm(
            hidden_states,
            weight=parameters["norm2_weight"],
            bias=parameters["norm2_bias"],
            epsilon=parameters["norm_epsilon"],
            memory_config=self.memory_config,
        )
        hidden_states = self.ttnn.linear(
            hidden_states,
            parameters["ffn_w1_weight"],
            bias=parameters["ffn_w1_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        hidden_states = self.ttnn.relu(hidden_states, memory_config=self.memory_config)
        hidden_states = self.ttnn.linear(
            hidden_states,
            parameters["ffn_w2_weight"],
            bias=parameters["ffn_w2_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        return self.ttnn.add(residual, hidden_states, memory_config=self.memory_config), (k_heads, v_heads)

    def _decode_step(self, hidden_states: torch.Tensor, positional_embedding: torch.Tensor, caches):
        hidden_tt = self._torch_hidden_to_tt(hidden_states)
        pos_tt = self._torch_hidden_to_tt(positional_embedding)
        new_caches = []
        for layer_index, parameters in enumerate(self.layer_parameters):
            hidden_tt, new_cache = self._apply_layer(
                hidden_tt,
                pos_tt,
                parameters,
                causal_bias=None,
                cache=caches[layer_index],
            )
            new_caches.append(new_cache)
        hidden_tt = self.ttnn.layer_norm(
            hidden_tt,
            weight=self.output_parameters["after_norm_weight"],
            bias=self.output_parameters["after_norm_bias"],
            epsilon=self.output_parameters["after_norm_epsilon"],
            memory_config=self.memory_config,
        )
        logits_tt = self.ttnn.linear(
            hidden_tt,
            self.output_parameters["decoder_weight"],
            bias=self.output_parameters["decoder_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        logits = self.ttnn.to_torch(logits_tt).squeeze(0).squeeze(0).squeeze(0).float()
        return logits, new_caches

    def _prefill(self, semantic_inputs: CosyVoiceSemanticInputs):
        hidden_tt = self._torch_hidden_to_tt(semantic_inputs.lm_input_embed)
        pos_tt = self._torch_hidden_to_tt(semantic_inputs.lm_input_positional_embedding)
        causal_bias = self._get_causal_bias(semantic_inputs.prefill_seq_len)
        caches = []
        for parameters in self.layer_parameters:
            hidden_tt, cache = self._apply_layer(hidden_tt, pos_tt, parameters, causal_bias=causal_bias, cache=None)
            caches.append(cache)
        hidden_tt = self.ttnn.layer_norm(
            hidden_tt,
            weight=self.output_parameters["after_norm_weight"],
            bias=self.output_parameters["after_norm_bias"],
            epsilon=self.output_parameters["after_norm_epsilon"],
            memory_config=self.memory_config,
        )
        logits_tt = self.ttnn.linear(
            hidden_tt,
            self.output_parameters["decoder_weight"],
            bias=self.output_parameters["decoder_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        logits = self.ttnn.to_torch(logits_tt).squeeze(0).squeeze(0)[-1].float()
        return logits, caches

    def _prepare_next_embeds(self, token_id: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        next_embedding = build_next_decode_embedding(self.output_parameters["speech_embedding_weight"], token_id)
        next_mask = torch.ones((1, 1, next_embedding.shape[1]), dtype=torch.bool, device=next_embedding.device)
        hidden_states, positional_embedding, _ = self.llm_module.llm.embed(next_embedding, next_mask, offset)
        return hidden_states, positional_embedding

    def generate(
        self,
        semantic_inputs: CosyVoiceSemanticInputs,
        *,
        sampling: int = 25,
    ) -> torch.Tensor:
        start = time.perf_counter()
        logits, caches = self._prefill(semantic_inputs)
        tokens: list[int] = []
        for step_index in range(semantic_inputs.max_decode_length):
            ignore_eos = step_index < semantic_inputs.min_decode_length
            next_token = int(self.llm_module.sampling_ids(logits.clone(), tokens, sampling, ignore_eos=ignore_eos))
            if next_token == self.llm_module.eos_token:
                break
            tokens.append(next_token)
            next_hidden, next_pos = self._prepare_next_embeds(next_token, semantic_inputs.prefill_seq_len + step_index)
            logits, caches = self._decode_step(next_hidden, next_pos, caches)
        wall_seconds = time.perf_counter() - start
        token_tensor = (
            torch.tensor(tokens, dtype=torch.int32).unsqueeze(0) if tokens else torch.zeros((1, 0), dtype=torch.int32)
        )
        return token_tensor, wall_seconds

    def teacher_forced_accuracy(
        self,
        semantic_inputs: CosyVoiceSemanticInputs,
        reference_tokens: torch.Tensor,
    ) -> float:
        if reference_tokens.numel() == 0:
            return 100.0
        logits, caches = self._prefill(semantic_inputs)
        predictions = [int(torch.argmax(logits).item())]
        for step_index in range(1, reference_tokens.shape[1]):
            next_hidden, next_pos = self._prepare_next_embeds(
                int(reference_tokens[0, step_index - 1].item()),
                semantic_inputs.prefill_seq_len + step_index - 1,
            )
            logits, caches = self._decode_step(next_hidden, next_pos, caches)
            predictions.append(int(torch.argmax(logits).item()))
        predicted = torch.tensor(predictions, dtype=torch.int32)
        target = reference_tokens.reshape(-1).to(torch.int32)
        return float((predicted == target).float().mean().item() * 100.0)
