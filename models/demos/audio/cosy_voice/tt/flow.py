from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CosyVoiceFlowConfig:
    input_size: int = 512
    output_size: int = 80
    vocab_size: int = 4096
    input_frame_rate: int = 50
    sample_rate: int = 22050
    hop_length: int = 256


@dataclass(frozen=True)
class CosyVoiceFlowInputs:
    source_speech_token: torch.Tensor
    source_speech_token_len: torch.Tensor
    prompt_speech_token: torch.Tensor
    prompt_speech_token_len: torch.Tensor
    prompt_speech_feat: torch.Tensor
    prompt_speech_feat_len: torch.Tensor
    flow_embedding: torch.Tensor
    normalized_embedding: torch.Tensor
    speaker_projection: torch.Tensor
    full_token: torch.Tensor
    full_token_len: torch.Tensor
    token_embedding: torch.Tensor
    decode_mel_length: int
    prompt_mel_length: int
    condition: torch.Tensor


@dataclass(frozen=True)
class CosyVoiceFlowBridgeParameters:
    input_embedding_weight: torch.Tensor
    speaker_weight: torch.Tensor
    speaker_bias: torch.Tensor
    encoder_proj_weight: torch.Tensor
    encoder_proj_bias: torch.Tensor


@dataclass(frozen=True)
class CosyVoiceFlowEncoderLayerParameters:
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
    norm_mha_weight: torch.Tensor
    norm_mha_bias: torch.Tensor
    norm_ff_weight: torch.Tensor
    norm_ff_bias: torch.Tensor


@dataclass(frozen=True)
class CosyVoiceFlowEncoderOutputParameters:
    after_norm_weight: torch.Tensor
    after_norm_bias: torch.Tensor
    epsilon: float = 1e-5


@dataclass(frozen=True)
class CosyVoiceFlowLengthRegulatorLayerParameters:
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    kernel_size: int
    padding: int


@dataclass(frozen=True)
class CosyVoiceFlowLengthRegulatorParameters:
    hidden_layers: tuple[CosyVoiceFlowLengthRegulatorLayerParameters, ...]
    output_weight: torch.Tensor
    output_bias: torch.Tensor
    num_groups: int = 1
    epsilon: float = 1e-5


def _empty_token_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 0), dtype=torch.int32, device=reference.device)


def _empty_length_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1,), dtype=torch.int32, device=reference.device)


def _empty_prompt_feature_tensor(reference: torch.Tensor, output_size: int) -> torch.Tensor:
    return torch.zeros((1, 0, output_size), dtype=torch.float32, device=reference.device)


def _default_flow_embedding(flow_module: Any, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (1, flow_module.spk_embed_affine_layer.in_features), dtype=torch.float32, device=reference.device
    )


def compute_decode_mel_length(
    token_count: int, *, input_frame_rate: int = 50, sample_rate: int = 22050, hop_length: int = 256
) -> int:
    return int(token_count / input_frame_rate * sample_rate / hop_length)


def build_prompt_plus_decode_tokens(prompt_token: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    prompt_len = prompt_token.shape[1]
    decode_len = token.shape[1]
    return torch.cat([prompt_token, token], dim=1), prompt_len, decode_len


def build_flow_condition(prompt_feat: torch.Tensor, decode_mel_length: int) -> torch.Tensor:
    condition = torch.zeros(
        (prompt_feat.shape[0], prompt_feat.shape[1] + decode_mel_length, prompt_feat.shape[2]),
        dtype=prompt_feat.dtype,
        device=prompt_feat.device,
    )
    condition[:, : prompt_feat.shape[1]] = prompt_feat
    return condition


def apply_flow_token_embedding_torch(
    token: torch.Tensor,
    token_len: torch.Tensor,
    parameters: CosyVoiceFlowBridgeParameters,
) -> torch.Tensor:
    mask = (torch.arange(token.shape[1], device=token.device).unsqueeze(0) < token_len.unsqueeze(1)).unsqueeze(-1)
    token = torch.clamp(token, min=0)
    return F.embedding(token, parameters.input_embedding_weight) * mask.to(parameters.input_embedding_weight.dtype)


def apply_flow_speaker_projection_torch(
    embedding: torch.Tensor,
    parameters: CosyVoiceFlowBridgeParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(embedding, dim=1)
    projected = F.linear(normalized, parameters.speaker_weight, parameters.speaker_bias)
    return normalized, projected


def apply_flow_encoder_projection_torch(
    hidden_states: torch.Tensor,
    parameters: CosyVoiceFlowBridgeParameters,
) -> torch.Tensor:
    return F.linear(hidden_states, parameters.encoder_proj_weight, parameters.encoder_proj_bias)


def extract_flow_bridge_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    input_embedding_prefix: str = "input_embedding",
    speaker_prefix: str = "spk_embed_affine_layer",
    encoder_proj_prefix: str = "encoder_proj",
) -> CosyVoiceFlowBridgeParameters:
    input_embedding_prefix = input_embedding_prefix.rstrip(".")
    speaker_prefix = speaker_prefix.rstrip(".")
    encoder_proj_prefix = encoder_proj_prefix.rstrip(".")
    return CosyVoiceFlowBridgeParameters(
        input_embedding_weight=state_dict[f"{input_embedding_prefix}.weight"],
        speaker_weight=state_dict[f"{speaker_prefix}.weight"],
        speaker_bias=state_dict[f"{speaker_prefix}.bias"],
        encoder_proj_weight=state_dict[f"{encoder_proj_prefix}.weight"],
        encoder_proj_bias=state_dict[f"{encoder_proj_prefix}.bias"],
    )


def extract_flow_encoder_layer_parameters(
    state_dict: dict[str, torch.Tensor],
    layer_num: int,
    *,
    prefix: str = "encoder.encoders",
) -> CosyVoiceFlowEncoderLayerParameters:
    layer_prefix = f"{prefix.rstrip('.')}.{layer_num}"
    return CosyVoiceFlowEncoderLayerParameters(
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
        norm_mha_weight=state_dict[f"{layer_prefix}.norm_mha.weight"],
        norm_mha_bias=state_dict[f"{layer_prefix}.norm_mha.bias"],
        norm_ff_weight=state_dict[f"{layer_prefix}.norm_ff.weight"],
        norm_ff_bias=state_dict[f"{layer_prefix}.norm_ff.bias"],
    )


def extract_flow_encoder_output_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "encoder.after_norm",
) -> CosyVoiceFlowEncoderOutputParameters:
    prefix = prefix.rstrip(".")
    return CosyVoiceFlowEncoderOutputParameters(
        after_norm_weight=state_dict[f"{prefix}.weight"],
        after_norm_bias=state_dict[f"{prefix}.bias"],
    )


def extract_flow_length_regulator_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "length_regulator.model",
) -> CosyVoiceFlowLengthRegulatorParameters:
    prefix = prefix.rstrip(".")
    hidden_layers = []
    for conv_index, norm_index in ((0, 1), (3, 4), (6, 7), (9, 10)):
        hidden_layers.append(
            CosyVoiceFlowLengthRegulatorLayerParameters(
                conv_weight=state_dict[f"{prefix}.{conv_index}.weight"],
                conv_bias=state_dict[f"{prefix}.{conv_index}.bias"],
                norm_weight=state_dict[f"{prefix}.{norm_index}.weight"],
                norm_bias=state_dict[f"{prefix}.{norm_index}.bias"],
                kernel_size=3,
                padding=1,
            )
        )
    return CosyVoiceFlowLengthRegulatorParameters(
        hidden_layers=tuple(hidden_layers),
        output_weight=state_dict[f"{prefix}.12.weight"],
        output_bias=state_dict[f"{prefix}.12.bias"],
    )


def build_flow_inputs(
    *,
    flow_module: Any,
    source_speech_token: torch.Tensor,
    source_speech_token_len: torch.Tensor,
    prompt_speech_token: torch.Tensor,
    prompt_speech_token_len: torch.Tensor,
    prompt_speech_feat: torch.Tensor,
    prompt_speech_feat_len: torch.Tensor,
    embedding: torch.Tensor,
) -> CosyVoiceFlowInputs:
    parameters = extract_flow_bridge_parameters(flow_module.state_dict())
    full_token, _, decode_token_len = build_prompt_plus_decode_tokens(prompt_speech_token, source_speech_token)
    full_token_len = prompt_speech_token_len + source_speech_token_len
    normalized_embedding, speaker_projection = apply_flow_speaker_projection_torch(embedding, parameters)
    token_embedding = apply_flow_token_embedding_torch(full_token, full_token_len, parameters)
    decode_mel_length = compute_decode_mel_length(
        decode_token_len,
        input_frame_rate=flow_module.input_frame_rate,
    )
    condition = build_flow_condition(prompt_speech_feat, decode_mel_length)
    return CosyVoiceFlowInputs(
        source_speech_token=source_speech_token,
        source_speech_token_len=source_speech_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        prompt_speech_feat=prompt_speech_feat,
        prompt_speech_feat_len=prompt_speech_feat_len,
        flow_embedding=embedding,
        normalized_embedding=normalized_embedding,
        speaker_projection=speaker_projection,
        full_token=full_token,
        full_token_len=full_token_len,
        token_embedding=token_embedding,
        decode_mel_length=decode_mel_length,
        prompt_mel_length=int(prompt_speech_feat.shape[1]),
        condition=condition,
    )


def build_flow_inputs_from_model_input(
    *,
    flow_module: Any,
    model_input: dict[str, Any],
    source_speech_token: torch.Tensor,
) -> CosyVoiceFlowInputs:
    prompt_speech_token = model_input.get("flow_prompt_speech_token")
    if prompt_speech_token is None:
        prompt_speech_token = _empty_token_tensor(source_speech_token)
    prompt_speech_token_len = model_input.get("flow_prompt_speech_token_len")
    if prompt_speech_token_len is None:
        prompt_speech_token_len = _empty_length_tensor(source_speech_token)
    prompt_speech_feat = model_input.get("prompt_speech_feat")
    if prompt_speech_feat is None:
        prompt_speech_feat = _empty_prompt_feature_tensor(source_speech_token, flow_module.output_size)
    prompt_speech_feat_len = model_input.get("prompt_speech_feat_len")
    if prompt_speech_feat_len is None:
        prompt_speech_feat_len = _empty_length_tensor(source_speech_token)
    embedding = model_input.get("flow_embedding")
    if embedding is None:
        embedding = _default_flow_embedding(flow_module, source_speech_token)
    return build_flow_inputs(
        flow_module=flow_module,
        source_speech_token=source_speech_token,
        source_speech_token_len=torch.tensor(
            [source_speech_token.shape[1]], dtype=torch.int32, device=source_speech_token.device
        ),
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        prompt_speech_feat=prompt_speech_feat,
        prompt_speech_feat_len=prompt_speech_feat_len,
        embedding=embedding,
    )


def prepare_tt_flow_condition(condition: torch.Tensor, mesh_device, *, dtype=None, memory_config=None):
    import ttnn  # noqa: PLC0415

    return ttnn.from_torch(
        condition,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def preprocess_tt_flow_bridge_parameters(parameters: CosyVoiceFlowBridgeParameters, *, dtype):
    from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight  # noqa: PLC0415

    import ttnn  # noqa: PLC0415

    return {
        "input_embedding_weight": ttnn.from_torch(
            parameters.input_embedding_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "speaker_weight": preprocess_linear_weight(parameters.speaker_weight, dtype=dtype),
        "speaker_bias": preprocess_linear_bias(parameters.speaker_bias, dtype=dtype),
        "encoder_proj_weight": preprocess_linear_weight(parameters.encoder_proj_weight, dtype=dtype),
        "encoder_proj_bias": preprocess_linear_bias(parameters.encoder_proj_bias, dtype=dtype),
    }


def preprocess_tt_flow_encoder_layer_parameters(parameters: CosyVoiceFlowEncoderLayerParameters, *, dtype):
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
        "norm_mha_weight": preprocess_layernorm_parameter(parameters.norm_mha_weight, dtype=dtype),
        "norm_mha_bias": preprocess_layernorm_parameter(parameters.norm_mha_bias, dtype=dtype),
        "norm_ff_weight": preprocess_layernorm_parameter(parameters.norm_ff_weight, dtype=dtype),
        "norm_ff_bias": preprocess_layernorm_parameter(parameters.norm_ff_bias, dtype=dtype),
        "norm_epsilon": 1e-12,
    }


def preprocess_tt_flow_encoder_output_parameters(parameters: CosyVoiceFlowEncoderOutputParameters, *, dtype):
    from ttnn.model_preprocessing import preprocess_layernorm_parameter  # noqa: PLC0415

    return {
        "after_norm_weight": preprocess_layernorm_parameter(parameters.after_norm_weight, dtype=dtype),
        "after_norm_bias": preprocess_layernorm_parameter(parameters.after_norm_bias, dtype=dtype),
        "after_norm_epsilon": parameters.epsilon,
    }


def preprocess_tt_flow_length_regulator_parameters(parameters: CosyVoiceFlowLengthRegulatorParameters, *, dtype):
    from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight  # noqa: PLC0415

    import ttnn  # noqa: PLC0415

    hidden_layers = []
    for layer in parameters.hidden_layers:
        hidden_layers.append(
            {
                "conv_weight": preprocess_linear_weight(
                    layer.conv_weight.reshape(layer.conv_weight.shape[0], -1),
                    dtype=dtype,
                ),
                "conv_bias": preprocess_linear_bias(layer.conv_bias, dtype=dtype),
                "norm_weight": ttnn.from_torch(
                    ttnn.create_group_norm_weight_bias_rm(
                        layer.norm_weight,
                        num_channels=layer.conv_weight.shape[0],
                        num_cores_x=1,
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "norm_bias": ttnn.from_torch(
                    ttnn.create_group_norm_weight_bias_rm(
                        layer.norm_bias,
                        num_channels=layer.conv_weight.shape[0],
                        num_cores_x=1,
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "kernel_size": layer.kernel_size,
                "padding": layer.padding,
            }
        )
    return {
        "hidden_layers": tuple(hidden_layers),
        "output_weight": preprocess_linear_weight(
            parameters.output_weight.reshape(parameters.output_weight.shape[0], -1),
            dtype=dtype,
        ),
        "output_bias": preprocess_linear_bias(parameters.output_bias, dtype=dtype),
        "num_groups": parameters.num_groups,
        "epsilon": parameters.epsilon,
    }


def interpolate_length_regulator_hidden_states(
    prompt_hidden_states: torch.Tensor,
    decode_hidden_states: torch.Tensor,
    prompt_mel_length: int,
    decode_mel_length: int,
    *,
    input_frame_rate: int = 50,
    sample_rate: int = 22050,
    hop_length: int = 256,
) -> torch.Tensor:
    segment_mel_length = int(20 / input_frame_rate * sample_rate / hop_length)
    if decode_hidden_states.shape[1] > 40:
        decode_head = F.interpolate(
            decode_hidden_states[:, :20].transpose(1, 2).contiguous(),
            size=segment_mel_length,
            mode="linear",
        )
        decode_mid = F.interpolate(
            decode_hidden_states[:, 20:-20].transpose(1, 2).contiguous(),
            size=decode_mel_length - segment_mel_length * 2,
            mode="linear",
        )
        decode_tail = F.interpolate(
            decode_hidden_states[:, -20:].transpose(1, 2).contiguous(),
            size=segment_mel_length,
            mode="linear",
        )
        decode_hidden_states = torch.cat([decode_head, decode_mid, decode_tail], dim=2)
    else:
        decode_hidden_states = F.interpolate(
            decode_hidden_states.transpose(1, 2).contiguous(),
            size=decode_mel_length,
            mode="linear",
        )
    if prompt_hidden_states.shape[1] != 0:
        prompt_hidden_states = F.interpolate(
            prompt_hidden_states.transpose(1, 2).contiguous(),
            size=prompt_mel_length,
            mode="linear",
        )
        return torch.cat([prompt_hidden_states, decode_hidden_states], dim=2)
    return decode_hidden_states


def apply_flow_length_regulator_torch(
    prompt_hidden_states: torch.Tensor,
    decode_hidden_states: torch.Tensor,
    prompt_mel_length: int,
    decode_mel_length: int,
    parameters: CosyVoiceFlowLengthRegulatorParameters,
    *,
    input_frame_rate: int = 50,
) -> tuple[torch.Tensor, int]:
    hidden_states = interpolate_length_regulator_hidden_states(
        prompt_hidden_states,
        decode_hidden_states,
        prompt_mel_length,
        decode_mel_length,
        input_frame_rate=input_frame_rate,
    )
    for layer in parameters.hidden_layers:
        hidden_states = F.conv1d(
            hidden_states,
            layer.conv_weight,
            layer.conv_bias,
            stride=1,
            padding=layer.padding,
        )
        hidden_states = F.group_norm(
            hidden_states,
            num_groups=parameters.num_groups,
            weight=layer.norm_weight,
            bias=layer.norm_bias,
            eps=parameters.epsilon,
        )
        hidden_states = F.mish(hidden_states)
    hidden_states = F.conv1d(hidden_states, parameters.output_weight, parameters.output_bias, stride=1, padding=0)
    return hidden_states.transpose(1, 2).contiguous(), prompt_mel_length + decode_mel_length


def apply_flow_encoder_layer_torch(
    hidden_states: torch.Tensor,
    positional_embedding: torch.Tensor,
    parameters: CosyVoiceFlowEncoderLayerParameters,
    *,
    num_heads: int = 8,
) -> torch.Tensor:
    head_dim = hidden_states.shape[-1] // num_heads

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm_mha_weight,
        parameters.norm_mha_bias,
        1e-12,
    )
    q = F.linear(hidden_states, parameters.q_weight, parameters.q_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, parameters.k_weight, parameters.k_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, parameters.v_weight, parameters.v_bias).view(1, -1, num_heads, head_dim).transpose(1, 2)

    pos = F.linear(positional_embedding, parameters.pos_weight, None).view(1, -1, num_heads, head_dim).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_with_bias_u = (q_t + parameters.pos_bias_u).transpose(1, 2)
    q_with_bias_v = (q_t + parameters.pos_bias_v).transpose(1, 2)

    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_with_bias_v, pos.transpose(-2, -1))
    if matrix_ac.shape != matrix_bd.shape:
        matrix_bd = _rel_shift_torch(matrix_bd)
    scores = (matrix_ac + matrix_bd) / torch.sqrt(torch.tensor(float(head_dim), device=hidden_states.device))
    attention = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(1, -1, num_heads * head_dim)
    attention_output = F.linear(attention_output, parameters.out_weight, parameters.out_bias)
    hidden_states = residual + attention_output

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm_ff_weight,
        parameters.norm_ff_bias,
        1e-12,
    )
    hidden_states = F.linear(hidden_states, parameters.ffn_w1_weight, parameters.ffn_w1_bias)
    hidden_states = F.silu(hidden_states)
    hidden_states = F.linear(hidden_states, parameters.ffn_w2_weight, parameters.ffn_w2_bias)
    return residual + hidden_states


def _rel_shift_torch(x: torch.Tensor) -> torch.Tensor:
    zero_pad = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
    shifted = x_padded[:, :, 1:].view_as(x)
    return shifted[:, :, :, : x.size(-1) // 2 + 1]


class CosyVoiceTTFlowBridge:
    def __init__(self, flow_module: Any, mesh_device, *, dtype=None, memory_config=None):
        import ttnn  # noqa: PLC0415

        self.ttnn = ttnn
        self.flow_module = flow_module
        self.mesh_device = mesh_device
        self.dtype = dtype or ttnn.bfloat16
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        parameters = preprocess_tt_flow_bridge_parameters(
            extract_flow_bridge_parameters(flow_module.state_dict()),
            dtype=self.dtype,
        )
        self.parameters = self._move_parameters_to_device(parameters)

    def _move_parameters_to_device(self, parameters: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for name, value in parameters.items():
            if isinstance(value, self.ttnn.Tensor):
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

    def embed_tokens(self, token: torch.Tensor, token_len: torch.Tensor) -> torch.Tensor:
        token_tt = self.ttnn.from_torch(
            token.reshape(1, 1, 1, token.shape[1]).to(torch.int32),
            device=self.mesh_device,
            dtype=self.ttnn.uint32,
            layout=self.ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
        )
        embedded = self.ttnn.embedding(
            token_tt,
            self.parameters["input_embedding_weight"],
            layout=self.ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        embedded = self.ttnn.reshape(embedded, (1, 1, token.shape[1], self.flow_module.input_size))
        if int(token_len.item()) != int(token.shape[1]):
            mask = (torch.arange(token.shape[1], device=token.device).unsqueeze(0) < token_len.unsqueeze(1)).unsqueeze(
                -1
            )
            mask_tt = self._torch_hidden_to_tt(mask.to(dtype=torch.float32))
            embedded = self.ttnn.mul(embedded, mask_tt, memory_config=self.memory_config)
        return self.ttnn.to_torch(embedded).squeeze(0).float()

    def project_speaker_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embedding, dim=1)
        projected = self.ttnn.linear(
            self._torch_hidden_to_tt(normalized),
            self.parameters["speaker_weight"],
            bias=self.parameters["speaker_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        return self.ttnn.to_torch(projected).squeeze(0).squeeze(0).float()

    def project_encoder_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.ttnn.linear(
            self._torch_hidden_to_tt(hidden_states),
            self.parameters["encoder_proj_weight"],
            bias=self.parameters["encoder_proj_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        return self.ttnn.to_torch(projected).squeeze(0).float()


class CosyVoiceTTFlowEncoder:
    def __init__(self, flow_module: Any, mesh_device, *, dtype=None, memory_config=None):
        import ttnn  # noqa: PLC0415

        self.ttnn = ttnn
        self.flow_module = flow_module
        self.mesh_device = mesh_device
        self.dtype = dtype or ttnn.bfloat16
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.num_heads = int(flow_module.encoder.encoders[0].self_attn.h)
        self.head_dim = int(flow_module.encoder.encoders[0].self_attn.d_k)
        self.hidden_dim = self.num_heads * self.head_dim
        state_dict = flow_module.state_dict()
        self.layer_parameters = [
            self._move_parameters_to_device(
                preprocess_tt_flow_encoder_layer_parameters(
                    extract_flow_encoder_layer_parameters(state_dict, layer_num=layer_idx),
                    dtype=self.dtype,
                )
            )
            for layer_idx in range(len(flow_module.encoder.encoders))
        ]
        self.output_parameters = self._move_parameters_to_device(
            preprocess_tt_flow_encoder_output_parameters(
                extract_flow_encoder_output_parameters(state_dict),
                dtype=self.dtype,
            )
        )

    def _move_parameters_to_device(self, parameters: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for name, value in parameters.items():
            if isinstance(value, self.ttnn.Tensor):
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

    def _apply_layer(self, hidden_states, positional_embedding, parameters):
        residual = hidden_states
        hidden_states = self.ttnn.layer_norm(
            hidden_states,
            weight=parameters["norm_mha_weight"],
            bias=parameters["norm_mha_bias"],
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
            self.ttnn.add(matrix_ac, matrix_bd, memory_config=self.memory_config),
            1.0 / torch.sqrt(torch.tensor(float(self.head_dim))).item(),
        )
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
            weight=parameters["norm_ff_weight"],
            bias=parameters["norm_ff_bias"],
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
        hidden_states = self.ttnn.silu(hidden_states, memory_config=self.memory_config)
        hidden_states = self.ttnn.linear(
            hidden_states,
            parameters["ffn_w2_weight"],
            bias=parameters["ffn_w2_bias"],
            memory_config=self.memory_config,
            dtype=self.dtype,
        )
        return self.ttnn.add(residual, hidden_states, memory_config=self.memory_config)

    def encode(self, token_embedding: torch.Tensor, token_len: torch.Tensor) -> torch.Tensor:
        masks = torch.ones(
            (token_embedding.shape[0], 1, token_embedding.shape[1]), dtype=torch.bool, device=token_embedding.device
        )
        embedded_hidden, positional_embedding, _ = self.flow_module.encoder.embed(token_embedding, masks)
        hidden_tt = self._torch_hidden_to_tt(embedded_hidden)
        pos_tt = self._torch_hidden_to_tt(positional_embedding)
        for parameters in self.layer_parameters:
            hidden_tt = self._apply_layer(hidden_tt, pos_tt, parameters)
        hidden_tt = self.ttnn.layer_norm(
            hidden_tt,
            weight=self.output_parameters["after_norm_weight"],
            bias=self.output_parameters["after_norm_bias"],
            epsilon=self.output_parameters["after_norm_epsilon"],
            memory_config=self.memory_config,
        )
        return self.ttnn.to_torch(hidden_tt).squeeze(0).float()


class CosyVoiceTTFlowLengthRegulator:
    def __init__(self, flow_module: Any, mesh_device, *, dtype=None, memory_config=None):
        import ttnn  # noqa: PLC0415

        self.ttnn = ttnn
        self.flow_module = flow_module
        self.mesh_device = mesh_device
        self.dtype = dtype or ttnn.bfloat16
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.input_frame_rate = int(flow_module.input_frame_rate)
        self.group_norm_core_grid = ttnn.CoreGrid(y=1, x=1)
        self.group_norm_input_mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(
                num_channel=flow_module.output_size,
                num_groups=1,
                num_cores_across_channel=1,
                data_type=ttnn.bfloat16,
            ),
            mesh_device,
        )
        self.parameters = self._move_parameters_to_device(
            preprocess_tt_flow_length_regulator_parameters(
                extract_flow_length_regulator_parameters(flow_module.state_dict()),
                dtype=self.dtype,
            )
        )

    def _move_parameters_to_device(self, parameters: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {
            "hidden_layers": [],
            "num_groups": parameters["num_groups"],
            "epsilon": parameters["epsilon"],
        }
        for layer in parameters["hidden_layers"]:
            moved_layer: dict[str, Any] = {}
            for name, value in layer.items():
                if isinstance(value, self.ttnn.Tensor):
                    moved_layer[name] = self.ttnn.to_device(value, self.mesh_device, memory_config=self.memory_config)
                else:
                    moved_layer[name] = value
            moved["hidden_layers"].append(moved_layer)
        moved["output_weight"] = self.ttnn.to_device(
            parameters["output_weight"], self.mesh_device, memory_config=self.memory_config
        )
        moved["output_bias"] = self.ttnn.to_device(
            parameters["output_bias"], self.mesh_device, memory_config=self.memory_config
        )
        return moved

    def _torch_nlc_to_tt(self, hidden_states: torch.Tensor):
        return self.ttnn.from_torch(
            hidden_states.unsqueeze(1).contiguous(),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=self.ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

    def _tt_to_torch_nlc(self, hidden_states) -> torch.Tensor:
        return self.ttnn.to_torch(hidden_states).squeeze(1).float()

    def _extract_temporal_patches(self, hidden_states: torch.Tensor, *, kernel_size: int, padding: int) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (padding, padding))
        hidden_states = hidden_states.unfold(dimension=2, size=kernel_size, step=1)
        return hidden_states.permute(0, 2, 1, 3).reshape(
            hidden_states.shape[0], -1, hidden_states.shape[1] * kernel_size
        )

    def _apply_linear_conv(self, hidden_states: torch.Tensor, *, weight, bias) -> torch.Tensor:
        return self.ttnn.linear(
            self._torch_nlc_to_tt(hidden_states),
            weight,
            bias=bias,
            memory_config=self.memory_config,
            dtype=self.dtype,
        )

    def inference(
        self,
        prompt_hidden_states: torch.Tensor,
        decode_hidden_states: torch.Tensor,
        prompt_mel_length: int,
        decode_mel_length: int,
        input_frame_rate: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        hidden_states = interpolate_length_regulator_hidden_states(
            prompt_hidden_states,
            decode_hidden_states,
            prompt_mel_length,
            decode_mel_length,
            input_frame_rate=input_frame_rate or self.input_frame_rate,
        )
        hidden_states = hidden_states.contiguous()
        for layer in self.parameters["hidden_layers"]:
            hidden_states_tt = self._apply_linear_conv(
                self._extract_temporal_patches(
                    hidden_states,
                    kernel_size=layer["kernel_size"],
                    padding=layer["padding"],
                ),
                weight=layer["conv_weight"],
                bias=layer["conv_bias"],
            )
            hidden_states_tt = self.ttnn.group_norm(
                hidden_states_tt,
                num_groups=self.parameters["num_groups"],
                input_mask=self.group_norm_input_mask,
                weight=layer["norm_weight"],
                bias=layer["norm_bias"],
                epsilon=self.parameters["epsilon"],
                memory_config=self.memory_config,
                core_grid=self.group_norm_core_grid,
                inplace=False,
                output_layout=self.ttnn.TILE_LAYOUT,
            )
            hidden_states_tt = self.ttnn.mish(hidden_states_tt, memory_config=self.memory_config)
            hidden_states = self._tt_to_torch_nlc(hidden_states_tt).transpose(1, 2).contiguous()
        hidden_states_tt = self._apply_linear_conv(
            hidden_states.transpose(1, 2).contiguous(),
            weight=self.parameters["output_weight"],
            bias=self.parameters["output_bias"],
        )
        return self._tt_to_torch_nlc(hidden_states_tt), prompt_mel_length + decode_mel_length


@dataclass(frozen=True)
class CosyVoiceFlowDecoderBlockParameters:
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    kernel_size: int
    padding: int
    num_groups: int = 8
    epsilon: float = 1e-5


@dataclass(frozen=True)
class CosyVoiceFlowDecoderResnetBlockParameters:
    time_mlp_weight: torch.Tensor
    time_mlp_bias: torch.Tensor
    block1: CosyVoiceFlowDecoderBlockParameters
    block2: CosyVoiceFlowDecoderBlockParameters
    res_conv_weight: torch.Tensor
    res_conv_bias: torch.Tensor


@dataclass(frozen=True)
class CosyVoiceFlowDecoderTransformerBlockParameters:
    norm1_weight: torch.Tensor
    norm1_bias: torch.Tensor
    attn_q_weight: torch.Tensor
    attn_k_weight: torch.Tensor
    attn_v_weight: torch.Tensor
    attn_out_weight: torch.Tensor
    attn_out_bias: torch.Tensor
    norm3_weight: torch.Tensor
    norm3_bias: torch.Tensor
    ff_proj_weight: torch.Tensor
    ff_proj_bias: torch.Tensor
    ff_activation: str
    ff_alpha: torch.Tensor | None
    ff_beta: torch.Tensor | None
    ff_out_weight: torch.Tensor
    ff_out_bias: torch.Tensor
    num_heads: int
    head_dim: int
    epsilon: float = 1e-5
    snake_alpha_logscale: bool = True


@dataclass(frozen=True)
class CosyVoiceFlowDecoderDownBlockParameters:
    resnet: CosyVoiceFlowDecoderResnetBlockParameters
    transformer_blocks: tuple[CosyVoiceFlowDecoderTransformerBlockParameters, ...]
    sample_weight: torch.Tensor
    sample_bias: torch.Tensor
    sample_stride: int
    sample_padding: int
    sample_transpose: bool = False


@dataclass(frozen=True)
class CosyVoiceFlowDecoderMidBlockParameters:
    resnet: CosyVoiceFlowDecoderResnetBlockParameters
    transformer_blocks: tuple[CosyVoiceFlowDecoderTransformerBlockParameters, ...]


@dataclass(frozen=True)
class CosyVoiceFlowDecoderUpBlockParameters:
    resnet: CosyVoiceFlowDecoderResnetBlockParameters
    transformer_blocks: tuple[CosyVoiceFlowDecoderTransformerBlockParameters, ...]
    sample_weight: torch.Tensor
    sample_bias: torch.Tensor
    sample_stride: int
    sample_padding: int
    sample_transpose: bool = False


@dataclass(frozen=True)
class CosyVoiceFlowDecoderParameters:
    time_linear_1_weight: torch.Tensor
    time_linear_1_bias: torch.Tensor
    time_linear_2_weight: torch.Tensor
    time_linear_2_bias: torch.Tensor
    down_blocks: tuple[CosyVoiceFlowDecoderDownBlockParameters, ...]
    mid_blocks: tuple[CosyVoiceFlowDecoderMidBlockParameters, ...]
    up_blocks: tuple[CosyVoiceFlowDecoderUpBlockParameters, ...]
    final_block: CosyVoiceFlowDecoderBlockParameters
    final_proj_weight: torch.Tensor
    final_proj_bias: torch.Tensor
    in_channels: int
    out_channels: int
    num_heads: int
    head_dim: int
    t_scheduler: str
    inference_cfg_rate: float
    time_scale: float = 1000.0


def _extract_flow_decoder_block_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    kernel_size: int,
    padding: int,
    num_groups: int = 8,
    epsilon: float = 1e-5,
) -> CosyVoiceFlowDecoderBlockParameters:
    prefix = prefix.rstrip(".")
    return CosyVoiceFlowDecoderBlockParameters(
        conv_weight=state_dict[f"{prefix}.block.0.weight"],
        conv_bias=state_dict[f"{prefix}.block.0.bias"],
        norm_weight=state_dict[f"{prefix}.block.1.weight"],
        norm_bias=state_dict[f"{prefix}.block.1.bias"],
        kernel_size=kernel_size,
        padding=padding,
        num_groups=num_groups,
        epsilon=epsilon,
    )


def _extract_flow_decoder_resnet_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    num_groups: int = 8,
    epsilon: float = 1e-5,
) -> CosyVoiceFlowDecoderResnetBlockParameters:
    prefix = prefix.rstrip(".")
    return CosyVoiceFlowDecoderResnetBlockParameters(
        time_mlp_weight=state_dict[f"{prefix}.mlp.1.weight"],
        time_mlp_bias=state_dict[f"{prefix}.mlp.1.bias"],
        block1=_extract_flow_decoder_block_parameters(
            state_dict,
            prefix=f"{prefix}.block1",
            kernel_size=3,
            padding=1,
            num_groups=num_groups,
            epsilon=epsilon,
        ),
        block2=_extract_flow_decoder_block_parameters(
            state_dict,
            prefix=f"{prefix}.block2",
            kernel_size=3,
            padding=1,
            num_groups=num_groups,
            epsilon=epsilon,
        ),
        res_conv_weight=state_dict[f"{prefix}.res_conv.weight"],
        res_conv_bias=state_dict[f"{prefix}.res_conv.bias"],
    )


def _extract_flow_decoder_transformer_block_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    num_heads: int,
    transformer_block_module: Any,
    epsilon: float = 1e-5,
) -> CosyVoiceFlowDecoderTransformerBlockParameters:
    prefix = prefix.rstrip(".")
    attn_q_weight = state_dict[f"{prefix}.attn1.to_q.weight"]
    inner_dim = int(attn_q_weight.shape[0])
    activation_module = transformer_block_module.ff.net[0]
    activation_name = type(activation_module).__name__.lower()
    ff_activation = "gelu"
    ff_alpha = None
    ff_beta = None
    if activation_name == "snakebeta":
        ff_activation = "snakebeta"
        ff_alpha = state_dict[f"{prefix}.ff.net.0.alpha"]
        ff_beta = state_dict[f"{prefix}.ff.net.0.beta"]
    elif activation_name == "geglu":
        ff_activation = "geglu"
    elif activation_name == "approximategelu":
        ff_activation = "gelu_approximate"
    return CosyVoiceFlowDecoderTransformerBlockParameters(
        norm1_weight=state_dict[f"{prefix}.norm1.weight"],
        norm1_bias=state_dict[f"{prefix}.norm1.bias"],
        attn_q_weight=attn_q_weight,
        attn_k_weight=state_dict[f"{prefix}.attn1.to_k.weight"],
        attn_v_weight=state_dict[f"{prefix}.attn1.to_v.weight"],
        attn_out_weight=state_dict[f"{prefix}.attn1.to_out.0.weight"],
        attn_out_bias=state_dict[f"{prefix}.attn1.to_out.0.bias"],
        norm3_weight=state_dict[f"{prefix}.norm3.weight"],
        norm3_bias=state_dict[f"{prefix}.norm3.bias"],
        ff_proj_weight=state_dict[f"{prefix}.ff.net.0.proj.weight"],
        ff_proj_bias=state_dict[f"{prefix}.ff.net.0.proj.bias"],
        ff_activation=ff_activation,
        ff_alpha=ff_alpha,
        ff_beta=ff_beta,
        ff_out_weight=state_dict[f"{prefix}.ff.net.2.weight"],
        ff_out_bias=state_dict[f"{prefix}.ff.net.2.bias"],
        num_heads=num_heads,
        head_dim=inner_dim // num_heads,
        epsilon=epsilon,
    )


def extract_flow_decoder_parameters(
    flow_decoder_module: Any,
    *,
    prefix: str = "estimator",
    num_groups: int = 8,
    epsilon: float = 1e-5,
) -> CosyVoiceFlowDecoderParameters:
    state_dict = flow_decoder_module.state_dict()
    estimator = flow_decoder_module.estimator
    prefix = prefix.rstrip(".")
    num_heads = int(estimator.down_blocks[0][1][0].attn1.heads)

    down_blocks: list[CosyVoiceFlowDecoderDownBlockParameters] = []
    for block_idx, (_, transformer_blocks, sample) in enumerate(estimator.down_blocks):
        block_prefix = f"{prefix}.down_blocks.{block_idx}"
        sample_prefix = f"{block_prefix}.2"
        if hasattr(sample, "conv"):
            sample_weight = state_dict[f"{sample_prefix}.conv.weight"]
            sample_bias = state_dict[f"{sample_prefix}.conv.bias"]
            sample_stride = int(sample.conv.stride[0])
            sample_padding = int(sample.conv.padding[0])
        else:
            sample_weight = state_dict[f"{sample_prefix}.weight"]
            sample_bias = state_dict[f"{sample_prefix}.bias"]
            sample_stride = int(sample.stride[0])
            sample_padding = int(sample.padding[0])
        down_blocks.append(
            CosyVoiceFlowDecoderDownBlockParameters(
                resnet=_extract_flow_decoder_resnet_parameters(
                    state_dict,
                    prefix=f"{block_prefix}.0",
                    num_groups=num_groups,
                    epsilon=epsilon,
                ),
                transformer_blocks=tuple(
                    _extract_flow_decoder_transformer_block_parameters(
                        state_dict,
                        prefix=f"{block_prefix}.1.{transformer_idx}",
                        num_heads=num_heads,
                        transformer_block_module=transformer_blocks[transformer_idx],
                        epsilon=epsilon,
                    )
                    for transformer_idx in range(len(transformer_blocks))
                ),
                sample_weight=sample_weight,
                sample_bias=sample_bias,
                sample_stride=sample_stride,
                sample_padding=sample_padding,
                sample_transpose=False,
            )
        )

    mid_blocks: list[CosyVoiceFlowDecoderMidBlockParameters] = []
    for block_idx, (_, transformer_blocks) in enumerate(estimator.mid_blocks):
        block_prefix = f"{prefix}.mid_blocks.{block_idx}"
        mid_blocks.append(
            CosyVoiceFlowDecoderMidBlockParameters(
                resnet=_extract_flow_decoder_resnet_parameters(
                    state_dict,
                    prefix=f"{block_prefix}.0",
                    num_groups=num_groups,
                    epsilon=epsilon,
                ),
                transformer_blocks=tuple(
                    _extract_flow_decoder_transformer_block_parameters(
                        state_dict,
                        prefix=f"{block_prefix}.1.{transformer_idx}",
                        num_heads=num_heads,
                        transformer_block_module=transformer_blocks[transformer_idx],
                        epsilon=epsilon,
                    )
                    for transformer_idx in range(len(transformer_blocks))
                ),
            )
        )

    up_blocks: list[CosyVoiceFlowDecoderUpBlockParameters] = []
    for block_idx, (_, transformer_blocks, sample) in enumerate(estimator.up_blocks):
        block_prefix = f"{prefix}.up_blocks.{block_idx}"
        sample_prefix = f"{block_prefix}.2"
        if hasattr(sample, "conv"):
            sample_weight = state_dict[f"{sample_prefix}.conv.weight"]
            sample_bias = state_dict[f"{sample_prefix}.conv.bias"]
            sample_stride = int(sample.conv.stride[0])
            sample_padding = int(sample.conv.padding[0])
            sample_transpose = True
        else:
            sample_weight = state_dict[f"{sample_prefix}.weight"]
            sample_bias = state_dict[f"{sample_prefix}.bias"]
            sample_stride = int(sample.stride[0])
            sample_padding = int(sample.padding[0])
            sample_transpose = False
        up_blocks.append(
            CosyVoiceFlowDecoderUpBlockParameters(
                resnet=_extract_flow_decoder_resnet_parameters(
                    state_dict,
                    prefix=f"{block_prefix}.0",
                    num_groups=num_groups,
                    epsilon=epsilon,
                ),
                transformer_blocks=tuple(
                    _extract_flow_decoder_transformer_block_parameters(
                        state_dict,
                        prefix=f"{block_prefix}.1.{transformer_idx}",
                        num_heads=num_heads,
                        transformer_block_module=transformer_blocks[transformer_idx],
                        epsilon=epsilon,
                    )
                    for transformer_idx in range(len(transformer_blocks))
                ),
                sample_weight=sample_weight,
                sample_bias=sample_bias,
                sample_stride=sample_stride,
                sample_padding=sample_padding,
                sample_transpose=sample_transpose,
            )
        )

    return CosyVoiceFlowDecoderParameters(
        time_linear_1_weight=state_dict[f"{prefix}.time_mlp.linear_1.weight"],
        time_linear_1_bias=state_dict[f"{prefix}.time_mlp.linear_1.bias"],
        time_linear_2_weight=state_dict[f"{prefix}.time_mlp.linear_2.weight"],
        time_linear_2_bias=state_dict[f"{prefix}.time_mlp.linear_2.bias"],
        down_blocks=tuple(down_blocks),
        mid_blocks=tuple(mid_blocks),
        up_blocks=tuple(up_blocks),
        final_block=_extract_flow_decoder_block_parameters(
            state_dict,
            prefix=f"{prefix}.final_block",
            kernel_size=3,
            padding=1,
            num_groups=num_groups,
            epsilon=epsilon,
        ),
        final_proj_weight=state_dict[f"{prefix}.final_proj.weight"],
        final_proj_bias=state_dict[f"{prefix}.final_proj.bias"],
        in_channels=int(estimator.in_channels),
        out_channels=int(estimator.out_channels),
        num_heads=num_heads,
        head_dim=int(estimator.down_blocks[0][1][0].attn1.to_q.weight.shape[0] // num_heads),
        t_scheduler=str(flow_decoder_module.t_scheduler),
        inference_cfg_rate=float(flow_decoder_module.inference_cfg_rate),
    )


def apply_flow_decoder_time_embedding_torch(
    timesteps: torch.Tensor,
    parameters: CosyVoiceFlowDecoderParameters,
) -> torch.Tensor:
    timesteps = timesteps.reshape(-1).float()
    half_dim = parameters.in_channels // 2
    scale = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=timesteps.device, dtype=torch.float32)) / (half_dim - 1))
    )
    sinusoidal = parameters.time_scale * timesteps.unsqueeze(1) * scale.unsqueeze(0)
    hidden_states = torch.cat([sinusoidal.sin(), sinusoidal.cos()], dim=-1)
    hidden_states = F.linear(hidden_states, parameters.time_linear_1_weight, parameters.time_linear_1_bias)
    hidden_states = F.silu(hidden_states)
    return F.linear(hidden_states, parameters.time_linear_2_weight, parameters.time_linear_2_bias)


def apply_flow_decoder_block_torch(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    parameters: CosyVoiceFlowDecoderBlockParameters,
) -> torch.Tensor:
    hidden_states = F.conv1d(
        hidden_states * mask,
        parameters.conv_weight,
        parameters.conv_bias,
        stride=1,
        padding=parameters.padding,
    )
    hidden_states = F.group_norm(
        hidden_states,
        num_groups=parameters.num_groups,
        weight=parameters.norm_weight,
        bias=parameters.norm_bias,
        eps=parameters.epsilon,
    )
    hidden_states = F.mish(hidden_states)
    return hidden_states * mask


def apply_flow_decoder_resnet_block_torch(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    time_embedding: torch.Tensor,
    parameters: CosyVoiceFlowDecoderResnetBlockParameters,
) -> torch.Tensor:
    residual = F.conv1d(hidden_states * mask, parameters.res_conv_weight, parameters.res_conv_bias, stride=1, padding=0)
    hidden_states = apply_flow_decoder_block_torch(hidden_states, mask, parameters.block1)
    hidden_states = hidden_states + F.linear(F.mish(time_embedding), parameters.time_mlp_weight, parameters.time_mlp_bias).unsqueeze(-1)
    hidden_states = apply_flow_decoder_block_torch(hidden_states, mask, parameters.block2)
    return hidden_states + residual


def _flow_mask_to_attention_bias(mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    attention_mask = mask.bool().repeat(1, hidden_states.shape[1], 1)
    return (1.0 - attention_mask.to(dtype=hidden_states.dtype)) * -1.0e10


def apply_flow_decoder_transformer_block_torch(
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor,
    parameters: CosyVoiceFlowDecoderTransformerBlockParameters,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm1_weight,
        parameters.norm1_bias,
        parameters.epsilon,
    )

    q = F.linear(hidden_states, parameters.attn_q_weight).view(
        hidden_states.shape[0], hidden_states.shape[1], parameters.num_heads, parameters.head_dim
    ).transpose(1, 2)
    k = F.linear(hidden_states, parameters.attn_k_weight).view(
        hidden_states.shape[0], hidden_states.shape[1], parameters.num_heads, parameters.head_dim
    ).transpose(1, 2)
    v = F.linear(hidden_states, parameters.attn_v_weight).view(
        hidden_states.shape[0], hidden_states.shape[1], parameters.num_heads, parameters.head_dim
    ).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) * (parameters.head_dim**-0.5)
    scores = scores + attention_bias.unsqueeze(1)
    attention = torch.softmax(scores, dim=-1)
    hidden_states = torch.matmul(attention, v).transpose(1, 2).contiguous().view(
        residual.shape[0], residual.shape[1], -1
    )
    hidden_states = F.linear(hidden_states, parameters.attn_out_weight, parameters.attn_out_bias)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        parameters.norm3_weight,
        parameters.norm3_bias,
        parameters.epsilon,
    )
    hidden_states = F.linear(hidden_states, parameters.ff_proj_weight, parameters.ff_proj_bias)
    if parameters.ff_activation == "snakebeta":
        assert parameters.ff_alpha is not None
        assert parameters.ff_beta is not None
        alpha = parameters.ff_alpha.exp() if parameters.snake_alpha_logscale else parameters.ff_alpha
        beta = parameters.ff_beta.exp() if parameters.snake_alpha_logscale else parameters.ff_beta
        hidden_states = hidden_states + (1.0 / (beta + 1.0e-9)) * torch.sin(hidden_states * alpha).pow(2)
    elif parameters.ff_activation == "geglu":
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * F.gelu(gate)
    elif parameters.ff_activation == "gelu_approximate":
        hidden_states = F.gelu(hidden_states, approximate="tanh")
    else:
        hidden_states = F.gelu(hidden_states)
    hidden_states = F.linear(hidden_states, parameters.ff_out_weight, parameters.ff_out_bias)
    return residual + hidden_states


def apply_flow_conditional_decoder_torch(
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    timesteps: torch.Tensor,
    parameters: CosyVoiceFlowDecoderParameters,
    *,
    spks: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    streaming: bool = False,
) -> torch.Tensor:
    if streaming:
        raise NotImplementedError("Streaming ConditionalDecoder is out of scope for the public CosyVoice TT path")

    time_embedding = apply_flow_decoder_time_embedding_torch(timesteps, parameters).to(dtype=x.dtype, device=x.device)
    hidden_states = torch.cat([x, mu], dim=1)
    if spks is not None:
        hidden_states = torch.cat([hidden_states, spks.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])], dim=1)
    if cond is not None:
        hidden_states = torch.cat([hidden_states, cond], dim=1)

    hiddens: list[torch.Tensor] = []
    masks = [mask]
    for block in parameters.down_blocks:
        mask_down = masks[-1]
        hidden_states = apply_flow_decoder_resnet_block_torch(hidden_states, mask_down, time_embedding, block.resnet)
        hidden_states_btc = hidden_states.transpose(1, 2).contiguous()
        attention_bias = _flow_mask_to_attention_bias(mask_down, hidden_states_btc)
        for transformer_block in block.transformer_blocks:
            hidden_states_btc = apply_flow_decoder_transformer_block_torch(
                hidden_states_btc, attention_bias, transformer_block
            )
        hidden_states = hidden_states_btc.transpose(1, 2).contiguous()
        hiddens.append(hidden_states)
        hidden_states = F.conv1d(
            hidden_states * mask_down,
            block.sample_weight,
            block.sample_bias,
            stride=block.sample_stride,
            padding=block.sample_padding,
        )
        masks.append(mask_down[:, :, ::2])

    masks = masks[:-1]
    mask_mid = masks[-1]
    for block in parameters.mid_blocks:
        hidden_states = apply_flow_decoder_resnet_block_torch(hidden_states, mask_mid, time_embedding, block.resnet)
        hidden_states_btc = hidden_states.transpose(1, 2).contiguous()
        attention_bias = _flow_mask_to_attention_bias(mask_mid, hidden_states_btc)
        for transformer_block in block.transformer_blocks:
            hidden_states_btc = apply_flow_decoder_transformer_block_torch(
                hidden_states_btc, attention_bias, transformer_block
            )
        hidden_states = hidden_states_btc.transpose(1, 2).contiguous()

    for block in parameters.up_blocks:
        mask_up = masks.pop()
        skip = hiddens.pop()
        hidden_states = torch.cat([hidden_states[:, :, : skip.shape[-1]], skip], dim=1)
        hidden_states = apply_flow_decoder_resnet_block_torch(hidden_states, mask_up, time_embedding, block.resnet)
        hidden_states_btc = hidden_states.transpose(1, 2).contiguous()
        attention_bias = _flow_mask_to_attention_bias(mask_up, hidden_states_btc)
        for transformer_block in block.transformer_blocks:
            hidden_states_btc = apply_flow_decoder_transformer_block_torch(
                hidden_states_btc, attention_bias, transformer_block
            )
        hidden_states = hidden_states_btc.transpose(1, 2).contiguous()
        if block.sample_transpose:
            hidden_states = F.conv_transpose1d(
                hidden_states * mask_up,
                block.sample_weight,
                block.sample_bias,
                stride=block.sample_stride,
                padding=block.sample_padding,
            )
        else:
            hidden_states = F.conv1d(
                hidden_states * mask_up,
                block.sample_weight,
                block.sample_bias,
                stride=block.sample_stride,
                padding=block.sample_padding,
            )

    hidden_states = apply_flow_decoder_block_torch(hidden_states, mask_up, parameters.final_block)
    output = F.conv1d(hidden_states * mask_up, parameters.final_proj_weight, parameters.final_proj_bias, stride=1, padding=0)
    return output * mask


def apply_flow_cfm_inference_torch(
    mu: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_timesteps: int,
    parameters: CosyVoiceFlowDecoderParameters,
    temperature: float = 1.0,
    spks: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    prompt_len: int = 0,
    cache: torch.Tensor | None = None,
    streaming: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn_like(mu).to(mu.dtype) * temperature
    if cache is None:
        cache = torch.zeros((mu.shape[0], mu.shape[1], 0, 2), dtype=mu.dtype, device=mu.device)
    cache_size = int(cache.shape[2])
    if cache_size != 0:
        z[:, :, :cache_size] = cache[:, :, :, 0]
        mu[:, :, :cache_size] = cache[:, :, :, 1]

    overlap_start = max(mu.shape[2] - 34, 0)
    z_cache = torch.cat([z[:, :, :prompt_len], z[:, :, overlap_start:]], dim=2)
    mu_cache = torch.cat([mu[:, :, :prompt_len], mu[:, :, overlap_start:]], dim=2)
    cache = torch.stack([z_cache, mu_cache], dim=-1)

    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if parameters.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    batch_size = int(mu.shape[0])
    conditioning_dtype = spks.dtype if spks is not None else mu.dtype
    x = z
    t = t_span[0].reshape(1)
    dt = t_span[1] - t

    for step in range(1, len(t_span)):
        x_in = torch.cat([x, x], dim=0).to(dtype=conditioning_dtype)
        mask_in = torch.cat([mask, mask], dim=0).to(dtype=conditioning_dtype)
        mu_in = torch.zeros((batch_size * 2, *mu.shape[1:]), dtype=conditioning_dtype, device=mu.device)
        mu_in[:batch_size] = mu.to(dtype=conditioning_dtype)
        t_in = t.expand(batch_size * 2).to(dtype=conditioning_dtype)
        spks_in = None
        if spks is not None:
            spks_in = torch.zeros((batch_size * 2, spks.shape[1]), dtype=conditioning_dtype, device=mu.device)
            spks_in[:batch_size] = spks.to(dtype=conditioning_dtype)
        cond_in = None
        if cond is not None:
            cond_in = torch.zeros((batch_size * 2, *cond.shape[1:]), dtype=conditioning_dtype, device=mu.device)
            cond_in[:batch_size] = cond.to(dtype=conditioning_dtype)

        dphi_dt = apply_flow_conditional_decoder_torch(
            x_in,
            mask_in,
            mu_in,
            t_in,
            parameters,
            spks=spks_in,
            cond=cond_in,
            streaming=streaming,
        )
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
        dphi_dt = ((1.0 + parameters.inference_cfg_rate) * dphi_dt) - (parameters.inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return x.float(), cache


class CosyVoiceTorchFlowDecoder:
    def __init__(self, flow_module: Any):
        self.flow_module = flow_module
        self.parameters = extract_flow_decoder_parameters(flow_module.decoder)

    def inference(
        self,
        *,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
        n_timesteps: int,
        prompt_len: int = 0,
        cache: torch.Tensor | None = None,
        temperature: float = 1.0,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_flow_cfm_inference_torch(
            mu=mu,
            mask=mask,
            n_timesteps=n_timesteps,
            parameters=self.parameters,
            temperature=temperature,
            spks=spks,
            cond=cond,
            prompt_len=prompt_len,
            cache=cache,
            streaming=streaming,
        )
