from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.audio.cosy_voice.demo.common import CosyVoiceCase, load_cases
from models.demos.audio.cosy_voice.demo.validate_tt import validate_accuracy_metrics, validate_audio_metrics
from models.demos.audio.cosy_voice.tt.flow import (
    CosyVoiceTorchFlowDecoder,
    CosyVoiceTTFlowBridge,
    CosyVoiceTTFlowEncoder,
    CosyVoiceTTFlowLengthRegulator,
    apply_flow_encoder_layer_torch,
    apply_flow_encoder_projection_torch,
    apply_flow_conditional_decoder_torch,
    apply_flow_length_regulator_torch,
    apply_flow_speaker_projection_torch,
    build_flow_condition,
    build_flow_inputs_from_model_input,
    compute_decode_mel_length,
    extract_flow_decoder_parameters,
    extract_flow_bridge_parameters,
    extract_flow_encoder_layer_parameters,
    extract_flow_length_regulator_parameters,
)
from models.demos.audio.cosy_voice.tt.llm import (
    apply_legacy_embed_torch,
    apply_semantic_layer_torch,
    build_autoregressive_attention_mask,
    build_semantic_inputs_from_model_input,
    compute_decode_length_bounds,
    extract_legacy_embed_parameters,
    extract_semantic_layer_parameters,
    extract_semantic_output_parameters,
)
from models.demos.audio.cosy_voice.tt.pipeline import CosyVoicePipeline
from models.demos.audio.cosy_voice.tt.reference import SemanticTokenResult

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo"
REFERENCE_REPO = Path("/vol/stor/reference_repos/CosyVoice")
MODEL_ROOT = Path("/root/.cache/cosyvoice_models")


def _dummy_semantic_tokens(length: int = 12) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.int32) % 128).reshape(1, -1)


def _build_projected_flow_hidden(_real_pipeline, case_name: str, token_length: int = 24):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}[case_name]
    prepared = _real_pipeline.prepare_case(case)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    flow_module = model.model.flow
    flow_inputs = build_flow_inputs_from_model_input(
        flow_module=flow_module,
        model_input=prepared.payload,
        source_speech_token=_dummy_semantic_tokens(token_length),
    )
    masks = torch.ones((1, 1, flow_inputs.token_embedding.shape[1]), dtype=torch.bool)
    hidden_states, positional_embedding, _ = flow_module.encoder.embed(flow_inputs.token_embedding, masks)
    for layer_idx in range(len(flow_module.encoder.encoders)):
        parameters = extract_flow_encoder_layer_parameters(flow_module.state_dict(), layer_num=layer_idx)
        hidden_states = apply_flow_encoder_layer_torch(hidden_states, positional_embedding, parameters, num_heads=8)
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        flow_module.encoder.after_norm.weight,
        flow_module.encoder.after_norm.bias,
        flow_module.encoder.after_norm.eps,
    )
    projected_hidden = apply_flow_encoder_projection_torch(
        hidden_states,
        extract_flow_bridge_parameters(flow_module.state_dict()),
    )
    return case, flow_module, flow_inputs, projected_hidden


def test_accuracy_manifest_is_public_and_valid():
    cases = load_cases(DEMO_ROOT / "accuracy_cases.json")
    assert {case.mode for case in cases} == {"sft", "zero_shot", "cross_lingual", "instruct"}
    assert {"zh", "en", "ja", "yue", "ko"}.issubset({case.language for case in cases})


def test_decode_length_bounds_match_public_ratios():
    min_len, max_len = compute_decode_length_bounds(text_token_len=48, prompt_text_token_len=8)
    assert min_len == 80
    assert max_len == 800


def test_autoregressive_attention_mask_is_lower_triangular():
    mask = build_autoregressive_attention_mask(4)
    assert mask.shape == (1, 4, 4)
    assert mask[0, 0, 3].item() is False
    assert mask[0, 3, 0].item() is True


def test_flow_condition_pads_decode_region():
    import torch

    prompt_feat = torch.ones((1, 10, 80))
    cond = build_flow_condition(prompt_feat, decode_mel_length=7)
    assert cond.shape == (1, 17, 80)
    assert cond[:, :10].sum().item() == 800
    assert cond[:, 10:].sum().item() == 0


def test_compute_decode_mel_length():
    assert compute_decode_mel_length(50) == 86


def test_flow_bridge_parameter_extraction_matches_public_checkpoint_shapes(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["sft_zh"]
    model, _ = _real_pipeline.reference.get_model(case.mode)
    parameters = extract_flow_bridge_parameters(model.model.flow.state_dict())
    assert parameters.input_embedding_weight.shape == (4096, 512)
    assert parameters.speaker_weight.shape == (80, 192)
    assert parameters.speaker_bias.shape == (80,)
    assert parameters.encoder_proj_weight.shape == (80, 512)
    assert parameters.encoder_proj_bias.shape == (80,)


def test_validate_accuracy_metrics_accepts_public_gate():
    validate_accuracy_metrics(96.0)


def test_validate_accuracy_metrics_rejects_regression():
    with pytest.raises(AssertionError):
        validate_accuracy_metrics(94.9)


def test_validate_audio_metrics_accepts_realistic_output():
    validate_audio_metrics(22050, 0.25)


@pytest.mark.parametrize(
    ("sample_rate", "audio_seconds"),
    [
        (0, 0.5),
        (22050, 0.01),
    ],
)
def test_validate_audio_metrics_rejects_invalid_outputs(sample_rate, audio_seconds):
    with pytest.raises(AssertionError):
        validate_audio_metrics(sample_rate, audio_seconds)


class _FakeLegacyEmbed(torch.nn.Module):
    def forward(self, x, mask, offset=0):
        return x + 4.0, torch.full_like(x, 5.0), mask


class _FakeSemanticLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm_input_size = 4
        self.sos = 0
        self.task_id = 1
        self.text_embedding = torch.nn.Embedding(32, 4)
        self.spk_embed_affine_layer = torch.nn.Linear(2, 4)
        self.llm_embedding = torch.nn.Embedding(2, 4)
        self.speech_embedding = torch.nn.Embedding(16, 4)
        self.llm = SimpleNamespace(embed=_FakeLegacyEmbed())

    def encode(self, text, text_lengths):
        return text + 3.0, text_lengths


def test_build_semantic_inputs_from_model_input_handles_optional_inputs():
    llm = _FakeSemanticLLM()
    payload = {
        "text": torch.tensor([[1, 2, 3]], dtype=torch.int32),
        "text_len": torch.tensor([3], dtype=torch.int32),
        "llm_embedding": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
    }
    semantic_inputs = build_semantic_inputs_from_model_input(llm_module=llm, model_input=payload)
    assert semantic_inputs.prompt_text.shape == (1, 0)
    assert semantic_inputs.prompt_speech_token.shape == (1, 0)
    assert semantic_inputs.prefill_seq_len == 6
    assert semantic_inputs.min_decode_length == 6
    assert semantic_inputs.max_decode_length == 60
    assert semantic_inputs.lm_input_embed.shape == semantic_inputs.lm_input.shape
    assert semantic_inputs.lm_input_mask.shape == (1, 1, semantic_inputs.prefill_seq_len)


@pytest.fixture(scope="module")
def _real_pipeline():
    if not REFERENCE_REPO.exists() or not MODEL_ROOT.exists():
        pytest.skip("CosyVoice reference repo or checkpoints are unavailable")
    pipeline = CosyVoicePipeline(
        reference_repo=str(REFERENCE_REPO),
        model_root=str(MODEL_ROOT),
        text_frontend=False,
    )
    yield pipeline
    pipeline.close()


@pytest.mark.parametrize(
    "case_name",
    ["sft_zh", "zero_shot_zh", "cross_lingual_en", "instruct_zh"],
)
def test_prepare_semantic_inputs_matches_public_mode_contract(_real_pipeline, case_name):
    case_map = {case.name: case for case in load_cases(DEMO_ROOT / "accuracy_cases.json")}
    case = case_map[case_name]
    prepared = _real_pipeline.prepare_case(case)
    semantic_inputs = _real_pipeline.prepare_semantic_inputs(case, prepared=prepared)
    assert semantic_inputs.prefill_seq_len > 0
    assert semantic_inputs.lm_input_embed.shape == semantic_inputs.lm_input.shape
    assert semantic_inputs.lm_input_positional_embedding.shape[-1] == semantic_inputs.lm_input.shape[-1]
    assert semantic_inputs.max_decode_length > semantic_inputs.min_decode_length >= 0
    if case.mode == "sft":
        assert semantic_inputs.prompt_text.shape[1] == 0
        assert semantic_inputs.prompt_speech_token.shape[1] == 0
        assert semantic_inputs.speaker_projection.shape[1] == 1
    if case.mode == "zero_shot":
        assert semantic_inputs.prompt_text.shape[1] > 0
        assert semantic_inputs.prompt_speech_token.shape[1] > 0
    if case.mode == "cross_lingual":
        assert semantic_inputs.prompt_text.shape[1] == 0
        assert semantic_inputs.prompt_speech_token.shape[1] == 0
    if case.mode == "instruct":
        assert semantic_inputs.prompt_text.shape[1] > 0
        assert semantic_inputs.speaker_projection.shape[1] == 0


@pytest.mark.parametrize(
    "case_name",
    ["sft_zh", "zero_shot_zh", "cross_lingual_en", "instruct_zh"],
)
def test_prepare_flow_inputs_matches_public_mode_contract(_real_pipeline, case_name):
    case_map = {case.name: case for case in load_cases(DEMO_ROOT / "accuracy_cases.json")}
    case = case_map[case_name]
    prepared = _real_pipeline.prepare_case(case)
    semantic_tokens = SemanticTokenResult(
        case=case,
        model_dir="/tmp/fake-model",
        tokens=_dummy_semantic_tokens(),
        wall_seconds=0.0,
    )
    flow_inputs = _real_pipeline.prepare_flow_inputs(case, semantic_tokens, prepared=prepared)
    assert flow_inputs.source_speech_token.shape[1] == semantic_tokens.tokens.shape[1]
    assert int(flow_inputs.source_speech_token_len.item()) == semantic_tokens.tokens.shape[1]
    assert int(flow_inputs.full_token_len.item()) == flow_inputs.full_token.shape[1]
    assert flow_inputs.token_embedding.shape == (1, flow_inputs.full_token.shape[1], 512)
    assert flow_inputs.speaker_projection.shape == (1, 80)
    assert flow_inputs.condition.shape == (1, flow_inputs.prompt_mel_length + flow_inputs.decode_mel_length, 80)
    assert flow_inputs.condition[:, : flow_inputs.prompt_mel_length].shape[1] == flow_inputs.prompt_mel_length
    if case.mode == "sft":
        assert flow_inputs.prompt_speech_token.shape[1] == 0
        assert flow_inputs.prompt_speech_feat.shape[1] == 0
    elif case.mode in {"zero_shot", "cross_lingual"}:
        assert flow_inputs.prompt_speech_token.shape[1] > 0
        assert flow_inputs.prompt_speech_feat.shape[1] > 0
    else:
        assert flow_inputs.prompt_speech_token.shape[1] == 0
        assert flow_inputs.prompt_speech_feat.shape[1] == 0


def test_legacy_embed_projection_matches_real_checkpoint(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["sft_zh"]
    prepared = _real_pipeline.prepare_case(case)
    semantic_inputs = _real_pipeline.prepare_semantic_inputs(case, prepared=prepared)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    parameters = extract_legacy_embed_parameters(model.model.llm.state_dict(), prefix="llm.embed.out")
    legacy_embed = apply_legacy_embed_torch(semantic_inputs.lm_input, parameters)
    assert legacy_embed.shape == semantic_inputs.lm_input.shape
    assert torch.allclose(
        legacy_embed * (semantic_inputs.lm_input.shape[-1] ** 0.5),
        semantic_inputs.lm_input_embed,
        atol=1e-4,
        rtol=1e-4,
    )


def test_semantic_layer_parameter_extraction_matches_real_checkpoint(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["sft_zh"]
    model, _ = _real_pipeline.reference.get_model(case.mode)
    layer = extract_semantic_layer_parameters(model.model.llm.state_dict(), layer_num=0, prefix="llm.encoders")
    assert layer.q_weight.shape == (1024, 1024)
    assert layer.k_weight.shape == (1024, 1024)
    assert layer.v_weight.shape == (1024, 1024)
    assert layer.out_weight.shape == (1024, 1024)
    assert layer.pos_weight.shape == (1024, 1024)
    assert layer.ffn_w1_weight.shape == (4096, 1024)
    assert layer.ffn_w2_weight.shape == (1024, 4096)
    assert layer.pos_bias_u.shape == (16, 64)
    assert layer.pos_bias_v.shape == (16, 64)


def test_semantic_output_parameter_extraction_matches_real_checkpoint(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["sft_zh"]
    model, _ = _real_pipeline.reference.get_model(case.mode)
    output = extract_semantic_output_parameters(model.model.llm.state_dict())
    assert output.after_norm_weight.shape == (1024,)
    assert output.decoder_weight.shape == (4097, 1024)
    assert output.decoder_bias.shape == (4097,)
    assert output.speech_embedding_weight.shape == (4096, 1024)


def test_flow_bridge_torch_helpers_match_reference_modules(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["zero_shot_zh"]
    prepared = _real_pipeline.prepare_case(case)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    flow_module = model.model.flow
    flow_inputs = build_flow_inputs_from_model_input(
        flow_module=flow_module,
        model_input=prepared.payload,
        source_speech_token=_dummy_semantic_tokens(24),
    )
    parameters = extract_flow_bridge_parameters(flow_module.state_dict())

    expected_embedding = flow_module.input_embedding(torch.clamp(flow_inputs.full_token, min=0))
    assert torch.allclose(flow_inputs.token_embedding, expected_embedding, atol=1e-4, rtol=1e-4)

    normalized_embedding, speaker_projection = apply_flow_speaker_projection_torch(
        flow_inputs.flow_embedding, parameters
    )
    assert torch.allclose(normalized_embedding, F.normalize(flow_inputs.flow_embedding, dim=1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(speaker_projection, flow_inputs.speaker_projection, atol=1e-4, rtol=1e-4)

    encoded_hidden, encoded_mask = flow_module.encoder(flow_inputs.token_embedding, flow_inputs.full_token_len)
    projected_hidden = apply_flow_encoder_projection_torch(encoded_hidden, parameters)
    reference_projected_hidden = flow_module.encoder_proj(encoded_hidden)
    assert encoded_mask.shape[-1] == flow_inputs.full_token.shape[1]
    assert torch.allclose(projected_hidden, reference_projected_hidden, atol=1e-4, rtol=1e-4)


def test_flow_bridge_ttnn_matches_reference_bridge_ops(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["zero_shot_zh"]
    prepared = _real_pipeline.prepare_case(case)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    flow_module = model.model.flow
    flow_inputs = build_flow_inputs_from_model_input(
        flow_module=flow_module,
        model_input=prepared.payload,
        source_speech_token=_dummy_semantic_tokens(24),
    )
    bridge = CosyVoiceTTFlowBridge(flow_module, _real_pipeline._get_device())
    encoded_hidden, _ = flow_module.encoder(flow_inputs.token_embedding, flow_inputs.full_token_len)
    parameters = extract_flow_bridge_parameters(flow_module.state_dict())

    tt_embedding = bridge.embed_tokens(flow_inputs.full_token, flow_inputs.full_token_len)
    tt_speaker_projection = bridge.project_speaker_embedding(flow_inputs.flow_embedding)
    tt_encoder_projection = bridge.project_encoder_output(encoded_hidden)

    assert torch.allclose(tt_embedding, flow_inputs.token_embedding, atol=2e-1, rtol=2e-1)
    assert torch.allclose(
        tt_speaker_projection,
        apply_flow_speaker_projection_torch(flow_inputs.flow_embedding, parameters)[1],
        atol=1e-1,
        rtol=1e-1,
    )
    assert torch.allclose(
        tt_encoder_projection,
        apply_flow_encoder_projection_torch(encoded_hidden, parameters),
        atol=2e-1,
        rtol=2e-1,
    )


def test_flow_encoder_layer_torch_matches_reference_layer(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["zero_shot_zh"]
    prepared = _real_pipeline.prepare_case(case)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    flow_module = model.model.flow
    flow_inputs = build_flow_inputs_from_model_input(
        flow_module=flow_module,
        model_input=prepared.payload,
        source_speech_token=_dummy_semantic_tokens(24),
    )
    masks = torch.ones((1, 1, flow_inputs.token_embedding.shape[1]), dtype=torch.bool)
    embedded_hidden, positional_embedding, _ = flow_module.encoder.embed(flow_inputs.token_embedding, masks)
    parameters = extract_flow_encoder_layer_parameters(flow_module.state_dict(), layer_num=0)
    chunk_mask = torch.ones(
        (1, embedded_hidden.shape[1], embedded_hidden.shape[1]),
        dtype=torch.bool,
        device=embedded_hidden.device,
    )
    reference_hidden, _, _, _ = flow_module.encoder.encoders[0](
        embedded_hidden, chunk_mask, positional_embedding, masks
    )
    torch_hidden = apply_flow_encoder_layer_torch(
        embedded_hidden,
        positional_embedding,
        parameters,
        num_heads=8,
    )
    assert torch.allclose(torch_hidden, reference_hidden, atol=1e-4, rtol=1e-4)


def test_flow_encoder_ttnn_matches_reference_encoder(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["zero_shot_zh"]
    prepared = _real_pipeline.prepare_case(case)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    flow_module = model.model.flow
    flow_inputs = build_flow_inputs_from_model_input(
        flow_module=flow_module,
        model_input=prepared.payload,
        source_speech_token=_dummy_semantic_tokens(24),
    )
    tt_encoder = CosyVoiceTTFlowEncoder(flow_module, _real_pipeline._get_device())
    tt_hidden = tt_encoder.encode(flow_inputs.token_embedding, flow_inputs.full_token_len)
    reference_hidden, _ = flow_module.encoder(flow_inputs.token_embedding, flow_inputs.full_token_len)
    assert torch.allclose(tt_hidden, reference_hidden, atol=3e-1, rtol=3e-1)


def test_flow_length_regulator_torch_matches_reference_module(_real_pipeline):
    _, flow_module, flow_inputs, projected_hidden = _build_projected_flow_hidden(_real_pipeline, "zero_shot_zh")
    prompt_token_len = int(flow_inputs.prompt_speech_token.shape[1])
    parameters = extract_flow_length_regulator_parameters(flow_module.state_dict())
    torch_hidden, torch_mel_length = apply_flow_length_regulator_torch(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        parameters,
        input_frame_rate=flow_module.input_frame_rate,
    )
    reference_hidden, reference_mel_length = flow_module.length_regulator.inference(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        flow_module.input_frame_rate,
    )
    assert torch_mel_length == reference_mel_length
    assert torch.allclose(torch_hidden, reference_hidden, atol=1e-4, rtol=1e-4)


def test_flow_length_regulator_ttnn_matches_reference_module(_real_pipeline):
    _, flow_module, flow_inputs, projected_hidden = _build_projected_flow_hidden(_real_pipeline, "zero_shot_zh")
    prompt_token_len = int(flow_inputs.prompt_speech_token.shape[1])
    tt_length_regulator = CosyVoiceTTFlowLengthRegulator(flow_module, _real_pipeline._get_device())
    tt_hidden, tt_mel_length = tt_length_regulator.inference(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        flow_module.input_frame_rate,
    )
    reference_hidden, reference_mel_length = flow_module.length_regulator.inference(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        flow_module.input_frame_rate,
    )
    assert tt_mel_length == reference_mel_length
    assert torch.allclose(tt_hidden, reference_hidden, atol=4e-1, rtol=4e-1)


def test_flow_decoder_parameter_extraction_matches_reference_decoder(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["zero_shot_zh"]
    model, _ = _real_pipeline.reference.get_model(case.mode)
    parameters = extract_flow_decoder_parameters(model.model.flow.decoder)
    assert len(parameters.down_blocks) == 2
    assert len(parameters.mid_blocks) == 12
    assert len(parameters.up_blocks) == 2
    assert parameters.in_channels == 320
    assert parameters.out_channels == 80
    assert parameters.num_heads == 8
    assert parameters.head_dim == 64
    assert parameters.final_proj_weight.shape == (80, 256, 1)


def test_flow_decoder_estimator_torch_matches_reference_module(_real_pipeline):
    _, flow_module, flow_inputs, projected_hidden = _build_projected_flow_hidden(_real_pipeline, "zero_shot_zh", token_length=8)
    prompt_token_len = int(flow_inputs.prompt_speech_token.shape[1])
    regulator_parameters = extract_flow_length_regulator_parameters(flow_module.state_dict())
    acoustic_hidden, _ = apply_flow_length_regulator_torch(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        regulator_parameters,
        input_frame_rate=flow_module.input_frame_rate,
    )
    mu = acoustic_hidden.transpose(1, 2).contiguous()
    mask = torch.ones((1, 1, mu.shape[-1]), dtype=torch.bool, device=mu.device)
    cond = flow_inputs.condition.transpose(1, 2).contiguous()
    parameters = extract_flow_decoder_parameters(flow_module.decoder)

    torch.manual_seed(0)
    reference_hidden = flow_module.decoder.estimator(
        x=torch.randn_like(mu),
        mask=mask,
        mu=mu,
        t=torch.ones((1,), dtype=mu.dtype),
        spks=flow_inputs.speaker_projection.to(dtype=mu.dtype),
        cond=cond,
    )

    torch.manual_seed(0)
    torch_hidden = apply_flow_conditional_decoder_torch(
        x=torch.randn_like(mu),
        mask=mask,
        mu=mu,
        timesteps=torch.ones((1,), dtype=mu.dtype),
        parameters=parameters,
        spks=flow_inputs.speaker_projection.to(dtype=mu.dtype),
        cond=cond,
    )
    assert torch.allclose(torch_hidden, reference_hidden, atol=1e-4, rtol=1e-4)


def test_flow_decoder_torch_matches_reference_module(_real_pipeline):
    _, flow_module, flow_inputs, projected_hidden = _build_projected_flow_hidden(_real_pipeline, "zero_shot_zh", token_length=8)
    prompt_token_len = int(flow_inputs.prompt_speech_token.shape[1])
    regulator_parameters = extract_flow_length_regulator_parameters(flow_module.state_dict())
    acoustic_hidden, total_mel_length = apply_flow_length_regulator_torch(
        projected_hidden[:, :prompt_token_len],
        projected_hidden[:, prompt_token_len:],
        flow_inputs.prompt_mel_length,
        flow_inputs.decode_mel_length,
        regulator_parameters,
        input_frame_rate=flow_module.input_frame_rate,
    )
    mu = acoustic_hidden.transpose(1, 2).contiguous()
    mask = torch.ones((1, 1, total_mel_length), dtype=torch.bool, device=mu.device)
    cond = flow_inputs.condition.transpose(1, 2).contiguous()
    cache = torch.zeros((1, 80, 0, 2), dtype=mu.dtype, device=mu.device)
    decoder = CosyVoiceTorchFlowDecoder(flow_module)

    torch.manual_seed(0)
    reference_feat, reference_cache = flow_module.decoder(
        mu=mu,
        mask=mask,
        spks=flow_inputs.speaker_projection.to(dtype=mu.dtype),
        cond=cond,
        n_timesteps=2,
        prompt_len=flow_inputs.prompt_mel_length,
        cache=cache.clone(),
    )

    torch.manual_seed(0)
    torch_feat, torch_cache = decoder.inference(
        mu=mu,
        mask=mask,
        spks=flow_inputs.speaker_projection.to(dtype=mu.dtype),
        cond=cond,
        n_timesteps=2,
        prompt_len=flow_inputs.prompt_mel_length,
        cache=cache.clone(),
    )
    assert torch.allclose(torch_feat, reference_feat, atol=1e-4, rtol=1e-4)
    assert torch.allclose(torch_cache, reference_cache, atol=1e-4, rtol=1e-4)


def test_semantic_layer_torch_matches_reference_layer(_real_pipeline):
    case = {item.name: item for item in load_cases(DEMO_ROOT / "accuracy_cases.json")}["sft_zh"]
    prepared = _real_pipeline.prepare_case(case)
    semantic_inputs = _real_pipeline.prepare_semantic_inputs(case, prepared=prepared)
    model, _ = _real_pipeline.reference.get_model(case.mode)
    parameters = extract_semantic_layer_parameters(model.model.llm.state_dict(), layer_num=0, prefix="llm.encoders")
    causal_mask = build_autoregressive_attention_mask(semantic_inputs.prefill_seq_len)
    reference_hidden, _, reference_cache, _ = model.model.llm.llm.encoders[0](
        semantic_inputs.lm_input_embed,
        causal_mask,
        semantic_inputs.lm_input_positional_embedding,
    )
    hidden_states, cache = apply_semantic_layer_torch(
        semantic_inputs.lm_input_embed,
        semantic_inputs.lm_input_positional_embedding,
        parameters,
        num_heads=16,
        causal=True,
    )
    reference_key, reference_value = torch.split(reference_cache, reference_cache.shape[-1] // 2, dim=-1)
    assert torch.allclose(hidden_states, reference_hidden, atol=1e-4, rtol=1e-4)
    assert torch.allclose(cache[0], reference_key, atol=1e-4, rtol=1e-4)
    assert torch.allclose(cache[1], reference_value, atol=1e-4, rtol=1e-4)


def test_pipeline_accuracy_uses_greedy_reference_targets():
    calls = {}

    class _FakeReference:
        def prepare_semantic_inputs(self, case, payload):
            calls["prepare_semantic_inputs"] = (case.name, payload)
            return "semantic-inputs"

        def generate_semantic_greedy_tokens(self, case, payload):
            calls["generate_semantic_greedy_tokens"] = (case.name, payload)
            return SemanticTokenResult(
                case=case,
                model_dir="/tmp/fake-model",
                tokens=torch.tensor([[1, 2, 3]], dtype=torch.int32),
                wall_seconds=0.1,
            )

        def generate_semantic_tokens(self, case, payload):  # pragma: no cover - should not be used
            raise AssertionError("sampled semantic tokens are not a valid public accuracy target")

    class _FakeGenerator:
        def teacher_forced_accuracy(self, semantic_inputs, tokens):
            calls["teacher_forced_accuracy"] = (semantic_inputs, tokens.clone())
            return 99.0

    pipeline = CosyVoicePipeline.__new__(CosyVoicePipeline)
    pipeline.reference = _FakeReference()
    pipeline.prepare_case = lambda case: SimpleNamespace(payload="prepared-payload")
    pipeline._get_semantic_generator = lambda mode: _FakeGenerator()

    case = CosyVoiceCase(name="fake", mode="sft", text="hi", speaker_id="spk")
    accuracy = pipeline.evaluate_semantic_token_accuracy(case)

    assert accuracy == 99.0
    assert calls["prepare_semantic_inputs"] == ("fake", "prepared-payload")
    assert calls["generate_semantic_greedy_tokens"] == ("fake", "prepared-payload")
    assert calls["teacher_forced_accuracy"][0] == "semantic-inputs"
    assert torch.equal(calls["teacher_forced_accuracy"][1], torch.tensor([[1, 2, 3]], dtype=torch.int32))
