# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical host-oracle, device-parity, and long-prompt attention regressions."""

import os

import pytest
import torch

from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    canvas_positions,
)


NEGATIVE_INFINITY = -1.0e9

_requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
_requires_device_integration = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run QB2 bidirectional attention integration tests",
)

if os.environ.get("DG_RUN_DEVICE") == "1":
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextRotaryEmbedding,
        Gemma4TextScaledWordEmbedding,
        apply_rotary_pos_emb,
    )

    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tests.test_factory import (
        find_layer_idx,
        num_layers_for_full_attention_group,
        parametrize_mesh_with_fabric,
    )
    from models.demos.gemma4.tests.unit.test_model import (
        _create_hf_model,
        _create_hf_text_config,
        _hf_model_state_to_tt_state,
    )
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.reference.denoise_loop import (
        denoise_block as reference_denoise_block,
    )
    from models.experimental.diffusion_gemma.reference.self_conditioning import (
        SelfConditioning,
    )
    from models.experimental.diffusion_gemma.tests.trajectory_pcc import (
        compare_trajectories,
    )
    from models.experimental.diffusion_gemma.tt.denoise_forward import (
        DenoiseLogitsAdapter,
        denoise_attention_forward,
        denoise_hidden_forward,
        denoise_logits_from_tokens,
        embed_canvas_tokens,
        read_prompt_kv_cache_slice,
    )
    from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block
    from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model
    from models.experimental.diffusion_gemma.tt.self_conditioning import (
        TtSelfConditioning,
    )
    from tests.ttnn.utils_for_testing import assert_with_pcc


def _mesh_1x4(function):
    if os.environ.get("DG_RUN_DEVICE") != "1":
        return function
    return parametrize_mesh_with_fabric([(1, 4)])(function)


def _attend(mask):
    return mask == 0


# Minimal host oracles for full, sliding-prefix, and local-window visibility.
def test_denoise_mask_is_fully_bidirectional_by_default():
    mask = build_canvas_denoise_mask(prompt_len=20, canvas_len=8)
    assert mask.shape == (8, 28)
    assert torch.all(mask == 0)


def test_sliding_attention_layer_type_windows_prompt_tail():
    prompt_len, canvas_len, sliding_window = 10, 6, 4
    attend = _attend(
        build_canvas_denoise_mask(
            prompt_len,
            canvas_len,
            layer_type="sliding_attention",
            sliding_window=sliding_window,
        )
    )

    keep_from = prompt_len - (sliding_window - 1)
    assert not attend[0, keep_from - 1]
    assert attend[0, keep_from]
    assert attend[0, prompt_len - 1]
    assert torch.all(attend[:, prompt_len:])
    assert torch.all(attend == attend[:1])


def test_local_window_is_symmetric_and_centered():
    prompt_len, canvas_len, window_half = 10, 12, 3
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=True,
        window_half=window_half,
    )
    attend = _attend(mask)
    query_positions = canvas_positions(prompt_len, canvas_len)

    assert mask.shape == (canvas_len, prompt_len + canvas_len)
    for row in range(canvas_len):
        keys = attend[row].nonzero(as_tuple=True)[0]
        assert int(keys.min()) == max(
            0,
            int(query_positions[row]) - window_half,
        )
        assert int(keys.max()) == min(
            prompt_len + canvas_len - 1,
            int(query_positions[row]) + window_half,
        )
        assert attend[row, int(query_positions[row])]


def _build_tt_model(
    mesh_device,
    hf_model,
    hf_text_config,
    *,
    num_layers,
    max_seq_len,
):
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tensor_parallel = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tensor_parallel),
    )
    return DiffusionGemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=(CCLManager(mesh_device, num_links=1) if tensor_parallel > 1 else None),
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=1,
        num_layers=num_layers,
        create_kv_cache=True,
    )


def _mesh_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def _to_torch(tt_tensor, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
    return ttnn.to_torch(tt_tensor)


def _to_device(mesh_device, value):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _to_device_tokens(mesh_device, value):
    return ttnn.from_torch(
        value.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _to_device_canvas_ids(mesh_device, value):
    return ttnn.from_torch(
        value.view(value.shape[0], 1, value.shape[1], 1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _torch_attention_reference(
    hf_model,
    hf_text_config,
    layer_idx,
    canvas_hidden,
    kv_hidden,
    mask,
):
    layer_type = hf_text_config.layer_types[layer_idx]
    attention = hf_model.layers[layer_idx].self_attn
    head_dim = attention.head_dim
    query_shape = (*canvas_hidden.shape[:-1], -1, head_dim)
    kv_shape = (*kv_hidden.shape[:-1], -1, head_dim)

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    position_ids = torch.arange(kv_hidden.shape[1]).unsqueeze(0)
    cosine, sine = rope(
        kv_hidden,
        position_ids,
        layer_type=layer_type,
    )
    query_cosine = cosine[:, -canvas_hidden.shape[1] :, :]
    query_sine = sine[:, -canvas_hidden.shape[1] :, :]

    query = attention.q_norm(attention.q_proj(canvas_hidden).view(query_shape))
    query = apply_rotary_pos_emb(
        query,
        query_cosine,
        query_sine,
        unsqueeze_dim=2,
    ).transpose(1, 2)
    key_linear = attention.k_proj(kv_hidden).view(kv_shape)
    value_linear = attention.v_proj(kv_hidden).view(kv_shape) if attention.v_proj is not None else key_linear
    key = attention.k_norm(key_linear)
    key = apply_rotary_pos_emb(
        key,
        cosine,
        sine,
        unsqueeze_dim=2,
    ).transpose(1, 2)
    value = attention.v_norm(value_linear).transpose(1, 2)
    if attention.num_key_value_groups != 1:
        key = key.repeat_interleave(
            attention.num_key_value_groups,
            dim=1,
        )
        value = value.repeat_interleave(
            attention.num_key_value_groups,
            dim=1,
        )
    out = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        is_causal=False,
        scale=1.0,
    )
    out = out.transpose(1, 2).reshape(
        canvas_hidden.shape[0],
        canvas_hidden.shape[1],
        -1,
    )
    return attention.o_proj(out)


def _torch_denoise_hidden_reference(
    hf_model,
    canvas_hidden,
    prompt_kv_hidden_by_layer,
    mask,
):
    hidden = canvas_hidden
    for layer_idx, layer in enumerate(hf_model.layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        kv_hidden = torch.cat(
            [prompt_kv_hidden_by_layer[layer_idx], normed],
            dim=1,
        )
        hidden = _torch_attention_reference(
            hf_model,
            hf_model.config,
            layer_idx,
            normed,
            kv_hidden,
            mask,
        )
        hidden = layer.post_attention_layernorm(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(hidden)
        if layer.enable_moe_block:
            hidden_1 = layer.post_feedforward_layernorm_1(hidden)
            hidden_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = layer.router(hidden_flat)
            hidden_2 = layer.pre_feedforward_layernorm_2(hidden_flat)
            hidden_2 = layer.experts(
                hidden_2,
                top_k_index,
                top_k_weights,
            )
            hidden_2 = hidden_2.reshape(residual.shape)
            hidden_2 = layer.post_feedforward_layernorm_2(hidden_2)
            hidden = hidden_1 + hidden_2
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = residual + hidden
        hidden = hidden * layer.layer_scalar
    return hf_model.norm(hidden)


def _torch_denoise_logits_reference(
    hf_model,
    canvas_hidden,
    prompt_kv_hidden_by_layer,
    mask,
):
    hidden = _torch_denoise_hidden_reference(
        hf_model,
        canvas_hidden,
        prompt_kv_hidden_by_layer,
        mask,
    )
    logits = hf_model.lm_head(hidden)
    cap = hf_model.config.final_logit_softcapping
    if cap and cap > 0:
        logits = torch.tanh(logits / cap) * cap
    return logits


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
@pytest.mark.parametrize(
    "layer_type",
    ["sliding_attention", "full_attention"],
)
def test_real_attention_denoise_mask_covers_prompt_prefix_for_layer_type(
    mesh_device,
    layer_type,
    reset_seeds,
):
    torch.manual_seed(5)
    prompt_len, canvas_len = 64, 256
    total_len = prompt_len + canvas_len
    base_config = _create_hf_text_config(vocab_size=256, num_layers=1)
    num_layers = 1 if layer_type == "sliding_attention" else num_layers_for_full_attention_group(base_config)
    hf_text_config = _create_hf_text_config(
        vocab_size=256,
        num_layers=num_layers,
    )
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(
        mesh_device,
        hf_model,
        hf_text_config,
        num_layers=num_layers,
        max_seq_len=total_len,
    )
    layer_idx = find_layer_idx(hf_text_config, layer_type)

    prompt_hidden = torch.randn(
        1,
        prompt_len,
        hf_text_config.hidden_size,
    )
    canvas_hidden = torch.randn(
        1,
        canvas_len,
        hf_text_config.hidden_size,
    )
    kv_hidden = torch.cat([prompt_hidden, canvas_hidden], dim=1)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        neg_inf=NEGATIVE_INFINITY,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_attention_reference(
            hf_model,
            hf_text_config,
            layer_idx,
            canvas_hidden,
            kv_hidden,
            mask,
        )

    tt_canvas_hidden = _to_device(
        mesh_device,
        canvas_hidden.unsqueeze(0),
    )
    tt_prompt_hidden = _to_device(
        mesh_device,
        prompt_hidden.unsqueeze(0),
    )
    tt_prompt_out = tt_model.layers[layer_idx].self_attn(
        tt_prompt_hidden,
        rope_mats=tt_model._get_rope_mats(
            layer_idx,
            seq_len=prompt_len,
        ),
        is_decode=False,
        keep_kv=True,
    )
    tt_prompt_out.deallocate(True)
    tt_prompt_kv = tt_model.layers[layer_idx].self_attn._last_kv
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=layer_idx,
        prompt_kv=tt_prompt_kv,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch(tt_out, mesh_device).squeeze(0)
    tt_prompt_kv[0].deallocate(True)
    tt_prompt_kv[1].deallocate(True)

    passing, message = assert_with_pcc(
        golden.float(),
        out.float(),
        0.99,
    )
    assert passing, message


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
def test_denoise_logits_forward_returns_full_canvas_logits(
    mesh_device,
    reset_seeds,
):
    torch.manual_seed(6)
    prompt_len, canvas_len, vocab_size = 64, 256, 256
    total_len = prompt_len + canvas_len
    hf_text_config = _create_hf_text_config(
        vocab_size=vocab_size,
        num_layers=1,
    )
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(
        mesh_device,
        hf_model,
        hf_text_config,
        num_layers=1,
        max_seq_len=total_len,
    )

    canvas_tokens = torch.randint(
        0,
        vocab_size,
        (1, canvas_len),
        dtype=torch.long,
    )
    previous_logits = torch.randn(1, canvas_len, vocab_size)
    self_conditioning_reference = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_reference.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_reference.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_reference.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_reference.down_proj.weight.data.clone(),
    }
    with torch.no_grad():
        canvas_hidden = hf_model.embed_tokens(canvas_tokens)
        conditioned_canvas_hidden = self_conditioning_reference.condition(
            canvas_hidden,
            previous_logits,
            hf_model.embed_tokens.weight,
        )
    prompt_tokens = torch.randint(
        0,
        vocab_size,
        (1, prompt_len),
        dtype=torch.long,
    )
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        neg_inf=NEGATIVE_INFINITY,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_denoise_logits_reference(
            hf_model,
            conditioned_canvas_hidden,
            [prompt_kv_hidden],
            mask,
        )
        golden_hidden = _torch_denoise_hidden_reference(
            hf_model,
            conditioned_canvas_hidden,
            [prompt_kv_hidden],
            mask,
        )

    tt_canvas_tokens = _to_device_tokens(mesh_device, canvas_tokens)
    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [
        read_prompt_kv_cache_slice(
            tt_model.tt_kv_cache[0],
            prompt_len=prompt_len,
        )
    ]
    tt_previous_logits = _to_device(
        mesh_device,
        previous_logits.unsqueeze(0),
    )
    tt_embedding_weight = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_logits = denoise_logits_from_tokens(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_tokens=tt_canvas_tokens,
        self_conditioning=self_conditioning,
        prev_logits=tt_previous_logits,
        self_conditioning_embedding_weight=tt_embedding_weight,
    )
    tt_canvas_hidden = embed_canvas_tokens(tt_model, tt_canvas_tokens)
    conditioned = self_conditioning.condition(
        tt_canvas_hidden,
        tt_previous_logits,
        tt_embedding_weight,
    )
    tt_canvas_hidden.deallocate(True)
    tt_hidden = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_hidden=conditioned,
    )
    logits = _to_torch(tt_logits, mesh_device).squeeze(0)
    hidden = _to_torch(tt_hidden, mesh_device).squeeze(0)
    for tt_key, tt_value in tt_prompt_kv_by_layer:
        tt_key.deallocate(True)
        tt_value.deallocate(True)

    _, hidden_pcc = comp_pcc(
        golden_hidden.float(),
        hidden.float(),
        pcc=0.0,
    )
    assert hidden_pcc >= 0.98
    passing, message = assert_with_pcc(
        golden.float(),
        logits.float(),
        0.98,
    )
    assert passing, message


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
@pytest.mark.parametrize(
    "enable_moe",
    [True, False],
    ids=["moe", "dense"],
)
def test_denoise_controller_real_logits_records_decision_flips(
    mesh_device,
    reset_seeds,
    enable_moe,
):
    torch.manual_seed(8)
    prompt_len, canvas_len, vocab_size, max_steps = 64, 256, 256, 2
    total_len = prompt_len + canvas_len
    hf_text_config = _create_hf_text_config(
        vocab_size=vocab_size,
        num_layers=1,
    )
    hf_text_config.enable_moe_block = enable_moe
    if enable_moe:
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(
        mesh_device,
        hf_model,
        hf_text_config,
        num_layers=1,
        max_seq_len=total_len,
    )

    prompt_tokens = torch.randint(
        0,
        vocab_size,
        (1, prompt_len),
        dtype=torch.long,
    )
    initial_canvas = torch.randint(
        0,
        vocab_size,
        (1, canvas_len),
        dtype=torch.long,
    )
    gumbel_noise = [torch.zeros(1, canvas_len, vocab_size) for _ in range(max_steps)]
    noise_tokens = [
        torch.randint(
            0,
            vocab_size,
            (1, canvas_len),
            dtype=torch.long,
        )
        for _ in range(max_steps)
    ]
    config = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=-1.0,
        stable_steps_to_halt=1,
        entropy_budget=0.1,
    )
    self_conditioning_reference = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_reference.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_reference.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_reference.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_reference.down_proj.weight.data.clone(),
    }
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        neg_inf=NEGATIVE_INFINITY,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    reference_logits_by_step = []

    class TorchLogitsAdapter:
        def __init__(self):
            self.previous_logits = None

        def __call__(self, canvas, step):
            with torch.no_grad():
                canvas_hidden = hf_model.embed_tokens(canvas)
                conditioned = self_conditioning_reference.condition(
                    canvas_hidden,
                    self.previous_logits,
                    hf_model.embed_tokens.weight,
                    enabled=self.previous_logits is not None,
                )
                logits = _torch_denoise_logits_reference(
                    hf_model,
                    conditioned,
                    [prompt_kv_hidden],
                    mask,
                )
                self.previous_logits = logits
                reference_logits_by_step.append(logits)
                return logits

    reference = reference_denoise_block(
        TorchLogitsAdapter(),
        initial_canvas,
        config,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [
        read_prompt_kv_cache_slice(
            tt_model.tt_kv_cache[0],
            prompt_len=prompt_len,
        )
    ]
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_embedding_weight = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    tt_adapter_base = DenoiseLogitsAdapter(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=tt_embedding_weight,
    )
    tt_logits_by_step = []

    def tt_adapter(canvas_tokens, step):
        logits = tt_adapter_base(canvas_tokens, step)
        tt_logits_by_step.append(_to_torch(logits, mesh_device).squeeze(0).float())
        return logits

    tt_result = denoise_block(
        tt_adapter,
        _to_device_canvas_ids(mesh_device, initial_canvas),
        config,
        gumbel_noise_fn=lambda step: _to_device(
            mesh_device,
            gumbel_noise[step].unsqueeze(0),
        ),
        noise_tokens_fn=lambda step: _to_device_canvas_ids(
            mesh_device,
            noise_tokens[step],
        ),
    )
    comparison = compare_trajectories(
        reference,
        tt_result,
        min_argmax_agreement=0.10,
        min_sampled_agreement=0.10,
        min_accept_iou=0.0,
        min_canvas_agreement=0.98,
        min_per_step_entropy_pcc=0.60,
        max_entropy_abs_err_threshold=0.50,
        committed_match_threshold=0.10,
        entropy_pcc_threshold=0.99,
    )
    accept_flips = [
        int((host.accept_mask != device.accept_mask).sum())
        for host, device in zip(
            reference.per_step,
            tt_result.per_step,
        )
    ]
    logits_pcc = [
        float(
            comp_pcc(
                reference_logits_by_step[index].float(),
                tt_logits_by_step[index].float(),
                pcc=0.0,
            )[1]
        )
        for index in range(max_steps)
    ]
    reference_top8_in_device = [
        float(
            (
                tt_logits_by_step[index].topk(k=8, dim=-1).indices
                == reference_logits_by_step[index].argmax(
                    dim=-1,
                    keepdim=True,
                )
            )
            .any(dim=-1)
            .float()
            .mean()
        )
        for index in range(max_steps)
    ]
    tt_adapter_base.reset()
    for tt_key, tt_value in tt_prompt_kv_by_layer:
        tt_key.deallocate(True)
        tt_value.deallocate(True)

    assert comparison.steps_match and comparison.halted_match
    assert comparison.passed, comparison
    assert reference.num_steps == tt_result.num_steps == max_steps
    assert not reference.halted and not tt_result.halted
    assert min(logits_pcc) >= (0.96 if enable_moe else 0.975)
    assert min(reference_top8_in_device) >= 0.80
    assert sum(accept_flips) <= (2 if enable_moe else 4)


def _run_canvas_sdpa(
    device,
    *,
    local_window=False,
    window_half=None,
    num_heads=8,
    num_kv_heads=8,
):
    torch.manual_seed(1234)
    prompt_len, canvas_len, head_dim = 256, 256, 256
    sequence_k = prompt_len + canvas_len
    query = torch.randn(1, num_heads, canvas_len, head_dim)
    key = torch.randn(1, num_kv_heads, sequence_k, head_dim)
    value = torch.randn(1, num_kv_heads, sequence_k, head_dim)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=local_window,
        window_half=window_half,
        neg_inf=NEGATIVE_INFINITY,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, sequence_k)

    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        key_reference = key.repeat_interleave(repeat, dim=1)
        value_reference = value.repeat_interleave(repeat, dim=1)
    else:
        key_reference, value_reference = key, value
    golden = torch.nn.functional.scaled_dot_product_attention(
        query,
        key_reference,
        value_reference,
        attn_mask=mask,
        is_causal=False,
    )

    tt_query = ttnn.from_torch(
        query,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_key = ttnn.from_torch(
        key,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_value = ttnn.from_torch(
        value,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_mask = ttnn.from_torch(
        mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        tt_query,
        tt_key,
        tt_value,
        attn_mask=tt_mask,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(tt_out)[:, :, :canvas_len, :]
    assert_with_pcc(golden, out, 0.99)


@_requires_device
@pytest.mark.use_module_device
def test_canvas_bidirectional(device):
    _run_canvas_sdpa(device)


@_requires_device
@pytest.mark.use_module_device
def test_gqa_bidirectional(device):
    _run_canvas_sdpa(device, num_heads=16, num_kv_heads=8)


@_requires_device
@pytest.mark.use_module_device
def test_sdpa_local_window_op(device):
    _run_canvas_sdpa(device, local_window=True, window_half=64)


CANVAS_LENGTH = 256


class _TinyGemma4Text(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = torch.nn.ModuleList([Gemma4TextDecoderLayer(config, layer_idx=0)])
        self.norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.lm_head = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )


def _tiny_attention_config(layer_type):
    layer_types = ["sliding_attention", "full_attention"] if layer_type == "sliding_attention" else ["full_attention"]
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=32,
        global_head_dim=32,
        layer_types=layer_types,
        sliding_window=1024,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_k_eq_v=False,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.0,
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        },
    )
    config._attn_implementation = "eager"
    return config


def _to_tt_state(hf_model):
    return {f"model.{key}": value for key, value in hf_model.state_dict().items()}


def _to_device_hidden(device, value):
    return ttnn.from_torch(
        value.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=_mesh_mapper(device),
    )


def _to_torch_hidden(device, value):
    is_mesh = hasattr(device, "shape") and device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(value)[0])
    return ttnn.to_torch(value)


def _torch_tiny_denoise_attention_reference(
    hf_model,
    layer_idx,
    prompt_hidden,
    canvas_hidden,
):
    config = hf_model.config
    layer_type = config.layer_types[layer_idx]
    attention = hf_model.layers[layer_idx].self_attn
    kv_hidden = torch.cat([prompt_hidden, canvas_hidden], dim=1)
    total_len = kv_hidden.shape[1]
    canvas_len = canvas_hidden.shape[1]
    rope = Gemma4TextRotaryEmbedding(config)
    cosine, sine = rope(
        kv_hidden,
        torch.arange(total_len).unsqueeze(0),
        layer_type=layer_type,
    )
    query_cosine = cosine[:, -canvas_len:, :]
    query_sine = sine[:, -canvas_len:, :]
    query_heads = attention.q_proj.out_features // attention.head_dim
    kv_heads = attention.k_proj.out_features // attention.head_dim
    query_shape = (
        *canvas_hidden.shape[:-1],
        query_heads,
        attention.head_dim,
    )
    kv_shape = (
        *kv_hidden.shape[:-1],
        kv_heads,
        attention.head_dim,
    )
    query = attention.q_norm(attention.q_proj(canvas_hidden).view(query_shape))
    query = apply_rotary_pos_emb(
        query,
        query_cosine,
        query_sine,
        unsqueeze_dim=2,
    ).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(kv_hidden).view(kv_shape))
    key = apply_rotary_pos_emb(
        key,
        cosine,
        sine,
        unsqueeze_dim=2,
    ).transpose(1, 2)
    value = attention.v_norm(attention.v_proj(kv_hidden).view(kv_shape)).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=False,
        scale=1.0,
    )
    out = out.transpose(1, 2).reshape(
        canvas_hidden.shape[0],
        canvas_len,
        config.hidden_size,
    )
    return attention.o_proj(out)


@_requires_device
@pytest.mark.use_module_device
@pytest.mark.parametrize(
    "layer_type",
    ["full_attention", "sliding_attention"],
)
def test_w2b_integrated_long_prompt_denoise_attention(
    device,
    layer_type,
):
    torch.manual_seed(47462)
    prompt_len = 33024
    total_len = prompt_len + CANVAS_LENGTH
    config = _tiny_attention_config(layer_type)
    hf_model = _TinyGemma4Text(config).eval()
    model_args = Gemma4ModelArgs.from_hf_config(config)
    model_args._hf_text_config = config
    tensor_parallel = device.shape[1] if hasattr(device, "shape") else 1
    mesh_config = MeshConfig(
        device.shape,
        decode=ModeConfig(tp=tensor_parallel),
    )
    tt_model = DiffusionGemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=_to_tt_state(hf_model),
        ccl_manager=(CCLManager(device, num_links=1) if tensor_parallel > 1 else None),
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=total_len,
        max_local_batch_size=1,
        num_layers=1,
        create_kv_cache=False,
    )

    prompt_hidden = torch.randn(1, prompt_len, config.hidden_size)
    canvas_hidden = torch.randn(
        1,
        CANVAS_LENGTH,
        config.hidden_size,
    )
    with torch.no_grad():
        golden = _torch_tiny_denoise_attention_reference(
            hf_model,
            0,
            prompt_hidden,
            canvas_hidden,
        )
    tt_prompt_hidden = _to_device_hidden(device, prompt_hidden)
    tt_canvas_hidden = _to_device_hidden(device, canvas_hidden)
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=0,
        prompt_hidden=tt_prompt_hidden,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch_hidden(device, tt_out).squeeze(0)

    assert_with_pcc(golden, out, 0.99)
    tt_prompt_hidden.deallocate(True)
    tt_canvas_hidden.deallocate(True)
    tt_out.deallocate(True)
