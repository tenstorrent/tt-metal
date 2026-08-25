# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical chunked-prefill, 256K RoPE, and reveal-mask golden gates."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.experimental.diffusion_gemma.reference.attention_mask import (
    build_canvas_denoise_mask,
    build_canvas_reveal_denoise_mask,
)
from models.experimental.diffusion_gemma.tt import chunked_prefill as cp
from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.utils_for_testing import assert_with_pcc

CHUNK_SIZE = 256
BLOCK_SIZE = 64
HIDDEN = 128
HEAD_DIM = 32
VOCAB = 256

requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run chunked-prefill correctness on a Tenstorrent device",
)


def _tiny_config(sliding_window):
    layer_types = ["sliding_attention", "full_attention"]
    config = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=HEAD_DIM,
        global_head_dim=HEAD_DIM,
        layer_types=layer_types,
        sliding_window=sliding_window,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_k_eq_v=False,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    )
    config._attn_implementation = "eager"
    return config


def _to_tt_state(config):
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextScaledWordEmbedding,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
                embed_scale=config.hidden_size**0.5,
            )
            self.layers = torch.nn.ModuleList(
                [Gemma4TextDecoderLayer(config, layer_idx=index) for index in range(config.num_hidden_layers)]
            )
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    model = _Tiny().eval()
    return {f"model.{key}": value for key, value in model.state_dict().items()}


def _alloc_caches(mesh_device, model, prompt_len, *, paged):
    config = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=prompt_len // BLOCK_SIZE) if paged else None
    return [
        init_kv_cache(
            mesh_device=mesh_device,
            config=layer.self_attn.config,
            max_batch_size=1,
            max_seq_len=prompt_len,
            paged_attention_config=config,
        )
        for layer in model.layers
    ]


def _last_token_logits(tt_logits, row):
    value = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    if value.dim() == 4:
        value = value.squeeze(0).squeeze(0)
    elif value.dim() == 3:
        value = value.squeeze(0)
    return value[row, :VOCAB]


def _chunked_vs_single_pcc(device, prompt_len, sliding_window):
    torch.manual_seed(47466)
    tp = device.shape[1] if hasattr(device, "shape") else 1
    config = _tiny_config(sliding_window)
    model_args = Gemma4ModelArgs.from_hf_config(config)
    model_args._hf_text_config = config
    mesh_config = MeshConfig(device.shape, decode=ModeConfig(tp=tp)) if hasattr(device, "shape") else None

    state = _to_tt_state(config)
    model = DiffusionGemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=state,
        ccl_manager=CCLManager(device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=prompt_len,
        max_local_batch_size=1,
        num_layers=config.num_hidden_layers,
        create_kv_cache=False,
    )

    input_ids = torch.randint(0, VOCAB, (1, prompt_len), dtype=torch.int64)
    replicate = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "shape") else None
    tokens_tt = ttnn.from_torch(
        input_ids,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate,
    )
    embeds = model.embed_tokens(tokens_tt)
    embeds_single = ttnn.to_layout(
        ttnn.reshape(embeds, (1, 1, prompt_len, HIDDEN)),
        ttnn.TILE_LAYOUT,
    )

    baseline_cache = _alloc_caches(device, model, prompt_len, paged=False)
    logits_single = model(
        embeds_single,
        is_decode=False,
        page_table=None,
        kv_caches=baseline_cache,
        input_ids_torch=input_ids,
        get_last_token=-1,
        batch_size=1,
    )
    single_last = _last_token_logits(logits_single, prompt_len - 1)
    logits_single.deallocate(True)

    paged_cache = _alloc_caches(device, model, prompt_len, paged=True)
    page_table_torch = cp.make_reference_page_table(prompt_len // BLOCK_SIZE, mesh_device=device)
    embeds = model.embed_tokens(tokens_tt)
    embeds_chunked = ttnn.to_layout(
        ttnn.reshape(embeds, (1, 1, prompt_len, HIDDEN)),
        ttnn.TILE_LAYOUT,
    )
    logits_chunked = cp.chunked_prefill(
        model,
        embeds_chunked,
        input_ids_torch=input_ids,
        embeds_torch=None,
        kv_cache=paged_cache,
        page_table_torch=page_table_torch,
        block_size=BLOCK_SIZE,
        chunk_size=CHUNK_SIZE,
    )
    chunked_last = _last_token_logits(logits_chunked, CHUNK_SIZE - 1)
    logits_chunked.deallocate(True)
    embeds_chunked.deallocate(True)

    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(single_last, chunked_last, pcc=0.999)
    return single_last, chunked_last, pcc


@requires_device
@pytest.mark.use_module_device
def test_chunked_prefill_matches_single(device):
    prompt_len, sliding_window = 512, 1024
    single_last, chunked_last, _ = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    assert_with_pcc(single_last, chunked_last, 0.999)


@requires_device
@pytest.mark.use_module_device
def test_chunked_prefill_sliding_past_window(device):
    prompt_len, sliding_window = 2048, 1024
    assert prompt_len > sliding_window
    single_last, chunked_last, _ = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    assert_with_pcc(single_last, chunked_last, 0.999)


ROPE_CACHE_LEN = 262144


def _rope_cache_model(cache_len):
    return SimpleNamespace(
        hf_config=SimpleNamespace(layer_types=["sliding_attention"]),
        rope_caches={
            "sliding_attention": (
                torch.zeros(1, 1, cache_len, 8),
                torch.zeros(1, 1, cache_len, 8),
            )
        },
    )


def test_get_rope_mats_reaches_256k():
    model = _rope_cache_model(ROPE_CACHE_LEN)
    cos, sin = DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=ROPE_CACHE_LEN)
    assert cos.shape[-2] == ROPE_CACHE_LEN
    assert sin.shape[-2] == ROPE_CACHE_LEN


CANVAS = 256


def _committed_columns(prompt_len: int, p_max: int, canvas_len: int) -> torch.Tensor:
    prefix = torch.arange(prompt_len)
    canvas = p_max + torch.arange(canvas_len)
    return torch.cat([prefix, canvas])


def test_no_leak_uncommitted_prefix_always_masked():
    p_max = 8192
    for prompt_len in (0, 4096):
        for layer_type, enforce_window in (("full_attention", False), ("sliding_attention", True)):
            mask = build_canvas_reveal_denoise_mask(
                prompt_len,
                CANVAS,
                p_max,
                layer_type=layer_type,
                sliding_window=1024,
                enforce_sliding_window=enforce_window,
            )
            assert tuple(mask.shape) == (CANVAS, p_max + CANVAS)
            uncommitted = mask[:, prompt_len:p_max]
            assert torch.isinf(uncommitted).all() and (uncommitted < 0).all()


def test_phase1_full_attn_bit_exact_to_allattend_golden():
    p_max = 8192
    for prompt_len in (32, 2048):
        reveal = build_canvas_reveal_denoise_mask(prompt_len, CANVAS, p_max, layer_type="full_attention")
        golden = build_canvas_denoise_mask(prompt_len, CANVAS, layer_type="full_attention")
        columns = _committed_columns(prompt_len, p_max, CANVAS)
        assert torch.equal(reveal[:, columns], golden)


def test_phase2_sliding_matches_golden_on_committed_span():
    p_max = 8192
    sliding_window = 1024
    for prompt_len in (1024, 4096):
        reveal = build_canvas_reveal_denoise_mask(
            prompt_len,
            CANVAS,
            p_max,
            layer_type="sliding_attention",
            sliding_window=sliding_window,
            enforce_sliding_window=True,
        )
        golden = build_canvas_denoise_mask(
            prompt_len,
            CANVAS,
            layer_type="sliding_attention",
            sliding_window=sliding_window,
        )
        columns = _committed_columns(prompt_len, p_max, CANVAS)
        assert torch.equal(torch.isfinite(reveal[:, columns]), torch.isfinite(golden))
