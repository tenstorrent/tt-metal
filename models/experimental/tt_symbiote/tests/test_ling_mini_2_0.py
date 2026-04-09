# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Ling-mini-2.0 with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model


def fast_greedy_generate(model, input_ids, attention_mask, max_new_tokens, past_key_values, mesh_device):
    """Greedy decode that avoids the __torch_dispatch__ overhead on logits.

    model.generate() routes every torch op on the logits TorchTTNNTensor through
    __torch_dispatch__ → dispatch_to_torch_wrapper → _unwrap_to_torch.  With a
    ~160k-token vocabulary, the accumulated Python-overhead of those dozens of
    dispatch calls per step × N steps dominates total runtime.

    This loop instead accesses outputs.logits.ttnn_tensor directly and calls
    to_torch_auto_compose exactly once per step, then does CPU argmax on the
    resulting small (trimmed) tensor.
    """
    torch_device = torch.device("cpu")
    vocab_size = int(getattr(model.config, "vocab_size", 0)) or 0

    eos_token_ids: set[int] = set()
    for src in (
        model.config.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            eos_token_ids.update(int(x) for x in src)
        else:
            eos_token_ids.add(int(src))

    model_kwargs: dict = {
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    model_kwargs = model._get_initial_cache_position(input_ids.shape[-1], torch_device, model_kwargs)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(generated, **model_kwargs)
        with torch.no_grad():
            outputs = model(**model_inputs, return_dict=True)

        # One device→host transfer per step — bypass __torch_dispatch__ entirely.
        tt_logits = getattr(outputs.logits, "ttnn_tensor", None)
        if tt_logits is not None:
            logits_cpu = to_torch_auto_compose(tt_logits, device=mesh_device).float()
            # Shape may be [1,1,V] or [1,V]; normalise to 2-D [batch, V].
            logits_last = logits_cpu.reshape(-1, logits_cpu.shape[-1])[-1:, :]
            if vocab_size:
                logits_last = logits_last[:, :vocab_size]
            next_token = int(torch.argmax(logits_last, dim=-1).item())
        else:
            logits_last = outputs.logits[:, -1, :]
            if vocab_size:
                logits_last = logits_last[:, :vocab_size]
            next_token = int(torch.argmax(logits_last, dim=-1).item())

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=getattr(model.config, "is_encoder_decoder", False),
        )

        generated = torch.cat(
            [generated, torch.tensor([[next_token]], dtype=input_ids.dtype, device=torch_device)],
            dim=-1,
        )

        if next_token in eos_token_ids:
            break

    return generated


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create a paged attention KV cache for Ling-mini-2.0.

    Args:
        model_config: Model configuration
        device: TTNN device
        batch_size: Batch size

    Returns:
        TTNNPagedAttentionKVCache instance
    """
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        config=config,
        device=None,
    ).to_device(device)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_ling_mini_2_0(mesh_device):
    """Test Ling-mini-2.0 model with TTNN acceleration."""

    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
        nn.Embedding: TTNNBailingPaddedEmbedding,
        model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }
    nn_to_ttnn_3 = {
        model.model.__class__: TTNNBailingMoeV2Model,
    }
    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    modules3 = register_module_replacement_dict(model, nn_to_ttnn_3, model_config=None)
    # After replacing all nn.Modules with TTNNModules, HF's model.device
    # (which calls next(self.parameters())) fails since no nn.Module params remain.
    # Patch it to return cpu — HF uses this for placing generated token tensors.
    type(model).device = property(lambda self: torch.device("cpu"))
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2, **modules3}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    print("Running inference with paged attention...")
    model.eval()
    torch.set_grad_enabled(False)

    # Warmup run without trace
    outputs = fast_greedy_generate(
        model, inputs["input_ids"], inputs.get("attention_mask"), 2, paged_cache, mesh_device
    )
    paged_cache.reset()
    # Actual run with trace
    outputs = fast_greedy_generate(
        model, inputs["input_ids"], inputs.get("attention_mask"), 4, paged_cache, mesh_device
    )
    paged_cache.reset()

    DispatchManager.clear_timings()
    outputs = fast_greedy_generate(
        model, inputs["input_ids"], inputs.get("attention_mask"), 128, paged_cache, mesh_device
    )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Ling-mini-2.0 PAGED ATTENTION OUTPUT: {decoded}")

    # Verify output is coherent (non-empty generated text)
    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
    TracedRun.release_all()
