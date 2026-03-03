# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import (
    DispatchManager,
    NormalRun,
    compose_transforms,
    no_dispatch,
    post_process_ttnn_module_output,
    set_device_wrap,
    to_ttnn_wrap,
    unwrap_to_torch,
    wrap_from_torch,
    wrap_to_torch_ttnn_tensor,
)
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
import transformers
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.moe import TTNNMoE
from models.experimental.tt_symbiote.modules.attention import (
    TTNNGlm4MoeLiteAttention,
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn
from models.experimental.tt_symbiote.core.torch_dispatcher import can_dispatch_to_torch, dispatch_to_torch

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


def create_paged_kv_cache(model_config, device, batch_size=1):
    config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=batch_size)
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.qk_head_dim,
        config=config,
        device=None,
    ).to_device(device)


@torch.no_grad()
def greedy_decode(model, input_ids, attention_mask, kv_cache, max_new_tokens, eos_token_id, mesh_device):
    out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=kv_cache, use_cache=True)
    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)
        if next_token.item() == eos_token_id:
            break

    ttnn.synchronize_device(mesh_device)
    return torch.cat([input_ids] + generated, dim=-1)


def _fast_module_run(self, *args, **kwds):
    transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
    func_args = tree_map(transform, args)
    other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
    func_kwargs = tree_map(transform, other_kwargs)
    func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
    self.preprocess_weights()
    self.move_weights_to_device()
    DispatchManager.set_current_module_name(self.module_name)
    result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
    DispatchManager.set_current_module_name(None)
    return result


def _fast_torch_dispatch(cls, func, types, args=(), kwargs=None):
    if can_dispatch_to_ttnn(func.name(), args, kwargs):
        return dispatch_to_ttnn(func.name(), args, kwargs)
    with no_dispatch():
        func_args = tree_map(unwrap_to_torch(func), args)
        func_kwargs = tree_map(unwrap_to_torch(func), kwargs)
        if can_dispatch_to_torch(func.name(), func_args, func_kwargs):
            func_res = dispatch_to_torch(func.name(), func_args, func_kwargs)
        else:
            func_res = func(*func_args, **func_kwargs)
        return tree_map(wrap_from_torch, func_res)


MESH_DEVICE_MAP = {
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
}


def _get_device_params():
    params = {"trace_region_size": 50000000, "num_command_queues": 2}
    num_devices = len(ttnn.get_device_ids())
    if num_devices > 1:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    return params


@pytest.mark.parametrize(
    "device_params",
    [_get_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128tok"])
def test_glm(mesh_device, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash")
    model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash", torch_dtype=torch.bfloat16)

    nn_to_ttnn = {
        model.model.layers[0].self_attn.__class__: TTNNGlm4MoeLiteAttention,
        model.model.layers[1].mlp.__class__: TTNNMoE,
    }
    nn_to_ttnn2 = {nn.Linear: TTNNLinearIColShardedWRowSharded}

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

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}
    for _, v in tqdm(all_modules.items(), desc="Preprocessing weights"):
        v.preprocess_weights()
        v.move_weights_to_device()

    model.eval()

    warmup_cache = create_paged_kv_cache(model.config, mesh_device)
    greedy_decode(
        model, inputs["input_ids"], inputs["attention_mask"], warmup_cache, 5, tokenizer.eos_token_id, mesh_device
    )
    DispatchManager.clear_timings()

    _orig_module_run = NormalRun.module_run
    _orig_torch_dispatch = NormalRun.torch_dispatch
    _orig_record_timing = DispatchManager.record_timing

    NormalRun.module_run = staticmethod(_fast_module_run)
    NormalRun.torch_dispatch = staticmethod(_fast_torch_dispatch)
    DispatchManager.record_timing = staticmethod(lambda *a, **kw: None)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    start_time = time.perf_counter()
    outputs = greedy_decode(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        kv_cache,
        max_new_tokens,
        tokenizer.eos_token_id,
        mesh_device,
    )
    end_time = time.perf_counter()

    NormalRun.module_run = _orig_module_run
    NormalRun.torch_dispatch = _orig_torch_dispatch
    DispatchManager.record_timing = _orig_record_timing

    total_time = end_time - start_time
    generated_tokens = outputs.shape[-1] - prompt_tokens
    tokens_per_second = generated_tokens / total_time

    decoded_output = tokenizer.decode(outputs[0][prompt_tokens:])
    print(f"\n{'='*80}")
    print(f"GLM OUTPUT: {decoded_output}")
    print(f"{'='*80}")
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Prompt tokens:        {prompt_tokens}")
    print(f"  Generated tokens:     {generated_tokens}")
    print(f"  Total time:           {total_time:.3f}s")
    print(f"  Throughput:           {tokens_per_second:.2f} tokens/s")
    print(f"  Time per token:       {(total_time / generated_tokens * 1000):.2f}ms")
    print(f"{'='*80}\n")
    DispatchManager.save_stats_to_file("glm_timing_stats.csv")
    TracedRun.release_all()
