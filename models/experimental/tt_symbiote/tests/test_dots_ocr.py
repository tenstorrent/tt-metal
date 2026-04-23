# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for dots.ocr model on N300 (1x2 Wormhole mesh) with tensor-parallel MLP.

Phase 2: Tensor-parallel MLP using column-parallel gate/up + row-parallel down.
MLP weights are sharded across 2 devices, residual stream stays replicated.
Attention and normalization remain non-distributed (replicated weights).
"""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import (
    DispatchManager,
    DistributedConfig,
    DistributedTensorConfig,
    TracedRun,
)
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
from models.experimental.tt_symbiote.modules.attention import (
    LlamaAttention,
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm
from models.experimental.tt_symbiote.utils.device_management import (
    DeviceInit,
    set_device,
)
from models.experimental.tt_symbiote.utils.module_replacement import (
    register_module_replacement_dict,
)


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

# Load from local HF cache to avoid Python import failure caused by the dot
# in "dots.ocr" — transformers' dynamic module loader creates a module path
# where Python interprets the dot as a package separator.
DOTS_OCR_LOCAL_PATH = "/home/salnahari/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def create_paged_kv_cache(model_config, device, batch_size=1):
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


class ReplicatedDeviceInit(DeviceInit):
    """DeviceInit that uses replication (not sharding) for all tensors.

    The default DeviceInit creates a DistributedConfig whose tensor_config uses
    ShardTensor2dMesh for mapping and ConcatMesh2dToTensor for composing.
    When all data is replicated (Phase 1), the sharding composer produces wrong
    shapes on to_torch (doubles the hidden dim or batch dim).

    This subclass forces:
    - mesh_mapper: ReplicateTensorToMesh (send same data to all devices)
    - mesh_composer: single-device composer via MeshComposerConfig with
      mesh_shape_override=MeshShape(1,1), so to_torch reads from device 0
      and preserves the original tensor shape.
    """

    DEVICE_TO_STATE_DICT = {}

    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        num_devices = device.get_num_devices()
        if num_devices <= 1:
            return DistributedConfig(device)

        single_device_composer = ttnn.create_mesh_composer(
            device,
            ttnn.MeshComposerConfig(
                dims=[0, 0],
                mesh_shape_override=ttnn.MeshShape(1, 1),
            ),
        )

        tensor_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            mesh_composer=single_device_composer,
            logical_shape_fn=lambda shape: shape,
        )

        config = object.__new__(DistributedConfig)
        config.mesh_device = device
        config.tensor_config = tensor_config
        config.ccl_manager = None
        return config


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_dots_ocr_n300(mesh_device):
    """Test dots.ocr on N300 (1x2 mesh) with tensor-parallel MLP."""
    model_name = DOTS_OCR_LOCAL_PATH

    print("Loading dots.ocr from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Pass 1: Replace MLP with tensor-parallel TTNNDotsOCRMLP
    nn_to_ttnn_mlp = {
        model.model.layers[0].mlp.__class__: TTNNDotsOCRMLP,
    }
    modules_mlp = register_module_replacement_dict(model, nn_to_ttnn_mlp, model_config=None)

    # Pass 2: Replace attention and normalization (nn-to-ttnn)
    nn_to_ttnn = {
        model.model.layers[0].self_attn.__class__: LlamaAttention,
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    }
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Pass 3: Replace remaining Linear layers (lm_head, etc.) with replicated TTNNLinear
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinear,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)

    # Prepare text-only input (no vision)
    messages = [
        {
            "role": "user",
            "content": "What is optical character recognition and how does it work?",
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

    type(model).device = property(lambda self: torch.device("cpu"))

    set_device(model, mesh_device, device_init=ReplicatedDeviceInit)

    all_modules = {**modules_mlp, **modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference with mesh device...")
    model.eval()
    torch.set_grad_enabled(False)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=kv_cache)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True, past_key_values=kv_cache)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=kv_cache)
    ttnn.synchronize_device(mesh_device)

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"dots.ocr N300 OUTPUT: {decoded}")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_n300_timing_stats.csv")
    TracedRun.release_all()
