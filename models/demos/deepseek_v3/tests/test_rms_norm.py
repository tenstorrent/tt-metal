# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3RMSNorm
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.cache import InMemoryCacheStorage, TensorCache
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import assert_hidden_dim_pcc, get_model_config, run_module_forward
from models.demos.deepseek_v3.utils.weight_spec import WeightSpecContext, create_weight_config_from_weight_spec


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
    ]
    + [
        ("prefill", seq_len)
        if seq_len == 128
        else pytest.param(
            "prefill",
            seq_len,
            marks=pytest.mark.skip(
                f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
            ),
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
@pytest.mark.parametrize(
    "reference_layernorm_path, RMSNormClass, hf_config_size_attr",
    [
        (None, DistributedRMSNorm, "hidden_size"),
        ("model.layers.0.input_layernorm", DistributedRMSNorm, "hidden_size"),
        ("model.layers.0.post_attention_layernorm", DistributedRMSNorm, "hidden_size"),
        (None, RMSNorm, "kv_lora_rank"),  # TODO: not properly tested here, needs fixing
        (None, RMSNorm, "q_lora_rank"),  # TODO: not properly tested here, needs fixing
        (
            "model.layers.0.self_attn.kv_a_layernorm",
            RMSNorm,
            "kv_lora_rank",
        ),  # TODO: not properly tested here, needs fixing
        (
            "model.layers.0.self_attn.q_a_layernorm",
            RMSNorm,
            "q_lora_rank",
        ),  # TODO: not properly tested here, needs fixing
    ],
)
def test_forward_pass(
    RMSNormClass,
    hf_config_size_attr,
    mode,
    seq_len,
    reference_layernorm_path,
    model_path,
    hf_config,
    mesh_device,
    ccl,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    num_module_layers, _ = mesh_device.shape
    hidden_size = getattr(hf_config, hf_config_size_attr)

    reference_model = DeepseekV3RMSNorm(
        hidden_size=hidden_size,
        eps=hf_config.rms_norm_eps,
    ).eval()

    # If we don't have a path to the reference weights we use random weights
    if reference_layernorm_path is not None:
        state_dict_for_cache = state_dict
        prefix = reference_layernorm_path
        layer_state = sub_state_dict(state_dict, reference_layernorm_path + ".")
        reference_model.load_state_dict({k: v.to(torch.float32) for k, v in layer_state.items()})
        layer_state = {k: v.to(torch.bfloat16) for k, v in layer_state.items()}
    else:
        layer_state = reference_model.to(torch.bfloat16).state_dict()
        state_dict_for_cache = {"layernorm.weight": layer_state["weight"]}
        prefix = "layernorm"

    torch_input = torch.randn(num_module_layers, 1, seq_len, hidden_size)
    reference_model = reference_model.to(torch.float32)
    reference_output = reference_model(torch_input)

    cache_storage = InMemoryCacheStorage()
    cache = TensorCache(state_dict_for_cache, hf_config.to_dict(), cache_storage)
    context = WeightSpecContext(resolver=lambda key: state_dict_for_cache[key])
    weight_spec = RMSNormClass.create_weight_spec(hf_config, mesh_device.shape, context.with_prefix(prefix))
    weight_config_inner = create_weight_config_from_weight_spec(weight_spec, prefix, cache, device=mesh_device)

    if RMSNormClass is DistributedRMSNorm:
        weight_config = {"rms_norm_post_all_gather": weight_config_inner}
    else:
        weight_config = weight_config_inner
    model_config = get_model_config(RMSNormClass, mode, hf_config, mesh_device)
    model_state = RMSNormClass.create_state(
        hf_config, mesh_device, *[ccl for _ in range(1) if RMSNormClass is DistributedRMSNorm]
    )
    run_config = create_run_config(model_config, weight_config, model_state)

    input_tensor_memory_config = (
        run_config["input_memory_config"] if RMSNormClass is DistributedRMSNorm else ttnn.L1_MEMORY_CONFIG
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_device.shape, dims=(0, -1 if RMSNormClass is DistributedRMSNorm else None)
        ),
        dtype=ttnn.bfloat16,
        memory_config=input_tensor_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = run_module_forward(RMSNormClass, mode, tt_input, run_config)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, dims=(0, -1)),
    )
    if RMSNormClass is RMSNorm:
        tt_output_torch = tt_output_torch[..., :hidden_size]

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.999)

    cache.summary()


if __name__ == "__main__":
    pytest.main([__file__])
