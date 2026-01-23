# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
    num_layers,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            num_layers=num_layers,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[
        0
    ].tokenizer  # TODO Should we support Data Parallel different models? If so, we need to support multiple tokenizers
    processor = model_args[0].processor
    return model_args, model, page_table, tt_kv_cache, tokenizer, processor


@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
def test_demo_text(
    optimizations,
    mesh_device,
):
    """
    Simple demo with limited dependence on reference code.
    """
    instruct = True
    max_seq_len = 1024
    data_parallel = 1
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    num_layers = None

    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    global_batch_size = 1  # input batch_size is interpreted as size per DP group

    # uneven split of devices per DP group not supported
    if data_parallel > num_devices or num_devices % data_parallel != 0:
        pytest.skip(f"Invalid number of DP groups: {data_parallel}, for {num_devices} devices")

    model_args, model, page_table, tt_kv_cache, tokenizer, processor = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        num_layers=num_layers,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    repeat_batch_prompts = [torch.randint(low=10, high=20, size=(1, 120), dtype=torch.int32)] * 2

    all_gather_results = []

    all_gather_input_tensor = ttnn.from_torch(
        torch.ones(1, 1, 32, 256),
        layout=ttnn.Layout.TILE,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    # When we call this all_gather before the prefill, the outputs of the following two all_gathers will be the same
    if "PERFORM_INITIAL_ALL_GATHER" in os.environ:
        ttnn.all_gather(
            all_gather_input_tensor,
            dim=3,
            num_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )

    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        if batch_idx != 0:
            for i in range(len(model)):
                for layer in model[i].layers:
                    k_cache, v_cache = layer.attention.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
            generator.prev_page_table = None

        generator.prefill_forward_text(
            input_prompts,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([120] * 1),
            enable_trace=False if "DISABLE_PREFILL_TRACE" in os.environ else True,
        )

        if batch_idx == 1:
            if "CLEAR_PROGRAM_CACHE" in os.environ:
                mesh_device.clear_program_cache()

        ttnn.synchronize_device(mesh_device)

        all_gather_results.append(
            ttnn.all_gather(
                all_gather_input_tensor,
                dim=3,
                num_links=1,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                cluster_axis=None,
                topology=ttnn.Topology.Linear,
            ).cpu(blocking=True)
        )

        ttnn.synchronize_device(mesh_device)

    first_output = all_gather_results[0]
    second_output = all_gather_results[1]

    # Convert ttnn tensors to torch tensors for comparison
    # Need mesh_composer since tensor is distributed on mesh device (sharded on dim=3)
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    first_output_torch = ttnn.to_torch(first_output, mesh_composer=mesh_composer)
    second_output_torch = ttnn.to_torch(second_output, mesh_composer=mesh_composer)

    # Compare using torch.allclose
    assert torch.allclose(first_output_torch, second_output_torch), "Outputs do not match"
