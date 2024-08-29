# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from transformers import FalconForCausalLM
from models.utility_functions import tt_tensors_to_torch_tensors


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device_mesh):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        tt_k_cache = tt_from_torch(
            k_cache,
            dtype=ttnn.bfloat16,
            device=device_mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        tt_v_cache = tt_from_torch(
            v_cache,
            dtype=ttnn.bfloat16,
            device=device_mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        kv_cache += ((tt_k_cache, tt_v_cache),)
    return kv_cache


def load_hf_model(model_location_generator, model_version):
    model_name = model_location_generator(model_version, model_subdir="Falcon")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict


def create_prefill_attn_mask_for_sharded_softmax(attention_mask, num_attn_heads, seq_len):
    if seq_len == 2048:
        num_slices = 16
    elif seq_len == 128:
        num_slices = 1
    elif seq_len == 1024:
        num_slices = 4
    else:
        raise ValueError("Unsupported seq_len for optimizations")

    attn_masks_per_slice = []
    attention_mask_starting_index_per_slice = 0
    slice_length = (num_attn_heads * seq_len) // num_slices
    number_of_attention_mask_elements_used_per_slice = slice_length - seq_len * (slice_length // seq_len)
    for _ in range(num_slices):
        torch_attn_mask_per_slice = torch.cat(
            [
                attention_mask[:, :, attention_mask_starting_index_per_slice:, :],
                attention_mask[:, :, :attention_mask_starting_index_per_slice, :],
            ],
            dim=2,
        )
        attn_masks_per_slice.append(torch_attn_mask_per_slice)

        attention_mask_starting_index_per_slice = (
            attention_mask_starting_index_per_slice + number_of_attention_mask_elements_used_per_slice
        ) % seq_len

    return attn_masks_per_slice


def get_rand_falcon_inputs(
    llm_mode,
    seq_len,
    batch,
    kv_cache_len,
    device_mesh,
    global_batch,
    head_dim,
    max_position_embeddings,
    configuration,
    model_config,
    num_layers=1,
    generate_attention_inputs=True,
):
    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    tt_attention_input = []
    tt_attention_mask = []
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        # assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        # Generate attention input and mask
        if generate_attention_inputs:
            attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
            attention_mask_bool = torch.ones(global_batch, 1, q_len, kv_len, dtype=bool).triu(diagonal=1)
            tt_attention_input = tt_from_torch(
                attention_input.unsqueeze(1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
            )

            if model_config["PREFILL_OPTIMIZED_MODE"] and seq_len in [2048, 128, 1024]:
                attn_masks = create_prefill_attn_mask_for_sharded_softmax(
                    (attention_mask_bool * -1e5),
                    configuration.num_attention_heads,
                    q_len,
                )
                tt_attention_mask = [
                    tt_from_torch(
                        attn_mask,
                        dtype=ttnn.bfloat4_b,
                        device=device_mesh,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
                    )
                    for attn_mask in attn_masks
                ]
            else:
                tt_attention_mask = tt_from_torch(
                    (attention_mask_bool * -100000).expand(-1, configuration.num_attention_heads, -1, -1),
                    dtype=model_config["DEFAULT_DTYPE"],
                    device=device_mesh,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
                )

        # Generate kvcache for each layer
        layer_past = None
        tt_layer_past = ()
        for layer in range(num_layers):
            tt_k_cache = torch.zeros(global_batch, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(global_batch, max_position_embeddings, head_dim)
            tt_k_cache = tt_from_torch(
                tt_k_cache.unsqueeze(1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
            )
            tt_v_cache = tt_from_torch(
                tt_v_cache.unsqueeze(1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
            )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        # Generate attention input and mask
        if generate_attention_inputs:
            attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
            tt_attention_input = tt_from_torch(
                attention_input.unsqueeze(1).transpose(0, 2),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=2),
            )

            attention_mask_bool = torch.zeros(global_batch, 1, q_len, kv_len, dtype=bool)
            kv_len_padded = (kv_len + 31) // 32 * 32
            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(global_batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
                ),
                dim=-1,
            )
            if model_config["l1_sharded"] == False:
                attention_mask_bool_padded = (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                    -1,
                    configuration.num_attention_heads,
                    -1,
                    -1,
                    # -1, 71, -1, -1
                )
                device_shard_dim = 2
            else:
                # Reshape width to tile-size since that is required by scale_mask_softmax_in_place with causal_mask=False (in falcon_attention.py)
                attention_mask_bool_padded = attention_mask_bool_padded.reshape(global_batch, 1, -1, 32) * -100000
                device_shard_dim = 0
            tt_attention_mask = tt_from_torch(
                attention_mask_bool_padded,
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=model_config["ATTN_MASK_MEMCFG"],
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=device_shard_dim),
            )
            if not model_config["l1_sharded"]:
                # Tilize attn masks
                tt_attention_mask = ttnn.tilize(
                    tt_attention_mask,
                    memory_config=model_config["ATTN_MASK_MEMCFG"],
                    dtype=model_config["ATTN_MASK_DTYPE"],
                )

        # Generate kvcache for each layer
        layer_past = ()
        tt_layer_past = ()
        for layer in range(num_layers):
            k_cache = torch.rand(global_batch, kv_cache_len, head_dim)
            v_cache = torch.rand(global_batch, kv_cache_len, head_dim)
            layer_past += ((k_cache.unsqueeze(1), v_cache.unsqueeze(1)),)

            tt_k_cache = torch.zeros(global_batch, max_position_embeddings, head_dim)
            tt_v_cache = torch.zeros(global_batch, max_position_embeddings, head_dim)
            tt_k_cache[:, :kv_cache_len, :] = k_cache
            tt_v_cache[:, :kv_cache_len, :] = v_cache
            tt_k_cache = tt_from_torch(
                tt_k_cache.unsqueeze(1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
            )
            tt_v_cache = tt_from_torch(
                tt_v_cache.unsqueeze(1),
                dtype=model_config["DEFAULT_DTYPE"],
                device=device_mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ShardTensorToMesh(device_mesh, dim=0),
            )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    if not generate_attention_inputs:
        return (layer_past, tt_layer_past, kv_len)
    else:
        return (
            attention_input,
            attention_mask_bool,
            layer_past,
            tt_attention_input,
            tt_attention_mask,
            tt_layer_past,
            kv_len,
        )


def concat_device_out_layer_present(device_mesh, tt_layer_present, seq_end_idx, end_idx_only=False):
    tt_layer_present = (
        tt_tensors_to_torch_tensors(tt_layer_present[0], device_mesh, concat_dim=0).squeeze(1),
        tt_tensors_to_torch_tensors(tt_layer_present[1], device_mesh, concat_dim=0).squeeze(1),
    )
    if not end_idx_only:
        tt_layer_present = (
            tt_layer_present[0][:, :seq_end_idx, :],
            tt_layer_present[1][:, :seq_end_idx, :],
        )
    else:
        tt_layer_present = (
            tt_layer_present[0][:, seq_end_idx, :],
            tt_layer_present[1][:, seq_end_idx, :],
        )
    return tt_layer_present


def concat_device_outputs(device_mesh, tt_out, llm_mode, tt_layer_present, seq_end_idx):
    concat_dim = 2 if llm_mode == "decode" else 0
    tt_out = tt_tensors_to_torch_tensors(tt_out, device_mesh, concat_dim=concat_dim).squeeze(1)
    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)
    tt_layer_present = concat_device_out_layer_present(device_mesh, tt_layer_present, seq_end_idx)
    return tt_out, tt_layer_present


def get_devices(device):
    # device is either a ttnn.DeviceMesh or a ttnn.Device
    if type(device) == ttnn.DeviceMesh:
        devices = device.get_devices()
    elif type(device) == ttnn.Device:
        devices = [device]
    else:
        raise ValueError(f"Unrecognized device type {type(device)}")
    return devices


def synchronize_devices(device):
    # device is either a ttnn.DeviceMesh or a ttnn.Device
    devices = get_devices(device)
    for device in devices:
        ttnn.synchronize_device(device)


def tt_from_torch(torch_tensor, dtype=None, device=None, layout=None, memory_config=None, mesh_mapper=None):
    # device is either a ttnn.DeviceMesh or a ttnn.Device
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        device=device,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper if type(device) == ttnn.DeviceMesh else None,
    )


def get_num_devices(device):
    # device is either a ttnn.DeviceMesh or a ttnn.Device
    if type(device) == ttnn.DeviceMesh:
        return device.get_num_devices()
    elif type(device) == ttnn.Device:
        return 1
    else:
        raise ValueError(f"Unrecognized device type {type(device)}")


def dump_device_profiler(device):
    # device is either a ttnn.DeviceMesh or a ttnn.Device
    devices = get_devices(device)
    for device in devices:
        ttnn.DumpDeviceProfiler(device)
