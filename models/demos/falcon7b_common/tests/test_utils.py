# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.falcon7b_common.reference.hf_modeling_falcon import FalconForCausalLM
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, devices):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        kv_cache_cur_layer = []
        for device in devices:
            k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
            v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
            tt_k_cache = torch2tt_tensor(k_cache, device)
            tt_v_cache = torch2tt_tensor(v_cache, device)
            kv_cache_cur_layer.append((tt_k_cache, tt_v_cache))
        kv_cache += (kv_cache_cur_layer,)
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
    devices,
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

        if generate_attention_inputs:
            attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
            attention_mask_bool = torch.ones(global_batch, 1, q_len, kv_len, dtype=bool).triu(diagonal=1)
        layer_past = None

        tt_layer_past = ()
        for layer in range(num_layers):
            tt_layer_past_cur = []
            for i, device in enumerate(devices):
                # Generate kvcache for each layer and attention mask once
                if generate_attention_inputs and layer == 0:
                    attention_input_i = attention_input[batch * i : batch * (i + 1)]
                    attention_mask_bool_i = attention_mask_bool[batch * i : batch * (i + 1)]
                    tt_attention_input.append(torch2tt_tensor(attention_input_i.unsqueeze(1), device))
                    if model_config["PREFILL_OPTIMIZED_MODE"] and seq_len in [2048, 128, 1024]:
                        attn_masks = create_prefill_attn_mask_for_sharded_softmax(
                            (attention_mask_bool_i * -1e5),
                            configuration.num_attention_heads,
                            q_len,
                        )
                        tt_attn_masks = [
                            torch2tt_tensor(attn_mask, device, tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT4_B)
                            for attn_mask in attn_masks
                        ]
                        tt_attention_mask.append(tt_attn_masks)
                    else:
                        tt_attention_mask.append(
                            torch2tt_tensor(
                                (attention_mask_bool_i * -100000).expand(-1, configuration.num_attention_heads, -1, -1),
                                device,
                            )
                        )
                tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
                tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
                tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
                tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
                tt_layer_past_cur.append((tt_k_cache, tt_v_cache))
            tt_layer_past += (tt_layer_past_cur,)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        if generate_attention_inputs:
            attention_input = (torch.rand(global_batch, q_len, configuration.hidden_size) * 2) - 1
            # attention_input = (torch.rand(batch, q_len, 4544) * 2) - 1
            attention_mask_bool = torch.zeros(global_batch, 1, q_len, kv_len, dtype=bool)
        layer_past = ()
        tt_layer_past = ()
        for layer in range(num_layers):
            k_cache = torch.rand(global_batch, kv_cache_len, head_dim)
            v_cache = torch.rand(global_batch, kv_cache_len, head_dim)
            layer_past += ((k_cache.unsqueeze(1), v_cache.unsqueeze(1)),)

            tt_layer_past_cur = []
            for i, device in enumerate(devices):
                # Generate kvcache for each layer and attention mask once
                if generate_attention_inputs and layer == 0:
                    tt_attention_input.append(
                        torch2tt_tensor(
                            attention_input[batch * i : batch * (i + 1)].unsqueeze(1).transpose(0, 2), device
                        )
                    )

                    kv_len_padded = (kv_len + 31) // 32 * 32
                    attention_mask_bool_padded = torch.cat(
                        (
                            attention_mask_bool[batch * i : batch * (i + 1)],
                            torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
                        ),
                        dim=-1,
                    )
                    tt_attention_mask.append(
                        torch2tt_tensor(
                            (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                                -1,
                                configuration.num_attention_heads,
                                -1,
                                -1,
                                # -1, 71, -1, -1
                            ),
                            device,
                        )
                    )
                tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
                tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
                tt_k_cache[:, :kv_cache_len, :] = k_cache[batch * i : batch * (i + 1)]
                tt_v_cache[:, :kv_cache_len, :] = v_cache[batch * i : batch * (i + 1)]
                tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
                tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
                tt_layer_past_cur.append((tt_k_cache, tt_v_cache))
            tt_layer_past += (tt_layer_past_cur,)

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


def concat_device_out_layer_present(num_devices, tt_layer_present, seq_end_idx, end_idx_only=False):
    for i in range(num_devices):
        tt_layer_present[i] = (
            tt2torch_tensor(tt_layer_present[i][0]).squeeze(1),
            tt2torch_tensor(tt_layer_present[i][1]).squeeze(1),
        )
        if not end_idx_only:
            tt_layer_present[i] = (
                tt_layer_present[i][0][:, :seq_end_idx, :],
                tt_layer_present[i][1][:, :seq_end_idx, :],
            )
        else:
            tt_layer_present[i] = (
                tt_layer_present[i][0][:, seq_end_idx, :],
                tt_layer_present[i][1][:, seq_end_idx, :],
            )
    tt_layer_present = (torch.concat([x[0] for x in tt_layer_present]), torch.concat([x[1] for x in tt_layer_present]))
    return tt_layer_present


def concat_device_outputs(num_devices, tt_out, llm_mode, tt_layer_present, seq_end_idx):
    for i in range(num_devices):
        tt_out[i] = tt2torch_tensor(tt_out[i]).squeeze(1)
        if llm_mode == "decode":
            tt_out[i] = tt_out[i].transpose(0, 1)
    tt_out = torch.concat(tt_out)
    tt_layer_present = concat_device_out_layer_present(num_devices, tt_layer_present, seq_end_idx)
    return tt_out, tt_layer_present
