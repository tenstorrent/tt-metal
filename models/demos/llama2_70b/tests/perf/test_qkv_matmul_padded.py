# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather, tt_all_gather_torch


class TtLlamaQKV_optimized(torch.nn.Module):
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads

        assert self.num_devices == 4 or self.num_devices == 8
        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        layer_name = f"{base_url}.{layer_num}"

        wq_str = f"{layer_name}.attention.wq.weight"
        wk_str = f"{layer_name}.attention.wk.weight"
        wv_str = f"{layer_name}.attention.wv.weight"

        self.qkv_list = []
        for i in range(self.num_devices):
            # Chunk weights
            wq_chunks = torch.chunk(self.state_dict[wq_str], self.n_heads, dim=0)
            wk_chunks = torch.chunk(self.state_dict[wk_str], self.n_kv_heads, dim=0)
            wv_chunks = torch.chunk(self.state_dict[wv_str], self.n_kv_heads, dim=0)

            # Select chunks for the current device
            wq_selected = torch.cat(wq_chunks[i * self.n_local_heads : (i + 1) * self.n_local_heads], dim=0)
            wk_selected = torch.cat(wk_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)
            wv_selected = torch.cat(wv_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            # Create interleaved qkv list
            n_repeat = self.n_heads // self.n_kv_heads
            qkv_interleaved = [
                [
                    wq[..., i * n_repeat * self.head_dim : (i + 1) * n_repeat * self.head_dim],
                    wk[..., i * self.head_dim : (i + 1) * self.head_dim],
                    wv[..., i * self.head_dim : (i + 1) * self.head_dim],
                ]
                for i in range(self.n_local_kv_heads)
            ]
            qkv_interleaved = [item for sublist in qkv_interleaved for item in sublist]

            # Concatenate Q, K, V for the current device
            qkv = torch.cat(qkv_interleaved, dim=-1)

            # pad 6 heads per group so that we can use 64 cores
            qkv = torch.cat([qkv, torch.zeros(qkv.shape[0], self.n_local_kv_heads * 6 * self.head_dim)], dim=-1)

            # Append the processed tensor to the list, assuming torch2tt_tensor is a defined method
            self.qkv_list.append(
                # tt_lib.tensor.pad(
                torch2tt_tensor(
                    qkv,
                    self.devices[i],
                    tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
                ),
                # output_tensor_shape = [1, 1, self.hidden_size, self.head_dim*self.n_local_kv_heads*16],
                # input_tensor_start = [0,0,0,0],
                # pad_value = 0,
                # )
            )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        print(f"x:0 shape:{x[0].shape()}")
        # qkv_weights = (8192, 10240)
        # x = (1, 1, 32, 8192)
        # fused_qkv_out = (1, 1, 32, 10240)

        # 4 devices: qkv_weights = (8192, 2560)
        # 8 devices: qkv_weights = (8192, 1280)

        # 4 devices per core:
        # K = 256
        # N = 80
        # compute_with_storage_grid_size=(8, 2)

        # 8 devices per core:
        # K = 256
        # N = 40
        # compute_with_storage_grid_size=(8, 1)
        attn_output = []

        for i in range(len(x)):
            print("qkv fused weighs shape:", self.qkv_list[i].shape())
            qkv_proj = tt_lib.operations.primary.matmul_1d(
                x[i],
                self.qkv_list[i],
                program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
                output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            )

            qkv_proj = tt_lib.tensor.sharded_to_interleaved(
                qkv_proj, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )

            qkv_proj = tt_lib.tensor.unpad(
                qkv_proj,
                output_tensor_start=[0, 0, 0, 0],
                output_tensor_end=[0, 0, self.max_batch_size - 1, self.head_dim * self.n_local_kv_heads * 10 - 1],
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
                # output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            )

            attn_output.append(qkv_proj)

            x[i].deallocate(True)

        if self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] != self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]:
            # for i in range(len(attn_output)):
            #     attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
            #         attn_output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            #     )
            for i in range(len(attn_output)):
                attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_output[i], sharded_mem_config=self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]
                )

        query_layer = []
        key_layer = []
        value_layer = []
        for i in range(len(attn_output)):
            (
                q_layer,  # [seqlen, n_local_heads, bsz, head_dim]
                k_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
                v_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            ) = tt_lib.tensor.nlp_create_qkv_heads(
                attn_output[i],
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )
            query_layer.append(q_layer)
            key_layer.append(k_layer)
            value_layer.append(v_layer)

            attn_output[i].deallocate(True)

        for i in range(len(query_layer)):
            query_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                query_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
            key_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                key_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
            value_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                value_layer[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )

        query_layer_allgather = tt_all_gather_torch(query_layer, dim=1)
        key_layer_allgather = tt_all_gather_torch(key_layer, dim=1)
        value_layer_allgather = tt_all_gather_torch(value_layer, dim=1)

        return query_layer_allgather[0], key_layer_allgather[0], value_layer_allgather[0]


class PytorchLlamaQKVModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attn = hf_reference_model.layers[layer_num].attention
        self.n_heads = 64
        self.kv_heads = 8
        self.head_dim = 128

        # Disable dropout
        self.attn.eval()

    def forward(self, x):
        xq = self.attn.wq(x)
        xk = self.attn.wk(x)
        xv = self.attn.wv(x)

        seqlen = xq.size(0)
        bsz = xq.size(2)

        xq = xq.view(seqlen, bsz, self.n_heads, self.head_dim).transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.view(seqlen, bsz, self.kv_heads, self.head_dim).transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xv = xv.view(seqlen, bsz, self.kv_heads, self.head_dim).transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim_

        return xq, xk, xv


def run_test_LlamaQKV(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    num_devices,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()
    print(state_dict.keys())

    # Prepare input
    torch.manual_seed(0)

    if seq_len == 1:
        input_shape = [seq_len, 1, batch, configuration.dim]
    else:
        input_shape = [batch, 1, seq_len, configuration.dim]

    attention_input = (torch.rand(input_shape) * 2) - 1
    layer_num = 0
    base_url = "layers"

    # Only 4 or 8 devices are supported, single device cant use full core grid for now.
    assert num_devices == 4 or num_devices == 8

    devices = [device for _ in range(num_devices)]  # Emulate fracturing on N chips
    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaQKV_model = PytorchLlamaQKVModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaQKV_model(attention_input)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaQKV_model = TtLlamaQKV_optimized(devices, state_dict, base_url, layer_num, model_config, configuration)

    tt_attention_input_host = torch2tt_tensor(attention_input, None, tt_dtype=model_config["LN_ATTN_OUTPUT_DTYPE"])
    tt_attention_input = []
    # TODO: Move sharding inputs to model
    for device in devices:
        tt_attention_input.append(tt_attention_input_host.to(device, model_config["LN_ATTN_OUTPUT_MEMCFG"]))

    tt_out = tt_LlamaQKV_model(tt_attention_input)
    tt_outs = [tt2torch_tensor(tt_o) for tt_o in tt_out]

    does_pass = True
    for i in range(len(pytorch_out)):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_outs[i], pcc)
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("n_devices", (8, 4))
@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (("llama-2-70B", 32, 1, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaQKV_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
    n_devices,
    use_program_cache,
):
    model_config = get_model_config(model_config_str, num_devices=n_devices)

    run_test_LlamaQKV(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        # tt_cache_path,
        # model_location_generator,
    )
