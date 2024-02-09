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


class TtLlamaQKV_optimized(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        self.n_heads = 64
        self.n_kv_heads = 8

        layer_name = f"{base_url}.{layer_num}"

        wq_str = f"{layer_name}.attention.wq.weight"
        wk_str = f"{layer_name}.attention.wk.weight"
        wv_str = f"{layer_name}.attention.wv.weight"

        self.wq = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wq_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
        )
        self.wk = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wk_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
        )
        self.wv = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wv_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
        )
        self.qkv_weights = tt_lib.tensor.concat([self.wq, self.wk, self.wv], dim=-1)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        print(f"x shape:{x.shape()}")

        x = tt_lib.tensor.interleaved_to_sharded(x, sharded_mem_config=self.model_config["QKV_MM_INPUT_MEMCFG"])
        # qkv_weights = (8192, 10240)
        # x = (1, 1, 32, 8192)
        # fused_qkv_out = (1, 1, 32, 10240)

        # compute_with_storage_grid_size=(8, 4),
        # in0_block_w=8,
        # out_subblock_h=1,
        # out_subblock_w=5,
        # per_core_M=1,
        # per_core_N=10,

        fused_qkv_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.qkv_weights,
            program_config=self.model_config["QKV_MM_PROGCFG"],
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
        )

        x.deallocate(True)
        if self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] != self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]:
            fused_qkv_out = tt_lib.tensor.sharded_to_interleaved(
                fused_qkv_out, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
            fused_qkv_out = tt_lib.tensor.interleaved_to_sharded(
                fused_qkv_out, sharded_mem_config=self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]
            )

        print(self.device.compute_with_storage_grid_size())
        (
            q_heads,  # [seqlen, n_heads, bsz, head_dim]
            k_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            fused_qkv_out,
            num_heads=64,
            num_kv_heads=8,
            transpose_k_heads=False,
            output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
        )
        fused_qkv_out.deallocate(True)

        return q_heads, k_heads, v_heads


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

        batch_size = 32

        xq = xq.view(1, batch_size, self.n_heads, self.head_dim)
        xk = xk.view(1, batch_size, self.kv_heads, self.head_dim)
        xv = xv.view(1, batch_size, self.kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        return xq, xk, xv


def run_test_LlamaQKV(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=True
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)

    if seq_len == 1:
        input_shape = [seq_len, 1, batch, configuration.dim]
    else:
        input_shape = [batch, 1, seq_len, configuration.dim]

    inp = (torch.rand(input_shape) * 2) - 1
    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaQKV_model = PytorchLlamaQKVModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaQKV_model(inp)

    # TT hardware execution -------------------------------------------------------------

    tt_LlamaQKV_model = TtLlamaQKV_optimized(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.dim,
        model_config,
        tt_cache_path=None,
    )

    tt_inp = torch2tt_tensor(inp, device)
    tt_out = tt_LlamaQKV_model(tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]
    # check outputs ----------------------------------------------------------------------
    for i in range(3):
        logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(3):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "llama-2-70B",
            32,
            1,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-SHARDED",))
def test_LlamaQKV_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
    use_program_cache,
):
    model_config = get_model_config(model_config_str)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaQKV(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
