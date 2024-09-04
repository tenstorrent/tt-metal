# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger
import torch
from torch import nn
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.t3000.llama2_70b.tt.llama_common import (
    get_llama_path,
    extract_pcc_from_log,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
)
from models.demos.t3000.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb,
    get_weight_cache_path,
)


class TtLlamaQKV(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        cache_path=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.layer_name = f"{base_url}.{layer_num}"

        wqkv_cache_str = f"{self.layer_name}.attention.wqkv_fused.weight"

        self.cache_path = cache_path

        self.qkv_list = []

        test_cache_path = get_weight_cache_path(self.cache_path, wqkv_cache_str, self.num_devices - 1, self.num_devices)
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, wqkv_cache_str, i, self.num_devices)
                self.qkv_list.append(
                    ttnn.load_tensor(str(tensor_cache_path)).to(self.devices[i], self.model_config["DRAM_MEMCFG"])
                )

    def forward(self, xs: ttnn.Tensor) -> ttnn.Tensor:
        fused_query_key_value = []
        for i in range(len(xs)):
            fused_query_key_value.append(
                ttnn.matmul(
                    xs[i],
                    self.qkv_list[i],
                    program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
        fused_query_key_value = tt_all_gather_torch(fused_query_key_value, dim=-1)
        return fused_query_key_value[0]


class PytorchLlamaQKVModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attn = hf_reference_model.layers[layer_num].attention

        # Disable dropout
        self.attn.eval()

    def forward(self, x):
        xq = self.attn.wq(x)
        xk = self.attn.wk(x)
        xv = self.attn.wv(x)

        # # Chunk tensors
        xq_chunks = xq.chunk(8, dim=-1)  # Chunk by 64 in the columns
        xk_chunks = xk.chunk(8, dim=-1)  # Chunk by 8
        xv_chunks = xv.chunk(8, dim=-1)  # Chunk by 8

        assembled_chunks = []
        # Assemble in 8q 1k 1v fashion
        for i in range(8):  # Loop to create 8 sets of 8q-1k-1v assemblies
            # For each set, we concatenate 8 q chunks, 1 k chunk, and 1 v chunk
            q_chunks_for_assembly = xq_chunks[i]
            k_chunk_for_assembly = xk_chunks[i]
            v_chunk_for_assembly = xv_chunks[i]

            # Concatenate the current 8q-1k-1v setup for this iteration
            current_assembly = torch.cat([q_chunks_for_assembly, k_chunk_for_assembly, v_chunk_for_assembly], dim=-1)
            assembled_chunks.append(current_assembly)

        # Concatenate assembled chunks
        result = torch.cat(assembled_chunks, dim=-1)
        return result
        # return torch.cat([xq_chunks[0], xk_chunks[0], xv_chunks[0]], dim=-1)


def run_test_LlamaQKV(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
    n_devices,
    emulated=False,
):
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices, emulated)

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=1, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    inp = (torch.rand(batch, 1, seq_len, configuration.dim) * 2) - 1
    layer_num = 0
    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaQKV_model = PytorchLlamaQKVModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaQKV_model(inp)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaQKV_model = TtLlamaQKV(
        devices,
        state_dict,
        base_url,
        layer_num,
        configuration.dim,
        model_config,
        cache_path=cache_path,
    )

    tt_inp = [torch2tt_tensor(inp.clone(), devices[i]) for i in range(n_devices)]

    tt_out = tt_LlamaQKV_model(tt_inp)
    tt_out = tt2torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass = True
    out_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")
    does_pass = does_pass and out_pass

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_devices, emulated",
    (
        (8, False),
        (8, True),
        (32, True),
    ),
    ids=(
        "8chip-T3000",
        "8chip-emulated",
        "32chip-emulated",
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (32, 1),
        (1, 128),
    ),
    ids=("decode", "prefill"),
)
@pytest.mark.parametrize("model_config_str, pcc", (("BFLOAT16-DRAM", 0.9997),))
def test_LlamaAttention_inference(
    batch,
    seq_len,
    pcc,
    model_config_str,
    n_devices,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str, num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaQKV(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        emulated,
    )
