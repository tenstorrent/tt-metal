# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from pathlib import Path

import tt_lib

from models.demos.llama2_70b.reference.llama.llama import Llama
from models.demos.llama2_70b.reference.llama.llama.model import precompute_freqs_cis

from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
from models.demos.llama2_70b.tt.llama_attention_optimized import TtLlamaAttention_optimized


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.layers[layer_num].attention

        # Disable dropout
        self.attention.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + 1]

        attn_mask = torch.zeros(batch, 1, 1, start_pos + 1)
        # attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        x: (batch, seq, hidden_dim)
        start_pos: int
        freqs_cis: ?
        mask: ?

        return: (batch, seq, hidden_dim)
        """
        result = self.attention(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def run_test_LlamaAttention_inference(
    devices,
    batch,
    seq_len,
    pcc,
    optimized,
    model_config,
    n_devices,
    emulated=False,
):
    if emulated:
        ckpt_dir = "/proj_sw/user_dev/llama-data-repacked-2/llama-2-70b/"
        tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
        cache_path = Path("/proj_sw/user_dev/llama-data-cache/weights-cache")
        device = devices[0]
        devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        ckpt_dir = model_config["DEFAULT_CKPT_DIR"]
        tokenizer_path = model_config["DEFAULT_TOKENIZER_PATH"]
        cache_path = model_config["DEFAULT_CACHE_PATH"]

    max_seq_len = 4096
    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=max_seq_len, max_batch_size=batch, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    print(state_dict.keys())

    # Prepare configs
    torch.manual_seed(0)
    layer_num = 0
    base_url = "layers"
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    print(f"Running optimized: {optimized}")
    print(f"Running emulated: {emulated}")
    print(f"Running on {n_devices} devices")

    # PyTorch model --------------------------------------------------------------------
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    # TT model -------------------------------------------------------------
    if optimized:
        tt_LlamaAttention_model = TtLlamaAttention_optimized(
            devices,
            state_dict,
            base_url,
            layer_num,
            model_config,
            configuration,
            emulated=emulated,
            cache_path=cache_path,
        )
    else:
        tt_LlamaAttention_model = TtLlamaAttention(
            devices, state_dict, base_url, layer_num, model_config, configuration
        )

    if not emulated:
        for device in devices:
            tt_lib.device.Synchronize(device)

    generation_start_pos = 120
    generation_length = 1
    all_tests_pass = True
    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
        pt_inp_normed = hugging_face_reference_model.layers[layer_num].attention_norm(pt_inp)
        tt_input = pt_inp_normed.clone()
        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        attention_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaAttention_model.prepare_inputs(
            pt_inp_normed, start_pos
        )

        pytorch_out = pytorch_LlamaAttention_model(
            attention_input,
            start_pos,
            freqs_cis,
            attn_mask,
        )

        # TT hardware execution -------------------------------------------------------------
        attention_input, start_pos, rot_mat, attn_mask = tt_LlamaAttention_model.prepare_inputs(tt_input, start_pos)

        tt_out = tt_LlamaAttention_model(
            attention_input,
            rot_mat,
            start_pos,
            attn_mask,
        )

        assert isinstance(tt_out, list)  # tt_out should be sharded on N devices
        tt_outs = [tt2torch_tensor(t) for t in tt_out]
        tt_out = torch.cat(tt_outs, dim=-1)
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")

        if does_pass:
            logger.info(f"[start_pos={start_pos}] Llama2-70b Attention output Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Llama2-70b Attention output Failed! PCC value is lower than {pcc}")
            all_tests_pass = False

    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    pytorch_layer_present = [
        pytorch_LlamaAttention_model.attention.cache_k.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
        pytorch_LlamaAttention_model.attention.cache_v.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
    ]
    # TT hardware execution -------------------------------------------------------------
    tt_layer_present = []
    for layer_past in tt_LlamaAttention_model.layer_past_list:
        tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
    # concat the pasts by heads
    if len(devices) > 1:
        tt_layer_present = [
            torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
        ]
    else:
        tt_layer_present = tt_layer_present[0]

    for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present):
        cache_length_to_check = generation_start_pos + generation_length
        cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
        cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
        logger.info(f"Output: {output_pcc}")

        if does_pass:
            logger.info(f"KV Cache Passed!")
        else:
            logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
            all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama2 Attention output Passed!")
    else:
        logger.warning("Llama2 Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@pytest.mark.parametrize(
    "batch, seq_len, pcc, optimized, n_devices, emulated",
    (
        (32, 1, 0.9997, True, 4, False),
        (32, 1, 0.9997, True, 8, False),
        (32, 1, 0.9997, True, 4, True),
        (32, 1, 0.9997, True, 8, True),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaAttention_inference(
    batch,
    seq_len,
    pcc,
    optimized,
    model_config_str,
    n_devices,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str, num_devices=n_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaAttention_inference(
        devices,
        batch,
        seq_len,
        pcc,
        optimized,
        model_config,
        n_devices,
        emulated=emulated,
    )
