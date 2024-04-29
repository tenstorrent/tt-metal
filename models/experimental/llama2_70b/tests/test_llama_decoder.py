# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.experimental.llama2_70b.tt.llama_decoder_galaxy import TtLlamaDecoder_galaxy
from models.experimental.llama2_70b.reference.llama.llama.model import precompute_freqs_cis
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)

# from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
#     comp_allclose,
#     comp_pcc,
# )
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
    extract_pcc_from_log,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    get_rot_transformation_mat,
)


class PytorchLlamaDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.decoder = hf_reference_model.layers[layer_num]

        # Disable dropout
        self.decoder.eval()

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

    def prepare_inputs_prefill(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        seq_len = x.size(1)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

        attn_mask = torch.full((seq_len, seq_len), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = attn_mask.expand(batch, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        x: (batch, seq, hidden_dim)
        start_pos: int
        freqs_cis: ?
        mask: ?

        return: (batch, seq, hidden_dim)
        """
        result = self.decoder(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def run_test_LlamaDecoder_inference(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
    n_devices,
    emulated=False,
):
    # Prepare paths and devices
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices, emulated)

    # Prepare configs
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch,
        n_layers=UNIT_TEST_N_LAYER,
        skip_model_load=False,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params
    model_name = "Llama3-70b" if configuration.vocab_size == 128256 else "Llama2-70b"
    head_dim = configuration.dim // configuration.n_heads

    # PyTorch model --------------------------------------------------------------------
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModel(hugging_face_reference_model, UNIT_TEST_LAYER_NUM)
    # TT model -------------------------------------------------------------------------
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = [torch2tt_tensor(transformation_mat_torch.clone(), device) for device in devices]
    if n_devices == 32:
        tt_LlamaDecoder_model = TtLlamaDecoder_galaxy(
            devices,
            state_dict,
            BASE_URL,
            UNIT_TEST_LAYER_NUM,
            model_config,
            configuration,
            batch,
            transformation_mats,
            emulated=emulated,
            cache_path=cache_path,
        )
    else:
        tt_LlamaDecoder_model = TtLlamaDecoder_optimized(
            devices,
            state_dict,
            BASE_URL,
            UNIT_TEST_LAYER_NUM,
            model_config,
            configuration,
            batch,
            transformation_mats,
            emulated=emulated,
            cache_path=cache_path,
        )

    if not emulated:
        for device in devices:
            tt_lib.device.Synchronize(device)

    all_tests_pass, all_pccs = True, []
    if model_config["LLM_MODE"] == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = 127
        generation_length = 1
    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
        tt_input = pt_inp.clone()
        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        if model_config["LLM_MODE"] == "prefill":
            x_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaDecoder_model.prepare_inputs_prefill(
                pt_inp, start_pos
            )
        else:
            x_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaDecoder_model.prepare_inputs(pt_inp, start_pos)

        pytorch_out = pytorch_LlamaDecoder_model(
            x_input,
            start_pos,
            freqs_cis,
            attn_mask,
        )

        # TT hardware execution -------------------------------------------------------------
        x_input, start_pos, rot_mat, attn_mask = tt_LlamaDecoder_model.prepare_inputs(tt_input, start_pos)

        tt_out = tt_LlamaDecoder_model(
            x_input,
            rot_mat,
            start_pos,
            attn_mask,
        )

        assert isinstance(tt_out, list)  # tt_out should be fractured or replicated on N devices
        assert len(tt_out) == len(devices)
        if n_devices == 32:
            tt_out = tt2torch_tensor(tt_out[0])
        else:
            tt_outs = [tt2torch_tensor(o) for o in tt_out]
            tt_out = torch.cat(tt_outs, dim=-1)
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")
        all_pccs.append(extract_pcc_from_log(output_pcc))

        if does_pass:
            logger.info(f"[start_pos={start_pos}] {model_name} Decoder output Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] {model_name} Decoder output Failed! PCC value is lower than {pcc}")
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    pytorch_layer_present = [
        pytorch_LlamaDecoder_model.decoder.attention.cache_k.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
        pytorch_LlamaDecoder_model.decoder.attention.cache_v.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
    ]
    # TT hardware output -----------------------------------------------------------------
    # TODO: Move to a helper function
    if n_devices == 32:
        all_k, all_v = [], []
        for i in range(4):  # 4 device groups
            tt_layer_present = []
            for layer_past in tt_LlamaDecoder_model.attention.attentions[i].layer_past_list:
                tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
            # concat the pasts by heads
            tt_layer_present = [
                torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
            ]
            all_k.append(tt_layer_present[0])
            all_v.append(tt_layer_present[1])
        # Concat across device groups
        all_k = torch.cat(all_k, dim=0)
        all_v = torch.cat(all_v, dim=0)
        tt_layer_present_all = [all_k, all_v]
    else:
        tt_layer_present = []
        for layer_past in tt_LlamaDecoder_model.attention.layer_past_list:
            tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
        # concat the pasts by heads
        tt_layer_present_all = [
            torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
        ]

    for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present_all):
        cache_length_to_check = generation_start_pos + generation_length
        if model_config["LLM_MODE"] == "prefill":
            cache_pt = cache_pt[:, :, :seq_len, :]
            cache_tt = cache_tt[:, :, :seq_len, :]
        else:
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
        logger.info(f"{model_name} Decoder output Passed!")
    else:
        logger.warning(f"{model_name} Decoder output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


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
    "batch, seq_len, pcc",
    ((32, 1, 0.9993), (1, 128, 0.9996), (1, 2048, 0.9994)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaDecoder_inference(
    batch,
    seq_len,
    pcc,
    n_devices,
    all_devices,
    emulated,
    use_program_cache,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaDecoder_inference(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        emulated,
    )
