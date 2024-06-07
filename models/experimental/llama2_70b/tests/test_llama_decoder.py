# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


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
    should_skip_model_load,
    check_kv_cache,
)


class PytorchLlamaDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num, rope_theta):
        super().__init__()
        self.decoder = hf_reference_model.layers[layer_num]
        self.rope_theta = rope_theta

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
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2, self.rope_theta)
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
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2, self.rope_theta)
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
    t3k_device_mesh,
    batch,
    seq_len,
    pcc,
    model_config,
    n_devices,
    emulated=False,
):
    # Prepare paths and devices
    t3k_device_mesh, ckpt_dir, tokenizer_path, cache_path = get_llama_path(
        t3k_device_mesh, model_config, n_devices, emulated
    )
    skip_model_load = should_skip_model_load()

    # Prepare configs
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch,
        n_layers=UNIT_TEST_N_LAYER,
        skip_model_load=skip_model_load,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params
    model_name = "Llama3-70b" if configuration.vocab_size == 128256 else "Llama2-70b"
    head_dim = configuration.dim // configuration.n_heads

    # PyTorch model --------------------------------------------------------------------
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModel(
        hugging_face_reference_model, UNIT_TEST_LAYER_NUM, configuration.rope_theta
    )
    # TT model -------------------------------------------------------------------------
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=model_config["DRAM_MEMCFG"],
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    transformation_mats = ttnn.to_device(transformation_mats, t3k_device_mesh)

    tt_LlamaDecoder_model = TtLlamaDecoder_optimized(
        t3k_device_mesh,
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

    all_tests_pass, all_pccs = True, []
    if model_config["LLM_MODE"] == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = UNIT_TEST_START_POS
        generation_length = UNIT_TEST_GENERATION_LENGTH
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

        tt_out = ttnn.from_device(tt_out)
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=3))
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

    tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_LlamaDecoder_model.attention.layer_past]
    tt_layer_present_all = [
        ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0)).transpose(0, 1)
        for lp in tt_layer_present_all
    ]

    cache_test_pass = check_kv_cache(
        pytorch_layer_present,
        tt_layer_present_all,
        generation_start_pos,
        generation_length,
        seq_len,
        model_config["LLM_MODE"] == "prefill",
        pcc,
    )
    all_tests_pass = all_tests_pass and cache_test_pass

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
    ((32, 1, 0.9993), (1, 128, 0.998), (1, 2048, 0.998)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaDecoder_inference(
    batch,
    seq_len,
    pcc,
    n_devices,
    t3k_device_mesh,
    emulated,
):
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_device_mesh.get_num_devices() < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_program_cache()

    inp = torch.rand(1, 1, 32, 32)
    for i in range(2):
        run_test_LlamaDecoder_inference(
            t3k_device_mesh,
            batch,
            seq_len,
            pcc,
            model_config,
            n_devices,
            emulated,
        )

        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            test_tensor = (
                ttnn.Tensor(
                    inp.reshape(-1).tolist(),
                    inp.shape,
                    ttnn.bfloat16,
                    ttnn.Layout.ROW_MAJOR,
                )
                .to(ttnn.Layout.TILE)
                .to(device)
            )
