# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
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
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
)


class PytorchLlamaModel(torch.nn.Module):
    def __init__(self, hf_reference_model):
        super().__init__()
        self.model = hf_reference_model

        # Disable dropout
        self.model.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def forward(self, x, start_pos):
        """
        x: (batch, seq)
        start_pos: int

        return: (batch, seq, hidden_dim)
        """
        return self.model(x, start_pos)


def run_test_LlamaModel_inference(
    t3k_device_mesh,
    batch,
    seq_len,
    pcc,
    model_config,
    n_layers,
    n_devices,
    emulated=False,
):
    # Prepare paths and devices
    t3k_device_mesh, ckpt_dir, tokenizer_path, cache_path = get_llama_path(
        t3k_device_mesh, model_config, n_devices, emulated
    )

    logger.info(f"Running num_layer: {n_layers}")
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=False,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params
    model_name = "Llama3-70b" if configuration.vocab_size == 128256 else "Llama2-70b"

    # PyTorch model --------------------------------------------------------------------
    pytorch_model = PytorchLlamaModel(hugging_face_reference_model)
    # TT model -------------------------------------------------------------------------
    tt_model = TtLlamaModel_optimized(
        t3k_device_mesh,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        batch,
        emulated=emulated,
        cache_path=cache_path,
    )

    # for device in devices:
    #     tt_lib.device.Synchronize(device)

    if model_config["LLM_MODE"] == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = UNIT_TEST_START_POS
        generation_length = UNIT_TEST_GENERATION_LENGTH
    all_tests_pass = True
    all_pccs, all_top1, all_top5 = [], [], []
    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        tt_inp_ids = pt_inp_ids.clone()
        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        pytorch_out = pytorch_model(
            pt_inp_ids,
            start_pos,
        )

        # TT hardware execution -------------------------------------------------------------
        tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_inp_ids, start_pos)

        tt_out = tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )
        del tt_inp_emb, rot_mat, attn_mask

        logger.info(f"Syncronizing devices for token idx {start_pos}")

        # for device in devices:
        #     tt_lib.device.Synchronize(device)
        #     if i % 8 == 0:
        #         tt_lib.device.DumpDeviceProfiler(device)

        logger.info(f"Done synchronizing devices")

        tt_out = ttnn.from_device(tt_out)
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=3))
        tt_out = tt_out[..., : configuration.vocab_size]
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze()  # [batch, hidden_dim]
        tt_out = tt_out.float()
        pytorch_out = pytorch_out.squeeze()  # [batch, hidden_dim]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")
        all_pccs.append(extract_pcc_from_log(output_pcc))

        kl_divs = scipy.stats.entropy(
            torch.nn.functional.softmax(pytorch_out, dim=-1), torch.nn.functional.softmax(tt_out, dim=-1), axis=-1
        )
        logger.info(f"Mean KL Divergence: {kl_divs.mean()}")

        # Write the code to check top-5 and top-1 accuracy. It should show the
        # percentage where the top-1 prediction in pytorch was in the top-5
        # predictions in tt.
        reference_top1 = np.argmax(pytorch_out, axis=-1)
        top1_acc = top_k_accuracy_score(reference_top1, tt_out, k=1, labels=np.arange(tt_out.shape[-1]))
        top5_acc = top_k_accuracy_score(reference_top1, tt_out, k=5, labels=np.arange(tt_out.shape[-1]))

        all_top1.append(top1_acc)
        all_top5.append(top5_acc)

        logger.info(f"Mean Top-1: {top1_acc}")
        logger.info(f"Mean Top-5: {top5_acc}")

        if does_pass:
            logger.info(f"[start_pos={start_pos}] {model_name} Model output Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] {model_name} Model output Failed! PCC value is lower than {pcc}")
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    logger.info(f"Average Top-1 over {len(all_top1)} tokens: {sum(all_top1) / len(all_top1)}")
    logger.info(f"Average Top-5 over {len(all_top5)} tokens: {sum(all_top5) / len(all_top5)}")
    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    pytorch_layer_present = [
        pytorch_model.model.layers[0]
        .attention.cache_k.clone()
        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        pytorch_model.model.layers[0]
        .attention.cache_v.clone()
        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
    ]
    # TT hardware execution -------------------------------------------------------------
    # tt_layer_present = []
    # for layer_past in tt_model.layers[0].attention.layer_past_list:
    #     tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
    # # concat the pasts by heads
    # tt_layer_present = [
    #     torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
    # ]
    tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layers[0].attention.layer_past]
    tt_layer_present_all = [
        ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=1)) for lp in tt_layer_present_all
    ]

    for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present_all):
        cache_length_to_check = generation_start_pos + generation_length + 1
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
        logger.info(f"{model_name} output Passed!")
    else:
        logger.warning(f"{model_name} output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "pcc, n_layers",
    (
        (0.999, 1),
        (0.998, 2),
        (0.99, 4),
        (0.99, 6),
        (0.99, 7),
        (0.99, 8),
        (0.99, 10),
        (0.99, 20),
        (0.99, 40),
        (0.99, 80),
    ),
    ids=("1L", "2L", "4L", "6L", "7L", "8L", "10L", "20L", "40L", "80L"),
)
@pytest.mark.parametrize(
    "n_devices, emulated",
    (
        (8, False),
        (8, True),
    ),
    ids=(
        "8chip-T3000",
        "8chip-emulated",
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1), (1, 128), (1, 2048)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    pcc,
    n_layers,
    n_devices,
    t3k_device_mesh,
    emulated,
    use_program_cache,
):
    # devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    # if len(devices) < n_devices and not emulated:
    #     pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaModel_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        n_layers,
        n_devices,
        emulated,
    )
