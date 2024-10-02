# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
import os

import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.tg.llama3_70b.tt.llama_model_galaxy import TtLlamaModel_galaxy
from models.demos.tg.llama3_70b.tt.llama_common import PytorchLlamaModel
from models.utility_functions import skip_for_grayskull
from models.demos.tg.llama3_70b.tt.llama_common import setup_llama_env
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_mesh_device,
    extract_pcc_from_log,
    BASE_URL,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    should_skip_model_load,
    check_kv_cache,
    ConcatMesh2DToTensor,
)
import gc


def run_test_LlamaModel_inference(
    mesh_device,
    cluster_shape,
    batch,
    seq_len,
    pcc,
    model_config,
    n_layers,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
    prompt_file=None,
):
    # Load prompt file if provided
    prompt = None
    if prompt_file:
        assert os.path.isfile(prompt_file), "Input file does not exist!"
        with open(prompt_file, "r") as f:
            prompt = f.read()

    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    logger.info(f"Running num_layer: {n_layers}")
    hugging_face_reference = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=model_config[llama_version],
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=skip_model_load,
    )
    hugging_face_reference_model = hugging_face_reference.model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # PyTorch model --------------------------------------------------------------------
    pytorch_model = PytorchLlamaModel(hugging_face_reference_model)
    # TT model -------------------------------------------------------------------------
    tt_model = TtLlamaModel_galaxy(
        mesh_device,
        cluster_shape,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        cache_path=cache_path,
        read_cache=True,
    )

    mode = "decode" if seq_len == 1 else "prefill"

    if mode == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = UNIT_TEST_START_POS
        generation_length = UNIT_TEST_GENERATION_LENGTH

    # Pre-process inputs in prompt mode
    if prompt:
        tokenizer = hugging_face_reference.tokenizer
        tokenized = tokenizer.encode(prompt, bos=True, eos=False)
        tokenized = torch.tensor(tokenized).unsqueeze(0)

        # Sliding window across sequence dimension for generation_length iterations
        tokenized = tokenized[:, : (seq_len + generation_length - 1) * batch]
        tokenized = torch.reshape(tokenized, (batch, (seq_len + generation_length - 1)))

        logger.info("Finished converting prompt to tokens.")

    all_tests_pass = True
    all_pccs, all_top1, all_top5 = [], [], []
    for i in range(generation_length):
        # Prepare input
        if prompt:
            pt_inp_ids = tokenized[:, i : i + seq_len]  # Slide window
            assert pt_inp_ids.shape == (batch, seq_len), f"Inputs must have shape {(batch, seq_len)}"
        else:
            pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        tt_inp_ids = pt_inp_ids.clone()

        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        logger.info(f"Running inference on PyTorch")
        pytorch_out = pytorch_model(
            pt_inp_ids,
            start_pos,
        )
        logger.info(f"Finished PyTorch inference")

        # TT hardware execution -------------------------------------------------------------
        tt_inp_emb, start_pos, rot_mat, cache_idxs, attn_masks = tt_model.prepare_inputs(
            tt_inp_ids, start_pos, mode=mode
        )
        tt_out = tt_model(tt_inp_emb, rot_mat, start_pos, cache_idxs=cache_idxs, attn_masks=attn_masks, mode=mode)
        del tt_inp_emb, rot_mat, attn_mask

        tt_out = ttnn.to_torch(
            tt_out, mesh_composer=ConcatMesh2DToTensor(mesh_device, dims=(1, 3), cluster_shape=cluster_shape)
        )
        tt_out = tt_out[:, 0:1, :, : configuration.vocab_size]
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze()  # [batch, hidden_dim]

        tt_out = tt_out.float()
        if mode == "decode":
            tt_out = tt_out[:batch]
            pytorch_out = pytorch_out.squeeze()  # [batch, hidden_dim]
        elif mode == "prefill":
            # Take only the last token to compare with PyTorch output
            tt_out = tt_out[-1, :].unsqueeze(0)
            pytorch_out = pytorch_out[:, -1, :]

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
            logger.info(f"[start_pos={start_pos}] {llama_version} Model output Passed!")
        else:
            logger.warning(
                f"[start_pos={start_pos}] {llama_version} Model output Failed! PCC value is lower than {pcc}"
            )
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    logger.info(f"Average Top-1 over {len(all_top1)} tokens: {sum(all_top1) / len(all_top1)}")
    logger.info(f"Average Top-5 over {len(all_top5)} tokens: {sum(all_top5) / len(all_top5)}")
    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    for layer_id in range(n_layers):
        print(f"Checking KV cache for layer {layer_id}")
        pytorch_layer_present = [
            pytorch_model.model.layers[layer_id]
            .attention.cache_k.clone()
            .permute(0, 2, 1, 3)[:batch, ...],  # [batch, n_kv_heads, seq, head_dim]
            pytorch_model.model.layers[layer_id]
            .attention.cache_v.clone()
            .permute(0, 2, 1, 3)[:batch, ...],  # [batch, n_kv_heads, seq, head_dim]
        ]

        tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_model.layers[layer_id].attention.layer_past]
        tt_layer_present_all = [
            ttnn.to_torch(
                lp, mesh_composer=ConcatMesh2DToTensor(mesh_device, dims=(0, 1), cluster_shape=cluster_shape)
            )[:batch, ...]
            for lp in tt_layer_present_all
        ]

        cache_test_pass = check_kv_cache(
            pytorch_layer_present,
            tt_layer_present_all,
            generation_start_pos,
            generation_length,
            seq_len,
            mode == "prefill",
            pcc,
        )
    all_tests_pass = all_tests_pass and cache_test_pass
    if all_tests_pass:
        logger.info(f"{llama_version} output Passed!")
    else:
        gc.collect()
        logger.warning(f"{llama_version} output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "cluster_shape, mesh_device", [pytest.param((4, 8), (8, 4), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("llama_version", ("llama3-tg", "llama3.1-tg"))
@pytest.mark.parametrize(
    "pcc,n_layers",
    [
        (0.995, 1),
        (0.993, 2),
        (0.993, 4),
        (0.993, 8),
        (0.99, 80),
    ],
    ids=("1L", "2L", "4L", "8L", "80L"),
)
@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (32, 1),
        #  (1, 32), (1, 256), (1, 8192), (1, 32768), (1, 128 * 1024)
    ],
    ids=[
        "decode",
        # "prefill_32", "prefill_256", "prefill_8k", "prefill_32k", "prefill_128k"
    ],
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    ((32, 2048), (16, 8192), (16, 32 * 1024), (16, 128 * 1024)),
    ids=(
        "short_context",
        "mid_long_context",
        "long_context",
        "super_long_context",
    ),
)
@pytest.mark.parametrize(
    "prompt_file",
    ("models/demos/t3000/llama2_70b/demo/data/a_tale_of_two_cities.txt", None),
    ids=("prompt_input", "rand_input"),
)
def test_LlamaModel_inference(
    batch,
    seq_len,
    pcc,
    n_layers,
    mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    prompt_file,
    cluster_shape,
    use_program_cache,
):
    if seq_len == 1 and batch != max_batch_size:
        pytest.skip(f"Input batch size should match max_batch_size")

    if batch == 1 and seq_len > max_context_len:
        pytest.skip(f"Prefill with seq_len={seq_len} is not supported with short context")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(mesh_device, model_config)

    run_test_LlamaModel_inference(
        mesh_device,
        cluster_shape,
        batch,
        seq_len,
        pcc,
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        prompt_file=prompt_file,
    )
