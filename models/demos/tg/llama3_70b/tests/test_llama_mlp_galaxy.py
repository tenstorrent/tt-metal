# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_common import (
    BASE_URL,
    MAX_SEQ_LEN,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_N_LAYER,
    ConcatMesh2DToTensor,
    ShardTensor2dMesh,
    check_mesh_device,
    comp_pcc,
    should_skip_model_load,
)
from models.demos.tg.llama3_70b.tt.llama_common import setup_llama_env
from models.demos.tg.llama3_70b.tt.llama_mlp_galaxy import TtLlamaMLP_galaxy
from models.utility_functions import skip_for_grayskull


class PytorchLlamaMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.layers[layer_num].feed_forward

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def tt_llama_mlp_prepare_inputs(llama_mlp_model, x, mode):
    if mode == "decode":
        num_users = 32
        M, K = num_users, llama_mlp_model.model_config["HIDDEN_SIZE"] // llama_mlp_model.cluster_shape[0]

        core_grid = ttnn.CoreGrid(y=1, x=8)
        act_mem_config = ttnn.create_sharded_memory_config(
            shape=(M // core_grid.y, K // core_grid.x),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        x_multichip = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=llama_mlp_model.mesh_device,
            memory_config=act_mem_config,
            mesh_mapper=ShardTensor2dMesh(
                llama_mlp_model.mesh_device, dims=(3, None), cluster_shape=llama_mlp_model.cluster_shape
            ),
        )
    elif mode == "prefill":
        x_multichip = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=llama_mlp_model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(
                llama_mlp_model.mesh_device, dims=(3, None), cluster_shape=llama_mlp_model.cluster_shape
            ),
        )

    return x_multichip


def run_test_LlamaMLP_inference(
    mesh_device,
    cluster_shape,
    batch,
    seq_len,
    pcc,
    model_config,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
):
    # Prepare paths and devices
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
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # Prepare input
    pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
    pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
    pt_inp_normed = hugging_face_reference_model.layers[UNIT_TEST_LAYER_NUM].ffn_norm(pt_inp)
    mode = "decode" if seq_len == 1 else "prefill"
    if mode == "decode":
        # shape should be (1, seq_len, batch, dim)
        pt_inp_normed = pt_inp_normed.unsqueeze(1).permute(2, 1, 0, 3)
    else:  # prefill
        # shape should be (1, batch, seq_len, dim)
        pt_inp_normed = pt_inp_normed.unsqueeze(0)

    tt_inp = pt_inp_normed.clone()

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaMLP_model = PytorchLlamaMLPModel(hugging_face_reference_model, UNIT_TEST_LAYER_NUM)
    pytorch_out = pytorch_LlamaMLP_model(pt_inp_normed)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaMLP_model = TtLlamaMLP_galaxy(
        mesh_device,
        cluster_shape,
        state_dict,
        BASE_URL,
        UNIT_TEST_LAYER_NUM,
        configuration.dim,
        model_config,
        cache_path=cache_path,
    )

    tt_mlp_input = tt_llama_mlp_prepare_inputs(tt_LlamaMLP_model, tt_inp, mode=mode)
    tt_out = tt_LlamaMLP_model(tt_mlp_input, mode=mode)

    tt_out = ttnn.to_torch(
        tt_out, mesh_composer=ConcatMesh2DToTensor(mesh_device, dims=(3, 1), cluster_shape=cluster_shape)
    )
    tt_out = tt_out[:, 0:1, :, :]

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info(f"{llama_version} TG MLP output Passed!")
    else:
        logger.warning(f"{llama_version} TG MLP output Failed!")
        gc.collect()
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "cluster_shape, mesh_device", [pytest.param((4, 8), (8, 4), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "llama_version",
    (("llama3-tg"),),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    [
        (32, 1, 0.9997),
        (1, 256, 0.9995),
    ],
    ids=[
        "decode",
        "prefill",
    ],
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        # (16, 8192),
    ),
    ids=(
        "short_context",
        # "long_context",
    ),
)
def test_LlamaMLP_inference(
    batch,
    seq_len,
    pcc,
    mesh_device,
    max_batch_size,
    max_context_len,
    llama_version,
    cluster_shape,
    use_program_cache,
):
    if batch > max_batch_size:
        pytest.skip(f"Decode with {batch} users is not supported with large context")

    if batch == 1 and seq_len > max_context_len:
        pytest.skip(f"Prefill with {seq_len=} is not supported with short context")

    if llama_version == "llama2" and seq_len > 2048:
        pytest.skip(f"Llama2 with {seq_len=} is not supported (max 2048)")

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(mesh_device, model_config)
    run_test_LlamaMLP_inference(
        mesh_device,
        cluster_shape,
        batch,
        seq_len,
        pcc,
        model_config,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
