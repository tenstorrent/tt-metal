# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn


def create_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    state_dict=None,
    hf_model_name="BAAI/bge-m3",
):
    """
    BGE-M3 version of create_tt_model that matches tt_transformers interface
    """
    from models.demos.wormhole.bge_m3.tt.model import BgeM3Model
    from models.demos.wormhole.bge_m3.tt.model_config import ModelArgs

    # Create BGE-M3 ModelArgs
    bge_m3_model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        hf_model_name=hf_model_name,
    )

    if not state_dict:
        state_dict = bge_m3_model_args.load_state_dict()

    # Create BGE-M3 model
    model = BgeM3Model(
        args=bge_m3_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
    )

    return bge_m3_model_args, model, state_dict
