# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill/decode for Tracy. Same as text prefill/decode tests with n_layers=1."""

import bz2
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.mistral_24b.tt.generator import MistralGenerator
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer
from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import (
    fabric_1d_trace_device_params,
    setup_vision_model_args,
)
from models.experimental.mistral_24b.tests.pipeline_tests.test_text_prefill_logits import (
    PROMPT_FILE,
    scale_page_params,
)

MESH = [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))]


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.parametrize("seq_len", (1024,), ids=["1k"])
@pytest.mark.parametrize("max_seq_len", (1024,), ids=["1k"])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 32, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
def test_single_layer_prefill(seq_len, max_seq_len, batch_size, page_params, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    page_params = scale_page_params(page_params, seq_len, batch_size)
    optimizations = lambda m: DecodersPrecision.accuracy(m.n_layers, m.model_name)
    model_args, instruct = setup_vision_model_args("instruct", max_seq_len, batch_size, mesh_device, optimizations)
    model_args.n_layers = 1

    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        encoded_prompt = model_args.encode_prompt(f.read(), instruct=instruct)[:seq_len]

    state_dict = model_args.load_state_dict()
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    page_table = torch.argsort(permutation).reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )

    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [[layer.attention.layer_past for layer in tt_model.layers]]
    generator = MistralGenerator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)

    generator.prefill_forward_text(
        torch.tensor(encoded_prompt).unsqueeze(0),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[seq_len],
        enable_trace=False,
        warmup_prefill=False,
    )
    ttnn.synchronize_device(mesh_device)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("max_seq_len", (1024,))
@pytest.mark.parametrize("page_params", [{"page_block_size": 32, "page_max_num_blocks": 1024}])
@pytest.mark.parametrize("device_params", fabric_1d_trace_device_params(num_command_queues=2), indirect=True)
@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
def test_single_layer_decode(max_seq_len, batch_size, page_params, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = Mode.DECODE
    optimizations = lambda m: DecodersPrecision.accuracy(m.n_layers, m.model_name)
    model_args, _ = setup_vision_model_args("instruct", max_seq_len, batch_size, mesh_device, optimizations)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()
    reference_model = model_args.reference_transformer(wrap=False, load_checkpoint=True)

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    page_table_tt = ttnn.from_torch(
        torch.argsort(permutation).reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        ),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

    current_pos = torch.tensor([0 for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    pt_decode_input = (
        torch.rand(batch_size, 1, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)) * 2
    ) - 1
    decode_input = model_args.prepare_residual_tensor_decode(
        pt_decode_input, model_args.get_residual_mem_config(mode, None)
    )
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
    rot_mats_local = (
        tt_model.rope_local_setup.get_rot_mats(current_pos) if hasattr(tt_model, "rope_local_setup") else None
    )

    tt_out = tt_model(
        decode_input,
        current_pos_tensor,
        rot_mats_global=rot_mats,
        rot_mats_local=rot_mats_local,
        mode=mode,
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        batch_size=batch_size,
    )
    ttnn.deallocate(tt_out)
    ttnn.synchronize_device(mesh_device)
