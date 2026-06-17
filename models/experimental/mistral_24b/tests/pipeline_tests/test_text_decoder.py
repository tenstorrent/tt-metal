# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode logits PCC test for Mistral-Small-3.1-24B text decoder.

Full model (40 layers + norm + lm_head) vs HuggingFace on synthetic random activations
for 32 decode steps. HF reference calls ``model.norm`` and ``lm_head`` directly.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import DynamicCache

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer
from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import setup_vision_model_args

# Mesh trace region (bytes) by architecture.
TRACE_REGION_SIZE_WORMHOLE = 30_000_000  # 30 MiB
TRACE_REGION_SIZE_BLACKHOLE = 35_000_000  # 35 MiB


def fabric_1d_trace_device_params(*, num_command_queues: int = 1):
    from models.common.utility_functions import is_wormhole_b0

    trace_region_size = TRACE_REGION_SIZE_WORMHOLE if is_wormhole_b0() else TRACE_REGION_SIZE_BLACKHOLE
    return [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": trace_region_size,
            "num_command_queues": num_command_queues,
        }
    ]


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.parametrize(
    "generation_length",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (1024,),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "device_params",
    fabric_1d_trace_device_params(num_command_queues=2),
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_text_decoder(max_seq_len, batch_size, generation_length, page_params, mesh_device, reset_seeds):
    """Decode logits PCC: full model (40 layers + norm + lm_head) vs HF on synthetic activations."""
    pcc_required = 0.98
    dtype = ttnn.bfloat8_b
    mode = Mode.DECODE

    weights = "instruct"
    optimizations = lambda margs: DecodersPrecision.accuracy(margs.n_layers, margs.model_name)
    model_args, _ = setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    state_dict = model_args.load_state_dict()

    reference_model = model_args.reference_transformer(wrap=False, load_checkpoint=True)
    reference_model.eval()
    ref_layers = reference_model.model.layers
    ref_rotary_emb = reference_model.model.rotary_emb
    ref_norm = reference_model.model.norm
    ref_lm_head = reference_model.lm_head

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )
    page_table_tt = ttnn.from_torch(
        page_table,
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

    seqlen = 1
    generation_start_pos = 0
    ref_kv_cache = DynamicCache()

    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
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

    mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
    )

    all_tests_pass = True
    for i in range(generation_length):
        logger.info(f"[Text Decoder] Generating token {i}")

        pt_decode_input = (
            torch.rand(
                batch_size, seqlen, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
            )
            * 2
        ) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.get_residual_mem_config(mode, None),
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
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        )
        ttnn.deallocate(tt_out)

        cache_position = torch.tensor([int(current_pos[0])])
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = ref_rotary_emb(pt_decode_input, position_ids)
        ref_hidden = pt_decode_input
        for layer in ref_layers:
            layer_out = layer(
                hidden_states=ref_hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=ref_kv_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            ref_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        ref_logits = ref_lm_head(ref_norm(ref_hidden))[:, :, : model_args.vocab_size]

        batch_cmp = ref_logits.shape[0]
        passing, pcc_message = comp_pcc(ref_logits, tt_output_torch[:batch_cmp], pcc_required)

        logger.info(comp_allclose(ref_logits, tt_output_torch[:batch_cmp]))
        logger.info(pcc_message)

        if not passing:
            logger.warning(f"Mistral-24B Text Decoder logits failed at position {current_pos[0].item()}")
            all_tests_pass = False

        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch_size)])
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

    assert all_tests_pass, f"Logits PCC below {pcc_required} for one or more decode iterations. Check warnings!"
