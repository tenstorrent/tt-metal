# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_attention import TtLlamaAttention
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3.tt.load_checkpoints import convert_meta_to_hf, convert_hf_to_meta, map_hf_to_meta_keys


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_llama_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    ref_model = model_args.reference_attention()
    ref_model.load_state_dict(partial_state_dict)

    from transformers import AutoModelForCausalLM

    hf_transformer = AutoModelForCausalLM.from_pretrained(model_args.DEFAULT_CKPT_DIR)
    hf_model = hf_transformer.model.layers[0].self_attn
    hf_model.eval()

    # Get the state dicts
    ref_state_dict = ref_model.attention.state_dict()  # should contain hf keys and weights
    hf_state_dict = hf_model.state_dict()

    for key in ["k_proj", "q_proj"]:
        for suffix in ["weight", "bias"]:
            print(
                f"{key}.{suffix}: ref matches hf : {torch.allclose(ref_state_dict[key + '.' + suffix], hf_state_dict[key + '.' + suffix])}"
            )

    print(" ".join(f"{x:+3.1f}" for x in ref_state_dict["k_proj.bias"]))
    print(" ".join(f"{x:+3.1f}" for x in hf_state_dict["k_proj.bias"]))
    # seq_len = 1

    # generation_start_pos = 0
    # generation_length = 10
    # all_tests_pass = True

    # # Setup RoPE transformation matrices
    # rope_setup = TtLlamaRotarySetup(
    #     mesh_device,
    #     batch_size,
    #     model_args.head_dim,
    #     model_args.max_seq_len,
    #     model_args.rope_theta,
    #     model_args.use_scaled_rope,
    #     model_args.orig_context_len,
    # )

    # transformation_mats = rope_setup.get_both_trans_mats()

    # page_table_tt = None
    # paged_attention_config = None

    # cos, sin = precompute_freqs(
    #     model_args.head_dim, model_args.max_seq_len * 2, model_args.rope_theta, model_args.use_scaled_rope
    # )
    # freqs_cis = torch.complex(cos, sin)

    # # Initial positions
    # current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])

    # for i in range(generation_length):
    #     # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
    #     pt_attention_input = torch.randn(batch_size, seq_len, model_args.dim) * 0.05

    #     # Get cos/sin matrices for the current position of each user
    #     rot_mats = rope_setup.get_rot_mats(current_pos)

    #     # In this test all users have the same position
    #     freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

    #     reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)
    #     hf_output =

    #     passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    #     logger.info(comp_allclose(reference_output, tt_output_torch))
    #     logger.info(f"PCC: {pcc_message}")
    #     if passing:
    #         logger.info(f"[pos={current_pos[0]}] Llama_Attention Passed!")
    #     else:
    #         logger.warning(f"[pos={current_pos[0]}] Llama_Attention Failed!")
    #         all_tests_pass = False

    #     # Increment position
    #     current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
    #     current_pos_tensor = ttnn.from_torch(
    #         current_pos,
    #         device=mesh_device,
    #         dtype=ttnn.int32,
    #         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    #     )

    #     check_kv_cache = True
    #     if check_kv_cache:
    #         # PyTorch output --------------------------------------------------------------------
    #         pytorch_layer_present = [
    #             reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
    #             reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
    #         ]
    #         # TT hardware execution -------------------------------------------------------------
    #         if paged_attention:
    #             tt_layer_present = [
    #                 (
    #                     ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
    #                         reverse_permutation
    #                     ]
    #                     .reshape(
    #                         model_args.max_batch_size,
    #                         paged_attention_config.max_num_blocks // model_args.max_batch_size,
    #                         model_args.n_kv_heads,
    #                         paged_attention_config.block_size,
    #                         model_args.head_dim,
    #                     )
    #                     .transpose(1, 2)
    #                     .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
    #                         :batch_size, ...
    #                     ]
    #                 )
    #                 for cache in tt_model.layer_past
    #             ]
    #         else:
    #             tt_layer_present = [
    #                 ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
    #                 for cache in tt_model.layer_past
    #             ]
    #         for label, cache_pt, cache_tt in zip(["K", "V"], pytorch_layer_present, tt_layer_present):
    #             cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + i + 1)
    #             cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
    #             cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
    #             does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
    #             logger.info(f"{label} cache output: {output_pcc}")
    #             if does_pass:
    #                 logger.info(f"{label} cache Passed!")
    #             else:
    #                 logger.warning(f"{label} Cache Failed! PCC value is lower than {pcc}")
    #                 all_tests_pass = False

    # if all_tests_pass:
    #     logger.info("Llama Attention output Passed!")
    # else:
    #     logger.warning("Llama Attention output Failed!")
    #     assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
