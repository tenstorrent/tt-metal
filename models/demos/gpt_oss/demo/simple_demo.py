import pytest
import torch
from loguru import logger
from tracy import signpost
from transformers import AutoTokenizer

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import TestFactory
from models.demos.gpt_oss.tt.model import Model
from models.demos.gpt_oss.tt.model_config import ModelArgs
from models.demos.gpt_oss.tt.model_mp import ModelWithMP
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format


@pytest.mark.timeout(6000)
@pytest.mark.parametrize("batch_size,", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 8), (4, 8)], ids=["1x8_mesh", "4x8_mesh"])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
    ],
    indirect=True,
    ids=["fabric_2d"],
)
@pytest.mark.parametrize(
    "prompt, iters",
    [
        ("Give me instructions to make the best birthday gift in the world. It must be unique.", 100),
        (
            "Write an essay desribing how LLMs work. Talk about how they are trained & can be fine tuned. Make it as detailed as possible.",
            200,
        ),
    ],
    ids=["prompt1", "prompt2"],
)
def test_gpt_oss_120b_mp(mesh_device, mesh_shape, batch_size, device_params, prompt, iters):
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    use_model_parallelism = True
    setup = TestFactory.setup_test(
        mesh_device, mesh_shape=mesh_shape, use_real_weights=False, use_model_parallelism=use_model_parallelism
    )
    model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=False, use_model_parallelism=use_model_parallelism)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(mp=mesh_device.shape[1], ep=mesh_device.shape[0], sp=1, tp=1),
        mp_enabled=True,
    )
    config = setup["config"]
    ccl_manager = setup["ccl_manager"]

    dtype = ttnn.bfloat8_b

    # Create model with model parallelism enabled
    tt_model = ModelWithMP(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict={},
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=str(model_args.weight_cache_path(dtype)),
        paged_attention_config=None,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=batch_size,
        users_row_sharded=False,
        use_throughput_experts=False,
        mesh_shape=mesh_shape,
    )

    # Load input_ids for testing
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    current_seq_len = input_ids.shape[1]

    (
        tt_embeds,
        rot_mats_global,
    ) = tt_model.prepare_inputs_prefill(
        tokens=input_ids
    )[:2]
    if tt_embeds.shape[2] % 32 != 0:
        # Pad to next multiple of 32 for non-row-sharded b<32
        pad_length = 32 - (tt_embeds.shape[2] % 32)
        tt_embeds = ttnn.pad(tt_embeds, [(0, pad_length), (0, 0)], value=0)
        logger.info(f"Padded tt_embeds to shape {tt_embeds.shape} as seqlen should be a multiple of 32.")
    signpost("prefill_start")
    # Run prefill
    tt_logits = tt_model.ttnn_prefill_forward(
        tt_embeds,
        user_id=0,
        rot_mats_global=rot_mats_global,
    )
    signpost("prefill_end")
    tt_logits_torch = ttnn.to_torch(tt_logits)
    next_token = tt_logits_torch.argmax(dim=-1)[:, :, current_seq_len]
    output = ""
    for iter in range(iters):
        signpost("decode_iteration_start")
        output += tokenizer.decode(next_token[0])
        logger.info(f"Decode iteration {iter}, next token ID: {next_token}, str = {tokenizer.decode(next_token[0])}")
        current_pos_torch = torch.tensor([[current_seq_len + iter]], dtype=torch.int32)
        tokens, current_pos_tt, rope_idx, _ = tt_model.prepare_inputs_decode(next_token, current_pos_torch)
        tt_logits = tt_model.ttnn_decode_forward(tokens, current_pos_tt, rot_mat_idxs=rope_idx)[0]
        tt_logits_torch = ttnn.to_torch(tt_logits)
        signpost("decode_iteration_end")
        next_token = tt_logits_torch.argmax(dim=-1)[:, :, 0]
    logger.info(f"Prompt : {prompt}")
    logger.info(f"Output : {output}")


@pytest.mark.timeout(6000)
@pytest.mark.parametrize("batch_size,", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 8), (4, 8)], ids=["1x8_mesh", "4x8_mesh"])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
    ids=["fabric_1d_ring"],
)
@pytest.mark.parametrize(
    "prompt, iters",
    [
        ("Give me instructions to make the best birthday gift in the world. It must be unique.", 100),
        (
            "Write an essay desribing how LLMs work. Talk about how they are trained & can be fine tuned. Make it as detailed as possible.",
            200,
        ),
    ],
    ids=["prompt1", "prompt2"],
)
def test_gpt_oss_120b_tp(mesh_device, mesh_shape, batch_size, device_params, prompt, iters):
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    use_model_parallelism = False
    setup = TestFactory.setup_test(
        mesh_device, mesh_shape=mesh_shape, use_real_weights=False, use_model_parallelism=use_model_parallelism
    )
    model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=False, use_model_parallelism=use_model_parallelism)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(mp=1, ep=mesh_device.shape[0], sp=1, tp=mesh_device.shape[1]),
        mp_enabled=False,
    )
    config = setup["config"]
    ccl_manager = setup["ccl_manager"]

    dtype = ttnn.bfloat8_b
    state_dict = {}
    load_model = False
    if load_model:
        state_dict_hf = model_args.load_state_dict(
            weights_path=model_args.model_path,
            dummy_weights=False,
            convert_to_meta_format=False,  # HF format for reference
        )

        # Convert to meta format for TT model
        state_dict = convert_hf_qkv_to_meta_format(state_dict_hf, config.head_dim)

    # Create model with model parallelism enabled
    tt_model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=str(model_args.weight_cache_path(dtype)),
        paged_attention_config=None,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=batch_size,
        users_row_sharded=False,
        use_throughput_experts=False,
    )

    # Load input_ids for testing
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    current_seq_len = input_ids.shape[1]

    (
        tt_embeds,
        rot_mats_global,
    ) = tt_model.prepare_inputs_prefill(
        tokens=input_ids
    )[:2]
    if tt_embeds.shape[2] % 32 != 0:
        # Pad to next multiple of 32 for non-row-sharded b<32
        pad_length = 32 - (tt_embeds.shape[2] % 32)
        tt_embeds = ttnn.pad(tt_embeds, [(0, pad_length), (0, 0)], value=0)
        logger.info(f"Padded tt_embeds to shape {tt_embeds.shape} as seqlen should be a multiple of 32.")

    signpost("prefill_start")
    # Run prefill
    tt_logits = tt_model.ttnn_prefill_forward(
        tt_embeds,
        user_id=0,
        rot_mats_global=rot_mats_global,
    )
    signpost("prefill_end")
    mesh_composer_dims = (0, 1)
    tt_logits_torch = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=mesh_composer_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )[:, :1, :, :]
    next_token = tt_logits_torch.argmax(dim=-1)[:, :, current_seq_len]
    output = ""
    for iter in range(iters):
        signpost("decode_iteration_start")
        output += tokenizer.decode(next_token[0])
        logger.info(f"Decode iteration {iter}, next token ID: {next_token}, str = {tokenizer.decode(next_token[0])}")
        current_pos_torch = torch.tensor([[current_seq_len + iter]], dtype=torch.int32)
        tokens, current_pos_tt, rope_idx, _ = tt_model.prepare_inputs_decode(next_token, current_pos_torch)
        tt_logits = tt_model.ttnn_decode_forward(tokens, current_pos_tt, rot_mat_idxs=rope_idx)[0]
        tt_logits_torch = ttnn.to_torch(
            tt_logits,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=mesh_composer_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )[:, :1, :, :]
        signpost("decode_iteration_end")
        next_token = tt_logits_torch.argmax(dim=-1)[:, :, 0]
    logger.info(f"Prompt : {prompt}")
    logger.info(f"Output : {output}")
