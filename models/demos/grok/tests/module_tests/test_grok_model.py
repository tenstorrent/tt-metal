import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.grok.tt.ccl import CCL_Manager
from models.demos.grok.tt.model import Transformer
from models.demos.grok.tt.model_config import TtModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (True,),
    ids=("paged_attention",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 2048}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_grok_model_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    prompts = ["What is your favorite condiment? "] * batch_size

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 64  # Load and run the full model

    # Load state dict for both attention and MLP/MoE components
    state_dict = model_args.load_weights_to_state_dict_no_experts()
    state_dict = model_args.load_experts_weights_to_state_dict(state_dict)

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.num_devices == 32 and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    tt_ccl = CCL_Manager(mesh_device)
    tt_model = Transformer(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        args=model_args,
        dtype=dtype,
        paged_attention_config=paged_attention_config,
    )
    logger.info(f"Model weights loaded")

    # Setup sampling and inputs

    # Grok 2.5 uses a SGLang tiktoken style tokenizer, see thread for workaround: https://huggingface.co/xai-org/grok-2/discussions/27
    tokenizer = AutoTokenizer.from_pretrained("alvarobartt/grok-2-tokenizer")
    encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
    encoded_prompts_tensor = torch.tensor(encoded_prompts)

    current_pos = torch.tensor([0 for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    tt_decode_input = ttnn.from_torch(
        encoded_prompts_tensor[:, :1].reshape(1, 1, 1, batch_size),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    all_outputs = []
    for i in range(1, max_seq_len + 1):
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        logger.info(f"[Model] Generating token {i}")
        logits = tt_model.ttnn_decode_forward(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            page_table=page_table_tt,
        )
        logger.info(f"[Model] Logit at index {i} generated")

        logits_host = tt_model.process_output_decode(logits, batch_size)
        next_token = torch.argmax(logits_host[:, -1], dim=-1)  # [batch_size] tensor
        next_token_text = tokenizer.decode(next_token.tolist())
        logger.info(f"Generated token {i}: text='{next_token_text[0]}'")

        # Increment current_pos
        current_pos = torch.tensor([i for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.num_devices == 32 and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        if i in range(len(encoded_prompts[0])):
            tt_decode_input = ttnn.from_torch(
                encoded_prompts_tensor[:, i : i + 1].reshape(1, 1, 1, batch_size),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            all_outputs.append(prompts[0][i : i + 1])
        else:
            tt_decode_input = ttnn.from_torch(
                next_token.reshape(1, 1, 1, batch_size),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            all_outputs.append(next_token_text)

        logger.info(f"Generated output: {all_outputs}")
