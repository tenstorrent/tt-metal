import os
from time import perf_counter

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.utility_functions import nearest_y

from ...reference.hf_utils import get_state_dict, load_tokenizer
from ...reference.modeling_gpt_oss import GptOssRotaryEmbedding
from ...tt.ccl import CCLManager
from ...tt.model import Model
from ...tt.rope import ApplyRotaryPosEmb

local_model_path = "models/demos/gpt_oss/reference"
tensor_cache_dir = (
    os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16") + "/ttnn_cache_demo"
)
local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")
tokenizer = load_tokenizer(local_weights_path)

BASE_PROMPT_LEN = 81  # Send empty prompt to apply_chat_template


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "generation_length",
    [
        200,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf16", "bf8", "bf4"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_model(
    mesh_device,
    generation_length,
    dtype,
    reset_seeds,
):
    # Prepare the prompt
    prompt = "How many r's in the word 'strawberry'?"

    padded_prefill_seq_len = nearest_y(len(prompt) + BASE_PROMPT_LEN, ttnn.TILE_SIZE)
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        max_length=padded_prefill_seq_len,
    )
    # print(f"Input ids: {inputs.input_ids}")
    print(f"Detokenized input: {tokenizer.decode(inputs.input_ids[0])}")

    decode_start_pos = (inputs.attention_mask[0] == 0).nonzero(as_tuple=True)[0][0].item()

    # Create configuration
    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)

    # Create input tensors (prefill)
    mask = torch.triu(torch.full((1, 1, padded_prefill_seq_len, padded_prefill_seq_len), -float("inf")), diagonal=1)
    sliding_mask = mask + torch.tril(
        torch.full((1, 1, padded_prefill_seq_len, padded_prefill_seq_len), -float("inf")),
        diagonal=-config.sliding_window,
    )

    rope_temp_tensor = torch.randn(1)
    RopeEmbeddings = GptOssRotaryEmbedding(config)
    position_ids = torch.arange(padded_prefill_seq_len).unsqueeze(0)
    cos, sin = RopeEmbeddings(rope_temp_tensor, position_ids)

    tt_mask = ttnn.from_torch(mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sliding_mask = ttnn.from_torch(sliding_mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    apply_rope = ApplyRotaryPosEmb(config)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    # Initialize TT model
    weights_type = "/real"
    model_state_dict = get_state_dict(local_weights_path, "", dtype=torch.bfloat16)
    ccl_manager = CCLManager(mesh_device)
    print("Initializing TT model")
    tt_model = Model(
        mesh_device,
        config,
        model_state_dict,
        ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_dir + weights_type,
    )
    print("TT model initialized successfully")

    # Convert to TTNN tensors
    tt_input_ids = ttnn.from_torch(
        inputs.input_ids, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )

    # Prefill
    print("Running TT model")
    tt_output = tt_model(
        input_ids=tt_input_ids,
        attention_masks={"full_attention": tt_mask, "sliding_attention": tt_sliding_mask},
        position_embeddings=rope_stuff,
    )

    # Handle prefill output
    tt_output_tensor = ttnn.get_device_tensors(tt_output)[0]

    outputs = ""

    print(f"Checking outputs:")
    prefill_out = ttnn.to_torch(tt_output_tensor)[:, decode_start_pos - 1, :]
    prefill_out_token_id = torch.argmax(prefill_out.float(), dim=-1)
    prefill_token_out = tokenizer.decode(prefill_out_token_id.flatten())
    outputs += prefill_token_out
    print(f"Prefill token output: {prefill_token_out}")

    # Generate
    iteration = 0
    prev_token_id = prefill_out_token_id.unsqueeze(0)
    while iteration < generation_length:
        cur_pos = decode_start_pos + iteration
        cur_seq_len = cur_pos + 1

        # Prepare inputs for the next iteration
        mask = torch.triu(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=1)[..., -1:, :]
        sliding_mask = (
            mask
            + torch.tril(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=-config.sliding_window)[
                ..., -1:, :
            ]
        )  # Only for 1 token because decode is causal

        position_ids = torch.tensor([cur_pos]).unsqueeze(0)
        cos, sin = RopeEmbeddings(rope_temp_tensor, position_ids)

        tt_mask = None  # No causal mask needed in decode mode
        tt_sliding_mask = ttnn.from_torch(
            sliding_mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        rope_stuff = (apply_rope, tt_cos, tt_sin)

        tt_input_id = ttnn.from_torch(
            prev_token_id, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )

        # Get output
        ta = perf_counter()
        tt_output = tt_model(
            input_ids=tt_input_id,
            attention_masks={"full_attention": tt_mask, "sliding_attention": tt_sliding_mask},
            position_embeddings=rope_stuff,
            position_idx=cur_pos if iteration == 0 else None,  # Only for the first iteration
        )

        # Handle output
        tt_output_tensor = ttnn.get_device_tensors(tt_output)[0]

        tt_output_tensor = ttnn.to_torch(tt_output_tensor)[:, 0, :]
        tb = perf_counter()
        print(f"Iteration {iteration} took {tb - ta:.4f} seconds and t/s: {1 / (tb - ta):.2f}")
        output_token_id = torch.argmax(tt_output_tensor.float(), dim=-1)
        output_token = tokenizer.decode(output_token_id.flatten())
        outputs += output_token
        print(f"Output: {outputs}")
        prev_token_id = output_token_id.unsqueeze(0)

        iteration += 1
