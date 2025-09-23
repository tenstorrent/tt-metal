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
from ...utils.general_utils import get_decode_mask

local_model_path = "models/demos/gpt_oss/reference"
tensor_cache_dir = (
    os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16") + "/ttnn_cache_demo"
)
local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")
tokenizer = load_tokenizer(local_weights_path)

BASE_PROMPT_LEN = 81  # Send empty prompt to apply_chat_template


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "generation_length",
    [
        200,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf16", "bf8", "bf4"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 42087296}], indirect=True
)
def test_model(
    mesh_device,
    generation_length,
    dtype,
    reset_seeds,
):
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))
    print("MESH DEVICE!", mesh_device)
    print("MESH SHAPE!", mesh_device.shape)
    tensor_cache_dir = (
        os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")
        + f"/ttnn_cache_{mesh_device.shape[0]}_{mesh_device.shape[1]}"
    )
    local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")

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
    print(f"Input ids: {inputs.input_ids.shape}")
    print(f"Detokenized input: {tokenizer.decode(inputs.input_ids[0])}")

    decode_start_pos = (inputs.attention_mask[0] == 0).nonzero(as_tuple=True)[0][0].item()
    print(f"DEBUG: decode_start_pos: {decode_start_pos}")
    # Create configuration
    config = AutoConfig.from_pretrained(local_weights_path, trust_remote_code=True)

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
    print(f"Input ids: {inputs.input_ids.shape}")
    print(f"DEBUG: decode_start_pos: {decode_start_pos}")

    print(f"Checking outputs:")
    prefill_out = ttnn.to_torch(tt_output_tensor)[:, decode_start_pos - 1, :]
    print(f"Prefill out shape: {prefill_out}")
    prefill_out_token_id = torch.argmax(prefill_out.float(), dim=-1)
    prefill_token_out = tokenizer.decode(prefill_out_token_id.flatten())
    outputs += prefill_token_out
    print(f"Prefill token output: {prefill_token_out}")
    ###### Decode Setup ######
    prev_token_id = prefill_out_token_id.unsqueeze(0)
    cur_pos = decode_start_pos

    # Prepare inputs
    position_ids = torch.tensor([cur_pos]).unsqueeze(0)
    cos, sin = RopeEmbeddings(rope_temp_tensor, position_ids)
    sliding_mask = get_decode_mask(position_ids[0].item(), config.sliding_window)
    sliding_mask = sliding_mask.repeat(1, config.num_attention_heads // mesh_device.shape[1], 1, 1).transpose(1, 2)

    tt_mask = None  # No causal mask needed in decode mode
    tt_sliding_mask = ttnn.from_torch(sliding_mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_position_idx = ttnn.from_torch(position_ids, device=mesh_device, dtype=ttnn.int32)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    tt_input_id = ttnn.from_torch(prev_token_id, device=mesh_device, dtype=ttnn.uint32)

    ###### Compile ######
    def run_decode():
        tt_output = tt_model(
            input_ids=tt_input_id,
            attention_masks={"full_attention": tt_mask, "sliding_attention": tt_sliding_mask},
            position_embeddings=rope_stuff,
            position_idx=tt_position_idx,
        )
        ttnn.plus_one(tt_position_idx)

        return tt_output

    print("Compiling decode model")
    tt_output = run_decode()

    # Reset tensors
    tt_position_idx_reset = ttnn.from_torch(position_ids, dtype=ttnn.int32)
    ttnn.copy_host_to_device_tensor(tt_position_idx_reset, tt_position_idx)

    print("Capturing decode trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_output = run_decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Reset tensors
    tt_position_idx_reset = ttnn.from_torch(position_ids, dtype=ttnn.int32)
    ttnn.copy_host_to_device_tensor(tt_position_idx_reset, tt_position_idx)

    # Generate
    print("Starting generation")
    iteration = 0
    while iteration < generation_length:
        cur_pos = decode_start_pos + iteration

        # Prepare inputs for the next iteration
        position_ids = torch.tensor([cur_pos]).unsqueeze(0)
        cos, sin = RopeEmbeddings(rope_temp_tensor, position_ids)
        sliding_mask = get_decode_mask(position_ids[0].item(), config.sliding_window)
        sliding_mask = sliding_mask.repeat(1, config.num_attention_heads // mesh_device.shape[1], 1, 1).transpose(1, 2)

        # Host tensors
        tt_sliding_mask_in = ttnn.from_torch(sliding_mask, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_cos_in = ttnn.from_torch(cos.unsqueeze(-2), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin_in = ttnn.from_torch(sin.unsqueeze(-2), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_input_id_in = ttnn.from_torch(prev_token_id, dtype=ttnn.uint32)

        # Copy host tensors to device
        ttnn.copy_host_to_device_tensor(tt_sliding_mask_in, tt_sliding_mask)
        ttnn.copy_host_to_device_tensor(tt_cos_in, tt_cos)
        ttnn.copy_host_to_device_tensor(tt_sin_in, tt_sin)
        ttnn.copy_host_to_device_tensor(tt_input_id_in, tt_input_id)

        # Run decode
        ta = perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        tt_output_tensor = ttnn.get_device_tensors(tt_output)[0]
        tt_output_tensor = tt_output_tensor.cpu(blocking=True, cq_id=0)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)[:, 0, :]
        tb = perf_counter()

        print(f"Iteration {iteration} took {tb - ta:.4f} seconds and t/s: {1 / (tb - ta):.2f}")

        output_token_id = torch.argmax(tt_output_tensor.float(), dim=-1)
        output_token = tokenizer.decode(output_token_id.flatten())
        outputs += output_token
        print(f"Output: {outputs}")
        prev_token_id = output_token_id.unsqueeze(0)

        iteration += 1

        if prev_token_id == tokenizer.eos_token_id:
            break

    print("Generation complete")
