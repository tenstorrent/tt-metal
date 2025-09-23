"""
Pytest test example demonstrating how to use the GPT-OSS demo model with the tt-transformers generator.
This test shows the integration between the demo model and the generator interface.

Usage: pytest models/demos/gpt_oss/examples/demo_with_generator.py -v
"""

import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gpt_oss.reference.hf_utils import get_state_dict, load_tokenizer
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.model import Model
from models.tt_transformers.tt.generator import Generator
from models.utility_functions import nearest_y

# Constants from original test
BASE_PROMPT_LEN = 81  # Send empty prompt to apply_chat_template


def create_model_args(config, mesh_device):
    """Create a simple model args object for the generator"""

    class ModelArgs:
        def __init__(self, config, mesh_device):
            self.max_batch_size = 1
            self.vocab_size = config.vocab_size
            self.max_prefill_chunk_size = 2048
            self.mesh_device = mesh_device

    return ModelArgs(config, mesh_device)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat4_b], ids=["bf4"])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 42087296}], indirect=True
)
def test_demo_with_generator(
    mesh_device,
    dtype,
    reset_seeds,
):
    """Example of using demo model with tt-transformers generator"""

    # Create (1,8) submesh from the pytest-provided (4,8) mesh
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 8)))

    print("=== GPT-OSS Demo with tt-transformers Generator ===")
    print("MESH DEVICE!", mesh_device)
    print("MESH SHAPE!", mesh_device.shape)

    # Setup paths
    local_weights_path = os.environ.get("GPT_OSS_WEIGHTS_PATH", "/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16")

    # Setup tensor cache directory based on actual mesh shape
    tensor_cache_dir = local_weights_path + f"/ttnn_cache_{mesh_device.shape[0]}_{mesh_device.shape[1]}"

    # Load configuration and tokenizer
    config = AutoConfig.from_pretrained(local_weights_path, trust_remote_code=True)
    # config.num_hidden_layers = 1
    tokenizer = load_tokenizer(local_weights_path)
    print("✓ Loaded configuration and tokenizer")

    # Load model weights
    model_state_dict = get_state_dict(local_weights_path, "", dtype=torch.bfloat16)
    print("✓ Loaded model weights")

    # Create demo model
    ccl_manager = CCLManager(mesh_device)
    demo_model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=model_state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_dir + "/real",
    )
    print("✓ Created demo model with generator-compatible interface")

    # Create model args for generator
    model_args = create_model_args(config, mesh_device)

    # Create generator with demo model
    generator = Generator(model=[demo_model], model_args=[model_args], mesh_device=mesh_device, tokenizer=tokenizer)
    print("✓ Created tt-transformers Generator with demo model")

    # Example usage - text generation
    print("\n=== Running Text Generation Example ===")

    # Prepare input using chat template (following original test pattern)
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
    print(f"Input prompt: {prompt}")
    print(f"Detokenized input: {tokenizer.decode(inputs.input_ids[0])}")
    print(f"Input tokens shape: {inputs.input_ids.shape}")

    # Calculate the correct position like original test_demo.py
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        decode_start_pos = (inputs.attention_mask[0] == 0).nonzero(as_tuple=True)[0][0].item()
        correct_last_token_pos = decode_start_pos - 1
    else:
        correct_last_token_pos = inputs.input_ids.shape[1] - 1

    # Prefill forward
    print("Running prefill...")
    batch_size = 1
    prompt_lens = torch.tensor([inputs.input_ids.shape[1]])

    prefill_logits = generator.prefill_forward_text(tokens=inputs.input_ids, prompt_lens=prompt_lens, empty_slots=[0])

    print(f"Prefill output shape: {prefill_logits.shape}")
    print(f"Prefill output: {prefill_logits}")

    # Get the first generated token
    next_token_id = torch.argmax(prefill_logits[0, 0], dim=-1)
    next_token = tokenizer.decode([next_token_id.item()])
    print(f"First generated token: '{next_token}' (ID: {next_token_id.item()})")

    # Decode forward for a few more tokens
    print("\nRunning decode for additional tokens...")
    current_pos = inputs.input_ids.shape[1]
    current_pos = 93
    generated_tokens = [next_token_id.item()]

    for i in range(50):  # Generate 5 more tokens
        decode_tokens = torch.tensor([[next_token_id.item()]])
        start_pos = torch.tensor([current_pos + i])

        decode_logits = generator.decode_forward_text(
            tokens=decode_tokens, start_pos=start_pos, enable_trace=False  # Disable tracing for simplicity
        )
        print(f"Decode logits shape: {decode_logits.shape}")

        next_token_id = torch.argmax(decode_logits[0, 0], dim=-1)
        next_token = tokenizer.decode([next_token_id.item()])
        generated_tokens.append(next_token_id.item())
        print(f"Token {i+1}: '{next_token}' (ID: {next_token_id.item()})")

    # Show full generated text
    full_generated = tokenizer.decode(generated_tokens)
    print(f"\nFull generated sequence: {prompt}{full_generated}")

    print("\n✅ Successfully demonstrated demo model with tt-transformers generator!")
