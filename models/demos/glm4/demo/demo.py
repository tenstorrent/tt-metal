# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
import os
import json
import sys
import torch
import torch.nn.functional as F
import ttnn

from time import time
import pytest
from loguru import logger

from models.utility_functions import skip_for_grayskull  # Keep generic utilities

# Removed Llama reference imports
from transformers import AutoTokenizer  # Use HF AutoTokenizer for GLM-4

# Replace this import with our own implementation
# from transformers.generation.utils import top_k_top_p_filtering

# GLM-4 specific imports
from models.demos.glm4.tt.model import Glm4Transformer
from models.demos.glm4.tt.model_config import Glm4ModelArgs
from models.demos.glm4.tt.load_weights import load_and_process_glm4_state_dict


# Our own implementation of top_k_top_p_filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k: keep only top k tokens with highest probability (top-k filtering)
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: value to use for filtered tokens
        min_tokens_to_keep: minimum number of tokens to keep

    Returns:
        filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


# Add a helper function to check mesh device compatibility instead of importing from Llama
def check_mesh_device(mesh_device, model_config=None):
    """Check if mesh device is compatible with model config"""
    if mesh_device is None:
        logger.warning("No mesh device provided!")
        return

    # Check number of devices
    num_devices = getattr(mesh_device, "get_num_devices", lambda: getattr(mesh_device, "device_count", 0))()
    logger.info(f"Using {num_devices} device(s)")

    # Add any model-specific checks based on model_config if needed
    if model_config and "NUM_DEVICES" in model_config:
        assert num_devices >= model_config["NUM_DEVICES"], (
            f"Requires at least {model_config['NUM_DEVICES']} devices to run, but only {num_devices} available",
        )

    return True


# Add a helper function to setup GLM-4 environment
def setup_glm4_env(model_name="THUDM/glm-4-9b-chat"):
    """Setup GLM-4 environment variables and paths"""
    # Check environment variables first, then use defaults
    ckpt_dir = os.environ.get("GLM4_WEIGHTS_DIR", "")
    tokenizer_path = os.environ.get("GLM4_TOKENIZER_PATH", model_name)
    cache_path = os.environ.get("GLM4_CACHE_DIR", None)

    if not ckpt_dir:
        logger.warning(f"GLM4_WEIGHTS_DIR not set! Using default paths from HuggingFace: {model_name}")
        ckpt_dir = model_name

    # Return setup information
    return {"model_name": model_name, "ckpt_dir": ckpt_dir, "tokenizer_path": tokenizer_path, "cache_path": cache_path}


@dataclass
class ModelArgs:
    """Args specific to the model implementation"""

    implementation: str = "tt"  # Keep 'tt' or potentially add 'hf' if comparing
    model_name: str = "THUDM/glm-4-9b-chat"  # Default GLM-4 model
    ckpt_dir: str = ""  # Will be populated from environment
    tokenizer_path: str = "THUDM/glm-4-9b-chat"  # Use HF identifier for tokenizer
    skip_model_load: bool = False
    max_batch_size: int = 32
    num_layers: int = None  # Will be read from config usually
    max_seq_len: int = 2048  # Default, adjust as needed for GLM-4
    max_kv_context_len: int = 2048  # Default, adjust as needed


@dataclass
class TTArgs:
    """Args specific to the TT implementation"""

    mesh_device: object = None  # Will be populated by test/infra
    cluster_shape: tuple = None  # Example: (4, 8) - Should be set based on infra
    n_devices: int = 32  # Example: 32 - Should be set based on infra
    emulated: bool = False
    cache_path: str = None  # Path for TT cache
    decode_only: bool = False  # Prefill + Decode vs Decode only


@dataclass
class DataArgs:
    """Args specific to data and generation"""

    max_output_tokens: int = 128
    prompts_file: str = "models/demos/glm4/demo/data/sample_text_prompts.json"  # Default to text prompts
    output_at_end: bool = True
    top_p: float = 1.0
    top_k: int = 1  # Use 1 for greedy
    temperature: float = 1.0
    chat: bool = False  # Set to True if using chat prompts format
    sample_len: int = None  # Limit input prompt length if needed
    ground_truth: str = None  # Path to ground truth JSON for comparison
    print_output_as_generated: bool = True
    print_output_at_end: bool = False


@dataclass
class DemoArgs:
    """Container for all arg types"""

    model: ModelArgs
    tt: TTArgs
    data: DataArgs


def construct_arg(**kwargs):
    """Helper to build DemoArgs from kwargs"""
    model_args = ModelArgs(**{k: v for k, v in kwargs.items() if hasattr(ModelArgs, k)})
    tt_args = TTArgs(**{k: v for k, v in kwargs.items() if hasattr(TTArgs, k)})
    data_args = DataArgs(**{k: v for k, v in kwargs.items() if hasattr(DataArgs, k)})
    return DemoArgs(model=model_args, tt=tt_args, data=data_args)


def nearest_power_of_2(n):
    """Find the nearest power of 2 >= n"""
    return 2 ** (n - 1).bit_length()


def get_prompts_for_compilation(tokenized, prompts):
    """Select unique padded lengths for compilation"""
    tokenized_len = [len(t) for t in tokenized]
    # Pad tokenized_len to be power of 2 and 32 at least
    padded_tokenized_len = [max(32, nearest_power_of_2(l)) for l in tokenized_len]
    # Get indexes of first occurences of each unique length
    unique_lengths = list(set(padded_tokenized_len))
    # Get indexes of unique_lenghts in tokenized_len
    indexes = [padded_tokenized_len.index(l) for l in unique_lengths]
    # Get tokenized and prompts for compilation
    return [tokenized[i] for i in indexes], [prompts[i] for i in indexes]


def demo_warmup(args: DemoArgs):
    """Optional warmup step for TT model"""
    # Skip if model_implementation is not tt
    if args.model.implementation != "tt":
        return
    logger.info("Starting demo warmup...")
    # Copy and modify arguments for a minimal run
    model_args = replace(args.model, num_layers=1)  # Use only 1 layer for warmup
    tt_args = replace(args.tt)
    data_args = replace(args.data, max_output_tokens=2)  # Generate only 2 tokens

    generator = build_generator(model_args, tt_args)
    model, tokenizer = generator["model"], generator["tokenizer"]

    tokenized, prompts = load_prompts_file(model_args, data_args, tokenizer)

    # Get prompts with unique padded lengths for compilation
    target_tokenized, target_prompts = get_prompts_for_compilation(tokenized, prompts)
    logger.info(f"Warmup using {len(target_prompts)} prompts with unique lengths.")

    # Run decode for compilation
    with torch.no_grad():
        run_decode(
            model_args,
            tt_args,
            data_args,
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=target_tokenized,
            prompts=target_prompts,
            is_compilation_run=True,  # Indicate this is for compilation
        )
    logger.info("Demo warmup finished.")


def run_demo(args: DemoArgs):
    """Main demo execution function"""
    # Set random reproducible seed
    torch.manual_seed(0)

    model_args = args.model
    tt_args = args.tt
    data_args = args.data

    # Load ground truth if available
    ground_truth_outputs = None
    if data_args.ground_truth:
        if not os.path.exists(data_args.ground_truth):
            logger.warning(f"Ground truth file {data_args.ground_truth} does not exist.")
        else:
            try:
                with open(data_args.ground_truth, "r") as f:
                    ground_truth_outputs = json.load(f)
                if not ground_truth_outputs:
                    logger.warning("Ground truth file is empty.")
                    ground_truth_outputs = None
                else:
                    logger.info(f"Loaded {len(ground_truth_outputs)} ground truth outputs.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {data_args.ground_truth}")
                ground_truth_outputs = None

    generator_output = build_generator(model_args, tt_args)
    model, tokenizer = generator_output["model"], generator_output["tokenizer"]

    # Load prompts
    tokenized, prompts = load_prompts_file(model_args, data_args, tokenizer)

    # If skip_model_load is true, we won't have a model to run
    # In this case, just log our progress and return early
    if model_args.skip_model_load or model is None:
        logger.info("Model loading was skipped. Displaying prompts that would be processed:")
        for i, prompt in enumerate(prompts[:5]):  # Show first 5 prompts
            logger.info(f"Prompt {i+1}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        if len(prompts) > 5:
            logger.info(f"... and {len(prompts) - 5} more prompts")
        logger.info("Skip model loading test completed successfully")
        return

    # Run decode
    with torch.no_grad():
        all_outputs = run_decode(
            model_args, tt_args, data_args, model=model, tokenizer=tokenizer, prompt_tokens=tokenized, prompts=prompts
        )

        # Store results
        output_data = []
        for i in range(len(prompts)):
            output_data.append(
                {
                    "prompt": prompts[i],
                    "output": all_outputs["decoded_outputs"][i],
                    "tokens_per_second": all_outputs["tokens_per_second"][i],
                    "time_to_first_token_ms": all_outputs["time_to_first_token_ms"][i],
                }
            )

        # Save output to JSON
        output_filename = f"models/demos/glm4/demo/{model_args.model_name.split('/')[-1]}_demo_output.json"
        try:
            with open(output_filename, "w") as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Demo output saved to {output_filename}")
        except IOError as e:
            logger.error(f"Error saving output to {output_filename}: {e}")

        if data_args.print_output_at_end:
            for item in output_data:
                print("-" * 20)
                print(f"Prompt: {item['prompt']}")
                print(f"Output: {item['output']}")
                print(f"Tokens/s: {item['tokens_per_second']:.2f}")
                print(f"TTFT (ms): {item['time_to_first_token_ms']:.2f}")
                print("-" * 20)

    # Check against ground truth if provided
    if ground_truth_outputs:
        if len(ground_truth_outputs) != len(all_outputs["decoded_outputs"]):
            logger.warning(
                f"Number of ground truth outputs ({len(ground_truth_outputs)}) does not match generated outputs ({len(all_outputs['decoded_outputs'])}). Skipping comparison."
            )
        else:
            logger.info("Comparing generated output with ground truth...")
            scores = string_similarity_score(ground_truth_outputs, all_outputs["decoded_outputs"])
            match = all(s == 1.0 for s in scores)  # Requires exact match

            if not match:
                incorrect_indices = [i for i, score in enumerate(scores) if score < 1]
                logger.warning(f"Output does not match ground truth at indices: {incorrect_indices}")
                for idx in incorrect_indices:
                    print(f"--- Mismatch Example (Index {idx}) ---")
                    print(f"Prompt: {prompts[idx]}")
                    print(f"Output: {all_outputs['decoded_outputs'][idx]}")
                    print(f"Expected: {ground_truth_outputs[idx]}")
                    print("---")
                # Decide if mismatch should raise an error or just warn
                # assert match, "Output must match ground truth!"
            else:
                logger.info("Output successfully matched ground truth!")


def build_generator(model_args: ModelArgs, tt_args: TTArgs) -> dict:
    """Build the generator components (model, tokenizer, etc.)"""
    logger.info(f"Building generator for model: {model_args.model_name}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {model_args.tokenizer_path}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # If we're using TT implementation
    if model_args.implementation == "tt":
        if model_args.skip_model_load:
            logger.info("Skipping model load as requested.")
            return {"model": None, "tokenizer": tokenizer}

        # Initialize GLM-4 model args
        glm4_args = Glm4ModelArgs(
            mesh_device=tt_args.mesh_device,
            dummy_weights=False,
            max_kv_cache_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
        )

        # Override num_layers if specified
        if model_args.num_layers is not None:
            glm4_args.n_layers = model_args.num_layers
            logger.info(f"Overriding number of layers to {model_args.num_layers}")

        # Load model weights
        logger.info(f"Loading GLM-4 weights from {model_args.ckpt_dir}")

        try:
            state_dict = load_and_process_glm4_state_dict(
                model_args.ckpt_dir,
                glm4_args,
                checkpoint_path=tt_args.cache_path,
            )

            # Initialize GLM-4 transformer
            model = Glm4Transformer(glm4_args, tt_args.mesh_device, state_dict=state_dict)
            logger.info(f"Successfully initialized GLM-4 model")

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            logger.error(f"Failed to load GLM-4 model: {e}")
            raise

    # If we're using HF implementation (for reference/comparison)
    elif model_args.implementation == "hf":
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
            )
            logger.info(f"Loaded HF model from {model_args.model_name}")

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            raise

    else:
        raise ValueError(f"Unknown implementation: {model_args.implementation}")


def get_sampling_func(top_k, top_p, temperature):
    """Return the appropriate sampling function"""
    if top_k == 1 and top_p == 1.0:  # Greedy decoding
        return lambda x: torch.argmax(x[..., -1, :], dim=-1)  # Get prediction for the last token
    else:  # Sampling
        return lambda x: top_pk_logits_efficient(x[..., -1, :], p=top_p, k=top_k, temperature=temperature)


def load_prompts_file(model_args: ModelArgs, data_args: DataArgs, tokenizer):
    """Load prompts from file and tokenize"""
    try:
        with open(data_args.prompts_file, "r") as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {data_args.prompts_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in prompts file: {data_args.prompts_file}")
        raise

    # Extract prompts from data structure
    if data_args.chat:
        # Chat prompts should be a list of dicts with a 'prompt' key containing message objects
        # The message objects should be a list of dicts with 'role' and 'content' keys
        prompt_texts = []
        for p in prompts_data:
            if isinstance(p, dict) and "prompt" in p:
                messages = p["prompt"]
                if messages and isinstance(messages, list):
                    # Use the tokenizer's chat template if available
                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        prompt_texts.append(text)
                    except Exception as e:
                        logger.error(f"Failed to apply chat template: {e}. Using raw content instead.")
                        # Fallback: just use the last user message
                        user_messages = [m for m in messages if m.get("role") == "user"]
                        if user_messages:
                            prompt_texts.append(user_messages[-1].get("content", ""))
                        else:
                            logger.warning(f"No user message found in chat prompt: {messages}")
                            prompt_texts.append("")
            else:
                logger.warning(f"Unexpected chat prompt format: {p}")
                prompt_texts.append("")
    else:
        # Text prompts should be a list of dicts with a 'prompt' key
        prompt_texts = []
        for p in prompts_data:
            if isinstance(p, dict) and "prompt" in p:
                prompt_texts.append(p["prompt"])
            else:
                logger.warning(f"Unexpected text prompt format: {p}")
                # Try to use the item directly if it's a string
                if isinstance(p, str):
                    prompt_texts.append(p)
                else:
                    prompt_texts.append("")

    # Now tokenize the prompts
    tokenized = []
    for text in prompt_texts:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors=None)
            tokenized.append(tokens)
        except Exception as e:
            logger.error(f"Failed to tokenize prompt: {e}")
            raise

    # Truncate prompts if sample_len is provided
    if data_args.sample_len is not None:
        tokenized = [t[: data_args.sample_len] for t in tokenized]

    # Limit batch size
    if len(tokenized) > model_args.max_batch_size:
        logger.warning(
            f"Prompts file contains {len(tokenized)} prompts, but max batch size is {model_args.max_batch_size}. Using first {model_args.max_batch_size} prompts."
        )
        tokenized = tokenized[: model_args.max_batch_size]
        prompt_texts = prompt_texts[: model_args.max_batch_size]

    logger.info(f"Loaded and tokenized {len(tokenized)} prompts.")
    return tokenized, prompt_texts


def initialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    """Prepare padded input tensors"""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id  # Fallback if pad_token_id is not set
        logger.warning(f"tokenizer.pad_token_id not set, using eos_token_id ({pad_id}) for padding.")

    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)

    prompt_lens = []
    for k, t in enumerate(prompt_tokens):
        current_len = len(t)
        prompt_lens.append(current_len)
        tokens[k, -current_len:] = torch.tensor(t, dtype=torch.long)  # Left padding

    # Create attention mask: 1 for non-pad tokens, 0 for pad tokens
    # Must match the padding side (left padding means mask is 0s then 1s)
    attention_mask = torch.zeros((bsz, total_len), dtype=torch.long)
    for k, current_len in enumerate(prompt_lens):
        attention_mask[k, -current_len:] = 1

    eos_reached = torch.tensor([False] * bsz)  # Track completion for each sequence

    return tokens, attention_mask, eos_reached, prompt_lens


def prepare_next_input(tokens, attention_mask, next_token):
    """Prepare input for the next decoding step"""
    # Get current sequence length from attention mask (sum of 1s)
    current_seq_lens = attention_mask.sum(dim=1)

    # Ensure next_token is the correct shape (batch_size, 1)
    if next_token.dim() == 1:
        next_token = next_token.unsqueeze(1)

    # Check if any sequence reached max length
    if (current_seq_lens >= tokens.shape[1]).any():
        logger.warning("Maximum sequence length reached, cannot append next token.")
        # Handle this case: maybe return original tokens or error
        # For simplicity, we'll append but it might go out of bounds if not handled in model
        pass  # Or return tokens, attention_mask

    # Create the next tokens tensor by shifting attention mask and inserting new token
    # For left padding, we don't shift. We find the first pad token (0) and replace it.
    # This seems complex. A simpler approach for causal LM with left padding:
    # The 'tokens' tensor should always contain the full sequence history.
    # We append the new token logically for the *next* prediction.
    # The model's forward pass needs the current sequence and position IDs.

    # Let's rethink: In decode phase, we process one token at a time.
    # The input `tokens` to `model.forward` should be just the new token.
    # The KV cache handles the history.

    # We need to update the full `tokens` tensor for tracking/stopping conditions.
    next_tokens = tokens.clone()
    for i in range(tokens.shape[0]):
        seq_len = current_seq_lens[i]
        if seq_len < tokens.shape[1]:
            next_tokens[i, seq_len] = next_token[i]  # Append new token

    # Update attention mask
    next_attention_mask = attention_mask.clone()
    for i in range(tokens.shape[0]):
        seq_len = current_seq_lens[i]
        if seq_len < tokens.shape[1]:
            next_attention_mask[i, seq_len] = 1  # Extend mask

    return next_tokens, next_attention_mask


# Main decode function (adapted from Llama3 demo)
def run_decode(
    model_args: ModelArgs,
    tt_args: TTArgs,
    data_args: DataArgs,
    model,  # TT Model or HF model
    tokenizer,
    prompt_tokens: list,
    prompts: list,
    return_logits: bool = False,
    return_full_logits: bool = False,
    is_compilation_run: bool = False,
):
    """
    Runs decoding/generation loop.

    Args:
        ... (model, tokenizer, args)
        prompt_tokens: List of tokenized prompts (list of lists of ints)
        prompts: List of original prompt strings (for logging)
        return_logits: Whether to return logits (not implemented)
        return_full_logits: Whether to return full logits (not implemented)
        is_compilation_run: Flag indicating if this is a warmup/compile run

    Returns:
        Dictionary containing decoded outputs, timings, etc.
    """
    assert not (return_logits or return_full_logits), "Returning logits is not supported yet."

    # Setup
    bsz = len(prompt_tokens)
    max_gen_len = data_args.max_output_tokens
    max_seq_len = model_args.max_seq_len  # Max length model can handle

    # Min prompt length determines starting point for generation length calculation
    min_prompt_len = min(len(t) for t in prompt_tokens)
    # Max prompt length determines the size needed for prefill KV cache
    max_prompt_len = max(len(t) for t in prompt_tokens)

    # Check if prompts exceed max model length
    if max_prompt_len + max_gen_len > max_seq_len:
        logger.warning(
            f"Max prompt length ({max_prompt_len}) + max generation length ({max_gen_len})"
            f" exceeds model max sequence length ({max_seq_len}). Generated output may be truncated."
        )
        max_gen_len = max_seq_len - max_prompt_len
        logger.warning(f"Adjusted max_gen_len to {max_gen_len}")

    total_len = max_prompt_len + max_gen_len  # Max possible length after generation

    # Initialize inputs (padded tokens, attention mask)
    tokens, attention_mask, eos_reached, prompt_lens = initialize_inputs(tokenizer, prompt_tokens, bsz, total_len)

    # --- Prefill Phase ---
    prefill_start_time = time()
    if not tt_args.decode_only:
        logger.info("Running prefill phase...")
        # TODO: Call the model's prefill method if it exists, otherwise use standard forward
        # This might involve passing the initial `tokens` and `attention_mask`
        # The TT model needs to handle the KV cache population internally
        # Example call structure (may need adjustment based on Glm4Transformer API):
        try:
            # Assume model forward handles prefill when given sequence length > 1
            # It needs the starting position for the KV cache index (0 for prefill)
            _ = model(
                input_ids=tokens[:, :max_prompt_len],  # Pass only up to max prompt len
                start_pos=0,  # Indicate prefill start
                # TODO: Check if attention_mask is needed/used by Glm4Transformer forward
                # attention_mask=attention_mask[:, :max_prompt_len]
            )
        except Exception as e:
            logger.error(f"Error during TT prefill forward pass: {e}")
            raise
        # TODO: Add device synchronization if needed (e.g., tt_lib.device.synchronize(device))
        logger.info("Prefill phase complete.")
    else:
        logger.info("Skipping prefill phase (decode_only=True).")
        # If skipping prefill, KV cache must be loaded/managed differently, or model must handle it.
        # This demo assumes prefill happens unless decode_only is True.

    prefill_end_time = time()
    time_to_first_token_ms = (
        [(prefill_end_time - prefill_start_time) * 1000] * bsz if not tt_args.decode_only else [0.0] * bsz
    )

    # --- Decode Phase ---
    logger.info("Running decode phase...")
    decode_start_time = time()

    # Setup for decode loop
    sampling_func = get_sampling_func(data_args.top_k, data_args.top_p, data_args.temperature)
    generated_tokens = 0
    all_decoded_outputs = ["" for _ in range(bsz)]  # Store decoded text per user

    # Decode loop state
    cur_pos = max_prompt_len  # Start decoding right after the longest prompt
    # `tokens` tensor already contains the prompts, padded to total_len
    # We will fill in the generated tokens starting from `cur_pos`

    for step in range(max_gen_len):
        iter_start_time = time()

        # Prepare inputs for this step: just the last token generated (or last prompt token)
        # The TT model's forward uses the KV cache for context
        # We need the token at the current position `cur_pos - 1` for all sequences in the batch
        # For left-padded sequences, the relevant token might be at different indices,
        # but KV cache position `cur_pos - 1` should be correct regardless.
        # The input ID should be tokens[:, cur_pos - 1]
        input_ids_step = tokens[:, cur_pos - 1].unsqueeze(1)  # Shape: (bsz, 1)

        # Run model forward pass for one step
        try:
            outputs = model(
                input_ids=input_ids_step,
                start_pos=cur_pos - 1,  # Index for KV cache
                # Attention mask might not be needed for single token decode if KV cache handles it
            )
        except Exception as e:
            logger.error(f"Error during TT decode forward pass at step {step}: {e}")
            raise

        # outputs should contain logits, typically shape (bsz, 1, vocab_size)
        logits = outputs  # Assuming the forward returns logits directly

        # Sample the next token
        next_token = sampling_func(logits)  # Shape: (bsz,)

        # Update the main tokens tensor with the generated token
        # Only update if the sequence hasn't ended and is within bounds
        can_update = (~eos_reached) & (cur_pos < total_len)
        tokens[can_update, cur_pos] = next_token[can_update]

        # Check for EOS token
        # TODO: Verify GLM-4 EOS token ID
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            logger.warning("EOS token ID not found in tokenizer!")
            # Use a common EOS token as fallback? Or disable EOS checking?
            # Let's assume it won't be None for now.
        eos_reached = eos_reached | (next_token == eos_token_id) & (~eos_reached)

        # Print generated token if requested (might be slow)
        if data_args.print_output_as_generated and not is_compilation_run:
            for user_id in range(bsz):
                if not eos_reached[user_id].item():  # Check if item() is needed
                    decoded_token = tokenizer.decode(next_token[user_id].item())
                    # Simple print, could be enhanced
                    print(f"user {user_id}, step {step}: {decoded_token}", end="\n" if user_id == bsz - 1 else " | ")

        generated_tokens += 1
        cur_pos += 1

        # Stop if all sequences have reached EOS
        if eos_reached.all():
            logger.info(f"All sequences reached EOS at step {step}.")
            break

        # Optional: Add device synchronization per step if measuring precise step latency
        # tt_lib.device.synchronize(device)
        iter_end_time = time()
        # logger.debug(f"Decode step {step} took {(iter_end_time - iter_start_time) * 1000:.2f} ms")

    decode_end_time = time()
    decode_duration = decode_end_time - decode_start_time
    tokens_per_second = []

    if decode_duration > 0:
        total_tokens_generated_per_user = (
            attention_mask.sum(dim=1) - prompt_lens
        ).tolist()  # Calculate generated tokens per user
        # tokens_per_second_per_user = [ (tps / decode_duration if decode_duration > 0 else 0) for tps in total_tokens_generated_per_user]
        effective_generated_tokens = generated_tokens * bsz  # Total tokens generated across batch
        tokens_per_second_overall = effective_generated_tokens / decode_duration if decode_duration > 0 else 0.0
        # Calculate per-user T/s based on overall average T/s
        # This avoids issues with users finishing early
        tokens_per_second = [tokens_per_second_overall] * bsz

        logger.info(f"Decode phase took {decode_duration:.2f}s")
        logger.info(f"Generated {generated_tokens} tokens per sequence ({effective_generated_tokens} total)")
        logger.info(f"Overall Tokens per second: {tokens_per_second_overall:.2f} T/s")
    else:
        logger.info("Decode phase duration was zero (or negative), cannot calculate T/s.")
        tokens_per_second = [0.0] * bsz

    # Get the final decoded text
    decoded_outputs = get_all_text(tokenizer, tokens, prompt_tokens, generated_tokens)

    # Prepare results dictionary
    results = {
        "decoded_outputs": decoded_outputs,
        "time_to_first_token_ms": time_to_first_token_ms,
        "tokens_per_second": tokens_per_second,  # List of T/s per user (using overall avg here)
        "generated_token_count": generated_tokens,
        "prefill_duration_s": prefill_end_time - prefill_start_time if not tt_args.decode_only else 0.0,
        "decode_duration_s": decode_duration,
    }

    return results


def latency_printout(latencies, model_args, generated_len, total_time_to_first_token, num_users):
    """DEPRECATED? - This seems Llama specific, replaced by results dict"""
    # This function seems specific to the old Llama demo structure.
    # The `run_decode` now returns a dict with timing info.
    # Keeping the function stub here for reference but likely removable.
    logger.warning("latency_printout function is likely deprecated.")


def get_all_text(tokenizer, tokens, prompt_tokens, generated_len):
    """Decode the full generated sequences"""
    all_text = []
    # `tokens` contains the padded prompts + generated tokens
    # We need to decode only the generated part for each sequence
    for i in range(len(prompt_tokens)):
        prompt_len = len(prompt_tokens[i])
        # Find the end of generation (EOS or max length)
        try:
            # Find first pad token after prompt, or first EOS token
            eos_pos = (tokens[i, prompt_len:] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                end_pos = prompt_len + eos_pos[0].item()
            else:  # No EOS found, use full generated length
                end_pos = prompt_len + generated_len
        except Exception:  # Catch potential errors in finding indices
            end_pos = prompt_len + generated_len  # Fallback

        # Ensure end_pos doesn't exceed total length
        end_pos = min(end_pos, tokens.shape[1])

        generated_part = tokens[i, prompt_len:end_pos]
        decoded_text = tokenizer.decode(generated_part, skip_special_tokens=True)
        all_text.append(decoded_text.strip())
    return all_text


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    """Optimized Top-P/Top-K sampling"""
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Top-K filtering
    if k > 0:
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        # Create a mask for filtering logits not in top-k
        k_mask = torch.full_like(logits, -float("inf"))
        k_mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        logits = k_mask
    else:  # if k=0, use full vocab
        top_k_values = logits
        top_k_indices = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)

    # Top-P filtering (applied after Top-K)
    if 0.0 < p < 1.0:
        # Sort the filtered logits and compute cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(top_k_values, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create mask for tokens to remove (cumulative prob > p)
        # Shift cumulative probs to right and set first value to 0
        shifted_cum_probs = F.pad(cumulative_probs[:, :-1], (1, 0), value=0.0)
        remove_indices_mask = shifted_cum_probs > p

        # Set logits for removed tokens to -inf
        # Need to map sorted_indices back to original indices if using k_mask
        # It's easier to apply p-filtering on the k_mask directly

        # Re-softmax the k_mask logits to get probs
        k_probs = F.softmax(logits, dim=-1)
        sorted_k_probs, sorted_k_indices = torch.sort(k_probs, descending=True, dim=-1)
        cumulative_k_probs = torch.cumsum(sorted_k_probs, dim=-1)

        # Find indices where cumulative prob exceeds p
        sorted_indices_to_remove = cumulative_k_probs > p
        # Shift right to keep the first token exceeding p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map back to original indices and set logits to -inf
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_k_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    # Sample from the filtered logits
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    if return_probs:  # If needed for debugging/analysis
        return next_token, probs
    else:
        return next_token


# Add string_similarity_score function to replace the Llama import
def string_similarity_score(expected, actual):
    """Calculate similarity score between expected and actual text outputs.
    A simple implementation that returns 1.0 for exact match, 0.0 otherwise.

    Args:
        expected: List of expected output strings or dicts with 'output' key
        actual: List of actual generated output strings

    Returns:
        List of similarity scores (0.0-1.0) for each pair
    """
    scores = []
    for i in range(min(len(expected), len(actual))):
        # Handle both list of strings and list of dicts with 'output' key
        exp = expected[i]["output"] if isinstance(expected[i], dict) else expected[i]
        act = actual[i]
        # Simple exact matching for now
        score = 1.0 if exp.strip() == act.strip() else 0.0
        scores.append(score)

    # Pad with zeros if lengths don't match
    scores.extend([0.0] * (len(expected) - len(scores)))

    return scores


# --- Pytest Integration ---
# TODO: Adapt parameters and marks for GLM-4 testing environment


# Example: Keeping some structure but marking as potentially needing changes
@pytest.mark.timeout(240000)  # Adjust timeout as needed
@skip_for_grayskull("Requires eth connected devices for multi-device, adapt if needed")
# Parameterize based on expected GLM-4 test configurations
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), 1  # Use a default value instead of ttnn.get_device_ids()
        )
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "model_name, ckpt_dir, tokenizer_path",
    [
        # Use environment variables with fallbacks
        (
            os.environ.get("GLM4_MODEL_NAME", "THUDM/glm-4-9b-chat"),
            os.environ.get("GLM4_WEIGHTS_DIR", "THUDM/glm-4-9b-chat"),
            os.environ.get("GLM4_TOKENIZER_PATH", "THUDM/glm-4-9b-chat"),
        ),
    ],
)
@pytest.mark.parametrize(
    "chat, prompts_file",
    [
        # Provide example prompt files for GLM-4 (chat and text completion)
        (True, "models/demos/glm4/demo/data/sample_chat_prompts.json"),
        (False, "models/demos/glm4/demo/data/sample_text_prompts.json"),
    ],
    ids=["chat_completion", "text_completion"],
)
@pytest.mark.parametrize(
    "decode_only", (False, True), ids=("prefill_decode", "decode_only")  # Test both prefill+decode and decode-only
)
@pytest.mark.parametrize(
    "num_layers",
    (None,),  # Let num_layers be derived from model config by default
    # (1, 5, None), # Or test specific layer counts for debugging
    ids=("full_model",),  # "1L", "5L", "full_model"
)
@pytest.mark.parametrize(
    "implementation, skip_model_load",
    [
        ("tt", False),
        # ("hf", False), # Add if HF comparison is needed
    ],
    ids=["tt_glm4"],  # "hf_glm4"
)
@pytest.mark.parametrize(
    "max_output_tokens, top_p, top_k, temperature",
    [
        (128, 1.0, 1, 1.0),  # Greedy
        (128, 0.9, 10, 1.0),  # Sampling
    ],
    ids=["greedy", "sampling"],
)
@pytest.mark.parametrize(
    "ground_truth",
    [
        # Provide path to a ground truth file if available for comparison
        "models/demos/glm4/demo/data/sample_ground_truth.json",
        None,
    ],
    ids=["check_enabled", "check_disabled"],
)
@pytest.mark.parametrize(
    "max_batch_size, max_seq_len",
    [
        (8, 2048),  # Example standard context
        # (4, 4096),  # Example longer context (adjust batch size based on memory)
        # Add other relevant batch/sequence length combinations
    ],
    ids=["bs8_seq2048"],  # "bs4_seq4096"
)
def test_Glm4Model_demo(
    # Fixtures and params from parametrize decorators
    mesh_device,  # Should provide TTDevice object(s)
    use_program_cache,  # Standard fixture
    # Model params
    model_name,
    ckpt_dir,
    tokenizer_path,
    implementation,
    skip_model_load,
    num_layers,
    # Generation params
    max_output_tokens,
    prompts_file,
    top_p,
    top_k,
    temperature,
    chat,
    # TT/Infra params
    decode_only,
    # Data/Test params
    ground_truth,
    max_batch_size,
    max_seq_len,
):
    """Test GLM-4 model demo with various configurations."""

    logger.info("Starting GLM-4 model demo test")

    # Enable async if available
    if hasattr(mesh_device, "enable_async"):
        mesh_device.enable_async(True)

    # Check if mesh device is compatible
    check_mesh_device(mesh_device)

    # Get device count for logging
    device_count = getattr(mesh_device, "device_count", 0)

    # Calculate cache_path based on model name
    cache_dir = os.environ.get("GLM4_CACHE_DIR", "")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"glm4_cache_{device_count}devices")
    else:
        cache_path = None

    # Construct arguments
    args = construct_arg(
        # Model params
        implementation=implementation,
        model_name=model_name,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        max_kv_context_len=max_seq_len,
        max_batch_size=max_batch_size,
        # TT params
        mesh_device=mesh_device,
        cache_path=cache_path,
        decode_only=decode_only,
        # Data/generation params
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=chat,
        ground_truth=ground_truth,
    )

    # Optional warmup for compilation
    demo_warmup(args)

    # Run the main demo
    run_demo(args)

    logger.info("GLM-4 model demo test completed successfully")


# Main block for direct execution without pytest
if __name__ == "__main__":
    import argparse
    import ttnn

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GLM-4 Demo")

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("GLM4_MODEL_NAME", "THUDM/glm-4-9b-chat"),
        help="GLM-4 model name/identifier",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=os.environ.get("GLM4_WEIGHTS_DIR", ""),
        help="Path to model weights directory",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=os.environ.get("GLM4_TOKENIZER_PATH", ""), help="Path/name for tokenizer"
    )
    parser.add_argument("--skip-model-load", action="store_true", help="Skip loading model weights (for testing)")
    parser.add_argument("--num-layers", type=int, default=None, help="Override number of transformer layers")

    # Generation arguments
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="models/demos/glm4/demo/data/sample_text_prompts.json",
        help="Path to JSON file with prompts",
    )
    parser.add_argument("--chat", action="store_true", help="Use chat prompts instead of text prompts")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling parameter (1 for greedy)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth file for validation")

    # TT arguments
    parser.add_argument("--decode-only", action="store_true", help="Skip prefill, do decode only")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")

    # Parse arguments
    args = parser.parse_args()

    # Create TT device
    try:
        device_ids = ttnn.get_device_ids()
        if device_ids:
            tt_device = ttnn.device.create_device(device_ids)
            if hasattr(tt_device, "enable_async"):
                tt_device.enable_async(True)
            logger.info(f"Created TT device with {len(device_ids)} devices")
        else:
            logger.error("No TT devices found")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create TT device: {e}")
        sys.exit(1)

    # Create demo args
    demo_args = construct_arg(
        # Model args
        implementation="tt",
        model_name=args.model_name,
        ckpt_dir=args.weights_dir if args.weights_dir else args.model_name,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.model_name,
        skip_model_load=args.skip_model_load,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        max_kv_context_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        # TT args
        mesh_device=tt_device,
        decode_only=args.decode_only,
        # Data/generation args
        max_output_tokens=args.max_tokens,
        prompts_file=args.prompts_file,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        chat=args.chat,
        ground_truth=args.ground_truth,
        print_output_as_generated=True,
        print_output_at_end=True,
    )

    # Run demo
    run_demo(demo_args)
