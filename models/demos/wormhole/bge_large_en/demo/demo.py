# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, BGE_SEQ_LENGTH

# Sample English texts for BGE demonstration
# BGE models work best with instruction prefix for retrieval tasks
instruction = "Represent this sentence for searching relevant passages: "
inputs = [
    [
        instruction + "Artificial intelligence is transforming how we interact with technology.",
        instruction + "AI is changing the way humans use computers and machines.",
        instruction + "Machine learning algorithms are revolutionizing data analysis.",
        instruction + "Deep learning networks can process complex patterns in data.",
        instruction + "Neural networks mimic the human brain's structure and function.",
        instruction + "Natural language processing enables computers to understand text.",
        instruction + "Computer vision allows machines to interpret visual information.",
        instruction + "The weather is sunny today with clear blue skies.",
    ]
]


def run_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator):
    inputs = inputs * device.get_num_devices()
    config = transformers.BertConfig.from_pretrained(model_name)
    # Set attention implementation if not set (required for reference model)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(
        inputs, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

    # Load reference PyTorch model
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix="", model_location_generator=model_location_generator
    )
    reference_sentence_embeddings = reference_module(
        input_ids,
        extended_attention_mask=extended_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).post_processed_output

    # Load and run TTNN model
    ttnn_module = BGEPerformantRunner(
        device=device,
        model_location_generator=model_location_generator,
        input_ids=input_ids,
        extended_mask=extended_mask,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        model_name=model_name,
    )
    ttnn_module._capture_bge_trace_2cqs()
    ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
    ttnn_sentence_embeddings = ttnn.to_torch(
        ttnn_out, dtype=torch.float32, mesh_composer=ttnn_module.runner_infra.output_mesh_composer
    )

    # Calculate cosine similarity for reference model
    cosine_sim_matrix1 = cosine_similarity(reference_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle1 = np.triu(cosine_sim_matrix1, k=1)
    similarities1 = upper_triangle1[upper_triangle1 != 0]
    mean_similarity1 = similarities1.mean()

    # Calculate cosine similarity for TTNN model
    cosine_sim_matrix2 = cosine_similarity(ttnn_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle2 = np.triu(cosine_sim_matrix2, k=1)
    similarities2 = upper_triangle2[upper_triangle2 != 0]
    mean_similarity2 = similarities2.mean()

    logger.info(f"Mean Cosine Similarity for Reference Model (PyTorch): {mean_similarity1:.4f}")
    logger.info(f"Mean Cosine Similarity for TTNN Model: {mean_similarity2:.4f}")
    similarity_diff = abs(mean_similarity1 - mean_similarity2)
    tolerance = 0.02  # 2% tolerance
    assert (
        similarity_diff < tolerance
    ), f"Cosine similarities differ by {similarity_diff:.4f}, which exceeds tolerance of {tolerance}"
    logger.info(f"✓ Cosine similarities are close (difference: {similarity_diff:.4f})")
    logger.info(f"✓ BGE-large-en-v1.5 demo completed successfully!")


@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("model_name, sequence_length", [("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH)])
def test_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator):
    return run_bge_demo_inference(device, inputs, model_name, sequence_length, model_location_generator)


@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("model_name, sequence_length", [("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH)])
def test_bge_demo_inference_dp(mesh_device, inputs, model_name, sequence_length, model_location_generator):
    return run_bge_demo_inference(mesh_device, inputs, model_name, sequence_length, model_location_generator)


def run_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator):
    """
    Demo function showing how to use BGE model with vLLM integration.

    This demonstrates the vLLM-compatible interface for embedding generation.
    For full vLLM server integration, see generator_vllm.py and vLLM documentation.
    """
    from models.demos.wormhole.bge_large_en.demo.generator_vllm import BGEForEmbedding

    inputs = inputs * device.get_num_devices()
    config = transformers.BertConfig.from_pretrained(model_name)
    # Set attention implementation if not set (required for reference model)
    if not hasattr(config, "_attn_implementation") or config._attn_implementation is None:
        config._attn_implementation = "eager"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Tokenize inputs
    encoded_input = tokenizer(
        inputs, padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    # Initialize vLLM-compatible model
    bge_model = BGEForEmbedding.initialize_vllm_model(
        hf_config=config,
        mesh_device=device,
        max_batch_size=input_ids.shape[0],
        max_seq_len=sequence_length,
        model_location_generator=model_location_generator,
    )

    # Generate embeddings using vLLM interface
    embeddings = bge_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {bge_model.get_embedding_dim()}")
    logger.info(f"✓ BGE vLLM integration demo completed successfully!")

    return embeddings


@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": BGE_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("model_name, sequence_length", [("BAAI/bge-large-en-v1.5", BGE_SEQ_LENGTH)])
def test_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator):
    """
    Test the vLLM integration for BGE embedding model.

    This test demonstrates how to use the BGE model with vLLM's interface,
    which enables OpenAI Embedding API compatibility.
    """
    return run_bge_vllm_demo(device, inputs, model_name, sequence_length, model_location_generator)
