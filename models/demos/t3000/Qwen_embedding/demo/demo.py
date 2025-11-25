import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.demo.simple_text_demo import prepare_generator_args
from models.tt_transformers.tt.common import create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


def get_QwenEmbeddingArgs():
    class QwenEmbeddingArgs(ModelArgs):
        def __init__(self, *args, **kwargs):
            HF_MODEL = os.getenv("HF_MODEL")
            assert (
                HF_MODEL == "Qwen/Qwen3-Embedding-8B"
            ), f"When QwenEmbeddingArgs is used, HF_MODEL must be Qwen/Qwen3-Embedding-8B, but got {HF_MODEL}"
            super().__init__(*args, **kwargs)

    return QwenEmbeddingArgs


class QwenEmbeddingModel:
    def __init__(self, device, data_parallel=1):
        self.generator_args_config = {
            "num_devices": device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1,
            "data_parallel": data_parallel,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": data_parallel,
            "optimizations": lambda model_args: DecodersPrecision.performance(
                model_args.n_layers, model_args.model_name
            ),
            "max_seq_len": 1024,
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
            "paged_attention": True,
            "num_layers": 10,
        }
        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            processor,
        ) = prepare_generator_args(
            **self.generator_args_config,
            model_factory_fn=lambda *args, **kwargs: create_tt_model(
                *args, **kwargs, ModelArgsClass=get_QwenEmbeddingArgs()
            ),
        )
        self.generator = Generator(self.model, self.model_args, device, self.tokenizer)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 70000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("data_parallel", [1])
def test_qwen_embedding_demo(
    device_params,
    mesh_device,
    data_parallel,
):
    model = QwenEmbeddingModel(mesh_device, data_parallel)

    # Run model to generate embeddings
    test_prompts = [
        "Embedding models convert text into vector representations.",
    ]

    logger.info(f"Testing Qwen3-Embedding-8B with {len(test_prompts)} prompts")

    embeddings = []
    for idx, prompt in enumerate(test_prompts):
        logger.info(f"Processing prompt {idx + 1}/{len(test_prompts)}: {prompt}")

        # Tokenize the prompt
        tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)

        logger.info(f"Prompt tokenized to {len(tokens)} tokens")

        # Run prefill to get embeddings (using the last hidden state)
        # For embedding models, we typically use the model output before the LM head
        logits = model.generator.prefill_forward_text(
            tokens_tensor,
            page_table=model.page_table,
            kv_cache=model.tt_kv_cache,
            prompt_lens=torch.tensor([len(tokens)], dtype=torch.long),
        )

        logger.info(f"Generated embedding with shape: {logits.shape}")
        embeddings.append(logits)

    logger.info(f"Successfully generated {len(embeddings)} embeddings")

    # Verify embeddings have expected dimensions
    for idx, embedding in enumerate(embeddings):
        logger.info(f"Embedding {idx + 1} shape: {embedding.shape}")
        assert embedding.shape[0] == 1, "Batch size should be 1"
        assert embedding.dim() >= 2, "Embedding should have at least 2 dimensions"

    logger.info("Qwen3-Embedding-8B demo completed successfully!")
