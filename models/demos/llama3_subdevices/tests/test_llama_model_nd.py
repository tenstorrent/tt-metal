# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    HostEmbedding,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.sampling import TTSampling
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.utility_functions import skip_for_blackhole


@torch.no_grad()
@skip_for_blackhole("Untested on blackhole!")
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "num_iters",
    (500,),
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
    "sampling_params",
    [
        {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
    ],
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(LlamaOptimizations.accuracy, id="accuracy"),
        # pytest.param(LlamaOptimizations.performance, id="performance"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1344544,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_llama_model_inference(
    num_iters,
    max_seq_len,
    batch_size,
    paged_attention,
    sampling_params,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    mode_accuracy = optimizations == LlamaOptimizations.accuracy
    instruct = True
    dummy_weights = True

    top_k = sampling_params["top_k"]
    if isinstance(top_k, int):
        top_k = [top_k] * batch_size
    top_p = sampling_params["top_p"]
    if isinstance(top_p, float):
        top_p = [top_p] * batch_size
    temperature = sampling_params["temperature"]
    if isinstance(temperature, float):
        temperature = [temperature] * batch_size
    seed = sampling_params["seed"]

    model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    prompts = ["Test"] * model_args.max_batch_size
    tokenizer = Tokenizer(model_args.tokenizer_path)
    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    generation_start_pos = 0

    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        temperature=temperature,
        tt_ccl=tt_model.tt_ccl,
    )
    logger.info("Model and caches loaded.")

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
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

    try:
        outputs = []
        for i in range(num_iters):
            logger.info(f"[Llama3 Model] Generating token {i}")

            decode_input = model_args.prepare_residual_tensor_decode(
                tt_decode_input,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )

            # Get cos/sin matrices for the current position of each user
            rot_mats = tt_model.rope_setup.get_rm_rot_mats(current_pos)

            # Run TT model
            tt_out = tt_model(
                decode_input,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )
            # Sampling
            tt_out_tok = tt_sampling(tt_out[0], top_k, top_p, seed)

            tt_out_tok_device0 = ttnn.get_device_tensors(tt_out_tok)[0]
            tt_out_tok_cpu = tt_out_tok_device0.cpu(blocking=True, cq_id=0)
            tt_output_torch = ttnn.to_torch(
                tt_out_tok_cpu,
            )

            # Only check user 0, see GH issue #16719
            tt_output_torch = tt_output_torch[..., :1, :]

            outputs.append(tt_output_torch)

        ##### Check outputs #####
        for arr in [outputs]:
            golden = arr[0]
            all_passing = True
            for i in range(len(arr)):
                logger.info(f"Checking output for iteration {i}")

                passing = torch.all(arr[i] == golden)

                if passing:
                    logger.info(f"Output for iteration {i} is equal to golden")
                else:
                    logger.warning(f"Output for iteration {i} is NOT equal to golden")

                all_passing = all_passing and passing

    except Exception as e:
        logger.error(e)
    finally:
        tt_model.tt_ccl.close()

    assert all_passing, "Not all outputs are equal to the golden output"
