# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import transformers
import transformers.generation.utils
import transformers.models.mistral.modeling_mistral
from loguru import logger

import ttnn

from ....blocks.rope import RopeConfig
from ....encoders.mistral3.model_mistral3 import Mistral3Encoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache, tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "skip_layers"),
    [
        pytest.param((1, 4), 0, id="1x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        # pytest.param(False, id="unmasked"),
    ],
)
def test_generation(*, mesh_device: ttnn.MeshDevice, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    tp_axis = 1
    max_length = 40

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    tokenizer = transformers.LlamaTokenizerFast.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="tokenizer")
    assert isinstance(tokenizer, transformers.LlamaTokenizerFast)

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    config = torch_model.model.language_model.config

    num_layers = len(torch_model.model.language_model.layers)
    del torch_model.model.language_model.layers[num_layers - skip_layers :]

    generation_config = torch_model.generation_config
    assert isinstance(generation_config, transformers.GenerationConfig)

    model = Mistral3Encoder(
        vocab_size=config.vocab_size,
        head_size=config.head_dim,
        embed_size=config.hidden_size,
        ff_size=config.intermediate_size,
        num_layers=config.num_hidden_layers - skip_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        norm_eps=config.rms_norm_eps,
        attn_qkv_bias=False,
        attn_out_bias=False,
        rope_config=RopeConfig(theta=config.rope_theta),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    state_dict = torch_model.state_dict()
    state_dict = Mistral3Encoder.convert_state(state_dict)
    if not cache.initialize_from_cache(
        tt_model=model,
        torch_state_dict=state_dict,
        model_name="flux2",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        model.load_torch_state_dict(state_dict)

    # This makes unmasked generation more similar in the two implementations, possibly because
    # `transformers` masks out padding tokens.
    tokenizer.pad_token_id = generation_config.bos_token_id

    out = tokenizer.__call__(
        ["Once upon a time", "Hello"],
        padding="longest",
        # padding side does not matter for our implementation but the
        # transformers library complains if right padding is used
        padding_side="left",
        return_tensors="pt",
        return_attention_mask=True,
    )
    tokens = out["input_ids"].to(torch_model.device)
    mask = out["attention_mask"].to(torch_model.device) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    generation_config.max_length = max_length
    generation_config.repetition_penalty = None  # repetition penalty is not implemented
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = True

    print("running ttnn model...")

    start_time = time.time()
    tt_out = model.generate(
        tt_tokens,
        mask=tt_mask,
        eos_tokens=generation_config.eos_token_id,
        max_length=generation_config.max_length,
        top_k=generation_config.top_k if generation_config.do_sample else 1,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
    )
    tt_tokens_out = tensor.to_torch(tt_out.tokens)

    print(f"generation took {time.time() - start_time:.2f} seconds")

    for i in range(tt_tokens_out.size(0)):
        print(tokenizer.decode(tt_tokens_out[i]))


@pytest.mark.parametrize(
    ("mesh_device", "skip_layers"),
    [
        pytest.param((1, 4), 0, id="1x4"),
        pytest.param((1, 8), 0, id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        pytest.param(False, id="unmasked"),
    ],
)
def test_guided_generation(*, mesh_device: ttnn.MeshDevice, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    tp_axis = 1
    max_length = 20

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    tokenizer = transformers.LlamaTokenizerFast.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="tokenizer")
    assert isinstance(tokenizer, transformers.LlamaTokenizerFast)

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    config = torch_model.model.language_model.config

    num_layers = len(torch_model.model.language_model.layers)
    del torch_model.model.language_model.layers[num_layers - skip_layers :]

    generation_config = torch_model.generation_config
    assert isinstance(generation_config, transformers.GenerationConfig)

    model = Mistral3Encoder(
        vocab_size=config.vocab_size,
        head_size=config.head_dim,
        embed_size=config.hidden_size,
        ff_size=config.intermediate_size,
        num_layers=config.num_hidden_layers - skip_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        norm_eps=config.rms_norm_eps,
        attn_qkv_bias=False,
        attn_out_bias=False,
        rope_config=RopeConfig(theta=config.rope_theta),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    state_dict = torch_model.state_dict()
    state_dict = Mistral3Encoder.convert_state(state_dict)
    if not cache.initialize_from_cache(
        tt_model=model,
        torch_state_dict=state_dict,
        model_name="flux2",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        model.load_torch_state_dict(state_dict)

    # This makes unmasked generation more similar in the two implementations, possibly because
    # `transformers` masks out padding tokens.
    tokenizer.pad_token_id = generation_config.bos_token_id

    out = tokenizer.__call__(
        ["Once upon a time", "Hello"],
        padding="longest",
        # padding side does not matter for our implementation but the
        # transformers library complains if right padding is used
        padding_side="left",
        return_tensors="pt",
        return_attention_mask=True,
    )
    tokens = out["input_ids"].to(torch_model.device)
    mask = out["attention_mask"].to(torch_model.device) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    generation_config.max_length = max_length
    generation_config.repetition_penalty = None  # repetition penalty is not implemented
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = True

    print("running torch model...")
    torch_mask_input = mask if mask is not None else torch.ones_like(tokens)
    out_ref = torch_model.generate(tokens, attention_mask=torch_mask_input)
    assert isinstance(out_ref, transformers.generation.utils.GenerateOutput)

    tokens_out = out_ref.sequences
    logits = torch.stack(out_ref.logits, dim=1)

    # diffusers somtimes generates longer sequences than max_length, in particular when `max_length
    # = 20` for whatever reason.
    tokens_out = tokens_out[:, :max_length]
    logits = logits[:, : max_length - tokens.size(1)]

    print("running ttnn model...")
    tt_out = model.generate(
        tt_tokens,
        guide=tokens_out,
        mask=tt_mask,
        eos_tokens=generation_config.eos_token_id,
        max_length=generation_config.max_length,
        top_k=generation_config.top_k if generation_config.do_sample else 1,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        return_logits=True,
    )

    tt_tokens_out = tensor.to_torch(tt_out.tokens)
    tt_logits = tensor.to_torch(ttnn.stack(tt_out.logits, dim=1), mesh_axes=[..., tp_axis])

    # To compare generated tokens, remove `guide` in the call to `model.generate`!
    # for i in range(tt_tokens_out.size(0)):
    #     print(tokenizer.decode(tokens_out[i]))
    #     print(tokenizer.decode(tt_tokens_out[i]))

    if mask is not None:
        # Masked positions on the start of the sequence contain random values from computing softmax over all -inf
        # so we remove them before comparison.
        _, s, d = logits.shape
        padded_mask = torch.nn.functional.pad(mask.bool(), [0, s - mask.size(1)], value=True)
        logits = logits.masked_select(padded_mask.unsqueeze(-1)).view([-1, d])
        tt_logits = tt_logits.masked_select(padded_mask.unsqueeze(-1)).view([-1, d])

    assert_quality(logits, tt_logits, ccc=0.9980, relative_rmse=0.063)

    assert tt_tokens_out.eq(tokens_out).all()


@pytest.mark.parametrize(
    ("mesh_device", "batch_size", "skip_layers"),
    [
        pytest.param((1, 1), 2, 32, id="1x1"),
        pytest.param((1, 2), 2, 22, id="1x2"),
        pytest.param((1, 4), 2, 0, id="1x4"),
        pytest.param((1, 8), 2, 0, id="1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "masked",
    [
        pytest.param(True, id="masked"),
        pytest.param(False, id="unmasked"),
    ],
)
def test_transformer(*, mesh_device: ttnn.MeshDevice, batch_size: int, skip_layers: int, masked: bool) -> None:
    torch.manual_seed(0)

    sequence_length = 512
    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = (
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
        )
        if tp_axis is not None
        else None
    )

    torch_model = transformers.Mistral3ForConditionalGeneration.from_pretrained(
        "black-forest-labs/FLUX.2-dev", subfolder="text_encoder"
    )
    config = torch_model.model.language_model.config

    num_layers = len(torch_model.model.language_model.layers)
    del torch_model.model.language_model.layers[num_layers - skip_layers :]

    model = Mistral3Encoder(
        vocab_size=config.vocab_size,
        head_size=config.head_dim,
        embed_size=config.hidden_size,
        ff_size=config.intermediate_size,
        num_layers=config.num_hidden_layers - skip_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        norm_eps=config.rms_norm_eps,
        attn_qkv_bias=False,
        attn_out_bias=False,
        rope_config=RopeConfig(theta=config.rope_theta),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    state_dict = torch_model.state_dict()
    state_dict = Mistral3Encoder.convert_state(state_dict)
    if not cache.initialize_from_cache(
        tt_model=model,
        torch_state_dict=state_dict,
        model_name="flux2",
        subfolder="text_encoder",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        model.load_torch_state_dict(state_dict)

    tokens = torch.randint(0, config.vocab_size, [batch_size, sequence_length])
    lengths = torch.randint(sequence_length // 4, 3 * sequence_length // 4, [batch_size])
    mask = torch.arange(sequence_length).flip([0]) < lengths.unsqueeze(1) if masked else None

    tt_tokens = tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    tt_mask = tensor.from_torch(mask, device=mesh_device) if mask is not None else None

    logger.info("running ttnn model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        mask=tt_mask,
        skip_final_linear=True,
        output_hidden_states=True,
    )
    tt_hidden_states_torch = [tensor.to_torch(t) for t in tt_hidden_states]

    logger.info("running torch model...")
    torch_mask_input = mask if mask is not None else torch.ones_like(tokens)
    with torch.no_grad():
        hidden_states = torch_model.forward(
            tokens, attention_mask=torch_mask_input, output_hidden_states=True
        ).hidden_states

    if mask is not None:
        # Masked positions on the start of the sequence contain undefined values from computing softmax over all -inf
        # so we remove them before comparison.
        _, _, d = hidden_states[0].shape
        hidden_states = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in hidden_states]
        tt_hidden_states_torch = [t.masked_select(mask.unsqueeze(-1)).view([-1, d]) for t in tt_hidden_states_torch]

    assert len(hidden_states) == len(tt_hidden_states_torch)

    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x, tt_x, pcc=0.998, relative_rmse=0.061)
