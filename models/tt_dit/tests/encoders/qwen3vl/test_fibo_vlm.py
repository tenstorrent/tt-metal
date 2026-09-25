# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Text-only generation for FIBO-vlm through `Qwen3VlEncoder`.

FIBO-vlm turns a natural-language prompt into the structured JSON that FIBO consumes. Free-running
greedy decoding cannot be compared token for token: one near-tie flip sends both models down
different but equally valid continuations. So both prefill and decode are asserted teacher-forced on the
reference's own token sequence, and free-running decode is asserted on what it is for -- emitting
the JSON.
"""

from __future__ import annotations

import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlCheckpoint
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

CHECKPOINT = "briaai/FIBO-vlm"
PROMPT = "A red bicycle leaning against a stone wall at sunset."
MAX_NEW_TOKENS = 32
JSON_PREFIX = '{"short_description":'


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_generation(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)

    tp_axis = 1

    ccl_manager = CCLManager(mesh_device, topology=default_topology(mesh_device) or ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(CHECKPOINT)
    torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.float32)
    text_config = torch_model.config.text_config

    model = Qwen3VlCheckpoint(CHECKPOINT).build(
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    prompt = encoded["input_ids"]
    prompt_len = prompt.shape[1]
    max_length = prompt_len + MAX_NEW_TOKENS

    logger.info("running torch model...")
    with torch.no_grad():
        reference = torch_model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_length=max_length,
            do_sample=False,
        )
        # Teacher-forced input is the reference sequence minus its last token; every position from
        # the end of the prompt onward is then predicting a generated token.
        forced = reference[:, :-1]
        out = torch_model.forward(
            input_ids=forced,
            attention_mask=torch.ones_like(forced),
            output_hidden_states=True,
        )
    assert not isinstance(out, tuple)
    hidden_states = list(out.hidden_states or [])
    logger.info(f"torch: {tokenizer.decode(reference[0, prompt_len:])!r}")

    generated_positions = slice(prompt_len - 1, forced.shape[1])

    logger.info("running ttnn model, free-running decode...")
    tt_out = model.generate(
        prompt,
        mask=None,
        eos_tokens=text_config.eos_token_id,
        max_length=max_length,
        top_k=1,
    )
    generated = tokenizer.decode(tt_out.tokens[0, prompt_len:])
    logger.info(f"ttnn:  {generated!r}")

    # Decode did its job if it produced the JSON FIBO consumes.
    assert generated.startswith(JSON_PREFIX)

    logger.info("running ttnn model, teacher-forced prefill...")
    tt_hidden_states = model.forward(
        tensor.from_torch(forced, device=mesh_device, dtype=ttnn.uint32),
        skip_final_linear=True,
        output_hidden_states=True,
    )
    tt_hidden_states_torch = [tensor.to_torch(t) for t in tt_hidden_states]
    assert len(hidden_states) == len(tt_hidden_states_torch)

    for x, tt_x in zip(hidden_states[-4:], tt_hidden_states_torch[-4:], strict=True):
        assert_quality(x[:, generated_positions], tt_x[:, generated_positions], pcc=0.9994, relative_rmse=0.04)

    logger.info("running ttnn model, teacher-forced decode...")
    # `guide` feeds the reference's own tokens back in, so each step's logits line up with the
    # reference logits at the generated positions.
    tt_out = model.generate(
        prompt,
        mask=None,
        eos_tokens=None,
        max_length=reference.shape[1],
        guide=reference,
        return_logits=True,
    )
    assert tt_out.logits is not None

    assert_quality(out.logits[:, generated_positions].float(), tt_out.logits, ccc=0.997, relative_rmse=0.08)
