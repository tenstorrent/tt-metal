# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Decode speed of FIBO-vlm through `Qwen3VlEncoder.generate`, with the checkpoint's sampling settings.

Timing only: the weights are random, the shapes are the checkpoint's.
"""

from __future__ import annotations

import time

import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.tt_dit.encoders.qwen3vl.model_qwen3vl import Qwen3VlEncoder
from models.tt_dit.encoders.transformer import MAX_CHUNK_SIZE, Cache
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor

CHECKPOINT = "briaai/FIBO-vlm"
PROMPT_LENGTH = 1024
NUM_STEPS = 32


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4"), pytest.param((1, 8), id="1x8")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 32_000_000}],
    indirect=True,
)
def test_decode(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    max_length = PROMPT_LENGTH + NUM_STEPS

    tp_axis = 1
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=default_topology(mesh_device) or ttnn.Topology.Linear)

    hf_config = transformers.AutoConfig.from_pretrained(CHECKPOINT)
    generation_config = transformers.GenerationConfig.from_pretrained(CHECKPOINT)
    encoder = Qwen3VlEncoder(
        Qwen3VlEncoder.config_from_hf(hf_config),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    logger.info("building random torch model...")
    torch_model = transformers.AutoModel.from_config(hf_config.text_config, dtype=torch.bfloat16)
    state = {f"model.language_model.{k}": v for k, v in torch_model.state_dict().items()}
    # The checkpoint ties the lm head to the token embedding.
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    encoder.load_torch_state_dict(Qwen3VlEncoder.convert_state(state))
    del torch_model, state

    torch_prompt = torch.randint(0, hf_config.text_config.vocab_size, (1, PROMPT_LENGTH))
    prompt = tensor.from_torch(torch_prompt, device=mesh_device, dtype=ttnn.uint32)

    # As in `generate`: a fresh cache sized for the whole sequence, the prompt prefilled into it.
    padded_length = -(-max_length // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

    prefill_times = []
    for _ in range(2):
        start = time.perf_counter()
        encoder.forward(
            prompt,
            cache=Cache(device=mesh_device, size=padded_length, batch_size=1),
            skip_final_linear=True,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_times.append(time.perf_counter() - start)

    logger.info(
        f"prefill {PROMPT_LENGTH} tokens: {prefill_times[-1] * 1e3:.1f} ms (first {prefill_times[0] * 1e3:.1f} ms)"
    )
    generated = {}
    for traced in (False, True):
        # Traced twice: the first call captures the trace, the second replays it.
        totals = []
        for _ in range(2 if traced else 1):
            torch.manual_seed(0)
            start = time.perf_counter()
            generated[traced] = encoder.generate(
                torch_prompt,
                mask=None,
                max_length=max_length,
                eos_tokens=None,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                temperature=generation_config.temperature,
                traced=traced,
            ).tokens
            totals.append(time.perf_counter() - start)
        mode = "traced" if traced else "untraced"
        for total, run in zip(totals, ("first", "second"), strict=False):
            logger.info(
                f"generate, {NUM_STEPS} tokens end to end, {mode}, {run}: {total * 1e3:.1f} ms, "
                f"{(total - prefill_times[-1]) / NUM_STEPS * 1e3:.2f} ms per token after the prefill"
            )
    mismatches = generated[True].ne(generated[False]).sum().item()
    logger.info(f"sampled tokens with the same seed, traced against untraced: {mismatches} of {NUM_STEPS} differ")
