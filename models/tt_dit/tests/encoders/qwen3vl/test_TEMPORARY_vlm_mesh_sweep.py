# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY: prefill and decode speed of FIBO-vlm at every device count of a Galaxy.

Timing only: the weights are zero, the shapes are the checkpoint's.

Not for committing -- a one-off sweep to pick the mesh the FIBO pipeline should give the VLM.
"""

from __future__ import annotations

import time

import pytest
import torch
import transformers
from loguru import logger

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.tt_dit.encoders.qwen3vl.model_qwen3vl_v2 import Qwen3VlEncoder
from models.tt_dit.encoders.transformer import MAX_CHUNK_SIZE, Cache
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.mesh import reshape_device

CHECKPOINT = "briaai/FIBO-vlm"
PROMPT_LENGTH = 1024
GENERATED_TOKENS = 1024
# Sampling as models/tt_dit/pipelines/fibo/vlm.py runs it, plus the checkpoint's top_k. Without a
# top_k, top-p sorts the whole 151936-token vocabulary on the host, 20 ms per token on one thread --
# a constant that does not shard, so it would flatten the very differences this sweep measures.
TOP_K = 20
TOP_P = 0.9
TEMPERATURE = 0.2

# A tensor-parallel model occupies every device of its mesh only when that mesh is a line: the
# parallel config names one axis, so on a 2D mesh the other axis replicates the model instead of
# sharding it, and a (2,4) run would be a 4-device model spread over 8 devices -- not comparable
# with the rest. So the sweep is over device counts, each in both orientations: 1xN shards along
# axis 1, Nx1 along axis 0. Those two differ in how the logical line lands on the physical 4x8,
# which is the thing being compared.
#
# `mesh_device` gets the smallest physical submesh holding the count; `logical_shape` is what that
# submesh is reshaped to for the run.
_PHYSICAL = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 16: (2, 8), 32: (4, 8)}


@pytest.mark.parametrize(
    ("mesh_device", "logical_shape"),
    [
        pytest.param(_PHYSICAL[count], shape, id=f"{count}dev_{shape[0]}x{shape[1]}")
        for count in _PHYSICAL
        for shape in dict.fromkeys(((1, count), (count, 1)))  # dedupes 1x1
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 32_000_000}],
    indirect=True,
)
def test_mesh_sweep(*, mesh_device: ttnn.MeshDevice, logical_shape: tuple[int, int]) -> None:
    """Generates a fixed GENERATED_TOKENS rather than stopping at an end token, so every
    configuration does the same work."""
    torch.set_num_threads(1)
    torch.manual_seed(0)

    devices = logical_shape[0] * logical_shape[1]
    cache_length = PROMPT_LENGTH + GENERATED_TOKENS
    hf_config = transformers.AutoConfig.from_pretrained(CHECKPOINT)
    torch_prompt = torch.randint(0, hf_config.text_config.vocab_size, (1, PROMPT_LENGTH))

    with reshape_device(mesh_device, logical_shape):
        tp_axis = 0 if logical_shape[0] > 1 else 1
        encoder = _encoder(hf_config, mesh_device, tp_axis=tp_axis)
        prompt = tensor.from_torch(torch_prompt, device=mesh_device, dtype=ttnn.uint32)

        # Prefill on its own, twice: the first pass compiles, the second is the reported one.
        padded_length = -(-cache_length // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE
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
        prefill = prefill_times[-1]

        # Two tokens, to compile the decode step and capture its trace. Sizing the cache to
        # cache_length here too makes that the same trace the measured run replays.
        encoder.generate(
            torch_prompt,
            mask=None,
            max_length=PROMPT_LENGTH + 2,
            cache_length=cache_length,
            eos_tokens=None,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            traced=True,
        )

        start = time.perf_counter()
        output = encoder.generate(
            torch_prompt,
            mask=None,
            max_length=cache_length,
            cache_length=cache_length,
            eos_tokens=None,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            traced=True,
        )
        total = time.perf_counter() - start

    generated = output.tokens.shape[1] - PROMPT_LENGTH
    assert generated == GENERATED_TOKENS, f"generated {generated} tokens, expected {GENERATED_TOKENS}"

    # `generate` prefills too, so the decode share is what is left over the prefill measured above.
    decode = total - prefill
    logger.info(
        f"vlm-sweep | {devices:2d} dev | {logical_shape[0]:2d}x{logical_shape[1]:<2d} | tp_axis={tp_axis} "
        f"| prefill {PROMPT_LENGTH} tok {prefill * 1e3:8.1f} ms "
        f"| decode {generated} tok {decode * 1e3:9.1f} ms "
        f"| {decode / generated * 1e3:6.2f} ms/tok | {generated / decode:7.1f} tok/s"
    )


def _encoder(hf_config: transformers.PretrainedConfig, mesh_device: ttnn.MeshDevice, *, tp_axis: int) -> Qwen3VlEncoder:
    """The text model with zeroed weights, tensor-parallel over `tp_axis` of the mesh."""
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=default_topology(mesh_device) or ttnn.Topology.Linear)
    encoder = Qwen3VlEncoder(
        Qwen3VlEncoder.config_from_hf(hf_config),
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    # On the meta device the model is built without any parameter data, so none of the random
    # initialisation runs; the values are irrelevant here and generating them costs minutes.
    with torch.device("meta"):
        torch_model = transformers.AutoModel.from_config(hf_config.text_config, dtype=torch.bfloat16)
    state = {
        f"model.language_model.{k}": torch.zeros(v.shape, dtype=v.dtype) for k, v in torch_model.state_dict().items()
    }
    # The checkpoint ties the lm head to the token embedding.
    state["lm_head.weight"] = state["model.language_model.embed_tokens.weight"]
    encoder.load_torch_state_dict(Qwen3VlEncoder.convert_state(state))
    return encoder
