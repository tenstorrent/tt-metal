# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The whole 48-layer Gemma-4 encoder on device with the shipped weights.

The parity tests cover the arithmetic but only one or two layers at a time. This runs the
real 12B stack — all 666 tensors, ~24 GB — tensor-parallel across the mesh, which is where
weight loading at scale and per-layer memory actually get exercised.

Slow by nature: reading the packed checkpoint off disk dominates.
"""

import json
import os
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.tt_dit.encoders.gemma4.model_gemma import Gemma4Config, Gemma4Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import line_params_req_exact_devices

CHECKPOINT = os.environ.get(
    "GEMMA4_CHECKPOINT",
    os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
)

SEQ_LEN = 128


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_full_gemma4_encoder_runs_on_shipped_weights(*, mesh_device):
    if not Path(CHECKPOINT).exists():
        pytest.skip(f"no Gemma-4 checkpoint at {CHECKPOINT}")

    with open(CHECKPOINT, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    config = Gemma4Config.from_hf_text_config(json.loads(header["__metadata__"]["gemma_config"])["text_config"])
    assert config.num_hidden_layers == 48

    logger.info(f"reading {CHECKPOINT}")
    state_dict = {}
    with safe_open(CHECKPOINT, "pt") as handle:
        # The encoder drops the vision/projection/tokenizer sidecars itself.
        keys = [key for key in handle.keys() if key.startswith("model.")]
        for index, key in enumerate(keys, 1):
            state_dict[key] = handle.get_tensor(key)
            # ~24 GB off disk is minutes of silence, which the device broker reaps as a hang.
            if index % 50 == 0 or index == len(keys):
                logger.info(f"read {index}/{len(keys)} tensors")

    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    encoder = Gemma4Encoder(config, mesh_device, ccl_manager, parallel_config, max_seq_len=SEQ_LEN)
    encoder.load_torch_state_dict(state_dict)
    del state_dict

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN))
    tt_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = encoder(tt_ids)

    # embedding + 48 layers + final norm
    assert len(hidden_states) == config.num_hidden_layers + 2

    for index in (0, 1, config.num_hidden_layers, config.num_hidden_layers + 1):
        host = ttnn.to_torch(ttnn.get_device_tensors(hidden_states[index])[0]).float()
        host = host.reshape(1, SEQ_LEN, config.hidden_size)
        logger.info(f"hidden[{index}]: std {host.std():.4f} absmax {host.abs().max():.2f}")
        assert torch.isfinite(host).all(), f"hidden[{index}] is not finite"
        # A dead or disconnected stage collapses to a constant; real Gemma activations do not.
        assert host.std() > 0.01, f"hidden[{index}] is degenerate"
