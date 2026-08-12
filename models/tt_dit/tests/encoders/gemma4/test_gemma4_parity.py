# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Numerical parity for the Gemma-4 encoder against HuggingFace.

The reference activations come from ``transformers>=5``'s ``Gemma4TextModel``, which
is the only Gemma-4 implementation available; the tt-metal env pins 4.53, so they are
generated out-of-process by ``gen_gemma4_reference.py`` and read back here. The stack
is narrowed (1024 hidden, 6 layers) but keeps the real head geometry — head_dim 256
sliding against 512 global, one global KV head, V tied to K — so the arithmetic under
test is the arithmetic the 12B checkpoint uses.

Regenerate with the instructions at the top of ``gen_gemma4_reference.py``; without the
file the test skips rather than fails, since the venv it needs is not part of CI.
"""

import os
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

REFERENCE = os.environ.get("GEMMA4_REFERENCE", "/tmp/g4ref/gemma4_reference.safetensors")

VOCAB = 1000
HIDDEN = 1024
NUM_LAYERS = 6


def _load_reference():
    if not Path(REFERENCE).exists():
        pytest.skip(f"no Gemma-4 reference at {REFERENCE}; generate it with gen_gemma4_reference.py")
    tensors = {}
    with safe_open(REFERENCE, "pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma4_encoder_matches_huggingface(*, mesh_device):
    reference = _load_reference()

    config = Gemma4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2048,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        max_position_embeddings=128,
    )

    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    input_ids = reference["input_ids"]
    seq_len = input_ids.shape[-1]

    encoder = Gemma4Encoder(config, mesh_device, ccl_manager, parallel_config, max_seq_len=seq_len)
    # The reference is a bare text model, so its keys carry no prefix; the packed LTX
    # checkpoint nests the same tree under model.*, which is what the encoder strips.
    encoder.load_torch_state_dict(
        {"model." + k[len("weight.") :]: v for k, v in reference.items() if k.startswith("weight.")}
    )

    tt_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = encoder(tt_ids)
    assert len(hidden_states) == NUM_LAYERS + 2

    def host(index):
        return ttnn.to_torch(ttnn.get_device_tensors(hidden_states[index])[0]).float().reshape(1, seq_len, HIDDEN)

    # reference hidden_states[i] is the input to layer i, so it lines up with ours
    # one-for-one until the tail, where the reference reports only the post-norm output.
    pairs = [(f"layer {idx} in", idx, idx) for idx in range(NUM_LAYERS)]
    pairs.append(("post-norm out", NUM_LAYERS + 1, NUM_LAYERS))

    # bf16 error compounds with depth — by the last layer even torch's own bf16 run only
    # holds ~93 % against fp32 — so the bar is that stage against the same bf16 floor, not
    # a fixed number. A structural bug falls far below the floor; dtype noise tracks it.
    failures = []
    for label, ours, theirs in pairs:
        got = host(ours).flatten().double()
        want = reference[f"hidden.{theirs}"].flatten().double()
        floor_want = reference[f"bf16.{theirs}"].flatten().double()

        def pcc(a, b):
            return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

        ours_pcc, floor_pcc = pcc(got, want), pcc(floor_want, want)
        logger.info(f"{label:>14}: PCC {ours_pcc * 100:.4f} % (torch bf16 floor {floor_pcc * 100:.4f} %)")
        if ours_pcc < floor_pcc - 0.005:
            failures.append(f"{label}: {ours_pcc * 100:.4f} % vs floor {floor_pcc * 100:.4f} %")

    assert not failures, "below the bf16 floor at: " + "; ".join(failures)
