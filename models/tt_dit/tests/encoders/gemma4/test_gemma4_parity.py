# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Numerical parity for the Gemma-4 encoder against HuggingFace.

The reference activations come from ``transformers>=5``'s ``Gemma4TextModel``, which is the
only Gemma-4 implementation available; the tt-metal env pins 4.53, so they are generated
out-of-process by ``gen_gemma4_reference.py`` and read back here.

Two shapes of test. The narrow one runs a randomly-initialised 1024-wide stack, keeping the
real head geometry — head_dim 256 sliding against 512 global, one global KV head, V tied to
K — to isolate the arithmetic. The real one runs two trained layers at full width, one of
each kind, and reads its weights from the shipped checkpoint so the loading path is under
test too.

Regenerate with the instructions at the top of ``gen_gemma4_reference.py``; without the
files the tests skip rather than fail, since the venv they need is not part of CI.
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

REFERENCE = os.environ.get("GEMMA4_REFERENCE", "/tmp/g4ref/gemma4_reference.safetensors")
REFERENCE_REAL = os.environ.get("GEMMA4_REFERENCE_REAL", "/tmp/g4ref/gemma4_reference_real.safetensors")
CHECKPOINT = os.environ.get(
    "GEMMA4_CHECKPOINT",
    os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
)

# Must match gen_gemma4_reference.py, which chooses the first global layer of the 48.
REAL_LAYERS = ((0, 0), (1, 5))


def _load(path):
    if not Path(path).exists():
        pytest.skip(f"no Gemma-4 reference at {path}; generate it with gen_gemma4_reference.py")
    with safe_open(path, "pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _checkpoint_text_config():
    if not Path(CHECKPOINT).exists():
        pytest.skip(f"no Gemma-4 checkpoint at {CHECKPOINT}")
    with open(CHECKPOINT, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return json.loads(header["__metadata__"]["gemma_config"])["text_config"]


def _run_encoder(config, mesh_device, state_dict, input_ids):
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    seq_len = input_ids.shape[-1]

    encoder = Gemma4Encoder(config, mesh_device, ccl_manager, parallel_config, max_seq_len=seq_len)
    encoder.load_torch_state_dict(state_dict)

    tt_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return encoder(tt_ids)


def _assert_tracks_bf16_floor(hidden_states, reference, *, num_layers, hidden_size, seq_len):
    assert len(hidden_states) == num_layers + 2

    def host(index):
        return ttnn.to_torch(ttnn.get_device_tensors(hidden_states[index])[0]).float().reshape(1, seq_len, hidden_size)

    def pcc(a, b):
        return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()

    # reference hidden_states[i] is the input to layer i, so it lines up with ours
    # one-for-one until the tail, where the reference reports only the post-norm output.
    pairs = [(f"layer {idx} in", idx, idx) for idx in range(num_layers)]
    pairs.append(("post-norm out", num_layers + 1, num_layers))

    # bf16 error compounds with depth — deep enough in, even torch's own bf16 run holds only
    # ~93 % against fp32 — so the bar is that stage against the same bf16 floor, not a fixed
    # number. A structural bug falls far below the floor; dtype noise tracks it.
    failures = []
    for label, ours, theirs in pairs:
        got, want = host(ours), reference[f"hidden.{theirs}"]
        ours_pcc, floor_pcc = pcc(got, want), pcc(reference[f"bf16.{theirs}"], want)
        logger.info(f"{label:>14}: PCC {ours_pcc * 100:.4f} % (torch bf16 floor {floor_pcc * 100:.4f} %)")
        if ours_pcc < floor_pcc - 0.005:
            failures.append(f"{label}: {ours_pcc * 100:.4f} % vs floor {floor_pcc * 100:.4f} %")

    assert not failures, "below the bf16 floor at: " + "; ".join(failures)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma4_encoder_matches_huggingface(*, mesh_device):
    reference = _load(REFERENCE)
    num_layers, hidden_size = 6, 1024

    config = Gemma4Config(
        vocab_size=1000,
        hidden_size=hidden_size,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        max_position_embeddings=128,
    )

    # The reference is a bare text model, so its keys carry no prefix; the packed LTX
    # checkpoint nests the same tree under model.*, which is what the encoder strips.
    state_dict = {"model." + k[len("weight.") :]: v for k, v in reference.items() if k.startswith("weight.")}

    input_ids = reference["input_ids"]
    hidden_states = _run_encoder(config, mesh_device, state_dict, input_ids)
    _assert_tracks_bf16_floor(
        hidden_states, reference, num_layers=num_layers, hidden_size=hidden_size, seq_len=input_ids.shape[-1]
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma4_encoder_matches_huggingface_on_shipped_weights(*, mesh_device):
    reference = _load(REFERENCE_REAL)

    text_config = dict(_checkpoint_text_config())
    text_config.update(num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"])
    config = Gemma4Config.from_hf_text_config(text_config)

    with safe_open(CHECKPOINT, "pt") as handle:
        keys = set(handle.keys())
        wanted = {k: k for k in ("model.embed_tokens.weight", "model.norm.weight")}
        for dst, src in REAL_LAYERS:
            prefix = f"model.layers.{src}."
            wanted.update({f"model.layers.{dst}." + k[len(prefix) :]: k for k in keys if k.startswith(prefix)})
        state_dict = {dst: handle.get_tensor(src) for dst, src in wanted.items()}

    input_ids = reference["input_ids"]
    hidden_states = _run_encoder(config, mesh_device, state_dict, input_ids)
    _assert_tracks_bf16_floor(
        hidden_states,
        reference,
        num_layers=2,
        hidden_size=config.hidden_size,
        seq_len=input_ids.shape[-1],
    )
