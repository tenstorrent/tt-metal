# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Context-length probe beyond the app limit, plus DRAM evidence for the per-layer prefill path.

Not part of the PCC suite: the app contract is ``max_length=8192`` and is covered there. This probe
builds one layer of each kind with ``max_seq_len`` raised to the probed length (the K/V page table and
RoPE tables grow with it) and runs a fresh prefill of the full probed length against the HF fp32
layer on the streamed real-prompt input. DRAM numbers come from ``ttnn.get_memory_view``:
after setup (weights, state, constants, input) and after the pass with the output still held.

The S=16384 golden is produced with the existing streamer::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 16384 --layers 0 3

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/probe/test_context_probe.py -q -s
"""

import json
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    bf16_round,
    check_pcc,
    reader,
    to_device,
    to_host,
)
from models.demos.pplx_decider_v1_27b.tt.decoder import PplxDecoderLayer
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations, PrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import PplxRotary

PROBE_LOG = Path("/local/ttuser/gtobar/artifacts/pplx_decider/logs/context_probe.jsonl")


def dram_view(device) -> dict:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    return dict(
        banks=banks,
        total_gib=banks * view.total_bytes_per_bank / 2**30,
        allocated_gib=banks * view.total_bytes_allocated_per_bank / 2**30,
        largest_free_per_bank_mib=view.largest_contiguous_bytes_free_per_bank / 2**20,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [8192, 16384], ids=["S8192", "S16384"])
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["L0_linear", "L3_full"])
def test_context_probe(device, layer_idx, seq_len):
    golden_path = ref.golden_input_path(ref.DEFAULT_GOLDEN_DIR, f"L{layer_idx}_input", seq_len)
    if not golden_path.exists():
        pytest.skip(f"No golden {golden_path}; generate it with the streamer (see module docstring)")
    args = PplxDeciderArgs.from_hf_config(reader().text_config, max_seq_len=seq_len)
    opts = Optimizations.build(device, policy=PrecisionPolicy.bfp8_weights(), max_seq_len=seq_len)
    x_host = bf16_round(torch.load(golden_path)[:, :seq_len])
    with torch.no_grad():
        expected = ref.layer_forward(
            ref.build_decoder_layer(reader(), layer_idx), x_host, ref.rotary_cos_sin(reader().text_config, seq_len)
        )

    dram_empty = dram_view(device)
    layer = PplxDecoderLayer.from_state_dict(
        reader().layer_state_dict(layer_idx), args=args, layer_idx=layer_idx, optimizations=opts
    )
    rotary = PplxRotary(args.rotary_dim, args.rope_theta, seq_len, device) if layer.kind == "full_attention" else None
    for module in (layer.input_norm, layer.post_norm, layer.mlp, layer.mixer):
        module.load_device_weights()
    x = to_device(x_host, device)
    dram_setup = dram_view(device)
    start = time.perf_counter()
    out = layer(x, rotary)
    ttnn.synchronize_device(device)
    first_pass_s = time.perf_counter() - start
    dram_after = dram_view(device)
    pcc = check_pcc(
        expected, to_host(out, expected.shape), module=f"probe_{layer.kind}", layer=layer_idx, seq_len=seq_len
    )
    row = dict(
        layer=layer_idx,
        kind=layer.kind,
        seq_len=seq_len,
        max_seq_len=seq_len,
        chunks=len(layer._chunks(seq_len)),
        pcc=pcc,
        first_pass_s=first_pass_s,
        dram_empty=dram_empty,
        dram_after_setup=dram_setup,
        dram_after_pass_output_held=dram_after,
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROBE_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("PROBE", json.dumps(row))
