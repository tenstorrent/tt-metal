# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context mode, one layer of each kind at 262144 tokens against the HF fp32 layer.

Exercises the path ``PplxDeciderModel.forward_long`` takes: the request as 128 pieces of 2048
tokens rewritten in place (``PplxDecoderLayer.forward_chunks``), the K/V cache from the shared pool
(``share_kv_cache``) sized for 262144 tokens, full-attention chunks of 2048 over the whole prefix
(chunked SDPA), and the DeltaNet state carried over 128 chunks. The input is the streamed
real-prompt layer input; the reference is the HF layer on CPU (fp32, several minutes at this length).

Golden (CPU, once)::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 262144 --layers 0 3

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/probe/test_long_context_layer.py -q -s
"""

import json
import time

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tests.test_utils import DEVICE_PARAMS, bf16_round, check_pcc, reader, to_device
from models.demos.pplx_decider_v1_27b.tt import paths
from models.demos.pplx_decider_v1_27b.tt.decoder import PplxDecoderLayer
from models.demos.pplx_decider_v1_27b.tt.model import LONG_CHUNK, dram_view
from models.demos.pplx_decider_v1_27b.tt.model_config import APP_MAX_LENGTH, LONG_MAX_LENGTH, PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations, PrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import PplxRotary

PROBE_LOG = paths.PROBE_LOG
PCC = 0.99


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [LONG_MAX_LENGTH], ids=["S262144"])
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["L0_linear", "L3_full"])
def test_long_context_layer(device, layer_idx, seq_len):
    golden_path = ref.golden_input_path(ref.DEFAULT_GOLDEN_DIR, f"L{layer_idx}_input", seq_len)
    if not golden_path.exists():
        pytest.skip(f"No golden {golden_path}; generate it with the streamer (see module docstring)")
    args = PplxDeciderArgs.from_hf_config(reader().text_config, max_seq_len=seq_len)
    opts = Optimizations.build(
        device,
        policy=PrecisionPolicy.default(),
        max_seq_len=seq_len,
        prefill_chunk=LONG_CHUNK,
        prefill_chunk_full_attention=APP_MAX_LENGTH,
        share_kv_cache=True,
    )
    x_host = bf16_round(torch.load(golden_path)[:, :seq_len])
    start = time.perf_counter()
    with torch.no_grad():
        expected = ref.layer_forward(
            ref.build_decoder_layer(reader(), layer_idx), x_host, ref.rotary_cos_sin(reader().text_config, seq_len)
        )
    hf_s = time.perf_counter() - start

    layer = PplxDecoderLayer.from_state_dict(
        reader().layer_state_dict(layer_idx), args=args, layer_idx=layer_idx, optimizations=opts
    )
    rotary = PplxRotary(args.rotary_dim, args.rope_theta, seq_len, device, head_dim=args.head_dim)
    for module in (layer.input_norm, layer.post_norm, layer.mlp, layer.mixer):
        module.load_device_weights()
    pieces = [to_device(x_host[:, s : s + LONG_CHUNK], device) for s in range(0, seq_len, LONG_CHUNK)]
    dram_setup = dram_view(device)
    start = time.perf_counter()
    layer.forward_chunks(pieces, rotary)
    ttnn.synchronize_device(device)
    tt_s = time.perf_counter() - start
    dram_after = dram_view(device)
    out = torch.cat([ttnn.to_torch(p).float().reshape(1, -1, args.hidden_size) for p in pieces], dim=1)
    last_pcc = float(torch.corrcoef(torch.stack([expected[0, -1], out[0, -1]]))[0, 1])
    pcc = check_pcc(
        expected, out, module=f"long_{layer.kind}", layer=layer_idx, seq_len=seq_len, threshold=PCC, policy="default"
    )
    row = dict(
        layer=layer_idx,
        kind=layer.kind,
        seq_len=seq_len,
        pieces=len(pieces),
        pcc=pcc,
        last_token_pcc=last_pcc,
        tt_pass_s=tt_s,
        hf_cpu_s=hf_s,
        dram_after_setup=dram_setup,
        dram_after_pass=dram_after,
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROBE_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("LONG_PROBE", json.dumps(row))
