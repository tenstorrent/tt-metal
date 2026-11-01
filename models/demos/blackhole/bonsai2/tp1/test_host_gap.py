# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-gap breakdown of one traced decode step: what lies between the device kernel sum and the e2e ms/token.

Each variant runs TIMED steps back to back, 5 times, synced at the end of each run; median ms/step:
    replay       execute_trace only                         -> device kernels + in-trace op-to-op gaps
    replay_sync  execute_trace + synchronize every step     -> + per-step launch/sync round trip
    write        host input prep + H2D write + replay       -> + input write (pipelined)
    full         gen.decode_forward(read=True) + argmax     -> what the e2e gate times
Also times host input prep alone and the logits readback of a finished step alone.

    HF_MODEL=<dense HF dir> pytest models/demos/blackhole/bonsai2/tp1/test_host_gap.py -s
"""
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tp1.test_decode_e2e import BLOCK_SIZE, NUM_BLOCKS
from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import Generator

WARMUP = 16
TIMED = 64


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_host_gap(mesh_device):
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE, enable_mtp=False
    )
    args = model.args
    ref = torch.load(f"models/tt_transformers/tests/reference_outputs/{args.model_name}.refpt")
    tokens = ref["reference_tokens"][0]
    model.allocate_kv_caches([NUM_BLOCKS, args.n_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1)
    model.reset_state()
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], mesh_device)
    prime_decode_trace(gen, model, tokens[:1].view(1, 1), torch.tensor([0]), page_table)
    trace_id = gen.trace_ids_decode[False][0]
    dev_inputs = gen.trace_inputs_decode[False][0]
    out = gen.trace_output_decode[False]

    def host_inputs(pos):
        return model.prepare_decode_inputs_host(tokens[pos].view(1, 1), torch.tensor([pos]), page_table)

    def replay(pos):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

    def replay_sync(pos):
        replay(pos)
        ttnn.synchronize_device(mesh_device)

    def write(pos):
        copy_host_to_device(host_tensors=host_inputs(pos), device_tensors=dev_inputs)
        replay(pos)

    def full(pos):
        o = gen.decode_forward(
            tokens[pos].view(1, 1),
            torch.tensor([pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        int((o[0] if isinstance(o, tuple) else o).squeeze().float().argmax())

    def bench(fn):
        ts = []
        for _ in range(5):
            for i in range(WARMUP):
                fn(i)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for i in range(TIMED):
                fn(WARMUP + i)
            ttnn.synchronize_device(mesh_device)
            ts.append((time.perf_counter() - t0) * 1e3 / TIMED)
        return statistics.median(ts), ts

    res = {name: bench(fn) for name, fn in [("replay", replay), ("replay_sync", replay_sync), ("write", write), ("full", full)]}

    t0 = time.perf_counter()
    for i in range(TIMED):
        host_inputs(i)
    prep_ms = (time.perf_counter() - t0) * 1e3 / TIMED

    replay(0)
    ttnn.synchronize_device(mesh_device)
    rb = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        gen.process_decode_output_host(gen.read_decode_output(out), is_tokens=False)
        rb.append((time.perf_counter() - t0) * 1e3)
    readback_ms = statistics.median(rb)

    print("\n=== host-gap breakdown, ms/step (median of 5 x", TIMED, "steps) ===")
    for k, (m, ts) in res.items():
        print(f"  {k:<12}{m:8.3f}   runs {[round(t, 3) for t in ts]}")
    print(f"  host input prep alone   {prep_ms:8.3f}")
    print(f"  logits readback alone   {readback_ms:8.3f}")
