# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MLP-level decode benchmark: DRAM-core (receiver-contiguous) prefetcher vs worker-core.

Builds the Llama-3.2-3B decode MLP (FF1/FF3/FF2 fed by the prefetcher) and trace-replays
a single decode forward ``BENCH_REPEATS`` times, reporting per-forward latency. The
prefetcher backend is selected by ``make_prefetcher`` via ``TT_METAL_USE_DRAM_CORE_PREFETCHER``
(1 = DRAM-core DRISC senders with recv-contig weights, 0 = worker-core BRISC/NCRISC senders),
so the two backends share an identical MLP / matmul / trace path and differ only in how
weights are pushed into the receiver ring. Run the test twice (env=1, then env=0) and compare.

Requires HF weights for ``meta-llama/Llama-3.2-3B`` and Blackhole. Decode-only.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import make_prefetcher

pytestmark = [
    run_for_blackhole("DRAM prefetcher benchmark requires Blackhole"),
    pytest.mark.skipif(
        not any(m in os.environ.get("HF_MODEL", "") for m in ("Llama-3.2-3B", "Llama-3.1-8B")),
        reason="HF_MODEL must point to a supported prefetcher model (Llama-3.2-3B or Llama-3.1-8B)",
    ),
]


@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 90000000}],
    indirect=True,
)
def test_mlp_bench(mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = Mode.DECODE
    seq_len = 32

    repeats = int(os.environ.get("BENCH_REPEATS", "50"))
    num_tensors = 3  # FF1/FF3/FF2
    # One MLP module's worth of weights per prefetch. The worker-core Prefetcher expects exactly
    # num_tensors*num_layers distinct inserted weights (one set per real decoder layer), so we use
    # num_layers=1 and re-prefetch per forward — the realistic decode pattern (prefetch once per
    # forward()) that both backends' run()/stop() lifecycles support.
    num_layers = 1

    # Pin both backends to the same ring for an apples-to-apples comparison. 3B's FF2 (N=dim=3072)
    # only divides cleanly at ring=32 (recv_per_bank=4); the worker-core auto-pick would otherwise
    # choose an invalid ring=64.
    recv_per_bank = int(os.environ.get("BENCH_RECV_PER_BANK", "4"))
    prefetcher = make_prefetcher(
        mesh_device, num_tensors=num_tensors, num_layers=num_layers, num_receiver_cores=recv_per_bank
    )
    backend = prefetcher.__class__.__name__
    prefetcher.init(mode)

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=1,
        max_seq_len=128,
        cache_hf=True,
        prefetcher=prefetcher,
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
        prefetcher=prefetcher,
    )

    prefetcher.prefetch()

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    def make_input():
        # MLP.forward deallocates its input, so each forward needs its own input tensor.
        return ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
            dtype=ttnn.bfloat8_b,
            memory_config=model_args.get_mlp_input_mem_config(mode, prefetcher),
            layout=ttnn.TILE_LAYOUT,
        )

    # Each decode forward is preceded by one prefetch (run) and followed by stop — the production
    # decode pattern. The full MLP forward contains a device read (CCL/reshard) so we can't trace it;
    # we time per-forward run()+forward()+stop() with a synchronize each iteration. Both backends run
    # identical MLP code, so the latency delta reflects the prefetcher-fed weight path. Inputs are
    # pre-built outside the timed region (MLP.forward deallocates its input).
    prefetcher.run()
    out = tt_model(make_input(), mode)
    prefetcher.stop()
    ttnn.synchronize_device(mesh_device)

    inputs = [make_input() for _ in range(repeats)]
    ttnn.synchronize_device(mesh_device)

    aiclk_pre = mesh_device.get_clock_rate_mhz() if hasattr(mesh_device, "get_clock_rate_mhz") else None
    per_forward_times = []
    bench_out = None
    for i in range(repeats):
        t0 = time.perf_counter()
        prefetcher.run()
        bench_out = tt_model(inputs[i], mode)
        prefetcher.stop()
        ttnn.synchronize_device(mesh_device)
        per_forward_times.append(time.perf_counter() - t0)
    assert bench_out is not None
    elapsed = sum(per_forward_times)

    if hasattr(prefetcher, "teardown"):
        prefetcher.teardown()

    per_forward_us = elapsed / repeats * 1e6
    sorted_us = sorted(t * 1e6 for t in per_forward_times)
    median_us = sorted_us[len(sorted_us) // 2]
    min_us = sorted_us[0]
    # 2 * M * K * N flops per matmul; FF1 + FF3 (dim->hidden) + FF2 (hidden->dim) per forward.
    dim = model_args.dim
    hidden = model_args.hidden_dim // model_args.num_devices
    flops = 2 * seq_len * (2 * dim * hidden + hidden * dim)
    tflops = flops / (median_us / 1e6) / 1e12
    logger.info(
        f"[mlp_bench] backend={backend} ring={prefetcher.ring_size} repeats={repeats} "
        f"per_forward mean={per_forward_us:.2f}us median={median_us:.2f}us min={min_us:.2f}us "
        f"aiclk={aiclk_pre}MHz -> {tflops:.3f} TFLOP/s (MLP, @median)"
    )
