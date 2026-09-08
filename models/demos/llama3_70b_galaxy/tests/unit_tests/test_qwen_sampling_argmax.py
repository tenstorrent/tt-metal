# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Isolated repro for the BH prefetcher-path force-argmax stale-token race.

Under decode trace replay, the sampling all_gather + untilize/argmax can return the previous
step's tokens if the gather is not pinned to the worker sub-device or if the argmax kernel
does not reset its semaphores. Eager mode masks this because per-op host dispatch latency
exceeds the gather time. This test always compiles eagerly, captures a trace, and replays
it with fresh column-sharded logits whose per-row maxima are known.
"""

import pytest
import torch
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.common.sampling.tt_sampling import TTSampling
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
            "trace_region_size": 23887872,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
def test_qwen_sampling_argmax(mesh_device, reset_seeds):
    model_args = TtQwenModelArgs(mesh_device, dummy_weights=True, max_batch_size=32, max_seq_len=256)
    model_args.n_layers = 1
    use_prefetcher = model_args.use_prefetcher
    logger.info(f"use_prefetcher={use_prefetcher} use_unfused_ccl={getattr(model_args, 'use_unfused_ccl', None)}")

    if not use_prefetcher:
        prefetcher_setup = None
        worker_sub_device_id = None
    else:
        prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, is_qwen=True)
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id

    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, is_qwen=True)

    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        k=torch.ones(32),
        p=torch.zeros(32),
        temp=torch.ones(32),
    )
    logger.info(f"force_argmax={tt_sampling.force_argmax_sampling}")

    batch = 32
    padded_vocab = model_args.padded_vocab_size  # 155648
    chunk = padded_vocab // 8  # 19456 per device column

    # Known max positions covering all 8 vocab chunks. Shift positions per iteration so a stale
    # gather buffer (previous iteration's data) also fails the check.
    positions = [198, 257, 11, 279, chunk + 5, 2 * chunk + 100, 3 * chunk + 7, 5 * chunk + 3000]

    n_iters = 5
    all_pass = True

    def make_input(it):
        x = torch.rand(1, 1, batch, padded_vocab) * 4.0 - 2.0
        expected = []
        for r in range(batch):
            pos = (positions[r % len(positions)] + it * 31) % padded_vocab
            x[0, 0, r, pos] = 30.0 + r
            expected.append(pos)
        return x, expected

    def to_dev(x):
        # Mirror the model: lm_head output is column-sharded (vocab) and row-replicated (after the
        # cross-row all-reduce), DRAM interleaved, bfloat8_b.
        return ttnn.from_torch(
            x,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=model_args.cluster_shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def score(it, tok, expected, tag=""):
        nonlocal all_pass
        got = ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).tolist()
        errs = [(r, expected[r], got[r]) for r in range(batch) if got[r] != expected[r]]
        logger.info(f"iter {it}{tag}: mismatches {len(errs)}")
        for r, exp, g in errs[:8]:
            logger.warning(
                f"iter {it}{tag} row {r}: expected {exp} got {g} (delta {g - exp} = {(g - exp) / chunk} chunks)"
            )
        if errs:
            all_pass = False

    # Mirror the model's trace lifecycle: eager compile call on a persistent input buffer, then
    # capture sampling into a trace and replay it with fresh data copied into the same buffer.
    x0, expected0 = make_input(0)
    x_dev = to_dev(x0)
    tok, _ = tt_sampling(x_dev)
    score(0, tok, expected0, tag=" [compile]")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tok, _ = tt_sampling(x_dev)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    for it in range(1, n_iters):
        x, expected = make_input(it)
        x_host = ttnn.from_torch(
            x,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=model_args.cluster_shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(x_host, x_dev)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        score(it, tok, expected, tag=" [replay]")

    ttnn.release_trace(mesh_device, trace_id)

    tt_ccl.close()
    assert all_pass, "on-device sampling argmax returned displaced indices"
