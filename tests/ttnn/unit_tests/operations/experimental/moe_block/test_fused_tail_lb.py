# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.fused_tail on a LoudBox (head_tail_lb_cases.py: SP 2 x TP 4, FABRIC_2D, seeded random
weights), the op on its own - accuracy, determinism, device time and the token loops in one call, each its own test.

From random owner rows and shared-expert output:
  * GLM (all-to-all): out = y_own + se in the residual's layout [rows, H / TP];
  * K3 (engine): out = rmsnorm(y_own) * norm @ latent_up^T + se.
Perf baselines: fused_tail_lb_perf_baselines.json."""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    PERF_RUNS,
    DeterminismMarker,
    PerfBaselines,
    median_us,
    traced_op_device_times,
)
from head_tail_lb_cases import (  # noqa: E402
    CALLS,
    MODELS,
    PARAMS,
    TP,
    Weights,
    _from_residual,
    _pcc,
    _q,
    _rmsnorm,
    _skip_unless_lb,
    _to_owner,
    _to_residual,
    mb,
)

BASELINES = PerfBaselines(HERE / "fused_tail_lb_perf_baselines.json")
TAIL_OPS = {"fused_tail": BLOCK_OPS["fused_tail"]}


# ---------------------------------------------------------------------------------------------------- tail
class Tail:
    def __init__(self, mesh, case):
        self.mesh, self.case = mesh, case
        self.model = model = MODELS[case.model]
        cfg = mb.MoEBlockConfig(chunk_tokens=case.chunk, num_links=2, persist_l1_small=True)
        self.res = mb.MoEBlockResources(mesh, model.dims(), cfg, mb.MoEDtypes())
        self.w = Weights(model, case.seed)
        g = torch.Generator().manual_seed(2000 + case.seed)
        chunk = case.chunk
        self.y_host = torch.randn(chunk, model.routed_width, generator=g).to(torch.bfloat16)
        self.se_host = (torch.randn(chunk, model.H, generator=g) * 0.5).to(torch.bfloat16)
        self.y = _to_owner(mesh, self.y_host)
        self.w_up = None
        if model.latent:
            # (K3: the shared expert's output in the residual's layout)
            self.se = _to_residual(mesh, self.se_host)
            f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
            self.w_up = mb.prepare_fused_tail_weights(
                f32(self.w.latent_up), f32(self.w.latent_norm), resources=self.res, dtype=ttnn.bfloat8_b
            )
        else:
            # (GLM: the owners' rows)
            self.se = _to_owner(mesh, self.se_host, layout=ttnn.TILE_LAYOUT)

    def reference(self):
        y, se = self.y_host.float(), self.se_host.float()
        if not self.model.latent:
            return y + se
        # (the op folds the norm's weight into W_up before its bf8 packing: the reference too)
        eps = self.model.dims().norm_eps
        up = _q(self.w.latent_up * self.w.latent_norm[None, :], ttnn.bfloat8_b)
        return _rmsnorm(y, 1.0, eps) @ up.T + se

    def __call__(self):
        return mb.fused_tail(self.y, self.se, resources=self.res, w_up=self.w_up)

    def close(self):
        self.res.release()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_accuracy(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    try:
        out = tail()
        got = _from_residual(mesh_device, out, case.chunk).float()
        ref = tail.reference()
        logger.info(f"FUSED_TAIL accuracy {case.id}:")
        _pcc("out", ref, got, 0.999 if tail.model.latent is None else 0.99)
        if tail.model.latent is None:
            exact = float((got.to(torch.bfloat16) == ref.to(torch.bfloat16)).float().mean())
            logger.info(f"  out = bf16(y_own + se) on {exact:.4f} of the elements")
            assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), "out != y_own + se"
    finally:
        tail.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_determinism(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    det = DeterminismMarker(mesh_device)
    try:
        for _ in range(CALLS):
            out = tail()
            det.add({"out": out})
            ttnn.deallocate(out)
        det.assert_identical(f"fused_tail {case.id}")
    finally:
        tail.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_perf(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    try:
        ttnn.deallocate(tail())  # (warm: program build)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = tail()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t = traced_op_device_times(mesh_device, tid, PERF_RUNS, TAIL_OPS)["fused_tail"]
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(out)
    finally:
        tail.close()
    us = median_us(t)
    logger.info(f"PERF fused_tail {case.id}: median {us:.1f} us over {len(t)} replays")
    BASELINES.check(f"fused_tail {case.id}", us, case.margin)


# ---------------------------------------------------------------------------------------------------- token loops
@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_loops_one_call(mesh_device, device_params, case):
    """MoEBlockConfig.token_loops 2: the tail over both loops' rows in one call (all_loops, its kernels loop) is
    bit-identical to a call a loop on each loop's rows (y_own, the shared expert's output and the output: loop 1's rows
    after loop 0's)."""
    _skip_unless_lb(mesh_device)
    k = 2
    if case.row_tokens % (32 * TP):
        pytest.skip(f"{case.id}: a loop's rows a mesh row ({case.row_tokens}) not a multiple of {32 * TP}")
    model = MODELS[case.model]
    chunk = k * case.chunk  # (a loop: the case's rows)
    cfg = mb.MoEBlockConfig(chunk_tokens=chunk, num_links=2, persist_l1_small=True)
    cfg.token_loops, cfg.chunk_splits = k, 1
    res = mb.MoEBlockResources(mesh_device, model.dims(), cfg, mb.MoEDtypes())
    try:
        w = Weights(model, case.seed)
        g = torch.Generator().manual_seed(2000 + case.seed)
        y = _to_owner(mesh_device, torch.randn(chunk, model.routed_width, generator=g).to(torch.bfloat16))
        se_host = (torch.randn(chunk, model.H, generator=g) * 0.5).to(torch.bfloat16)
        w_up = None
        if model.latent:
            se = _to_residual(mesh_device, se_host)
            f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
            w_up = mb.prepare_fused_tail_weights(
                f32(w.latent_up), f32(w.latent_norm), resources=res, dtype=ttnn.bfloat8_b
            )
        else:
            se = _to_owner(mesh_device, se_host, layout=ttnn.TILE_LAYOUT)

        def rows(t, i):
            n, wd = int(t.shape[-2]) // k, int(t.shape[-1])
            return ttnn.slice(t, [0, 0, i * n, 0], [1, 1, (i + 1) * n, wd], [1, 1, 1, 1])

        ref = ttnn.concat([mb.fused_tail(rows(y, i), rows(se, i), resources=res, w_up=w_up) for i in range(k)], dim=2)
        for _ in range(3):  # (repeated: the loop words re-armed between calls)
            got = mb.fused_tail(y, se, resources=res, w_up=w_up, all_loops=True)
            diff = sum(
                int((ttnn.to_torch(a) != ttnn.to_torch(b)).sum())
                for a, b in zip(ttnn.get_device_tensors(got), ttnn.get_device_tensors(ref))
            )
            logger.info(f"FUSED_TAIL one call {case.id} x {k} loops: {diff} elements differ from a call a loop")
            assert diff == 0
    finally:
        res.release()
