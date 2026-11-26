# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the logprobs trace of an argmax_ids verify (tt/spec_logprobs.py VerifyLogprobs) on controlled logits.

The trace runs as the model runs it: a trace of its own over a persistent buffer holding each device's vocabulary shard
[1, 1, rows, V / TP] bfloat16 (V = 248320), warmed eagerly, captured with program cache misses disallowed and replayed,
at the full verify width (B = 8 users x T = 4 = 32 rows) and at the row counts of the verify row buckets (4, 8, 16
rows). For each logits kind, the payload of the argmax and of tokens deeper in each row's top-32 (the tokens a device
draw can take) is compared with vLLM's raw logprobs of the same bfloat16 logits (Sampler.compute_logprobs in float32
and log_softmax in float64): the logprob within LOGPROB_TOL, the rank equal (or a lower bound when the token ties a
shard's last candidate), the top-20 ids those of a stable sort of the row and their logprobs within LOGPROB_TOL. The
largest difference seen is logged.

    pytest models/demos/blackhole/qwen36/tests/test_spec_logprobs.py    (Blackhole, up to 2 chips; MESH_DEVICE=P150x8)
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_logprobs import TOP_LOGPROBS, VerifyLogprobs, payload_from_reads
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_TOP_K

VOCAB = 248320
MESH = _resolve_mesh_shape(2)
# The largest |device - float64 log_softmax| a payload logprob may have: the device sums 31040 float32 terms per shard
# relative to the row maximum and adds the shards' sums in float32.
LOGPROB_TOL = 1e-4
KINDS = ("random", "peaked", "coarse", "flat")


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _logits(kind, rows, seed):
    """bfloat16-valued logits [rows, V] of a ``kind``: random (N(0, 3)), peaked (random plus a few large logits, as a
    confident next token), coarse (rounded to 0.5: thousands of exact ties) and flat (N(0, 0.3): a nucleus far wider
    than the candidates)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, VOCAB, generator=g) * (0.3 if kind == "flat" else 3.0)
    if kind == "peaked":
        for r in range(rows):
            idx = torch.randint(0, VOCAB, (6,), generator=g)
            x[r, idx] += torch.tensor([18.0, 15.0, 14.0, 12.0, 11.0, 10.0])
    if kind == "coarse":
        x = torch.round(x / 0.5) * 0.5
    return x.to(torch.bfloat16).float()


def _capture(mesh, fn):
    mesh.set_program_cache_misses_allowed(False)
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    ttnn.synchronize_device(mesh)
    return trace


def _reads(lp):
    return [ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in lp.outputs]


@_parametrize
@pytest.mark.parametrize("B, T, whole", [(8, 4, False), (1, 4, False), (1, 4, True)])
def test_the_payload_equals_vllms_logprobs_of_the_same_logits(mesh_device, B, T, whole):
    """``whole``: every device holds the whole vocabulary, as when the verify gathers its logits itself."""
    R = B * T
    lp = VerifyLogprobs(mesh_device, B, T, VOCAB, shards=_shards(mesh_device))
    bufs = {
        rows: ttnn.from_torch(
            torch.zeros(1, 1, rows, VOCAB),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if whole else _mapper(mesh_device),
        )  # fmt: skip
        for rows in sorted({R, min(R, 16), min(R, 8), 4}, reverse=True)
    }
    for buf in bufs.values():
        lp.body(buf)
    ttnn.synchronize_device(mesh_device)
    traces = {rows: _capture(mesh_device, lambda b=buf: lp.body(b)) for rows, buf in bufs.items()}
    from vllm.v1.sample.sampler import Sampler

    worst = worst_top = 0.0
    lower_bounds = 0
    try:
        for i, kind in enumerate(KINDS):
            for rows, buf in bufs.items():
                x = _logits(kind, rows, seed=10 * i + rows)
                h = ttnn.from_torch(
                    x.reshape(1, 1, rows, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if whole else _mapper(mesh_device),
                )  # fmt: skip
                ttnn.copy_host_to_device_tensor(h, buf)
                ttnn.execute_trace(mesh_device, traces[rows], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                order = torch.sort(x, dim=1, descending=True, stable=True).indices
                g = torch.Generator().manual_seed(rows + i)
                depth = torch.randint(0, SAMPLER_TOP_K, (rows, 1), generator=g)
                depth[::3] = 0  # argmax rows
                chosen = torch.zeros(R, dtype=torch.int64)
                chosen[:rows] = order.gather(1, depth).reshape(-1)
                live = torch.zeros(B, dtype=torch.bool)
                live[: rows // T] = True
                out = payload_from_reads(_reads(lp), chosen.reshape(B, T), live, B, T)
                got_lp, rank, top_lp, top_ids = (t.reshape(R, *t.shape[2:])[:rows] for t in out)
                want = torch.log_softmax(x.double(), dim=-1)
                ref = Sampler.gather_logprobs(Sampler.compute_logprobs(x), TOP_LOGPROBS, chosen[:rows])
                err = (got_lp.double() - want.gather(1, chosen[:rows, None]).reshape(-1)).abs().max()
                err_ref = (got_lp - ref.logprobs[:, 0]).abs().max()
                top_err = (top_lp.double() - want.gather(1, order[:, :TOP_LOGPROBS])).abs().max()
                worst, worst_top = max(worst, float(err), float(err_ref)), max(worst_top, float(top_err))
                where = f"{kind}, {rows} rows"
                assert err <= LOGPROB_TOL and err_ref <= LOGPROB_TOL, f"{where}: logprob off by {float(err):.2e}"
                assert top_err <= LOGPROB_TOL, f"{where}: top logprobs off by {float(top_err):.2e}"
                assert torch.equal(top_ids.to(torch.int64), order[:, :TOP_LOGPROBS]), f"{where}: top ids differ"
                want_rank = (x >= x.gather(1, chosen[:rows, None])).sum(dim=1).to(torch.int32)
                assert torch.equal(want_rank, ref.selected_token_ranks.to(torch.int32))
                tied = rank != want_rank
                lower_bounds += int(tied.sum())
                assert bool((rank[tied] < want_rank[tied]).all()), f"{where}: rank above vLLM's"
        logger.info(
            f"logprobs payload vs float64 log_softmax: max |diff| {worst:.2e} (tokens), {worst_top:.2e} (top-20); "
            f"{lower_bounds} lower-bound ranks"
        )
    finally:
        for t in traces.values():
            ttnn.release_trace(mesh_device, t)
        for buf in bufs.values():
            ttnn.deallocate(buf)


@_parametrize
def test_logprobs_trace_device_time(mesh_device):
    """The full-width logprobs trace's replay time (wall clock over 50 replays after a warm one), logged."""
    B, T = 8, 4
    lp = VerifyLogprobs(mesh_device, B, T, VOCAB, shards=_shards(mesh_device))
    buf = ttnn.from_torch(
        _logits("random", B * T, 1).reshape(1, 1, B * T, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, mesh_mapper=_mapper(mesh_device),
    )  # fmt: skip
    lp.body(buf)
    ttnn.synchronize_device(mesh_device)
    trace = _capture(mesh_device, lambda: lp.body(buf))
    try:
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / n * 1e3
        logger.info(f"logprobs trace: {ms:.3f} ms per replay at {B * T} rows, TP={_shards(mesh_device)}")
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(buf)
