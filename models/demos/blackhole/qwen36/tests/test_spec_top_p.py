# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the certificate sampler of a verify with top-k off and top_p < 1 rows (tt/spec_top_p.py TopPCertificate)
on controlled logits.

The trace runs as the model runs it: a trace of its own over a persistent buffer holding each device's vocabulary shard
[1, 1, rows, V / TP] bfloat16 (V = 248320), warmed eagerly, captured with program cache misses disallowed, and replayed
with a fresh sampling plan per verify (the plan of spec_top_p.sampler_params). B = 8 users x T = 4 columns:

* users 0 to 3: peaked rows whose nucleus lies inside the top-32 at their (temperature, top_p): q >= top_p + CERT_EPS
  on every device, and over N replays each column's draws follow vLLM's full-vocabulary nucleus
  (sampling_reference.target_probs with top-k off): Pearson chi-square p-value above P_MIN per column;
* user 4: rows with a plateau of 200 tokens, whose nucleus reaches far past the top-32: q < top_p, so
  spec_top_p.failed_rows marks every column, and the redo from the logits read back from the devices (the reads the engine makes) follows vLLM's
  nucleus (chi-square over seeds);
* user 5: a top-k 20 top_p 0.9 row next to them draws as on the sampler trace (target_probs under either top-p rule);
* user 6 greedy, user 7 padding: they draw nothing.

    pytest models/demos/blackhole/qwen36/tests/test_spec_top_p.py    (Blackhole, up to 2 chips; MESH_DEVICE=P150x8)
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.sampling_reference import chi_square_p, target_probs
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_sample import VerifySampler, sampling_plan
from models.demos.blackhole.qwen36.tt.spec_top_p import (
    CERT_EPS,
    TopPCertificate,
    certificate_rows,
    failed_rows,
    redo_draws,
    sampler_params,
)

VOCAB = 248320
MESH = _resolve_mesh_shape(2)
P_MIN = 1e-4
B, T = 8, 4
# (temperature, top_k, top_p) per user; top_k VOCAB is off, as the plugin sends it.
USERS = [
    (1.0, VOCAB, 0.9),
    (0.7, VOCAB, 0.95),
    (1.2, VOCAB, 0.8),
    (0.6, VOCAB, 0.5),
    (1.0, VOCAB, 0.9),
    (1.0, 20, 0.9),
    (0.0, 1, 1.0),
    (0.0, 1, 1.0),
]


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _row_logits(r, g):
    """Users 0 to 3 and 5: a peaked row (a few large logits over N(0, 1)); user 4: a plateau of 200 tokens 8 above
    the rest, whose nucleus holds far more than 32 tokens; the greedy and padding users: N(0, 1)."""
    u = r // T
    x = torch.randn(VOCAB, generator=g)
    if u == 4:
        x[torch.randperm(VOCAB, generator=g)[:200]] += 8.0
        return x.to(torch.bfloat16).float()
    if u in (0, 1, 2, 3, 5):
        idx = torch.randperm(VOCAB, generator=g)[:12]
        x[idx] += torch.linspace(9.0, 4.0, 12)
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


def _params():
    return SimpleNamespace(
        temperature=[u[0] for u in USERS],
        top_k=[u[1] for u in USERS],
        top_p=[u[2] for u in USERS],
        seed=[None] * B,
    )


@_parametrize
def test_certificate_rows_draw_vllms_nucleus_and_failed_rows_redo_from_the_device_logits(mesh_device):
    g = torch.Generator().manual_seed(2026)
    logits = torch.stack([_row_logits(r, g) for r in range(B * T)])
    sampler = VerifySampler(mesh_device, B, T, VOCAB, shards=_shards(mesh_device))
    cert_dev = TopPCertificate(sampler)
    buf = ttnn.from_torch(
        logits.reshape(1, 1, B * T, VOCAB), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=_mapper(mesh_device),
    )  # fmt: skip
    params = _params()
    cert = certificate_rows(params, B, VOCAB)
    assert cert.rows == [True] * 5 + [False] * 3
    positions = [100 + 9 * u for u in range(B - 1)] + [-1]
    plan = sampling_plan(sampler_params(params, cert), positions, T)
    sampler.stage(plan)
    cert_dev.stage(cert, plan, T)
    cert_dev.body(buf)
    ttnn.synchronize_device(mesh_device)
    trace = _capture(mesh_device, lambda: cert_dev.body(buf))
    N = 2000
    try:
        draws = []
        for i in range(N):
            plan = sampling_plan(sampler_params(params, cert), [p + i if p >= 0 else p for p in positions], T)
            sampler.stage(plan)
            cert_dev.stage(cert, plan, T)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            per_device = [
                ttnn.to_torch(d).reshape(-1)[: B * T].to(torch.int64) for d in ttnn.get_device_tensors(sampler.draws)
            ]
            assert all(torch.equal(d, per_device[0]) for d in per_device), "devices drew different tokens"
            draws.append(per_device[0])
        draws = torch.stack(draws)
        qs = [ttnn.to_torch(d).reshape(-1)[: B * T].float() for d in ttnn.get_device_tensors(cert_dev.q)]
        assert all(torch.allclose(q, qs[0]) for q in qs), "devices computed different q"
        q = qs[0]
        fail = failed_rows(q, cert, T)
        logger.info(f"certificate q per user: {[round(float(q[u * T]), 4) for u in range(B)]}")
        assert fail.tolist() == [False] * (4 * T) + [True] * T + [False] * (3 * T)
        for u in range(4):
            temp, _, p = USERS[u]
            assert bool((q[u * T : u * T + T] >= p + CERT_EPS).all())
            for j in range(T):
                r = u * T + j
                pv = chi_square_p(torch.bincount(draws[:, r], minlength=VOCAB), target_probs(logits[r], temp, 0, p))
                logger.info(f"user {u} column {j}: chi-square p {pv:.3g}")
                assert pv > P_MIN, f"row {r}: the certificate draws do not follow the nucleus (p = {pv})"
        temp, k, p = USERS[5]
        for j in range(T):
            counts = torch.bincount(draws[:, 5 * T + j], minlength=VOCAB)
            by_rule = [
                chi_square_p(counts, target_probs(logits[5 * T + j], temp, k, p, rule)) for rule in ("vllm", "at_most")
            ]
            assert max(by_rule) > P_MIN, f"the top-k 20 row next to the certificate rows: {by_rule}"
        # the redo of the failed user from the logits the engine reads back
        parts = [ttnn.to_torch(ttnn.from_device(t, blocking=True)) for t in ttnn.get_device_tensors(buf)]
        full = torch.cat([t.reshape(-1, t.shape[-1])[: B * T] for t in parts], dim=1).float()
        assert torch.equal(full, logits), "the logits read back differ from the staged ones"
        temp, _, p = USERS[4]
        redone = [int(redo_draws(torch.zeros(B, T), full, fail, cert, [s] * (B * T), T)[4, 0]) for s in range(1, 1501)]
        pv = chi_square_p(
            torch.bincount(torch.tensor(redone), minlength=VOCAB), target_probs(logits[4 * T], temp, 0, p)
        )
        logger.info(f"failed user 4: redo chi-square p {pv:.3g}")
        assert pv > P_MIN, f"the redo does not follow the nucleus (p = {pv})"
    finally:
        ttnn.release_trace(mesh_device, trace)
        ttnn.deallocate(buf)
