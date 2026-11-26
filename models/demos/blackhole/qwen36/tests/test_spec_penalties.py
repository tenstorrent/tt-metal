# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the verify penalties of Qwen36MTPContractForCausalLM (tt/spec_penalties.py VerifyPenalties) on controlled
logits, as the model runs them: persistent buffers of each device's vocabulary shard [1, 1, rows, V / TP] bfloat16
(V = 248320), the bodies warmed eagerly and captured as traces with program cache misses disallowed, the counts written
by the eager rebuild and add passes.

* The penalty trace over the B-user rows and over a verify row bucket's rows: column j of every penalty row is vLLM's
  apply_penalties over the committed output plus drafts 1 .. j (spec_penalties.reference_penalties), rounded once to
  bfloat16; every token of a neutral row, and every token a frequency or presence penalty does not touch, is bit for bit
  what was written; a greedy penalty row's verify ids are the float32 argmax of its penalized logits (the plugin's host
  path), across the shards; the other lanes keep their ids.
* The commit trace of a chained verify adds each tracked user's commit (its ids, or its draws, up to the drafter pack's
  count) and leaves the other users.

    pytest models/demos/blackhole/qwen36/tests/test_spec_penalties.py    (Blackhole; MESH_DEVICE=P150x8)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax
from models.demos.blackhole.qwen36.tt.spec_penalties import (
    StepPenalties,
    VerifyPenalties,
    counts_of,
    reference_penalties,
)
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, draft_anchor, drafter_pack_rows, pack_width

VOCAB = 248320
MESH = _resolve_mesh_shape(8)
B, T = 8, 4
R = B * T
BUCKET = 2  # users of the verify row bucket whose trace runs too
# user -> (repetition, frequency, presence, greedy): 0-2 one penalty each, 3 all three, 4 a repetition below 1;
# 3 and 4 are drawn rows (their ids stay), 5-7 neutral.
PENALTIES = {
    0: (1.3, 0.0, 0.0, True),
    1: (1.0, 0.4, 0.0, True),
    2: (1.0, 0.0, 0.5, True),
    3: (1.15, 0.2, 0.3, False),
    4: (0.8, 0.0, 0.0, False),
}


def _parametrize(fn):
    # The fused argmax all-gathers each shard's maximum.
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _shards(mesh):
    return mesh.get_num_devices()


def _mapper(mesh):
    return ttnn.ShardTensorToMesh(mesh, dim=3) if _shards(mesh) > 1 else ttnn.ReplicateTensorToMesh(mesh)


def _replicated(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _write(mesh, host, dev, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mapper=None):
    mapper = ttnn.ReplicateTensorToMesh(mesh) if mapper is None else mapper
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=mapper), dev)


def _logits_buffer(mesh, rows):
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=_mapper(mesh),
    )


def _read_logits(mesh, buf):
    rows = int(buf.shape[2])
    if _shards(mesh) > 1:
        return ttnn.to_torch(buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).reshape(rows, VOCAB)
    return ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).reshape(rows, VOCAB)


def _lanes(t, n=32):
    """[devices, n] words of the [1, 1, 1, >= n] uint32 ``t``."""
    return torch.stack([ttnn.to_torch(d).reshape(-1)[:n].to(torch.int64) for d in ttnn.get_device_tensors(t)])


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


def _bf16_ulps(a, b):
    """|a - b| in units of b's bfloat16 ulp (a, b float32 holding bfloat16 values)."""
    scale = torch.pow(2.0, torch.floor(torch.log2(b.abs().clamp_min(1e-30))) - 7)
    return ((a - b).abs() / scale).max().item() if a.numel() else 0.0


class _Rig:
    """A VerifyPenalties over persistent stand-ins of what the model binds it to: the verify step buffer, the verify
    ids lanes, the sampler's draws, the drafter pack and the vocabulary shard of the B-user and the bucket verify."""

    def __init__(self, mesh):
        self.mesh = mesh
        fused = FusedShardArgmax(mesh, VOCAB, _shards(mesh))
        self.pen = VerifyPenalties(mesh, B, T, VOCAB, shards=_shards(mesh), argmax=lambda x: fused(x, LANES))
        self.step = _replicated(mesh, torch.zeros(VERIFY_STEP_ROWS, pack_width(R), dtype=torch.int32), ttnn.uint32)
        self.ids = _replicated(mesh, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        self.draws = _replicated(mesh, torch.zeros(1, 1, 1, R, dtype=torch.int32), ttnn.uint32)
        self.pack = _replicated(
            mesh, torch.zeros(drafter_pack_rows(T - 1), pack_width(R), dtype=torch.int32), ttnn.uint32
        )
        self.logits = {R: _logits_buffer(mesh, R), BUCKET * T: _logits_buffer(mesh, BUCKET * T)}
        self.pen.bind(step=self.step, ids_out=self.ids, draws=self.draws, pack=self.pack)
        for buf in self.logits.values():
            self.pen.apply_body(buf)
        self.pen.commit_body()
        ttnn.synchronize_device(mesh)
        self.pen.warm_eager()
        self.apply = {rows: _capture(mesh, lambda b=buf: self.pen.apply_body(b)) for rows, buf in self.logits.items()}
        self.commit = _capture(mesh, self.pen.commit_body)

    def stage_tokens(self, tokens):
        step = torch.zeros(VERIFY_STEP_ROWS, pack_width(R), dtype=torch.int32)
        step[0, :R] = tokens.reshape(-1).to(torch.int32)
        _write(self.mesh, step, self.step, ttnn.uint32)

    def replay(self, trace):
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
        finally:
            self.mesh.set_program_cache_misses_allowed(True)

    def release(self):
        for trace in (*self.apply.values(), self.commit):
            ttnn.release_trace(self.mesh, trace)
        self.pen.release()


def _step_penalties():
    rep, freq, pres = torch.ones(B), torch.zeros(B), torch.zeros(B)
    for u, (r, f, p, _) in PENALTIES.items():
        rep[u], freq[u], pres[u] = r, f, p
    live = torch.tensor([u in PENALTIES for u in range(B)])
    return StepPenalties(rep, freq, pres, live)


def _greedy_lanes():
    return [int(PENALTIES.get(r // T, (0, 0, 0, False))[3]) for r in range(R)]


@_parametrize
def test_the_penalty_trace_scores_every_column_as_vllm_does(mesh_device):
    rig = _Rig(mesh_device)
    g = torch.Generator().manual_seed(4)
    try:
        prompts = {
            u: torch.randint(0, VOCAB, (n,), generator=g) for u, n in zip(range(B), (300, 9, 40, 1000, 77, 5, 5, 5))
        }
        outputs = {
            u: torch.randint(0, VOCAB, (n,), generator=g) for u, n in zip(range(B), (60, 400, 25, 33, 12, 3, 3, 3))
        }
        for u in range(B):  # a frequent token, and one repeated in the prompt
            outputs[u][:10] = outputs[u][0]
        tokens = torch.randint(0, VOCAB, (B, T), generator=g)
        tokens[:, 0] = torch.stack([outputs[u][-1] for u in range(B)])
        tokens[0, 2] = outputs[0][0]  # a draft already in the history
        tokens[1, 3] = tokens[1, 0]  # a padded column repeats column 0
        rig.pen.rebuild({u: (prompts[u], outputs[u]) for u in range(B)})
        rig.pen.stage(_step_penalties(), _greedy_lanes())
        rig.stage_tokens(tokens)
        for rows in (R, BUCKET * T):
            raw = (torch.randn(rows, VOCAB, generator=g) * 3).to(torch.bfloat16)
            for r in range(rows):  # each row's maximum is a token its penalties lower
                u = r // T
                raw[r, int(outputs[u][0])] = raw[r].max() + 0.25
                raw[r, int(prompts[u][0])] = raw[r].max() + 0.125
            _write(mesh_device, raw.reshape(1, 1, rows, VOCAB), rig.logits[rows], ttnn.bfloat16, ttnn.TILE_LAYOUT,
                   _mapper(mesh_device))  # fmt: skip
            ids_before = torch.arange(1000, 1032, dtype=torch.int32).reshape(1, 1, 1, 32)
            _write(mesh_device, ids_before, rig.ids, ttnn.uint32)
            rig.replay(rig.apply[rows])
            got = _read_logits(mesh_device, rig.logits[rows]).to(torch.float32)
            lanes = _lanes(rig.ids)
            assert bool((lanes == lanes[0]).all()), "the devices of the mesh hold different ids"
            worst, exact, greedy_flips = 0.0, 0, 0
            for r in range(rows):
                u, j = divmod(r, T)
                x = raw[r].to(torch.float32)
                if u not in PENALTIES:
                    assert torch.equal(got[r], x), f"neutral row {r} changed"
                    assert int(lanes[0, r]) == 1000 + r
                    continue
                rep, freq, pres, greedy = PENALTIES[u]
                history = torch.cat([outputs[u], tokens[u, 1 : j + 1]])
                want32 = reference_penalties(x, prompts[u], history, rep, freq, pres)
                want = want32.to(torch.bfloat16).to(torch.float32)
                touched = counts_of(history, VOCAB) > 0
                if rep != 1.0:
                    touched |= counts_of(prompts[u], VOCAB) > 0
                assert torch.equal(got[r, ~touched], x[~touched]), f"row {r}: a token no penalty touches changed"
                worst = max(worst, _bf16_ulps(got[r, touched], want[touched]))
                exact += int(torch.equal(got[r], want))
                if greedy:
                    assert int(lanes[0, r]) == int(want32.argmax()), f"row {r}: not the float32 argmax"
                    greedy_flips += int(want.argmax()) != int(want32.argmax())
                else:
                    assert int(lanes[0, r]) == 1000 + r, f"drawn row {r}: its ids lane changed"
            penalized = sum(1 for r in range(rows) if r // T in PENALTIES)
            logger.info(
                f"[penalties] {rows} rows: {exact} of {penalized} penalized rows bit-exact to the bfloat16 reference, "
                f"worst {worst:.3f} bf16 ulp; {greedy_flips} greedy rows whose bfloat16 argmax would differ"
            )
            assert worst <= 1.0, f"a penalized logit is {worst} bf16 ulp off the reference"
    finally:
        rig.release()


@_parametrize
def test_the_commit_trace_adds_each_tracked_users_commit(mesh_device):
    rig = _Rig(mesh_device)
    g = torch.Generator().manual_seed(5)
    try:
        outputs = {u: torch.randint(0, VOCAB, (20,), generator=g) for u in range(B)}
        rig.pen.rebuild({u: (torch.tensor([], dtype=torch.int64), outputs[u]) for u in range(B)})
        ids = torch.randint(0, VOCAB, (B, T), generator=g)
        draws = torch.randint(0, VOCAB, (B, T), generator=g)
        m = [u % T for u in range(B)]
        lanes = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        lanes[0, 0, 0, :R] = ids.reshape(-1).to(torch.int32)
        _write(mesh_device, lanes, rig.ids, ttnn.uint32)
        _write(mesh_device, draws.reshape(1, 1, 1, R).to(torch.int32), rig.draws, ttnn.uint32)
        pack = torch.zeros(drafter_pack_rows(T - 1), pack_width(R), dtype=torch.int32)
        pack[draft_anchor(T - 1), :B] = torch.tensor([u * T + m[u] for u in range(B)], dtype=torch.int32)
        _write(mesh_device, pack, rig.pack, ttnn.uint32)
        tracked = [u < 6 for u in range(B)]
        drawn = [int(u % 2) for u in range(B)]
        rig.pen.stage(_step_penalties(), [0] * R, track=tracked, drawn=drawn)
        rig.replay(rig.commit)
        counts, _ = rig.pen.read(range(B))
        for u in range(B):
            commit = (draws if drawn[u] else ids)[u, : m[u] + 1]
            history = torch.cat([outputs[u], commit]) if tracked[u] else outputs[u]
            assert torch.equal(counts[u], counts_of(history, VOCAB)), f"user {u}"
        rig.pen.add({0: torch.tensor([7, 7, 248319])})
        assert int(rig.pen.read([0])[0][0, 7]) == int(counts[0, 7]) + 2
    finally:
        rig.release()
