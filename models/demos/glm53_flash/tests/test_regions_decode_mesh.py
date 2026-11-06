# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GLM53_REGIONS_DECODE=stream`` on a (2, 4) block (doc/mesh/regions_decode.md): the one-copy decode -- the stream
/ dispatch ops reading the regions prefill path's per-expert lists through the REGIONS_SRC kernels, no stacked copy
-- against (1) today's fallback without the stacked copy (the regions kernel at decode shapes) and (2) the stacked
streaming decode (``GLM53_REGIONS_KEEP_STACKED=1``), on the same prefilled state, eager and traced, over a contiguous
decode ladder.

Model A is the one-copy configuration (``GLM53_PREFILL_EXPERTS=regions GLM53_REGIONS_KEEP_STACKED=0
GLM53_REGIONS_DECODE=stream``: the stacked stream tensors are never uploaded, ``TtRoutedExperts.w13 is None``). The
prompts are prefilled ONCE in 1024-token chunks (512-row slabs per chip; the KDA L1 window of
tests/test_decode_packed_inputs_mesh.py); the 128-token prompt of user slot 3 goes through the regions-sourced
DISPATCH ops (128 gathered rows) and the long chunks through the regions kernel, so the prefill state itself covers
the dispatch variant. Four passes from the same state: S1 = stream eager, R1 = the regions fallback eager
(``GLM53_REGIONS_DECODE=regions`` per call: the knob is read per forward, like the glue knobs), S2 = stream traced,
R2 = regions traced. Then model B (``GLM53_REGIONS_KEEP_STACKED=1 GLM53_REGIONS_DECODE=off``: the regions prefill
with the stacked copy kept for decode) prefills the same prompts and runs T1 = stacked eager, T2 = stacked traced.
Asserted: S1 == S2, R1 == R2, T1 == T2 (eager vs traced), and the headline S1 == T1 == T2 bit for bit on the greedy
ids and the logits (the regions-sourced ops do the same products in the same order as the stacked ops; the prefill
states agree because both models prefill through the regions path and the regions-sourced dispatch ops are
bit-identical to the stacked dispatch ops). R1 vs S1 is logged (logits PCC, greedy agreement; a 0.95 PCC bar: the
regions kernel sums in a different order) and the replay tok/s of all three decodes are logged.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \\
        pytest models/demos/glm53_flash/tests/test_regions_decode_mesh.py -s
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream.regions import REGIONS_DECODE_ENV
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
N_STEPS = int(os.environ.get("GLM53_REGIONS_STEPS", "12"))  # contiguous decode steps per pass
MAX_SEQ_LEN = 2304
PREFILL_CHUNK = 512 * MESH[0]  # 1024-token chunks on 2 rows: 512-row slabs per chip (the KDA L1 window)
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
# prompt lengths per user slot of a row (the rows hold the same prompts); slot 3 = 128 tokens: one chunk of 64 rows
# per chip = 128 gathered rows, the dispatch-op range (33..256) -> the regions-sourced dispatch ops in model A
PROMPT_LENS = [2040, 2044, 2048, 128]
REGIONS_PCC_BAR = 0.95


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-3")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _prompt(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long) % 1000 + 1000


def _moe_experts(model: GLM53MeshModel) -> list:
    return [dec.layer.mlp.experts for dec in model.layers if dec.is_moe]


class _Ladder:
    """The decode steps of one pass: per step the ids (distinct per user slot, seeded, the same on every row like
    the prompts) and the positions (``prompt length + step`` per user, the same on every row)."""

    def __init__(self, model: GLM53MeshModel, lens: list[int], n_steps: int):
        self.model = model
        U = model.total_users
        g = torch.Generator().manual_seed(4321)
        self.steps = [
            (
                torch.randint(0, model.cfg.vocab_size, (B,), generator=g).repeat(model.mesh.rows),
                [lens[u % B] + k for u in range(U)],
            )
            for k in range(n_steps)
        ]

    def read(self, outs) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        _, logits, ids = outs
        host_ids = m.ids_to_host(ids).clone()
        host_logits = (
            m.mesh.to_host(logits, rows=2 if m.mesh.rows > 1 else None).float().reshape(m.total_users, -1).clone()
        )
        return host_ids, host_logits

    def load(self, ids, pos):
        self.model.load_ids(ids)
        self.model.set_positions(pos)

    def eager(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for ids, pos in self.steps:
            self.load(ids, pos)
            outs = self.model.decode_step(self.model.ids_dev, self.model.positions)
            out.append(self.read(outs))
            for t in outs:
                if t is not None:
                    ttnn.deallocate(t)
        return out

    def traced(self, trace_id, trace_outs) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]:
        out, walls = [], []
        for ids, pos in self.steps:
            self.load(ids, pos)
            t0 = time.perf_counter()
            self.model.replay(trace_id)
            walls.append(time.perf_counter() - t0)
            out.append(self.read(trace_outs))
        return out, walls


def _prefill_all(model: GLM53MeshModel, lens: list[int]):
    for u, n in enumerate(lens):
        streams, logits, _ = model.prefill_chunked(_prompt(n), user=u, chunk=PREFILL_CHUNK)
        ttnn.deallocate(streams)
        if logits is not None:
            ttnn.deallocate(logits)
    ttnn.synchronize_device(model.device)


def _assert_identical(name: str, a, b, steps):
    for k, ((ia, la), (ib, lb)) in enumerate(zip(a, b)):
        pos = steps[k][1]
        assert torch.equal(
            ia, ib
        ), f"{name}: greedy ids differ at step {k} (positions {pos}): {ia.tolist()} vs {ib.tolist()}"
        assert torch.equal(la, lb), (
            f"{name}: logits differ at step {k} (positions {pos}): {int((la != lb).sum())} values, "
            f"max abs {(la - lb).abs().max().item():.3g}"
        )


def _compare_soft(name: str, a, b) -> tuple[float, float]:
    """(min logits PCC over the steps, greedy agreement) of two passes that are not expected to be bit-identical."""
    pccs, agree, n = [], 0, 0
    for (ia, la), (ib, lb) in zip(a, b):
        pccs.append(pcc(la, lb))
        agree += int((ia == ib).sum())
        n += ia.numel()
    logger.info(
        f"{name}: logits pcc min {min(pccs):.5f} mean {sum(pccs) / len(pccs):.5f}, greedy ids agree {agree}/{n}"
    )
    return min(pccs), agree / n


def _tok_s(walls: list[float], U: int) -> str:
    ms = sum(walls[2:]) / max(1, len(walls) - 2) * 1e3  # skip the first replays (program cache of the trace)
    return f"{ms:.2f} ms/step = {1e3 / ms:.1f} tok/s per user, {U * 1e3 / ms:.0f} tok/s aggregate"


def _build(mesh_device, layers, cfg) -> GLM53MeshModel:
    t0 = time.time()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    logger.info(f"model built in {time.time() - t0:.0f} s")
    return model


def _release(model: GLM53MeshModel):
    for dec in model.layers:
        dec.release_weights()
    model.release_prefill_traces()
    del model
    gc.collect()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_regions_decode_one_copy_vs_stacked(mesh_device, monkeypatch):
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    lens = [PROMPT_LENS[u % len(PROMPT_LENS)] for u in range(B)]
    assert max(lens) + N_STEPS < MAX_SEQ_LEN and all(n % 4 == 0 for n in lens), (lens, N_STEPS, MAX_SEQ_LEN)
    monkeypatch.setenv("GLM53_PREFILL_EXPERTS", "regions")
    monkeypatch.delenv("GLM53_PARTITION_USERS", raising=False)

    # ================================================================ model A: the one-copy configuration
    monkeypatch.setenv("GLM53_REGIONS_KEEP_STACKED", "0")
    monkeypatch.setenv(REGIONS_DECODE_ENV, "stream")
    model = _build(mesh_device, layers, cfg)
    U = model.total_users
    experts = _moe_experts(model)
    assert experts, "the layer range holds no MoE layer (layers >= 3)"
    for ex in experts:
        assert ex.one_copy and ex.w13 is None and ex.w2 is None, "model A must not hold the stacked copy"
        assert ex.regions_stream_ok, ex.regions_stream_reason
        assert ex._regions_src is not None and ex._regions_src.E == ex.E_local
    plan = experts[0]._regions_src.plan
    logger.info(
        f"mesh {MESH} layers {layers} users/row {B}: regions plan ppb {plan.ppb}, gate|up {plan.gate_packet_tiles}-tile "
        f"packets x {plan.gate_pages_per_col} per column, down {plan.cols} columns x {plan.down_pages_per_expert} packets"
    )
    ladder = _Ladder(model, lens, N_STEPS)
    ids0, pos0 = ladder.steps[0]
    _prefill_all(model, lens)  # once, before any decode step
    used = set().union(*(ex.paths_used for ex in experts))
    assert "dispatch_regions" in used, f"the 128-token prompt should take the regions-sourced dispatch ops: {used}"
    assert "regions" in used, f"the long chunks should take the regions kernel: {used}"
    assert not (used & {"stream", "dispatch"}), f"model A has no stacked copy to stream from: {used}"

    # ---- S1: the one-copy stream decode, eager
    model.warmup_decode(ids0, pos0)
    model.reset_kda_states()
    out_s1 = ladder.eager()
    assert all(ex.last_path == "stream_regions" for ex in experts), [ex.last_path for ex in experts]
    logger.info(f"S1 (regions-sourced stream, eager): ids of the last step {out_s1[-1][0].tolist()}")

    # ---- R1: today's fallback (the regions kernel at decode shapes), eager; its decode-shaped constants are built
    # here, before any capture (the trace rule)
    monkeypatch.setenv(REGIONS_DECODE_ENV, "regions")
    model.warmup_decode(ids0, pos0)
    model.reset_kda_states()
    out_r1 = ladder.eager()
    assert all(ex.last_path == "regions" for ex in experts), [ex.last_path for ex in experts]
    p_min, agree = _compare_soft("R1 (regions kernel) vs S1 (regions-sourced stream)", out_r1, out_s1)
    assert p_min > REGIONS_PCC_BAR, f"the regions fallback diverged from the stream decode: pcc {p_min:.4f}"

    # ---- S2: the one-copy stream decode, traced
    monkeypatch.setenv(REGIONS_DECODE_ENV, "stream")
    trace_id, trace_outs = model.capture(ids0, pos0)
    assert all(ex.last_path == "stream_regions" for ex in experts), [ex.last_path for ex in experts]
    model.reset_kda_states()
    out_s2, walls_s2 = ladder.traced(trace_id, trace_outs)
    _assert_identical("S2 (stream traced) vs S1 (stream eager)", out_s2, out_s1, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"PERF S2 (regions-sourced stream, traced): {_tok_s(walls_s2, U)}")

    # ---- R2: the regions fallback, traced
    monkeypatch.setenv(REGIONS_DECODE_ENV, "regions")
    trace_id, trace_outs = model.capture(ids0, pos0)
    assert all(ex.last_path == "regions" for ex in experts), [ex.last_path for ex in experts]
    model.reset_kda_states()
    out_r2, walls_r2 = ladder.traced(trace_id, trace_outs)
    _assert_identical("R2 (regions traced) vs R1 (regions eager)", out_r2, out_r1, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"PERF R2 (regions kernel fallback, traced): {_tok_s(walls_r2, U)}")
    _release(model)
    del ladder, experts
    gc.collect()
    ttnn.synchronize_device(mesh_device)

    # ================================================================ model B: the stacked copy kept for decode
    monkeypatch.setenv("GLM53_REGIONS_KEEP_STACKED", "1")
    monkeypatch.setenv(REGIONS_DECODE_ENV, "off")
    model = _build(mesh_device, layers, cfg)
    experts = _moe_experts(model)
    for ex in experts:
        assert not ex.one_copy and ex.w13 is not None and ex.w2 is not None, "model B keeps the stacked copy"
    ladder = _Ladder(model, lens, N_STEPS)
    assert ladder.steps[0][0].tolist() == ids0.tolist() and ladder.steps[0][1] == pos0  # the same seeded ladder
    _prefill_all(model, lens)
    used = set().union(*(ex.paths_used for ex in experts))
    assert "dispatch" in used and "regions" in used and "dispatch_regions" not in used, used

    # ---- T1: the stacked stream decode, eager
    model.warmup_decode(ids0, pos0)
    model.reset_kda_states()
    out_t1 = ladder.eager()
    assert all(ex.last_path == "stream" for ex in experts), [ex.last_path for ex in experts]
    _assert_identical("T1 (stacked stream eager) vs S1 (regions-sourced stream eager)", out_t1, out_s1, ladder.steps)

    # ---- T2: the stacked stream decode, traced
    trace_id, trace_outs = model.capture(ids0, pos0)
    model.reset_kda_states()
    out_t2, walls_t2 = ladder.traced(trace_id, trace_outs)
    _assert_identical("T2 (stacked stream traced) vs T1", out_t2, out_t1, ladder.steps)
    _assert_identical("T2 (stacked stream traced) vs S2 (regions-sourced stream traced)", out_t2, out_s2, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(f"PERF T2 (stacked stream, traced): {_tok_s(walls_t2, U)}")
    logger.info(
        f"PERF mesh {MESH} layers {len(layers)} users {U}: regions-sourced stream {_tok_s(walls_s2, U)}; "
        f"stacked stream {_tok_s(walls_t2, U)}; regions kernel fallback {_tok_s(walls_r2, U)}; "
        f"one-copy decode bit-identical to the stacked decode over {N_STEPS} steps (eager and traced); "
        f"regions fallback vs stream: logits pcc min {p_min:.5f}, greedy agreement {agree:.3f}"
    )
    _release(model)
