# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GLM53_DECODE_PACKED_INPUTS`` on a (2, 4) block: the one-upload decode inputs (``tt/packed_inputs.py``,
``GLM53MeshModel.load_decode_inputs`` + the device-side unpack at the start of ``decode_step``) against the direct
per-field refreshes, eager and traced, bit-identical greedy ids and logits over a contiguous decode ladder that
crosses an index_kpool boundary at every step, the first sparse pool (2047 -> 2048) and the dense bound
(2050 -> 2051) for different users at different steps, with a dense user alongside; the unpacked field tensors
(ids, positions, pool_ids, ring_slots, cur_pos, tail_pool, valid_len_rows, the per-user valid_len list) are read
back per chip and compared with the direct path's host values; the user-partitioned cache is the second case.

Four passes over the same ladder, each starting from the same prefilled state (every user's prompt re-prefilled
after the pass's warm-up / capture, so every value a step reads was written by this pass): direct eager, packed
eager (+ the field checks), packed traced, direct traced. The packed segments are allocated before the first
capture (the trace rule), the per-length prefill traces are not used (eager prefills only).

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \\
        pytest models/demos/glm53_flash/tests/test_decode_packed_inputs_mesh.py -s
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import SparseDecodeMeta, dense_exact_max_pos
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import PACKED_INPUTS_ENV, GLM53MeshModel, PackedDecodeInputs
from models.demos.glm53_flash.tt.packed_inputs import PackedDecodeLayout

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
N_STEPS = int(os.environ.get("GLM53_PACKED_STEPS", "16"))  # contiguous decode steps per pass
MAX_SEQ_LEN = 2304
WARM_POS = MAX_SEQ_LEN - 4  # the warm-up / capture position: its state writes are never read by the ladder
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
# prompt lengths per user slot of a row (the rows hold the same prompts): user 0 crosses 2047 -> 2048 at step 8 and
# 2050 -> 2051 at step 11, user 1 at steps 4 / 7, user 2 starts sparse and crosses 2050 at step 3, user 3 stays dense
PROMPT_LENS = [2040, 2044, 2048, 600]


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-3")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _prompt(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long) % 1000 + 1000


class _Ladder:
    """The decode steps of one pass: per step the ids (distinct per user slot, seeded, the same on every row like
    the prompts) and the positions (``prompt length + step`` per user, the same on every row)."""

    def __init__(self, model: GLM53MeshModel, lens: list[int], n_steps: int):
        self.model = model
        U = model.total_users
        g = torch.Generator().manual_seed(1234)
        self.steps = [
            (
                torch.randint(0, model.cfg.vocab_size, (B,), generator=g).repeat(model.mesh.rows),
                [lens[u % B] + k for u in range(U)],
            )
            for k in range(n_steps)
        ]

    def read(self, outs) -> tuple[torch.Tensor, torch.Tensor]:
        """(greedy ids [U] int64, logits [U, V] fp32) of a step's outputs (one chip read per mesh row each)."""
        m = self.model
        _, logits, ids = outs
        host_ids = m.ids_to_host(ids).clone()
        host_logits = (
            m.mesh.to_host(logits, rows=2 if m.mesh.rows > 1 else None).float().reshape(m.total_users, -1).clone()
        )
        return host_ids, host_logits

    def eager(self, load, on_step=None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for k, (ids, pos) in enumerate(self.steps):
            load(k, ids, pos)
            outs = self.model.decode_step(self.model.ids_dev, self.model.positions)
            if on_step is not None:
                on_step(k, ids, pos)
            out.append(self.read(outs))
            for t in outs:
                if t is not None:
                    ttnn.deallocate(t)
        return out

    def traced(
        self, trace_id, trace_outs, load
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float], list[float]]:
        """Replays; returns the outputs, the per-step (load + replay) walls and the load-only walls (seconds)."""
        out, walls, loads = [], [], []
        for k, (ids, pos) in enumerate(self.steps):
            t0 = time.perf_counter()
            load(k, ids, pos)
            t1 = time.perf_counter()
            self.model.replay(trace_id)
            walls.append(time.perf_counter() - t0)
            loads.append(t1 - t0)
            out.append(self.read(trace_outs))
        return out, walls, loads


def _prefill_all(model: GLM53MeshModel, lens: list[int]):
    """Every user slot's prompt (eager, sequence-parallel over the rows: slot u of every row holds it): the KDA
    slots, the latent cache rows [0, S) and the pooled-key index are rewritten from position 0."""
    for u, n in enumerate(lens):
        streams, logits, _ = model.prefill(_prompt(n), user=u)
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


def _chip_values(model: GLM53MeshModel, t: ttnn.Tensor, chip: int) -> list[int]:
    return [int(v) for v in ttnn.to_torch(model.mesh.device_tensors(t)[chip]).view(torch.int32).reshape(-1).tolist()]


def _check_fields(
    model: GLM53MeshModel, layout: PackedDecodeLayout, ids: torch.Tensor, pos: list[int], *, per_user: bool
):
    """After an unpack: every persistent field tensor, read per chip, equals the direct path's host values (the
    planner's expectation and, independently, tt/dsa.py's ``SparseDecodeMeta.host_values`` / the position formulas)."""
    mesh, dp, cfg = model.mesh, model.positions, model.cfg
    host_ids = model.embedding.tokens_to_host(ids, pad_to=model.total_users).reshape(-1)
    valid, tail, cur = SparseDecodeMeta.host_values(pos, cfg)
    P, Dm = cfg.index_kpool, dense_exact_max_pos(cfg)
    tensors = {"ids": model.ids_dev, "positions": dp.positions, "pool_ids": dp.pool_ids, "ring_slots": dp.ring_slots}
    if dp.dense_positions is not None:
        tensors["dense_positions"] = dp.dense_positions
    tensors.update(
        {"cur_pos": dp.sparse.cur_pos, "tail_pool": dp.sparse.tail_pool, "valid_len_rows": dp.sparse.valid_len_rows}
    )
    if per_user:
        for u, vt in enumerate(dp.sparse.valid_len):
            tensors[f"valid_len[{u}]"] = vt
    for r in range(mesh.rows):
        for c in range(mesh.cols):
            chip = r * mesh.cols + c
            exp = layout.expected_chip_fields(host_ids, pos, r, c)
            first, end = layout.chip_users(r, c)
            direct = {
                "ids": host_ids[r * B : (r + 1) * B].tolist(),
                "positions": pos[first:end],
                "pool_ids": [p // P for p in pos[first:end]],
                "ring_slots": [p % P for p in pos[first:end]],
                "dense_positions": [min(p, Dm) for p in pos[first:end]],
                "cur_pos": cur[first:end].tolist(),
                "tail_pool": tail.reshape(-1)[first:end].tolist(),
                "valid_len_rows": valid[first:end].tolist(),
            }
            for u in range(end - first):
                direct[f"valid_len[{u}]"] = [int(valid[first + u])]
            for name, t in tensors.items():
                got = _chip_values(model, t, chip)
                assert (
                    got == exp[name] == direct[name]
                ), f"chip {chip} field {name}: device {got}, planner {exp[name]}, direct {direct[name]}"
            seg = layout.segment_of(model.packed.host, layout.segment(r, c))
            dev_seg = ttnn.to_torch(mesh.device_tensors(model.packed.dev)[chip]).view(torch.int32)
            assert torch.equal(
                dev_seg.reshape(-1), seg.reshape(-1)
            ), f"chip {chip}: packed segment differs from the host segment"


def _tok_s(walls: list[float], U: int) -> str:
    ms = sum(walls[2:]) / max(1, len(walls) - 2) * 1e3  # skip the first replays (program cache of the trace)
    return f"{ms:.2f} ms/step = {1e3 / ms:.1f} tok/s per user, {U * 1e3 / ms:.0f} tok/s aggregate"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("partition", ["0", "1"])
def test_packed_decode_inputs_bit_identical(mesh_device, monkeypatch, partition):
    if partition == "1" and B % MESH[1]:
        pytest.skip(f"the user partition needs users_per_row {B} % cols {MESH[1]} == 0")
    monkeypatch.setenv("GLM53_PARTITION_USERS", partition)
    monkeypatch.delenv(PACKED_INPUTS_ENV, raising=False)
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
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
    U = model.total_users
    lens = [PROMPT_LENS[u % len(PROMPT_LENS)] for u in range(B)]
    assert max(lens) + N_STEPS < WARM_POS, (lens, N_STEPS, WARM_POS)
    ladder = _Ladder(model, lens, N_STEPS)
    warm_ids = torch.full((U,), 1001, dtype=torch.long)
    warm_pos = [WARM_POS] * U
    logger.info(
        f"mesh {MESH} layers {layers} users/row {B} partition {partition}: prompts {lens}, {N_STEPS} steps per pass"
    )

    def direct_load(k, ids, pos):
        model.load_ids(ids)
        model.set_positions(pos)

    def packed_load(k, ids, pos):  # both packed routes: the one-call load and the load_ids + set_positions pair
        if k % 2 == 0:
            model.load_decode_inputs(ids, pos)
        else:
            model.load_ids(ids)
            model.set_positions(pos)

    # ---- pass A: the direct per-field refreshes, eager (compiles every program, the sparse buffers included)
    model.warmup_decode(warm_ids, warm_pos)
    assert model.packed is None
    _prefill_all(model, lens)
    out_a = ladder.eager(direct_load)
    logger.info(f"pass A (direct, eager): ids of the last step {out_a[-1][0].tolist()}")

    # ---- pass B: the packed inputs, eager; the segments are allocated here, before the first capture
    monkeypatch.setenv(PACKED_INPUTS_ENV, "1")
    model.warmup_decode(warm_ids, warm_pos)
    assert model.packed is not None and model.packed.layout.dense == (model.positions.dense_positions is not None)
    layout = model.packed.layout
    logger.info(
        f"packed layout: segment {layout.segment_shape} int32, {layout.segments} segments, fields {list(layout.fields)}"
    )
    # the per-user valid_len list (consumed by the unbatched top-k only): its own unpack, checked once, eagerly
    per_user = dataclasses.replace(layout, per_user_valid_len=True)
    extra = PackedDecodeInputs(model.mesh, per_user, per_user.pack(model._host_ids, model.positions.host))
    ids0, pos0 = ladder.steps[0]
    model.layers[0].update_decode_positions(model.positions, pos0, upload=False)
    model._host_ids = model.embedding.tokens_to_host(ids0, pad_to=U).reshape(-1)
    extra.upload(per_user.pack(model._host_ids, pos0))
    extra.unpack(model.ids_dev, model.positions)
    ttnn.synchronize_device(mesh_device)
    saved_packed = model.packed
    model.packed = extra  # _check_fields reads model.packed.host / .dev
    _check_fields(model, per_user, ids0, pos0, per_user=True)
    model.packed = saved_packed
    ttnn.deallocate(extra.dev)
    _prefill_all(model, lens)
    out_b = ladder.eager(
        packed_load, on_step=lambda k, ids, pos: _check_fields(model, layout, ids, pos, per_user=False)
    )
    _assert_identical("packed eager vs direct eager", out_b, out_a, ladder.steps)
    logger.info("pass B (packed, eager): bit-identical to pass A; every unpacked field matches the direct values")

    # ---- pass C: the packed inputs, traced (the unpack is part of the captured step)
    trace_id, trace_outs = model.capture(warm_ids, warm_pos)
    _prefill_all(model, lens)
    out_c, walls_c, loads_c = ladder.traced(trace_id, trace_outs, packed_load)
    _assert_identical("packed traced vs direct eager", out_c, out_a, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(
        f"PERF pass C (packed, traced): {_tok_s(walls_c, U)} (host wall per step incl. the one upload: "
        f"{sum(loads_c[2:]) / max(1, len(loads_c) - 2) * 1e3:.2f} ms; one chip read per row for the ids)"
    )

    # ---- pass D: the direct refreshes, traced (the knob off again; the segments stay allocated, unused)
    monkeypatch.delenv(PACKED_INPUTS_ENV, raising=False)
    model.packed = None
    trace_id, trace_outs = model.capture(warm_ids, warm_pos)
    _prefill_all(model, lens)
    out_d, walls_d, loads_d = ladder.traced(trace_id, trace_outs, direct_load)
    _assert_identical("direct traced vs direct eager", out_d, out_a, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(
        f"PERF pass D (direct, traced): {_tok_s(walls_d, U)} (host wall per step incl. the per-field uploads: "
        f"{sum(loads_d[2:]) / max(1, len(loads_d) - 2) * 1e3:.2f} ms)"
    )
    rows_agree = sum(torch.equal(i[: U // 2], i[U // 2 :]) for i, _ in out_a) if model.mesh.rows == 2 else None
    logger.info(
        f"PERF mesh {MESH} layers {len(layers)} users {U} partition {partition}: packed traced {_tok_s(walls_c, U)}; "
        f"direct traced {_tok_s(walls_d, U)}; all four passes bit-identical over {N_STEPS} steps"
        + (f"; rows agree on the greedy ids in {rows_agree}/{N_STEPS} steps (info)" if rows_agree is not None else "")
    )
