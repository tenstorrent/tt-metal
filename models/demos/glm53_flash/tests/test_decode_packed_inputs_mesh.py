# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GLM53_DECODE_PACKED_INPUTS`` on a (2, 4) block: the one-upload decode inputs (``tt/packed_inputs.py``,
``GLM53MeshModel.load_decode_inputs`` + the device-side unpack at the start of ``decode_step``) against the direct
per-field refreshes, eager and traced, bit-identical greedy ids and logits over a contiguous decode ladder that
crosses an index_kpool boundary at every step, the first sparse pool (2047 -> 2048) and the dense bound
(2050 -> 2051) for different users at different steps, with a dense user alongside; the unpacked field tensors
(ids, positions, pool_ids, ring_slots, cur_pos, tail_pool, valid_len_rows, the per-user valid_len list) are read
back per chip and compared with the direct path's host values; the user-partitioned cache is the second case.

Four passes over the same ladder from the same state: direct eager, packed eager (+ the field checks), packed
traced, direct traced; then the resident chain (``GLM53_RESIDENT_DECODE=1``, ``tt/resident_decode.py``): the
captured step feeds its greedy ids back and advances the live positions on device, one load at the chain start and
replays with no upload, bit-identical to the direct eager chain (step k feeds step k-1's greedy ids), every
device-advanced field equal to the host formulas after each replay, and a parked slot (live 0) keeping its position.
Pass F (``GLM53_DECODE_SLOT_MASK=1``, the live mask as the ``kda_decode`` write gate): slot 0's prompt prefilled
in two chunks with masked decode steps of the other slots between them; its KDA carry is byte-identical across
the parked steps and its chain after the second chunk equals the uninterleaved reference (the interleave's model
side, doc/mesh/resident_decode.md step 4). The prompts are prefilled ONCE, before any decode step, in ``PREFILL_CHUNK``-token chunks
(512-row slabs per chip, the chunked prefill the adapter uses) because of the L1 hazard below. Every pass warms
up / captures at the ladder's first positions (the prompt ends, multiples of 4: the warm-up's latent row and ring
slot are rewritten by step 0 before anything reads them and no pool completes there) and resets the KDA states
before its ladder, so every value a step reads was written by the prefill or by this pass (the pools / rows of the
previous pass beyond the current position are never ranked or attended). The packed segments are allocated before
the first capture (the trace rule); the per-length prefill traces are not used.

L1 hazard of long prefill slabs on a (2, 4) block (the first two block runs of this test failed on it, with and
without a decode step before the prefill; nothing of the packed path was involved): the KDA sequence-parallel
prefill keeps its chunk-preparation terms and activations in interleaved L1 under the static per-core budgets of
``deepseek_v3_d_p/tt/kda/config.py`` (sized for Kimi-K3 on the Galaxy: 24 heads, 640 rows per device). GLM has
16 KDA heads per chip (64 / TP4) and 64 KiB of chunk terms per head per 32-row chunk, so a 1024-row slab (a
2048-token chunk on 2 rows) holds 32 MiB of terms = 273 KiB per core on the 12x10 grid (under the 320 KiB budget,
so L1) plus the qkv / decay / norm / partial-output / summary buffers: 608 KiB per core in all (observed: L1
buffers from 950016 B up to the 1.5 MiB top). ``ttnn.experimental.kda.affine_exclusive_scan`` then statically
allocates a fixed ~896 KiB of dataflow buffers per worker core (K = V = 128: 256 tile slots, independent of the
slab length) whose region ends at 1029280 B, 77 KiB into those buffers -> "Statically allocated dataflow buffers
in program N clash with L1 buffers on core range [0-0 - 11-1]" at the first prefill of the first KDA layer.
512-row slabs use ~320 KiB per core (safe by ~200 KiB), 768-row slabs ~450 KiB (safe by ~80 KiB), 1024-row slabs
clash; from 1216 rows the terms exceed the 37.5 MiB budget and spill to DRAM, which is why the 8x4 serving chunks
(16 K tokens = 2048-row slabs) and the 128-token warm-ups (64-row slabs) never hit it. By the same arithmetic any
chunk whose padded length gives ~800-1200 rows per chip is in the window: 1600-2400-token chunks on a 2-row
block (the 2048-token prefill trace of ``DEFAULT_LENGTHS`` included), 6.4-9.6 K-token single-chunk prompts on
the 8x4 (unverified on the device); keep block prefill chunks <= 1024 tokens (512-row slabs).

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
from models.demos.glm53_flash.tt.packed_inputs import RESIDENT_ENV, SLOT_MASK_ENV, PackedDecodeLayout

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
N_STEPS = int(os.environ.get("GLM53_PACKED_STEPS", "16"))  # contiguous decode steps per pass
MAX_SEQ_LEN = 2304
# the prefill chunk: 512-row slabs per chip (the module docstring's L1 hazard: 1024-row slabs clash in the KDA scan)
PREFILL_CHUNK = 512 * MESH[0]
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

    def eager_chain(self, load):
        """The greedy chain from the first step (the resident reference): step k feeds step k-1's greedy ids at the
        first positions + k. Returns (outputs, the (ids, positions) fed per step)."""
        out, steps = [], []
        ids, pos0 = self.steps[0]
        for k in range(len(self.steps)):
            pos = [p + k for p in pos0]
            steps.append((ids, pos))
            load(k, ids, pos)
            outs = self.model.decode_step(self.model.ids_dev, self.model.positions)
            out.append(self.read(outs))
            for t in outs:
                if t is not None:
                    ttnn.deallocate(t)
            ids = out[-1][0]
        return out, steps

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
    """Every user slot's prompt (eager, sequence-parallel over the rows: slot u of every row holds it) in
    ``PREFILL_CHUNK``-token chunks (the chunked-prefill contract: ``start`` offsets, 512-row slabs per chip; the
    module docstring's L1 hazard): the KDA slots, the latent cache rows [0, S) and the pooled-key index are written
    from position 0. Runs once, before any decode step."""
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


def _chip_values(model: GLM53MeshModel, t: ttnn.Tensor, chip: int) -> list[int]:
    return [int(v) for v in ttnn.to_torch(model.mesh.device_tensors(t)[chip]).view(torch.int32).reshape(-1).tolist()]


def _check_fields(
    model: GLM53MeshModel,
    layout: PackedDecodeLayout,
    ids: torch.Tensor,
    pos: list[int],
    *,
    per_user: bool,
    packed: bool = True,
):
    """After an unpack (or a resident step's device-side advance, ``packed=False``: no segment to compare): every
    persistent field tensor, read per chip, equals the direct path's host values (the planner's expectation and,
    independently, tt/dsa.py's ``SparseDecodeMeta.host_values`` / the position formulas)."""
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
            if not packed:
                continue
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
    monkeypatch.setenv(PACKED_INPUTS_ENV, "0")  # the direct path first (the knob defaults on since 2026-10-27)
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
    assert max(lens) + N_STEPS < MAX_SEQ_LEN and all(n % 4 == 0 for n in lens), (lens, N_STEPS, MAX_SEQ_LEN)
    ladder = _Ladder(model, lens, N_STEPS)
    # every pass warms up / captures at the ladder's first positions (rewritten by step 0) with its ids
    ids0, pos0 = ladder.steps[0]
    logger.info(
        f"mesh {MESH} layers {layers} users/row {B} partition {partition}: prompts {lens}, {N_STEPS} steps per pass"
    )
    _prefill_all(model, lens)  # once, before any decode step

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
    model.warmup_decode(ids0, pos0)
    assert model.packed is None
    model.reset_kda_states()  # every pass decodes the ladder from the zero KDA state
    out_a = ladder.eager(direct_load)
    logger.info(f"pass A (direct, eager): ids of the last step {out_a[-1][0].tolist()}")

    # ---- pass B: the packed inputs, eager; the segments are allocated here, before the first capture
    monkeypatch.setenv(PACKED_INPUTS_ENV, "1")
    model.warmup_decode(ids0, pos0)
    assert model.packed is not None and model.packed.layout.dense == (model.positions.dense_positions is not None)
    layout = model.packed.layout
    logger.info(
        f"packed layout: segment {layout.segment_shape} int32, {layout.segments} segments, fields {list(layout.fields)}"
    )
    # the per-user valid_len list (consumed by the unbatched top-k only): its own unpack of the warm-up's values
    # (ids0 / pos0, what the model's segment holds now), checked once, eagerly, before the first capture
    per_user = dataclasses.replace(layout, per_user_valid_len=True)
    extra = PackedDecodeInputs(model.mesh, per_user, per_user.pack(model._host_ids, model.positions.host))
    extra.unpack(model.ids_dev, model.positions)
    ttnn.synchronize_device(mesh_device)
    saved_packed = model.packed
    model.packed = extra  # _check_fields reads model.packed.host / .dev
    _check_fields(model, per_user, ids0, pos0, per_user=True)
    model.packed = saved_packed
    ttnn.deallocate(extra.dev)
    model.reset_kda_states()
    out_b = ladder.eager(
        packed_load, on_step=lambda k, ids, pos: _check_fields(model, layout, ids, pos, per_user=False)
    )
    _assert_identical("packed eager vs direct eager", out_b, out_a, ladder.steps)
    logger.info("pass B (packed, eager): bit-identical to pass A; every unpacked field matches the direct values")

    # ---- pass C: the packed inputs, traced (the unpack is part of the captured step)
    trace_id, trace_outs = model.capture(ids0, pos0)
    model.reset_kda_states()
    out_c, walls_c, loads_c = ladder.traced(trace_id, trace_outs, packed_load)
    _assert_identical("packed traced vs direct eager", out_c, out_a, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(
        f"PERF pass C (packed, traced): {_tok_s(walls_c, U)} (host wall per step incl. the one upload: "
        f"{sum(loads_c[2:]) / max(1, len(loads_c) - 2) * 1e3:.2f} ms; one chip read per row for the ids)"
    )

    # ---- pass D: the direct refreshes, traced (the knob off again; the segments stay allocated, unused)
    monkeypatch.setenv(PACKED_INPUTS_ENV, "0")
    model.packed = None
    trace_id, trace_outs = model.capture(ids0, pos0)
    model.reset_kda_states()
    out_d, walls_d, loads_d = ladder.traced(trace_id, trace_outs, direct_load)
    _assert_identical("direct traced vs direct eager", out_d, out_a, ladder.steps)
    ttnn.release_trace(mesh_device, trace_id)
    logger.info(
        f"PERF pass D (direct, traced): {_tok_s(walls_d, U)} (host wall per step incl. the per-field uploads: "
        f"{sum(loads_d[2:]) / max(1, len(loads_d) - 2) * 1e3:.2f} ms)"
    )
    # ---- pass E: the resident chain (GLM53_RESIDENT_DECODE=1 with the packed inputs): the captured step feeds its
    # greedy ids back and advances the live positions on device; one load at the chain start, then replays that
    # upload nothing. Reference: the direct eager chain from (ids0, pos0) (traced == eager by passes C / D).
    model.packed = saved_packed
    monkeypatch.setenv(PACKED_INPUTS_ENV, "1")
    model.reset_kda_states()
    chain_ref, chain_steps = ladder.eager_chain(packed_load)
    monkeypatch.setenv(RESIDENT_ENV, "1")
    model.warmup_decode(ids0, pos0)  # allocates the feedback shadows (before the capture), compiles the feedback
    assert model.resident is not None
    trace_id, trace_outs = model.capture(ids0, pos0)
    model.reset_kda_states()
    t0 = time.perf_counter()
    model.load_decode_inputs(ids0, pos0, live=[True] * U)  # the chain start: the one host upload
    load_e = time.perf_counter() - t0
    out_e, walls_e = [], []
    for k in range(N_STEPS):
        t0 = time.perf_counter()
        model.replay(trace_id)
        walls_e.append(time.perf_counter() - t0)
        out_e.append(ladder.read(trace_outs))
        model.advance_positions_host()
        # after the replay the persistent inputs hold the NEXT step's values: this step's greedy ids at pos0 + k + 1
        _check_fields(model, layout, out_e[-1][0], [p + k + 1 for p in pos0], per_user=False, packed=False)
    _assert_identical("resident traced vs direct eager chain", out_e, chain_ref, chain_steps)
    # the live mask: a parked slot keeps its position (and its fed-back id), the others advance
    live = [u % 2 == 0 for u in range(U)]
    pos_k = [p + N_STEPS for p in pos0]
    model.load_decode_inputs(out_e[-1][0], pos_k, live=live)
    model.replay(trace_id)
    model.advance_positions_host()
    ids_k = ladder.read(trace_outs)[0]
    _check_fields(
        model, layout, ids_k, [p + (1 if v else 0) for p, v in zip(pos_k, live)], per_user=False, packed=False
    )
    assert model.positions.host == [p + (1 if v else 0) for p, v in zip(pos_k, live)]
    ttnn.release_trace(mesh_device, trace_id)
    monkeypatch.delenv(RESIDENT_ENV, raising=False)
    logger.info(
        f"PERF pass E (resident, traced): {_tok_s(walls_e, U)} (replay only, no per-step upload; the chain-start load "
        f"{load_e * 1e3:.2f} ms); bit-identical to the direct eager chain over {N_STEPS} steps; the live mask holds"
    )
    # ---- pass F: the decode slot mask (GLM53_DECODE_SLOT_MASK=1 over the resident chain: the live mask is the
    # kda_decode program's per-user write gate, doc/mesh/resident_decode.md step 4). Slot 0's prompt is prefilled in
    # two chunks with K masked decode steps of the other slots between them (slot 0 parked at the chunk end with a
    # pad id, live 0): its KDA carry is byte-identical across the parked steps (the gate), and its decode chain
    # after the second chunk equals the uninterleaved reference (the DSA row / pool the parked steps wrote at the
    # chunk end are rewritten by the second chunk, which starts there). Not wired for the user partition.
    if partition == "0":
        monkeypatch.setenv(RESIDENT_ENV, "1")
        monkeypatch.setenv(SLOT_MASK_ENV, "1")
        model.resident = None  # re-allocated with the gate page by the warm-up's staging (before the capture)
        model.warmup_decode(ids0, pos0)  # the gated twin program compiles here
        assert model.supports_slot_mask and model.positions.write_gate is not None
        trace_id, trace_outs = model.capture(ids0, pos0)
        n0, c0 = lens[0], PREFILL_CHUNK
        assert c0 < n0 and n0 % 4 == 0, (c0, n0)
        prompt0 = _prompt(n0)
        flat0 = [r * B for r in range(model.mesh.rows)]  # slot index 0 of every row (the rows hold the same prompt)
        K = 4

        def chunk(start: int, end: int):
            streams, logits, _ = model.prefill(prompt0[start:end], user=0, start=start)
            ttnn.deallocate(streams)
            if logits is not None:
                ttnn.deallocate(logits)

        def carry0() -> list[tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for dec in model.layers:
                if not dec.is_kda:
                    continue
                rec = dec._user_recurrent(dec.kda_batched.recurrent, 0)
                conv = dec._require_weights().attn.user_conv_state(dec.kda_batched, 0)
                out.append((ttnn.to_torch(rec).clone(), ttnn.to_torch(conv).clone()))
                ttnn.deallocate(rec)
                ttnn.deallocate(conv)
            return out

        def chain(n_steps: int):
            model.load_decode_inputs(ids0, pos0, live=[True] * U)
            out = []
            for _ in range(n_steps):
                model.replay(trace_id)
                out.append(ladder.read(trace_outs))
                model.advance_positions_host()
            return out

        # the interleaved run: chunk 1, K parked steps, chunk 2, the chain
        model.reset_kda_states()
        chunk(0, c0)
        before = carry0()
        live = [u % B != 0 for u in range(U)]
        ids_f = ids0.clone()
        ids_f[flat0] = 0
        pos_f = [c0 if u % B == 0 else pos0[u] for u in range(U)]
        model.load_decode_inputs(ids_f, pos_f, live=live)
        for _ in range(K):
            model.replay(trace_id)
            model.advance_positions_host()
        ttnn.synchronize_device(model.device)
        assert model.positions.host[flat0[0]] == c0, model.positions.host
        after = carry0()
        for layer_k, ((ra, ca), (rb, cb)) in enumerate(zip(before, after)):
            assert torch.equal(ra, rb) and torch.equal(ca, cb), (
                f"pass F: slot 0's KDA carry of KDA layer #{layer_k} changed across {K} parked steps: recurrent "
                f"{int((ra != rb).sum())} values, conv {int((ca != cb).sum())} values"
            )
        chunk(c0, n0)
        out_f = chain(N_STEPS)
        # the reference: the same two chunks with nothing between them, the same chain
        model.reset_kda_states()
        chunk(0, c0)
        chunk(c0, n0)
        out_r = chain(N_STEPS)
        for k, ((i_f, l_f), (i_r, l_r)) in enumerate(zip(out_f, out_r)):
            assert torch.equal(
                i_f[flat0], i_r[flat0]
            ), f"pass F: slot 0's greedy ids differ at step {k}: {i_f[flat0].tolist()} vs {i_r[flat0].tolist()}"
            assert torch.equal(l_f[flat0], l_r[flat0]), (
                f"pass F: slot 0's logits differ at step {k}: {int((l_f[flat0] != l_r[flat0]).sum())} values, "
                f"max abs {(l_f[flat0] - l_r[flat0]).abs().max().item():.3g}"
            )
        ttnn.release_trace(mesh_device, trace_id)
        monkeypatch.delenv(SLOT_MASK_ENV, raising=False)
        monkeypatch.delenv(RESIDENT_ENV, raising=False)
        logger.info(
            f"pass F (slot mask): slot 0's KDA carry byte-identical across {K} parked steps in {len(before)} KDA "
            f"layers; its chain after the second chunk bit-identical to the uninterleaved reference over {N_STEPS} steps"
        )
    rows_agree = sum(torch.equal(i[: U // 2], i[U // 2 :]) for i, _ in out_a) if model.mesh.rows == 2 else None
    logger.info(
        f"PERF mesh {MESH} layers {len(layers)} users {U} partition {partition}: packed traced {_tok_s(walls_c, U)}; "
        f"direct traced {_tok_s(walls_d, U)}; resident traced {_tok_s(walls_e, U)}; all five passes bit-identical "
        f"over {N_STEPS} steps"
        + (f"; rows agree on the greedy ids in {rows_agree}/{N_STEPS} steps (info)" if rows_agree is not None else "")
    )
