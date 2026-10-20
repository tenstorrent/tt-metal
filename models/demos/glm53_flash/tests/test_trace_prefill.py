# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced fixed-length prefill (tt/trace_prefill.py) vs the eager prefill of the resident mesh model on a small
mesh with a few layers (``GLM53_MODEL_LAYERS=0-5``: KDA + dense MLP 0-2, DSA + MoE 3, KDA + MoE 4-5).

``test_slot_write_ops`` (seconds, no weights): the device-side slot-write primitives on small tensors (the fp32
recurrent-state view + ``paged_fill_cache(batch_idx_tensor)``, the conv-history mask blend, the ring block fill, the
one-hot HiFi4 row selection), all exact.

``test_traced_vs_eager``: for every captured length (128 / 256 / 512 with ``max_seq_len`` 512 on one row) a prompt
shorter than the length goes through the eager prefill and through the trace, into the same user slot: same
first token, last-token logits PCC >= ``GLM53_TRACE_PCC`` (0.999), output stream rows [0, S) PCC, and the device
state the slot holds afterwards (KDA recurrent + conv carries, latent pages, complete pooled keys, the ring rows
the prompt defines) vs the eager prefill's. Then replay-twice (a second prompt through the same trace, another
slot, gives ITS eager result, not the previous prompt's), decode continuity (2 eager decode steps after the eager
vs the traced prefill give the same greedy ids) and TTFT eager vs traced (3 iterations each).

Order of the device work: every eager pass runs BEFORE the captures (its results are kept on the host), the traced
replays after, because a persistent tensor created lazily by the eager path after a capture (e.g. a new
``TtKDA.scalar`` value) could alias that trace's transient buffers.

    TT_VISIBLE_DEVICES=3 GLM53_MESH_SHAPE=1,1 GLM53_MODEL_LAYERS=0-5 \\
      pytest models/demos/glm53_flash/tests/test_trace_prefill.py -s --timeout 1500

Knobs: GLM53_MODEL_B (users per row, 2), GLM53_TRACE_MAX_SEQ (512), GLM53_MODEL_EXPERT_DTYPE /
GLM53_MODEL_EXPERT_DOWN_DTYPE (default bfp4 here: three MoE layers on one chip), GLM53_MESH_CACHE,
GLM53_MODEL_TRACE_MB (256), GLM53_TRACE_PCC (0.999), GLM53_TRACE_STATE_PCC (0.999).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

DUMP = Path("bringup/references/glm53_flash/L45_S128")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "2"))  # users per row
MAX_SEQ = int(os.environ.get("GLM53_TRACE_MAX_SEQ", "512"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "256"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp4")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp4")]
PCC_BAR = float(os.environ.get("GLM53_TRACE_PCC", "0.999"))
STATE_PCC_BAR = float(os.environ.get("GLM53_TRACE_STATE_PCC", "0.999"))
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
N_TTFT = 3
S_STABLE = 90  # the replay-twice prompt (same trace as S=100, another slot)
S_DECODE = 102  # the decode-continuity prompt (S % 4 == 2: the ring holds two rows of the open pool)


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-5")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


# ------------------------------------------------------------------------------------------ readbacks
def _shard0(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


# ------------------------------------------------------------------------------- slot-write primitives
@pytest.mark.timeout(300)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_slot_write_ops(mesh_device):
    """The device-side slot writes of the traced prefill on small tensors (seconds, no weights): the fp32 recurrent
    state through ``ttnn.experimental.view`` + ``paged_fill_cache(batch_idx_tensor)`` (whole-tile copy into one
    slot, the other slots untouched), the tap-major bf16 conv history through the one-hot row x column broadcast
    masks (exact), the ring block fill, and the one-hot HiFi4 row selection (exact)."""
    torch.manual_seed(0)
    dev = mesh_device
    Bu, H, K, V, taps, W = 2, 4, 128, 128, 3, 256
    up = lambda t, dtype, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev)
    # recurrent: slot 1 <- new, slot 0 unchanged
    rec_h = torch.randn(Bu, H, K, V)
    new_h = torch.randn(1, H, K, V)
    rec, new = up(rec_h, ttnn.float32), up(new_h, ttnn.float32)
    nblk = H * K // 32
    pt = up(torch.arange(Bu * nblk, dtype=torch.int32).reshape(Bu, nblk), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    idx = up(torch.tensor([1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    rec_v = ttnn.experimental.view(rec, ttnn.Shape([Bu * nblk, 1, 32, V]))
    new_v = ttnn.experimental.view(new, ttnn.Shape([1, 1, H * K, V]))
    ttnn.experimental.paged_fill_cache(rec_v, new_v, pt, batch_idx_tensor=idx)
    got = ttnn.to_torch(rec).float()
    assert torch.equal(got[0], rec_h[0]), "slot 0 of the recurrent state was touched"
    assert torch.equal(got[1], new_h[0]), f"slot 1 != new (max diff {(got[1] - new_h[0]).abs().max()})"
    # conv history [taps, B, W] bf16 tiles: dst * keep + new * mask
    conv_h = torch.randn(taps, Bu, W).bfloat16()
    newc_h = torch.randn(taps, 1, W).bfloat16()
    mask = torch.zeros(taps, Bu, 1)
    mask[:, 1, 0] = 1.0
    conv, newc = up(conv_h, ttnn.bfloat16), up(newc_h, ttnn.bfloat16)
    m, keep = up(mask, ttnn.bfloat16), up(1.0 - mask, ttnn.bfloat16)
    ttnn.multiply(conv, keep, output_tensor=conv)
    rows = ttnn.multiply(newc, m)
    assert tuple(rows.shape) == (taps, Bu, W), rows.shape
    ttnn.add(conv, rows, output_tensor=conv)
    got = ttnn.to_torch(conv)
    assert torch.equal(got[:, 0], conv_h[:, 0]), "row 0 of the conv history was touched"
    assert torch.equal(got[:, 1], newc_h[:, 0]), "row 1 of the conv history != new"
    # ring block fill [B,1,32,D] <- [1,1,32,D] into slot 1
    D = 128
    ring_h, tile_h = torch.randn(Bu, 1, 32, D).bfloat16(), torch.randn(1, 1, 32, D).bfloat16()
    ring, tile = up(ring_h, ttnn.bfloat16), up(tile_h, ttnn.bfloat16)
    rpt = up(torch.arange(Bu, dtype=torch.int32).reshape(Bu, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.paged_fill_cache(ring, tile, rpt, batch_idx_tensor=idx)
    got = ttnn.to_torch(ring)
    assert torch.equal(got[0], ring_h[0]) and torch.equal(got[1], tile_h[0]), "ring block fill"
    # one-hot HiFi4 selection of rows 96..99 (and row 35 for the head) out of 128 bf16 rows: exact
    T = 128
    x_h = torch.randn(1, 1, T, D).bfloat16()
    sel_h = torch.zeros(1, 1, 32, T)
    for j, p in enumerate((96, 97, 98, 99, 35)):
        sel_h[0, 0, j, p] = 1.0
    hifi4 = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out = ttnn.matmul(
        up(sel_h, ttnn.bfloat16), up(x_h, ttnn.bfloat16), dtype=ttnn.bfloat16, compute_kernel_config=hifi4
    )
    got = ttnn.to_torch(out)[0, 0]
    assert torch.equal(got[:4], x_h[0, 0, 96:100]) and torch.equal(got[4], x_h[0, 0, 35]), "one-hot selection not exact"
    assert torch.equal(got[5:], torch.zeros_like(got[5:])), "zero selector rows are not zero"
    logger.info("slot-write primitives: recurrent paged fill, conv mask blend, ring fill, one-hot selection all exact")


def _owner_logits(model: GLM53MeshModel, logits: ttnn.Tensor, owner_row: int) -> torch.Tensor:
    """Row 0 of the owner row's logits (the eager single row / the traced selected row) -> [V] fp32 on the host."""
    shard = model.mesh.device_tensors(logits)[owner_row * model.mesh.cols]
    return ttnn.to_torch(shard).float()[0, 0, 0].clone()


def _stream_rows(model: GLM53MeshModel, streams: ttnn.Tensor, S: int) -> torch.Tensor:
    return model.mesh.to_host(streams, rows=2 if model.mesh.rows > 1 else None).float()[0, 0, :S].clone()


def _slot_state(model: GLM53MeshModel, user: int, S: int) -> dict[str, torch.Tensor]:
    """Host copies of what slot ``user`` holds after a prefill of ``S`` tokens (chip 0): per KDA layer the
    recurrent [H,K,V] and conv [taps,W] carries, per DSA layer the latent rows [0,S), the complete pooled keys
    [0, S//4) and the ring rows this prompt wrote (slots [0, S % 4))."""
    out = {}
    for dec in model.layers:
        li = dec.layer_idx
        if dec.is_kda:
            out[f"L{li}.recurrent"] = _shard0(dec.kda_batched.recurrent).float()[user].clone()
            conv = _shard0(dec.kda_batched.convolution).float()
            out[f"L{li}.conv"] = (conv[:, user] if dec.layer.attn.conv_tiled else conv[user]).clone()
        else:
            cache = dec.cache
            slot = cache.prefill_slot(user)
            nblk = -(-S // cache.block_size)
            ids = cache.page_table_host[slot, :nblk].tolist()
            lat = _shard0(cache.cache).float()  # [max_blocks, 1, 32, L]
            out[f"L{li}.latent"] = lat[ids].reshape(nblk * cache.block_size, -1)[:S].clone()
            idx = cache.index
            pools = _shard0(idx.pool_cache).float()[slot * idx.pool_blocks : (slot + 1) * idx.pool_blocks]
            out[f"L{li}.pools"] = pools.reshape(-1, idx.D)[: S // idx.P].clone()
            tail = S % idx.P
            if tail:
                out[f"L{li}.ring_k"] = _shard0(idx.ring_k).float()[slot, 0, :tail].clone()
                out[f"L{li}.ring_g"] = _shard0(idx.ring_g).float()[slot, 0, :tail].clone()
    return out


def _compare_states(ref: dict, out: dict, tag: str) -> float:
    worst = 1.0
    for k, a in ref.items():
        b = out[k]
        p = pcc(a, b) if a.numel() > 1 and float(a.std()) > 0 else float(torch.equal(a, b))
        mad = float((a - b).abs().max())
        logger.info(f"  {tag} {k}: pcc {p:.6f} max|diff| {mad:.3e} (|ref| max {float(a.abs().max()):.3e})")
        worst = min(worst, p)
    return worst


def _decode_ids(model: GLM53MeshModel, first: int, S: int, n: int) -> list[int]:
    """``n`` eager decode steps of user 0 from its first token at position ``S`` (the other slots idle at 1)."""
    U = model.total_users
    out, tok, pos = [], first, [S] + [1] * (U - 1)
    for _ in range(n):
        model.load_ids(torch.tensor([tok] + [0] * (U - 1), dtype=torch.long))
        model.set_positions(pos)
        streams, logits, ids = model.decode_step(model.ids_dev, model.positions)
        tok = int(model.ids_to_host(ids)[0])
        out.append(tok)
        pos = [p + 1 for p in pos]
        for t in (streams, logits, ids):
            if t is not None:
                ttnn.deallocate(t)
    return out


# ----------------------------------------------------------------------------------------------- test
@pytest.mark.timeout(1500)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_traced_vs_eager(mesh_device):
    layers = _layer_range()
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=B,
        max_seq_len=MAX_SEQ,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    U = model.total_users
    logger.info(
        f"model: mesh {MESH} layers {layers} B/row {B} max_seq_len {MAX_SEQ} built in {time.perf_counter() - t0:.0f} s"
    )
    base = torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1).long()  # 128 real tokens
    pool = torch.cat([base, base.flip(0), base])  # 384 tokens for the longer lengths

    def prompt(S: int, offset: int = 0) -> torch.Tensor:
        return pool[offset : offset + S].clone()

    lengths = model.trace_prefill_lengths()
    cases = [(S, u) for S, u in ((100, 0), (128, 1), (200, 0), (384, 1)) if any(T >= S for T in lengths)]
    logger.info(f"trace lengths {lengths}; cases (S, user) {cases}")

    # ---- phase 1 (eager, before any capture): the persistent decode inputs + the eager decode compile, then every
    # eager prefill of the test with its results kept on the host
    model.warmup_decode(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()
    eager = {}
    for S, u in cases:
        streams, logits, owner = model.prefill(prompt(S), user=u)
        tok = model.first_token(logits, owner)
        eager[(S, u)] = (
            tok,
            owner,
            _owner_logits(model, logits, owner),
            _stream_rows(model, streams, S),
            _slot_state(model, u, S),
        )
        ttnn.deallocate(streams)
        ttnn.deallocate(logits)
    streams, logits, owner = model.prefill(prompt(S_STABLE, 50), user=1)  # the replay-twice prompt
    eager_stable = (model.first_token(logits, owner), _owner_logits(model, logits, owner))
    ttnn.deallocate(streams)
    ttnn.deallocate(logits)
    streams, logits, owner = model.prefill(prompt(S_DECODE, 20), user=0)  # the decode-continuity prompt
    tok_dec = model.first_token(logits, owner)
    ttnn.deallocate(streams)
    ttnn.deallocate(logits)
    eager_decode = [tok_dec] + _decode_ids(model, tok_dec, S_DECODE, 2)
    eager_walls = []
    for _ in range(N_TTFT):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        streams, logits, owner = model.prefill(prompt(100), user=0)
        model.first_token(logits, owner)
        eager_walls.append(time.perf_counter() - t0)
        ttnn.deallocate(streams)
        ttnn.deallocate(logits)

    # ---- phase 2: the traces (every persistent input allocated, every length compiled, then captured)
    t0 = time.perf_counter()
    captured = model.capture_prefill()
    logger.info(f"captured prefill traces {captured} in {time.perf_counter() - t0:.0f} s")
    assert sorted(captured) == lengths

    # ---- phase 3: traced replays vs the eager results
    worst_pcc, worst_state, mismatches = 1.0, 1.0, []
    for S, u in cases:
        tok_e, owner_e, le, rows_e, state_e = eager[(S, u)]
        T = min(T for T in model.prefill_traces if T >= S)
        streams, logits, ids, owner_t = model.prefill_traced(prompt(S), user=u)
        tok_t = model.first_token(ids, owner_t)
        lt = _owner_logits(model, logits, owner_t)
        rows_t = _stream_rows(model, streams, S)
        state_t = _slot_state(model, u, S)
        assert owner_t == owner_e, (owner_t, owner_e)
        assert int(lt.argmax()) == tok_t, "the traced greedy id is not the argmax of its own selected row"
        p_log, p_rows = pcc(le, lt), pcc(rows_e, rows_t)
        logger.info(
            f"S={S} user {u} (trace T={T}): first token eager {tok_e} traced {tok_t} "
            f"({'match' if tok_e == tok_t else 'DIFFERENT'}); last-token logits pcc {p_log:.6f} "
            f"max|diff| {float((le - lt).abs().max()):.3e}; stream rows [0,{S}) pcc {p_rows:.6f} "
            f"max|diff| {float((rows_e - rows_t).abs().max()):.3e}"
        )
        worst_state = min(worst_state, _compare_states(state_e, state_t, f"S={S} u{u}"))
        worst_pcc = min(worst_pcc, p_log, p_rows)
        if tok_e != tok_t:
            mismatches.append((S, u, tok_e, tok_t))

    # replay-twice stability: prompt A then prompt B through the same trace (B into another slot) -> B's result
    _, logits, ids, owner = model.prefill_traced(prompt(100), user=0)
    la = _owner_logits(model, logits, owner)
    _, logits, ids, owner = model.prefill_traced(prompt(S_STABLE, 50), user=1)
    tok_b, lb = model.first_token(ids, owner), _owner_logits(model, logits, owner)
    p_stable = pcc(eager_stable[1], lb)
    logger.info(
        f"replay-twice: prompt B (S={S_STABLE}, slot 1) after prompt A (S=100, slot 0): first token traced {tok_b} "
        f"eager {eager_stable[0]} ({'match' if tok_b == eager_stable[0] else 'DIFFERENT'}); logits pcc vs eager B "
        f"{p_stable:.6f}, vs A's replay {pcc(la, lb):.4f}"
    )
    assert not torch.equal(la, lb), "the second replay returned the first prompt's logits"

    # decode continuity: the traced prefill's state (KDA carries, cache pages, index) feeds the eager decode
    _, logits, ids, owner = model.prefill_traced(prompt(S_DECODE, 20), user=0)
    tok_dec_t = model.first_token(ids, owner)
    traced_decode = [tok_dec_t] + _decode_ids(model, tok_dec_t, S_DECODE, 2)
    logger.info(f"decode continuity (S={S_DECODE}): first + 2 decode ids eager {eager_decode} traced {traced_decode}")

    traced_walls = []
    for _ in range(N_TTFT):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        _, logits, ids, owner = model.prefill_traced(prompt(100), user=0)
        model.first_token(ids, owner)
        traced_walls.append(time.perf_counter() - t0)
    logger.info(
        f"PERF-TTFT mesh {MESH} layers {len(layers)} S=100: eager min {min(eager_walls) * 1e3:.1f} ms "
        f"(mean {sum(eager_walls) / N_TTFT * 1e3:.1f}), traced min {min(traced_walls) * 1e3:.1f} ms "
        f"(mean {sum(traced_walls) / N_TTFT * 1e3:.1f}) -> x{min(eager_walls) / min(traced_walls):.1f}"
    )
    model.release_prefill_traces()

    assert not mismatches, f"first token differs (S, user, eager, traced): {mismatches}"
    assert worst_pcc >= PCC_BAR, f"traced vs eager pcc {worst_pcc} below {PCC_BAR}"
    assert worst_state >= STATE_PCC_BAR, f"slot state pcc {worst_state} below {STATE_PCC_BAR}"
    assert tok_b == eager_stable[0] and p_stable >= PCC_BAR, "replay-twice: the second prompt's result is off"
    assert traced_decode == eager_decode, "decode after the traced prefill diverges from decode after the eager one"
