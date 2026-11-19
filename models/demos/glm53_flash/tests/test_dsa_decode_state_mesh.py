# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B of ``GLM53_DSA_DECODE_STATE`` (decode ledger item B5): the traced multi-user sparse decode with the fused
``dsa_decode_state`` block + the absorb folds ("1") against today's composite ops ("0"), on real layer-3 weights and
the R4097 attention inputs, users beyond position 2047 (sparse: the selected pool ids matter) and, on the block, a
dense user as well. Both paths run in lockstep on their own caches from the same prefill; per step:

* attention / layer output PCC between the paths (>= 0.9998 with the folds, 0.9999 with ``GLM53_DSA_FOLD=none``:
  the bars of ``test_dsa.py::test_dsa_fused_vs_composite``);
* the selected pool ids of every sparse user (the persistent ``topk_rows`` output the gather read) identical;
* the index rings (k and gate rows) and the pooled-key cache bit-identical, the latent cache bit-identical at the
  end (``GLM53_DSS_STRICT=0`` reports instead of asserting: the first failure candidate of the ledger item);
* the state path captured in a trace and replayed (the composite stays eager) for the last steps.

``test_decode_state_vs_composite_chip0``: the attention alone (``TtDSA`` + ``TtLatentCache``) on one chip, B = 4 /
32 users at the same position. ``test_decode_state_partitioned_block``: the full layer (``FunctionalDecoder``) on a
(2, 4) block with the user-partitioned heads2users cache (4 users per row = one per chip: three sparse prefixes and
one dense), both rows fed the same prompts (rows must agree byte for byte within a path: the batch-independence
check of ``test_mesh_batch_independence.py``, which also runs as is under ``GLM53_DSA_DECODE_STATE=1``).
``GLM53_DSS_MODES=0,latent`` A/Bs the fallback mode instead (index state identical by construction).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_dsa_decode_state_mesh.py -s -k chip0
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \\
    MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_dsa_decode_state_mesh.py -s -k block
End to end (first token / greedy decode with the knob): ``GLM53_DSA_DECODE_STATE=1`` in front of the
``tests/test_mesh_model.py -k chain`` and ``tests/test_mesh_batch_independence.py`` commands of doc/mesh/README.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import attention_input, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import SparseDecodeMeta, TtDSA, TtLatentCache, decode_position_tensors, dsa_folds
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder, host_streams_to_device
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

LAYER = 3
R4097 = Path("bringup/references/glm53_flash/R4097")
ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
MODES = tuple(os.environ.get("GLM53_DSS_MODES", "0,1").split(","))  # the A/B pair ("0,latent" = the fallback)
STRICT = os.environ.get("GLM53_DSS_STRICT", "1") == "1"  # bit-identity of rings / pools / latent asserted
N_EAGER, N_TRACE = int(os.environ.get("GLM53_DSS_EAGER", "6")), int(os.environ.get("GLM53_DSS_TRACE", "4"))
PCC_BAR = 0.9999 if dsa_folds(strict=False) == (False, False) else 0.9998
SPARSE_FROM = 2047  # a user at position a >= 2047 ranks real pools (n = (a+1)//4 >= 512); below, the top-k is ignored

needs_dump = pytest.mark.skipif(not (R4097 / f"layer{LAYER}_prefill.pt").exists(), reason=f"no {R4097} dump")


def _load():
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    x_in = torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"]  # [1, 4097, 4, 4096] mHC streams
    return raw, cfg, x_in


def _equal_report(name: str, a: torch.Tensor, b: torch.Tensor, failures: list[str]):
    """Bit-identity of two bf16 device tensors (as fp32), asserted or reported (``STRICT``)."""
    if torch.equal(a, b):
        return True
    d = (a - b).abs()
    ia, ib = a.to(torch.bfloat16).view(torch.int16).int(), b.to(torch.bfloat16).view(torch.int16).int()
    msg = (
        f"{name}: {int((d > 0).sum())} of {d.numel()} values differ, max abs {d.max().item():.3g}, "
        f"max {int((ia - ib).abs().max())} bf16 ulps"
    )
    if a.dim() == 4 and a.shape[-2] == 32:  # [blocks, 1, 32, D] caches: which rows (pool / ring ids) differ
        rows = d.amax(dim=-1).reshape(a.shape[0], 32)  # max abs per (block, row)
        big = (rows > 1e-2).nonzero().tolist()
        blocks = sorted({b for b, _ in big})
        detail = ", ".join(f"b{b}r{r}={rows[b, r].item():.3g}" for b, r in big[:12])
        msg += (
            f"; rows with |d| > 1e-2: {len(big)} in blocks {blocks[:8]}"
            f"{' (first 12: ' + detail + ')' if detail else ''}; rows with 1-ulp noise: {int(((rows > 0) & (rows <= 1e-2)).sum())}"
        )
    logger.warning(msg)
    failures.append(msg)
    return False


def _ids_report(name: str, top_a: torch.Tensor, top_b: torch.Tensor, users: list[int], failures: list[str]):
    """``top_*`` [B,1,1,K] uint32: the ranked pool ids of every user; the SETS of the listed users must agree."""
    for u in users:
        sa, sb = set(top_a[u, 0, 0].tolist()), set(top_b[u, 0, 0].tolist())
        if sa != sb:
            msg = f"{name}: user {u}: {len(sa ^ sb)} of {len(sa)} selected pool ids differ"
            logger.warning(msg)
            failures.append(msg)


# --------------------------------------------------------------------------------------- one chip, attention
@needs_dump
@pytest.mark.timeout(1500)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 64 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_decode_state_vs_composite_chip0(mesh_device, B):
    raw, cfg, x_in = _load()
    attn = {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")}
    rows = attention_input(raw, cfg, x_in)  # [4097, 4096] bf16: the attention inputs of the real prompt
    P, K = cfg.index_kpool, cfg.index_select_pools
    S0 = 2112  # every user beyond 2047: the selected pools are real
    n_steps = N_EAGER + 1 + N_TRACE
    assert S0 + n_steps + 7 * B <= rows.shape[0]
    paths = {m: TtDSA(mesh_device, cfg, attn, layer_idx=LAYER, decode_state=m) for m in MODES}
    caches = {m: TtLatentCache(mesh_device, cfg, max_batch=B, max_seq_len=S0 + 64, block_size=32) for m in MODES}
    x_pre = ttnn.from_torch(
        rows[:S0].reshape(1, 1, S0, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    for m in MODES:
        for u in range(B):
            paths[m].prefill_forward(x_pre, caches[m], user=u, seq_len=S0)
    ttnn.deallocate(x_pre)
    # persistent decode inputs (refreshed in place: the trace replays read them)
    x_dev = ttnn.from_torch(
        torch.zeros(1, 1, B, cfg.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    pos = [S0] * B
    pos_t, pool_t, ring_t = decode_position_tensors(mesh_device, pos, P)
    meta = SparseDecodeMeta.from_positions(mesh_device, pos, cfg)

    def load(step: int):
        nonlocal pos
        pos = [S0 + step] * B
        xt = torch.stack([rows[S0 + step + 7 * u] for u in range(B)]).reshape(1, 1, B, -1)  # distinct tokens per user
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), x_dev)
        for t, vals in ((pos_t, pos), (pool_t, [p // P for p in pos]), (ring_t, [p % P for p in pos])):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                t,
            )
        meta.update(pos, cfg)

    def run(m: str) -> ttnn.Tensor:
        return paths[m].decode_forward(x_dev, caches[m], pos_t, pool_t, ring_t, host_positions=pos, sparse_meta=meta)

    failures: list[str] = []
    worst = 1.0

    def compare(step: int, ys: dict[str, torch.Tensor], how: str):
        nonlocal worst
        a, b = MODES
        p = pcc(ys[a], ys[b])
        worst = min(worst, p)
        rk = {m: ttnn.to_torch(caches[m].index.rings()[0]).float() for m in MODES}
        rg = {m: ttnn.to_torch(caches[m].index.rings()[1]).float() for m in MODES}
        pools = {m: ttnn.to_torch(caches[m].index.pool_cache).float() for m in MODES}
        tops = {m: ttnn.to_torch(paths[m]._top_buf[B]) for m in MODES}
        _ids_report(f"step {step} ({how})", tops[a], tops[b], list(range(B)), failures)
        same = all(
            _equal_report(f"step {step} ({how}) {n}", x[a], x[b], failures)
            for n, x in (("ring_k", rk), ("ring_g", rg), ("pools", pools))
        )
        logger.info(
            f"  step {step} ({how}, pos {pos[0]}): output pcc {a} vs {b} {p:.6f}; index state {'identical' if same else 'DIFFERS'}"
        )

    for step in range(N_EAGER):
        load(step)
        ys = {}
        for m in MODES:
            y = run(m)
            ys[m] = ttnn.to_torch(y).float()[0, 0]
            ttnn.deallocate(y)
        compare(step, ys, "eager")
    # the state path captured in a trace (warm-up + capture consume the same token twice: the program's writes at
    # one position are idempotent), the composite path stays eager
    load(N_EAGER)
    traced = MODES[1]
    y0 = run(MODES[0])
    ys = {MODES[0]: ttnn.to_torch(y0).float()[0, 0]}
    ttnn.deallocate(y0)
    warm = run(traced)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = run(traced)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ys[traced] = ttnn.to_torch(y_tr).float()[0, 0]
    assert pcc(ttnn.to_torch(warm).float()[0, 0], ys[traced]) > 0.99999, "capture run differs from its warm-up"
    ttnn.deallocate(warm)
    compare(N_EAGER, ys, "capture")
    for step in range(N_EAGER + 1, n_steps):
        load(step)
        y0 = run(MODES[0])
        ys = {MODES[0]: ttnn.to_torch(y0).float()[0, 0]}
        ttnn.deallocate(y0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ys[traced] = ttnn.to_torch(y_tr).float()[0, 0]
        compare(step, ys, "trace replay")
    ttnn.release_trace(mesh_device, tid)
    lat = {m: ttnn.to_torch(caches[m].cache).float() for m in MODES}
    _equal_report("latent cache (end)", lat[MODES[0]], lat[MODES[1]], failures)
    logger.info(
        f"chip 0, B={B}, modes {MODES}: worst output pcc {worst:.6f} over {n_steps} steps; {len(failures)} mismatches"
    )
    assert worst > PCC_BAR, f"output pcc {worst} below {PCC_BAR}"
    if STRICT:
        assert not failures, "\n".join(failures[:20])


# ----------------------------------------------------------------------------- (2,4) block, partitioned cache
@needs_dump
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(ROWS, COLS, trace_region_size=64 * 1024 * 1024)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_decode_state_partitioned_block(mesh_device, monkeypatch):
    raw, cfg, x_in = _load()
    mesh = GLMMesh.from_device(mesh_device)
    B = COLS  # one user per chip (users_per_chip = 1): three sparse prefixes and one dense
    S = [2112, 2080, 2049, 600][:B] + [2112] * max(0, B - 4)
    U = ROWS * B
    n_steps = N_EAGER + 1 + N_TRACE
    assert max(S) + n_steps <= x_in.shape[1]
    max_seq_len = -(-max(S) // (32 * ROWS)) * (32 * ROWS) + 64
    decs = {}
    for m in MODES:
        monkeypatch.setenv("GLM53_DSA_DECODE_STATE", m)  # read at TtDSA construction
        decs[m] = FunctionalDecoder.from_state_dict(
            raw,
            hf_config=cfg,
            layer_idx=LAYER,
            mesh_device=mesh_device,
            max_batch=B,
            max_seq_len=max_seq_len,
            partition_users=True,
            prefill_partition="heads2users",
        )
        assert decs[m].cache.partition and decs[m].cache.users_per_chip == 1
        assert decs[m].layer.attn.decode_state == m
        for u in range(B):
            y = decs[m].prefill_forward(
                host_streams_to_device(x_in[:, : S[u]], mesh_device, mesh=mesh), user=u, seq_len=S[u]
            )
            ttnn.deallocate(y)
    B_loc = decs[MODES[0]].cache.users_per_chip

    def packed(step: int) -> torch.Tensor:
        toks = [x_in[0, S[u % B] + step].reshape(1, -1).float() for u in range(U)]  # the prompt's next real token
        return torch.stack(toks, dim=0).reshape(1, 1, U, -1)

    positions = lambda step: [S[u % B] + step for u in range(U)]
    streams = {m: mesh.upload(packed(0), ttnn.float32, rows=2) for m in MODES}
    dps = {m: decs[m].make_decode_positions(positions(0)) for m in MODES}

    def load(step: int):
        for m in MODES:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    packed(step), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_rows(2)
                ),
                streams[m],
            )
            decs[m].update_decode_positions(dps[m], positions(step))

    failures: list[str] = []
    worst = 1.0

    def shards(t: ttnn.Tensor) -> list[torch.Tensor]:
        return [ttnn.to_torch(d).float() for d in mesh.device_tensors(t)]

    def compare(step: int, outs: dict[str, ttnn.Tensor], how: str):
        nonlocal worst
        a, b = MODES
        sh = {m: shards(outs[m]) for m in MODES}
        for m in MODES:  # columns replicated after the TP all-reduce; rows fed the same prompts agree byte for byte
            for r in range(ROWS):
                for c in range(1, COLS):
                    assert torch.equal(sh[m][r * COLS + c], sh[m][r * COLS]), (step, m, r, c, "columns differ")
                if r and not torch.equal(sh[m][r * COLS], sh[m][0]):
                    failures.append(f"step {step} ({how}) mode {m}: row {r} differs from row 0 (batch independence)")
        for r in range(ROWS):
            for u in range(B):
                p = pcc(sh[a][r * COLS][0, 0, u], sh[b][r * COLS][0, 0, u])
                worst = min(worst, p)
        pos = positions(step)
        tops = {m: [ttnn.to_torch(d) for d in mesh.device_tensors(decs[m].layer.attn._top_buf[B_loc])] for m in MODES}
        same = True
        for chip in range(ROWS * COLS):
            r, c = divmod(chip, COLS)
            sparse_local = [j for j in range(B_loc) if pos[r * B + c * B_loc + j] >= SPARSE_FROM]
            _ids_report(f"step {step} ({how}) chip {chip}", tops[a][chip], tops[b][chip], sparse_local, failures)
            ca, cb = decs[a].cache, decs[b].cache
            for name, ta, tb in (
                ("ring_k", ca.index.rings()[0], cb.index.rings()[0]),
                ("ring_g", ca.index.rings()[1], cb.index.rings()[1]),
                ("pools", ca.index.pool_cache, cb.index.pool_cache),
            ):
                va, vb = shards(ta)[chip], shards(tb)[chip]
                same &= _equal_report(f"step {step} ({how}) chip {chip} {name}", va, vb, failures)
        logger.info(
            f"  step {step} ({how}): worst (row, user) pcc so far {worst:.6f}; index state {'identical' if same else 'DIFFERS'}"
        )

    for step in range(N_EAGER):
        if step:
            load(step)
        outs = {m: decs[m].decode_forward(streams[m], positions=dps[m]) for m in MODES}
        compare(step, outs, "eager")
        for y in outs.values():
            ttnn.deallocate(y)
    load(N_EAGER)
    traced = MODES[1]
    y0 = decs[MODES[0]].decode_forward(streams[MODES[0]], positions=dps[MODES[0]])
    tid, y_tr, warm = decs[traced].capture_decode_trace(streams[traced], dps[traced])
    compare(N_EAGER, {MODES[0]: y0, traced: warm}, "capture warm-up")
    ttnn.deallocate(y0)
    ttnn.deallocate(warm)
    for step in range(N_EAGER + 1, n_steps):
        load(step)
        y0 = decs[MODES[0]].decode_forward(streams[MODES[0]], positions=dps[MODES[0]])
        decs[traced].replay_decode(tid)
        compare(step, {MODES[0]: y0, traced: y_tr}, "trace replay")
        ttnn.deallocate(y0)
    ttnn.release_trace(mesh_device, tid)
    for chip, (la, lb) in enumerate(zip(shards(decs[MODES[0]].cache.cache), shards(decs[MODES[1]].cache.cache))):
        _equal_report(f"latent cache (end) chip {chip}", la, lb, failures)
    logger.info(
        f"block ({ROWS},{COLS}) partitioned, modes {MODES}: worst (row, user) pcc {worst:.6f}; {len(failures)} mismatches"
    )
    assert worst > PCC_BAR, f"layer output pcc {worst} below {PCC_BAR}"
    if STRICT:
        assert not failures, "\n".join(failures[:20])
