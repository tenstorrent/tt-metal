# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Slot-level prefix caching on a (2, 4) block (``doc/serving/prefix_caching.md``, ``tt/generator_vllm.py``): the
served adapter prefills prompt A, decodes a response while another user decodes alongside (A's slot is retained
and parked at its end), then serves the multi-turn request A + response + new turn with ``num_computed_tokens``
= the content's last 32-boundary from the slot's KDA snapshot (the chunk ``[snapshot, end)`` only), and the
result is compared with a from-scratch prefill of the whole prompt in the SAME physical slot: the first token
and the first decode tokens agree (a differing token is accepted only at a near-tie of the reference logits: the
prefill chunk scan and the decode recurrence of the KDA layers are different op sequences, so the states agree
to bf16/fp32 rounding, not bit for bit) and the logits correlate above 0.999. The hit is served twice: the first
hit pays any first-time program compile of the tail chunk, the second (warm) hit is the steady-state cost; the
PERF line reports both. The miss path (a prompt that differs inside the cached prefix) prefills from 0 in another
slot and matches its own from-scratch reference. Both snapshot stores run (``device`` ring, ``host`` copies).
1024-token device chunks (512-row slabs: the L1 hazard of ``test_decode_packed_inputs_mesh.py``), no prefill
traces (the eager chunked path is the hit path). ``GLM53_VLLM_APC_ALLOC_CHECK=1`` is on: the adapter refuses the
first call site (snapshot save, hit restore, prefill row, release) after which the DRAM footprint differs from the
post-capture baseline (doc/serving/prefix_caching.md s.6), so a buffer allocated under the live decode trace --
the 2026-10-28 hang class -- fails the test by name instead of wedging the block later.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \\
        pytest models/demos/glm53_flash/tests/test_serving_prefix_cache_mesh.py -s

Expected: 2 passed (one per store), ~10 min with a warm weight cache. First failure candidates: (1) a
``paged_fill_cache`` / ``view`` shape refusal in ``DeviceKdaSnapshotRing.save`` (the ring view on the entry axis;
compare with ``FunctionalDecoder._write_user_kda_state_slot``), (2) the conv mask blend on the ``[taps, E, W]``
ring (``E`` beyond one tile row), (3) ``ttnn.slice`` of the ring entry (tile-aligned on dim 0: should hold),
(4) the parked decode: a retained slot at position 1540 runs the sparse DSA path on the block (positions >
2050 only: here dense), (5) the hit's logits PCC below 0.999 = the restored state is wrong (then check the host
store first: it isolates the paged fill from the copy semantics).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.vllm_config import APC_ALLOC_CHECK_ENV, ServingGeometry

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))  # users per row
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
MAX_SEQ_LEN = 4096
CHUNK = 1024  # two device chunks for prompt A on any row count (the snapshot at 1024 the test asserts)
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
LEN_A = 1500  # prompt A: two device chunks (a prefill snapshot at 1024)
N_RESP = 40  # A's response: crosses 1504 and 1536 (the rolling decode snapshot ends at 1536)
LEN_TURN = 50  # the new turn
N_CHECK = 5  # decode steps compared after the prefill
PCC_MIN = 0.999
NEAR_TIE = 0.05  # a differing greedy token is accepted when the reference's top-2 logit gap is below this


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-3")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _ids(n: int, seed: int, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, min(vocab, 100000), (n,), generator=g).to(torch.int32)


def _greedy(n):
    from types import SimpleNamespace

    return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


class _Served:
    """The adapter driven like the plugin: host logits out (``sampling_params=None``) so every step yields logits."""

    def __init__(self, adapter: GLM53FlashForCausalLM):
        self.a = adapter
        self.n = adapter.geometry.request_capacity

    def prefill(self, logical: int, ids: torch.Tensor, start: int) -> torch.Tensor:
        """``[vocab]`` fp32 logits of the prompt's last token; the chunk ``[start, len)`` is what vLLM scheduled."""
        out = self.a.prefill_forward(
            ids.reshape(1, -1),
            prompt_lens=[int(ids.numel())],
            start_pos=torch.tensor([start]),
            empty_slots=[logical],
            enable_trace=False,
        )
        return out[0, 0].clone()

    def decode(self, feed: dict[int, tuple[int, int]]) -> dict[int, torch.Tensor]:
        """``{logical: (token, position)}`` -> ``{logical: logits [vocab]}``."""
        tokens = torch.zeros(self.n, 1, dtype=torch.int32)
        positions = torch.full((self.n,), -1, dtype=torch.long)
        for logical, (tok, pos) in feed.items():
            tokens[logical, 0] = tok
            positions[logical] = pos
        logits, _ = self.a.decode_forward(
            tokens,
            positions,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        return {logical: logits[logical, 0].clone() for logical in feed}

    def generate(self, logical: int, first_logits: torch.Tensor, position: int, n: int, side: dict | None = None):
        """``n`` greedy decode steps from ``first_logits`` at ``position``: (tokens fed, logits per step)."""
        toks, outs = [], []
        tok = int(first_logits.argmax())
        for k in range(n):
            feed = {logical: (tok, position + k)}
            if side:
                feed.update(side(k))
            res = self.decode(feed)
            toks.append(tok)
            outs.append(res[logical])
            tok = int(res[logical].argmax())
        return toks, outs


def _assert_agree(name: str, got: list[torch.Tensor], ref: list[torch.Tensor], pcc_min: float = PCC_MIN):
    for k, (g, r) in enumerate(zip(got, ref)):
        pcc = _pcc(g, r)
        tg, tr = int(g.argmax()), int(r.argmax())
        top2 = torch.topk(r, 2).values
        gap = float(top2[0] - top2[1])
        logger.info(f"{name} step {k}: pcc {pcc:.6f} top-1 {tg} vs {tr} (ref top-2 gap {gap:.3f})")
        assert pcc >= pcc_min, f"{name} step {k}: logits pcc {pcc:.5f} < {pcc_min}"
        assert (
            tg == tr or gap < NEAR_TIE
        ), f"{name} step {k}: token {tg} != {tr} and the reference gap {gap:.3f} is no near-tie"


def _slot_state(a, logical: int, n_rows: int) -> dict[str, torch.Tensor]:
    """Every layer's device state of the request in logical slot ``logical`` after its prefill, on the host: per KDA
    layer the recurrent state and the conv history of its user, per DSA layer the latent rows [0, n_rows) of its user
    (the owner column's pages), the pooled-key index of those rows and the raw ring; read before any decode step."""
    model = a.model
    s = a.slots.physical(logical)
    row, user = int(s.row), int(s.user)
    out: dict[str, torch.Tensor] = {}
    cols = model.mesh.cols
    for i, dec in enumerate(model.layers):
        if dec.is_kda:
            chip = row * cols
            rec = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.recurrent)[chip])[user]
            conv = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.convolution)[chip])[:, user]
            out[f"L{i} kda.recurrent"] = rec.float().clone()
            out[f"L{i} kda.conv"] = conv.float().clone()
            continue
        cache = dec.state.cache if hasattr(dec, "state") else dec.cache
        owner = cache.owner_of(user) if cache.partition else 0
        chip = row * cols + (owner or 0)
        local = cache.prefill_slot(user)
        n_blocks = -(-int(n_rows) // cache.block_size)
        ids = cache.page_table_host[local, :n_blocks].tolist()
        pages = ttnn.to_torch(model.mesh.device_tensors(cache.cache)[chip])  # [blocks, 1, 32, L]
        out[f"L{i} dsa.latent"] = pages[ids].reshape(-1, pages.shape[-1])[: int(n_rows)].float().clone()
        idx = cache.index
        n_pools = -(-int(n_rows) // 4)
        pools = ttnn.to_torch(model.mesh.device_tensors(idx.pool_cache)[chip])  # [max_batch * pool_blocks, 1, 32, D]
        pools = pools.reshape(-1, idx.pool_blocks * 32, pools.shape[-1])[local, :n_pools]
        out[f"L{i} index.pools"] = pools.float().clone()
        out[f"L{i} index.ring_k"] = ttnn.to_torch(model.mesh.device_tensors(idx.ring_k)[chip])[local].float().clone()
    return out


def _diff_states(name: str, got: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> None:
    """Log, per component, the largest absolute difference and the first differing row (bit-identical = none)."""
    for key in ref:
        g, r = got[key], ref[key]
        if g.shape != r.shape:
            logger.info(f"{name}: {key}: shape {tuple(g.shape)} vs {tuple(r.shape)}")
            continue
        d = (g - r).abs()
        if not bool((d > 0).any()):
            logger.info(f"{name}: {key}: identical")
            continue
        flat = d.reshape(d.shape[0], -1) if d.dim() > 1 else d.reshape(-1, 1)
        rows = torch.nonzero(flat.max(dim=1).values > 0).reshape(-1)
        logger.info(
            f"{name}: {key}: max |diff| {float(d.max()):.4g} (ref |max| {float(r.abs().max()):.4g}); "
            f"{int(rows.numel())} of {flat.shape[0]} rows differ, first {int(rows[0])}, last {int(rows[-1])}"
        )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("store", ["device", "host"])
def test_served_prefix_cache_hit_matches_from_scratch(mesh_device, monkeypatch, store):
    if B % MESH[1]:
        pytest.skip(f"the user partition needs users_per_row {B} % cols {MESH[1]} == 0")
    monkeypatch.setenv("GLM53_PARTITION_USERS", "1")
    monkeypatch.setenv("GLM53_PREFILL_PARTITION", "heads2users")
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", "128,2048")  # the start-1024 continuation compiles at boot
    monkeypatch.setenv("GLM53_VLLM_APC_TAIL_SPLIT", "0")
    monkeypatch.setenv(APC_ALLOC_CHECK_ENV, "1")  # every APC call site keeps the post-capture DRAM footprint
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
    entries = 2
    model.allocate_kda_snapshots(entries, store)  # before any capture
    g = ServingGeometry(
        rows=MESH[0],
        cols=MESH[1],
        users_per_row=B,
        max_seq_len=MAX_SEQ_LEN,
        partitioned=True,
        device_prefill_chunk=CHUNK,
        pad_token_id=cfg.pad_token_id,
        traced_prefill_lengths=(),  # eager chunks only (the hit path); no prefill trace (2048 = 1024-row slabs)
    )
    a = GLM53FlashForCausalLM(model, g, pad_token_id=cfg.pad_token_id, prefix_cache_entries=entries)
    t0 = time.perf_counter()
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=False)
    a.warmup_model_decode(enable_trace=False, max_batch_size=B, num_blocks=1)
    a.warmup_model_decode(enable_trace=True, max_batch_size=B, num_blocks=1)
    logger.info(f"[{store}] warm-up + decode capture in {time.perf_counter() - t0:.0f} s")
    srv = _Served(a)
    vocab = cfg.vocab_size
    A = _ids(LEN_A, 1, vocab)
    turn = _ids(LEN_TURN, 2, vocab)
    side_prompt = _ids(200, 3, vocab)

    # ---- pass 1: prompt A in logical 0, its response, with a second user decoding alongside; A's slot retained
    la = srv.prefill(0, A, 0)
    s0 = a.slots.physical(0)
    user = s0.user
    assert a.apc.content(s0.row, user).snapshots == {1024: 1}, a.apc.content(s0.row, user).snapshots
    resp, _ = srv.generate(0, la, LEN_A, N_RESP)
    content = a.apc.content(s0.row, user)
    assert content.length == LEN_A + N_RESP and content.snapshots == {1024: 1, 1536: 0}, content.snapshots
    a.release_request(0)
    assert a.apc.park_positions() == {(s0.row, user): LEN_A + N_RESP}
    lb = srv.prefill(1, side_prompt, 0)  # another request decodes while A's slot is parked at 1540
    assert a.slots.physical(1).user != user
    srv.generate(1, lb, 200, 3)
    a.release_request(1)

    # ---- pass 2: the multi-turn request, vLLM's hit = 1536 (the content's full blocks); served from the CHUNK-ALIGNED
    # snapshot at 1024 (``SlotPrefixCache.hit_align``: the continued chunk starts at a warmed device-chunk multiple,
    # the rows [1024, 1536) are prefilled again), twice: the first hit and the warm (steady-state) hit (its content is
    # the first hit's + 5 decoded tokens, truncated back by the hit; entry 0 still holds 1536)
    full = torch.cat([A, torch.tensor(resp, dtype=torch.int32), turn])
    assert full.numel() == LEN_A + N_RESP + LEN_TURN
    hits = []  # (ms, logits, tokens, decode logits) per hit
    for n_hit in range(2):
        t0 = time.perf_counter()
        hit_logits = srv.prefill(2, full, 1536)
        hit_ms = (time.perf_counter() - t0) * 1000
        s2 = a.slots.physical(2)
        assert (s2.user, a.apc.stats["hits"], a.apc.stats["downgraded"]) == (user, n_hit + 1, 0), (s2.user, a.apc.stats)
        hit_toks, hit_outs = srv.generate(2, hit_logits, full.numel(), N_CHECK)
        a.release_request(2)
        hits.append((hit_ms, hit_logits, hit_toks, hit_outs))
    hit_ms, hit_logits, hit_toks, hit_outs = hits[0]
    warm_ms, warm_logits, warm_toks, warm_outs = hits[1]
    _assert_agree(f"[{store}] warm hit vs first hit", [warm_logits] + warm_outs, [hit_logits] + hit_outs)
    logger.info(
        f"[{store}] the two hits are {'bit-identical' if torch.equal(warm_logits, hit_logits) else 'not bit-identical'} "
        f"on the prefill logits; tokens {hit_toks} vs {warm_toks}"
    )

    # ---- pass 3: the from-scratch reference of the whole prompt in the SAME physical slot
    a.apc.forget(s2.row, user)  # drop the resident content so the balance rule picks the same user again
    for key in a.apc.retained_slots():
        a.apc.forget(*key)
    t0 = time.perf_counter()
    ref_logits = srv.prefill(3, full, 0)
    ref_ms = (time.perf_counter() - t0) * 1000
    assert a.slots.physical(3).user == user, "the reference must run in the hit's slot (per-column numerics)"
    ref_toks, ref_outs = srv.generate(3, ref_logits, full.numel(), N_CHECK)
    a.release_request(3)
    _assert_agree(f"[{store}] hit vs from-scratch prefill", [hit_logits] + hit_outs, [ref_logits] + ref_outs)
    _assert_agree(f"[{store}] warm hit vs from-scratch prefill", [warm_logits] + warm_outs, [ref_logits] + ref_outs)
    logger.info(
        f"PERF [{store}] hit prefill [{CHUNK}, {full.numel()}) (aligned restore of the 1536 hit) first {hit_ms:.0f} ms, "
        f"warm {warm_ms:.0f} ms vs "
        f"from-scratch [0, {full.numel()}) {ref_ms:.0f} ms; tokens hit {hit_toks} warm {warm_toks} ref {ref_toks}; "
        f"apc {a.apc.stats}"
    )

    # ---- miss: a stale hit (one token changed inside the "cached" prefix, A resident in `user`): no slot matches,
    # the request prefills from 0 in a slot without resident content, and equals its own from-scratch run there
    for key in a.apc.retained_slots():
        a.apc.forget(*key)
    srv.prefill(0, A, 0)  # A resident again (retained with its prefill snapshot at 1024 only)
    a.release_request(0)
    altered = full.clone()
    altered[700] = (int(altered[700]) + 1) % vocab
    miss_logits = srv.prefill(1, altered, 1536)
    assert a.apc.stats["downgraded"] == 1
    miss_row, miss_user = a.slots.physical(1).row, a.slots.physical(1).user
    assert miss_user != user, "a free slot without resident content is taken before the retained one"
    miss_toks, miss_outs = srv.generate(1, miss_logits, altered.numel(), N_CHECK)
    a.release_request(1)
    a.apc.forget(miss_row, miss_user)  # A stays retained in `user`, so the balance + rank rule picks miss_user again
    ref2_logits = srv.prefill(1, altered, 0)
    assert a.slots.physical(1).user == miss_user, (a.slots.physical(1).user, miss_user)
    ref2_toks, ref2_outs = srv.generate(1, ref2_logits, altered.numel(), N_CHECK)
    _assert_agree(f"[{store}] miss (downgraded) vs from-scratch", [miss_logits] + miss_outs, [ref2_logits] + ref2_outs)
    # a downgraded miss IS a from-scratch prefill in the same slot: bit-identical
    assert torch.equal(miss_logits, ref2_logits) and miss_toks == ref2_toks
    a.release_persistent_capture()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_served_vllm_cached_prefix_miss_with_budget_split(mesh_device, monkeypatch):
    """What vLLM's OWN prefix cache does to the adapter (the 2026-10-29 APC smoke on 45 layers): two prompts share a
    2048-token prefix. While A is mid-prefill vLLM reports the shared blocks computed for B (``num_computed_tokens``
    2048) and splits the step's token budget, so B's first scheduled chunk ends at 2112 (64 new tokens). The adapter
    holds no snapshot of that prefix in a retained slot (A's slot is live), recomputes ``[0, 2112)`` in B's own slot
    (device chunks 1024, 1024, 128) and then continues B from the chunk-UNALIGNED 2176 while A continues from 2048.
    Both decode greedily and must agree with their from-scratch prefills (the layer-level chunk plans are
    bit-identical -- ``test_chunked_prefill``; this is the adapter's miss + continuation + decode path)."""
    if B % MESH[1]:
        pytest.skip(f"the user partition needs users_per_row {B} % cols {MESH[1]} == 0")
    monkeypatch.setenv("GLM53_PARTITION_USERS", "1")
    monkeypatch.setenv("GLM53_PREFILL_PARTITION", "heads2users")
    # the ladder walks a 4096-token prompt too: the prompts here are 2348 tokens, their prefix copies / continued
    # chunks beyond 2048 must be warm under the seal (the served ladder reaches 32768)
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", "128,2048,4096")
    monkeypatch.setenv("GLM53_VLLM_APC_TAIL_SPLIT", "0")
    monkeypatch.setenv(APC_ALLOC_CHECK_ENV, "1")
    # what initialize_vllm_model sets in serving: the whole-prefix latent copies key on the device chunk, so a
    # continuation from any start reads a warmed width (the adapter is built directly here)
    monkeypatch.setenv("GLM53_PREFIX_BUCKET", str(CHUNK))
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
    entries = 2
    model.allocate_kda_snapshots(entries, "host")
    g = ServingGeometry(
        rows=MESH[0],
        cols=MESH[1],
        users_per_row=B,
        max_seq_len=MAX_SEQ_LEN,
        partitioned=True,
        device_prefill_chunk=CHUNK,
        pad_token_id=cfg.pad_token_id,
        traced_prefill_lengths=(),
    )
    a = GLM53FlashForCausalLM(model, g, pad_token_id=cfg.pad_token_id, prefix_cache_entries=entries)
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=False)
    a.warmup_model_decode(enable_trace=False, max_batch_size=B, num_blocks=1)
    a.warmup_model_decode(enable_trace=True, max_batch_size=B, num_blocks=1)
    srv = _Served(a)
    vocab = cfg.vocab_size
    shared = _ids(2 * CHUNK, 11, vocab)
    A = torch.cat([shared, _ids(300, 12, vocab)])
    Bp = torch.cat([shared, _ids(300, 13, vocab)])
    n = int(A.numel())
    split_end = 2 * CHUNK + 128  # a 128-aligned odd end: the served r3 shape (36736 = 287 x 128), the static index path
    # A's first two device chunks (two vLLM steps), then B joins with the shared prefix "computed" and 64 tokens of budget
    srv.prefill(0, A[:CHUNK], 0)
    srv.prefill(0, A[: 2 * CHUNK], CHUNK)
    t0 = time.perf_counter()
    srv.prefill(1, Bp[:split_end], 2 * CHUNK)
    miss_ms = (time.perf_counter() - t0) * 1000
    sa, sb = a.slots.physical(0), a.slots.physical(1)
    assert (sa.row, sa.user) != (sb.row, sb.user)
    logger.info(f"budget split: B's miss recompute [0, {split_end}) in {miss_ms:.0f} ms; apc {a.apc.stats}")
    la = srv.prefill(0, A, 2 * CHUNK)  # A continues from 2048
    lb = srv.prefill(1, Bp, split_end)  # B continues from the chunk-unaligned 2112
    # both decode together (a prefilled slot that is not in the decode batch counts as mid-prefill)
    toks_a, outs_a, toks_b, outs_b = [], [], [], []
    ta, tb = int(la.argmax()), int(lb.argmax())
    for k in range(N_CHECK):
        res = srv.decode({0: (ta, n + k), 1: (tb, n + k)})
        toks_a.append(ta)
        toks_b.append(tb)
        outs_a.append(res[0])
        outs_b.append(res[1])
        ta, tb = int(res[0].argmax()), int(res[1].argmax())
    a.release_request(0)
    a.release_request(1)
    for key in a.apc.retained_slots():
        a.apc.forget(*key)
    # B2: the SAME chunk plan as B (1024, 1024, 128, 172) through plain continuations (no cached-prefix miss): splits
    # the adapter's miss / recompute path from the chunk plan's own numerics
    srv.prefill(2, Bp[:CHUNK], 0)
    srv.prefill(2, Bp[: 2 * CHUNK], CHUNK)
    srv.prefill(2, Bp[:split_end], 2 * CHUNK)
    lb2 = srv.prefill(2, Bp, split_end)
    state_b2 = _slot_state(a, 2, n)  # before any decode step: what the chunk plan left for the decode
    toks_b2, outs_b2 = srv.generate(2, lb2, n, N_CHECK)
    a.release_request(2)
    for key in a.apc.retained_slots():
        a.apc.forget(*key)
    # the from-scratch references (the adapter's own 1024-token chunk plan from 0)
    ra = srv.prefill(0, A, 0)
    same_a = (a.slots.physical(0).row, a.slots.physical(0).user) == (sa.row, sa.user)
    rtoks_a, routs_a = srv.generate(0, ra, n, N_CHECK)
    a.release_request(0)
    rb = srv.prefill(1, Bp, 0)
    same_b = (a.slots.physical(1).row, a.slots.physical(1).user) == (sb.row, sb.user)
    state_rb = _slot_state(a, 1, n)
    _diff_states("B2's plan (1024+1024+128+172) vs from scratch (1024+1024+300)", state_b2, state_rb)
    rtoks_b, routs_b = srv.generate(1, rb, n, N_CHECK)
    a.release_request(1)
    logger.info(
        f"budget split: A tokens {toks_a} ref {rtoks_a} (same slot {same_a}, prefill logits equal {torch.equal(la, ra)}); "
        f"B tokens {toks_b} ref {rtoks_b} (same slot {same_b}, equal {torch.equal(lb, rb)}); apc {a.apc.stats}"
    )
    pb_b2 = [_pcc(x, y) for x, y in zip([lb] + outs_b, [lb2] + outs_b2)]
    logger.info(
        f"budget split: B (miss path) vs B2 (plain continuations, same plan) per step pcc {[f'{v:.6f}' for v in pb_b2]}; tokens B {toks_b} B2 {toks_b2}"
    )
    _assert_agree("A continued at 2048 vs from scratch", [la] + outs_a, [ra] + routs_a)
    _assert_agree(
        "B2: the plan 1024+1024+128+172 by plain continuations vs from scratch",
        [lb2] + outs_b2,
        [rb] + routs_b,
        pcc_min=0.99,
    )
    # B's prefill ran a different chunk plan than its reference (1024 + 1024 + 128 + 172 vs 1024 + 1024 + 300): the
    # sequence-parallel KDA scan groups the recurrence by the chunk length, so the carries differ at the ulp and the MoE
    # flips near-tie expert choices (0.998 on the prefill logits at 4 layers; tests/test_mesh_local_kda.py
    # test_sp_prefill_padded_ragged_chunk_carry); the tokens must still agree or differ only at a reference near-tie
    _assert_agree(
        "B: vLLM-cached 2048 miss, chunk end 2176, continued vs from scratch",
        [lb] + outs_b,
        [rb] + routs_b,
        pcc_min=0.99,
    )
    a.release_persistent_capture()
