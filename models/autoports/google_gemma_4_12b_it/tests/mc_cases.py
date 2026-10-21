# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic scenarios run on both the single-chip baseline and the QB2 multichip decoder.

A scenario ("case") builds its inputs from seeds, runs one or more decoder layers through a ``Backend`` and returns
``{label: output}``. ``single_chip_baseline.py`` runs every case on one chip (``OptimizedDecoder``, 1x1 mesh, no
fabric) in its own process and stores the outputs; ``test_multichip_decoder.py`` runs the same case on the
``[1, 4]`` mesh (``MultichipDecoder``) and compares output by output. The single-chip layer cannot share the
process: a chip-0 submesh of the open QB2 mesh corrupted both layers' results (work log, item 4).

With ``want=True`` a case also returns the HF float32 reference of each output (``{label: (got, want)}``).
"""

import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import BLOCK, CANARY_BLOCK, Pages
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    HFLayer,
    real_layer_state_dict,
    real_prompt_ids,
    synthetic_hidden,
)

REPO = Path(__file__).resolve().parents[4]
CACHE_DIR = Path(os.environ.get("GEMMA4_MC_CACHE", REPO / "generated" / "gemma4_12b_it_multichip"))
BASELINE_DIR = CACHE_DIR / "single_chip_baseline"
ACT_DIR = CACHE_DIR / "real_activations"

# Representative layers of every layer kind x precision group (tt/optimized_decoder.py layer_policy):
# 0 sliding BFP8, 5 full BFP8, 13 sliding BFP4 attention + MLP (60-core single-chip residual), 17 full BFP4
# attention + MLP, 24 sliding BFP4 MLP + BFP4 QKV, 25 sliding BFP4 gate/up + BFP4 attention, 35 full BFP4 gate/up,
# 47 full with HiFi2 prefill matmuls; 10 sliding with BF16 prefill MLP intermediates.
POLICY_LAYERS = [0, 5, 10, 13, 17, 24, 25, 35, 47]
LENGTHS = [1, 17, 65, 100, 1000, 1025, 1344, 2049, 4097, 8193]
SHORT_LENGTHS = [100, 1344, 4097]


def kind_of(cfg, layer_idx):
    return "sliding" if cfg.layer_types[layer_idx] == "sliding_attention" else "full"


def stats_layer(cfg, layer_idx):
    """Synthetic activations use the recorded statistics of the representative layer of the same kind."""
    return 0 if kind_of(cfg, layer_idx) == "sliding" else 5


REAL_POOL = 4099  # tokens of the real prompt whose layer inputs are cached for the non-representative layers


def layer_inputs(cfg, layer_idx, n, seed=0, offset=0):
    """``[1, n, H]`` inputs of ``layer_idx``: synthetic (recorded per-channel statistics) for the representative
    layers 0 and 5; for every other layer a slice of the HF float32 stack's real activations of a real chat prompt
    starting at token ``offset``. Synthetic statistics of layer 0 / 5 are off-distribution for deeper layers (the
    BFP4 layers then score ~0.99 against HF on one chip too, work log item 9)."""
    if layer_idx in (0, 5):
        return synthetic_hidden(layer_idx, n, seed=seed)
    pool = max(REAL_POOL, offset + n)
    return real_inputs(cfg, layer_idx, pool)[:, offset : offset + n].contiguous()


# ---------------------------------------------------------------------------------------------- real activations
def real_inputs(cfg, layer_idx, tokens):
    """HF float32 input of ``layer_idx`` for the first ``tokens`` tokens of the real chat prompt (disk-cached).

    Causal layers make a prefix of a longer prompt's activations equal to the shorter prompt's, so the cache keeps
    the longest prompt computed per layer. Layers are computed in order from the deepest cached layer below.
    """
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    def path(i, n):
        return ACT_DIR / f"layer{i:02d}_{n}.pt"

    def cached(i):
        best = None
        for p in ACT_DIR.glob(f"layer{i:02d}_*.pt"):
            n = int(p.stem.split("_")[1])
            if n >= tokens and (best is None or n < best[0]):
                best = (n, p)
        return best

    hit = cached(layer_idx)
    if hit:
        return torch.load(hit[1])[:, :tokens]
    start = max((i for i in range(layer_idx) if cached(i)), default=None)
    if start is None:
        from models.autoports.google_gemma_4_12b_it.tests.hf_reference import real_hidden

        x = real_hidden(cfg, 0, real_prompt_ids(tokens))
        torch.save(x, path(0, tokens))
        start = 0
    else:
        x = torch.load(cached(start)[1])[:, :tokens]
    for i in range(start, layer_idx):
        ref = HFLayer(cfg, i, real_layer_state_dict(i))
        x = ref.prefill(x, ref.new_cache(), slice_len=2048)
        torch.save(x, path(i + 1, tokens))
    return x


# ---------------------------------------------------------------------------------------------- backends
class Backend:
    """Device adapter: builds a decoder and moves the decoder's inputs/outputs across the test boundary."""

    name = None

    def __init__(self, device):
        self.device = device

    def ints(self, t):
        return ttnn.from_torch(
            t.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self._replicate(),
        )

    def host_ints(self, t):
        return ttnn.from_torch(
            t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._replicate()
        )

    def decode_in(self, dec, x):
        """``[1, B, H]`` -> the decoder's decode residual layout (replicated on a mesh)."""
        return ttnn.from_torch(
            x.reshape(1, 1, x.shape[-2], x.shape[-1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=dec.decode_input_memory_config(x.shape[-2]),
            mesh_mapper=self._replicate(),
        )

    def host_decode(self, x):
        return ttnn.from_torch(
            x.reshape(1, 1, x.shape[-2], x.shape[-1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate(),
        )

    def _replicate(self):
        return None

    def sync(self):
        ttnn.synchronize_device(self.device)


class SingleChip(Backend):
    name = "single"

    def build(self, cfg, layer_idx, sd, num_blocks, **kw):
        from models.autoports.google_gemma_4_12b_it.tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder.from_state_dict(
            sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=self.device, num_blocks=num_blocks, **kw
        )

    def prefill_in(self, dec, x):
        return ttnn.from_torch(
            x.reshape(1, 1, x.shape[-2], x.shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def prefill_out(self, t):
        return ttnn.to_torch(t).float().reshape(1, t.shape[-2], t.shape[-1])

    def decode_out(self, t):
        return ttnn.to_torch(t).float().reshape(1, t.shape[-2], t.shape[-1])

    def cache_blocks(self, dec, blocks):
        """Per-chip view of the K/V cache blocks ``blocks``: ``[(k, v)]`` with one entry (one chip)."""
        return [tuple(ttnn.to_torch(c).float()[blocks] for c in dec.kv_cache)]


class Multichip(Backend):
    name = "multichip"

    def _replicate(self):
        return ttnn.ReplicateTensorToMesh(self.device)

    def build(self, cfg, layer_idx, sd, num_blocks, **kw):
        from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import MultichipDecoder

        return MultichipDecoder.from_state_dict(
            sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=self.device, num_blocks=num_blocks, **kw
        )

    def prefill_in(self, dec, x):
        return ttnn.from_torch(
            x.reshape(1, 1, x.shape[-2], x.shape[-1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=dec.prefill_input_mesh_mapper(),
        )

    def prefill_out(self, t):
        out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=3)).float()
        return out.reshape(1, t.shape[-2], -1)

    def decode_out(self, t):
        reps = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)).float()
        for i in range(1, reps.shape[0]):
            assert torch.equal(reps[0], reps[i]), f"decode output replica {i} differs from replica 0"
        return reps[0].reshape(1, t.shape[-2], t.shape[-1])

    def cache_blocks(self, dec, blocks):
        per = [
            ttnn.to_torch(c, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)).float() for c in dec.kv_cache
        ]
        n = dec.num_blocks
        return [tuple(c[d * n : (d + 1) * n][blocks] for c in per) for d in range(per[0].shape[0] // n)]


class TracedDecode:
    """Decode trace with persistent device inputs (residual-layout hidden state, positions, page table)."""

    def __init__(self, be, dec, batch, table, user_split=None):
        self.be, self.dec, self.batch = be, dec, batch
        # multichip only: force the full-layer user-split choice (decode_forward(user_split=...)); None = layer default
        self.kw = {"user_split": user_split} if user_split is not None and hasattr(dec, "use_user_split") else {}
        self.x = be.decode_in(dec, torch.zeros(1, batch, dec.hidden_size))
        self.pos = be.ints(torch.zeros(batch))
        self.page_table = be.ints(table)
        self.trace_id = None
        self.out = None

    def set_inputs(self, x, pos, table):
        ttnn.copy_host_to_device_tensor(self.be.host_decode(x), self.x)
        ttnn.copy_host_to_device_tensor(self.be.host_ints(pos), self.pos)
        ttnn.copy_host_to_device_tensor(self.be.host_ints(table), self.page_table)

    def capture(self):
        self.dec.decode_forward(self.x, current_pos=self.pos, page_table=self.page_table, **self.kw)
        self.be.sync()
        self.trace_id = ttnn.begin_trace_capture(self.be.device, cq_id=0)
        self.out = self.dec.decode_forward(self.x, current_pos=self.pos, page_table=self.page_table, **self.kw)
        ttnn.end_trace_capture(self.be.device, self.trace_id, cq_id=0)
        self.be.sync()

    def replay(self):
        ttnn.execute_trace(self.be.device, self.trace_id, cq_id=0, blocking=False)
        return self.out

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.be.device, self.trace_id)
            self.trace_id = None


# ---------------------------------------------------------------------------------------------- cases
class Recorder:
    def __init__(self, want):
        self.want = want
        self.out = {}

    def add(self, label, got, want=None):
        assert label not in self.out, label
        self.out[label] = (got, want) if self.want else got


def case_lengths(be, cfg, layer_idx, want=False, lengths=None):
    """Fresh requests of non-aligned lengths on one instance (fresh shuffled pages each), two decodes after each.

    Real weights, synthetic activations (per-channel statistics of the same layer kind). Covers tile, block,
    window, 2D-matmul/minimal_matmul crossover (1280 / 1344) and default-chunk (8192) boundaries.
    """
    rec = Recorder(want)
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd) if want else None
    lengths = lengths or (LENGTHS if layer_idx in (0, 5) else SHORT_LENGTHS)
    max_pages = (max(lengths) + 2 + BLOCK - 1) // BLOCK
    dec = be.build(cfg, layer_idx, sd, num_blocks=2 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=1, seed=layer_idx)
    for i, n in enumerate(lengths):
        pages.release(0)
        pages.ensure(0, n + 2)
        table = be.ints(pages.table)
        x = layer_inputs(cfg, layer_idx, n + 2, seed=100 + i)
        cache = ref.new_cache() if want else None
        w = ref.prefill(x[:, :n], cache, slice_len=2048) if want else None
        got = be.prefill_out(dec.prefill_forward(be.prefill_in(dec, x[:, :n]), page_table=table))
        rec.add(f"prefill[S={n}]", got, w)
        for p in (n, n + 1):
            out = dec.decode_forward(
                be.decode_in(dec, x[:, p : p + 1]), current_pos=be.ints(torch.tensor([p])), page_table=table
            )
            rec.add(
                f"decode[S={n},pos={p}]", be.decode_out(out), ref.forward(x[:, p : p + 1], cache, p) if want else None
            )
    return rec.out


def case_continuation(be, cfg, layer_idx, want=False):
    """One 12000-token prompt prefilled three ways into separate users, then decoded from each user's cache:
    one call (8192 + 3808-row chunks), explicit continuations at 1088 / 3200 / 8256 (page-aligned starts across
    the window and the default chunk), and a stale-page reuse by a new 300-token request."""
    rec = Recorder(want)
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd) if want else None
    n, steps = 12000, 2
    max_pages = (n + steps + BLOCK - 1) // BLOCK
    dec = be.build(cfg, layer_idx, sd, num_blocks=3 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=3, seed=11)
    x = synthetic_hidden(stats_layer(cfg, layer_idx), n + steps, seed=5)
    cache = ref.new_cache() if want else None
    w = ref.prefill(x[:, :n], cache, slice_len=2048) if want else None
    w_dec = [ref.forward(x[:, p : p + 1], cache, p) for p in range(n, n + steps)] if want else [None] * steps
    pages.ensure(0, n + steps)
    pages.ensure(1, n + steps)
    table = be.ints(pages.table)
    rec.add(
        "one_call_prefill",
        be.prefill_out(dec.prefill_forward(be.prefill_in(dec, x[:, :n]), page_table=table, user_id=0)),
        w,
    )
    parts = []
    bounds = (0, 1088, 3200, 8256, n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out = dec.prefill_forward(be.prefill_in(dec, x[:, a:b]), page_table=table, user_id=1, start_pos=a)
        parts.append(be.prefill_out(out))
    rec.add("continuation_prefill", torch.cat(parts, dim=1), w)
    for user in (0, 1):
        row = be.ints(pages.table[user : user + 1])
        for s, p in enumerate(range(n, n + steps)):
            out = dec.decode_forward(
                be.decode_in(dec, x[:, p : p + 1]), current_pos=be.ints(torch.tensor([p])), page_table=row
            )
            rec.add(f"decode_after[user={user},pos={p}]", be.decode_out(out), w_dec[s])
    pages.release(0)
    pages.ensure(2, 301)
    y = synthetic_hidden(stats_layer(cfg, layer_idx), 301, seed=99)
    ycache = ref.new_cache() if want else None
    wy = ref.prefill(y[:, :300], ycache) if want else None
    table = be.ints(pages.table)
    rec.add(
        "reused_prefill",
        be.prefill_out(dec.prefill_forward(be.prefill_in(dec, y[:, :300]), page_table=table, user_id=2)),
        wy,
    )
    out = dec.decode_forward(
        be.decode_in(dec, y[:, 300:301]), current_pos=be.ints(torch.tensor([300])), page_table=be.ints(pages.table[2:3])
    )
    rec.add("reused_decode", be.decode_out(out), ref.forward(y[:, 300:301], ycache, 300) if want else None)
    rec.add("canary_clean", torch.tensor([float(all(_canary_clean(be, dec)))]))
    return rec.out


def _canary_clean(be, dec):
    return [bool((k.abs().max() == 0) and (v.abs().max() == 0)) for k, v in be.cache_blocks(dec, [CANARY_BLOCK])]


def case_traced(be, cfg, layer_idx, batch, want=False, steps=6, stress_steps=0, stress_every=60, user_split=None):
    """Traced decode at ``batch`` users with per-replay refresh of hidden states, positions and page table.

    Users are prefilled to different lengths (page edges, window crossings during decode); with batch > 3 user 3 is
    inactive (-1) for the first three replays. ``stress_steps`` more replays follow, recorded every
    ``stress_every`` replays. The last replay is repeated with identical inputs (bitwise determinism).
    """
    rec = Recorder(want)
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd) if want else None
    total = steps + stress_steps
    g = torch.Generator().manual_seed(batch + layer_idx)
    if batch == 1:
        lengths = [1086]
    else:
        lengths = torch.randint(1, 1300, (batch,), generator=g).tolist()
        lengths[:4] = [1, 63, 1020, 1279]
    max_pages = (max(lengths) + total + BLOCK) // BLOCK
    dec = be.build(cfg, layer_idx, sd, num_blocks=batch * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=batch, seed=batch)
    xs, caches = [], []
    for u, n in enumerate(lengths):
        x = layer_inputs(cfg, layer_idx, n + total, seed=1000 + u, offset=13 * u)
        xs.append(x)
        if want:
            caches.append(ref.new_cache())
            ref.prefill(x[:, :n], caches[-1])
        pages.ensure(u, n)
        dec.prefill_forward(be.prefill_in(dec, x[:, :n]), page_table=be.ints(pages.table), user_id=u)
    inactive = {3} if batch > 3 else set()
    trace = TracedDecode(be, dec, batch, pages.table, user_split=user_split)
    pos = torch.tensor(lengths)
    done = [0] * batch
    last = None
    for step in range(total):
        if step == 3:
            inactive.clear()
        cur = []
        for u in range(batch):
            if u in inactive:
                cur.append(-1)
            else:
                pages.ensure(u, int(pos[u]) + 1)
                cur.append(int(pos[u]))
        x = torch.cat([xs[u][:, lengths[u] + done[u] : lengths[u] + done[u] + 1] for u in range(batch)], 1)
        cur = torch.tensor(cur)
        trace.set_inputs(x, cur, pages.table)
        if step == 0:
            trace.capture()
        out = trace.replay()
        score = step < steps or (step + 1) % stress_every == 0
        got = be.decode_out(out) if score else None
        for u in range(batch):
            if cur[u] < 0:
                continue
            p = int(cur[u])
            w = ref.forward(xs[u][:, p : p + 1], caches[u], p) if want else None
            if score:
                rec.add(f"step{step}[user={u},pos={p}]", got[:, u : u + 1], w)
            done[u] += 1
            pos[u] += 1
        last = (x, cur)
    x, cur = last
    trace.set_inputs(x, cur, pages.table)
    a = be.decode_out(trace.replay()).clone()
    trace.set_inputs(x, cur, pages.table)
    b = be.decode_out(trace.replay())
    rec.add("replay_bitwise_equal", torch.tensor([float(torch.equal(a, b))]))
    trace.release()
    rec.add("canary_clean", torch.tensor([float(all(_canary_clean(be, dec)))]))
    return rec.out


def case_cache_layout(be, cfg, layer_idx, want=False):
    """Paged K/V written by prefill (1100 tokens) and two decodes, read back per chip from the used blocks."""
    rec = Recorder(want)
    sd = real_layer_state_dict(layer_idx)
    n = 1100
    dec = be.build(cfg, layer_idx, sd, num_blocks=40)
    pages = Pages(dec.num_blocks, 20, num_users=1, seed=3)
    pages.ensure(0, n + 2)
    table = be.ints(pages.table)
    x = synthetic_hidden(stats_layer(cfg, layer_idx), n + 2, seed=8)
    dec.prefill_forward(be.prefill_in(dec, x[:, :n]), page_table=table)
    for p in (n, n + 1):
        dec.decode_forward(be.decode_in(dec, x[:, p : p + 1]), current_pos=be.ints(torch.tensor([p])), page_table=table)
    blocks = pages.table[0, : pages.owned[0]].long().tolist()
    for d, (k, v) in enumerate(be.cache_blocks(dec, blocks)):
        # [blocks, heads, block, hd] -> [heads, tokens, hd]; positions beyond n + 2 are never written
        rec.add(f"k[chip={d}]", k.permute(1, 0, 2, 3).reshape(k.shape[1], -1, k.shape[-1])[:, : n + 2])
        rec.add(f"v[chip={d}]", v.permute(1, 0, 2, 3).reshape(v.shape[1], -1, v.shape[-1])[:, : n + 2])
    rec.add("canary_clean", torch.tensor([float(all(_canary_clean(be, dec)))]))
    return rec.out


def case_stack(be, cfg, layers, batch, want=False, steps=4, shared_step_inputs=False, prefill_label="prefill"):
    """A stack of layers: prefill and traced decode chain each layer's output into the next layer's input on device,
    with no conversion between layers (the stacked-decoder layout contract).

    ``shared_step_inputs``: the decode step computes the RoPE rows once per layer kind and, on the mesh, the
    user-split inputs once (``decode_rope_rows`` / ``decode_split_inputs``) inside the captured step and passes them
    to every layer, as a full-model generator does. ``prefill_label``: label of the chained prefill outputs (the
    test applies a diagnostic bar to ``prefill_chain_diag``)."""
    rec = Recorder(want)
    shared = {}
    decs, refs = [], []
    lengths = [700 + 37 * u for u in range(batch)]
    max_pages = (max(lengths) + steps + BLOCK) // BLOCK
    for i in layers:
        sd = real_layer_state_dict(i)
        decs.append(be.build(cfg, i, sd, num_blocks=batch * max_pages + 1, shared=shared))
        refs.append(HFLayer(cfg, i, sd) if want else None)
    pages = Pages(decs[0].num_blocks, max_pages, num_users=batch, seed=5)
    # every user is a real prompt from its start (BOS); prefixes of different lengths (stack_precision_probe.py:
    # a request starting mid-prompt makes its first rows unusually sensitive to rounding in a chained stack)
    xs = [layer_inputs(cfg, layers[0], n + steps, seed=300 + u) for u, n in enumerate(lengths)]
    caches = [[r.new_cache() for r in refs] for _ in range(batch)] if want else None
    for u, n in enumerate(lengths):
        pages.ensure(u, n + steps)
        table = be.ints(pages.table)
        h = be.prefill_in(decs[0], xs[u][:, :n])
        for d in decs:
            h = d.prefill_forward(h, page_table=table, user_id=u)
        w = None
        if want:
            w = xs[u][:, :n].float()
            for r, c in zip(refs, caches[u]):
                w = r.prefill(w, c)
        rec.add(f"{prefill_label}[user={u}]", be.prefill_out(h), w)
    # decode: one trace over the whole stack
    x0 = be.decode_in(decs[0], torch.zeros(1, batch, decs[0].hidden_size))
    pos = be.ints(torch.zeros(batch))
    pt = be.ints(pages.table)

    def stack_step():
        h = x0
        if not shared_step_inputs:
            for d in decs:
                h = d.decode_forward(h, current_pos=pos, page_table=pt)
            return h
        rows = {}
        for d in decs:
            if d.layer_type not in rows:
                rows[d.layer_type] = d.decode_rope_rows(pos, batch)
        split = decs[0].decode_split_inputs(pos, pt) if hasattr(decs[0], "decode_split_inputs") else None
        for d in decs:
            kw = {"rope_rows": rows[d.layer_type]}
            if split is not None:
                kw["split_inputs"] = split
            h = d.decode_forward(h, current_pos=pos, page_table=pt, **kw)
        for t in [t for r in rows.values() for t in r] + list(split or ()):
            ttnn.deallocate(t)
        return h

    tid = None
    for s in range(steps):
        x = torch.cat([xs[u][:, lengths[u] + s : lengths[u] + s + 1] for u in range(batch)], 1)
        cur = torch.tensor([lengths[u] + s for u in range(batch)])
        ttnn.copy_host_to_device_tensor(be.host_decode(x), x0)
        ttnn.copy_host_to_device_tensor(be.host_ints(cur), pos)
        if s == 0:
            stack_step()
            be.sync()
            tid = ttnn.begin_trace_capture(be.device, cq_id=0)
            out = stack_step()
            ttnn.end_trace_capture(be.device, tid, cq_id=0)
        ttnn.execute_trace(be.device, tid, cq_id=0, blocking=False)
        got = be.decode_out(out)
        for u in range(batch):
            w = None
            if want:
                w = xs[u][:, int(cur[u]) : int(cur[u]) + 1].float()
                for r, c in zip(refs, caches[u]):
                    w = r.forward(w, c, int(cur[u]))
            rec.add(f"decode[step={s},user={u}]", got[:, u : u + 1], w)
    ttnn.release_trace(be.device, tid)
    return rec.out


def case_real(be, cfg, layer_idx, want=False, tokens=1024):
    """Real activations of a real chat prompt: prefill ``tokens``, decode the next two tokens (eager)."""
    rec = Recorder(want)
    sd = real_layer_state_dict(layer_idx)
    x = real_inputs(cfg, layer_idx, tokens + 2)
    w_all = None
    if want:
        ref = HFLayer(cfg, layer_idx, sd)
        w_all = ref.prefill(x, ref.new_cache(), slice_len=2048)
    pages_n = (tokens + 2 + BLOCK - 1) // BLOCK
    dec = be.build(cfg, layer_idx, sd, num_blocks=pages_n + 1)
    pages = Pages(dec.num_blocks, pages_n, num_users=1, seed=layer_idx)
    pages.ensure(0, tokens + 2)
    table = be.ints(pages.table)
    got = be.prefill_out(dec.prefill_forward(be.prefill_in(dec, x[:, :tokens]), page_table=table))
    rec.add("prefill", got, w_all[:, :tokens] if want else None)
    for p in (tokens, tokens + 1):
        out = dec.decode_forward(
            be.decode_in(dec, x[:, p : p + 1]), current_pos=be.ints(torch.tensor([p])), page_table=table
        )
        rec.add(f"decode[pos={p}]", be.decode_out(out), w_all[:, p : p + 1] if want else None)
    return rec.out


def case_registry(cfg):
    """Case id -> callable(be, want) for every case with a single-chip baseline."""
    reg = {}
    for i in POLICY_LAYERS:
        reg[f"lengths_L{i}"] = lambda be, want, i=i: case_lengths(be, cfg, i, want)
    for i in (0, 5):
        reg[f"continuation_L{i}"] = lambda be, want, i=i: case_continuation(be, cfg, i, want)
        reg[f"cache_layout_L{i}"] = lambda be, want, i=i: case_cache_layout(be, cfg, i, want)
        for b in (1, 32):
            reg[f"traced_L{i}_b{b}"] = lambda be, want, i=i, b=b: case_traced(be, cfg, i, b, want)
        reg[f"stress_L{i}_b32"] = lambda be, want, i=i: case_traced(be, cfg, i, 32, want, steps=3, stress_steps=300)
    for i in (13, 17):
        reg[f"traced_L{i}_b32"] = lambda be, want, i=i: case_traced(be, cfg, i, 32, want)
    # full-layer user-split decode attention (B >= 8, a multiple of 4): 2, 3 and 4 users per chip
    for b in (8, 12, 16):
        reg[f"traced_L5_b{b}"] = lambda be, want, b=b: case_traced(be, cfg, 5, b, want)
    # the generator's per-context choice (use_user_split): the unsplit path at B=32 and the split path at B=4
    reg["traced_L5_b32_unsplit"] = lambda be, want: case_traced(be, cfg, 5, 32, want, user_split=False)
    reg["traced_L5_b4_split"] = lambda be, want: case_traced(be, cfg, 5, 4, want, user_split=True)
    for layers in ((3, 4, 5, 6), (11, 12, 13)):
        for b in (1, 32):
            name = "-".join(str(v) for v in layers)
            reg[f"stack_{name}_b{b}"] = lambda be, want, layers=layers, b=b: case_stack(be, cfg, layers, b, want)
    # layers 11..17 on real activations: two full layers (11 BFP8, 17 BFP4) share the mesh's user-split state, and the
    # per-step RoPE rows and user-split inputs are computed once inside the captured step (the full-model contract).
    # Its 7-layer chained prefill (unchanged short-chunk path, five BFP4 layers) is scored with a diagnostic bar.
    reg["stack_11-17_b16_shared"] = lambda be, want: case_stack(
        be, cfg, tuple(range(11, 18)), 16, want, steps=6, shared_step_inputs=True, prefill_label="prefill_chain_diag"
    )
    for i in range(cfg.num_hidden_layers):
        reg[f"real_L{i}"] = lambda be, want, i=i: case_real(be, cfg, i, want)
    for i in (0, 5, 13, 17):
        reg[f"real_long_L{i}"] = lambda be, want, i=i: case_real(be, cfg, i, want, tokens=8192)
    return reg
