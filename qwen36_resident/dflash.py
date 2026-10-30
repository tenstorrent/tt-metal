# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ttnn DFlash2 drafter (models/demos/blackhole/qwen36/tt/dflash2_tp.py, DFlash2DrafterTP) on the resident
model's mesh.

The drafter runs as ordinary ttnn ops (and traces) on CQ0 between resident launches. It cannot read the resident
lm_head (BankLayout tiles with the final norm folded in), so it gets its own vocab-sharded bf8 copy of the
checkpoint's lm_head ([H, V / n] per chip, the drafter applies its own final norm), a TT_CCL for its all-reduces
and the target's token embedding on the host. Its context taps are the target's residual after layers
cfg["taps"], dim-fractured: chip c holds columns [c H / n, (c + 1) H / n) of every tap row.
"""
import hashlib
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, H, resolve_weights_dir
from models.demos.blackhole.qwen36.tt.dflash2_tp import DFlash2DrafterTP
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
from qwen36_resident import weights as Q

# 1: restore the initial recurrent state before every prompt, also when the prefill handoff overwrites it (A/B)
FULL_RESET = os.environ.get("RESIDENT_PREFILL_FULL_RESET", "0") == "1"
# 0: no host synchronization between the drafter's context fill and the prompt's verify step (the first-step race, A/B
# only; see _fill_context)
FILL_SYNC = os.environ.get("RESIDENT_DFLASH_FILL_SYNC", "1") == "1"


def ccl_topology():
    """Ring on the boxes whose chips form one (ModelArgs.ccl_topology), else Linear"""
    ring = (ttnn.cluster.ClusterType.P300_X2, ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P150_X8)
    return ttnn.Topology.Ring if ttnn.cluster.get_cluster_type() in ring else ttnn.Topology.Linear


def cache_root():
    root = os.environ.get("TT_CACHE_PATH")
    return f"{root}/resident_dflash" if root else None


DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}


def lm_head_vshard(mesh, ck, cache_dir=None, dtype=ttnn.bfloat8_b):
    """the checkpoint's lm_head as [H, V / n] per chip (vocab-sharded, no norm folded; bf8 by default)"""
    tag = "" if dtype == ttnn.bfloat8_b else f".{str(dtype).split('.')[-1]}"
    name = f"{cache_dir}/lm_head.vshard{tag}" if cache_dir else None
    w = ck.get("lm_head.weight")  # [V, H]
    return ttnn.as_tensor(
        w,
        preprocess=lambda t: t.T.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=name,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def build_drafter(mesh, ck, weights_dir=None, embed=None, block=None):
    """DFlash2DrafterTP on `mesh` for the target checkpoint ck; weights cached under $TT_CACHE_PATH/resident_dflash.
    DFL_MLP_DTYPE / DFL_LM_DTYPE (bf16 | bf8 | bf4): the drafter's MLP weights (default bf16) and its lm_head
    (default bf8)"""
    weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)
    real = os.path.realpath(resolve_weights_dir(weights_dir))
    root = cache_root()
    if root:
        os.makedirs(root, exist_ok=True)
    cache = f"{root}/{os.path.basename(real)}-{hashlib.sha1(real.encode()).hexdigest()[:10]}" if root else None
    emb = (Q.embedding(ck) if embed is None else embed).to(torch.bfloat16)
    mlp_dtype = DTYPES[os.environ.get("DFL_MLP_DTYPE", "bf16")]
    lm = lm_head_vshard(mesh, ck, root, DTYPES[os.environ.get("DFL_LM_DTYPE", "bf8")])
    drafter = DFlash2DrafterTP(
        mesh,
        weights_dir,
        emb,
        TT_CCL(mesh),
        ccl_topology(),
        lm,
        lm_vocab_sharded=True,
        block=block,
        mlp_dtype=mlp_dtype,
        cache_dir=cache,
    )
    logger.info(
        f"[resident-dflash] drafter {real}: block {drafter.block}, taps {drafter.taps}, windows {drafter.windows}, "
        f"selector {drafter.has_selector}, topology {ccl_topology()}"
    )
    return drafter


def tap_buffers(mesh, ntaps, rows, values=None):
    """`ntaps` persistent dim-fractured [1, 1, rows, H / n] bf16 tap tensors (zeros, or values [ntaps, rows, H])"""
    out = []
    for i in range(ntaps):
        x = torch.zeros(1, 1, rows, H) if values is None else values[i].reshape(1, 1, rows, H)
        out.append(
            ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
        )
    return out


class ResidentDFlash:
    """Greedy DFlash2 speculative decode of one user on the resident engine (host driven, v1).

    The target runs R-row verify steps (ResidentModel(verify=R)) whose hubs write the drafter's context taps; the
    drafter runs between the target's launches on CQ0. State: the target has consumed positions < p and t (the
    pending token at p) is known. An iteration drafts R - 1 tokens from t over the context 0 .. p - 1, verifies
    rows [t, d_1 .. d_R-1] at p .. p + R - 1, accepts the longest prefix of drafts that equal the greedy tokens of
    the rows before them (a of them), extends the drafter's context with the taps of rows 0 .. a and continues
    from row a (the bonus token, the greedy token of row a, becomes the pending one at p + a + 1). The prompt goes
    through the same verify steps, R tokens per step; with the pipeline-parallel prefill (prefill=True) a prompt but its
    last token is prefilled and handed off, the prefill's taps of the last `window` positions (the drafter's sliding
    window) fill the drafter's context on device, and the last token goes through a verify step."""

    PREFILL_CHUNKS = (128, 256, 512, 1024)

    def __init__(self, mesh, ck, max_pos=4096, R=8, layers=None, weights_dir=None, prefill=False, window=2048):
        from models.demos.blackhole.qwen36.tt.dflash2 import load_config
        from qwen36_resident.model import Dims, ResidentModel, State, is_attn

        cfg = ck.config
        layers = layers or cfg["num_hidden_layers"]
        interval = cfg["full_attention_interval"]
        self.mesh, self.R, self.vocab = mesh, R, cfg["vocab_size"]
        d = self.d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=self.vocab)
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        emb = Q.embedding(ck)
        self.emb = emb.float()
        weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)
        tap_layers = list(load_config(weights_dir)["taps"])
        self.pp, self.window = None, window
        self.use_prefill = prefill  # (False: every prompt token through verify steps, e.g. as a reference)
        self.tap_log = self.draft_log = None  # (diagnostics: lists -> the taps / drafts of every step)
        self.prompt_times = None  # (a list -> the prefill path's stage ends, synchronized: (name, perf_counter))
        if prefill:
            from qwen36_resident.prefill.pp_prefill import PPPrefill

            C = max(self.PREFILL_CHUNKS)
            max_len = -(-max_pos // C) * C
            max_pos = max_len + 64
            self.pp = PPPrefill(
                mesh,
                ck,
                layers,
                interval,
                chunk=list(self.PREFILL_CHUNKS),
                max_len=max_len,
                taps=tap_layers,
                tap_window=window,
            )
            self.fill_rows = window + 64  # the drafter's context fill: one bucket (one program)
            # per tap: 1 on the chip whose prefill stage computes its layer, 0 on the others
            n, Ls = mesh.get_num_devices(), layers // mesh.get_num_devices()
            self.tap_masks = [
                ttnn.from_torch(
                    torch.eye(n)[L // Ls].reshape(n, 1, 1, 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                for L in tap_layers
            ]
        self.taps = tap_buffers(mesh, len(tap_layers), R)
        st = State(d, layers - n_attn, n_attn, 0, max_pos, zero=True, kv=False)
        self.m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            st,
            layers,
            interval,
            True,
            verify=R,
            taps=list(zip(tap_layers, self.taps)),
        )
        self.drafter = build_drafter(mesh, ck, weights_dir, embed=emb, block=R)
        nb = -(-max_pos // 64)
        self.drafter.alloc(torch.arange(nb, dtype=torch.int32).reshape(1, nb), 64, self.taps)
        self.max_pos = max_pos
        self.ring = self.slot = 0
        self.prev = None
        # every program once, eagerly, before any trace is captured; then the target's trace (the drafter's are
        # captured on their first armed call)
        self._step([0] * R, 0)
        self.m.argmax()
        self.drafter.extend_context(0, 0, traced=False)
        self.drafter.draft(0, 1, traced=False)
        ttnn.synchronize_device(mesh)
        if self.pp is not None:
            self._fill_context(1)  # compiles the context fill
            self.pp.capture()
            self.pp.handoff(self.m, 1)  # compiles the handoff kernels
        self.m.reset()
        self.m.capture_trace()
        self.drafter.arm_traces()
        self.stop()

    def _step(self, toks, p):
        self.m.step(
            self.m.token_state(
                self.emb[list(toks)], list(range(p, p + self.R)), ring=self.ring, slot=self.slot, prev=self.prev
            )
        )

    def _commit(self, a):
        self.ring, self.slot, self.prev = self.ring + 1 + a, (self.slot + a) % self.R, a

    def _prompt(self, ids, draft_context=True):
        """the prompt through verify steps (R tokens each, all accepted): returns the first generated token"""

        def mark(name):
            if self.prompt_times is not None:
                ttnn.synchronize_device(self.mesh)
                self.prompt_times.append((name, time.perf_counter()))

        P, R = len(ids), self.R
        prefill = self.pp is not None and self.use_prefill and P > 1 and draft_context
        mark("begin")
        # the initial recurrent state (a ~37 MB / chip host upload, ~80 ms) only where no handoff overwrites it: the
        # handoff writes every GDN layer's state into slot 0 and the conv history rows the first step reads (the ring's
        # first block), and reset never restores the KV caches (stale rows are masked)
        self.m.reset(state=not prefill or FULL_RESET)
        self.ring = self.slot = 0
        self.prev = None
        start = 0
        if prefill:
            # the prefill and its handoff (GDN states into the committed base, conv histories at ring 0, KV) of all but
            # the last token, and the drafter's context from the prefill's taps
            start = P - 1
            mark("start")
            self.pp.reset()
            self.pp.run(torch.tensor(ids[:start]))
            mark("prefill")
            self.pp.handoff(self.m, start)
            mark("handoff")
            self._fill_context(start)
            mark("fill")
        for p0 in range(start, P, R):
            k = min(R, P - p0)
            self._step(list(ids[p0 : p0 + k]) + [ids[p0 + k - 1]] * (R - k), p0)
            if self.tap_log is not None:  # (diagnostics: the taps of every prompt step)
                comp = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
                self.tap_log.append(
                    (p0, torch.stack([ttnn.to_torch(t, mesh_composer=comp).reshape(-1, H)[:k] for t in self.taps]))
                )
            if draft_context:
                self.drafter.extend_context(p0, k)
            g = self.m.argmax()
            self._commit(k - 1)
        self._last_row = k - 1
        return g[k - 1]

    def _fill_context(self, n_tokens, host=False):
        """the drafter's context (its window) from the last prefill's taps of positions < n_tokens, one fill of
        fill_rows rows from s0 = the window's first block. On device: each tap's rows masked to the chip that holds
        them, one reduce-scatter -> the taps dim-fractured as the drafter reads them. host: through the host
        (read_taps; the reference)"""
        s0 = max(0, n_tokens - self.window) // 64 * 64
        rows, nt = self.fill_rows, len(self.tap_masks)
        if host:
            taps, _ = self.pp.read_taps(n_tokens, self.window)
            full = torch.zeros(nt, rows, H)
            full[:, : taps.shape[1]] = taps.float()
            dev = tap_buffers(self.mesh, nt, rows, full)
        else:
            masked = []
            for x, m in zip(self.pp.tap_rows(s0, rows), self.tap_masks):
                masked.append(ttnn.mul(x, m))
                ttnn.deallocate(x)
            cat = ttnn.concat(masked, dim=2)  # [1, 1, taps x rows, H], each chip its taps' rows (others zero)
            for x in masked:
                ttnn.deallocate(x)
            dr = self.drafter
            fr = tt_all_reduce(  # (on a 1-D mesh: the reduce-scatter alone, [1, 1, taps x rows, H / n]; frees cat)
                cat,
                self.mesh,
                dr.tt_ccl,
                cluster_axis=0,
                dim=3,
                topology=dr.topo,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dev = [ttnn.slice(fr, (0, 0, t * rows, 0), (1, 1, (t + 1) * rows, fr.shape[-1])) for t in range(nt)]
            ttnn.deallocate(fr)
        self.drafter.fill_context(dev, s0, valid_len=n_tokens - s0)
        for t in dev:
            ttnn.deallocate(t)
        if FILL_SYNC:
            # every chip done with the fill before the verify step is enqueued: a resident step's hubs push their first
            # round into the peers' hub slots (CB 0, mlp_hub.cpp) as soon as they start, so a chip still running the
            # fill's last ops (their CBs at those L1 addresses) loses its peers' slots and corrupts its own fill. The
            # draft does the same before every later step (DFlash2DrafterTP.draft). 2026-10-30, whenever one chip lagged
            # (chip 1's AICLK throttle under sustained decode): first-step logits all 0 (first token 0, "!") and a
            # drafter context that accepted no draft for the whole request
            ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------ serving (one request at a time)
    def start(self, ids):
        """A request's prompt (as generate's): returns the logits [vocab] of the last prompt row. The first token is
        their argmax (first_token), the token the device continues from; next_block returns the tokens after it."""
        ids = [int(t) for t in ids]
        assert len(ids) <= self.max_prompt_tokens, f"prompt of {len(ids)} > max_prompt_tokens {self.max_prompt_tokens}"
        self.stop()  # (a request still open is dropped)
        self.first_token = int(self._prompt(ids))
        lg = self.m.logits()[self._last_row]
        self._p, self._t = len(ids), self.first_token
        return lg[:, : self.d.vocab_chip].reshape(-1)[: self.vocab].clone()

    def next_block(self, n, limit):
        """The next n tokens of the request, or fewer once a verify step would pass position min(limit, max_pos)
        (its rows run at p .. p + R - 1). Each iteration commits 1 .. R tokens; the ones past n wait for the next
        call."""
        end = min(int(limit), self.max_pos)
        while len(self._carry) < n and self._p + self.R <= end:
            p, t = self._p, self._t
            drafts = self.drafter.draft(t, p)
            self._step([t] + drafts, p)
            g = self.m.argmax()
            a = 0
            while a < self.R - 1 and drafts[a] == g[a]:
                a += 1
            self._carry += drafts[:a] + [g[a]]
            self._accepts.append(a)
            self.drafter.extend_context(p, a + 1)
            self._commit(a)
            self._p, self._t = p + a + 1, g[a]
        block, self._carry = self._carry[:n], self._carry[n:]
        return block

    def stop(self):
        """End the request: the tokens not yet returned are dropped (idempotent; safe before any start)"""
        self._carry, self._p, self._t, self.first_token, self._accepts = [], 0, None, None, []

    def stats(self):
        """the current request's speculation: verify iterations, tokens committed, mean drafts accepted"""
        n = len(self._accepts)
        return dict(iterations=n, committed=n + sum(self._accepts), mean_accept=sum(self._accepts) / n if n else 0.0)

    @property
    def max_prompt_tokens(self):
        """the longest prompt start() takes (room for a verify step and the drafter's block past it)"""
        return self.max_pos - 2 * self.R - 64

    def generate(self, ids, n_new, timing=False):
        """greedy spec decode: (n_new generated tokens, stats); timing: synchronize between the parts and time them"""
        ids = [int(t) for t in ids]
        P, R = len(ids), self.R
        assert P + n_new + 2 * R + 64 <= self.max_pos
        t0 = time.perf_counter()
        t = self._prompt(ids)
        t_first = time.perf_counter() - t0
        gen, accepts, parts = [t], [], dict(draft=[], verify=[], extend=[])
        p = P
        t1 = time.perf_counter()
        while len(gen) < n_new:
            ts = time.perf_counter()
            drafts = self.drafter.draft(t, p)
            tv = time.perf_counter()
            self._step([t] + drafts, p)
            g = self.m.argmax()
            te = time.perf_counter()
            a = 0
            while a < R - 1 and drafts[a] == g[a]:
                a += 1
            gen += drafts[:a] + [g[a]]
            accepts.append(a)
            if self.tap_log is not None:  # (diagnostics: the committed rows' taps, the drafts)
                comp = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
                self.tap_log.append(
                    (p, torch.stack([ttnn.to_torch(t, mesh_composer=comp).reshape(-1, H)[: a + 1] for t in self.taps]))
                )
                self.draft_log.append((p, t, list(drafts), a))
            self.drafter.extend_context(p, a + 1)
            if timing:
                ttnn.synchronize_device(self.mesh)
                parts["draft"].append(tv - ts)
                parts["verify"].append(te - tv)
                parts["extend"].append(time.perf_counter() - te)
            self._commit(a)
            p, t = p + a + 1, g[a]
        ttnn.synchronize_device(self.mesh)
        dt = time.perf_counter() - t1
        stats = dict(
            first_token_s=t_first,
            iterations=len(accepts),
            mean_accept=sum(accepts) / len(accepts),
            tokens_per_iter=(len(gen) - 1) / len(accepts),
            tok_s=(len(gen) - 1) / dt,
            ms_per_iter=1e3 * dt / len(accepts),
            accepts=accepts,
        )
        if timing:
            med = lambda xs: 1e3 * sorted(xs)[len(xs) // 2]
            stats.update({f"{k}_ms": med(v) for k, v in parts.items()})
        return gen[:n_new], stats

    def greedy(self, ids, n_new, other=None, prefill=False):
        """the verify model's own greedy tokens, one token per step (row 0; the reference spec decode reproduces).
        other (another decode's tokens): also returns, per generated position, this decode's logit of its own token
        minus its logit of the other's token (0 where they agree; the margin by which they differ).
        prefill: the prompt as generate() takes it (the pipeline-parallel prefill path), else through verify steps"""
        ids = [int(t) for t in ids]
        t = self._prompt(ids, draft_context=prefill)
        gen, p, gaps = [t], len(ids), [0.0]
        while len(gen) < n_new:
            self._step([t] * self.R, p)
            if self.tap_log is not None:  # (diagnostics: row 0's taps)
                comp = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
                self.tap_log.append(
                    (p, torch.stack([ttnn.to_torch(x, mesh_composer=comp).reshape(-1, H)[:1] for x in self.taps]))
                )
            t = self.m.argmax()[0]
            if other is not None and len(gen) < len(other) and other[len(gen)] != t:
                lg = self.m.logits()[0][:, : self.d.vocab_chip].reshape(-1)[: self.vocab]
                gaps.append(float(lg[t] - lg[other[len(gen)]]))
            else:
                gaps.append(0.0)
            gen.append(t)
            self._commit(0)
            p += 1
        return (gen, gaps) if other is not None else gen
