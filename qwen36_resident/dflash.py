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
from models.tt_transformers.tt.ccl import TT_CCL
from qwen36_resident import weights as Q


def ccl_topology():
    """Ring on the boxes whose chips form one (ModelArgs.ccl_topology), else Linear"""
    ring = (ttnn.cluster.ClusterType.P300_X2, ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P150_X8)
    return ttnn.Topology.Ring if ttnn.cluster.get_cluster_type() in ring else ttnn.Topology.Linear


def cache_root():
    root = os.environ.get("TT_CACHE_PATH")
    return f"{root}/resident_dflash" if root else None


def lm_head_vshard(mesh, ck, cache_dir=None):
    """the checkpoint's lm_head as [H, V / n] bf8 per chip (vocab-sharded, no norm folded)"""
    name = f"{cache_dir}/lm_head.vshard" if cache_dir else None
    w = ck.get("lm_head.weight")  # [V, H]
    return ttnn.as_tensor(
        w,
        preprocess=lambda t: t.T.contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=name,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def build_drafter(mesh, ck, weights_dir=None, embed=None, block=None):
    """DFlash2DrafterTP on `mesh` for the target checkpoint ck; weights cached under $TT_CACHE_PATH/resident_dflash"""
    weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)
    real = os.path.realpath(resolve_weights_dir(weights_dir))
    root = cache_root()
    if root:
        os.makedirs(root, exist_ok=True)
    cache = f"{root}/{os.path.basename(real)}-{hashlib.sha1(real.encode()).hexdigest()[:10]}" if root else None
    emb = (Q.embedding(ck) if embed is None else embed).to(torch.bfloat16)
    lm = lm_head_vshard(mesh, ck, root)
    drafter = DFlash2DrafterTP(
        mesh,
        weights_dir,
        emb,
        TT_CCL(mesh),
        ccl_topology(),
        lm,
        lm_vocab_sharded=True,
        block=block,
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
    through the same verify steps, R tokens per step."""

    def __init__(self, mesh, ck, max_pos=4096, R=8, layers=None, weights_dir=None):
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
        # every program once, eagerly, before any trace is captured; then the target's trace (the drafter's are
        # captured on their first armed call)
        self._step([0] * R, 0)
        self.m.argmax()
        self.drafter.extend_context(0, 0, traced=False)
        self.drafter.draft(0, 1, traced=False)
        ttnn.synchronize_device(mesh)
        self.m.reset()
        self.m.capture_trace()
        self.drafter.arm_traces()

    def _step(self, toks, p):
        self.m.step(
            self.m.token_state(self.emb[list(toks)], list(range(p, p + self.R)), ring=self.ring, slot=self.slot)
        )

    def _commit(self, a):
        self.ring, self.slot = self.ring + 1 + a, (self.slot + a) % self.R

    def _prompt(self, ids, draft_context=True):
        """the prompt through verify steps (R tokens each, all accepted): returns the first generated token"""
        self.m.reset()
        self.ring = self.slot = 0
        P, R = len(ids), self.R
        for p0 in range(0, P, R):
            k = min(R, P - p0)
            self._step(list(ids[p0 : p0 + k]) + [ids[p0 + k - 1]] * (R - k), p0)
            if draft_context:
                self.drafter.extend_context(p0, k)
            g = self.m.argmax()
            self._commit(k - 1)
        return g[k - 1]

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

    def greedy(self, ids, n_new):
        """the verify model's own greedy tokens, one token per step (row 0; the reference spec decode reproduces)"""
        ids = [int(t) for t in ids]
        t = self._prompt(ids, draft_context=False)
        gen, p = [t], len(ids)
        while len(gen) < n_new:
            self._step([t] * self.R, p)
            t = self.m.argmax()[0]
            gen.append(t)
            self._commit(0)
            p += 1
        return gen
