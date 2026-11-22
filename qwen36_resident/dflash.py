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
from models.demos.blackhole.qwen36.tt.dflash2 import H, resolve_weights_dir
from models.demos.blackhole.qwen36.tt.dflash2_tp import DFlash2DrafterTP
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
from qwen36_resident import weights as Q

# 1: restore the initial recurrent state before every prompt, also when the prefill handoff overwrites it (A/B)
FULL_RESET = os.environ.get("RESIDENT_PREFILL_FULL_RESET", "0") == "1"
# 0: no host synchronization between the drafter's context fill and the prompt's verify step (the first-step race, A/B
# only; see _fill_context)
FILL_SYNC = os.environ.get("RESIDENT_DFLASH_FILL_SYNC", "1") == "1"
# verify rows (ResidentDFlashRows): the drafter's KV as a per-user ring of window + 2 blocks (DFlash2 is all
# sliding-window: a key older than the window is never read), not max_pos per user (20 KB / token: 8 users x 256K =
# 42 GB). 0: each user's own max_pos blocks (the A/B reference)
DRAFTER_RING = os.environ.get("RESIDENT_DFLASH_RING", "1") == "1"


# The drafter checkpoint the resident engine was validated with, pinned to its commit: resolve_weights_dir finds
# snapshots/<rev> in the HF cache offline (HF_HUB_OFFLINE=1) without a refs/main entry, which a cache populated by
# `hf download --revision` or a copied snapshot lacks. DFLASH_WEIGHTS (a local dir or org/name[@rev]) overrides it.
DRAFTER_WEIGHTS = "incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"


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
    weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DRAFTER_WEIGHTS)
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
        weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DRAFTER_WEIGHTS)
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

    def _fill_context(self, n_tokens, host=False, phys=0):
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
        self.drafter.fill_context(dev, s0, phys=phys, valid_len=n_tokens - s0)  # (phys: the drafter slot, multi-user)
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


class ResidentDFlashRows:
    """Greedy DFlash2 speculative decode of U users in one launch (DB1 prototype, host driven): the target runs the
    verify rows of ResidentModel(verify=R, rows=True, groups=U) — row group u is user u's verify of R rows — and the
    drafter runs one U x R layout over U context slots (DFlash2DrafterTP.alloc with U page-table rows). The verify
    rows' hubs write user u's taps into rows u R .. of the tap tiles, the drafter's user-major rows. Per iteration:
    every user drafts R - 1 tokens from its pending token, the U users' rows verify in one step, each user accepts its
    longest matching prefix (the device's per-row greedy, fed_token), extends its context by its accepted rows and
    continues from its row a (ring + 1 + a, slot + a). Prompts go through the same verify steps (R tokens per step,
    every user's prompt the same length here)."""

    def __init__(
        self,
        mesh,
        ck,
        users=4,
        max_pos=4096,
        R=8,
        weights_dir=None,
        block_size=64,
        prefill=False,
        window=2048,
        kv_pool_blocks=None,
    ):
        from qwen36_resident.model import Dims, ResidentModel, State, is_attn

        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        self.mesh, self.R, self.U, self.vocab = mesh, R, users, cfg["vocab_size"]
        d = self.d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=self.vocab)
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        emb = Q.embedding(ck)
        self.emb = emb.float()
        weights_dir = weights_dir or os.environ.get("DFLASH_WEIGHTS", DRAFTER_WEIGHTS)
        from models.demos.blackhole.qwen36.tt.dflash2 import load_config

        tap_layers = list(load_config(weights_dir)["taps"])
        # (prefill: the pipeline-parallel prefill with the drafter's taps, as ResidentDFlash's: each user's prompt but
        # its last token prefilled, handed off into its first state slot, its drafter context filled from the taps)
        self.pp, self.window = None, window
        if prefill:
            from qwen36_resident.prefill.pp_prefill import PPPrefill

            C = max(ResidentDFlash.PREFILL_CHUNKS)
            max_len = -(-max_pos // C) * C
            max_pos = max_len + 64
            self.pp = PPPrefill(
                mesh,
                ck,
                layers,
                interval,
                chunk=list(ResidentDFlash.PREFILL_CHUNKS),
                max_len=max_len,
                taps=tap_layers,
                tap_window=window,
            )
            self.fill_rows = window + 64
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
        self.taps = tap_buffers(mesh, len(tap_layers), users * R)
        embed = ttnn.from_torch(
            emb.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        nb = -(-max_pos // block_size)
        sts = [State(d, layers - n_attn, n_attn, 0, max_pos, zero=True, kv=False) for _ in range(users)]
        # (kv_pool_blocks: a serving pool of the plugin's block count, every user on block 0 until its request's table
        # arrives — SpecRowsSession.start; else a pool of each user's own blocks)
        own = [list(range(u * nb, (u + 1) * nb)) for u in range(users)]
        self.m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            sts,
            layers,
            interval,
            lm_head=True,
            embed=embed,
            kv_pool_blocks=kv_pool_blocks or users * nb,
            block_tables=[[0]] * users if kv_pool_blocks else own,
            kv_block_size=block_size,
            rows=True,
            groups=users * R // 8,  # (8 rows per group: a user per group at R = 8, two at R = 4)
            verify=R,
            taps=list(zip(tap_layers, self.taps)),
        )
        self.drafter = build_drafter(mesh, ck, weights_dir, embed=emb, block=R)
        widest = max(self.drafter.windows) if all(self.drafter.windows) else 0
        if DRAFTER_RING and widest:  # (window + 2 blocks: a block row never evicts a key it attends; >= the fill)
            ring = -(-(max(widest, self.window if prefill else 0) + 2 * block_size) // block_size) * block_size
            nbd = ring // block_size
            self.drafter.alloc(
                torch.arange(users * nbd, dtype=torch.int32).reshape(users, nbd),
                block_size,
                self.taps,
                ring_tokens=ring,
            )
        else:
            self.drafter.alloc(torch.arange(users * nb, dtype=torch.int32).reshape(users, nb), block_size, self.taps)
        self.drafter_ring = self.drafter.ring
        self.max_pos = max_pos
        self._reset_ptrs()
        # every program once, eagerly, then the target's trace (the drafter's on their first armed calls)
        self._step([[0] * R] * users, [0] * users)
        self.greedy_rows()
        self.drafter.extend_context([0] * users, [0] * users, traced=False)
        self.drafter.draft([0] * users, [1] * users, traced=False)
        ttnn.synchronize_device(mesh)
        if self.pp is not None:  # (the prefill's programs before any trace: a fill, a capture, a handoff)
            self._fill_context(1, phys=0)
            self.pp.capture()
            self.pp.handoff(self.m, 1, slot=0, block_table=self.m.block_tables[0])
        self.m.reset()
        self.m.capture_trace()
        self.drafter.arm_traces()

    def _fill_context(self, n_tokens, phys=0):
        ResidentDFlash._fill_context(self, n_tokens, phys=phys)

    def _reset_ptrs(self):
        U = self.U
        self.ring, self.slot, self.prev = [0] * U, [0] * U, [None] * U

    def _step(self, toks, pos):
        """one verify step: toks[u] = user u's R tokens at pos[u] .. pos[u] + R - 1"""
        x0 = torch.stack([self.emb[list(t)] for t in toks])
        self.m.step(self.m.token_state_vrows(x0, pos, self.ring, self.slot, self.prev))

    def greedy_rows(self):
        """the device's greedy token of every row ([U][R], the hub's argmax)"""
        fed = self.m.fed_token()
        return [[fed[u * self.R + r][0] for r in range(self.R)] for u in range(self.U)]

    def _commit(self, accepts):
        R = self.R
        for u, a in enumerate(accepts):
            self.ring[u], self.slot[u], self.prev[u] = self.ring[u] + 1 + a, (self.slot[u] + a) % R, a

    def _prompt(self, prompts, draft_context=True):
        """the users' prompts (one length) through verify steps, R tokens each, all accepted: the first generated
        token per user"""
        P, R, U = len(prompts[0]), self.R, self.U
        assert all(len(p) == P for p in prompts), "the prototype takes prompts of one length"
        self.m.reset()
        self._reset_ptrs()
        start = 0
        if self.pp is not None and P > 1:
            # each user's prompt but its last token: PP prefill, handoff into the user's first state slot (u R), its
            # drafter context (slot u) from the prefill's taps; the last tokens then go through one verify step
            start = P - 1
            for u, ids in enumerate(prompts):
                self.pp.reset()
                self.pp.run(torch.as_tensor(ids[:start]))
                self.pp.handoff(self.m, start, slot=u * R, block_table=self.m.block_tables[u * R])
                if draft_context:
                    self._fill_context(start, phys=u)
        for p0 in range(start, P, R):
            k = min(R, P - p0)
            self._step([list(ids[p0 : p0 + k]) + [ids[p0 + k - 1]] * (R - k) for ids in prompts], [p0] * U)
            if draft_context:
                self.drafter.extend_context([p0] * U, [k] * U)
            g = self.greedy_rows()
            self._commit([k - 1] * U)
        return [g[u][k - 1] for u in range(U)]

    def generate(self, prompts, n_new):
        """greedy spec decode of the U users: (per user n_new tokens, stats)"""
        prompts = [[int(t) for t in p] for p in prompts]
        U, R, P = self.U, self.R, len(prompts[0])
        assert P + n_new + 2 * R + 64 <= self.max_pos
        t0 = time.perf_counter()
        t = self._prompt(prompts)
        t_first = time.perf_counter() - t0
        gen = [[t[u]] for u in range(U)]
        p, accepts, parts = [P] * U, [], dict(draft=[], verify=[], extend=[])
        t1 = time.perf_counter()
        while min(len(g) for g in gen) < n_new:
            ts = time.perf_counter()
            drafts = self.drafter.draft(list(t), list(p))
            tv = time.perf_counter()
            self._step([[t[u]] + drafts[u] for u in range(U)], p)
            g = self.greedy_rows()
            if getattr(self, "logit_log", None) is not None:  # (diagnostics: {(user, position): logits})
                lg = self.m.logits()[:, :, : self.d.vocab_chip].reshape(U * R, -1)[:, : self.vocab]
                for u in range(U):
                    for r in range(R):
                        self.logit_log[("spec", u, p[u] + r)] = (r, lg[u * R + r].float().clone())
            te = time.perf_counter()
            acc = []
            for u in range(U):
                a = 0
                while a < R - 1 and drafts[u][a] == g[u][a]:
                    a += 1
                gen[u] += drafts[u][:a] + [g[u][a]]
                acc.append(a)
            accepts.append(acc)
            self.drafter.extend_context(list(p), [a + 1 for a in acc])
            ttnn.synchronize_device(self.mesh)
            parts["draft"].append(tv - ts)
            parts["verify"].append(te - tv)
            parts["extend"].append(time.perf_counter() - te)
            self._commit(acc)
            p = [p[u] + acc[u] + 1 for u in range(U)]
            t = [g[u][acc[u]] for u in range(U)]
        dt = time.perf_counter() - t1
        flat = [a for acc in accepts for a in acc]
        med = lambda xs: 1e3 * sorted(xs)[len(xs) // 2]  # noqa: E731
        stats = dict(
            first_token_s=t_first,
            iterations=len(accepts),
            mean_accept=sum(flat) / len(flat),
            tokens_per_iter=sum(len(g) - 1 for g in gen) / len(accepts) / U,
            tok_s_per_user=sum(len(g) - 1 for g in gen) / U / dt,
            tok_s_total=sum(len(g) - 1 for g in gen) / dt,
            ms_per_iter=1e3 * dt / len(accepts),
            **{f"{k}_ms": med(v) for k, v in parts.items()},
        )
        return [g_[:n_new] for g_ in gen], stats

    def greedy(self, prompts, n_new):
        """the verify model's own greedy tokens of the U users, one token per step (row 0; spec decode reproduces it)"""
        prompts = [[int(t) for t in p] for p in prompts]
        U, R = self.U, self.R
        t = self._prompt(prompts, draft_context=False)
        gen, p = [[t[u]] for u in range(U)], [len(prompts[0])] * U
        while len(gen[0]) < n_new:
            self._step([[t[u]] * R for u in range(U)], p)
            g = self.greedy_rows()
            if getattr(self, "logit_log", None) is not None:
                lg = self.m.logits()[:, :, : self.d.vocab_chip].reshape(U * R, -1)[:, : self.vocab]
                for u in range(U):
                    self.logit_log[("greedy", u, p[u])] = lg[u * R].float().clone()
            t = [g[u][0] for u in range(U)]
            for u in range(U):
                gen[u].append(t[u])
            self._commit([0] * U)
            p = [x + 1 for x in p]
        return gen


class SpecRowsSession:
    """The runner-model spec contract's per-step API over a ResidentDFlashRows (DB3: the vLLM class maps the runner's
    calls onto these). Per user u (its rows u R .. of the verify launch, drafter slot u): start(u, ids) prefills a
    prompt (PP prefill + handoff + its drafter context, then its last token through a verify step with the others
    held) and returns that row's logits (the runner samples the first token from them); verify(tokens, positions,
    accepted, active) applies each active user's previous commit (accepted = tokens it committed: 1 after a start or
    a step that accepted no draft, up to R) and runs one verify step of [token, drafts] per user, returning the
    device greedy [U][R]; propose(anchors, positions) extends each user's drafter context by what its last verify
    committed (nothing right after a start) and drafts R - 1 tokens from its anchor at its position."""

    def __init__(self, eng):
        assert eng.pp is not None, "SpecRowsSession: the engine's PP prefill (ResidentDFlashRows(prefill=True))"
        self.e = eng
        U = eng.U
        self.pos = [0] * U  # the next position (row 0 of the user's next verify)
        self.started = [False] * U  # (a start pending its first propose: no verify taps to extend by)
        self.last = [None] * U  # (the last verify's row-0 position, for the propose's context extension)
        self.committed = [1] * U

    def start(self, u, ids, table=None):
        e, R, U = self.e, self.e.R, self.e.U
        ids = [int(t) for t in ids]
        P = len(ids)
        if table is not None:
            tbl = [list(t) for t in e.m.block_tables]
            for r in range(u * R, (u + 1) * R):
                tbl[r] = list(table)
            e.m.set_page_table(tbl)
        if P > 1:
            e.pp.reset()
            e.pp.run(torch.as_tensor(ids[: P - 1]))
            e.pp.handoff(e.m, P - 1, slot=u * R, block_table=e.m.block_tables[u * R])
            e._fill_context(P - 1, phys=u)
        e.ring[u], e.slot[u], e.prev[u] = 0, 0, None
        toks = [[0] * R for _ in range(U)]
        toks[u] = [ids[-1]] * R
        pos = list(self.pos)
        pos[u] = P - 1
        x0 = torch.stack([e.emb[list(t)] for t in toks])
        e.m.step(e.m.token_state_vrows(x0, pos, e.ring, e.slot, e.prev, active=[k == u for k in range(U)]))
        lg = e.m.logits()[u * R, :, : e.d.vocab_chip].reshape(-1)[: e.vocab].float().clone()
        n_valid = [0] * U
        n_valid[u] = 1
        e.drafter.extend_context([max(p, 0) for p in pos], n_valid)
        e.ring[u], e.slot[u], e.prev[u] = 1, 0, 0  # (the last prompt token committed as row 0)
        self.pos[u], self.started[u], self.committed[u] = P, True, 1
        return lg

    def verify(self, tokens, positions, active):
        """one verify step: per active user its committed token and R - 1 drafts (tokens[u]) at positions[u] (row 0);
        the commit its previous step left (recorded by propose) is applied first. Returns the device greedy [U][R]"""
        e, R, U = self.e, self.e.R, self.e.U
        for u in range(U):
            if not active[u]:
                continue
            if not self.started[u]:  # (the previous verify's commit: accepted drafts committed[u] - 1)
                a = self.committed[u] - 1
                e.ring[u], e.slot[u], e.prev[u] = e.ring[u] + 1 + a, (e.slot[u] + a) % R, a
            assert int(positions[u]) == self.pos[u], (u, int(positions[u]), self.pos[u])
        x0 = torch.stack([e.emb[[int(t) for t in tokens[u]]] for u in range(U)])
        e.m.step(e.m.token_state_vrows(x0, [int(p) for p in positions], e.ring, e.slot, e.prev, active=list(active)))
        g = e.greedy_rows()
        for u in range(U):
            if active[u]:
                self.last[u], self.started[u] = int(positions[u]), False
        return g

    def propose(self, anchors, positions, accepted, active):
        """after a commit (a start or a verify): a user that just verified records what it committed (accepted: 1 + the
        drafts accepted) and extends its drafter context by those rows' taps; then every user drafts R - 1 tokens from
        its anchor (its last committed token) at its position. Returns [U][R - 1]"""
        e, U = self.e, self.e.U
        n_valid = [0] * U
        for u in range(U):
            if active[u] and not self.started[u] and self.last[u] is not None:
                self.committed[u] = int(accepted[u])
                self.pos[u] = self.last[u] + self.committed[u]
                n_valid[u] = self.committed[u]
        slot0 = [max(self.last[u] if self.last[u] is not None else 0, 0) for u in range(U)]
        e.drafter.extend_context(slot0, n_valid)
        return e.drafter.draft([int(a) for a in anchors], [int(p) for p in positions])
