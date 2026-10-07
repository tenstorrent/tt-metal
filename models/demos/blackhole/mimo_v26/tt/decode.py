# SPDX-License-Identifier: Apache-2.0
"""Decode engine for MiMo-V2.6-Flash on the 4x8 mesh: one decode step for 32 user slots per mesh row (up to 128 users), per-row and per-slot positions, dead-slot
MoE routing spread, final norm + vocab-sharded lm_head + per-device argmax candidates, optionally captured in a trace (capture once, replay).

Inputs per step (all device-resident persistent tensors updated with copy_host_to_device_tensor, so the trace never sees host allocations):
  tok  uint32 [1,32] per device (row r's 32 slot tokens), cur int32 [32] / idx uint32 [1,32] positions of the token being fed, dead mask for the MoE.
Outputs: per-device local (max logit, argmax) of the lm_head shard -> host picks the global argmax over the 8 vocab shards (lowest vocab id on ties = HF argmax).
"""
import os

import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt import verify_dev as vd
from models.demos.blackhole.mimo_v26.tt.attention import decode_pos_io, decode_pos_set
from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm, rms_decode_sharded

HIDDEN, SLOTS, T64 = 4096, 32, 64


def global_argmax(lg, tok_out, vocab_local):
    """pick + write into tok_out (see global_pick)"""
    pick = global_pick(lg, vocab_local)
    nxt = ttnn.reshape(ttnn.to_layout(ttnn.typecast(pick, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, lg.shape[2]))
    ttnn.copy(nxt, tok_out)
    return tok_out


def global_pick(lg, vocab_local):
    """Greedy token of every row from the vocab-sharded logits lg [1,1,U,V/8] (bf16 tile, one shard per mesh column), all on device -> float32 tile [1,1,U,1]
    (ids < 2^24 are exact). Ties between columns go to the lowest column = lowest vocab id (same rule as the host path).
    """
    U = lg.shape[2]
    vals = ttnn.max(lg, dim=3, keepdim=True)  # [1,1,U,1]
    idx = ttnn.argmax(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)  # uint32 [1,1,U,1] local ids
    # global id = local + column * V/8 : the column offset is a per-device constant tensor
    gidx = ttnn.add(
        ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32), _col_offset(lg.device(), U, vocab_local)
    )
    vpad = ttnn.pad(vals, [(0, 0), (0, 0), (0, 0), (0, 31)], 0.0)  # [1,1,U,32]
    gpad = ttnn.pad(gidx, [(0, 0), (0, 0), (0, 0), (0, 31)], 0.0)
    gv = ttnn.all_gather(vpad, dim=3, cluster_axis=1)  # [1,1,U,256]: column c occupies [32c, 32c+32), value at 32c
    gi = ttnn.all_gather(gpad, dim=3, cluster_axis=1)
    cv = ttnn.slice(gv, [0, 0, 0, 0], [1, 1, U, 256], [1, 1, 1, 32])  # [1,1,U,8]
    ci = ttnn.slice(gi, [0, 0, 0, 0], [1, 1, U, 256], [1, 1, 1, 32])
    m = ttnn.max(cv, dim=3, keepdim=True)
    hit = ttnn.typecast(ttnn.eq(cv, m), ttnn.float32)  # 1 where the column holds the max
    # first hit wins: weight columns 8..1 and take the argmax of hit*weight
    w = _col_weights(lg.device())
    sc = ttnn.multiply(hit, w)
    mx = ttnn.max(sc, dim=3, keepdim=True)
    onehot = ttnn.typecast(ttnn.eq(sc, mx), ttnn.float32)
    onehot = ttnn.multiply(onehot, hit)
    return ttnn.sum(ttnn.multiply(onehot, ci), dim=3, keepdim=True)  # [1,1,U,1] float32 global id (tile)


_COL_OFF = {}
_COL_W = {}


def _col_weights(mesh):
    if id(mesh) not in _COL_W:  # device constants must exist before a trace capture (no host writes while capturing)
        _COL_W[id(mesh)] = ttnn.from_torch(
            torch.arange(8, 0, -1, dtype=torch.float32).reshape(1, 1, 1, 8),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    return _COL_W[id(mesh)]


def _col_offset(mesh, U, vocab_local):
    """float32 tile [1,1,U,1] per device = mesh column * vocab_local"""
    key = (id(mesh), U, vocab_local)
    if key not in _COL_OFF:
        rows, cols = mesh.shape
        t = torch.zeros(rows, cols, 1, U, 1)
        for c in range(cols):
            t[:, c] = float(c * vocab_local)
        _COL_OFF[key] = ttnn.from_torch(
            t.reshape(rows, cols, U, 1),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1)),
        )
        _COL_OFF[key] = ttnn.reshape(_COL_OFF[key], (1, 1, U, 1))
    return _COL_OFF[key]


class LiveMask:
    """Persistent live-slot mask for tt/moe_decode.MiMoMoEDecode: bf16 tile [rows,1,64,1] sharded over mesh rows, 1.0 = live slot (slots 32..63 are pad rows, always 0).
    Dead slots get their routing spread round-robin over all experts (they would otherwise all pick the same experts and overflow a chip's dispatch buffer).
    """

    def __init__(self, mesh):
        self.mesh = mesh
        rows = mesh.shape[0]
        self.rowmap = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))
        self.t = ttnn.from_torch(
            torch.zeros(rows, 1, T64, 1).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.rowmap,
        )

    def set(self, alive):
        """alive: bool [rows, 32]."""
        m = torch.zeros(alive.shape[0], 1, T64, 1)
        m[:, 0, : alive.shape[1], 0] = alive.float()
        host = ttnn.from_torch(m.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rowmap)
        ttnn.copy_host_to_device_tensor(host, self.t)
        return host


class Decoder:
    def __init__(
        self,
        model,
        trace=True,
        keep_logits=False,
        spread=True,
        U=SLOTS,
        block=1,
        capture=None,
        feedback=False,
        max_pos=None,
        eos=(),
        chunk=16,
        vdev=False,
    ):
        """U = real decode users per mesh row. block = tokens per user per step: 1 = plain decode; 8 = spec-decode VERIFY with `block` virtual users per real user
        (row v = block*u + t feeds token t of user u at position pos_u + t; U*block <= 32). Every decode tensor has B = U*block rows.
        capture: layer indices whose outputs are kept as an extra trace output `feat` [1,1,B,len(capture)*4096] (bf16, on device; layer 47 = before the final norm).
        """
        self.m, self.mesh, self.keep_logits, self.use_trace, self.U = model, model.mesh, keep_logits, trace, U
        self.block, self.B, self.capture = block, U * block, tuple(capture) if capture else ()
        assert self.B <= SLOTS, "U*block must be <= 32 rows per mesh row"
        self.feat = None
        self.upd = None
        # feedback: the step samples the next tokens on device (global_argmax), writes them into the token input in place and advances cur/idx by one, so consecutive steps need no
        # host input copies and only the [rows,U] token tensor is read back; the host refreshes the inputs (set_inputs) only when a user finishes.
        self.feedback = bool(feedback) and block == 1
        self._last = None
        self.rows = self.mesh.shape[0]
        self.rowmap = ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None))
        self.pos = decode_pos_io(self.mesh, self.B)
        self.tok = ttnn.from_torch(
            torch.zeros(self.rows, self.B, dtype=torch.int32),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.rowmap,
        )
        self.live = LiveMask(self.mesh) if spread else None
        self.upd = None
        self.upd_mode = os.environ.get("MIMO_VERIFY_WRITE", "skip")
        if block > 1 and self.upd_mode in ("seq", "skip"):
            n_upd = U if self.upd_mode == "seq" else self.B
            self.upd = [self._persist_u(torch.zeros(self.rows, n_upd, dtype=torch.int32)) for _ in range(block)]
        self.vocab_local = model.lm_head.shape[-1]
        self.max_pos, self.eos, self.N = max_pos if max_pos is not None else 1 << 20, tuple(sorted(eos)), chunk
        self.c0 = 0  # host mirror of the device step counter (ring write index)
        self.vdev = bool(vdev) and block > 1
        if self.vdev:
            assert (
                self.upd_mode == "skip" and self.upd is not None
            ), "device-resident verify needs MIMO_VERIFY_WRITE=skip"
            vd.init_state(self)
        if self.feedback:
            self._init_fb_state()
        self._tr = None
        self.out = None
        self.steps = 0

    def _init_fb_state(self):
        # device-resident decode state (float32 tiles, row sharded): alive flag, position of the token being fed, and a [N,B] ring of sampled tokens written at the
        # device step counter, so the host only enqueues trace replays and reads the ring once per chunk of N steps
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        mk = lambda t, mm=rep: ttnn.from_torch(
            t, device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mm
        )
        self.alive = mk(torch.zeros(self.rows, 1, self.B, 1), self.rowmap)
        self.posf = mk(torch.zeros(self.rows, 1, self.B, 1), self.rowmap)
        self.cnt = mk(torch.zeros(1, 1, 1, 1))
        self.iota = mk(torch.arange(self.N, dtype=torch.float32).reshape(1, 1, self.N, 1))
        self.ring = mk(torch.zeros(1, 1, self.N, self.B))

    def set_feedback(self, flag):
        """switch between host-fed and device-resident decode (drops the trace; the next run re-captures)"""
        assert self.block == 1
        self.release()
        self.feedback = bool(flag)
        if flag and not hasattr(self, "alive"):
            self._init_fb_state()

    def _persist_u(self, t):
        return ttnn.from_torch(
            t.reshape(-1), device=self.mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rowmap
        )

    # ------------------------------------------------------------------ one step (device)
    def _forward(self):
        m = self.m
        if self.vdev:
            vd.prologue(self)
        e = ttnn.embedding(self.tok, m.embed, layout=ttnn.TILE_LAYOUT)  # [1,32,4096] per device
        h = ttnn.reshape(e, (1, 1, self.B, HIDDEN))
        caps = []
        ropes = {}
        if os.environ.get("MIMO_ROPE_HOIST", "1") == "1":
            for lyr in m.layers:  # cos/sin of the step: one set per layer kind (SWA theta 1e4, global theta 1e7)
                if lyr.swa not in ropes:
                    ropes[lyr.swa] = lyr.attn.rope_prep(self.pos["idx"], self.B)
        for li, lyr in enumerate(m.layers):
            h2 = lyr.decode(
                h,
                self.pos["cur"],
                self.pos["idx"],
                live=self.live.t if self.live is not None else None,
                block=self.block,
                upd=self.upd,
                rope=ropes.get(lyr.swa),
            )
            ttnn.deallocate(h)
            h = h2
            if li in self.capture:
                caps.append(ttnn.clone(h))
        for r_ in ropes.values():
            for t_ in r_:
                ttnn.deallocate(t_)
        feat = None
        if caps:
            feat = ttnn.concat(caps, dim=3) if len(caps) > 1 else caps[0]
            for c in caps if len(caps) > 1 else []:
                ttnn.deallocate(c)
        if os.environ.get("MIMO_NORM", "sharded") == "sharded" and h.shape[2] <= 32:
            x = rms_decode_sharded(h, m.w_norm)
        else:
            x = ttnn.rms_norm(h, epsilon=EPS, weight=m.w_norm, compute_kernel_config=_k_norm())
        ttnn.deallocate(h)
        lg = ttnn.linear(
            x,
            m.lm_head,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        )
        ttnn.deallocate(x)
        if self.vdev:
            vd.epilogue(self, global_pick(lg, self.vocab_local))
            if not self.keep_logits:
                ttnn.deallocate(lg)
                lg = None
            return self.tok, None, lg, feat
        if self.feedback:
            self._device_update(global_pick(lg, self.vocab_local))
            if not self.keep_logits:
                ttnn.deallocate(lg)
                lg = None
            return self.tok, None, lg, feat
        vals = ttnn.max(lg, dim=3, keepdim=True)
        idx = ttnn.argmax(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
        if not self.keep_logits:
            ttnn.deallocate(lg)
            lg = None
        return vals, idx, lg, feat

    def _device_update(self, pick):
        """End of a feedback step, all on device: record the sampled tokens in the ring at the step counter, update the alive flag (eos / end of context), advance the
        positions of live users, and write next step's token / cur / idx / live-mask inputs in place."""
        F, RM, B = ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.B
        f = lambda t: ttnn.typecast(t, F)
        alive, pos = self.alive, self.posf
        # ring[cnt] = token of users that were alive at the start of the step
        tokrow = ttnn.transpose(ttnn.multiply(pick, alive), -2, -1)  # [1,1,1,B]
        oh = f(ttnn.eq(self.iota, self.cnt))  # [1,1,N,1]
        ring = ttnn.add(ttnn.multiply(self.ring, ttnn.subtract(1.0, oh)), ttnn.multiply(oh, tokrow))
        ttnn.copy(ring, self.ring)
        c1 = ttnn.add(self.cnt, 1.0)
        ttnn.copy(ttnn.multiply(c1, f(ttnn.lt(c1, float(self.N)))), self.cnt)
        # alive / positions
        hit = None
        for e in self.eos:
            x = f(ttnn.eq(pick, float(e)))
            hit = x if hit is None else ttnn.add(hit, x)
        a_new = alive
        if hit is not None:
            a_new = ttnn.multiply(a_new, f(ttnn.eq(hit, 0.0)))
        p1 = ttnn.add(pos, 1.0)
        a_new = ttnn.multiply(a_new, f(ttnn.lt(p1, float(self.max_pos - 1))))
        posn = ttnn.multiply(p1, a_new)
        tokn = ttnn.multiply(pick, a_new)
        ttnn.copy(a_new, self.alive)
        ttnn.copy(posn, self.posf)
        rm = lambda t, dt, shp: ttnn.reshape(ttnn.typecast(ttnn.to_layout(t, RM), dt), shp)
        ttnn.copy(rm(posn, ttnn.int32, (B,)), self.pos["cur"])
        ttnn.copy(rm(posn, ttnn.uint32, (1, B)), self.pos["idx"])
        ttnn.copy(rm(tokn, ttnn.uint32, (1, B)), self.tok)
        if self.live is not None:
            ttnn.copy(
                ttnn.typecast(ttnn.pad(a_new, [(0, 0), (0, 0), (0, T64 - B), (0, 0)], 0.0), ttnn.bfloat16), self.live.t
            )

    # ------------------------------------------------------------------ host interface
    def set_inputs(self, tokens, positions, dead):
        """tokens int [rows,32]; positions int [rows,32] (position of the token being fed); dead bool [rows,32]. Dead slots are forced to token 0 / position 0."""
        self._last = (tokens, positions, dead)
        if (
            self.block > 1
        ):  # tokens [rows,U,block], positions [rows,U] = position of token 0 of each block, dead [rows,U]
            assert tokens.dim() == 3 and tokens.shape[2] == self.block
            positions = (positions.unsqueeze(-1) + torch.arange(self.block)).reshape(tokens.shape[0], -1)
            tokens = tokens.reshape(tokens.shape[0], -1)
            dead = dead.repeat_interleave(self.block, dim=1)
        tokens = torch.where(dead, torch.zeros_like(tokens), tokens)
        positions = torch.where(dead, torch.zeros_like(positions), positions)
        upd_hosts = []
        if self.upd is not None:
            for t in range(self.block):
                if self.upd_mode == "skip":
                    ut = torch.full_like(positions, -1)
                    ut[:, t :: self.block] = positions[:, t :: self.block]
                else:
                    ut = positions[:, t :: self.block]
                h = ttnn.from_torch(
                    ut.to(torch.int32).contiguous().reshape(-1),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.rowmap,
                )
                ttnn.copy_host_to_device_tensor(h, self.upd[t])
                upd_hosts.append(h)
        host = ttnn.from_torch(
            tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rowmap
        )
        ttnn.copy_host_to_device_tensor(host, self.tok)
        keep = [host] + upd_hosts + decode_pos_set(self.mesh, self.pos, positions)
        if self.live is not None:
            keep.append(self.live.set(~dead))
        if self.feedback:
            rows = tokens.shape[0]
            for t_, dev in (((~dead).float(), self.alive), (positions.float(), self.posf)):
                h = ttnn.from_torch(
                    t_.reshape(rows, 1, self.B, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=self.rowmap,
                )
                ttnn.copy_host_to_device_tensor(h, dev)
                keep.append(h)
            hz = ttnn.from_torch(
                torch.zeros(1, 1, 1, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ttnn.copy_host_to_device_tensor(hz, self.cnt)
            keep.append(hz)
            self.c0 = 0
        # host->device copies are asynchronous: keep the host tensors alive and wait until they landed (freed host memory read by the device = row-wide
        # random glitches, observed as run-to-run nondeterminism; see LOG)
        ttnn.synchronize_device(self.mesh)
        self._keep = keep

    def _ensure_trace(self):
        if self._tr is None:
            chk = os.environ.get("MIMO_CHECK_CAP")
            self._forward()  # eager warmup/compile (capacity check on)
            ttnn.synchronize_device(self.mesh)
            os.environ["MIMO_CHECK_CAP"] = "0"
            self._tr = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.out = self._forward()
            ttnn.end_trace_capture(self.mesh, self._tr, cq_id=0)
            if chk is None:
                os.environ.pop("MIMO_CHECK_CAP")
            else:
                os.environ["MIMO_CHECK_CAP"] = chk
            ttnn.synchronize_device(self.mesh)
            if self.vdev and getattr(self, "_vlast", None) is not None:
                vd.set_state(self, *self._vlast)  # the eager warm-up round advanced the state: restore it
            if self.feedback and self._last is not None:
                self.set_inputs(
                    *self._last
                )  # the eager warm-up step advanced tokens/positions in place: restore the inputs of this step

    def run(self):
        """Run one step with the inputs currently on device; returns (next token ids [rows,32], max logit [rows,32][, logits [rows,32,V]])."""
        if self.use_trace:
            self._ensure_trace()
            ttnn.execute_trace(self.mesh, self._tr, cq_id=0, blocking=True)
            vals, idx, lg, self.feat = self.out
        else:
            vals, idx, lg, feat = self._forward()
            if self.feat is not None:
                ttnn.deallocate(self.feat)
            self.feat = feat
        self.steps += 1
        if self.feedback:  # only the sampled tokens: ring row of this step, read from the column-0 device of every row
            tok = self._read_ring(1)[0]
            out = (tok, None)
            if self.keep_logits:
                out += (
                    ttnn.to_torch(
                        lg,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape)
                        ),
                    ).float()[:, 0],
                )
            return out
        comp = ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape))
        v = ttnn.to_torch(vals, mesh_composer=comp).float()[:, 0, :, :]  # [rows,32,cols]
        ix = ttnn.to_torch(idx, mesh_composer=comp)[:, 0, :, :].long()  # local ids [rows,32,cols]
        cols = v.shape[-1]
        glob = ix + torch.arange(cols).reshape(1, 1, -1) * self.vocab_local
        best = v.max(-1).values
        cand = torch.where(v == best.unsqueeze(-1), glob, torch.full_like(glob, 1 << 30))
        tok = cand.min(-1).values  # lowest vocab id among tied maxima
        out = (tok, best)
        if self.keep_logits:
            out += (
                ttnn.to_torch(
                    lg,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh, dims=(0, 3), mesh_shape=tuple(self.mesh.shape)),
                ).float()[:, 0],
            )
        if not self.use_trace:
            for t in (vals, idx) + ((lg,) if lg is not None else ()):
                ttnn.deallocate(t)
        return out

    @staticmethod
    def accept_prefix(argmax, tokens):
        """Greedy spec-decode acceptance. tokens [rows,U,block] = [last accepted token, draft_1..draft_{block-1}], argmax [rows,U,block] = target greedy token after each block
        position (from verify). Draft j is accepted iff argmax[j-1] == tokens[j] for all earlier drafts too. Returns n_acc [rows,U] (0..block-1); the tokens to emit this round
        are argmax[..., :n_acc+1] (the accepted drafts equal argmax[..., :n_acc], the last one is the bonus token) and the next block starts with argmax[..., n_acc].
        """
        ok = argmax[..., :-1] == tokens[..., 1:]
        return torch.cumprod(ok.long(), dim=-1).sum(-1)

    def verify(self, tokens, positions, dead):
        """Spec-decode verify: tokens [rows,U,block] (last accepted token + drafts), positions [rows,U] of token 0, dead [rows,U]. One traced step over U*block virtual users.
        Returns (argmax [rows,U,block] long: the target's greedy next token after each block position, max logit [rows,U,block]); self.feat holds the captured hidden states.
        """
        assert self.block > 1
        self.set_inputs(tokens, positions, dead)
        out = self.run()
        rows = tokens.shape[0]
        return out[0].reshape(rows, self.U, self.block), out[1].reshape(rows, self.U, self.block)

    def set_verify_state(self, anchor, pos, alive):
        """device-resident verify: initial anchor token / position of token 0 / alive, long/bool [rows,U] (host write once per generation)"""
        assert self.vdev
        vd.set_state(self, anchor, pos, alive)
        if self.use_trace:
            self._ensure_trace()  # warm-up + capture now (restores this state afterwards) so that drafts written next are not wiped by the restore

    def write_drafts(self, drafts):
        """blk_in[...,1:8] = drafts (device tensor [..,U,7]); device ops only (usable inside the spec agent's own traces)"""
        vd.write_drafts(self, drafts)

    def verify_enqueue(self, n=1):
        """enqueue n verify rounds (non-blocking trace replays; state, accept_prefix and the round ring are all on device); no host sync"""
        assert self.vdev and self.use_trace
        self._ensure_trace()
        for _ in range(n):
            ttnn.execute_trace(self.mesh, self._tr, cq_id=0, blocking=False)
        self.steps += n

    def read_rounds(self, n):
        return vd.read_rounds(self, n)

    def _read_ring(self, n):
        """tokens sampled by the last n steps (ring rows c0..c0+n, wraps) -> long [n, rows, B]; one blocking read per mesh row (column-0 device)."""
        dts = ttnn.get_device_tensors(self.ring)
        cols = self.mesh.shape[1]
        ring = torch.stack([ttnn.to_torch(dts[r * cols]).reshape(self.N, -1)[:, : self.B] for r in range(self.rows)], 1)
        idx = [(self.c0 + j) % self.N for j in range(n)]
        self.c0 = (self.c0 + n) % self.N
        return ring[idx].round().long()

    def run_chunk(self, n):
        """n <= N decode steps with no host work inside: n non-blocking trace replays on device-resident state (tokens, positions, alive mask, step counter), then ONE bulk
        ring read. Returns tokens long [n, rows, B] (0 for users that were already dead at that step; a user's eos token is included).
        """
        assert self.feedback and self.use_trace and 0 < n <= self.N
        self._ensure_trace()
        for _ in range(n):
            ttnn.execute_trace(self.mesh, self._tr, cq_id=0, blocking=False)
        self.steps += n
        return self._read_ring(n)

    def release(self):
        if self._tr is not None:
            ttnn.release_trace(self.mesh, self._tr)
            self._tr = None
        if self.out is not None:
            for t in self.out:
                if t is not None and t is not self.tok:  # feedback: tok is a persistent input, not a trace output
                    ttnn.deallocate(t)
            self.out = None

    def close(self):
        """release the trace and every U-dependent device buffer (tokens, positions, live mask)"""
        self.release()
        for n_ in ("alive", "posf", "cnt", "iota", "ring") + (vd.STATE_NAMES if self.vdev else ()):
            if hasattr(self, n_):
                ttnn.deallocate(getattr(self, n_))
        for t in (
            (self.tok, self.pos["cur"], self.pos["idx"])
            + tuple(self.upd or ())
            + ((self.live.t,) if self.live is not None else ())
        ):
            ttnn.deallocate(t)
