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
from models.demos.blackhole.mimo_v26.tt.attention import decode_pos_io, decode_pos_set
from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm
from models.demos.blackhole.mimo_v26.tt.moe_prefill import NUM_ROUTED, TOPK, spread_pad_routing

HIDDEN, SLOTS, T64 = 4096, 32, 64


class DeadRouting:
    """Replaces the router output of dead slots and of the 32 zero pad rows (decode MoE runs on 64 tokens per row) by a round-robin spread over all experts
    (spread_pad_routing), selected per row by a persistent mask updated each step."""

    def __init__(self, mesh):
        self.mesh = mesh
        rows = mesh.shape[0]
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.rowmap = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))
        idx, w = spread_pad_routing(1, T64, TOPK, NUM_ROUTED)
        mk = lambda t, dt: ttnn.from_torch(
            t, device=mesh, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        self.spread_idx = mk(idx.reshape(1, 1, T64, TOPK).to(torch.int32), ttnn.int32)
        self.spread_w = mk(w.reshape(1, 1, T64, TOPK).bfloat16(), ttnn.bfloat16)
        self.mask = ttnn.from_torch(
            torch.ones(rows, 1, T64, TOPK).bfloat16(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.rowmap,
        )

    def set(self, dead):
        """dead: bool [rows, 32] (True = dead/finished/no user)."""
        rows = dead.shape[0]
        m = torch.ones(rows, 1, T64, TOPK)
        m[:, 0, : dead.shape[1]] = dead.float().unsqueeze(-1)  # 1 where dead
        host = ttnn.from_torch(m.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rowmap)
        ttnn.copy_host_to_device_tensor(host, self.mask)

    def override(self, route, L=None):
        """route = (weights [T,1,1,k] bf16 RM, indices [T,1,1,k] uint16 RM) as returned by MiMoGate.route -> same format with dead rows replaced."""
        wts, idx = route
        T, k = wts.shape[0], wts.shape[3]
        assert T == T64 and k == TOPK
        i32 = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, (1, 1, T, k)), ttnn.TILE_LAYOUT), ttnn.int32)
        w16 = ttnn.to_layout(ttnn.reshape(wts, (1, 1, T, k)), ttnn.TILE_LAYOUT)
        ni = ttnn.where(self.mask, self.spread_idx, i32)
        nw = ttnn.where(self.mask, self.spread_w, w16)
        for t in (i32, w16, wts, idx):
            ttnn.deallocate(t)
        new_idx = ttnn.view(ttnn.to_layout(ttnn.typecast(ni, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k))
        new_w = ttnn.view(ttnn.to_layout(nw, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k))
        ttnn.deallocate(ni)
        ttnn.deallocate(nw)
        return new_w, new_idx


class Decoder:
    def __init__(self, model, trace=True, keep_logits=False, spread=True):
        self.m, self.mesh, self.keep_logits, self.use_trace = model, model.mesh, keep_logits, trace
        self.rows = self.mesh.shape[0]
        self.rowmap = ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None))
        self.pos = decode_pos_io(self.mesh)
        self.tok = ttnn.from_torch(
            torch.zeros(self.rows, SLOTS, dtype=torch.int32),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.rowmap,
        )
        self.dead = DeadRouting(self.mesh) if spread else None
        self.vocab_local = model.lm_head.shape[-1]
        self._tr = None
        self.out = None
        self.steps = 0

    # ------------------------------------------------------------------ one step (device)
    def _forward(self):
        m = self.m
        e = ttnn.embedding(self.tok, m.embed, layout=ttnn.TILE_LAYOUT)  # [1,32,4096] per device
        h = ttnn.reshape(e, (1, 1, SLOTS, HIDDEN))
        for lyr in m.layers:
            h2 = lyr.decode(h, self.pos["cur"], self.pos["idx"], dead=self.dead)
            ttnn.deallocate(h)
            h = h2
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
        vals = ttnn.max(lg, dim=3, keepdim=True)
        idx = ttnn.argmax(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
        if not self.keep_logits:
            ttnn.deallocate(lg)
            lg = None
        return vals, idx, lg

    # ------------------------------------------------------------------ host interface
    def set_inputs(self, tokens, positions, dead):
        """tokens int [rows,32]; positions int [rows,32] (position of the token being fed); dead bool [rows,32]. Dead slots are forced to token 0 / position 0."""
        tokens = torch.where(dead, torch.zeros_like(tokens), tokens)
        positions = torch.where(dead, torch.zeros_like(positions), positions)
        host = ttnn.from_torch(
            tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rowmap
        )
        ttnn.copy_host_to_device_tensor(host, self.tok)
        decode_pos_set(self.mesh, self.pos, positions)
        if self.dead is not None:
            self.dead.set(dead)

    def run(self):
        """Run one step with the inputs currently on device; returns (next token ids [rows,32], max logit [rows,32][, logits [rows,32,V]])."""
        if self.use_trace:
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
            ttnn.execute_trace(self.mesh, self._tr, cq_id=0, blocking=True)
            vals, idx, lg = self.out
        else:
            vals, idx, lg = self._forward()
        self.steps += 1
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

    def release(self):
        if self._tr is not None:
            ttnn.release_trace(self.mesh, self._tr)
            self._tr = None
