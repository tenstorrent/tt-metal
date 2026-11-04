# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b TP (PRODUCTION_PATH.md, doc/ttnn/07b-mla-megakernel/shard_contract.md): the sparse_dsa attention half as
ONE program per chip of the mesh, following Stage 14's tt/tp8_layer.py:

  every chip runs MkMla on its head shard (MeshHeadShard: heads [H_r r, H_r (r + 1)), H_r = 64 / TP = 8 at TP8, with
  q_b's columns, W_uk / W_uv and o_proj's rows of those heads); q_a | kv_a, the norms, the latent cache (every chip
  writes the same latent row into its own copy), mhc_pre / mhc_post and the DSA indexer are replicated;
  o_proj (row shard) -> a PARTIAL [T, 4096] -> FABRIC (tt/fabric_ar.py FabricPhase, "cb" mode: the readers hand their
  packed partial rows to the 2 lanes, kernels/mla/ra.cpp MK_FAB_*, which all-reduce over the mesh and signal the
  owners) -> mhc_post on the owners reads the reduced rows from the lanes (mhc_post/reader_p.cpp MK_POST_Y_FABRIC)
  -> s1 (identical on every chip).

MkMla builds its program for the whole mesh (the per-rank weights are mesh-sharded tensors: one buffer address on every
chip); this module captures the ProgramDescriptor (ttnn.generic_op intercepted during the call), adds the fabric hooks,
appends the lane kernels / CBs / fabric connections per chip and launches ONE MeshProgramDescriptor (only the lanes'
arguments differ between chips). A 1-chip mesh (TP-sized shards on one chip, `vtp`) runs the program unhooked: the
per-chip TP program minus the collective (its a is that chip's partial)."""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import _Capture, _rt_dict, _src_is

DRAM = ttnn.DRAM_MEMORY_CONFIG


class MeshHeadShard:
    """mk_mla.HeadShard for every chip of a mesh at once: rank r = mesh device r (row-major, ShardTensorToMesh's order)
    holds heads [H_r r, H_r (r + 1)); q_b / o_proj as mesh-sharded DS linears, `host_weights` = the per-rank host slices
    MkMla re-lays out (W_uk, W_uv, q_b for the folded norm); everything else is the (replicated) decoder attention's.
    vtp: TP-vtp-sized shards on fewer chips (ranks 0 .. ndev - 1 only; timing / 1-chip checks)."""

    FULL_KEYS = ("qb", "wo", "wuk", "wuv")

    def __init__(self, attn, mesh, *, vtp: int | None = None, host_full: dict | None = None, keep_full: bool = False):
        """host_full: {qb [q_lora, H dqk], wo [H dv, hidden], wuk [H, dqk, kv_lora], wuv [H, kv_lora, dv]} fp32 host
        copies of the decoder attention's weights (tt/tp_dsa.py's slice sidecar of an earlier attach_dsa) in place of
        the device read-backs (_host0: 4 reads + untilizes per layer, 8.5 s of the 11-layer attach); keep_full: keep
        them as self.host_full after the construction (a sidecar save)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear

        ndev = int(mesh.get_num_devices())
        tp = int(vtp or ndev)
        if ndev > tp or attn.h % tp:
            raise ValueError(f"{attn.h} heads, TP {tp} on {ndev} chips")
        hh, hr = attn.h, attn.h // tp
        self.device, self.ocfg, self.fcfg, self.tp, self.ndev = mesh, attn.ocfg, attn.fcfg, tp, ndev
        self.h, self.q_lora, self.kv_lora, self.dqk, self.dv = hr, attn.q_lora, attn.kv_lora, attn.dqk, attn.dv
        self.eps, self.scale, self.block_size = attn.eps, attn.scale, attn.block_size
        self.ds_qa_kva, self.qa_norm, self.kva_norm = attn.ds_qa_kva, attn.qa_norm, attn.kva_norm
        shapes = dict(qb=(attn.q_lora, attn.ds_qb.n), wo=(hh * attn.dv, attn.ds_o.n), wuk=(hh, attn.dqk, attn.kv_lora),
                      wuv=(hh, attn.kv_lora, attn.dv))  # fmt: skip
        if host_full is not None:
            got = {k: tuple(int(v) for v in host_full[k].shape) for k in self.FULL_KEYS}
            if got != shapes:
                raise ValueError(f"MeshHeadShard host_full shapes {got} != {shapes}")
            qb, wo, wuk, wuv = (host_full[k].float() for k in self.FULL_KEYS)
        else:
            host = MK._host0
            qb = host(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n]
            wo = host(attn.ds_o.weight).reshape(hh * attn.dv, -1)[:, : attn.ds_o.n]
            wuk = host(attn.w_uk).reshape(hh, attn.dqk, attn.kv_lora)
            wuv = host(attn.w_uv).reshape(hh, attn.kv_lora, attn.dv)
        self.host_full = dict(qb=qb, wo=wo, wuk=wuk, wuv=wuv) if keep_full else None
        heads = [slice(hr * r, hr * (r + 1)) for r in range(ndev)]
        qbs = [qb[:, s.start * attn.dqk : s.stop * attn.dqk] for s in heads]
        wos = [wo[s.start * attn.dv : s.stop * attn.dv] for s in heads]

        def sharded(blocks, dtype, dim):
            full = torch.cat([b.contiguous() for b in blocks], dim=dim)
            mapper = ttnn.ShardTensorToMesh(mesh, dim=dim) if ndev > 1 else None
            return ttnn.from_torch(full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                   mesh_mapper=mapper)  # fmt: skip

        self.ds_qb = DSLinear(mesh, sharded(qbs, attn.ds_qb.weight.dtype, -1), role="mla_qb", cfg=attn.ocfg,
                              ckc=attn.ds_qb.ckc, dtype=MK.BF16)  # fmt: skip
        self.ds_o = DSLinear(mesh, sharded(wos, attn.ds_o.weight.dtype, 0), role="mla_o", cfg=attn.ocfg,
                             ckc=attn.ds_o.ckc, dtype=MK.BF16)  # fmt: skip
        self.host_weights = dict(w_uk=[wuk[s] for s in heads], w_uv=[wuv[s] for s in heads], qb=qbs)
        self.heads = heads


def _with_rt(kd, extra: list, defines: list):
    """KernelDescriptor `kd` (a copy) with `defines` and `extra` appended to every core's runtime args at ONE index (the
    longest core's argument count: shorter cores are zero-padded) -> (kd, that index)"""
    rt = _rt_dict(kd)
    n = max(len(v) for v in rt.values())
    for (x, y), v in rt.items():
        kd.runtime_args[x][y].extend([0] * (n - len(v)) + [int(a) for a in extra])
    kd.defines = list(defines)
    return kd, n


class TPMla:
    """The sparse_dsa attention half, TP over every chip of `mesh` (decode rows t <= 32; t > 4: the flat residual /
    s1, Stage 14 native t): MkMla on MeshHeadShard + the
    in-program fabric all-reduce of o_proj's partial. Same call / outputs as MkMla (o_lat / o: the chip's heads; s1,
    latent, hn, ...: identical on every chip). `a` is NOT written on a fabric mesh (the partial goes to the lanes and
    the reduced rows only live in the lanes' result buffer, read by mhc_post)."""

    def __init__(self, mesh, mhc_args: dict, attn, *, vtp: int | None = None, shard: MeshHeadShard | None = None,
                 ts=(1, 2, 3, 4), fabrics: dict | None = None, host_full: dict | None = None, keep_full: bool = False,
                 **mk_kwargs):  # fmt: skip
        """shard: an earlier TPMla's .shard (resident per-rank weights: only the program objects are rebuilt). ts: the
        row counts served (one FabricPhase each, any of 1..32); fabrics: {t: FabricPhase} of an earlier TPMla with the same lane cores
        (Stage 14b: every sparse_dsa layer of the stack shares one set of lane buffers / counters)."""
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        self.shard = shard or MeshHeadShard(attn, mesh, vtp=vtp, host_full=host_full, keep_full=keep_full)
        self.mk = MK.MkMla(mesh, mhc_args, self.shard, **mk_kwargs)
        grid = mesh.compute_with_storage_grid_size()
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in self.mk.used]
        self.lanes = free[:2]
        assert len(self.lanes) == 2, "two free cores for the fabric lanes"
        self.nofab = self.R * self.C == 1
        # rows -> FabricPhase (the packed row count is a compile-time value of the readers): built (and probed: a launch +
        # read-back) HERE for every T, never on a call that may sit inside a trace capture
        if self.nofab:
            self.fabrics = {}
        elif fabrics is not None:
            # 14b int26: a sharer whose own MkMla leaves the owner's lane cores free adopts them (Stage 17's MTP layer
            # beside a GLM53_STACK_DSA main sparse_dsa layer: MkMla(dsa) occupies other cores, so the first two free
            # cores of the two differ while the owner's lanes are free here too)
            own = [tuple(int(v) for v in c) for c in next(iter(fabrics.values())).lanes]
            if own != [tuple(int(v) for v in c) for c in self.lanes] and all(c not in self.mk.used for c in own):
                self.lanes = own
            assert all([tuple(int(v) for v in c) for c in f.lanes] == [tuple(int(v) for v in c) for c in self.lanes]
                       for f in fabrics.values()), "shared fabrics: other lane cores"  # fmt: skip
            assert set(ts) <= set(fabrics), (ts, sorted(fabrics))
            self.fabrics = fabrics
        else:
            self.fabrics = {t: self._new_fabric(t) for t in ts}
        self.calls = 0

    def _new_fabric(self, t: int):
        f = FAR.FabricPhase(self.mesh, t, lanes=self.lanes, mode="cb")
        f.probe()
        return f

    def fabric(self, t: int):
        # a native-t width bucket (t > 4 without its own phase): the 32-row phase, whole-tile packets — the t > 4
        # consumer path (a b = 8 / 16 phase packs rows: StackBackend's widths hung the mesh with them, 2026-10-20)
        if t not in self.fabrics and t > 4 and 32 in self.fabrics:
            return self.fabrics[32]
        return self.fabrics[t]

    def _hooked(self, io, desc, t: int):
        """The captured MkMla program + the fabric phase -> MeshProgramDescriptor (one program per chip)."""
        fab = self.fabric(t)
        o_bit = self.mk.phase_order.index("o")
        n_readers = len(self.mk.last_readers.assign)
        kernels, seen = [], {"ra": 0, "own": 0}
        for kd in desc.kernels:
            if _src_is(kd, "mla/ra.cpp"):
                rt = max(len(v) for v in _rt_dict(kd).values())
                defs = list(kd.defines) + [("MK_FAB_MASK", str(1 << o_bit)), ("MK_FAB_BROWS", str(fab.brows)),
                                           ("MK_FAB_RT", str(rt))]  # fmt: skip
                kernels.append(_with_rt(kd, fab.producer_args(), defs)[0])
                seen["ra"] += 1
            elif _src_is(kd, "mla/owner_dm0.cpp"):  # mhc_post's reader: the reduced rows from the 2 lanes
                rt = max(len(v) for v in _rt_dict(kd).values())
                defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(rt))]
                kernels.append(_with_rt(kd, fab.consumer_args(), defs)[0])
                seen["own"] += 1
            else:
                kernels.append(kd)
        assert seen["ra"] >= 1 and seen["own"] >= 1, seen
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                lk, lcbs = fab.lane_parts(r, c, 1, n_readers, self.mk.owner_noc, cons_sem=MK.SEM_A)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk, semaphores=list(desc.semaphores),
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        return list(io) + fab.io(), mpd

    def __call__(
        self, x, cache, page_table, cur_pos, *, t: int, dsa_state: dict | None = None, keep=None, stack: bool = False
    ):
        """MkMla's call on every chip (+ the all-reduce of a): -> MkMla's outputs dict. keep: as MkMla's, applied after
        the launch (under the capture the program is not enqueued yet). stack: MkMla's stack mode (Stage 14b: x = the
        persistent stacked residual)."""
        self.calls += 1
        if self.nofab:
            return self.mk(x, cache, page_table, cur_pos, t=t, dsa_state=dsa_state, keep=keep, stack=stack)
        with _Capture() as cap:
            out = self.mk(x, cache, page_table, cur_pos, t=t, dsa_state=dsa_state, stack=stack)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked(io, desc, t)
        ttnn.generic_op(io2, mpd)
        if keep is not None:
            for k in [k for k in out if k not in keep]:
                v = out.pop(k)
                if v is not None:  # stack mode: no x_out
                    ttnn.deallocate(v)
        return out
