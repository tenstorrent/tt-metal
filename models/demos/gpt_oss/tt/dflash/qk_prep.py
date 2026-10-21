# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter's per-layer q / k prep as one program (fused_decode/kernels/draft_qk_prep_*.cpp), replacing
ttnn rms_norm + to_memory_config + rotary_embedding_llama for Q and for K, the V to_memory_config, and the per-pass
get_rot_mats (2 embeddings, 2 transposes, 2 interleaved_to_sharded):

  one core per draft row r (the head-sharded layout's core of user r): Q / K tiles of row r ([heads, 64], heads in
  tile rows) -> per-head RMSNorm over head_dim with the q_norm / k_norm weight -> RoPE (Meta layout, decode form:
  y = x * cos + (x . T) * sin per 32-column tile, cos / sin = rope cache row pos[r]) -> this core's shard of the
  head-sharded Q / K outputs; V tiles copied into its shard of the V output.
"""

import struct

import torch

import ttnn
from models.tt_transformers.tt.common import get_rot_transformation_mat

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
HEAD_GRID_X = 8  # head_mc's grid width (user r on core (r % 8, r // 8) from core_offset)


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class DraftQKPrep:
    def __init__(
        self,
        mesh_device,
        rows,
        nq,
        nkv,
        head_dim,
        eps,
        head_mc,
        cos_cache,
        sin_cache,
        core_offset=(0, 0),
        kv_core=None,
    ):
        """head_mc: the head-sharded memory config (user r on core (r % 8, r // 8)); cos_cache / sin_cache: the
        RotarySetup row-major [1, 1, positions, head_dim] BF16 caches; kv_core: the fused KV write's core (None: no
        fused KV write, parts() without kv)."""
        assert head_dim == 2 * TILE and nq <= TILE and nkv <= TILE
        self.mesh, self.M, self.nq, self.nkv, self.hd, self.eps = mesh_device, rows, nq, nkv, head_dim, eps
        self.head_mc, self.cos, self.sin = head_mc, cos_cache, sin_cache
        # joint (nq + nkv <= 16, the drafter's context pass): q's and k's heads in one tile (k rows nq .. nq + nkv - 1),
        # normalized and rotated at once with per-row weights (half the compute phases); else q and k tiles apart (the
        # draft pass: 16 + 2 heads per device)
        self.joint = nq + nkv <= 16
        self.defines = [("PREP_JOINT", "1"), ("NQ", str(nq)), ("NKV", str(nkv))] if self.joint else []
        # core_offset: head_mc's grid starts there (user r on core offset + (r % 8, r // 8))
        x0, y0 = core_offset
        self.core_list = [ttnn.CoreCoord(x0 + r % HEAD_GRID_X, y0 + r // HEAD_GRID_X) for r in range(rows)]
        self.kv_core = kv_core
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.trans = ttnn.from_torch(
            get_rot_transformation_mat(TILE).reshape(1, 1, TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def gamma_rows(self, w):
        """Joint mode, per-row weights: w [rows <= 16, head_dim] -> 2 full tiles (row h = w[h])."""
        assert self.joint
        g = torch.zeros(1, 1, TILE, self.hd)
        g[0, 0, : w.shape[0]] = w.float()
        return ttnn.from_torch(
            g,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def gamma(self, q_norm, k_norm):
        """[q_norm | k_norm] (head_dim each) as full tiles (every row = the weight): 4 tiles in DRAM; joint: 2 tiles,
        rows 0 .. nq - 1 = q_norm, rows nq .. nq + nkv - 1 = k_norm."""
        if self.joint:
            return self.gamma_rows(
                torch.cat(
                    [
                        q_norm.float().reshape(1, -1).expand(self.nq, -1),
                        k_norm.float().reshape(1, -1).expand(self.nkv, -1),
                    ]
                )
            )
        w = torch.cat([q_norm.float().reshape(-1), k_norm.float().reshape(-1)])
        assert w.numel() == 2 * self.hd
        return ttnn.from_torch(
            w.reshape(1, 1, 1, -1).expand(1, 1, TILE, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, q, k, v, gamma, rot_pos):
        kernels, cbs, sems, io, outs = self.parts(q, k, v, gamma, rot_pos)
        ttnn.generic_op(io + list(outs), ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return outs

    def parts(self, q, k, v, gamma, rot_pos, kv=None, fuse_fn=None, sem_base=0):
        """(kernels, cbs, semaphores, io, (q, k, v)) of the program; the outputs are the head-sharded q / k / v.
        q / k / v: [1, 32, heads, 64] BF16 tiles in DRAM or L1 (row r = user r); gamma: from gamma(); rot_pos: UINT32
        [1, 32] (row r's rope position). kv = (DraftKVWrite, k_cache, v_cache, pos, page_table): the K / V rows' cache
        write in the same program (on core kv_core, after every prep core's semaphore; semaphore ids sem_base, sem_base
        + 1).
        fuse_fn(qo, (ko, vo)) -> (kernels, cbs, io, noc_cores, sem_id, semaphores): an op in the same program (the
        drafter's block attention): the K and V write kernels each increment its cores' semaphore sem_id once the
        head-sharded k / v are written (the op reads the block's rows from them)."""
        alloc = lambda h: ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, TILE, h, self.hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, self.head_mc
        )
        qo, ko, vo = alloc(self.nq), alloc(max(self.nkv, 1)), alloc(self.nkv or self.nq)  # (nkv 0: V has nq heads)
        cb_q, cb_k, cb_g, cb_cs, cb_t, cb_scl, cb_scr, cb_sq, cb_rs, cb_t1, cb_xn, cb_t2, cb_qo, cb_ko = range(14)
        f32, bf16 = ttnn.float32, ttnn.bfloat16

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(cb_q, bf16, 2048, 2),
            cb(cb_k, bf16, 2048, 2),
            cb(cb_g, bf16, 2048, 4),
            cb(cb_cs, bf16, 2048, 4),
            cb(cb_t, bf16, 2048, 1),
            cb(cb_scl, bf16, 2048, 1),
            cb(cb_scr, bf16, 384, 1),
            cb(cb_sq, f32, 4096, 1),
            cb(cb_rs, f32, 4096, 2),
            cb(cb_t1, f32, 4096, 2),
            cb(cb_xn, bf16, 2048, 2),
            cb(cb_t2, bf16, 2048, 2),
            cb(cb_qo, bf16, 2048, 2),
            cb(cb_ko, bf16, 2048, 2),
        ]
        row_bytes = self.hd * 2
        reader_ct = [cb_q, cb_k, cb_g, cb_cs, cb_t, cb_scl, cb_scr, row_bytes]
        for t in (q, k, v, gamma, self.cos, self.sin, rot_pos, self.trans):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [
            cb_q,
            cb_k,
            cb_g,
            cb_cs,
            cb_t,
            cb_scl,
            cb_sq,
            cb_rs,
            cb_t1,
            cb_xn,
            cb_t2,
            cb_qo,
            cb_ko,
            _f32_bits(1.0 / self.hd),
            _f32_bits(self.eps),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        notify = []
        if kv is not None:
            assert self.kv_core is not None, "the fused KV write needs kv_core"
            p = self.mesh.worker_core_from_logical_core(self.kv_core)
            notify = [p.x, p.y]
        for r, c in enumerate(self.core_list):
            reader_rt[c.x][c.y] = [
                q.buffer_address(),
                k.buffer_address(),
                v.buffer_address(),
                gamma.buffer_address(),
                self.cos.buffer_address(),
                self.sin.buffer_address(),
                rot_pos.buffer_address(),
                self.trans.buffer_address(),
                vo.buffer_address(),
                r,
            ] + notify
            writer_rt[c.x][c.y] = [qo.buffer_address(), ko.buffer_address()] + notify
        dm = ttnn.DataMovementConfigDescriptor
        defs = ([("PREP_NOTIFY", "1")] if kv is not None else []) + self.defines + [("PREP_SEM_BASE", str(sem_base))]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_qk_prep_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defs,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_qk_prep_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=[cb_qo, cb_ko],
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defs,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_qk_prep_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=self.defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        sems, io = [], [q, k, v, gamma, self.cos, self.sin, rot_pos, self.trans]
        fused = fuse_fn(qo, (ko, vo)) if fuse_fn is not None else None
        if kv is not None:
            writer, k_cache, v_cache, pos, page_table = kv
            kk, kcb = writer.parts(
                ko,
                vo,
                k_cache,
                v_cache,
                pos,
                page_table,
                core=self.kv_core,
                cb_id=14,
                wait=(sem_base, self.M),
                notify=(fused[3], fused[4]) if fused is not None else None,
            )
            kernels += kk
            cbs += kcb
            both = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(self.kv_core, self.kv_core)]))
            sems = [ttnn.SemaphoreDescriptor(sem_base + i, ttnn.CoreType.WORKER, both, 0) for i in (0, 1)]
            io += [pos, page_table, k_cache, v_cache]
        if fused is not None:
            assert kv is not None, "the fused op waits for the K / V cache write"
            f_kernels, f_cbs, f_io, _, f_sem, f_sems = fused[:6]
            kernels += list(f_kernels)
            cbs += list(f_cbs)
            sems += [ttnn.SemaphoreDescriptor(f_sem, ttnn.CoreType.WORKER, f_kernels[0].core_ranges, 0)] + list(f_sems)
            io += [t for t in f_io if all(t is not u for u in io + [qo, ko, vo])]
        return kernels, cbs, sems, io, (qo, ko, vo)
