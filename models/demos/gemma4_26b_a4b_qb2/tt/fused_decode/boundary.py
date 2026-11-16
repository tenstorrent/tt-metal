# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode layer boundaries of Gemma 4 26B on the QB2 1x4 mesh: the TP all-reduce, the sandwich norms, the
residual add and the next layer's input norm as one program per boundary (kernels/bnd_*.cpp).

Residual contract (as models/demos/gpt_oss/tt/fused_decode/boundary.py): between the boundaries the residual stream
is a *flat* BF16 vector replicated on every device (hidden value h at byte 2 h of `tiles` 2 KB tile pages on one
core, zero padded). Two persistent buffers alternate (entry / moe write buffer 0, attn writes buffer 1), so the
traced step reuses fixed addresses.

Producers and consumers stay ttnn ops: each boundary gathers row 0 of its producers' [1, 1, 32, hidden] BF16 TILE
outputs into a flat partial, multicasts it over the fabric into slot `ring index` of every device's receive buffer
(one per call site, "attn" / "moe", alternating, so a slot is never rewritten before every device consumed it), sums
the slots in slot order (bit-identical on every device) and writes its outputs as row 0 of persistent TILE tensors
(rows 1..31 stay zero):

  entry(hidden):                 x = input_norm(hidden)                        -> qkv_in
  attn(o_proj partial):          res += post_attention_norm(sum)
                                 n = norm(res): router_in = n * router_scale (FP32), expert_in = n * g_expert,
                                 shared_in = n [* g_shared]
  moe(shared, routed partials):  res = (res + ff_norm(ff1_norm(sum_s) + ff2_norm(sum_r))) * layer_scalar
                                 x = next_norm(res) (the next layer's input norm, or the final norm) -> qkv_in
"""

import os
import struct
from pathlib import Path

import torch

import ttnn

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32
PAGE = TILE * TILE * 2  # BF16 tile page
SITES = ("attn", "moe")
FAST_RECONFIG = os.environ.get("GEMMA4_26B_WS5_BND_FAST", "1") != "0"


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class GemmaBoundary:
    """Persistent state of the batch-1 decode boundaries (module doc); one instance per model."""

    CB_PART, CB_RES, CB_RECV, CB_G, CB_SCALER, CB_VF, CB_SYNC = 0, 1, 2, 3, 4, 5, 6
    CB_A, CB_B, CB_SS, CB_RS, CB_NB = 8, 9, 10, 11, 12
    CB_RES_OUT, CB_O0 = 16, 17  # outputs CB_O0 .. CB_O0 + 2

    def __init__(
        self,
        mesh,
        hidden,
        eps,
        *,
        core=(0, 0),
        shared_dtype=ttnn.float32,
        round_n=True,
        flat_moe=False,
        flat_router=False,
    ):
        """round_n: the attn boundary rounds its normed vector to BF16 before the router / expert / shared scales (as
        the ttnn RMSNorm output of the unfused path); False keeps it FP32. flat_moe: expert_in and shared_in are flat
        BF16 vectors on the boundary core (moe_decode x_flat consumers) instead of row 0 of TILE tensors."""
        self.round_n = round_n
        self.flat_moe = flat_moe
        self.flat_router = flat_router  # router_in flat FP32 on the boundary core (fused_decode/router.py)
        assert not flat_moe or shared_dtype == ttnn.bfloat16
        assert not flat_router or flat_moe
        self.mesh = mesh
        self.hidden, self.eps = hidden, eps
        self.ring = mesh.shape[1]
        assert mesh.shape[0] == 1 and hidden % TILE == 0
        self.T = -(-hidden // (TILE * TILE))
        self.htiles = hidden // TILE
        self.core = ttnn.CoreCoord(*core)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        phys = mesh.worker_core_from_logical_core(self.core)
        self.noc_xy = (phys.x, phys.y)
        self.coords = [ttnn.MeshCoordinate(0, c) for c in range(self.ring)]
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def flat(pages, dtype=ttnn.bfloat16):
            config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.cores, (TILE, pages * TILE), ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                torch.zeros(1, 1, TILE, pages * TILE),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=config,
                mesh_mapper=rep,
            )

        self.res = [flat(self.T), flat(self.T)]
        self.part = flat(2 * self.T)
        all_cores = mesh.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(all_cores.x - 1, all_cores.y - 1))]
        )
        self.sites = {
            "attn": (flat(self.ring * self.T), ttnn.create_global_semaphore(mesh, all_cores, 0)),
            "moe": (flat(self.ring * 2 * self.T), ttnn.create_global_semaphore(mesh, all_cores, 0)),
        }

        def rows(dtype):
            return ttnn.from_torch(
                torch.zeros(1, 1, 1, hidden),  # logical one row, tile padded (zeros)
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        # Consumer inputs: row 0 rewritten by every boundary, rows 1..31 zero.
        self.qkv_in = rows(ttnn.bfloat16)
        self.router_in = flat(self.T, ttnn.float32) if flat_router else rows(ttnn.float32)
        # shared_in: the plain normed vector (shared-MLP gamma folded into its weights)
        self.expert_in = flat(self.T) if flat_moe else rows(ttnn.bfloat16)
        self.shared_in = flat(self.T) if flat_moe else rows(shared_dtype)
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.max_chunk = max_payload // PAGE * PAGE

    # Host-side weights -------------------------------------------------------------------------------------------
    def gammas(self, vectors):
        """Norm weights as one flat BF16 row-major DRAM page: `vectors` (each [hidden], None = ones) back to back,
        `T` tile pages each."""
        n = self.T * TILE * TILE
        flat = torch.zeros(1, 1, 1, len(vectors) * n)
        for i, v in enumerate(vectors):
            flat[0, 0, 0, i * n : i * n + self.hidden] = 1.0 if v is None else v.float().reshape(-1)
        return ttnn.from_torch(
            flat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def fp32_vector(self, v):
        flat = torch.zeros(1, 1, 1, self.T * TILE * TILE)
        flat[0, 0, 0, : self.hidden] = v.float().reshape(-1)
        return ttnn.from_torch(
            flat,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    # Program -----------------------------------------------------------------------------------------------------
    def _cb(self, index, pages, dtype=ttnn.bfloat16, tensor=None):
        page = PAGE if dtype == ttnn.bfloat16 else 2 * PAGE
        fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
        if tensor is not None:
            cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
            cb.total_size = pages * page
            cb.format_descriptors = fmt
            return cb
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.cores, format_descriptors=fmt)

    def _run(
        self,
        mode,
        parts,
        gamma,
        gcount,
        res_in,
        res_out,
        outs,
        vf=None,
        scalar=1.0,
        out2_gamma=False,
        flat_outs=(),
        pregathered=False,
        tokens=None,
        table=None,
    ):
        """outs: written as row 0 of TILE tensors (half each by bnd_reader.cpp / bnd_sender.cpp); flat_outs: the next
        compute outputs, packed straight into flat tensors on the boundary core. pregathered: the producers wrote the
        flat partial buffer (self.part) themselves; `parts` are then placeholders."""
        T = self.T
        for part in parts:
            # 32 B face-row reads: a DRAM source would need 64 B aligned destinations (Blackhole).
            assert part.memory_config().buffer_type == ttnn.BufferType.L1, "boundary inputs must be in L1"
            assert part.dtype == ttnn.bfloat16 and part.layout == ttnn.TILE_LAYOUT and part.padded_shape[-2] == TILE
        site = {0: None, 1: "attn", 2: "moe", 3: None}[mode]
        slots = self.ring if site else 0
        npart = len(parts)
        recv, sem = self.sites[site] if site else (None, None)
        sem_addr = ttnn.get_global_semaphore_address(sem) if site else 0
        cbs = [
            self._cb(self.CB_RES, T, tensor=res_out if mode in (0, 3) else res_in),
            self._cb(self.CB_G, gcount * T),
            self._cb(self.CB_SCALER, 1),
            self._cb(self.CB_A, 2 * T, ttnn.float32),
            self._cb(self.CB_B, 2 * T, ttnn.float32),
            self._cb(self.CB_SS, 2, ttnn.float32),
            self._cb(self.CB_RS, 2, ttnn.float32),
            self._cb(self.CB_NB, T, ttnn.bfloat16 if self.round_n else ttnn.float32),
            self._cb(self.CB_SYNC, 1),
        ]
        if mode == 3:
            cbs.append(self._cb(self.CB_PART, T + 1))  # token entry: the aligned stage of the embedding row
            cbs.append(self._cb(self.CB_RES_OUT, T, tensor=res_out))
        if mode in (1, 2):
            cbs += [
                self._cb(self.CB_PART, npart * T, tensor=self.part),
                self._cb(self.CB_RECV, slots * npart * T, tensor=recv),
                self._cb(self.CB_RES_OUT, T, tensor=res_out),
            ]
        if vf is not None:
            cbs.append(self._cb(self.CB_VF, T, ttnn.float32))
        fp32_mask = 0
        for i, o in enumerate(outs):
            cbs.append(self._cb(self.CB_O0 + i, T, o.dtype))
            fp32_mask |= (o.dtype == ttnn.float32) << i
        for i, o in enumerate(flat_outs):
            # packed by the compute straight into a flat tensor on the boundary core (no row writes)
            cbs.append(self._cb(self.CB_O0 + len(outs) + i, T, o.dtype, tensor=o))

        def acc(t):
            return ttnn.TensorAccessorArgs(t).get_compile_time_args()

        p0 = parts[0] if parts else gamma
        p1 = parts[1] if npart > 1 else p0
        vft = vf if vf is not None else gamma
        reader_ct = [self.CB_PART, self.CB_RES, self.CB_RECV, self.CB_G, self.CB_SCALER, self.CB_VF, T, npart]
        reader_ct += [slots, self.ring, 1 if mode == 0 else 0, self.htiles, gcount, 1 if vf is not None else 0]
        o = list(outs) + [outs[0] if outs else self.qkv_in] * (3 - len(outs))
        reader_ct += [self.CB_SYNC, 1 if pregathered else 0, len(outs), self.CB_O0, fp32_mask]
        reader_ct += acc(p0) + acc(p1) + acc(gamma) + acc(vft) + acc(o[0]) + acc(o[1]) + acc(o[2])
        reader_ct += [1 if mode == 3 else 0] + acc(tokens if mode == 3 else gamma) + acc(table if mode == 3 else gamma)
        reader_rt = ttnn.RuntimeArgs()
        reader_rt[self.core.x][self.core.y] = (
            [
                p0.buffer_address(),
                p1.buffer_address(),
                gamma.buffer_address(),
                vft.buffer_address(),
                sem_addr,
            ]
            + [t.buffer_address() for t in o]
            + ([tokens.buffer_address(), table.buffer_address()] if mode == 3 else [0, 0])
        )
        compute_ct = [self.CB_RECV, self.CB_RES, self.CB_G, self.CB_SCALER, self.CB_VF, self.CB_A, self.CB_B]
        compute_ct += [self.CB_SS, self.CB_RS, self.CB_RES_OUT, self.CB_O0, T, slots, max(npart, 1), mode]
        compute_ct += [_f32_bits(1.0 / self.hidden), _f32_bits(self.eps), 1 if out2_gamma else 0, self.CB_NB]
        compute_rt = ttnn.RuntimeArgs()
        compute_rt[self.core.x][self.core.y] = [_f32_bits(scalar)]
        payload = npart * T * PAGE
        chunk = min(payload, self.max_chunk) if payload else PAGE
        io = list(parts) + [gamma, res_in, res_out] + list(outs) + list(flat_outs)
        io += [vf] if vf is not None else []
        if site:
            io.append(recv)

        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            c = coord[1]
            fwd, bwd = (self.ring - 1 - c, c) if site else (0, 0)
            sender_ct = [payload, chunk, fwd, bwd, slots, self.CB_PART, npart * T, T, self.htiles, len(outs)]
            sender_ct += [self.CB_O0, fp32_mask] + acc(o[0]) + acc(o[1]) + acc(o[2])
            sender_ct += [self.CB_SYNC, 1 if pregathered else 0, npart] + acc(p0) + acc(p1)
            sender_rt = ttnn.RuntimeArgs()
            sender_rt[self.core.x][self.core.y] = (
                [
                    self.part.buffer_address(),
                    (recv.buffer_address() + c * payload) if site else 0,
                    self.noc_xy[0],
                    self.noc_xy[1],
                    sem_addr,
                ]
                + [t.buffer_address() for t in o]
                + [p0.buffer_address(), p1.buffer_address()]
            )
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "bnd_reader.cpp"),
                    core_ranges=self.cores,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "bnd_compute.cpp"),
                    core_ranges=self.cores,
                    compile_time_args=compute_ct,
                    runtime_args=compute_rt,
                    defines=[("FAST_RECONFIG", "1")] if FAST_RECONFIG else [],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "bnd_sender.cpp"),
                    core_ranges=self.cores,
                    compile_time_args=sender_ct,
                    runtime_args=sender_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                ),
            ]
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
            if site:
                fabric_args = []
                src = self.mesh.get_fabric_node_id(coord)
                for step, hops in ((1, fwd), (-1, bwd)):
                    fabric_args.append(1 if hops > 0 else 0)
                    if hops > 0:
                        dst = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c + step))
                        fabric_args += ttnn.setup_fabric_connection(
                            mesh_device=self.mesh,
                            src_fabric_node_id=src,
                            dst_fabric_node_id=dst,
                            link_idx=0,
                            program_descriptor=program,
                            worker_core=self.core,
                        )
                program.kernels[2].runtime_args[self.core.x][self.core.y].extend(fabric_args)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op(io, mesh_program)

    # Boundaries --------------------------------------------------------------------------------------------------
    def entry_token(self, tokens, table, gamma, scale):
        """Token entry: the residual is row tokens[0] of the replicated row-major embedding `table` ([vocab, hidden]
        BF16, DRAM, one row per page) times `scale`; returns qkv_in (the first layer's normed input)."""
        self._run(3, [], gamma, 1, self.res[0], self.res[0], [self.qkv_in], scalar=scale, tokens=tokens, table=table)
        return self.qkv_in

    def entry(self, hidden, gamma):
        """hidden: the embedded token, [1, 1, 32, hidden] BF16 TILE (row 0). gamma: gammas([input_norm]). Returns
        qkv_in (the first layer's normed input); the residual is res[0]."""
        self._run(0, [hidden], gamma, 1, self.res[0], self.res[0], [self.qkv_in])
        return self.qkv_in

    def attn(self, partial, gamma, router_scale, *, shared_gamma=False):
        """partial: this device's o_proj output, [1, 1, 32, hidden] BF16 TILE (row 0). gamma: gammas([post_attention,
        expert, shared]) (shared_gamma=False: gammas([post_attention, expert]), shared_in = the plain normed vector);
        router_scale: fp32_vector(router input scale). Returns (router_in FP32, expert_in BF16, shared_in FP32)."""
        outs = [self.router_in, self.expert_in, self.shared_in]
        self._run(
            1,
            [partial],
            gamma,
            3 if shared_gamma else 2,
            self.res[0],
            self.res[1],
            [] if self.flat_router else (outs[:1] if self.flat_moe else outs),
            vf=router_scale,
            out2_gamma=shared_gamma,
            flat_outs=outs if self.flat_router else (outs[1:] if self.flat_moe else ()),
        )
        return self.router_in, self.expert_in, self.shared_in

    def moe_pregathered(self, gamma, layer_scalar):
        """moe() with the shared / routed partials already written flat into self.part (pages [0, T) / [T, 2 T)) by
        their producers (moe_decode flat_out=(boundary.part, 0 / T * 2048))."""
        self._run(
            2,
            [self.part, self.part],
            gamma,
            4,
            self.res[1],
            self.res[0],
            [self.qkv_in],
            scalar=layer_scalar,
            pregathered=True,
        )
        return self.qkv_in

    def moe(self, shared, routed, gamma, layer_scalar):
        """shared, routed: this device's shared-MLP and routed-expert partial sums ([1, 1, 32, hidden] BF16 TILE, row
        0). gamma: gammas([ff1, ff2, ff, next input norm (or the final norm)]). Returns qkv_in (the next normed input).
        """
        self._run(2, [shared, routed], gamma, 4, self.res[1], self.res[0], [self.qkv_in], scalar=layer_scalar)
        return self.qkv_in

    def residual(self):
        """Host copy of the residual (res[0]) of device 0, [hidden] float (tests)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.res[0])[0]).float()
        return flat_from_tile_rows(t, self.hidden)


def flat_from_tile_rows(mat, hidden):
    """Inverse of the device byte order: a [.., 32, 32 * pages] tile container read back with to_torch -> flat."""
    pages = mat.shape[-1] // TILE
    t = mat.reshape(2, 16, pages, 2, 16).permute(2, 0, 3, 1, 4)
    return t.reshape(-1)[:hidden]


def flat_to_tile_rows(vec, pages):
    """Host helper: a flat vector -> the [32, 32 * pages] matrix whose TILE-layout device bytes are the vector."""
    flat = torch.zeros(pages * TILE * TILE, dtype=vec.dtype)
    flat[: vec.numel()] = vec.reshape(-1)
    return flat.reshape(pages, 2, 2, 16, 16).permute(1, 3, 0, 2, 4).reshape(TILE, pages * TILE)
