# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``RouterTopk``: the Gemma 4 router tail (softmax, top-k, sum-normalize, per-expert scale) straight into the
local experts' routing tables, as one program.

Input ``logits`` is the router projection ``[1, 1, B, E]`` (bf16, TILE, DRAM; ``B <= 32`` users on one row tile,
``E`` experts, a multiple of 32), ``pes`` the per-expert scale ``[1, 1, 1, E]`` (bf16, ROW_MAJOR) and ``base`` a
``[1, 1, 1, 16]`` uint32 ROW_MAJOR tensor whose first word is this chip's first global expert (the ``(ep, tp)``
sharding of ``experts/weights.py``: chip ``(r, c)`` holds experts ``[(r*tp + c) * E_loc, +E_loc)``), so the same
program on every chip of the mesh selects its own expert columns. Outputs, both ROW_MAJOR bf16 DRAM:

* ``w`` ``[1, 1, E_loc, 32]``: row ``e`` = the users' routing weights of local expert ``e`` (zero where a user did
  not select it, rows ``>= B`` zero) -- the weight table ``custom_ops/moe_combine`` consumes;
* ``sparsity`` ``[1, 1, 1, E_loc]``: 1.0 where some user selected the local expert -- the ``sparse_matmul`` mask.

HF: ``probs = softmax(logits); v, i = topk(probs, k); w = v / v.sum() * per_expert_scale[i]``; softmax then top-k
equals top-k of the logits then ``exp(l_j - max) / sum_selected``, so only the ``k`` selected exponentials are
taken (fp32, one bf16 rounding). User ``u``'s row is ranked by core ``u`` (RISC-V scalar code: 32 cores, ~10 us),
the local weight vectors are gathered on core 0 through a semaphore, which transposes them into the expert rows,
writes both outputs and resets the semaphore for the next trace replay. This replaces ``softmax, topk (+fillpad),
sum (fillpad + reduce), div, zeros_like, scatter (5 programs), mul, matmul (expert select), max, to_layout,
permute``: 15 programs -> 1.

``op_packed`` is the same program in its PACKED mode for the decode MoE kernel path (``GEMMA4_MOE_DECODE_KERNEL``
rows): instead of the local weight tables it writes the per-user ``(values, ids)`` as one ``[1, 1, 32, 2k]`` bf16
ROW_MAJOR tensor (columns ``[0, k)`` = sum-normalized top-k weights times ``per_expert_scale[id]``, descending;
``[k, 2k)`` = the global expert ids as bf16, exact below 256; rows ``>= B`` carry weight 0 and the distinct ids
``0..k-1``). No gather, no semaphore. It replaces ``softmax, topk (+fillpad), sum (fillpad + reduce), mul, gather
(+fillpad), div`` (10 programs) and feeds ``custom_ops/moe_frame_prep`` / the lane gather as one tensor.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class RouterTopk:
    CB_LOGITS = 0  # the logit tiles of the row tile
    CB_SCRATCH = 1  # pes, base, the send buffer, the output staging
    CB_GATHER = 2  # core 0: one 64-byte slot per user (the user's local weight vector)
    SEM = 0
    MAX_EXPERTS = 256
    MAX_LOCAL_EXPERTS = 64  # the gathered form: 64-byte gather slots per user, E_loc staging rows of 64 B

    @classmethod
    def supports(cls, num_experts: int, top_k: int, packed: bool = False, e_loc: int | None = None) -> bool:
        """Whether the router programs can take ``num_experts`` logits columns and a top-``top_k``: the expert
        count is a tile multiple up to MAX_EXPERTS (the row tile's columns; the kernel ranks a row from a
        ``uint32_t key[E]`` on its stack) and k <= 32; the PACKED form also needs 32-byte rows of 2k bf16 (k a
        multiple of 8); the gathered form's local expert count ``e_loc`` (when given) is even and at most
        MAX_LOCAL_EXPERTS (the leader's 64-byte gather slots and staging rows). A model outside this (the CI's
        reduced-expert configurations) keeps the shared ttnn chain; callers decide with this, the programs assert."""
        e, k = int(num_experts), int(top_k)
        if e <= 0 or e % TILE or e > cls.MAX_EXPERTS or not 1 <= k <= TILE:
            return False
        if e_loc is not None and not (1 <= int(e_loc) <= cls.MAX_LOCAL_EXPERTS and int(e_loc) % 2 == 0):
            return False
        return not packed or (4 * k) % TILE == 0

    @staticmethod
    def golden(logits: torch.Tensor, pes: torch.Tensor, base: int, e_loc: int, k: int):
        """torch reference (HF order): logits [1,1,B,E], pes [E] -> (w [1,1,e_loc,32], sparsity [1,1,1,e_loc])."""
        B, E = logits.shape[2], logits.shape[3]
        probs = torch.softmax(logits.float().reshape(B, E), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        wk = v / v.sum(dim=-1, keepdim=True) * pes.float()[i]
        dense = torch.zeros(B, E).scatter(1, i, wk)
        local = dense[:, base : base + e_loc]  # [B, e_loc]
        w = torch.zeros(1, 1, e_loc, TILE)
        w[0, 0, :, :B] = local.t()
        sparsity = (local != 0).any(dim=0).float().reshape(1, 1, 1, e_loc)
        return w, sparsity

    @staticmethod
    def make_base(mesh_device, mesh_config, e_loc: int, ep: int, tp: int):
        """The per-chip ``[1, 1, 1, 16]`` uint32 first-local-expert tensor, sharded like ``expert_select``."""
        is_mesh = hasattr(mesh_device, "shape")
        vals = torch.zeros(ep, tp, 1, 16, dtype=torch.int32)
        for r in range(ep):
            for c in range(tp):
                vals[r, c, 0, :] = (r * tp + c) * e_loc
        mapper = None
        if is_mesh:
            tp_axis = int(getattr(mesh_config, "tp_axis", 1)) if mesh_config is not None else 1
            ep_dim = 0 if ep > 1 else None
            tp_dim = 1 if tp > 1 else None
            dims = (tp_dim, ep_dim) if tp_axis == 0 else (ep_dim, tp_dim)
            mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=dims)
        return ttnn.from_torch(
            vals,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    @staticmethod
    def golden_packed(logits: torch.Tensor, pes: torch.Tensor, k: int):
        """torch reference of the packed form: logits [1,1,B,E], pes [E] -> [1,1,32,2k] float (rows >= B: weight 0,
        ids 0..k-1)."""
        B, E = logits.shape[2], logits.shape[3]
        probs = torch.softmax(logits.float().reshape(B, E), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        wk = v / v.sum(dim=-1, keepdim=True) * pes.float()[i]
        out = torch.zeros(1, 1, TILE, 2 * k)
        out[0, 0, :B, :k] = wk
        out[0, 0, :B, k:] = i.float()
        out[0, 0, B:, k:] = torch.arange(k, dtype=torch.float32)
        return out

    @classmethod
    def op_packed(cls, logits: ttnn.Tensor, pes: ttnn.Tensor, base: ttnn.Tensor, k: int):
        """PACKED mode (see the module docstring): ``[1, 1, 32, 2k]`` bf16 ROW_MAJOR DRAM (one page per user row)."""
        assert 1 <= k <= 32 and (4 * k) % 32 == 0, f"packed rows of 2k bf16 must be 32-byte aligned (k={k})"
        return cls._run(logits, pes, base, e_loc=2, k=k, packed=True)

    @classmethod
    def op(cls, logits: ttnn.Tensor, pes: ttnn.Tensor, base: ttnn.Tensor, e_loc: int, k: int):
        return cls._run(logits, pes, base, e_loc=e_loc, k=k, packed=False)

    @classmethod
    def _run(cls, logits: ttnn.Tensor, pes: ttnn.Tensor, base: ttnn.Tensor, e_loc: int, k: int, packed: bool):
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype == ttnn.bfloat16, "logits must be bf16 TILE"
        _, _, B, E = (int(v) for v in logits.shape)
        assert B <= TILE and logits.padded_shape[-2] == TILE, f"one row tile of users expected, got {logits.shape}"
        assert cls.supports(E, 1), f"experts {E} must be a multiple of 32 (<= {cls.MAX_EXPERTS})"
        assert pes.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(int(v) for v in pes.shape) == (
            1,
            1,
            1,
            E,
        ), "pes RM [1,1,1,E]"
        assert base.layout == ttnn.ROW_MAJOR_LAYOUT and base.dtype == ttnn.uint32, "base uint32 RM"
        assert cls.supports(E, k, packed=packed, e_loc=None if packed else e_loc), (E, k, e_loc, packed)
        device = logits.device()
        if packed:
            w_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, TILE, 2 * k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            sp_out = w_out  # unused in PACKED mode (bound so the accessor args stay the same)
        else:
            w_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, e_loc, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            sp_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, e_loc]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
        gx, gy = compute_grid(device)  # (x, y), capped by GEMMA4_TEST_GRID
        n_cores = TILE  # one user row per core (rows >= B idle); core 0 is the leader
        assert gx * gy >= n_cores
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        leader = device.worker_core_from_logical_core(cores[0])
        tile_bytes = TILE * TILE * 2
        cols_t = E // TILE
        slot_bytes = 64 if e_loc <= 32 else 128
        scratch_bytes = 2048 + e_loc * 64

        def cb(idx, total, page, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_LOGITS, cols_t * tile_bytes, tile_bytes),
            cb(cls.CB_SCRATCH, scratch_bytes, scratch_bytes),
            cb(cls.CB_GATHER, TILE * slot_bytes, TILE * slot_bytes),
        ]
        ct = [
            cls.CB_LOGITS,
            cls.CB_SCRATCH,
            cls.CB_GATHER,
            cols_t,
            E,
            e_loc,
            k,
            pes.buffer_page_size(),
            base.buffer_page_size(),
            w_out.buffer_page_size(),
            sp_out.buffer_page_size(),
            slot_bytes,
            cls.SEM,
            leader.x,
            leader.y,
        ]
        for t in (logits, pes, base, w_out, sp_out):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ct.append(1 if packed else 0)
        rt = ttnn.RuntimeArgs()
        for u, core in enumerate(cores):
            rt[core.x][core.y] = [
                logits.buffer_address(),
                pes.buffer_address(),
                base.buffer_address(),
                w_out.buffer_address(),
                sp_out.buffer_address(),
                u,
                B,
            ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "router_topk.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        sems = [ttnn.SemaphoreDescriptor(cls.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        if packed:
            ttnn.generic_op([logits, pes, base, w_out], program)
            return w_out
        ttnn.generic_op([logits, pes, base, w_out, sp_out], program)
        return w_out, sp_out
