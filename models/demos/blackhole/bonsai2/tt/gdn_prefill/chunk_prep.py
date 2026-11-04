# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GatedDeltaNet chunked prefill: the preprocessing for ttnn.transformer.gated_delta_attn_seq in one op.

models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py builds the scan kernel's 8 inputs with
about 150 ttnn ops per call on [heads x chunks, 128, 128] fp32 tensors in DRAM (decay prefix sums, the decay mask,
k k^T, the diagonal normalisation, the block inverses by a 30-step Horner solve). Every (head, chunk) is
independent, so kernels/prep_*.cpp computes all of one (head, chunk) on one core in L1 (130 cores share the
heads x chunks items). chunk_gated_delta_rule_seq_fused is a drop-in for that module's chunk_gated_delta_rule_seq for
whole chunks without padding; anything else goes to the original."""

import os
import struct

import torch

import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tt/gdn_prefill/kernels/"
C = 128


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class ChunkPrep:
    def __init__(self, device):
        e = torch.zeros(32, 32)
        e[31, :] = 1.0
        consts = [
            torch.triu(torch.ones(32, 32)),  # TRIU (with the diagonal)
            torch.ones(32, 32),  # ONES
            torch.tril(torch.ones(32, 32)),  # TRIL
            torch.tril(torch.ones(32, 32), -1),  # STRIL
            torch.eye(32),  # EYE
            e,  # E31: row 31 of ones
            torch.zeros(32, 32),  # ZERO
            torch.triu(torch.ones(32, 32), 1),  # STRIU
        ]
        self.consts = ttnn.from_torch(torch.cat(consts, 0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.device = device

    def __call__(self, q, k, v, beta, g, alpha, token_major=None):
        """q (scaled), k, v [BH, T, 128], beta [BH, T, 1], g [BH, T]: fp32 TILE, T a multiple of 128.
        -> L_unit, v_beta_sc, k_bd_sc, intra, q_decay, k_decay_t [BH, NC, 128, 128], dl_exp [BH, NC, 1, 1],
        L_inv [BH, NC, 128, 32]."""
        device = self.device
        BH, T = list(beta.shape)[0], list(beta.shape)[1]
        # token_major: q, k, v are one bf16 [1, T, W] tensor (the conv output); dict(Wt, rf, q_off, k_off, v_off,
        # eps, scale): tile-column offsets, value heads per key head, the q/k L2-norm eps and q's scale.
        tm = token_major
        NC = T // C
        items = BH * NC

        def alloc(shape):
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )

        outs = [alloc([BH, NC, C, C]) for _ in range(6)] + [alloc([BH, NC, 1, 1]), alloc([BH, NC, C, 32])]
        grid = device.compute_with_storage_grid_size()
        ncores = min(grid.x * grid.y, items)
        coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        ins = [q, k, v, beta, g, self.consts]
        rd_c = [t.buffer_address() for t in ins] + [items, ncores, grid.x, NC, T // 32]
        if tm:
            rd_c += [tm["Wt"], tm["rf"], tm["q_off"], tm["k_off"], tm["v_off"]]
        wr_c = [t.buffer_address() for t in outs] + [items, ncores, grid.x]
        cp_c = [items, ncores, grid.x, _f32_bits(alpha)]
        cp_c += [_f32_bits(tm["eps"]), _f32_bits(tm["scale"])] if tm else [0, 0]
        dm = [("TOKEN_MAJOR", "1")] if tm else []
        rd_ct = [8]
        for t in ins:
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = []
        for t in outs:
            wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        pages = {0: 16, 1: 16, 2: 8, 3: 4, 4: 4, 5: 8, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 10, 13: 16}
        pages.update({14: 16, 15: 10, 16: 4, 18: 4, 19: 4, 20: 4, 21: 4, 23: 4, 24: 8, 25: 1, 26: 4, 27: 1})
        if tm:
            pages.update({17: 1, 28: 16, 29: 16})
        bf16_in = {0, 1, 2} if tm else set()

        def cbd(i, p):
            fmt, size = (ttnn.bfloat16, 2048) if i in bf16_in else (ttnn.float32, 4096)
            return ttnn.CBDescriptor(
                total_size=size * p,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=size)],
            )

        cbs = [cbd(i, p) for i, p in pages.items()]
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "prep_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=rd_c,
                    defines=dm,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "prep_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    common_runtime_args=wr_c,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "prep_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    common_runtime_args=cp_c,
                    defines=dm,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                        math_approx_mode=False,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op(ins + outs, program)
        return outs


_PREP = {}


def chunk_gated_delta_rule_seq_fused(
    q, k, v, beta, g, chunk_size=128, scale=None, initial_state=None, mesh_device=None, cached_masks=None,
    valid_len=None,
):  # fmt: skip
    """Same contract as ttnn_delta_rule_seq.chunk_gated_delta_rule_seq (returns o [BH, T, V], final_state)."""
    import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as seq

    BH, T, K = list(q.shape)
    whole = valid_len is None or (not isinstance(valid_len, (list, tuple)) and valid_len >= T)
    if (
        chunk_size != C
        or T % C
        or K != 128
        or list(v.shape)[2] != 128
        or not whole
        or seq._os.environ.get("QWEN_GDN_INV_DOUBLING", "0") != "0"
    ):
        return _ORIG(q, k, v, beta, g, chunk_size, scale, initial_state, mesh_device, cached_masks, valid_len)
    if scale is None:
        scale = K**-0.5
    alpha = float(os.environ.get("QWEN_GDN_DIAG_ALPHA", "0.25"))
    # One instance per process (one device): the GDN code may pass a different Python object for the same device,
    # and a lazy first build inside a trace capture would write the constant tiles where the replay never sees them.
    key = "device"
    if key not in _PREP:
        _PREP[key] = ChunkPrep(mesh_device)
    qs = ttnn.multiply(q, scale)
    beta3 = ttnn.reshape(beta, [BH, T, 1])
    L_unit, v_beta_sc, k_bd_sc, intra, q_decay, k_decay_t, dl_exp, L_inv = _PREP[key](qs, k, v, beta3, g, alpha)
    ttnn.deallocate(qs)
    S0 = None
    if initial_state is not None:
        S0 = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, 128]), ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    out_4d, final_state = ttnn.transformer.gated_delta_attn_seq(
        L_unit, v_beta_sc, k_bd_sc, intra, q_decay, k_decay_t, dl_exp, L_inv,
        initial_state=S0, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # fmt: skip
    for t in (L_unit, v_beta_sc, k_bd_sc, intra, q_decay, k_decay_t, dl_exp, L_inv):
        ttnn.deallocate(t)
    if out_4d.dtype != ttnn.float32:
        out_4d = ttnn.typecast(out_4d, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.reshape(out_4d, [BH, T, 128], memory_config=ttnn.L1_MEMORY_CONFIG)
    return o, final_state


_ORIG = None


def install(device=None):
    """Route ttnn_delta_rule_seq.chunk_gated_delta_rule_seq (used by its adapter) through the fused preprocessing.
    device: build the constant tiles now (a host write inside a trace capture would not be replayed)."""
    global _ORIG
    import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as seq

    if device is not None and "device" not in _PREP:
        _PREP["device"] = ChunkPrep(device)

    if _ORIG is None:
        _ORIG = seq.chunk_gated_delta_rule_seq
        seq.chunk_gated_delta_rule_seq = chunk_gated_delta_rule_seq_fused
