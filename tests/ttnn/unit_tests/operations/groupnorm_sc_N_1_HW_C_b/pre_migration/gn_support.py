# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared case builder, independent reference and output-contract checks for the groupnorm_sc_N_1_HW_C_b
pre-migration suite. Nothing here imports the op's planner or descriptor: the op is only ever called through the
route fixture the test passes in.

Reference: the golden suite's ``pytorch_groupnorm_sc_N_1_HW_C`` (fp32 torch GroupNorm, result in the input dtype),
fed the values the device actually holds (``ttnn.to_torch`` of the input, gamma and beta before the call: exact for
bf16 / fp32, the quantized values for bfloat8_b). Tolerances: the golden ``helpers.TOLERANCES`` per dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
    TOLERANCES,
    block_shard_memory_config,
    pytorch_groupnorm_sc_N_1_HW_C,
)
from eval.metrics import check_output

BF16, F32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DEFAULT_EPS = 1e-5

TORCH_DTYPE = {BF16: torch.bfloat16, F32: torch.float32, BF8: torch.bfloat16}


def _round_up(n, m):
    return -(-n // m) * m


@dataclass(frozen=True)
class Case:
    """One call of the op. ``grid`` = (grid_x, grid_y) of an L1 BLOCK_SHARDED input (ROW_MAJOR orientation, the
    golden ``block_shard_memory_config`` geometry); None = DRAM interleaved."""

    shape: tuple
    num_groups: int
    dtype: object = BF16
    layout: object = TILE
    affine: str = "gamma_beta"  # gamma_beta | gamma_only | no_affine
    affine_dtype: object = BF16
    affine_layout: object = RM
    grid: tuple | None = None
    in_place: bool = False
    eps: float = DEFAULT_EPS
    pad_fill: float | None = None  # DRAM TILE only: written into the tile padding of a logical-shape view
    pattern: str = "groups"  # groups: per-group scale / offset and a per-image shift; offset10: N(10, 1) (mean >> std)
    tolerance: tuple | None = None  # (pcc, rel-RMS); None = the golden TOLERANCES[dtype]
    canary: bool = False  # sharded only: a neighbour L1 tensor with the input's spec that must stay untouched

    @property
    def sharded(self):
        return self.grid is not None

    def with_(self, **kw):
        return replace(self, **kw)


def torch_tensors(case: Case, seed: int):
    """(x, gamma, beta) on the host, in the torch dtype of each ttnn dtype. ``groups``: per-group offsets in [-2, 2),
    scales in [0.5, 1.5) and a per-image shift make a group / image mix-up move the result far outside tolerance.
    ``offset10``: the golden regression's offset-heavy activations (N(0, 1) + 10), where a one-pass E[x^2] - E[x]^2
    variance loses the signal."""
    N, _, HW, C = case.shape
    G = case.num_groups
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(case.shape, generator=gen, dtype=torch.float32)
    if case.pattern == "groups":
        Cg = C // G
        scale = (0.5 + torch.rand(G, generator=gen)).repeat_interleave(Cg)
        offset = (torch.rand(G, generator=gen) * 4.0 - 2.0).repeat_interleave(Cg)
        image = torch.linspace(-0.5, 0.5, N).view(N, 1, 1, 1) if N > 1 else torch.zeros(1)
        x = x * scale + offset + image
    elif case.pattern == "offset10":
        x = x + 10.0
    else:
        raise ValueError(f"unknown pattern {case.pattern!r}")
    x = x.to(TORCH_DTYPE[case.dtype])
    gamma = beta = None
    if case.affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn((1, 1, 1, C), generator=gen).to(TORCH_DTYPE[case.affine_dtype])
    if case.affine == "gamma_beta":
        beta = torch.randn((1, 1, 1, C), generator=gen).to(TORCH_DTYPE[case.affine_dtype])
    return x, gamma, beta


def input_memory_config(case: Case, device):
    if not case.sharded:
        return ttnn.DRAM_MEMORY_CONFIG
    return block_shard_memory_config(case.shape, case.grid, layout=case.layout, dtype=case.dtype, device=device)


def to_device_input(case: Case, x, device, memory_config):
    if case.pad_fill is None:
        return ttnn.from_torch(x, dtype=case.dtype, layout=case.layout, device=device, memory_config=memory_config)
    assert case.layout == TILE and not case.sharded, "pad_fill is a DRAM TILE case"
    N, _, HW, C = case.shape
    Hp, Cp = _round_up(HW, 32), _round_up(C, 32)
    padded = torch.full((N, 1, Hp, Cp), case.pad_fill, dtype=x.dtype)
    padded[..., :HW, :C] = x
    t = ttnn.from_torch(padded, dtype=case.dtype, layout=TILE, device=device, memory_config=memory_config)
    t = ttnn.reshape(t, [N, 1, HW, C], [N, 1, Hp, Cp])
    assert tuple(t.shape) == tuple(case.shape)
    return t


def to_device_affine(t, case: Case, device):
    if t is None:
        return None
    return ttnn.from_torch(
        t, dtype=case.affine_dtype, layout=case.affine_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def expect_dispatches(op, n=1):
    """Source route: the op is one ``ttnn.generic_op`` per call (0 when it refuses). Native route: never any."""
    want = n if op.mode == "source" else 0
    assert op.last_dispatches == want, f"{op.mode} call made {op.last_dispatches} generic_op dispatches, want {want}"


class Prepared:
    """A case's device tensors, its reference and the state the output contract is checked against."""

    def __init__(self, case: Case, device, seed: int = 0):
        self.case = case
        self.device = device
        x, gamma, beta = torch_tensors(case, seed)
        self.in_cfg = input_memory_config(case, device)
        self.tt_x = to_device_input(case, x, device, self.in_cfg)
        self.canary = self.canary_host = None
        if case.canary:
            assert case.sharded
            self.canary_host = torch.full(tuple(case.shape), 7.25, dtype=TORCH_DTYPE[case.dtype])
            self.canary = ttnn.from_torch(
                self.canary_host, dtype=case.dtype, layout=case.layout, device=device, memory_config=self.in_cfg
            )
        self.tt_gamma = to_device_affine(gamma, case, device)
        self.tt_beta = to_device_affine(beta, case, device)
        # The values the device holds: the reference's inputs and the "untouched" baseline.
        self.x_host = ttnn.to_torch(self.tt_x)
        self.gamma_host = None if gamma is None else ttnn.to_torch(self.tt_gamma)
        self.beta_host = None if beta is None else ttnn.to_torch(self.tt_beta)
        self.expected = reference(case, self.x_host, self.gamma_host, self.beta_host)
        self.in_addr = self.tt_x.buffer_address()
        self.in_padded_shape = list(self.tt_x.padded_shape)

    def kwargs(self, **override):
        kw = {}
        if self.tt_gamma is not None:
            kw["gamma"] = self.tt_gamma
        if self.tt_beta is not None:
            kw["beta"] = self.tt_beta
        if self.case.eps != DEFAULT_EPS:
            kw["eps"] = self.case.eps
        if self.case.in_place:
            kw["in_place"] = True
        kw.update(override)
        return kw

    def call(self, op, **override):
        out = op(self.tt_x, self.case.num_groups, **self.kwargs(**override))
        expect_dispatches(op)
        return out

    def check(self, out, expected=None, tolerance=None):
        check_result(self, out, self.expected if expected is None else expected, tolerance)


def reference(case: Case, x, gamma, beta, eps=None):
    return pytorch_groupnorm_sc_N_1_HW_C(
        x, case.num_groups, gamma=gamma, beta=beta, eps=case.eps if eps is None else eps
    )


def expected_padded_shape(shape, layout):
    shape = list(shape)
    if layout == TILE:
        shape[-2], shape[-1] = _round_up(shape[-2], 32), _round_up(shape[-1], 32)
    return shape


def check_result(p: Prepared, out, expected, tolerance=None):
    """The whole output contract of one call: placement, metadata, values, memory effects."""
    case = p.case
    assert isinstance(out, ttnn.Tensor), f"the op returns one ttnn.Tensor, got {type(out)}"
    cfg = out.memory_config()
    out_layout = case.layout if case.sharded else TILE
    if case.sharded:
        assert cfg.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED, cfg
        assert cfg.buffer_type == ttnn.BufferType.L1, cfg
        in_ss, out_ss = p.in_cfg.shard_spec, cfg.shard_spec
        assert list(out_ss.shape) == list(in_ss.shape), f"shard shape {out_ss.shape} != input {in_ss.shape}"
        assert out_ss.grid == in_ss.grid, f"shard grid {out_ss.grid} != input {in_ss.grid}"
        assert out_ss.orientation == in_ss.orientation
    else:
        assert cfg.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, cfg
        assert cfg.buffer_type == ttnn.BufferType.DRAM, cfg
    assert out.storage_type() == ttnn.StorageType.DEVICE
    # A sharded output has the input's spec, padding included (a ROW_MAJOR block shard pads the width to
    # grid_x x shard width); a DRAM output is TILE, padded to whole tiles.
    want_padded = p.in_padded_shape if case.sharded else expected_padded_shape(case.shape, out_layout)
    assert list(out.padded_shape) == want_padded, f"padded shape {out.padded_shape}, want {want_padded}"
    if case.in_place:
        assert out.buffer_address() == p.in_addr, "in_place=True must return the input's own buffer"
    else:
        assert out.buffer_address() != p.in_addr, "an out-of-place output must not share the input's buffer"
    check_output(
        out,
        expected,
        shape=list(case.shape),
        dtype=case.dtype,
        expected_layout=out_layout,
        tolerance=tolerance or case.tolerance or TOLERANCES[case.dtype],
    )
    # Memory effects: out of place, the input is untouched; gamma / beta are never written; nothing past the
    # input's own shards is written (the canary is the neighbouring L1 allocation of the same spec).
    if not case.in_place:
        assert torch.equal(ttnn.to_torch(p.tt_x), p.x_host), "the op modified its (out-of-place) input"
    if p.gamma_host is not None:
        assert torch.equal(ttnn.to_torch(p.tt_gamma), p.gamma_host), "the op modified gamma"
    if p.beta_host is not None:
        assert torch.equal(ttnn.to_torch(p.tt_beta), p.beta_host), "the op modified beta"
    if p.canary is not None:
        assert torch.equal(ttnn.to_torch(p.canary), p.canary_host), "the op wrote outside its own shards"


def fresh_cache(device):
    """A known starting state: caching on, no entries."""
    device.enable_program_cache()
    device.clear_program_cache()
    assert device.num_program_cache_entries() == 0


def entries(device):
    return device.num_program_cache_entries()
