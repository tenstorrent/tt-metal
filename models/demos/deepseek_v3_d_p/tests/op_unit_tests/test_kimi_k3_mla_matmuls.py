# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-op matmul unit tests for the Kimi-K3 MLA at its exact chunked-prefill per-device shapes.

Why a separate file from ``test_mla_matmuls.py``: that file is the 128k/100k **single-shot** set --
module-level ``NUM_HEADS = 128``, seq lens 8192/6400, a flat 20-field parametrize tuple, and
``mesh_device`` pinned to ``(8, 4)``. K3's set is the **chunked 640** set. Expressing it there would
mean fighting those constants; here each case is named and the geometry is stated once.

Shapes are **per device** at ``S_loc = 640``, ``tp = 4``, ``H_loc = 96/4 = 24`` -- the geometry of
the op audit in ``models/demos/deepseek_v3_d_p/docs/KIMI_K3_MLA.md`` §3. Those per-device shapes are
identical on SP8xTP4 (Galaxy) and SP2xTP4 (an 8-chip 2x4 box), so this file runs on either;
weights and activations are replicated and every device runs the same matmul. That is deliberate:
the question here is "is this program config valid and accurate at this shape", which is what a
tuning sweep needs, with no CCL in the measurement. End-to-end sharding correctness is covered by
``tests/test_mla.py::test_kimi_k3_mla`` and the gate epilogue by ``test_kimi_k3_gate.py``.

What actually changes vs Kimi-K2.6 (F3 in the doc says "all six tuned 640 configs are invalidated",
which is true of the *gating tag* but overstates the *work*):

  * ``q_a_proj``, ``kv_a_proj_with_mqa`` -- per-device shapes are **identical** to K2.6
    (K = hidden/tp = 1792 either way). K2.6's 640 configs should transfer verbatim; they are only
    skipped because they declare ``num_heads: 64``.
  * ``o_proj`` -- K widens 2048 -> 3072, but N is the full 7168 (K-sharded via mapper_tp0) for both,
    and K2.6's ``in0_block_w=8`` divides both K_t=64 and K_t=96. Expect it to transfer too.
  * ``q_b_proj`` -- N genuinely widens 3072 -> 4608 per device. Needs a new config.
  * ``wkv_b1`` / ``wkv_b2`` -- batched, and the batch is ``H_loc``: 16 -> 24. K2.6's ``per_core_M=4``
    puts ``24 * (20/4) = 120`` blocks on a 110-core grid, so these genuinely overflow and need
    ``per_core_M >= 5``. This is the one place the batch increase is not free.
  * ``g_proj`` -- new op.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG

PCC_REQUIRED = 0.99

# Available core grid is 12x10, but due to di/dt and throttling problems the model uses 11x10.
COMPUTE_GRID = (11, 10)
NUM_CORES = COMPUTE_GRID[0] * COMPUTE_GRID[1]

# --- Kimi-K3 per-device geometry at tp=4 (see module docstring) ---
TP = 4
HIDDEN = 7168
HIDDEN_LOC = HIDDEN // TP  # 1792
NUM_HEADS = 96
HEADS_LOC = NUM_HEADS // TP  # 24
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
S_LOC = 640

TILE = 32


@dataclass(frozen=True)
class MMCase:
    """One MLA matmul at its per-device shape, with the program config under test."""

    name: str
    # in0 / in1 as [b, batch, M, K] x [b, batch, K, N]; batch > 1 => a batched (per-head) matmul.
    batch: int
    m: int
    k: int
    n: int
    program_config: object
    out_dtype: object = ttnn.bfloat16
    act_mem_config: object = ttnn.DRAM_MEMORY_CONFIG
    out_mem_config: object = ttnn.L1_MEMORY_CONFIG
    # Expected host-side activation applied to the reference when the config fuses one.
    fused_activation_ref: Optional[str] = None

    @property
    def is_batched(self) -> bool:
        return self.batch > 1


def _mcast_2d(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fused_activation=None):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=COMPUTE_GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=fused_activation,
    )


# Per-device shapes, from the op audit (docs/KIMI_K3_MLA.md §3). Program configs are NOT written here
# -- they are pulled from mla_config.py through the same resolution ttMLA uses, so this test always
# exercises what actually ships and cannot drift from it.
#
#   (weight name in MLA_MATMUL_CONFIG, batch, M, K, N)
K3_MM_SHAPES = [
    # hidden_states @ q_a_proj. K = hidden/tp; unchanged from K2.6.
    ("q_a_proj", 1, S_LOC, HIDDEN_LOC, Q_LORA_RANK),
    # q_a latent @ q_b_proj. N widens 3072 -> 4608 per device (N_t 96 -> 144).
    ("q_b_proj", 1, S_LOC, Q_LORA_RANK, HEADS_LOC * QK_HEAD_DIM),
    # hidden_states @ kv_a_proj_with_mqa; unchanged from K2.6.
    ("kv_a_proj_with_mqa", 1, S_LOC, HIDDEN_LOC, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
    # q_nope @ wkv_b1, batched over the 24 local heads (K2.6: 16).
    ("wkv_b1", HEADS_LOC, S_LOC, QK_NOPE_HEAD_DIM, KV_LORA_RANK),
    # sdpa_out @ wkv_b2, batched over 24 heads.
    ("wkv_b2", HEADS_LOC, S_LOC, KV_LORA_RANK, V_HEAD_DIM),
    # concat_heads out @ o_proj. K widens 2048 -> 3072; N is the full 7168 either way.
    ("o_proj", 1, S_LOC, HEADS_LOC * V_HEAD_DIM, HIDDEN),
    # NEW: all-gathered hidden @ g_proj (output gate), with sigmoid fused into the matmul.
    ("g_proj", 1, S_LOC, HIDDEN, NUM_HEADS * V_HEAD_DIM // TP),
]


def _k3_mla_stub():
    """A ttMLA carrying only what the config resolvers read, tagged as Kimi-K3.

    Resolving through ttMLA rather than indexing MLA_MATMUL_CONFIG directly is the point: a slot may
    hold several candidates and only _select_cfg / _cfg_matches know which one K3 gets.
    """
    mla = object.__new__(ttMLA)
    mla.num_heads = NUM_HEADS
    mla.q_lora_rank = Q_LORA_RANK
    mla.is_chunked = True
    mla._is_dsa_family = False
    # _resolve_mm_cfg (used by _gate_sigmoid_fused) indexes this the way __init__ builds it.
    mla.mm_configs = {name: MLA_MATMUL_CONFIG.get(name, {}) for name, *_ in K3_MM_SHAPES}
    return mla


def _build_cases():
    stub = _k3_mla_stub()
    cases = []
    for name, batch, m, k, n in K3_MM_SHAPES:
        cfg = stub._select_cfg(MLA_MATMUL_CONFIG.get(name, {}).get(S_LOC))
        assert cfg is not None, f"no Kimi-K3 tuned config for {name!r} at seq_len_local={S_LOC}"
        pc = cfg["program_config"]
        fused = getattr(pc, "fused_activation", None)
        cases.append(
            MMCase(
                name=name,
                batch=batch,
                m=m,
                k=k,
                n=n,
                program_config=pc,
                out_dtype=cfg["out_dtype"],
                act_mem_config=cfg["act_mem_config"],
                out_mem_config=cfg["out_mem_config"],
                fused_activation_ref="sigmoid" if fused is not None else None,
            )
        )
    # g_proj without the fused activation: ttMLA's untuned path applies a standalone ttnn.sigmoid
    # instead (see _gate_sigmoid_fused), so both forms need coverage.
    g = next(c for c in cases if c.name == "g_proj")
    cases.append(
        MMCase(
            name="g_proj_no_fusion",
            batch=g.batch,
            m=g.m,
            k=g.k,
            n=g.n,
            program_config=_mcast_2d(
                in0_block_w=g.program_config.in0_block_w,
                out_subblock_h=g.program_config.out_subblock_h,
                out_subblock_w=g.program_config.out_subblock_w,
                per_core_M=g.program_config.per_core_M,
                per_core_N=g.program_config.per_core_N,
            ),
            out_dtype=g.out_dtype,
            act_mem_config=g.act_mem_config,
            out_mem_config=g.out_mem_config,
            fused_activation_ref=None,
        )
    )
    return cases


K3_MM_CASES = _build_cases()


def test_k3_g_proj_config_fuses_sigmoid():
    """The shipped g_proj config must fuse sigmoid.

    ttMLA keys off exactly this (``_gate_sigmoid_fused``) to decide whether to apply a standalone
    ``ttnn.sigmoid``. If the fused activation were dropped from the config without ttMLA noticing,
    the gate would silently stop being a gate.
    """
    stub = _k3_mla_stub()
    cfg = stub._select_cfg(MLA_MATMUL_CONFIG["g_proj"][S_LOC])
    assert cfg is not None
    fused = getattr(cfg["program_config"], "fused_activation", None)
    assert fused is not None, "shipped Kimi-K3 g_proj config no longer fuses an activation"
    assert fused.op_type == ttnn.UnaryOpType.SIGMOID, f"g_proj fuses {fused.op_type}, expected SIGMOID"
    assert stub._gate_sigmoid_fused(S_LOC), "ttMLA does not agree that the shipped g_proj config fuses sigmoid"


def test_k3_mm_cases_fit_the_grid():
    """Pure arithmetic: no case may ask for more blocks than the grid has cores.

    Runs without a device so a bad program config is caught at collection speed rather than as an
    opaque device-side failure. The batched (MatmulMultiCoreReuse) path distributes
    ``batch * (M_t/per_core_M) * (N_t/per_core_N)`` blocks over cores; the 2D multicast path lays
    ``per_core_M`` down the 10 rows and ``per_core_N`` across the 11 columns.
    """
    failures = []
    for case in K3_MM_CASES:
        m_t, k_t, n_t = case.m // TILE, case.k // TILE, case.n // TILE
        pc = case.program_config
        if case.is_batched:
            blocks = case.batch * -(-m_t // pc.per_core_M) * -(-n_t // pc.per_core_N)
            if blocks > NUM_CORES:
                failures.append(f"{case.name}: {blocks} blocks > {NUM_CORES} cores")
        else:
            if pc.per_core_M * COMPUTE_GRID[1] < m_t:
                failures.append(f"{case.name}: per_core_M {pc.per_core_M} x {COMPUTE_GRID[1]} rows < M_t {m_t}")
            if pc.per_core_N * COMPUTE_GRID[0] < n_t:
                failures.append(f"{case.name}: per_core_N {pc.per_core_N} x {COMPUTE_GRID[0]} cols < N_t {n_t}")
        if k_t % pc.in0_block_w:
            failures.append(f"{case.name}: in0_block_w {pc.in0_block_w} does not divide K_t {k_t}")
        if pc.out_subblock_h * pc.out_subblock_w > 8:
            failures.append(f"{case.name}: out_subblock {pc.out_subblock_h}x{pc.out_subblock_w} > 8 tiles")
    assert not failures, "program config / shape mismatches:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("mesh_device", [(2, 4), (8, 4)], ids=["2x4", "8x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("case", K3_MM_CASES, ids=[c.name for c in K3_MM_CASES])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 matmul configs are tuned for Blackhole")
def test_kimi_k3_mla_mm(mesh_device, case: MMCase):
    """One Kimi-K3 MLA matmul at its per-device shape, against a torch reference."""
    torch.manual_seed(42)
    in0 = torch.randn(1, case.batch, case.m, case.k, dtype=torch.bfloat16)
    in1 = torch.randn(1, case.batch, case.k, case.n, dtype=torch.bfloat16) * 0.02

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_in0 = ttnn.from_torch(
        in0,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=case.act_mem_config,
        mesh_mapper=replicate,
    )
    tt_in1 = ttnn.from_torch(
        in1,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,  # every MLA projection is bf8 on device
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    # Same compute kernel config ttMLA uses (mla.py: default_compute_kernel_config).
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    tt_out = ttnn.linear(
        tt_in0,
        tt_in1,
        memory_config=case.out_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=case.out_dtype,
        program_config=case.program_config,
    )
    ttnn.synchronize_device(mesh_device)

    expected = torch.matmul(in0.float(), in1.float())
    if case.fused_activation_ref == "sigmoid":
        expected = torch.sigmoid(expected)
    elif case.fused_activation_ref is not None:
        raise AssertionError(f"unhandled fused activation {case.fused_activation_ref!r}")

    # Every device ran the same replicated matmul; check device 0's copy.
    actual = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]

    assert tuple(actual.shape) == (1, case.batch, case.m, case.n), f"shape {tuple(actual.shape)}"
    passing, pcc = comp_pcc(expected, actual.float(), PCC_REQUIRED)
    logger.info(f"K3 {case.name}: [{case.batch},{case.m},{case.k}] @ [{case.k},{case.n}] PCC={pcc}")
    assert passing, f"K3 {case.name} PCC {pcc} < {PCC_REQUIRED}"


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 matmul configs are tuned for Blackhole")
def test_k3_g_proj_fused_sigmoid_matches_standalone(mesh_device):
    """g_proj with sigmoid fused must equal g_proj followed by a standalone ttnn.sigmoid.

    ``ttMLA`` picks between these two at runtime (``_gate_sigmoid_fused``, keyed on whether the
    resolved tuned config carries a ``fused_activation``), so they have to agree -- otherwise
    promoting a tuned config into ``mla_config.py`` silently changes the model's numerics, or the
    model double-applies sigmoid.
    """
    fused_case = next(c for c in K3_MM_CASES if c.name == "g_proj")
    plain_case = next(c for c in K3_MM_CASES if c.name == "g_proj_no_fusion")
    assert fused_case.fused_activation_ref == "sigmoid" and plain_case.fused_activation_ref is None

    torch.manual_seed(42)
    in0 = torch.randn(1, 1, fused_case.m, fused_case.k, dtype=torch.bfloat16)
    in1 = torch.randn(1, 1, fused_case.k, fused_case.n, dtype=torch.bfloat16) * 0.02

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    kwargs = dict(device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)
    tt_in0 = ttnn.from_torch(in0, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)
    tt_in1 = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    linear_kwargs = dict(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )

    fused = ttnn.linear(tt_in0, tt_in1, program_config=fused_case.program_config, **linear_kwargs)
    plain = ttnn.linear(tt_in0, tt_in1, program_config=plain_case.program_config, **linear_kwargs)
    standalone = ttnn.sigmoid(plain, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    fused_torch = ttnn.to_torch(fused, mesh_composer=composer)[:1].float()
    standalone_torch = ttnn.to_torch(standalone, mesh_composer=composer)[:1].float()

    # Both round through bf16 twice; they should be near-identical, not merely correlated.
    passing, pcc = comp_pcc(standalone_torch, fused_torch, 0.9999)
    max_abs = (standalone_torch - fused_torch).abs().max().item()
    logger.info(f"K3 g_proj fused vs standalone sigmoid: PCC={pcc} max_abs={max_abs}")
    assert passing, f"fused sigmoid disagrees with standalone: PCC {pcc}, max_abs {max_abs}"
