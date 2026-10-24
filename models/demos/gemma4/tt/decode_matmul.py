# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit 1D-multicast program configs for the dense decode matmuls (opt-in).

A plain ``ttnn.linear`` on a one-tile-high activation picks the 1D systolic config with ``in0_block_w=1``:
the activation is multicast one K tile at a time, so a K=2816 projection runs 88 multicast rounds and
reads its weight at ~15% of the DRAM peak (26B-A4B TP=4: gate_up 43 us for 6.1 MB, wqkv 52 us for 11.5 MB,
router 35 us for 0.7 MB, measured on a P150 with trace replay). The same kernel with a wider K block
(``in0_block_w`` = largest divisor of the K tiles up to ``GEMMA4_DECODE_MM_IN0_BLOCK_W``, default 8) and
one N tile per core over the whole worker grid halves those times.

A matmul with an explicit program config silently drops the default math fidelity from HiFi2 to LoFi
(``create_matmul_attributes``), so the config comes with an explicit HiFi2 compute config. The K blocks
accumulate in fp32 Dest (``fp32_dest_acc_en``): with the auto path's packer-L1 accumulation a wide block
sums its tiles in 16-bit Dest and the per-op PCC vs an fp32 reference falls from 0.99995 (auto) to
0.99976 (22-tile block); with fp32 Dest it is 0.99999 for every block width at the same speed.
``GEMMA4_DECODE_MM_FP32_ACC=0`` selects the packer-L1 accumulation instead.

Per-site opt-in flags (all default off): ``GEMMA4_MLP_DECODE_1D`` (shared MLP gate_up / down),
``GEMMA4_ATTN_DECODE_1D`` (wqkv / o_proj), ``GEMMA4_ROUTER_DECODE_1D`` (router projection).
"""

import math
import os

import ttnn
from models.demos.gemma4.tt.mesh_orientation import decode_flag

TILE = ttnn.TILE_SIZE


def flag(name: str) -> bool:
    """A decode fusion switch: the env var when set, else the build profile's default (on for the 4-chip Blackhole
    MoE build, off elsewhere; tt/mesh_orientation.py)."""
    return decode_flag(name) == "1"


def is_decode_rows(x) -> bool:
    """True for a ``[1, 1, M<=32, K]`` activation (one tile of rows): the only shape the 1D decode config serves."""
    shape = x.shape
    return len(shape) == 4 and int(shape[0]) == 1 and int(shape[1]) == 1 and 1 <= int(shape[2]) <= TILE


def _largest_divisor(n: int, max_div: int) -> int:
    for d in range(max(1, min(n, max_div)), 0, -1):
        if n % d == 0:
            return d
    return 1


def decode_in0_block_w(k: int, max_block_w=None) -> int:
    """K-tile block width: the largest divisor of the K tiles not above ``max_block_w``
    (``GEMMA4_DECODE_MM_IN0_BLOCK_W``, default 8)."""
    if max_block_w is None:
        max_block_w = max(1, int(os.environ.get("GEMMA4_DECODE_MM_IN0_BLOCK_W", "8")))
    return _largest_divisor(math.ceil(k / TILE), max_block_w)


def decode_1d_progcfg(mesh_device, k: int, n: int, max_block_w=None):
    """1D mcast_in0 config for ``[1, 1, <=32, k] @ [k, n]``: one N tile per core across the device's
    worker grid (more when N has more tiles than cores), K streamed in ``decode_in0_block_w`` blocks."""
    grid = mesh_device.compute_with_storage_grid_size()
    n_tiles = math.ceil(n / TILE)
    per_core_n = max(1, math.ceil(n_tiles / (grid.x * grid.y)))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=decode_in0_block_w(k, max_block_w),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_compute_kernel_config(mesh_device):
    """HiFi2 (the auto path's fidelity) with fp32 Dest accumulation across the K blocks (default); the
    auto path's packer-L1 accumulation on ``GEMMA4_DECODE_MM_FP32_ACC=0``. ``GEMMA4_DECODE_MM_FIDELITY=lofi``
    selects LoFi, the fidelity every matmul with an explicit program config and no compute config runs at
    (``create_matmul_attributes``: the DRAM-sharded decode matmuls and the lm_head of the dense variants)."""
    fp32_acc = os.environ.get("GEMMA4_DECODE_MM_FP32_ACC", "1") == "1"
    fidelity = os.environ.get("GEMMA4_DECODE_MM_FIDELITY", "hifi2").lower()
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi if fidelity == "lofi" else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=not fp32_acc,
    )


class DecodeLinear:
    """``ttnn.linear`` on an interleaved weight with the 1D decode config for one-tile activations and the
    op default for everything else (prefill rows, batched leading dims)."""

    def __init__(self, weight, mesh_device, k: int, n: int, enabled: bool, max_block_w=None):
        self.weight = weight
        self.k = k
        self.n = n
        self.program_config = decode_1d_progcfg(mesh_device, k, n, max_block_w) if enabled else None
        self.compute_kernel_config = decode_compute_kernel_config(mesh_device) if enabled else None

    def __call__(self, x, memory_config=None, compute_kernel_config=None):
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = memory_config
        if self.program_config is not None and is_decode_rows(x):
            kwargs["program_config"] = self.program_config
            kwargs["compute_kernel_config"] = (
                compute_kernel_config if compute_kernel_config is not None else self.compute_kernel_config
            )
        elif compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        return ttnn.linear(x, self.weight, **kwargs)


# --- DRAM-sharded decode kernel: the multi-core pipeline rows ----------------------------------------------------
#
# The DRAM-sharded decode kernel (tt/dram_sharded.py) streams each weight shard with ONE reader core per DRAM bank;
# its multi-core pipeline (``use_multicore_pipeline`` with ``num_workers_per_dram_bank`` >= 2) puts 2-4 readers on
# every bank and reads the activation straight from its L1 shards. On the dense 31B / 12B per-chip shapes at the
# shipped decode layouts it is the one program-config knob that moves every DRAM-sharded matmul (single-chip knob
# lab, traced, PCC vs the fp32 golden >= the LoFi kernel of today; _logs/g7_matmul_knob_lab.md, 2026-10-24): 31B
# tp4 wqkv 54.2 -> 53.2 us, o_proj 33.1 -> 29.1, gate_up 128.8 -> 124.4, down 69.3 -> 65.5; tp8 o_proj 21.5 -> 17.8,
# gate_up 67.3 -> 63.4, down 40.1 -> 33.5 (wqkv tp8: no gain, no row). Rows are keyed by (model tag, op, tp);
# Wormhole and any grid of <= 64 worker cores keep the single-reader kernel (the pipeline needs workers x 8 banks
# worker cores and was measured on the 12x10 / 13x10 Blackhole grids only). ``GEMMA4_DRAM_SHARD_PIPELINE`` is the
# switch: ``0`` = the single-reader kernel everywhere; ``1`` = the pipeline with fp32 Dest accumulation
# (``pipeline_fp32_acc``); ``l1acc`` = the pipeline with the packer-L1 accumulation (perf reference only); the
# default is ``PIPELINE_DEFAULT`` (off until the model-level traced A/B and the token-accuracy subset pass);
# a comma list of op names (``gate_up,down``) restricts the rows to those ops (``pipeline_ops_allowed``).

PIPELINE_ENV = "GEMMA4_DRAM_SHARD_PIPELINE"
PIPELINE_DEFAULT = "0"
MODEL_TAG_BY_HIDDEN = {5376: "31B", 3840: "12B"}
MIN_PIPELINE_GRID_CORES = 65
# (model, op, tp) -> (workers per DRAM bank, in0_block_w)
DRAM_SHARDED_PIPELINE_ROWS = {
    ("31B", "qkv", 4): (3, 2),
    ("31B", "out_proj", 4): (4, 4),
    ("31B", "out_proj", 8): (4, 4),
    ("31B", "gate_up", 4): (4, 1),
    ("31B", "gate_up", 8): (4, 1),
    ("31B", "down", 4): (4, 1),
    ("31B", "down", 8): (4, 1),
    ("12B", "down", 4): (3, 1),
    ("12B", "down", 8): (4, 2),
    ("12B", "gate_up", 4): (3, 2),
    ("12B", "gate_up", 8): (4, 1),
    ("12B", "out_proj", 4): (4, 4),
    ("12B", "out_proj", 8): (4, 2),
}


def model_tag(hidden_size):
    """The dense model a per-chip shape belongs to, from its hidden size (5376 = 31B, 3840 = 12B), else None."""
    return MODEL_TAG_BY_HIDDEN.get(int(hidden_size))


def dram_sharded_pipeline_row(hidden_size, op: str, tp: int, grid_xy, blackhole: bool, default=None):
    """``(workers per bank, in0_block_w)`` for a DRAM-sharded decode matmul, or None for the single-reader kernel:
    the switch, Wormhole, a grid of fewer than ``MIN_PIPELINE_GRID_CORES`` worker cores, a model / op / tp without a
    row, or a row whose readers would not fit the grid."""
    env = os.environ.get(PIPELINE_ENV, default if default is not None else PIPELINE_DEFAULT)
    if env.strip().lower() in ("0", "false", "no", "off", ""):
        return None
    if not blackhole:
        return None
    gx, gy = int(grid_xy[0]), int(grid_xy[1])
    if gx * gy < MIN_PIPELINE_GRID_CORES:
        return None
    row = DRAM_SHARDED_PIPELINE_ROWS.get((model_tag(hidden_size), op, int(tp)))
    if row is None or row[0] * 8 > gx * gy:
        return None
    if not pipeline_ops_allowed(op):
        return None
    return row


def pipeline_fp32_acc() -> bool:
    """Whether pipelined rows accumulate the whole K in fp32 Dest (the switch's ``1`` / ``on``; no L1 partials, more
    accurate than the single-reader kernel). ``GEMMA4_DRAM_SHARD_PIPELINE=l1acc`` keeps the packer-L1 form (16-bit
    Dest groups, Float16_b L1 accumulator, bf16 partial exchange across the bank's workers): faster on the lab's
    random data but 4 / 385 top-1 tokens worse on the 12B token-accuracy subset."""
    return os.environ.get(PIPELINE_ENV, PIPELINE_DEFAULT).strip().lower() != "l1acc"


def pipeline_ops_allowed(op: str) -> bool:
    """The switch may name the ops the rows apply to (``GEMMA4_DRAM_SHARD_PIPELINE=gate_up,down``: the MLP only;
    ``out_proj``): every other value (``1``, ``on``, ``l1acc``) allows all ops."""
    env = os.environ.get(PIPELINE_ENV, PIPELINE_DEFAULT).strip().lower()
    if env in ("1", "on", "true", "yes", "l1acc", "0", "false", "no", "off", ""):
        return True
    return op in {tok.strip() for tok in env.split(",")}


def pipeline_row_for(mesh_device, hidden_size, op: str, tp: int):
    """The row for a model build on ``mesh_device`` (its arch and worker grid)."""
    from models.common.utility_functions import is_blackhole

    grid = mesh_device.compute_with_storage_grid_size()
    return dram_sharded_pipeline_row(hidden_size, op, tp, (grid.x, grid.y), is_blackhole())
