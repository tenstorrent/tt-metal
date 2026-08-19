# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config sweep for the seven matmuls of the Qwen3.5 VISION TOWER.

WHY THIS EXISTS
---------------
A tower profile (``tests/test_vision_tower_pcc.py`` under Tracy + ``tt-perf-report``) tags every one
of the tower's matmuls **SLOW** -- both DRAM% and FLOP% low, i.e. neither bandwidth- nor math-bound,
so a program config is on the table. Five of the seven run on the *auto* config or on a config
sized for the wrong M, and two run HiFi4 on bfloat8_b weights. This sweep finds, per matmul, the
fastest (program config, fidelity, chunk size, output memory) point that still passes a PCC gate,
and prints it as a ready-to-paste snippet.

THE SHAPES ARE THE MODEL'S, NOT THIS TEST'S
-------------------------------------------
Every shape / dtype / baseline config comes from :mod:`vision_matmul_specs`, which derives them
from ``VisionModelArgs`` -- the same object the tower is built from -- and cross-checks them
against a real ``DropInVisionTransformer.forward`` (``test_vision_matmul_specs_match_model``).
Run that test first: it is the gate that keeps a sweep winner from being a shape nothing runs.

WHAT IS SWEPT (staged coordinate descent, one axis at a time)
------------------------------------------------------------
  0. baseline          -- exactly what the model does today (the number to beat)
  1. fidelity          -- HiFi4 / HiFi4_fp16 / HiFi2 / HiFi2_fp16 / LoFi. ``fp32_dest_acc_en=False``
                          also lifts the output-subblock cap from h*w <= 4 to <= 8.
  2. chunk size        -- the tower reshapes ``[1,1,S,K] -> [1,S/C,C,K]`` before each matmul; C is
                          a free knob (the reshape is metadata-only on a TILE tensor).
  3. grid              -- never hard-coded: candidates come from the real
                          ``compute_with_storage_grid_size()``.
  4. in0_block_w       -- divisors of K_tiles (K-streaming pipeline depth).
  5. out_subblock      -- (h, w) with h*w <= cap.
  6. fused activation  -- ``activation="gelu"`` on the auto path dispatches a SEPARATE unary op
                          (matmul.cpp: ``user_fused_activation && !user_core_coord`` ->
                          ``unary_chain``). Moving GELU into the program config removes that op.
  7. folded bias       -- ``qkv`` / ``wo`` / ``fc2`` add their bias as their own binary op; try
                          folding it into ``ttnn.linear(bias=...)``.
  8. output memory     -- DRAM vs L1 interleaved (the report's "place input 0 in L1" advice is a
                          no-op here; the writeback is what the next op re-reads).

Run (N300 / Qwen3.5-9B)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
      models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py -v -s

Run (T3K / Qwen3.6-27B). ``HF_MODEL`` points at the LOCAL config dir: ``ModelArgs`` takes
``CKPT_DIR = HF_MODEL`` and ``model_name`` from its basename, so this needs no checkpoint and no hub
fetch -- and the same trick runs the PCC gate and the tower profile on the 27B::

    MESH_DEVICE=T3K HF_MODEL=$PWD/models/tt_transformers/model_params/Qwen3.6-27B pytest \\
      models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py -v -s

One family only, and a wider search::

    QWEN36_SWEEP_FAMILIES=qkv,wo QWEN36_SWEEP_PASSES=2 MESH_DEVICE=N300 \\
      HF_MODEL=Qwen/Qwen3.5-9B pytest models/.../test_sweep_vision_matmuls.py -v -s -k sweep

TIMING METHOD
-------------
Host wall-clock over ``reps`` back-to-back launches with a single ``synchronize_device`` at the
end, best of ``TRIALS`` batches. Legitimate here only because every one of these matmuls is
0.7-3.3 ms -- ~100x the ~30 us dispatch cost. The harness prints ``baseline`` next to the
``tt-perf-report`` number for the same op so you can confirm the isolation IS the same
experiment before trusting any ranking.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tests.perf.vision_matmul_specs import (
    FAMILIES,
    MatmulSpec,
    assert_specs_match,
    capture_specs,
    derive_specs,
    padded_seq_len,
)
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

# The demo image: demo/benchmark_vision.py and demo/vision_demo.py both default to this grid.
DEMO_GRID = (1, 86, 128)

WEIGHT_DTYPE = ttnn.bfloat8_b
PCC_THRESHOLD = 0.99
# Anything inside this band of the best time is noise, not a winner (see methodology.md §3).
# Measured run-to-run spread of an identical config on this harness is ~10%, hence 5% is already
# generous; every kept winner is re-checked in-model by re-profiling the tower.
NOISE_FRACTION = 0.05
TRIALS = 3
TARGET_BATCH_SECONDS = 0.10

# In-model device time (us) per family, from `tt-perf-report` on the tower for QWEN3.5-9B ON N300
# with the demo grid -- a reference for that configuration only. Printed next to this harness's own
# baseline so you can confirm the isolation reproduces the in-model number before trusting a ranking.
# `untuned` folds in the separate op each change absorbed (qkv's 937 us bias add, mlp_fc1's 1234 us
# GELU); `tuned` is the same profile as currently shipped.
PROFILE_BASELINE_US = {
    #             untuned  tuned
    "patch_embed": (865, 730),
    "qkv": (3281 + 937, 1411),
    "wo": (2725, 294),
    "mlp_fc1": (1863 + 1234, 1991),
    "mlp_fc2": (1670, 995),
    "merger_fc1": (707, 708),
    "merger_fc2": (764, 757),
}

TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
# Usable L1 per Tensix core, minus room for kernels / semaphores / the l1_small allocator.
L1_CB_BUDGET = 1_400_000


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1024 * 1024 * 1024} if _MULTI else {}),
    }
]


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(p.strip() for p in raw.split(",") if p.strip())


# ------------------------------------------------------------------------------------ candidates


@dataclass(frozen=True)
class Cand:
    """One point in the sweep space."""

    chunk: int
    fidelity: str
    variant: str = "2d"  # "auto" | "2d" | "1d"
    grid: tuple[int, int] | None = None
    ibw: int | None = None
    sbh: int | None = None
    sbw: int | None = None
    obh: int | None = None
    obw: int | None = None
    out_mem: str = "dram"
    # Where input 0 lives. "dram"/"l1" are interleaved; "shard" is BLOCK_SHARDED L1, the classic
    # 2D-mcast layout (M over grid_y, K over grid_x), which additionally requires grid_x | K_tiles
    # and pins in0_block_w to K_tiles/grid_x.
    in0_mem: str = "dram"
    fuse_act: bool = False
    fold_bias: bool = False
    # Only set when reproducing an existing program config verbatim: the model's own configs do not
    # always derive per_core_* from the grid the way `per_core_dims` does, and a baseline that
    # "fixes" them silently is not the baseline (VISION_WO_PREFILL_PROGCFG sizes per_core_M for
    # 2048 rows while the matmul it configures runs 1024 -- worth 24 of 64 cores).
    pcm: int | None = None
    pcn: int | None = None

    def label(self, spec: MatmulSpec) -> str:
        if self.variant == "auto":
            core = "auto"
        else:
            pcm, pcn = per_core_dims(spec, self)
            core = (
                f"{self.variant} g{self.grid[0]}x{self.grid[1]} ibw{self.ibw} " f"sb{self.sbh}x{self.sbw} pc{pcm}x{pcn}"
            )
            if (self.obh, self.obw) != (None, None):
                core += f" ob{self.obh}x{self.obw}"
        extra = "".join(
            [
                f" C{self.chunk}",
                f" {self.fidelity}",
                "" if self.in0_mem == "dram" else f" in0-{self.in0_mem}",
                " L1out" if self.out_mem == "l1" else "",
                " fusedact" if self.fuse_act else "",
                " foldbias" if self.fold_bias else "",
            ]
        )
        return core + extra


def divisors(n: int, hi: int | None = None) -> list[int]:
    out = [d for d in range(1, n + 1) if n % d == 0]
    return [d for d in out if hi is None or d <= hi]


def per_core_dims(spec: MatmulSpec, cand: Cand) -> tuple[int, int]:
    if cand.pcm is not None and cand.pcn is not None:
        return cand.pcm, cand.pcn
    m_t = cand.chunk // ttnn.TILE_SIZE
    gx, gy = cand.grid
    if cand.variant == "1d":
        return m_t, math.ceil(spec.n_tiles / (gx * gy))
    return math.ceil(m_t / gy), math.ceil(spec.n_tiles / gx)


def cores_used(spec: MatmulSpec, cand: Cand) -> int:
    if cand.variant == "auto":
        return 0
    m_t = cand.chunk // ttnn.TILE_SIZE
    pcm, pcn = per_core_dims(spec, cand)
    return math.ceil(m_t / pcm) * math.ceil(spec.n_tiles / pcn)


def est_l1_bytes(spec: MatmulSpec, cand: Cand, fp32_dest: bool) -> int:
    """Rough per-core CB footprint, used only to skip candidates that cannot possibly compile."""
    if cand.variant == "auto":
        return 0
    pcm, pcn = per_core_dims(spec, cand)
    obh = cand.obh or pcm
    obw = cand.obw or pcn
    # A block-sharded in0 is read in place: no mcast in0 CB (its cost is counted as held L1).
    in0 = 0 if cand.in0_mem == "shard" else obh * cand.ibw * TILE_BYTES[spec.in0_dtype] * 2
    in1 = cand.ibw * obw * TILE_BYTES[spec.in1_dtype] * 2
    out = obh * obw * TILE_BYTES[spec.out_dtype]
    interm = obh * obw * (4096 if fp32_dest else TILE_BYTES[spec.out_dtype])
    return in0 + in1 + out + interm


def subblock_candidates(pcm: int, pcn: int, obh: int, obw: int, cap: int) -> list[tuple[int, int]]:
    out = []
    for h in divisors(obh):
        for w in divisors(obw):
            if h * w <= cap:
                out.append((h, w))
    # Largest area first: that is the DST-utilisation ordering.
    return sorted(out, key=lambda hw: (-hw[0] * hw[1], -hw[1]))


def build_progcfg(spec: MatmulSpec, cand: Cand, activation) -> Any:
    if cand.variant == "auto":
        return None
    pcm, pcn = per_core_dims(spec, cand)
    common = dict(
        compute_with_storage_grid_size=cand.grid,
        in0_block_w=cand.ibw,
        out_subblock_h=cand.sbh,
        out_subblock_w=cand.sbw,
        per_core_M=pcm,
        per_core_N=pcn,
        fused_activation=activation if cand.fuse_act else None,
    )
    if cand.obh is not None:
        common["out_block_h"] = cand.obh
    if cand.obw is not None:
        common["out_block_w"] = cand.obw
    # Mirror VisionModelArgs.vision_mm_plan: an unchunked activation is one batch, which the model
    # configures with fuse_batch=True.
    fuse_batch = spec.rows == cand.chunk
    if cand.variant == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**common, fuse_batch=fuse_batch, mcast_in0=True)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**common, transpose_mcast=False, fuse_batch=fuse_batch)


def valid(spec: MatmulSpec, cand: Cand, fp32_dest: bool) -> str | None:
    """Return a rejection reason, or None if the candidate is worth dispatching."""
    if spec.rows % cand.chunk or cand.chunk % ttnn.TILE_SIZE:
        return "chunk does not divide rows"
    if cand.variant == "auto":
        return None
    if spec.k_tiles % cand.ibw:
        return "in0_block_w does not divide K_tiles"
    pcm, pcn = per_core_dims(spec, cand)
    l1 = est_l1_bytes(spec, cand, fp32_dest)
    if l1 > L1_CB_BUDGET:
        return f"est L1 {l1 // 1024} KB > budget"
    obh, obw = cand.obh or pcm, cand.obw or pcn
    if pcm % obh or pcn % obw:
        return "out_block does not divide per_core"
    if obh % cand.sbh or obw % cand.sbw:
        return "out_subblock does not divide out_block"
    cap = 4 if fp32_dest else 8
    if cand.sbh * cand.sbw > cap:
        return f"subblock area {cand.sbh * cand.sbw} > cap {cap}"
    if cand.in0_mem == "shard":
        gx, gy = cand.grid
        # A block-sharded in0 shards K across grid_x, so the K block each core holds IS
        # in0_block_w -- the config cannot choose it freely.
        if spec.k_tiles % gx or cand.ibw != spec.k_tiles // gx:
            return f"block-sharded in0 needs in0_block_w == K_tiles/grid_x ({spec.k_tiles}/{gx})"
        # ttnn shards the flattened height, so a batched activation puts every row on the grid at
        # once -- per_core_M must then cover all of it, i.e. the single-shot case.
        if spec.rows % (gy * ttnn.TILE_SIZE) or (spec.rows // ttnn.TILE_SIZE) // gy != pcm:
            return f"block-sharded in0 needs per_core_M == rows_tiles/grid_y ({spec.rows // 32}/{gy})"
        held = spec.rows * spec.k * TILE_BYTES[spec.in0_dtype] // (ttnn.TILE_SIZE**2) // (gx * gy)
        if held + l1 > L1_CB_BUDGET:
            return f"sharded in0 {held // 1024} KB + CBs {l1 // 1024} KB > budget"
    if cand.variant == "1d":
        m_t = cand.chunk // ttnn.TILE_SIZE
        if math.ceil(m_t / pcm) * math.ceil(spec.n_tiles / pcn) > cand.grid[0] * cand.grid[1]:
            return "num_blocks_total > num_cores"
    return None


# ---------------------------------------------------------------------------------------- runner


@dataclass
class Result:
    cand: Cand
    label: str
    us: float | None
    pcc: float | None
    cores: int
    error: str | None = None
    # Device time of the separate op(s) this candidate still leaves behind (the bias add the model
    # dispatches after the matmul). Folding the bias in makes the MATMUL slower but removes that op,
    # so ranking on the matmul alone would reject a win.
    extra_us: float = 0.0

    @property
    def ok(self) -> bool:
        return self.us is not None and self.error is None

    @property
    def total(self) -> float:
        return math.inf if not self.ok else self.us + self.extra_us


class FamilyBench:
    """Holds the device tensors for one matmul family and times candidates against them."""

    def __init__(self, mesh_device, model_args, spec: MatmulSpec):
        self.mesh_device = mesh_device
        self.spec = spec
        self.grid_size = mesh_device.compute_with_storage_grid_size()
        self.max_grid = (self.grid_size.x, self.grid_size.y)

        self.fidelities = {}
        for name in ("hifi4", "hifi4_fp16", "hifi2", "hifi2_fp16", "lofi"):
            ckc = getattr(model_args, f"compute_kernel_config_{name}", None)
            if ckc is not None:
                self.fidelities[name] = ckc
        assert spec.fidelity_name in self.fidelities, f"unknown baseline fidelity {spec.fidelity_name}"

        torch.manual_seed(0)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        # Per-device shapes: every device runs this identical matmul on its own shard, so a
        # replicated tensor of the LOCAL shape reproduces the in-model device op exactly.
        a = torch.randn(1, 1, spec.rows, spec.k) * 0.1
        b = torch.randn(1, 1, spec.k, spec.n) * 0.05
        self.a_base = ttnn.from_torch(
            a,
            dtype=spec.in0_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.b = ttnn.from_torch(
            b,
            dtype=spec.in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.bias = None
        if spec.has_bias or spec.separate_bias:
            bias = torch.randn(spec.n) * 0.02
            self.bias = ttnn.from_torch(
                bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        # Reference from the tensors AS QUANTISED on device, so PCC measures the matmul's own
        # numerics (fidelity, accumulation order) and not the bf8b round-trip.
        a_q = ttnn.to_torch(ttnn.get_device_tensors(self.a_base)[0]).float().reshape(spec.rows, spec.k)
        b_q = ttnn.to_torch(ttnn.get_device_tensors(self.b)[0]).float().reshape(spec.k, spec.n)
        raw = a_q @ b_q
        bias_row = (
            None
            if self.bias is None
            else ttnn.to_torch(ttnn.get_device_tensors(self.bias)[0]).float().reshape(-1)[: spec.n]
        )

        def _finish(with_bias: bool):
            ref = raw + bias_row if (with_bias and bias_row is not None) else raw
            if spec.activation == "gelu":
                return torch.nn.functional.gelu(ref)
            if spec.activation is not None:
                raise NotImplementedError(f"reference for activation {spec.activation}")
            return ref

        # Two references: a `fold_bias` candidate produces bias-added output, so comparing it
        # against the bias-free reference would report a bogus PCC drop.
        self.ref_nobias = _finish(False)
        self.ref_bias = _finish(True) if bias_row is not None else self.ref_nobias
        self.ref = self.ref_bias if spec.has_bias else self.ref_nobias

        self.activation_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False) if spec.activation == "gelu" else None
        self._chunk_views: dict[int, Any] = {}
        self._in0: Any = None  # in0 in the candidate's layout, live only for the current measurement
        self._conv_cache: dict[Any, float] = {}
        self.reps = 4
        # Cost of the bias add the model dispatches as its own op right after this matmul (only
        # meaningful when the bias is applied to the matmul output as-is; a row-parallel family
        # applies it after the collective, on a narrower tensor, and folding it in there would be
        # summed TP times by the all-reduce).
        self.sep_bias_us = 0.0
        if spec.separate_bias and not spec.bias_after_collective:
            self.sep_bias_us = self._time_bias_add()

    def _time_bias_add(self) -> float:
        spec = self.spec
        out = ttnn.linear(
            self.view(spec.chunk),
            self.b,
            compute_kernel_config=self.fidelities[spec.fidelity_name],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=spec.out_dtype,
        )
        ttnn.synchronize_device(self.mesh_device)
        best = math.inf
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            for _ in range(4):
                ttnn.deallocate(ttnn.add(out, self.bias, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(self.mesh_device)
            best = min(best, (time.perf_counter() - t0) / 4)
        ttnn.deallocate(out)
        logger.info(f"  separate bias add on the {spec.name} output costs {best * 1e6:.1f} us")
        return best * 1e6

    def in0_for(self, cand: Cand):
        """The in0 tensor in the layout `cand` asks for, built ONCE and reused.

        Built outside the timed loop on purpose: this measures the matmul given that layout, i.e.
        the best case where the PRODUCER emits it. `conversion_us` separately reports what it costs
        to get there from DRAM, which is the honest number when no producer can be changed.
        """
        view = self.view(cand.chunk)
        if cand.in0_mem == "dram":
            return view
        # Built fresh per candidate and freed by `release_in0`. Caching these would leave every
        # variant's 432 KB/core resident for the rest of the family's sweep, which starves later
        # candidates of the very L1 they are being measured for -- it produced one spurious
        # "collides with L1 buffers" failure on a config the budget arithmetic says fits.
        return ttnn.to_memory_config(view, self._in0_mem_config(cand))

    def release_in0(self, cand: Cand, tensor) -> None:
        if cand.in0_mem != "dram" and tensor is not None:
            ttnn.deallocate(tensor)

    def _in0_mem_config(self, cand: Cand):
        if cand.in0_mem == "l1":
            return ttnn.L1_MEMORY_CONFIG
        gx, gy = cand.grid
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
                [self.spec.rows // gy, self.spec.k // gx],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def conversion_us(self, cand: Cand) -> float:
        """Time to move in0 from DRAM into `cand`'s layout -- 0 for the DRAM baseline."""
        if cand.in0_mem == "dram":
            return 0.0
        key = ("conv",) + (cand.chunk, cand.in0_mem, cand.grid, cand.ibw)
        if key not in self._conv_cache:
            cfg = self._in0_mem_config(cand)
            view = self.view(cand.chunk)
            ttnn.deallocate(ttnn.to_memory_config(view, cfg))
            ttnn.synchronize_device(self.mesh_device)
            best = math.inf
            for _ in range(TRIALS):
                t0 = time.perf_counter()
                for _ in range(4):
                    ttnn.deallocate(ttnn.to_memory_config(view, cfg))
                ttnn.synchronize_device(self.mesh_device)
                best = min(best, (time.perf_counter() - t0) / 4)
            self._conv_cache[key] = best * 1e6
        return self._conv_cache[key]

    def view(self, chunk: int):
        """`[1, 1, rows, K] -> [1, rows/C, C, K]` -- metadata-only on a TILE tensor, as in-model."""
        if chunk not in self._chunk_views:
            batch = self.spec.rows // chunk
            self._chunk_views[chunk] = (
                self.a_base if batch == 1 else ttnn.reshape(self.a_base, [1, batch, chunk, self.spec.k])
            )
        return self._chunk_views[chunk]

    def _call(self, cand: Cand, progcfg):
        spec = self.spec
        kwargs = dict(
            compute_kernel_config=self.fidelities[cand.fidelity],
            memory_config=ttnn.L1_MEMORY_CONFIG if cand.out_mem == "l1" else ttnn.DRAM_MEMORY_CONFIG,
            dtype=spec.out_dtype,
        )
        if progcfg is not None:
            kwargs["program_config"] = progcfg
        if spec.activation is not None and not cand.fuse_act:
            kwargs["activation"] = spec.activation
        if spec.has_bias or (spec.separate_bias and cand.fold_bias):
            kwargs["bias"] = self.bias
        return ttnn.linear(self._in0 if self._in0 is not None else self.in0_for(cand), self.b, **kwargs)

    def run(self, cand: Cand, want_pcc: bool = False) -> Result:
        spec = self.spec
        fp32_dest = bool(self.fidelities[cand.fidelity].fp32_dest_acc_en)
        reason = valid(spec, cand, fp32_dest)
        label = cand.label(spec)
        if reason is not None:
            return Result(cand, label, None, None, cores_used(spec, cand), f"skipped: {reason}")

        progcfg = None
        try:
            progcfg = build_progcfg(spec, cand, self.activation_param)
        except Exception as exc:  # pragma: no cover - invalid config combination
            return Result(cand, label, None, None, cores_used(spec, cand), f"progcfg: {type(exc).__name__}: {exc}")

        try:
            self._in0 = self.in0_for(cand)
            out = self._call(cand, progcfg)  # compile + warm the program cache
            ttnn.synchronize_device(self.mesh_device)
        except Exception as exc:
            self.release_in0(cand, self._in0)
            self._in0 = None
            msg = str(exc).splitlines()
            return Result(cand, label, None, None, cores_used(spec, cand), f"run: {msg[0][:160] if msg else exc}")

        pcc = None
        if want_pcc:
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(spec.rows, spec.n)
            ref = self.ref_bias if (spec.has_bias or cand.fold_bias) else self.ref_nobias
            _, pcc = comp_pcc(ref, got, PCC_THRESHOLD)
            pcc = float(pcc)
        ttnn.deallocate(out)

        best = math.inf
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            for _ in range(self.reps):
                o = self._call(cand, progcfg)
                ttnn.deallocate(o)
            ttnn.synchronize_device(self.mesh_device)
            best = min(best, (time.perf_counter() - t0) / self.reps)
        self.release_in0(cand, self._in0)
        self._in0 = None
        extra = 0.0 if cand.fold_bias else self.sep_bias_us
        return Result(cand, label, best * 1e6, pcc, cores_used(spec, cand), extra_us=extra)


# ----------------------------------------------------------------------------------- the search


def baseline_cand(spec: MatmulSpec, bench: FamilyBench) -> Cand:
    """The candidate that reproduces exactly what the model does today."""
    pc = spec.baseline_progcfg
    out_mem = "l1" if spec.out_l1 else "dram"
    if pc is None:
        return Cand(chunk=spec.chunk, fidelity=spec.fidelity_name, variant="auto", out_mem=out_mem)
    # The tower already fuses the activation into this program config, so the baseline must too.
    grid = (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
    return Cand(
        chunk=spec.chunk,
        fidelity=spec.fidelity_name,
        variant="2d",
        grid=grid,
        ibw=pc.in0_block_w,
        sbh=pc.out_subblock_h,
        sbw=pc.out_subblock_w,
        obh=pc.out_block_h,
        obw=pc.out_block_w,
        pcm=pc.per_core_M,
        pcn=pc.per_core_N,
        fuse_act=spec.activation_fused,
        out_mem=out_mem,
    )


def reflow(cand: Cand, **kw) -> Cand:
    """Derive a candidate whose per_core_* / out_block are re-derived from grid and chunk.

    Used whenever an axis that FEEDS per_core_* moves, so a pinned baseline value cannot leak
    into a swept candidate.
    """
    return replace(cand, pcm=None, pcn=None, obh=None, obw=None, **kw)


def seed_cand(spec: MatmulSpec, bench: FamilyBench, fidelity: str, chunk: int) -> Cand | None:
    """First LEGAL 2D point, most cores first.

    Must search rather than guess: the widest-grid pick blows the L1 CB budget on the merger
    shapes (per_core_M=43 -> a 2.3 MB output CB), and a family whose seed is skipped would
    otherwise never get a 2D config swept at all.
    """
    ibw = max([d for d in divisors(spec.k_tiles, 4)] or [1])
    for grid in grid_options(spec, bench, chunk):
        cand = fix_subblock(
            spec, bench, Cand(chunk=chunk, fidelity=fidelity, variant="2d", grid=grid, ibw=ibw, sbh=1, sbw=1)
        )
        if valid(spec, cand, bool(bench.fidelities[fidelity].fp32_dest_acc_en)) is None:
            return cand
    return None


def fix_subblock(spec: MatmulSpec, bench: FamilyBench, cand: Cand) -> Cand:
    """Re-legalise the subblock after a grid / out_block / fidelity move."""
    if cand.variant == "auto":
        return cand
    fp32_dest = bool(bench.fidelities[cand.fidelity].fp32_dest_acc_en)
    pcm, pcn = per_core_dims(spec, cand)
    obh, obw = cand.obh or pcm, cand.obw or pcn
    for h, w in subblock_candidates(pcm, pcn, obh, obw, 4 if fp32_dest else 8):
        trial = replace(cand, sbh=h, sbw=w)
        if valid(spec, trial, fp32_dest) is None:
            return trial
    return replace(cand, sbh=1, sbw=1)


def chunk_options(spec: MatmulSpec) -> list[int]:
    tiles = spec.rows // ttnn.TILE_SIZE
    opts = {spec.chunk}
    for d in divisors(tiles):
        rows = d * ttnn.TILE_SIZE
        if 512 <= rows <= 8192:
            opts.add(rows)
    return sorted(opts)


def grid_options(spec: MatmulSpec, bench: FamilyBench, chunk: int) -> list[tuple[int, int]]:
    gx_max, gy_max = bench.max_grid
    m_t = chunk // ttnn.TILE_SIZE
    ys = sorted(set(divisors(m_t, gy_max)) | {gy_max})
    xs = sorted(set(divisors(spec.n_tiles, gx_max)) | {gx_max})
    # Only grids that use most of the array are interesting; keep the biggest handful.
    opts = [(x, y) for x, y in itertools.product(xs, ys) if x * y >= 0.35 * gx_max * gy_max]
    return sorted(opts, key=lambda g: -g[0] * g[1])[:10]


def sweep_family(bench: FamilyBench, spec: MatmulSpec, passes: int) -> list[Result]:
    """Staged coordinate descent. Each stage varies ONE axis around the current best."""
    results: list[Result] = []

    def evaluate(cands, *, want_pcc=False, stage=""):
        stage_results = []
        for cand in cands:
            res = bench.run(cand, want_pcc=want_pcc)
            results.append(res)
            stage_results.append(res)
            if res.ok:
                logger.info(
                    f"  [{stage}] {res.us:8.1f} us  cores {res.cores:>3}  "
                    f"pcc {'-' if res.pcc is None else f'{res.pcc:.5f}'}  {res.label}"
                )
            else:
                logger.debug(f"  [{stage}] --  {res.label}  ({res.error})")
        return [r for r in stage_results if r.ok]

    # ---- stage 0: the baseline the model ships -------------------------------------------
    base = baseline_cand(spec, bench)
    base_results = evaluate([base], want_pcc=True, stage="baseline")
    if not base_results:
        raise AssertionError(f"{spec.name}: the model's own config failed to run: {results[-1].error}")
    baseline = base_results[0]
    bench.reps = max(3, min(24, int(TARGET_BATCH_SECONDS / (baseline.us * 1e-6))))
    untuned, tuned = PROFILE_BASELINE_US.get(spec.name, (0, 0))
    logger.info(
        f"  baseline {baseline.total:.1f} us (in-model device time: {untuned} us untuned, "
        f"{tuned} us as currently tuned) -- reps={bench.reps}"
    )

    # ---- stage 1: fidelity (also flips the subblock cap via fp32_dest_acc_en) -------------
    fid_cands = [fix_subblock(spec, bench, replace(base, fidelity=fid)) for fid in bench.fidelities]
    ok = evaluate(fid_cands, want_pcc=True, stage="fidelity")
    ok = [r for r in ok if r.pcc is None or r.pcc >= PCC_THRESHOLD]
    best_fid = min(ok, key=lambda r: r.total).cand.fidelity if ok else spec.fidelity_name

    def best_2d() -> Result | None:
        """Best 2D result so far. Kept SEPARATE from the global best: if `auto` happens to lead,
        every later `replace(best, grid=...)` would silently be a no-op on an auto config and the
        family would never get a program config swept at all."""
        cands = [r for r in results if r.ok and r.cand.variant == "2d"]
        return min(cands, key=lambda r: r.total) if cands else None

    # ---- stage 2..5: coordinate descent around the best 2D point -------------------------
    seed = seed_cand(spec, bench, best_fid, spec.chunk)
    if seed is not None:
        evaluate([seed], stage="seed")
    work = best_2d()

    for p in range(passes):
        if work is None:
            logger.info("  no legal 2D program config for this shape; keeping the auto path")
            break
        # chunk size -- reflow per_core_*, they are derived from it
        cands = [
            fix_subblock(spec, bench, reflow(work.cand, chunk=c)) for c in chunk_options(spec) if c != work.cand.chunk
        ]
        evaluate(cands, stage=f"chunk p{p}")
        work = best_2d()

        # grid -- also reflows per_core_*
        cands = [
            fix_subblock(spec, bench, reflow(work.cand, grid=g))
            for g in grid_options(spec, bench, work.cand.chunk)
            if g != work.cand.grid
        ]
        evaluate(cands, stage=f"grid p{p}")
        work = best_2d()

        # in0_block_w
        cands = [
            fix_subblock(spec, bench, replace(work.cand, ibw=b))
            for b in divisors(spec.k_tiles, 32)
            if b != work.cand.ibw
        ]
        evaluate(cands, stage=f"ibw p{p}")
        work = best_2d()

        # out_subblock, and out_block (which is what actually sizes the output CB)
        cand = work.cand
        pcm, pcn = per_core_dims(spec, cand)
        fp32_dest = bool(bench.fidelities[cand.fidelity].fp32_dest_acc_en)
        cands = [
            replace(cand, sbh=h, sbw=w)
            for h, w in subblock_candidates(pcm, pcn, cand.obh or pcm, cand.obw or pcn, 4 if fp32_dest else 8)
            if (h, w) != (cand.sbh, cand.sbw)
        ]
        cands += [fix_subblock(spec, bench, replace(cand, obh=obh)) for obh in divisors(pcm) if obh != pcm]
        evaluate(cands, stage=f"subblock p{p}")
        work = best_2d()

    def leader() -> Result:
        return min([r for r in results if r.ok], key=lambda r: r.total)

    # ---- stage 6/7: fuse the activation into the matmul, fold the separate bias in --------
    extras = []
    if spec.activation is not None and leader().cand.variant != "auto" and not leader().cand.fuse_act:
        extras.append(replace(leader().cand, fuse_act=True))
    if spec.separate_bias and not spec.bias_after_collective:
        extras.append(replace(leader().cand, fold_bias=True))
    if extras:
        evaluate(extras, want_pcc=True, stage="fuse")

    # ---- stage 8: output memory config (the writeback) ------------------------------------
    tops = sorted([r for r in results if r.ok], key=lambda r: r.total)[:3]
    evaluate([replace(r.cand, out_mem="l1") for r in tops if r.cand.out_mem == "dram"], stage="L1 out")

    # ---- stage 8b: INPUT 0 layout -- what `tt-perf-report` nags about on every SLOW matmul ---
    # L1-interleaved needs only the producer's memory_config flipped. BLOCK_SHARDED is the classic
    # 2D-mcast layout but pins in0_block_w to K_tiles/grid_x and puts the whole activation on the
    # grid at once, so it is often illegal here -- the skip reason says which.
    in0_cands = []
    seed_2d = next(
        (r.cand for r in sorted([r for r in results if r.ok], key=lambda r: r.total) if r.cand.variant == "2d"), None
    )
    if seed_2d is not None:
        # in0 in L1 competes with the OUTPUT for the same L1: at 12288 rows each of these tensors is
        # tens of MB per device, so try in0-L1 with the output back in DRAM, and at the SMALLEST
        # in0_block_w too (the CBs are what collide with the L1 buffer).
        for ibw in sorted({seed_2d.ibw, min(divisors(spec.k_tiles, 8) or [1])}):
            in0_cands.append(fix_subblock(spec, bench, replace(seed_2d, in0_mem="l1", out_mem="dram", ibw=ibw)))
            in0_cands.append(fix_subblock(spec, bench, replace(seed_2d, in0_mem="l1", out_mem="l1", ibw=ibw)))
        # BLOCK_SHARDED in0 gets its OWN grid and an unchunked activation: the layout pins
        # in0_block_w = K_tiles/grid_x and per_core_M = rows_tiles/grid_y, so it is only expressible
        # as the single-shot 2D matmul. Give it the largest grid_x that divides K_tiles.
        gx_s = max(divisors(spec.k_tiles, bench.max_grid[0]) or [1])
        gy_s = bench.max_grid[1]
        if spec.rows % (gy_s * ttnn.TILE_SIZE) == 0:
            in0_cands.append(
                fix_subblock(
                    spec,
                    bench,
                    replace(
                        seed_2d,
                        in0_mem="shard",
                        out_mem="dram",
                        chunk=spec.rows,
                        grid=(gx_s, gy_s),
                        ibw=spec.k_tiles // gx_s,
                        obh=None,
                        obw=None,
                        pcm=None,
                        pcn=None,
                    ),
                )
            )
    for res in evaluate(in0_cands, stage="in0 layout"):
        conv = bench.conversion_us(res.cand)
        logger.info(
            f"    in0={res.cand.in0_mem}: matmul {res.us:.1f} us"
            f" (+{conv:.1f} us if it must be converted from DRAM rather than produced there)"
        )

    # ---- stage 9: re-walk fidelity on the WINNING config ---------------------------------
    # Coordinate descent pinned fidelity in stage 1 against the model's (untuned) config, so the
    # accuracy/speed trade has to be re-priced at the winning point -- LoFi is usually fastest but
    # a bf16xbf16 matmul feeding 27 blocks may not want it.
    win = leader().cand
    evaluate(
        [fix_subblock(spec, bench, replace(win, fidelity=f)) for f in bench.fidelities if f != win.fidelity],
        want_pcc=True,
        stage="refidelity",
    )

    # Re-measure the baseline last: an identical config re-runs within ~10% on this harness, and a
    # cold first measurement would inflate every speedup.
    again = bench.run(base, want_pcc=False)
    if again.ok and again.us < baseline.us:
        logger.info(f"  baseline re-measured {again.us:.1f} us (was {baseline.us:.1f}) -- keeping the faster")
        baseline.us = again.us

    # PCC for everything still in contention.
    for res in sorted([r for r in results if r.ok], key=lambda r: r.total)[:6]:
        if res.pcc is None:
            checked = bench.run(res.cand, want_pcc=True)
            res.pcc = checked.pcc
            res.us = min(res.us, checked.us or res.us)

    return results


# ------------------------------------------------------------------------------------ reporting


def report(spec: MatmulSpec, results: list[Result], baseline: Result) -> Result | None:
    ok = sorted([r for r in results if r.ok], key=lambda r: r.total)
    peak_flops = spec.flops()
    lines = [
        "",
        f"===== {spec.name}: {spec.rows}x{spec.k}x{spec.n} (batch {spec.batch} x M {spec.chunk}) =====",
        f"  baseline  {baseline.total:8.1f} us"
        f"{f' (matmul {baseline.us:.0f} + bias op {baseline.extra_us:.0f})' if baseline.extra_us else ''}"
        f"   pcc {baseline.pcc if baseline.pcc is None else f'{baseline.pcc:.5f}'}   {baseline.label}",
        f"  {'rank':<4} {'us':>9} {'vs base':>8} {'TFLOP/s':>8} {'cores':>5}  {'pcc':>8}  config",
    ]
    for i, r in enumerate(ok[:15], 1):
        tflops = peak_flops / (r.total * 1e-6) / 1e12
        lines.append(
            f"  {i:<4} {r.total:9.1f} {baseline.total / r.total:7.2f}x {tflops:8.1f} {r.cores or '-':>5}  "
            f"{'-' if r.pcc is None else f'{r.pcc:.5f}':>8}  {r.label}"
        )
    failed = [r for r in results if not r.ok and not (r.error or "").startswith("skipped")]
    if failed:
        lines.append(f"  ({len(failed)} candidates failed to compile/run, e.g. {failed[0].error})")
    # The accuracy/speed trade at the winning config, so a caller can trade a few us for PCC.
    win_shape = replace(ok[0].cand, fidelity="")
    walk = [r for r in ok if replace(r.cand, fidelity="") == win_shape and r.pcc is not None]
    if len(walk) > 1:
        lines.append(
            "  fidelity walk at the winning config: "
            + ", ".join(f"{r.cand.fidelity} {r.total:.0f}us/pcc {r.pcc:.5f}" for r in walk)
        )
    logger.info("\n".join(lines))

    winners = [r for r in ok if r.pcc is not None and r.pcc >= PCC_THRESHOLD]
    if not winners:
        logger.warning(f"{spec.name}: no PCC-passing candidate")
        return None
    winner = winners[0]
    if winner.total > baseline.total * (1 - NOISE_FRACTION):
        logger.info(f"{spec.name}: baseline is already best (within the {NOISE_FRACTION:.0%} noise band)")
        return None
    logger.info(
        f"{spec.name}: WINNER {winner.total:.1f} us vs baseline {baseline.total:.1f} us "
        f"({baseline.total / winner.total:.2f}x, -{baseline.total - winner.total:.0f} us/call) pcc {winner.pcc:.5f}\n"
        f"{snippet(spec, winner.cand)}"
    )
    return winner


def snippet(spec: MatmulSpec, cand: Cand) -> str:
    if cand.variant == "auto":
        return f"    # {spec.name}: leave on auto, fidelity={cand.fidelity}, chunk={cand.chunk}"
    pcm, pcn = per_core_dims(spec, cand)
    cls = (
        "MatmulMultiCoreReuseMultiCast1DProgramConfig"
        if cand.variant == "1d"
        else "MatmulMultiCoreReuseMultiCastProgramConfig"
    )
    tail = "fuse_batch=False, mcast_in0=True," if cand.variant == "1d" else "transpose_mcast=False, fuse_batch=False,"
    act = "ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False)" if cand.fuse_act and spec.activation == "gelu" else "None"
    return "\n".join(
        [
            f"    # {spec.name}: chunk={cand.chunk} rows, fidelity={cand.fidelity}, "
            f"out={'L1' if cand.out_mem == 'l1' else 'DRAM'}"
            f"{', bias folded into the matmul' if cand.fold_bias else ''}",
            f"    ttnn.{cls}(",
            f"        compute_with_storage_grid_size=({cand.grid[0]}, {cand.grid[1]}),",
            f"        in0_block_w={cand.ibw},",
            f"        out_subblock_h={cand.sbh},",
            f"        out_subblock_w={cand.sbw},",
            *([f"        out_block_h={cand.obh},"] if cand.obh else []),
            *([f"        out_block_w={cand.obw},"] if cand.obw else []),
            f"        per_core_M={pcm},",
            f"        per_core_N={pcn},",
            f"        {tail}",
            f"        fused_activation={act},",
            "    )",
        ]
    )


# --------------------------------------------------------------------------------------- fixtures


def _model_args(mesh_device, n_patches: int) -> VisionModelArgs:
    """Config only -- the sweep never touches weight values, just shapes and dtypes."""
    return VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=padded_seq_len(n_patches))


# ------------------------------------------------------------------------------------------ tests


@pytest.mark.timeout(3600)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("grid", [DEMO_GRID], ids=[f"patches{math.prod(DEMO_GRID)}"])
def test_vision_matmul_specs_match_model(mesh_device, device_params, grid, tmp_path):
    """Gate: the analytic sweep table == the matmuls a real tower forward issues.

    This is what makes the sweep the same experiment as the model. It builds the full
    ``DropInVisionTransformer`` at depth 1, runs one forward with ``ttnn.linear`` patched, and diffs every shape / dtype / program config / fidelity
    against :func:`derive_specs`.
    """
    del device_params
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
    from models.tt_transformers.tt.ccl import TT_CCL

    mesh_device.enable_program_cache()
    n_patches = math.prod(grid)
    model_args = _model_args(mesh_device, n_patches)
    vcfg = model_args.hf_config.vision_config
    vcfg.depth = 1

    specs = derive_specs(model_args, n_patches)
    logger.info("analytic vision matmul table:\n  " + "\n  ".join(s.summary() for s in specs.values()))

    torch.manual_seed(0)
    reference_model = Qwen3_5VisionModel(vcfg).eval()
    model = DropInVisionTransformer(
        reference_model,
        model_args,
        dtype=WEIGHT_DTYPE,
        debug=False,
        tt_ccl=TT_CCL(mesh_device),
        weight_cache_path=tmp_path / "vision_sweep_weights",
    )
    pixel_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    calls = capture_specs(model, torch.randn(n_patches, pixel_dim), torch.tensor([grid], dtype=torch.long))

    logger.info(
        "observed vision matmuls:\n  "
        + "\n  ".join(
            f"{c.family:<11} in0 {c.in0_shape} {c.in0_dtype} x in1 {c.in1_shape} {c.in1_dtype} "
            f"-> {c.out_dtype} in {c.out_buffer}"
            f"{' bias' + str(c.bias_shape) if c.bias_shape else ''}"
            f"{' act=' + str(c.activation) if c.activation else ''}"
            f"{' PROGCFG' if c.progcfg else ' auto'}"
            for c in calls.values()
        )
    )
    assert_specs_match(specs, calls)


@pytest.mark.timeout(14400)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("grid", [DEMO_GRID], ids=[f"patches{math.prod(DEMO_GRID)}"])
def test_sweep_vision_matmuls(mesh_device, device_params, grid):
    """Sweep every vision-tower matmul and print the fastest PCC-passing config per family."""
    del device_params
    mesh_device.enable_program_cache()
    n_patches = math.prod(grid)
    model_args = _model_args(mesh_device, n_patches)
    specs = derive_specs(model_args, n_patches)

    families = _env_list("QWEN36_SWEEP_FAMILIES", FAMILIES)
    passes = int(os.environ.get("QWEN36_SWEEP_PASSES", "1"))
    unknown = set(families) - set(specs)
    assert not unknown, f"unknown families {sorted(unknown)}; known: {sorted(specs)}"

    logger.info(
        f"sweeping {list(families)} on mesh {MESH_SHAPE} "
        f"(grid {mesh_device.compute_with_storage_grid_size()}), {n_patches} patches -> "
        f"{padded_seq_len(n_patches)} padded, TP={model_args.cluster_shape[1]}, "
        f"replicated_acts={model_args.vision_replicated_acts}"
    )

    summary = []
    for name in families:
        spec = specs[name]
        logger.info(f"\n######## {spec.summary()}")
        bench = FamilyBench(mesh_device, model_args, spec)
        results = sweep_family(bench, spec, passes)
        baseline = next(r for r in results if r.ok)
        winner = report(spec, results, baseline)
        summary.append((spec, baseline, winner))
        for t in (bench.a_base, bench.b, bench.bias):
            if t is not None:
                ttnn.deallocate(t)

    total_base = sum(b.total for _, b, _ in summary)
    total_best = sum((w or b).total for _, b, w in summary)
    lines = [
        "",
        "===== vision matmul sweep summary =====",
        f"  {'family':<12} {'baseline':>10} {'best':>10} {'speedup':>8}  config",
    ]
    for spec, base, win in summary:
        best = win or base
        lines.append(
            f"  {spec.name:<12} {base.total:10.1f} {best.total:10.1f} {base.total / best.total:7.2f}x  "
            f"{best.label if win else '(baseline)'}"
        )
    lines.append(f"  {'TOTAL':<12} {total_base:10.1f} {total_best:10.1f} {total_base / total_best:7.2f}x")
    # Per-block vs once-per-image, so the numbers can be scaled to the full 27-block tower.
    per_block = [s for s in summary if s[0].name in ("qkv", "wo", "mlp_fc1", "mlp_fc2")]
    pb_base = sum(b.total for _, b, _ in per_block)
    pb_best = sum((w or b).total for _, b, w in per_block)
    depth = model_args.hf_config.vision_config.depth
    lines += [
        f"  per-block matmuls (x{depth} blocks): {pb_base:.0f} -> {pb_best:.0f} us/block "
        f"= {pb_base * depth / 1000:.1f} -> {pb_best * depth / 1000:.1f} ms/tower",
        "  NOTE: these are host wall-clock/rep and read ~10-25% above the in-model device time; "
        "re-profile the tower to confirm every winner.",
    ]
    logger.info("\n".join(lines))
