# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Test constants and helpers shared across this folder."""

from __future__ import annotations

import dataclasses
import inspect
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import RING_SIZE, MoonViTConfig
from models.common.utility_functions import comp_pcc
from models.experimental.kimi_k3_moonvit_v2.tt.ccl import ParallelFactor, TT_CCL

# Depth of the shallow model that the full-model tests run (`shallow_reference`, `shallow_config`).
TEST_LAYERS = 2

# Fixtures (conftest.py), each with random weights or the checkpoint at $KIMI_K3_HF_MODEL:
# `reference` has every layer, `shallow_reference` the first TEST_LAYERS.
WEIGHTS = ["random", "checkpoint"]
REFERENCE = pytest.mark.parametrize("reference", WEIGHTS, indirect=True)
SHALLOW_REFERENCE = pytest.mark.parametrize("shallow_reference", WEIGHTS, indirect=True)

# Layers the per-layer tests cover. With random weights every layer is alike, so one is enough;
# with the checkpoint, the first, middle and last. MOONVIT_TEST_LAYERS (e.g. "5" or "0,20") sets
# the layers for both.
LAYERS_ENV = "MOONVIT_TEST_LAYERS"


def _test_layers() -> dict[str, list[int]]:
    value = os.environ.get(LAYERS_ENV)
    if value:
        layers = [int(layer) for layer in value.split(",")]
        return {"random": layers, "checkpoint": layers}
    n = MoonViTConfig().num_hidden_layers
    return {"random": [0], "checkpoint": [0, n // 2, n - 1]}


REFERENCE_LAYERS = pytest.mark.parametrize(
    ("reference", "layer"),
    [
        pytest.param(weights, layer, id=f"{weights}-layer{layer}")
        for weights, layers in _test_layers().items()
        for layer in layers
    ],
    indirect=["reference"],
)

# (t, h, w) in patches. h and w are even, as the processor's 28 px padding guarantees.
GRIDS = {
    "square": [(1, 8, 8)],
    "wide": [(1, 6, 20)],
    "tall": [(1, 20, 6)],
    # Matches the 64x64 position table exactly, so it skips interpolation.
    "native_table": [(1, 64, 64)],
}

# Device grids for attention, the block and the full model. Their CPU golden builds L x L scores per
# head, which rules out 65,536 patches here.
ATTENTION_GRIDS = pytest.mark.parametrize("grid", [[(1, 64, 64)], [(1, 40, 100)]], ids=["64x64", "40x100"])

# Device grids for the parts that work on each row alone (everything but attention). Their CPU golden
# grows only with the row count, so it runs at the full padded length in seconds. Both meshes run the
# same image; one P150 runs one ring chip's quarter of it (`MeshRows`):
#   full:     fills the padded length exactly; real rows on every ring chip, no padding
#   max_side: also fills it, with the longest side the processor allows (512)
#   partial:  nearly full; the real rows end inside a tile (not a multiple of 32) on the last chip
#   small:    a small image, almost all padding; every real row is on the first chip
ROW_GRID_SIZES = {"full": (256, 256), "max_side": (512, 128), "partial": (250, 250), "small": (40, 100)}
ROW_GRIDS = pytest.mark.parametrize("size", list(ROW_GRID_SIZES))


def row_grid(size: str) -> list[tuple[int, int, int]]:
    h, w = ROW_GRID_SIZES[size]
    return [(1, h, w)]


# The ring: 4 Blackhole chips along mesh axis 1, 2 links between neighbours. On a LoudBox (8 chips,
# 2x4 mesh, torus links along the 4-long axis) it is the first row: `mesh_device` in conftest.py
# opens all 8 chips and hands the test that row. `FABRIC_2D_TORUS_X` because `FABRIC_1D_RING` cannot
# route the ring's wraparound link there.
LOUDBOX_SHAPE = (2, RING_SIZE)
RING_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}
RING4 = pytest.param((1, RING_SIZE), 1, 2, RING_FABRIC, id="ring1x4")
# One P150 standing in for one chip of that ring: it runs that chip's quarter of the padded sequence
# (`device_config`, `MeshRows`), never the whole of it.
P150 = pytest.param((1, 1), None, 1, {}, id="p150")


def mesh_params(*params):
    return pytest.mark.parametrize(
        ("mesh_device", "sp_axis", "num_links", "device_params"),
        list(params),
        indirect=["mesh_device", "device_params"],
    )


# Attention needs the ring. Parts that work on each row alone also run on one P150.
RING = mesh_params(RING4)
P150_AND_RING = mesh_params(P150, RING4)

# Traced tests (`Trace`): the ring with DRAM set aside for the captured commands.
TRACE_REGION_SIZE = 100_000_000
RING_TRACE = mesh_params(
    pytest.param((1, RING_SIZE), 1, 2, {**RING_FABRIC, "trace_region_size": TRACE_REGION_SIZE}, id="ring1x4")
)
# Untimed replays, then timed ones (`Trace.time_replays`).
TRACE_WARMUP = 2
TRACE_ITERATIONS = 10

# ttsim (TT_METAL_SIMULATOR) simulates one chip at about 1/100 of silicon speed, too slow for the
# full padded length (16,384 rows per chip). Sim tests opt in with `@SIMULATOR`; on ttsim, conftest.py
# skips every other device test. Each row-wise part has one: one chip, random weights, SIMULATOR_GRID
# in a config padded to SIMULATOR_ROWS (`simulator_config`), so the chip runs its quarter, 128 rows.
# They also run on hardware.
ON_SIMULATOR = bool(os.environ.get("TT_METAL_SIMULATOR"))
SIMULATOR_MARK = "simulator"
SIMULATOR_ROWS = 512
# 120 rows, all in the first chip's quarter: the real rows end inside a tile, padding after them.
SIMULATOR_GRID = [(1, 6, 20)]


def SIMULATOR(test):
    """Mark `test` as a sim test: the `simulator` mark, one P150, and random weights if it takes
    `reference`."""
    marks = [getattr(pytest.mark, SIMULATOR_MARK), mesh_params(P150)]
    if "reference" in inspect.signature(test).parameters:
        marks.append(pytest.mark.parametrize("reference", ["random"], indirect=True))
    for mark in marks:
        test = mark(test)
    return test


def simulator_config(config: MoonViTConfig) -> MoonViTConfig:
    """`config` padded to SIMULATOR_ROWS, for every sim test: the checks, host prep and device tables
    all follow `max_patches`, and `device_config` gives one chip its quarter, as for the full length."""
    return dataclasses.replace(config, max_patches=SIMULATOR_ROWS)


def device_config(config: MoonViTConfig, sp_axis: int | None) -> MoonViTConfig:
    """The config the TT modules build with. On one P150 (`sp_axis` None) the padded length is one ring
    chip's share, `max_patches / RING_SIZE` rows, so the chip sees the shapes a ring chip sees. Host
    prep and `check_grid` keep the full config: the input is the ring's, and `MeshRows` cuts the
    chip's quarter from it."""
    if sp_axis is not None:
        return config
    return dataclasses.replace(config, max_patches=config.max_patches // RING_SIZE)


def ring_parallel(mesh_device, sp_axis: int | None, num_links: int) -> tuple[ParallelFactor | None, TT_CCL | None]:
    """`(sequence_parallel, ccl)` for `resolve_parallel`; `(None, None)` on a single device."""
    if sp_axis is None:
        return None, None
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    return ParallelFactor(sp_factor, sp_axis), TT_CCL(mesh_device, num_links=num_links, topology=ttnn.Topology.Ring)


def quality(golden: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """How close `actual` is to `golden`:

    - `pcc`: Pearson correlation (`comp_pcc`). Blind to a scale or offset error.
    - `relative_rmse`: RMS error over the golden's standard deviation. Catches what PCC misses.
    - `ccc`: concordance correlation, PCC lowered by any scale or offset difference.
    - `max_abs`, `max_rel`: the largest element differences (as `comp_allclose` reports them; `max_rel`
      over nonzero golden values), for diagnosis.
    """
    golden, actual = golden.float(), actual.float()
    _, pcc = comp_pcc(golden, actual, 0.0)
    g, a = golden.flatten().double(), actual.flatten().double()
    std_g, std_a = g.std().item(), a.std().item()
    cov = ((g - g.mean()) * (a - a.mean())).mean().item()
    ccc = 2 * cov / (std_g**2 + std_a**2 + (g.mean() - a.mean()).item() ** 2)
    relative_rmse = (g - a).pow(2).mean().sqrt().item() / std_g
    diff = (golden - actual).abs()
    nonzero = golden != 0
    max_rel = (diff[nonzero] / golden[nonzero].abs()).max().item() if nonzero.any() else 0.0
    max_abs = diff.max().item()
    return {"pcc": pcc, "relative_rmse": relative_rmse, "ccc": ccc, "max_abs": max_abs, "max_rel": max_rel}


def format_quality(q: dict[str, float]) -> str:
    return (
        f"PCC {q['pcc']:.6f}, relative RMSE {q['relative_rmse']:.2%}, CCC {q['ccc']:.6f}, "
        f"max abs {q['max_abs']:.3g}, max rel {q['max_rel']:.3g}"
    )


def quality_failures(q: dict[str, float], pcc: float, relative_rmse: float | None) -> list[str]:
    failures = []
    if not q["pcc"] >= pcc:
        failures.append(f"PCC {q['pcc']:.6f} < {pcc}")
    if relative_rmse is not None and not q["relative_rmse"] <= relative_rmse:
        failures.append(f"relative RMSE {q['relative_rmse']:.2%} > {relative_rmse:.2%}")
    return failures


def assert_quality(
    golden: torch.Tensor, actual: torch.Tensor, pcc: float = 0.99, relative_rmse: float | None = None, label: str = ""
) -> dict[str, float]:
    """Assert PCC >= `pcc` and relative RMSE <= `relative_rmse`; log every `quality` measure.

    Thresholds follow tt_dit's `assert_quality` (PCC plus relative RMSE); CCC and the largest element
    differences are logged, not asserted. Each test's relative RMSE limit is about twice the largest
    value measured on Blackhole over its cases (random and checkpoint weights, both meshes), rounded
    up. Tests whose forward is not written yet have none (`relative_rmse=None` logs it only)."""
    label = label or os.environ.get("PYTEST_CURRENT_TEST", "quality").split(" ")[0]
    q = quality(golden, actual)
    logger.info(f"{label}: {format_quality(q)}")
    failures = quality_failures(q, pcc, relative_rmse)
    assert not failures, f"{label}: {failures}"
    return q


def random_tensor(*shape: int, dtype: torch.dtype = torch.bfloat16, seed: int = 0) -> torch.Tensor:
    """Standard-normal test input from its own seeded generator, so it does not depend on the global
    RNG or on which tests ran before."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def random_patches(config: MoonViTConfig, grid_thws: list[tuple[int, int, int]], seed: int = 0):
    """Patches in the processor's output range ((x/255 - 0.5)/0.5 lies in [-1, 1])."""
    n = sum(t * h * w for t, h, w in grid_thws)
    g = torch.Generator().manual_seed(seed)
    pixels = torch.rand((n, config.in_channels, config.patch_size, config.patch_size), generator=g) * 2 - 1
    return pixels, torch.tensor(grid_thws, dtype=torch.int64)


def reference_rope(reference, grid_thws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(cu_seqlens, max_seqlen, rope_freqs_cis)` as `MoonViT3dEncoder.forward` builds them."""
    rope_freqs_cis = reference.vision_tower.encoder.rope_2d.get_freqs_cis(
        grid_thws=grid_thws, device=torch.device("cpu")
    )
    lengths = torch.cat(
        (
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        )
    )
    max_seqlen = lengths.max()
    cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens, max_seqlen, rope_freqs_cis


def sp_shard(x: torch.Tensor, mesh_device, sp_axis: int | None, rows: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """(L, C) or (H, L, C) on the host -> [1, 1 or H, rows, C] `dtype` on device, zero-padded to `rows`,
    split along `sp_axis`, or replicated when `sp_axis` is None. fp32 for the encoder's residual stream
    (block inputs) and the position embeddings added into it."""
    x = x if x.ndim == 3 else x[None]
    if x.shape[-2] > rows:
        raise ValueError(f"{x.shape[-2]} rows do not fit the padded length {rows}")
    padded = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[-2]))[None]
    if sp_axis is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        dims = [None, None]
        dims[sp_axis] = 2
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    return ttnn.from_torch(padded, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def sp_gather(x: ttnn.Tensor, mesh_device, sp_axis: int | None, logical_n: int) -> torch.Tensor:
    """The inverse of `sp_shard`: -> (logical_n, C), or (H, logical_n, C) when H > 1.

    Replicas are stacked on dim 0 and the first is kept.
    """
    if sp_axis is None:
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    else:
        dims = [0, 0]
        dims[sp_axis] = 2
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    out = ttnn.to_torch(x, mesh_composer=composer)[0, :, :logical_n]
    return out[0] if out.shape[0] == 1 else out


@dataclasses.dataclass(frozen=True)
class MeshRows:
    """The rows of one padded sequence that the mesh under test holds: `n` real rows padded to `rows`,
    the ring's length, in the order the ring splits them (window order for patches in the model).

    On the ring (`sp_axis` set) that is every row. On one chip (`sp_axis` None) it is one ring chip's
    quarter, `rows // RING_SIZE` rows: the last chip that holds real rows, so the quarter has real rows
    and, unless the image fills the padded length, the padding after them, as that ring chip has.

    Host tensors have their rows on dim 0, real (`n`) or already padded (`rows`).
    """

    n: int
    rows: int
    sp_axis: int | None

    @property
    def share(self) -> int:
        """Padded rows on the mesh."""
        return self.rows if self.sp_axis is not None else self.rows // RING_SIZE

    @property
    def start(self) -> int:
        """The first padded row on the mesh."""
        if self.sp_axis is not None:
            return 0
        return min((self.n - 1) // self.share, RING_SIZE - 1) * self.share

    @property
    def real(self) -> slice:
        """The real rows on the mesh."""
        return slice(self.start, min(self.start + self.share, self.n))

    def take(self, x: torch.Tensor) -> torch.Tensor:
        """The real rows on the mesh: the rows of a golden to compare with `gather`."""
        return x[self.real]

    def held(self, x: torch.Tensor) -> torch.Tensor:
        """`x` zero-padded to `rows`, then the `share` rows on the mesh."""
        return host_prep.pad_rows(x, self.rows)[self.start : self.start + self.share]

    def shard(self, x: torch.Tensor, mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """(L, C) or (L, H, C) on the host -> the mesh's rows on device, [1, 1 or H, share/sp, C]."""
        x = self.held(x)
        return sp_shard(x if x.ndim == 2 else x.transpose(0, 1), mesh_device, self.sp_axis, self.share, dtype)

    def gather(self, x: ttnn.Tensor, mesh_device) -> torch.Tensor:
        """The inverse of `shard` with the padding dropped: -> (real rows, C) or (real rows, H, C)."""
        out = sp_gather(x, mesh_device, self.sp_axis, self.real.stop - self.real.start)
        return out if out.ndim == 2 else out.transpose(0, 1)


class Trace:
    """One captured trace of `forward()`, replayed on command queue 0.

    `forward` must read only inputs that keep their address (tensors kept alive, or `DeviceInput`s),
    and must have run once untraced first: that compiles its programs and allocates the ring attention
    semaphores and buffers, which cannot happen inside a capture. Every replay writes `output`. Use it
    as a context manager, so the trace is released when a check fails.
    """

    def __init__(self, mesh_device, forward) -> None:
        self.mesh_device = mesh_device
        self.trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            self.output = forward()
        except BaseException:
            ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
            ttnn.release_trace(mesh_device, self.trace_id)
            raise
        ttnn.end_trace_capture(mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *exc) -> None:
        ttnn.release_trace(self.mesh_device, self.trace_id)

    def _execute(self) -> None:
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)

    def replay(self) -> ttnn.Tensor:
        """Run once on the current inputs and wait for it; returns `output`."""
        self._execute()
        ttnn.synchronize_device(self.mesh_device)
        return self.output

    def time_replays(self, label: str, iterations: int = TRACE_ITERATIONS, warmup: int = TRACE_WARMUP) -> float:
        """Seconds per replay on the current inputs: `warmup` untimed replays, then `iterations` back to
        back with one wait for the device at the end, so that wait is spread over all of them (as the
        CCL perf tests time their traces). Nothing is written or read in between. Logged, not asserted."""
        for _ in range(warmup):
            self._execute()
        ttnn.synchronize_device(self.mesh_device)
        start = time.perf_counter()
        for _ in range(iterations):
            self._execute()
        ttnn.synchronize_device(self.mesh_device)
        seconds = (time.perf_counter() - start) / iterations
        logger.info(f"{label}: {seconds * 1e3:.3f} ms per traced replay ({iterations} replays)")
        return seconds


def assert_same_as_untraced(untraced: torch.Tensor, traced: torch.Tensor, label: str = "") -> None:
    """A replay runs the same programs on the same inputs as the untraced run, so it must match it
    exactly."""
    assert torch.equal(
        untraced, traced
    ), f"{label}: traced output differs from the untraced run ({format_quality(quality(untraced, traced))})"
