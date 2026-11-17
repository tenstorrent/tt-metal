# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Test constants and helpers shared across this folder."""

from __future__ import annotations

import copy
import dataclasses
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import (
    CHECKPOINT_ENV,
    checkpoint_path,
    load_vision_state_dict,
)
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision import KimiK3VisionModel, load_vision_config
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.ccl import ParallelFactor, TT_CCL
from models.experimental.kimi_k3_moonvit_v2.tt.device_input import sp_mesh_mapper
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import (
    RING_SIZE,
    MoonViTConfig,
    check_grid,
    compute_kernel_config,
    resolve_parallel,
)
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope

# Weights. A test that builds `reference` carries one or both of these marks,
# and conftest.py runs it once per mark: SYNTHETIC with random weights, REAL with the checkpoint at
# $KIMI_K3_HF_MODEL. `-m synthetic` or `-m real` selects by weights.
SYNTHETIC_MARK = "synthetic"
REAL_MARK = "real"
WEIGHTS = [SYNTHETIC_MARK, REAL_MARK]
SYNTHETIC = getattr(pytest.mark, SYNTHETIC_MARK)
REAL = getattr(pytest.mark, REAL_MARK)

# Layers, for tests that take `layer`. Independent of the weights.
NUM_LAYERS = MoonViTConfig().num_hidden_layers


def layer_id(layer: int) -> str:
    return f"layer{layer}"


# MOONVIT_TEST_LAYER (e.g. "5") runs that one layer in place of SAMPLED_LAYERS.
LAYERS_ENV = "MOONVIT_TEST_LAYER"


def _sampled_layers() -> list[int]:
    value = os.environ.get(LAYERS_ENV)
    if not value:
        return [0, NUM_LAYERS // 2, NUM_LAYERS - 1]
    if not value.isdigit() or int(value) >= NUM_LAYERS:
        raise ValueError(f"${LAYERS_ENV} must be one layer, 0 to {NUM_LAYERS - 1}; got {value!r}")
    return [int(value)]


# The first, middle and last layer, or the one layer of $MOONVIT_TEST_LAYER.
SAMPLED_LAYERS = pytest.mark.parametrize("layer", _sampled_layers(), ids=layer_id)
# Every layer, whatever $MOONVIT_TEST_LAYER says.
ALL_LAYERS = pytest.mark.parametrize("layer", range(NUM_LAYERS), ids=layer_id)

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
# grows only with the row count, so it runs at the full padded length in seconds. One P150 runs one
# ring chip's quarter of the image (`MeshRows`):
#   full:     fills the padded length exactly; real rows on every ring chip, no padding
#   max_side: also fills it, with the longest side the processor allows (512)
#   partial:  nearly full; the real rows end inside a tile (not a multiple of 32) on the last chip
#   small:    a small image, almost all padding; every real row is on the first chip
PADDING_LAYOUT_SIZES = {"full": (256, 256), "max_side": (512, 128), "partial": (250, 250), "small": (40, 100)}
PADDING_LAYOUT_GRIDS = pytest.mark.parametrize("size", list(PADDING_LAYOUT_SIZES))


def padding_layout_grid(size: str) -> list[tuple[int, int, int]]:
    h, w = PADDING_LAYOUT_SIZES[size]
    return [(1, h, w)]


# Attention and block tests with real rows on every chip: the ring runs the production padded length
# (`max_patches`) on a PADDING_LAYOUT_SIZES image with real rows on every chip. The CPU golden at that
# length is too slow for every query row, so it computes SAMPLED_ROWS_PER_CHIP query rows of each chip,
# each against every key (`goldens.use_sampled_attention`), and each chip's rows are checked on their own.
REAL_ROWS_FROM_EACH_CHIP_ASSERTED_GRIDS = pytest.mark.parametrize("size", ["full", "max_side", "partial"])
SAMPLED_ROWS_PER_CHIP = 256


def sampled_rows(n: int, rows: int, per_chip: int = SAMPLED_ROWS_PER_CHIP, seed: int = 0) -> list[torch.Tensor]:
    """Per ring chip, sorted window-order indices of `per_chip` of the real rows it holds, for `n` real
    rows padded to `rows`: its first and last `per_chip // 4` real rows (where the image ends inside a
    tile on the last chip, the rows before the padding) and the rest drawn between them. A chip with
    fewer real rows gives all of them; one with none, an empty tensor."""
    share = rows // RING_SIZE
    edge = per_chip // 4
    generator = torch.Generator().manual_seed(seed)
    chips = []
    for chip in range(RING_SIZE):
        real = torch.arange(chip * share, min((chip + 1) * share, n))
        if len(real) <= per_chip:
            chips.append(real)
            continue
        middle = real[edge:-edge]
        drawn = middle[torch.randperm(len(middle), generator=generator)[: per_chip - 2 * edge]]
        chips.append(torch.cat([real[:edge], drawn, real[-edge:]]).sort().values)
    return chips


def _label(label: str) -> str:
    """`label`, or the running test's id when it is empty."""
    return label or os.environ.get("PYTEST_CURRENT_TEST", "quality").split(" ")[0]


def assert_quality_per_chip(
    golden: torch.Tensor,
    actual: torch.Tensor,
    chips: list[torch.Tensor],
    pcc: float = 0.99,
    relative_rmse: float | None = None,
    label: str = "",
) -> None:
    """`assert_quality` on each chip's rows (`chips`, as `sampled_rows` gives them) of `golden` and
    `actual`, both window order. Every chip is checked and logged before any failure is raised, so a
    wrong chip cannot hide behind three right ones."""
    label = _label(label)
    parts = []
    for chip, rows in enumerate(chips):
        assert len(rows), f"{label}: chip {chip} holds no real rows"
        parts.append((f"chip {chip} ({len(rows)} rows)", golden[rows], actual[rows]))
    assert_quality_parts(parts, pcc, relative_rmse, label)


def assert_quality_parts(
    parts: list[tuple[str, torch.Tensor, torch.Tensor]], pcc: float, relative_rmse: float | None, label: str
) -> None:
    """`quality` of each `(name, golden, actual)` in `parts`. Every part is checked and logged before
    any failure is raised."""
    failed = []
    for name, golden, actual in parts:
        q = quality(golden, actual)
        logger.info(f"{label} {name}: {format_quality(q)}")
        failed += [f"{name} {failure}" for failure in quality_failures(q, pcc, relative_rmse)]
    assert not failed, f"{label}: {failed}"


# The ring: 4 Blackhole chips along mesh axis 1, 2 links between neighbours. On a LoudBox (8 chips,
# 2x4 mesh, torus links along the 4-long axis) it is the first row: `mesh_device` in conftest.py
# opens all 8 chips and hands the test that row. On a Galaxy (8x4) it opens all 32 and does the same. `FABRIC_2D_TORUS_X` because `FABRIC_1D_RING` cannot
# route the ring's wraparound link there.
LOUDBOX_SHAPE = (2, RING_SIZE)
RING_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}
RING_NUM_LINKS = 2
# Traced tests (`Trace`) set aside this much DRAM for the captured commands.
TRACE_REGION_SIZE = 100_000_000


def mesh_params(shape: tuple[int, int], sp_axis: int | None, num_links: int, device_params: dict, id: str):
    """Run a test on one mesh: parametrizes `mesh_device`, `sp_axis`, `num_links` and `device_params`."""
    return pytest.mark.parametrize(
        ("mesh_device", "sp_axis", "num_links", "device_params"),
        [pytest.param(shape, sp_axis, num_links, device_params, id=id)],
        indirect=["mesh_device", "device_params"],
    )


def ring_params(**device_params):
    """The ring, with `device_params` added to its fabric config."""
    return mesh_params((1, RING_SIZE), 1, RING_NUM_LINKS, {**RING_FABRIC, **device_params}, "ring1x4")


# Attention needs the ring, so only tests with attention in them use it. Parts that work on each row
# alone run on one P150, standing in for one chip of the ring: it runs that chip's quarter of the
# padded sequence (`device_config`, `MeshRows`), never the whole of it.
RING = ring_params()
RING_TRACE = ring_params(trace_region_size=TRACE_REGION_SIZE)
P150 = mesh_params((1, 1), None, 1, {}, "p150")

# Untimed replays, then timed ones (`Trace.time_replays`).
TRACE_WARMUP = 2
TRACE_ITERATIONS = 10

# ttsim (TT_METAL_SIMULATOR) simulates one chip far slower than silicon, too slow for the
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
    """Mark `test` as a sim test: the `simulator` mark and one P150. Weights come from the test's
    own SYNTHETIC or REAL."""
    return P150(getattr(pytest.mark, SIMULATOR_MARK)(test))


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
    - `nonfinite`: NaN and Inf values in `actual`. `comp_pcc` sets them to zero, so PCC alone misses a few.
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
    nonfinite = int((~torch.isfinite(actual)).sum())
    return {
        "pcc": pcc,
        "relative_rmse": relative_rmse,
        "ccc": ccc,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonfinite": nonfinite,
    }


def format_quality(q: dict[str, float]) -> str:
    return (
        f"PCC {q['pcc']:.6f}, relative RMSE {q['relative_rmse']:.2%}, CCC {q['ccc']:.6f}, "
        f"max abs {q['max_abs']:.3g}, max rel {q['max_rel']:.3g}, {q['nonfinite']} NaN or Inf"
    )


def quality_failures(q: dict[str, float], pcc: float, relative_rmse: float | None) -> list[str]:
    failures = []
    if q["nonfinite"]:
        failures.append(f"{q['nonfinite']} NaN or Inf values")
    if not q["pcc"] >= pcc:
        failures.append(f"PCC {q['pcc']:.6f} < {pcc}")
    if relative_rmse is not None and not q["relative_rmse"] <= relative_rmse:
        failures.append(f"relative RMSE {q['relative_rmse']:.2%} > {relative_rmse:.2%}")
    return failures


def assert_quality(
    golden: torch.Tensor, actual: torch.Tensor, pcc: float = 0.99, relative_rmse: float | None = None, label: str = ""
) -> dict[str, float]:
    """Assert no NaN or Inf, PCC >= `pcc` and relative RMSE <= `relative_rmse`; log every `quality` measure.

    Thresholds follow tt_dit's `assert_quality` (PCC plus relative RMSE); CCC and the largest element
    differences are logged, not asserted. Each test's relative RMSE limit is about twice the largest
    value measured on Blackhole over its cases (random and checkpoint weights, both meshes), rounded
    up. `relative_rmse=None` logs it without asserting it."""
    label = _label(label)
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


def build_reference(weights: str) -> KimiK3VisionModel:
    """bf16, eval mode, as the checkpoint runs (upstream RMSNorm eps follows the input dtype)."""
    path = checkpoint_path()
    if weights == REAL_MARK and path is None:
        pytest.skip(f"set ${CHECKPOINT_ENV} to a Kimi-K3 checkpoint folder")
    vision_config = load_vision_config(path)
    torch.manual_seed(0)
    model = KimiK3VisionModel(vision_config)
    if weights == REAL_MARK:
        model.load_state_dict(load_vision_state_dict(path))
    else:
        # RMSNorm initializes to ones, which would hide a norm weight that fails to load.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" in name:
                    param.copy_(1.0 + 0.1 * torch.randn_like(param))
    return model.to(torch.bfloat16).eval()


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


def sp_shard(
    x: torch.Tensor, mesh_device, sp_axis: int | None, rows: int, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
) -> ttnn.Tensor:
    """(L, C) or (H, L, C) on the host -> [1, 1 or H, rows, C] `dtype` on device, zero-padded to `rows`,
    split along `sp_axis`, or replicated when `sp_axis` is None. fp32 for the encoder's residual stream
    (block inputs) and the position embeddings added into it; row major for the patches, which the
    patch embed tiles on device."""
    x = x if x.ndim == 3 else x[None]
    if x.shape[-2] > rows:
        raise ValueError(f"{x.shape[-2]} rows do not fit the padded length {rows}")
    padded = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[-2]))[None]
    mapper = sp_mesh_mapper(mesh_device, sp_axis)
    return ttnn.from_torch(padded, device=mesh_device, dtype=dtype, layout=layout, mesh_mapper=mapper)


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


def logical_length_tensor(mesh_device, n: int) -> ttnn.Tensor:
    """The unpadded patch count as the `logical_n` the ring attention reads: [1, 1, 1, 1] uint32, row
    major, replicated. The model writes it into a `DeviceInput` instead, so a trace can replay it."""
    return ttnn.from_torch(
        torch.tensor([n], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


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
    def real_slice(self) -> slice:
        """The real rows on the mesh."""
        return slice(self.start, min(self.start + self.share, self.n))

    def real_rows(self, x: torch.Tensor) -> torch.Tensor:
        """The real rows on the mesh: the rows of a golden to compare with `gather`."""
        return x[self.real_slice]

    def padded_rows(self, x: torch.Tensor) -> torch.Tensor:
        """`x` zero-padded to `rows`, then the `share` rows on the mesh."""
        return host_prep.pad_rows(x, self.rows)[self.start : self.start + self.share]

    def shard(self, x: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        """(L, C) or (L, H, C) on the host -> the mesh's rows on device, [1, 1 or H, share/sp, C]."""
        x = self.padded_rows(x)
        return sp_shard(x if x.ndim == 2 else x.transpose(0, 1), mesh_device, self.sp_axis, self.share, dtype, layout)

    def gather(self, x: ttnn.Tensor, mesh_device) -> torch.Tensor:
        """The inverse of `shard` with the padding dropped: -> (real rows, C) or (real rows, H, C)."""
        out = sp_gather(x, mesh_device, self.sp_axis, self.real_slice.stop - self.real_slice.start)
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


def check_ring_layer(
    mesh_device,
    sp_axis: int,
    num_links: int,
    config: MoonViTConfig,
    reference,
    layer: int,
    grid: list[tuple[int, int, int]],
    *,
    tt_class,
    reference_call,
    state_prefixes: tuple[str, ...] | None = None,
    input_dtype=ttnn.bfloat16,
    relative_rmse: float,
    real_rows_from_each_chip: bool,
) -> None:
    """`tt_class` (built like `TT_Vision_Attention`, loaded from encoder block `layer`, only the keys
    starting with `state_prefixes` when given) on the ring against `reference_call(block, x, cu_seqlens,
    max_seqlen, rope_freqs_cis)`. The device runs window-ordered rows.

    By default the golden is the bf16 block over every row, compared as a whole in raster order. With
    `real_rows_from_each_chip` (an image with real rows on every chip, at the production padded length)
    the fp32 golden covers `sampled_rows` of each chip only, each against every key
    (`goldens.use_sampled_attention`), and each chip's rows are checked on their own."""
    # Imported here: goldens.py imports this module.
    from models.experimental.kimi_k3_moonvit_v2.tests.goldens import to_fp32, use_sampled_attention

    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = tt_class(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    block = reference.vision_tower.encoder.blocks[layer]
    state = block.state_dict()
    if state_prefixes is not None:
        state = {k: v for k, v in state.items() if k.startswith(state_prefixes)}
    tt.load_torch_state_dict(state)

    grid_thws = torch.tensor(grid)
    h, w = check_grid(grid_thws, config)
    kernel = config.merge_kernel_size
    x = random_tensor(h * w, config.hidden_size)
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    if real_rows_from_each_chip:
        chips = sampled_rows(h * w, config.max_patches)
        query_rows = host_prep.window_order(h, w, kernel)[torch.cat(chips)].sort().values
        with torch.no_grad(), use_sampled_attention(query_rows):
            golden = reference_call(to_fp32(copy.deepcopy(block)), x.float(), cu_seqlens, max_seqlen, rope_freqs_cis)
    else:
        with torch.no_grad():
            golden = reference_call(block, x, cu_seqlens, max_seqlen, rope_freqs_cis)

    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, h, w))
    x_window = host_prep.to_window_order(x, h, w, kernel)
    logical_n = logical_length_tensor(mesh_device, h * w)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches, input_dtype), rope, logical_n)
    out = sp_gather(out, mesh_device, sp_axis, h * w)
    if real_rows_from_each_chip:
        golden = host_prep.to_window_order(golden, h, w, kernel)
        assert_quality_per_chip(golden, out, chips, relative_rmse=relative_rmse)
    else:
        assert_quality(golden, host_prep.from_window_order(out, h, w, kernel), relative_rmse=relative_rmse)


def assert_same_as_untraced(untraced: torch.Tensor, traced: torch.Tensor, label: str = "") -> None:
    """A replay runs the same programs on the same inputs as the untraced run, so it must match it
    exactly."""
    assert torch.equal(
        untraced, traced
    ), f"{label}: traced output differs from the untraced run ({format_quality(quality(untraced, traced))})"
