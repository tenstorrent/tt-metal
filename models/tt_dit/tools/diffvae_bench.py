# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the DiffVAE timing instruments.

The pytest timing cases in ``tests/models/vae/test_diffvae_ltx.py`` and the ``time_module`` CLI
call the same functions here, so a number from either is comparable with a number from the other.
The only thing the two paths do differently is acquire the mesh: pytest through the ``mesh_device``
fixture, the CLI through :func:`open_mesh`.

Every instrument here has the same shape. A *bench* is a built module plus one ``step`` that runs
it once on inputs it owns; :func:`timed` warms the program cache and measures. The deterministic
blocks take an *arm*, a :class:`DetBlockOptions` naming the fusions the block builds with.

What a run measures is carried in two objects, so the defaults are the same in the tests and the
tool: a :class:`DiffVAEOptions` for how the decoder is built (the pytest ``diffvae_options`` fixture
or ``time_module``'s flags produce it) and a :class:`BenchOptions` for the instrument itself
(latent frames, iterations, collectives, the stage-5 grid).
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.tt_dit.layers.neighborhood_attention_plan import build_device_plan, plan_na3d
from models.tt_dit.models.vae.diffvae_ltx import (
    DetBlockOptions,
    DiffVAEDecoder,
    DiffVAEOptions,
    NABlock,
    decoder_config,
    rope_tables,
)
from models.tt_dit.models.vae.diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config, Grid
from models.tt_dit.models.vae.diffvae_ops import wshard
from models.tt_dit.models.vae.diffvae_rope import default_rope_dim_split
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import timing_tree

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)

HEAD_DIM = 64
SP_AXIS, TP_AXIS = 1, 0
BRICKED = "bricked_sp_w_sharded"

#: The 1080p decode's latent (H, W); every instrument's default geometry derives from it.
LATENT_HW = (34, 60)

# ---------------------------------------------------------------------------
# Deterministic-block arms and geometry
# ---------------------------------------------------------------------------

#: The unflagged block every arm is compared against.
BASELINE = DetBlockOptions()

#: arm -> block options. Every arm and the baseline run the bricked W-sharded executor;
#: ``recommended`` is the production set for the W-sharded stages.
ARMS: dict[str, DetBlockOptions] = {
    "fused_qkv": DetBlockOptions(fused_qkv=True),
    "colpar_qkv": DetBlockOptions(colpar_qkv=True),
    "fused_swiglu": DetBlockOptions(fused_swiglu=True),
    "tp_mlp": DetBlockOptions(tp_mlp=True),
    "recommended": DetBlockOptions(colpar_qkv=True, fused_rope=True, fused_swiglu=True),
}

#: Stage 1 runs replicated with no TP axis, so only the arms that need neither can reach it.
STAGE1_ARMS: dict[str, DetBlockOptions] = {
    "swiglu": DetBlockOptions(fused_swiglu=True),
    "qkv": DetBlockOptions(fused_qkv=True),
    "qkv_rope": DetBlockOptions(fused_qkv=True, fused_rope=True),
    "recommended": DetBlockOptions(fused_qkv=True, fused_rope=True, fused_swiglu=True),
}

#: (label, dim, kernel, full dims, blocks in that stage) for the W-sharded deterministic stages, at
#: the s34x60 decode's geometry.
STAGES = [
    ("stage2", 1024, (3, 7, 7), (6, 68, 120), 6),
    ("stage3", 512, (3, 5, 5), (11, 68, 120), 4),
    ("stage4", 512, (3, 5, 5), (21, 136, 240), 2),
]

#: Stage 1 of the same decode: (dim, kernel, dims, blocks). Its W=60 does not divide the size-8 axis,
#: so it runs replicated on the linear-order executor with no TP axis.
STAGE1 = (2048, (3, 7, 7), (6, 34, 60), 4)


def arm_options(arm: str, table: dict[str, DetBlockOptions] = ARMS) -> DetBlockOptions:
    """``"baseline"`` is the unflagged block; anything else must be a key of ``table``."""
    if arm == "baseline":
        return BASELINE
    return table[arm]


def _assert_built(block: NABlock, arm: DetBlockOptions, tp_axis: int | None) -> None:
    """The block built exactly the forms ``arm`` resolves to at this ``tp_axis``."""
    expected = arm.resolve(tp_axis)
    built = DetBlockOptions(
        fused_qkv=block.attn.fused_qkv,
        colpar_qkv=block.attn.colpar_qkv,
        fused_rope=block.attn.fused_rope,
        fused_swiglu=block.mlp.fused,
        tp_mlp=block.mlp.tp_mlp,
    )
    assert built == expected, f"built {built}, expected {expected}"


def build_det_block(mesh, dim: int, kernel, arm: DetBlockOptions, backend: str = BRICKED) -> NABlock:
    """An NABlock built with ``arm`` on ``backend``, asserting the forms actually took."""
    block = NABlock(
        dim,
        kernel,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend=backend,
        ccl_manager=CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        options=arm,
    )
    _assert_built(block, arm, TP_AXIS)
    assert block.attn.na3d_backend == backend
    return block


def build_stage1_block(mesh, arm: DetBlockOptions) -> NABlock:
    """A replicated stage-1 block, asserting the forms took and that the column-parallel ones did not."""
    dim, kernel, _, _ = STAGE1
    block = NABlock(
        dim,
        kernel,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend="linear_order",
        ccl_manager=None,
        sp_axis=None,
        tp_axis=None,
        options=arm,
    )
    assert block.attn.tp == 1
    _assert_built(block, arm, None)
    assert block.attn.colpar_qkv is False, "colpar needs a tp_axis to shard the weight over"
    return block


def det_inputs(mesh, dim: int, dims):
    """Tokens per chip, this chip's local dims, and the W-partitioned RoPE tables."""
    sp = int(list(mesh.shape)[SP_AXIS])
    t, h, w = dims
    tokens = t * h * (w // sp)
    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh)
    cos = ttnn.mesh_partition(cos, dim=3, cluster_axis=SP_AXIS)
    sin = ttnn.mesh_partition(sin, dim=3, cluster_axis=SP_AXIS)
    return tokens, (t, h, w // sp), cos, sin


def stage1_plan(mesh, dims, kernel):
    """The linear-order executor query-shards across the mesh; the plan takes the CCL for that."""
    return build_device_plan(
        plan_na3d(dims, kernel),
        mesh_device=mesh,
        ccl_manager=CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear),
    )


def named_params(module, prefix: str = ""):
    """``named_parameters`` is not recursive; walk the children."""
    for name, param in module.named_parameters():
        yield f"{prefix}{name}", param
    for name, child in module.named_children():
        yield from named_params(child, f"{prefix}{name}.")


def seeded(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """A deterministic weight for ``name``: norms near one, everything else at fan-in scale."""
    g = torch.Generator().manual_seed(7)
    if "norm" in name:
        return (1.0 + 0.05 * torch.randn(*shape, generator=g)).to(torch.float32)
    scale = shape[-2] ** -0.5 if len(shape) > 1 else 0.02
    return (torch.randn(*shape, generator=g) * scale).to(torch.float32)


def fill(module) -> None:
    """Load seeded weights into every parameter. Timing needs sane magnitudes, not a checkpoint."""
    for name, param in named_params(module):
        param.load_torch_tensor(seeded(name, tuple(param.total_shape)))


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchOptions:
    """What an instrument runs with, as distinct from how the decoder is built (:class:`DiffVAEOptions`).

    Every field is optional so a test or tool keeps its own default when the caller says nothing:
    the accessor methods take that default.
    """

    #: Latent frames for the decoder instruments; output frames are ``8 * T - 7`` (19 -> 145).
    latent_t: int | None = None
    #: Timed iterations after the warm-ups.
    iters: int | None = None
    #: Collectives. Linear/1-link is what the committed gate baselines were recorded with; the
    #: runner ships ring + 2 links.
    topology: ttnn.Topology = ttnn.Topology.Linear
    num_links: int | None = None
    #: The stage-5 grid for the block instrument.
    grid: Grid | None = None

    def latent_frames(self, default: int = 19) -> int:
        return default if self.latent_t is None else self.latent_t

    def iterations(self, default: int = 10) -> int:
        return default if self.iters is None else self.iters

    def ccl(self, mesh, *, default_links: int = 1) -> CCLManager:
        return CCLManager(
            mesh, num_links=default_links if self.num_links is None else self.num_links, topology=self.topology
        )

    @property
    def fabric(self) -> ttnn.FabricConfig:
        """The fabric that matches :attr:`topology`: a ring closes the wraparound link."""
        return ttnn.FabricConfig.FABRIC_1D_RING if self.topology is ttnn.Topology.Ring else ttnn.FabricConfig.FABRIC_1D

    def stage5_grid(self, default: Grid) -> Grid:
        return default if self.grid is None else self.grid


def parse_topology(name: str) -> ttnn.Topology:
    """``"linear"`` or ``"ring"``, as the CLI flags and pytest options spell them."""
    try:
        return {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[name.lower()]
    except KeyError:
        raise ValueError(f"unknown topology {name!r}; expected 'linear' or 'ring'") from None


def parse_grid(spec: str, *, batch: int = 1) -> Grid:
    """``"TxHxW"`` to a :class:`Grid`."""
    t, h, w = (int(v) for v in spec.lower().split("x"))
    return Grid(batch=batch, t=t, h=h, w=w)


def parse_stride(spec: str) -> tuple[int, int, int]:
    """``"t,h,w"`` to a GNA stride."""
    parts = tuple(int(v) for v in spec.split(","))
    if len(parts) != 3:
        raise ValueError(f"gna stride {spec!r} needs three comma-separated ints")
    return parts


def production_decoder(mesh, config: dict, ccl: CCLManager, options: DiffVAEOptions) -> DiffVAEDecoder:
    """A decoder built with ``options``; :meth:`DiffVAEOptions.production` is how the runner ships it.
    Weights are not loaded here."""
    return DiffVAEDecoder(config, mesh_device=mesh, ccl_manager=ccl, options=options)


def describe(options: DiffVAEOptions) -> str:
    """The tag the decode timing lines carry."""
    tp = f"+TP(axis {options.stage5_tp_axis})" if options.stage5_tp_axis is not None else ""
    det = ""
    if options.stages_sp_axis is not None:
        det = f"+detSP({options.stages_backend},sp_axis={options.stages_sp_axis},tp_axis={options.stages_tp_axis})"
    sharded = "W-SP" if options.stage5_sp_axis is not None else "replicated"
    return f"{sharded}({options.stage5_backend}){tp}{det} slab={options.slab_frames}"


def loaded_production_decoder(
    mesh, options: DiffVAEOptions, ccl: CCLManager, *, checkpoint: Path = CHECKPOINT
) -> tuple[DiffVAEDecoder, dict]:
    """:func:`production_decoder` with the checkpoint loaded; returns the decoder and its config."""
    config = decoder_config(checkpoint)
    dec = production_decoder(mesh, config, ccl, options)
    dec.load_checkpoint(checkpoint)
    return dec, config


def latent(config: dict, t: int, hw: tuple[int, int] = LATENT_HW, *, seed: int = 0) -> torch.Tensor:
    lh, lw = hw
    return torch.randn(1, config["in_channels"], t, lh, lw, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Benches: a built module and one step on inputs it owns
# ---------------------------------------------------------------------------


@dataclass
class Bench:
    """One timing instrument. ``step`` runs the module once; ``depth`` is how many of these the decode
    runs, so ``ms * depth`` is the stage's share of a decode."""

    label: str
    step: Callable[[], None]
    depth: int = 1
    tokens_per_chip: int = 0
    describe: str = ""
    _release: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def close(self) -> None:
        for release in self._release:
            release()
        self._release.clear()


def timed(bench: Bench, mesh, iters: int, *, warmup: int = 2) -> float:
    """Milliseconds per ``step`` after ``warmup`` calls, both sides synchronised."""
    for _ in range(warmup):  # warm the program cache
        bench.step()
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(iters):
        bench.step()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / iters * 1000


def block_line(bench: Bench, ms: float) -> str:
    """The one-line report every block timer prints: ``[stageN] ... ms/block  xdepth = ... ms``."""
    return (
        f"[{bench.label:7s}] {bench.describe} tokens/chip={bench.tokens_per_chip:7d} "
        f"| {ms:8.2f} ms/block  x{bench.depth} = {ms * bench.depth:8.1f} ms"
    )


def det_block_bench(mesh, stage, arm: DetBlockOptions) -> Bench:
    """One W-sharded deterministic NABlock of ``stage`` (a :data:`STAGES` row) under ``arm``."""
    label, dim, kernel, dims, depth = stage
    block = build_det_block(mesh, dim, kernel, arm)
    fill(block)
    tokens, local, cos, sin = det_inputs(mesh, dim, dims)
    state = {
        "x": ttnn.from_torch(
            torch.randn(tokens, dim, generator=torch.Generator().manual_seed(3)),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    }

    def step() -> None:
        state["x"] = block(state["x"], dims=local, cos=cos, sin=sin, device_plan=None)

    bench = Bench(
        label,
        step,
        depth=depth,
        tokens_per_chip=tokens,
        describe=f"dim={dim:5d} heads={dim // HEAD_DIM:3d} dims={dims} w_local={local[2]:3d}",
    )
    bench._release.append(lambda: ttnn.deallocate(state["x"]))
    return bench


def stage1_bench(mesh, arm: DetBlockOptions) -> Bench:
    """The replicated stage-1 NABlock under ``arm``. Every chip holds the whole volume."""
    dim, kernel, dims, depth = STAGE1
    t, h, w = dims
    tokens = t * h * w
    plan = stage1_plan(mesh, dims, kernel)
    block = build_stage1_block(mesh, arm)
    fill(block)
    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh)
    state = {
        "x": ttnn.from_torch(
            torch.randn(tokens, dim, generator=torch.Generator().manual_seed(3)),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    }

    def step() -> None:
        state["x"] = block(state["x"], dims=dims, cos=cos, sin=sin, device_plan=plan)

    bench = Bench(
        "stage1",
        step,
        depth=depth,
        tokens_per_chip=tokens,
        describe=f"dim={dim:5d} heads={dim // HEAD_DIM:3d} dims={dims} replicated",
    )
    bench._release.append(lambda: ttnn.deallocate(state["x"]))
    return bench


#: The shipped 1080p 25-frame stage-5 grid.
STAGE5_GRID = Grid(batch=1, t=121, h=128, w=192)


def _flat(x: torch.Tensor, channels: int) -> torch.Tensor:
    """``(B, T, H, W, C)`` -> the module's ``(1, B, sites, C)`` layout."""
    return x.reshape(1, x.shape[0], -1, channels)


def diff_block_bench(mesh, grid: Grid = STAGE5_GRID, *, ccl: CCLManager | None = None) -> Bench:
    """One stage-5 DiffusionNABlock in the production 2-D SP x TP config on the bricked executor.

    Weights are filled rather than loaded: this measures the block, not the checkpoint. The
    inputs are built in the same order as ``DiffVAEStage5.forward``, so the block sees exactly what
    it sees in a decode: context W-sharded to this chip's band, modulation from the shared AdaLN,
    one RoPE table set per band, and, under keep-bricked, activations already in bricked order.
    """
    cfg = DiffVAEStage5Config()
    sp = int(list(mesh.shape)[SP_AXIS])
    assert grid.w % sp == 0, f"W={grid.w} not divisible by sp={sp}"

    model = DiffVAEStage5(
        cfg,
        mesh_device=mesh,
        dtype=ttnn.bfloat16,
        ccl_manager=ccl or CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear),
        na3d_backend=BRICKED,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )
    fill(model)

    g = torch.Generator().manual_seed(99)
    context = torch.randn(grid.batch, grid.t, grid.h, grid.w, cfg.context_channels, generator=g)
    x_t = torch.randn(
        grid.batch, cfg.out_channels, grid.t, grid.h * cfg.patch_size, grid.w * cfg.patch_size, generator=g
    )

    tt_context = ttnn.from_torch(
        _flat(context, cfg.context_channels).contiguous(), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_context = wshard(tt_context, (grid.t, grid.h, grid.w), sp_axis=model.sp_axis)
    bands = model.bands(grid)
    x_bands = model.embed_x_t(x_t, bands)
    timestep = ttnn.from_torch(
        torch.tensor([0.7] * grid.batch).reshape(1, 1, -1, 1), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32
    )
    modulation = model.modulation(timestep, grid.batch)
    tables = model.rope_tables(grid)
    band_tables = tuple(tables.frames(band.pad_lo, band.pad_hi) for band in bands)
    brick = model._stage5_brick(grid) if model._keep_bricked else None
    if brick is not None:
        tt_context = model._brick_activation(tt_context, model._local_volume(grid), brick)
        x_bands = [
            model._brick_activation(band_x, model._local_volume(grid, t=band.hi - band.lo), brick)
            for band_x, band in zip(x_bands, bands)
        ]

    block = model.diff_blocks[0]
    state = {"x": x_bands}

    def step() -> None:
        state["x"] = block(state["x"], tt_context, modulation, grid, bands, band_tables, brick=brick)

    return Bench(
        "stage5",
        step,
        depth=cfg.num_blocks,
        tokens_per_chip=grid.t * grid.h * (grid.w // sp),
        describe=f"dim={cfg.dim} heads={cfg.dim // cfg.head_dim} grid={(grid.t, grid.h, grid.w)} bands={len(bands)}",
    )


def block_sections(calls: int) -> list[tuple[str, float, float]]:
    """``(label, ms per call, share)`` for the last root's direct children, pooled by label.

    A block's own spans would each become a root; run the bench under one
    ``timing_tree.span(mesh, ..., root=True)`` so they pool under it instead. Empty unless
    ``TT_DIT_STAGE_TIMING=1`` recorded anything.
    """
    if not timing_tree.ENABLED or not timing_tree.roots():
        return []
    merged: dict[str, float] = {}
    for child in timing_tree.roots()[-1].children:
        merged[child.label] = merged.get(child.label, 0.0) + child.incl_ms
    total = sum(merged.values()) or 1.0
    return [(label, ms / calls, 100 * ms / total) for label, ms in sorted(merged.items(), key=lambda kv: -kv[1])]


def render_sections(sections: list[tuple[str, float, float]]) -> str:
    lines = [f"{'section':44s} {'ms/block':>10} {'share':>7}"]
    lines += [f"{label:44s} {ms:10.2f} {share:6.1f}%" for label, ms, share in sections]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whole-decode instruments
# ---------------------------------------------------------------------------


def timed_decode(decoder: DiffVAEDecoder, latent: torch.Tensor, mesh, *, output_type: str = "float"):
    """Warm-up decode, then one timed decode; returns the output and the time in ms."""
    decoder.forward(latent, output_type=output_type, seed=0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    out = decoder.forward(latent, output_type=output_type, seed=0)
    ttnn.synchronize_device(mesh)
    return out, (time.perf_counter() - t0) * 1000.0


def upload_latent(decoder: DiffVAEDecoder, latent: torch.Tensor, mesh) -> ttnn.Tensor:
    """The raw latent in the ROW_MAJOR ``(1, C, T, H*W)`` form the device-preproc path reads.

    A trace refuses host-to-device writes during capture, so the upload happens once, outside the
    captured region, and the buffer is handed to ``decode`` / ``forward_context`` as ``latent_tt``.
    """
    b, c, t, h, w = latent.shape
    return ttnn.from_torch(
        latent.reshape(1, c, t, h * w).contiguous(), device=mesh, dtype=decoder.dtype, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def read_chip0(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()


def trace_region(
    decoder: DiffVAEDecoder, region: str, latent: torch.Tensor, raw: ttnn.Tensor
) -> Callable[[], ttnn.Tensor]:
    """The device graph a trace captures, with both host boundaries outside it.

    ``decode`` is the whole thing: ghost pad, conv_in, the four deterministic stages, the crop, and
    stage 5's eight diffusion blocks, with the PCIe pull deferred to the caller by ``device_out``.
    ``det_context`` stops at the W-sharded stage-5 context.
    """
    if region == "decode":
        return lambda: decoder.decode(latent, seed=0, latent_tt=raw, device_out=True)
    if region == "det_context":
        return lambda: decoder.forward_context(latent, gather_output=False, latent_tt=raw)[0]
    raise ValueError(f"unknown trace region {region!r}; expected 'decode' or 'det_context'")


@dataclass
class TraceReport:
    eager_ms: list[float]
    replay_ms: list[float]
    enqueue_ms: float | None
    enqueue_total_ms: float | None
    max_abs_diff: float
    identical: bool


def trace_replay(
    mesh, run: Callable[[], ttnn.Tensor], iters: int, *, probe_dispatch: bool = False, log=print
) -> TraceReport:
    """Capture ``run`` as a ttnn trace after two eager passes, replay it ``iters`` times, and check the
    replay against the last eager output.

    The first eager pass builds program cache, device plans, persistent CCL buffers and the cached
    shard offsets, none of which may be created inside the capture; the second is the honest eager
    number. ``probe_dispatch`` adds an unsynchronised eager call to measure the host enqueue cost
    alone, which is the entire budget a trace can reclaim. ``run`` must do no host-to-device writes.

    ``TT_DIT_STAGE_TIMING`` should be unset: its per-stage timers sync the mesh, and a synchronize
    inside a captured region is both untraceable and a distorted measurement.

    ``run`` must do no host-to-device writes; the capture refuses them with "Writes are not
    supported during trace capture", once per chip. ``ttnn.zeros`` / ``ttnn.full`` with a device
    argument are such writes; ``utils.tensor.full`` is the device-side form.
    """
    eager_ms = []
    eager_ref = None
    for i in range(2):
        t0 = time.perf_counter()
        out = run()
        ttnn.synchronize_device(mesh)
        eager_ms.append((time.perf_counter() - t0) * 1000)
        log(f"[eager {i}] {eager_ms[-1]:8.1f} ms  out={tuple(out.shape)}")
        eager_ref = read_chip0(out)
        ttnn.deallocate(out)

    enqueue = total = None
    if probe_dispatch:
        # ttnn dispatch is asynchronous, so returning from the call without synchronising measures
        # the host enqueue cost alone.
        t0 = time.perf_counter()
        out = run()
        enqueue = (time.perf_counter() - t0) * 1000
        ttnn.synchronize_device(mesh)
        total = (time.perf_counter() - t0) * 1000
        log(
            f"[dispatch] host enqueue {enqueue:8.1f} ms | total {total:8.1f} ms | host share {100 * enqueue / total:5.2f}%"
        )
        ttnn.deallocate(out)

    log("[trace] begin_trace_capture")
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out_t = run()
    except BaseException:
        # An op that cannot be captured (a host write, most often) raises mid-region. Close the
        # region before unwinding: with it left open, close_mesh_device blocks and the mesh has to
        # be reset, and the exception itself is never printed.
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.release_trace(mesh, tid)
        raise
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    log(f"[trace] captured id={tid} out={tuple(out_t.shape)}")

    replay_ms = []
    for i in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        replay_ms.append((time.perf_counter() - t0) * 1000)
        log(f"[replay {i}] {replay_ms[-1]:8.1f} ms")

    got = read_chip0(out_t)
    diff = float((eager_ref - got).abs().max())
    identical = bool(torch.equal(eager_ref, got))
    log(f"[check] max|eager-replay| = {diff:.6f}  identical={identical}")
    ttnn.release_trace(mesh, tid)
    return TraceReport(eager_ms, replay_ms, enqueue, total, diff, identical)


@dataclass
class TracedForwardReport:
    eager_ms: float
    capture_ms: float
    replay_ms: list[float]
    identical: bool


def traced_forward_check(
    decoder: DiffVAEDecoder, latent: torch.Tensor, mesh, *, output_type: str = "float", replays: int = 2, log=print
) -> TracedForwardReport:
    """The decoder's own traced ``forward`` (the pipeline's path once ``_vae_traced`` is set) against its
    eager ``forward`` on the same latent: the first traced call captures, the rest replay, and every
    result must equal the eager pixels exactly. Leaves the decoder untraced with its trace released.
    """
    eager, eager_ms = timed_decode(decoder, latent, mesh, output_type=output_type)
    log(f"[eager] {eager_ms:8.1f} ms  out={tuple(eager.shape)}")

    def same(a, b) -> bool:
        return bool(torch.equal(a, b)) if isinstance(a, torch.Tensor) else bool((a == b).all())

    decoder._vae_traced = True
    try:
        t0 = time.perf_counter()
        got = decoder.forward(latent, output_type=output_type, seed=0)
        ttnn.synchronize_device(mesh)
        capture_ms = (time.perf_counter() - t0) * 1000
        identical = same(eager, got)
        log(f"[capture+run] {capture_ms:8.1f} ms  identical={identical}")
        replay_ms = []
        for i in range(replays):
            t0 = time.perf_counter()
            got = decoder.forward(latent, output_type=output_type, seed=0)
            ttnn.synchronize_device(mesh)
            replay_ms.append((time.perf_counter() - t0) * 1000)
            ok = same(eager, got)
            identical = identical and ok
            log(f"[replay {i}] {replay_ms[-1]:8.1f} ms  identical={ok}")
    finally:
        decoder._vae_traced = False
        decoder.release_trace()
    return TracedForwardReport(eager_ms, capture_ms, replay_ms, identical)


@dataclass
class TraceValidation:
    eager_reproducible: bool
    replay_matches_a: bool
    replay_follows_input: bool
    replay_matches_b: bool
    replay_ms: float


def trace_validate(decoder: DiffVAEDecoder, mesh, latent_a: torch.Tensor, latent_b: torch.Tensor, *, log=print):
    """Prove a captured trace of the deterministic stages re-executes rather than leaving a stale buffer.

    A bit-identical replay is not evidence on its own: the capture's output allocation can reuse
    the address the eager output was just freed from, so a replay that did nothing would still read
    the right numbers. Two checks close that gap: poison the output before replaying, and change the
    input in place after capture and require the output to follow.
    """
    raw = upload_latent(decoder, latent_a, mesh)
    raw_b = upload_latent(decoder, latent_b, mesh)

    def run(lat):
        ctx, _ = decoder.forward_context(lat, gather_output=False, latent_tt=raw)
        return ctx

    out = run(latent_a)
    ref_a = read_chip0(out)
    ttnn.deallocate(out)
    out = run(latent_a)
    ref_a2 = read_chip0(out)
    ttnn.deallocate(out)
    reproducible = bool(torch.equal(ref_a, ref_a2))
    log(f"[eager] A reproducible: {reproducible}")

    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out_t = run(latent_a)
    except BaseException:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.release_trace(mesh, tid)
        raise
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    log(f"[trace] captured {tid}")

    # CHECK 1: poison the output, then replay. A no-op replay leaves the poison behind.
    ttnn.copy(ttnn.zeros_like(out_t), out_t)
    log(f"[check1] output poisoned to zeros: max|.|={float(read_chip0(out_t).abs().max()):.6f}")
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    got_a = read_chip0(out_t)
    matches_a = bool(torch.equal(got_a, ref_a))
    log(f"[check1] after replay == eager A: {matches_a}  max|diff|={float((got_a - ref_a).abs().max()):.6f}")

    # CHECK 2: change the input in place. Replay must follow it.
    ttnn.copy(raw_b, raw)
    t0 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    replay_ms = (time.perf_counter() - t0) * 1000
    got_b = read_chip0(out_t)
    out = run(latent_b)
    ref_b = read_chip0(out)
    ttnn.deallocate(out)
    follows = not torch.equal(got_b, got_a)
    matches_b = bool(torch.equal(got_b, ref_b))
    log(f"[check2] replay on new input differs from A: {follows}")
    log(f"[check2] replay on new input == eager B  : {matches_b}  max|diff|={float((got_b - ref_b).abs().max()):.6f}")
    log(f"[check2] replay {replay_ms:8.1f} ms")
    ttnn.release_trace(mesh, tid)
    return TraceValidation(reproducible, matches_a, follows, matches_b, replay_ms)


# ---------------------------------------------------------------------------
# Process plumbing for the CLI (pytest gets these from its fixtures)
# ---------------------------------------------------------------------------

#: Enough for the whole 1080p decode graph.
TRACE_REGION_SIZE = 200_000_000


@contextlib.contextmanager
def open_mesh(shape=(4, 8), *, fabric: ttnn.FabricConfig | None = None, trace_region_size: int | None = None):
    """Open a mesh the way the ``mesh_device`` fixture does: fabric first, the same default device
    params, fabric disabled again on close."""
    from tests.scripts.common import get_updated_device_params

    params = {}
    if trace_region_size is not None:
        params["trace_region_size"] = trace_region_size
    params = get_updated_device_params(params)
    if fabric is not None:
        ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@contextlib.contextmanager
def heartbeat(interval: float = 20.0) -> Iterator[None]:
    """Keep a no-output watchdog fed; checkpoint loads and trace captures are silent for minutes."""
    stop = threading.Event()

    def beat() -> None:
        n = 0
        while not stop.wait(interval):
            n += interval
            print(f"[heartbeat] {n:.0f}s elapsed, still working", flush=True)

    threading.Thread(target=beat, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
