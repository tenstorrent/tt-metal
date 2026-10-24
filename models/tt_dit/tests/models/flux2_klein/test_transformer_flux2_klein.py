# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The full FLUX.2 Klein DiT, one denoising step on the mesh vs diffusers' ``Flux2Transformer2DModel`` on CPU with
the real weights (``tests/models/flux3_klein/test_transformer_flux3_klein.py`` pattern).

Oracle: ``Flux2Transformer2DModel.forward`` (``transformer_flux2.py:1180-1381``) in bf16 under
``torch.inference_mode()`` on CPU, called exactly as ``Flux2KleinPipeline.__call__`` calls it (``pipeline_flux2_klein.py
:841-862``): ``hidden_states = [latents | reference tokens]`` bf16, ``timestep = bf16(t) / 1000``, ``guidance=None``,
``encoder_hidden_states`` the 512 prompt rows, ``img_ids`` / ``txt_ids`` int64, the output sliced to the latent rows.
The device side runs ``Flux2KleinTransformer``: ``sequence_layout`` / ``prepare_plan`` / ``embed_prompt`` once per
request, one ``forward`` per sample. Random inputs (seeded) stand in for the encoders: the DiT is gated in isolation,
the encoders have their own rows.

Rows (the (4, 8) ring with 8 KB packets; ``layout`` sp8tp4 = SP8 axis 1 x TP4 axis 0 (the default) and sp4tp8;
``size`` 4b / 9b, a missing snapshot skips):
  T1 test_distilled_real[size, layout]  the DISTILLED checkpoint at full depth: (a) T2I 1024: image 4096 + text 512,
                          no reference; (b) 1-ref: image 4096 + reference 4096 (``reference_ids(64, 64, 0)``) + text
                          512; both at step 0 of the 4-step schedule. A second forward on the same plan must reproduce
                          the first bit for bit (the step loop).
  T2 test_base_real[size, layout]  the BASE checkpoint: T2I 1024 at step 0 of the 50-step schedule and the CFG pair
                          (the same latents with a second, "empty-prompt" text) compared on velocity_uncond,
                          velocity_cond and ``cfg_velocity(u, c, 4.0)`` (guidance amplifies the per-forward deviation).

Gate on the ``[S_img, 128]`` velocity: ``assert_quality(pcc >= VELOCITY_PCC, relative_rmse <= VELOCITY_RMSE)``, the
0.995 floor of the FLUX 3 Klein port's goldens (its per-forward floor measured 0.9998-0.99997 on the ring); applied
here unmeasured and never loosened.

Weights: the real rows build ONE torch ``Flux2Transformer2DModel`` per snapshot per session (``common.reference_
transformer``), lazily: only when the device cache (``utils.cache.load_model``, ``TT_DIT_CACHE_DIR``) or the oracle
cache misses. ORACLE CACHE: ``$FLUX2_KLEIN_ORACLE_CACHE``; key = the DiT blob, variant, size, case, seed, step, the
sha256 of every input tensor and ``common.reference_code_sha`` (the installed diffusers file + versions). A full-depth
4B oracle on 4608 rows takes ~1-3 min on a node CPU; the 1-ref row (8704 rows) longer. Never run two at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.flux2_klein.config_flux2_klein import Flux2KleinConfig
from ....models.transformers.flux2_klein.transformer_flux2_klein import Flux2KleinPlan, Flux2KleinTransformer
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux2_klein.packing import (
    FLUX2_KLEIN_IN_CHANNELS,
    FLUX2_KLEIN_TEXT_LENGTH,
    latent_ids,
    pad_rows,
    reference_ids,
    text_ids,
)
from ....pipelines.flux2_klein.scheduler import (
    FLUX2_KLEIN_GUIDANCE,
    FLUX2_KLEIN_STEPS,
    Flux2KleinScheduler,
    Flux2KleinStep,
)
from ....pipelines.flux2_klein.weights_dit_flux2_klein import load_flux2_klein_transformer_weights
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    BF16,
    DEFAULT_LAYOUT,
    LAYOUTS,
    MESH_SHAPE,
    SIZES,
    cached_oracle,
    dit_blob,
    measure,
    reference_code_sha,
    reference_transformer,
    require_checkpoint_dir,
    sha256_of,
)
from .device_params import ring_params_8k_flux2_klein_transformer

# Velocity gate (never loosened; see the module docstring).
VELOCITY_PCC = 0.995
VELOCITY_RMSE = 0.10
GRID = (64, 64)  # 1024 x 1024: 4096 latent rows


# --- host inputs ------------------------------------------------------------------------------------------------------


@dataclass
class Request:
    """One sample's host inputs in the pipeline's dtypes: image fp32 (bf16-valued), text and cond bf16, ids int64."""

    image: tuple[torch.Tensor, torch.Tensor]
    text: tuple[torch.Tensor, torch.Tensor]
    image_cond: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def image_rows(self) -> int:
        return int(self.image[0].shape[0])

    @property
    def cond_rows(self) -> int:
        return 0 if self.image_cond is None else int(self.image_cond[0].shape[0])

    @property
    def tag(self) -> str:
        return f"img{self.image_rows}+cond{self.cond_rows}+txt{self.text[0].shape[0]}"

    def spatial_rows(self) -> torch.Tensor:
        """``[S_img + S_cond, 128]`` fp32: the latents followed by the reference tokens (``P:846-848``)."""
        if self.image_cond is None:
            return self.image[0]
        return torch.cat([self.image[0], self.image_cond[0].float()])

    def spatial_ids(self) -> torch.Tensor:
        if self.image_cond is None:
            return self.image[1]
        return torch.cat([self.image[1], self.image_cond[1]])

    def inputs_sha(self) -> str:
        parts = [sha256_of(t) for t in (*self.image, *self.text)]
        if self.image_cond is not None:
            parts += [sha256_of(t) for t in self.image_cond]
        return "".join(parts)


def random_image(seed: int, grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    return torch.randn(h * w, FLUX2_KLEIN_IN_CHANNELS, generator=g).to(BF16).float(), latent_ids(h, w)


def random_text(seed: int, joint_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(FLUX2_KLEIN_TEXT_LENGTH, joint_dim, generator=g).to(BF16), text_ids(FLUX2_KLEIN_TEXT_LENGTH)


def random_cond(seed: int, grid: tuple[int, int], index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    return torch.randn(h * w, FLUX2_KLEIN_IN_CHANNELS, generator=g).to(BF16), reference_ids(h, w, index)


# --- reference --------------------------------------------------------------------------------------------------------


def reference_velocity(snapshot: Path, request: Request, step: Flux2KleinStep) -> torch.Tensor:
    """``Flux2Transformer2DModel.forward`` exactly as the pipeline calls it at ``step``; fp32 ``[S_img, 128]``."""
    dit = reference_transformer(snapshot)
    timestep = torch.tensor([step.timestep], dtype=torch.float32).to(BF16) / 1000  # P:841, P:853
    t0 = time.time()
    with torch.inference_mode():
        out = dit(
            hidden_states=request.spatial_rows().to(BF16)[None],
            encoder_hidden_states=request.text[0][None],
            timestep=timestep,
            guidance=None,
            img_ids=request.spatial_ids(),
            txt_ids=request.text[1],
            return_dict=False,
        )[0]
    velocity = out[0, : request.image_rows].float().clone()  # P:862
    logger.info(f"reference forward {request.tag} step {step.index} (t={step.timestep:.4f}) in {time.time() - t0:.1f}s")
    return velocity


def cached_velocity(
    snapshot: Path, variant: str, size: str, case: str, seed: int, request: Request, step: Flux2KleinStep
):
    fields = {
        "blob": dit_blob(snapshot),
        "variant": variant,
        "size": size,
        "case": case,
        "seed": seed,
        "step_index": step.index,
        "timestep": repr(step.timestep),
        "inputs": request.inputs_sha(),
        "ref_sha": reference_code_sha(),
    }
    return cached_oracle(
        fields,
        f"{variant} {size} {case} seed {seed} step {step.index}",
        lambda: {"velocity": reference_velocity(snapshot, request, step)},
    )["velocity"]


# --- device -----------------------------------------------------------------------------------------------------------

mesh_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[MESH_SHAPE, ring_params_8k_flux2_klein_transformer]],
    ids=["bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)
layout_rows = pytest.mark.parametrize("layout", [DEFAULT_LAYOUT, *(n for n in LAYOUTS if n != DEFAULT_LAYOUT)])
size_rows = pytest.mark.parametrize("size", list(SIZES))


@dataclass
class Mesh:
    device: ttnn.MeshDevice
    sp_axis: int
    tp_axis: int
    ccl: CCLManager
    parallel_config: DiTGParallelConfigNoCFG
    tag: str

    @property
    def sp(self) -> int:
        return self.parallel_config.sequence_parallel.factor


def open_mesh(mesh_device, layout: str) -> Mesh:
    sp_axis, tp_axis = LAYOUTS[layout]
    shape = tuple(mesh_device.shape)
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=shape[sp_axis], mesh_axis=sp_axis),
    )
    tag = f"{shape[0]}x{shape[1]}_{layout}_ring"
    logger.info(f"mesh {tag}: SP {shape[sp_axis]} over axis {sp_axis}, TP {shape[tp_axis]} over axis {tp_axis}")
    return Mesh(mesh_device, sp_axis, tp_axis, ccl, parallel_config, tag)


def dram_per_bank(device) -> str:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip"
    )


def device_transformer(mesh: Mesh, variant: str, size: str, snapshot: Path) -> Flux2KleinTransformer:
    config = Flux2KleinConfig.from_snapshot(snapshot)
    assert config.size == size, f"{snapshot} holds a {config.size} transformer, requested {size}"
    model = Flux2KleinTransformer(
        config, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config, shard_prompt=True
    )
    t0 = time.time()
    state = load_flux2_klein_transformer_weights(
        model, variant, size, snapshot, parallel_config=mesh.parallel_config, mesh_device=mesh.device
    )
    logger.info(
        f"RESULT weights[{variant}/{size}] {mesh.tag}: cache {state} in {time.time() - t0:.1f}s; "
        f"{dram_per_bank(mesh.device)}"
    )
    return model


@dataclass
class DeviceRequest:
    """A request uploaded once: the padded spatial rows, the hoisted prompt embedding and the plan."""

    request: Request
    plan: Flux2KleinPlan
    spatial: ttnn.Tensor
    prompt_embedded: ttnn.Tensor

    def deallocate(self) -> None:
        ttnn.deallocate(self.spatial)
        ttnn.deallocate(self.prompt_embedded)
        self.plan.deallocate()


def upload_request(mesh: Mesh, model: Flux2KleinTransformer, request: Request) -> DeviceRequest:
    """The transformer's tensor contract: spatial ``[1, L_pad, 128]`` bf16 SP-sharded with zero pad rows, the prompt
    ``[1, 512, joint]`` bf16 SP-sharded under ``shard_prompt`` -> ``embed_prompt``, the plan of the layout."""
    t0 = time.time()
    layout = model.sequence_layout(request.image_rows, request.cond_rows)
    plan = model.prepare_plan(layout, request.spatial_ids(), request.text[1])
    rows = pad_rows(request.spatial_rows().to(BF16), layout.padded)[0]
    spatial = tensor.from_torch(
        rows[None].contiguous(), device=mesh.device, dtype=ttnn.bfloat16, mesh_axes=[None, mesh.sp_axis, None]
    )
    prompt_axis = mesh.sp_axis if model.shard_prompt else None
    tokens = tensor.from_torch(
        request.text[0][None].contiguous(), device=mesh.device, dtype=ttnn.bfloat16, mesh_axes=[None, prompt_axis, None]
    )
    prompt_embedded = model.embed_prompt(tokens)
    ttnn.deallocate(tokens)
    ttnn.synchronize_device(mesh.device)
    logger.info(
        f"request {request.tag} {mesh.tag}: L_real={layout.real} L_pad={layout.padded} rows/device={layout.rows}; "
        f"plan + hoisted prompt in {time.time() - t0:.2f}s"
    )
    return DeviceRequest(request, plan, spatial, prompt_embedded)


def device_velocity(mesh: Mesh, model: Flux2KleinTransformer, dev: DeviceRequest, step: Flux2KleinStep) -> torch.Tensor:
    """One device forward at ``step``; fp32 ``[S_img, 128]``."""
    t_embed = tensor.from_torch(
        torch.full((1, 1), step.embedding_t, dtype=torch.float32),
        device=mesh.device,
        dtype=ttnn.float32,
        mesh_axes=[None, None],
    )
    ttnn.synchronize_device(mesh.device)
    t0 = time.time()
    out = model.forward(spatial=dev.spatial, prompt_embedded=dev.prompt_embedded, t_embed=t_embed, plan=dev.plan)
    ttnn.synchronize_device(mesh.device)
    wall = time.time() - t0
    assert tuple(out.shape) == (1, dev.spatial.shape[1], FLUX2_KLEIN_IN_CHANNELS), tuple(out.shape)
    velocity = tensor.to_torch(out, mesh_axes=[None, mesh.sp_axis, None])[0, : dev.request.image_rows].float()
    ttnn.deallocate(out)
    ttnn.deallocate(t_embed)
    logger.info(
        f"device forward {dev.request.tag} {mesh.tag} step {step.index} (embedding_t={step.embedding_t}) in "
        f"{wall:.2f}s host wall"
    )
    assert torch.isfinite(velocity).all(), "non-finite velocity"
    return velocity


def check(name: str, reference: torch.Tensor, got: torch.Tensor) -> None:
    p, r, m = measure(reference, got)
    logger.info(f"RESULT {name}: pcc={p:.6f} rel_rmse={r:.5f} max_abs={m:.4g} (gate {VELOCITY_PCC} / {VELOCITY_RMSE})")
    assert_quality(reference, got, pcc=VELOCITY_PCC, relative_rmse=VELOCITY_RMSE)


# --- T1: distilled, T2I and one reference -----------------------------------------------------------------------------


@mesh_rows
@layout_rows
@size_rows
def test_distilled_real(mesh_device, layout: str, size: str) -> None:
    snapshot = require_checkpoint_dir("distilled", size)
    mesh = open_mesh(mesh_device, layout)
    model = device_transformer(mesh, "distilled", size, snapshot)
    joint_dim = model.config.joint_dim
    scheduler = Flux2KleinScheduler(FLUX2_KLEIN_STEPS["distilled"], GRID[0] * GRID[1])
    step = scheduler.step(0)

    seed = 200
    image = random_image(seed, GRID)
    text = random_text(seed + 1, joint_dim)
    cond = random_cond(seed + 2, GRID, 0)
    for case, request in (("t2i", Request(image, text)), ("ref1", Request(image, text, cond))):
        reference = cached_velocity(snapshot, "distilled", size, case, seed, request, step)
        dev = upload_request(mesh, model, request)
        got = device_velocity(mesh, model, dev, step)
        check(f"distilled_real[{case}] {size} {mesh.tag} {request.tag}", reference, got)
        if case == "t2i":
            again = device_velocity(mesh, model, dev, step)
            assert torch.equal(got, again), "the second forward of the same step differs from the first"
            logger.info(f"RESULT distilled_real[t2i] {size} {mesh.tag}: second forward bit-equal to the first")
        dev.deallocate()
    model.deallocate_weights()
    ttnn.synchronize_device(mesh.device)


# --- T2: base, T2I and the CFG pair -----------------------------------------------------------------------------------


@mesh_rows
@layout_rows
@size_rows
def test_base_real(mesh_device, layout: str, size: str) -> None:
    snapshot = require_checkpoint_dir("base", size)
    mesh = open_mesh(mesh_device, layout)
    model = device_transformer(mesh, "base", size, snapshot)
    joint_dim = model.config.joint_dim
    scheduler = Flux2KleinScheduler(FLUX2_KLEIN_STEPS["base"], GRID[0] * GRID[1])
    step = scheduler.step(0)
    guidance = FLUX2_KLEIN_GUIDANCE["base"]

    seed = 300
    image = random_image(seed, GRID)
    cond_request = Request(image, random_text(seed + 1, joint_dim))
    uncond_request = Request(image, random_text(seed + 2, joint_dim))
    reference_c = cached_velocity(snapshot, "base", size, "t2i", seed, cond_request, step)
    reference_u = cached_velocity(snapshot, "base", size, "t2i_uncond", seed, uncond_request, step)
    dev_c = upload_request(mesh, model, cond_request)
    dev_u = upload_request(mesh, model, uncond_request)
    got_c = device_velocity(mesh, model, dev_c, step)
    got_u = device_velocity(mesh, model, dev_u, step)
    name = f"base_real {size} {mesh.tag} {cond_request.tag}"
    check(f"{name}: velocity_cond", reference_c, got_c)
    check(f"{name}: velocity_uncond", reference_u, got_u)
    check(
        f"{name}: cfg_velocity(u, c, {guidance})",
        Flux2KleinScheduler.cfg_velocity(reference_u.to(BF16), reference_c.to(BF16), guidance).float(),
        Flux2KleinScheduler.cfg_velocity(got_u.to(BF16), got_c.to(BF16), guidance).float(),
    )
    dev_c.deallocate()
    dev_u.deallocate()
    model.deallocate_weights()
    ttnn.synchronize_device(mesh.device)
