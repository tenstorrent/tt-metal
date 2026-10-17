# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M3: the full FLUX 3 Image Klein DiT, one denoising step on device vs the vendored torch ``DiT``.

Oracle: ``reference/flux3_klein/dit.py::DiT.forward`` in bf16 under ``torch.inference_mode()`` +
``torch.autocast("cpu", dtype=torch.bfloat16)`` (SDPBackend.MATH), exactly as ``sampling.denoise`` runs it, on a
``packing.pack``-built ``Flux3KleinPacked`` converted to the reference ``Packed`` with ``timesteps["image"].fill_(t)``
(the ``_velocity`` step of ``test_reference_flux3_klein.py``). The device side runs ``Flux3KleinTransformer``:
``embed_text`` / ``embed_image_cond`` once per request, one ``forward`` per sample and step. Mesh handling, the
``check`` wrapper (logs ``RESULT ...: pcc= rel_rmse= max_abs=``, then ``assert_quality``) and the mesh rows are the
M1 test's (see its module docstring for why the (2,2) mesh is a submesh of the (4,8) parent).

Rows (each on the (2,2) Linear submesh for iteration and the (4,8) Ring with 8 KB packets as the gate):
  T1 test_fast_random[t2i|ref80] x [base|flow_map]   random DiT (N(0, 0.02) linears, QK scales U(0.5, 1.5)) at
                          DEPTH 1 / DEPTH_SINGLE 2 (``common.patch_reference_depth``); image 32x32 = 1024, text 80,
                          cond 8x10 = 80 or none; base: ``Flux3KleinScheduler(50).step(0)``, flow map:
                          ``Flux3KleinScheduler(16).step(0)`` with ``timesteps_ref = schedule[1]``.
  T2 test_base_real       the BASE checkpoint at full depth (5 / 28): (a) T2I 1024: image 4096 + text 320, no cond;
                          (b) 1-ref: image 4096 + cond 4096 + text 320; (c) the CFG step on T2I: ONE reference call on
                          the two-sample pack [unconditional (text 80), conditional (text 320)] vs two device forwards,
                          compared on velocity_uncond, velocity_cond and ``cfg_velocity(u, c, 3.5)``.
  T3 test_distilled_real_flow_map   the DISTILLED checkpoint, ``flow_map=True`` (359 keys strict), T2I 4096 + 320 at
                          ``Flux3KleinScheduler(16).step(0)`` with ``timesteps_ref = schedule[1]``; liveness: the same
                          forward with ``t_ref_embed`` of step 8 must differ from the step-0 oracle (pcc < 0.999).
  T4 test_strict_load_negative   base state into ``flow_map=True`` (missing ``modulation.ref_time_in.*``), distilled
                          state into ``flow_map=False`` (unexpected), one dropped single-block key, one extra key, and
                          the positive control (the exact parameter set); plus the real checkpoints' key sets vs the
                          reference ``DiT.state_dict()`` (357 / 359 keys) when the checkpoints are present.

Gate on the ``[L_image, 128]`` velocity: ``assert_quality(pcc >= 0.996, relative_rmse <= 0.09)`` (``VELOCITY_PCC`` /
``VELOCITY_RMSE`` below, with the measured floors). Never loosened.

Weights: the real rows build ONE torch ``DiT`` per variant per test session (meta-device ``DiT`` +
``ref_checkpoint.load_checkpoint``, 11 GB bf16), lazily: only when the device cache (``utils.cache.load_model``,
``TT_DIT_CACHE_DIR``) or the oracle cache misses. Never run two full-depth CPU oracles at once (one takes 1-3 min and
~12 GB on the node; the two-sample CFG pack 3-5 min).

ORACLE CACHE: ``FLUX3_KLEIN_ORACLE_CACHE`` names a directory (unset -> always recompute; the test passes with an empty
directory). Key = sha256 of
``"{blob}|{variant}|{case}|{seed}|{depth}|{depth_single}|{flow_map}|{step_index}|{inputs}|{ref_sha}"`` with
``blob`` = the HuggingFace blob name of ``model.safetensors`` (``checkpoint_blob``: the basename of the snapshot
symlink's own target, which in a hub cache is the file's sha256; NOT the fully resolved path, which a shared blob
store may rename), ``inputs`` = sha256 over the request's ``(tokens, ids)`` bytes per mode in ``FLUX3_KLEIN_MODES``
order for every sample plus the step's ``(t_model, t_next_model)`` (``inputs_sha``: the oracle's actual inputs, so a
change to the request generators or the schedule can never serve a stale velocity) and ``ref_sha`` = sha256 of the
bytes of ``reference/flux3_klein/dit.py`` + ``pipelines/flux3_klein/packing.py`` + ``pipelines/flux3_klein/scheduler.py``.
Files: ``<digest>.safetensors`` (the velocity tensors of the case) + ``<digest>.json`` (the key fields). A miss computes,
writes and logs ``ORACLE MISS <digest>``; a hit logs ``ORACLE HIT <digest>``. Random rows are never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.flux3_klein.modulation_flux3_klein import timestep_inputs, timesteps_to_device
from ....models.transformers.flux3_klein.transformer_flux3_klein import (
    FLUX3_KLEIN_DEPTH,
    FLUX3_KLEIN_DEPTH_SINGLE,
    Flux3KleinSequencePlan,
    Flux3KleinTransformer,
)
from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.packing import (
    FLUX3_KLEIN_MODES,
    Flux3KleinSample,
    image_ids,
    padded_length,
    reference_ids,
    text_ids,
)
from ....pipelines.flux3_klein.scheduler import (
    FLUX3_KLEIN_GUIDANCE,
    FLUX3_KLEIN_STEPS,
    Flux3KleinScheduler,
    Flux3KleinStep,
)
from ....reference.flux3_klein import checkpoint as ref_checkpoint
from ....reference.flux3_klein import dit as ref_dit
from ....utils import cache, tensor
from .common import EMPTY_PROMPT_TEXT_LENGTH, patch_reference_depth, sha256_of, variant_checkpoint_dir
from .test_transformer_block_flux3_klein import Mesh, check, mesh_rows, open_mesh

BF16 = torch.bfloat16

# Velocity gate (PLAN.md M3). Measured floors (pcc / relative rmse, ledger 2026-10-17 08:49 and 08:58 rows):
#   real weights, full depth 5 / 28: 0.99975-0.99997 / 0.0080-0.0225 over the (2,2) submesh and the (4,8) ring. The
#   lowest-margin row is the CFG-combined velocity cfg_velocity(u, c, 3.5) (0.999767 / 0.0219 on the (2,2) submesh,
#   0.999750 / 0.0225 on the ring): guidance 3.5 amplifies the device-vs-reference deviation of the two forwards,
#   each of which sits at 0.99996-0.99997 / 0.0080-0.0096. Next come the 1-ref row (ring 0.999818 / 0.0192, submesh
#   0.999914 / 0.0132) and the distilled flow map (ring 0.999822 / 0.0194, submesh 0.999878 / 0.0159); T2I is
#   0.99997 / 0.0080-0.0081 on both meshes. A 0.9998 on a real row is this floor, not a regression.
#   random weights, DEPTH 1 / 2 (T1): 0.99988-0.99992 / 0.0129-0.0159; the DEPTH 1 / 1 prototype measured
#   0.99990-0.99994 / 0.012-0.014 on both meshes (design probe B).
# Bug calibration from M1 (block level, host sweep): a modulation chunk order shifted by one head or a scale without
# the +1 measures 0.970 / 0.243, swiglu halves swapped 0.970 / 0.243. Thresholds: never loosened.
VELOCITY_PCC = 0.996
VELOCITY_RMSE = 0.09
# T3 liveness: a flow-map forward with another step's t_next must land this far from the step-0 oracle.
LIVENESS_PCC_MAX = 0.999

ORACLE_CACHE_ENV = "FLUX3_KLEIN_ORACLE_CACHE"
REFERENCE_FILES = (
    Path(__file__).resolve().parents[3] / "reference" / "flux3_klein" / "dit.py",
    Path(__file__).resolve().parents[3] / "pipelines" / "flux3_klein" / "packing.py",
    Path(__file__).resolve().parents[3] / "pipelines" / "flux3_klein" / "scheduler.py",
)
FAST_DEPTH = (1, 2)
FULL_DEPTH = (FLUX3_KLEIN_DEPTH, FLUX3_KLEIN_DEPTH_SINGLE)

# One torch DiT per variant per session (11 GB each), built lazily by `torch_dit`.
_TORCH_DITS: dict[str, ref_dit.DiT] = {}


# --- host inputs ------------------------------------------------------------------------------------------------------


@dataclass
class Request:
    """One sample's host inputs: ``(tokens, ids)`` per mode, the pipeline's dtypes (image fp32 bf16-valued, text and
    cond bf16; image / cond ids fp32, text ids int64)."""

    image: tuple[torch.Tensor, torch.Tensor]
    text: tuple[torch.Tensor, torch.Tensor]
    image_cond: tuple[torch.Tensor, torch.Tensor] | None = None

    def sample(self) -> Flux3KleinSample:
        return Flux3KleinSample(text=self.text, image=self.image, image_cond=self.image_cond)

    @property
    def lengths(self) -> dict[str, int]:
        return {
            "image": self.image[0].shape[0],
            "image_cond": 0 if self.image_cond is None else self.image_cond[0].shape[0],
            "text": self.text[0].shape[0],
        }

    @property
    def ids(self) -> dict[str, torch.Tensor]:
        ids = {"image": self.image[1], "text": self.text[1]}
        if self.image_cond is not None:
            ids["image_cond"] = self.image_cond[1]
        return ids

    @property
    def tag(self) -> str:
        n = self.lengths
        return f"img{n['image']}+cond{n['image_cond']}+txt{n['text']}"


def random_image(seed: int, grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    return torch.randn(h * w, 128, generator=g).to(BF16).float(), image_ids(h, w)


def random_text(seed: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(length, 20480, generator=g).to(BF16), text_ids(length)


def random_cond(seed: int, grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    return torch.randn(h * w, 128, generator=g).to(BF16), reference_ids(h, w, 0)


# --- reference ----------------------------------------------------------------------------------------------------------


def random_dit(flow_map: bool, seed: int) -> ref_dit.DiT:
    """Seeded random reference DiT at the CURRENT module depth: linears N(0, 0.02), QK-norm scales U(0.5, 1.5)."""
    torch.manual_seed(seed)
    dit = ref_dit.DiT(flow_map)
    with torch.no_grad():
        for name, p in dit.named_parameters():
            if name.endswith("scale"):
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0.0, 0.02)
    return dit.to(BF16).eval()


def torch_dit(variant: str, snapshot: Path) -> ref_dit.DiT:
    """The real ``DiT`` of ``variant`` (full depth, bf16, eval), loaded once per session on first use."""
    if variant not in _TORCH_DITS:
        assert (ref_dit.DEPTH, ref_dit.DEPTH_SINGLE) == FULL_DEPTH, "the real model needs the unpatched depth"
        flow_map = variant == "distilled"
        assert ref_checkpoint.is_flow_map(str(snapshot)) == flow_map, snapshot
        t0 = time.time()
        with torch.device("meta"):
            dit = ref_dit.DiT(flow_map)
        _TORCH_DITS[variant] = ref_checkpoint.load_checkpoint(dit, str(snapshot), "cpu", BF16)
        logger.info(
            f"torch DiT[{variant}] loaded from {snapshot.name} ({len(_TORCH_DITS[variant].state_dict())} keys) "
            f"in {time.time() - t0:.1f}s"
        )
    return _TORCH_DITS[variant]


def reference_packed(packed: packing.Flux3KleinPacked, t: float) -> ref_dit.Packed:
    """The reference ``Packed`` of a ``packing.pack`` result with the image rows' timesteps filled like the sampler."""
    timesteps = {m: packed.timesteps[m].clone() for m in FLUX3_KLEIN_MODES}
    timesteps["image"].fill_(t)
    return ref_dit.Packed(
        data=dict(packed.data), ids=dict(packed.ids), timesteps=timesteps, mode_indices=dict(packed.mode_indices)
    )


def reference_velocity(
    dit: ref_dit.DiT, samples: list[Flux3KleinSample], scheduler: Flux3KleinScheduler, step_index: int
) -> torch.Tensor:
    """``DiT.forward`` on the pack of ``samples`` at ``scheduler.step(step_index)`` exactly as ``sampling.denoise`` runs
    it; fp32 copy of the bf16 velocity in ``packed.data["image"]`` order (all samples concatenated)."""
    step = scheduler.step(step_index)
    ours = packing.pack(samples)
    packed = reference_packed(ours, step.t)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=BF16):
        target = scheduler.schedule[step_index + 1].reshape(1) if dit.flow_map else None
        out = dit(packed, timesteps_ref=target).float().clone()
    logger.info(
        f"reference DiT.forward flow_map={dit.flow_map} samples={len(samples)} rows={ours.sequence_length} "
        f"step {step_index} (t_model={step.t_model}) in {time.time() - t0:.1f}s"
    )
    return out


# --- oracle cache -------------------------------------------------------------------------------------------------------


def reference_sha() -> str:
    """sha256 of the bytes of ``REFERENCE_FILES`` (the oracle's code)."""
    h = hashlib.sha256()
    for path in REFERENCE_FILES:
        h.update(path.read_bytes())
    return h.hexdigest()


def checkpoint_blob(snapshot: Path) -> str:
    """The HuggingFace blob name of ``snapshot / model.safetensors``: the basename of the symlink's OWN target (in a hub
    cache the file's sha256), not of the fully resolved path, which a shared blob store may rename; for a plain file
    (no symlink) the name plus its byte size."""
    path = snapshot / "model.safetensors"
    if path.is_symlink():
        return Path(os.readlink(path)).name
    return f"{path.name}:{path.stat().st_size}"


def inputs_sha(samples: list[Flux3KleinSample], step: Flux3KleinStep) -> str:
    """sha256 over the oracle's inputs: every sample's ``(tokens, ids)`` bytes per mode in ``FLUX3_KLEIN_MODES`` order
    (``"-"`` for an absent mode) and the step's ``(t_model, t_next_model)``."""
    h = hashlib.sha256()
    for sample in samples:
        for mode in FLUX3_KLEIN_MODES:
            pair = getattr(sample, mode)
            h.update(b"-" if pair is None else f"{sha256_of(pair[0])}{sha256_of(pair[1])}".encode())
    h.update(f"{step.t_model!r}|{step.t_next_model!r}".encode())
    return h.hexdigest()


def oracle_cache_dir() -> Path | None:
    value = os.environ.get(ORACLE_CACHE_ENV)
    return Path(value) if value else None


def oracle_key(
    snapshot: Path,
    variant: str,
    case: str,
    seed: int,
    depth: tuple[int, int],
    flow_map: bool,
    step_index: int,
    inputs: str,
) -> tuple[str, dict]:
    """``(digest, fields)`` of one oracle entry (see the module docstring's ORACLE CACHE)."""
    fields = {
        "blob": checkpoint_blob(snapshot),
        "variant": variant,
        "case": case,
        "seed": seed,
        "depth": depth[0],
        "depth_single": depth[1],
        "flow_map": flow_map,
        "step_index": step_index,
        "inputs": inputs,
        "ref_sha": reference_sha(),
    }
    key = "|".join(str(fields[k]) for k in fields)
    return hashlib.sha256(key.encode()).hexdigest(), fields


def cached_oracle(
    *,
    snapshot: Path,
    variant: str,
    case: str,
    seed: int,
    step_index: int,
    samples: list[Flux3KleinSample],
    step: Flux3KleinStep,
    compute,
) -> dict[str, torch.Tensor]:
    """``compute()`` -> ``{name: tensor}`` through the oracle cache (see the module docstring); ``samples`` and ``step``
    are the inputs ``compute`` will feed the reference (keyed by ``inputs_sha``)."""
    from safetensors.torch import load_file, save_file

    flow_map = variant == "distilled"
    digest, fields = oracle_key(
        snapshot, variant, case, seed, FULL_DEPTH, flow_map, step_index, inputs_sha(samples, step)
    )
    root = oracle_cache_dir()
    if root is not None and (root / f"{digest}.safetensors").is_file():
        logger.info(f"ORACLE HIT {digest} ({variant} {case} seed {seed} step {step_index})")
        return load_file(str(root / f"{digest}.safetensors"))
    logger.info(f"ORACLE MISS {digest} ({variant} {case} seed {seed} step {step_index})")
    tensors = {k: v.contiguous() for k, v in compute().items()}
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(root / f"{digest}.safetensors"))
        (root / f"{digest}.json").write_text(json.dumps(fields, indent=1))
        logger.info(f"oracle written to {root / digest}.safetensors")
    return tensors


# --- device -------------------------------------------------------------------------------------------------------------


def device_transformer(mesh: Mesh, *, flow_map: bool, depth: tuple[int, int]) -> Flux3KleinTransformer:
    return Flux3KleinTransformer(
        flow_map=flow_map,
        depth=depth[0],
        depth_single=depth[1],
        device=mesh.device,
        ccl_manager=mesh.ccl,
        parallel_config=mesh.parallel_config,
    )


def load_real_weights(mesh: Mesh, model: Flux3KleinTransformer, variant: str, snapshot: Path) -> None:
    """``utils.cache.load_model``: the device cache under ``TT_DIT_CACHE_DIR`` when complete, else the torch state
    (strict load, 357 / 359 reference keys) and a cache write. Logs HIT / MISS and the wall time."""
    model_name = f"flux3_klein_{variant}_{snapshot.name}"
    subfolder = f"transformer_D{model.depth}_S{model.depth_single}{'_flowmap' if model.flow_map else ''}"
    cache_dir = (
        cache.model_cache_dir(
            model_name=model_name,
            subfolder=subfolder,
            parallel_config=mesh.parallel_config,
            mesh_shape=tuple(mesh.device.shape),
            mesh_device=mesh.device,
            required=False,
        )
        if cache.cache_dir_is_set()
        else None
    )
    complete = cache_dir is not None and (cache_dir / cache.CACHE_DICT_FILE).is_file()
    logger.info(f"device weight cache {'HIT' if complete else 'MISS'}: {cache_dir}")
    t0 = time.time()
    cache.load_model(
        model,
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=mesh.parallel_config,
        mesh_shape=tuple(mesh.device.shape),
        mesh_device=mesh.device,
        get_torch_state_dict=lambda: torch_dit(variant, snapshot).state_dict(),
    )
    logger.info(
        f"RESULT weights[{variant}] {mesh.tag}: {'cache load' if complete else 'torch load + cache write'} "
        f"in {time.time() - t0:.1f}s"
    )


@dataclass
class DeviceRequest:
    """A request uploaded once: image tokens, hoisted text / cond projections and the plan."""

    request: Request
    plan: Flux3KleinSequencePlan
    image_tokens: ttnn.Tensor
    text_embedded: ttnn.Tensor
    image_cond_embedded: ttnn.Tensor | None

    def deallocate(self) -> None:
        for t in (self.image_tokens, self.text_embedded, self.image_cond_embedded):
            if t is not None:
                ttnn.deallocate(t)
        self.plan.deallocate()


def upload_rows(mesh: Mesh, rows: torch.Tensor) -> ttnn.Tensor:
    """``[L, C]`` -> bf16 ``[1, L_pad, C]`` ``mesh_axes=[None, sp, None]`` with zero alignment rows."""
    padded = packing.pad_rows(rows.to(BF16), padded_length(rows.shape[0], mesh.sp))[0]
    return tensor.from_torch(padded[None], device=mesh.device, mesh_axes=[None, mesh.sp_axis, None])


def upload_request(mesh: Mesh, model: Flux3KleinTransformer, request: Request) -> DeviceRequest:
    t0 = time.time()
    layout = model.sequence_layout(request.lengths)
    plan = model.prepare_plan(layout, request.ids)
    image_tokens = upload_rows(mesh, request.image[0])
    text_embedded = model.embed_text(upload_rows(mesh, request.text[0]))
    cond_embedded = (
        model.embed_image_cond(upload_rows(mesh, request.image_cond[0])) if request.image_cond is not None else None
    )
    ttnn.synchronize_device(mesh.device)
    logger.info(
        f"request {request.tag} {mesh.tag}: layout L_real={layout.total} L_pad={layout.total_padded} R={layout.rows_total}"
        f" padded={layout.padded}; plan + hoisted projections in {time.time() - t0:.2f}s"
    )
    return DeviceRequest(request, plan, image_tokens, text_embedded, cond_embedded)


def device_velocity(
    mesh: Mesh,
    model: Flux3KleinTransformer,
    dev: DeviceRequest,
    step,
    *,
    t_ref_step=None,
    trace=None,
) -> torch.Tensor:
    """One device forward at ``step``; the flow map takes ``t_next`` from ``t_ref_step`` (default: ``step``).
    fp32 ``[L_img, 128]``."""
    t_embed, _ = timestep_inputs(step, False)
    t_ref = timestep_inputs(step if t_ref_step is None else t_ref_step, True)[1] if model.flow_map else None
    tt_t, tt_ref = timesteps_to_device(t_embed, t_ref, device=mesh.device)
    ttnn.synchronize_device(mesh.device)
    t0 = time.time()
    out = model(
        image_tokens=dev.image_tokens,
        text_embedded=dev.text_embedded,
        image_cond_embedded=dev.image_cond_embedded,
        t_embed=tt_t,
        t_ref_embed=tt_ref,
        plan=dev.plan,
        trace=trace,
    )
    ttnn.synchronize_device(mesh.device)
    wall = time.time() - t0
    length = dev.request.lengths["image"]
    assert tuple(out.shape) == (1, dev.plan.layout.padded["image"] // mesh.sp, 128), tuple(out.shape)
    velocity = tensor.to_torch(out, mesh_axes=[None, mesh.sp_axis, None])[0, :length].float()
    ttnn.deallocate(out)
    logger.info(
        f"device forward {dev.request.tag} {mesh.tag} step {step.index} (embedding_t={step.embedding_t}"
        f"{'' if t_ref is None else f', embedding_t_next={float(t_ref[0, 0])}'}) in {wall:.2f}s host wall"
    )
    assert torch.isfinite(velocity).all(), "non-finite velocity"
    return velocity


# --- T1: random weights, DEPTH 1 / 2, both models, with and without a reference ------------------------------------------


@mesh_rows
@pytest.mark.parametrize("variant", ["base", "flow_map"])
@pytest.mark.parametrize("case", ["t2i", "ref80"])
def test_fast_random(
    mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, case: str, variant: str, monkeypatch
) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    patch_reference_depth(monkeypatch, ref_dit, FAST_DEPTH)
    flow_map = variant == "flow_map"
    dit = random_dit(flow_map, seed=50)
    request = Request(
        image=random_image(51, (32, 32)),
        text=random_text(52, EMPTY_PROMPT_TEXT_LENGTH),
        image_cond=random_cond(53, (8, 10)) if case == "ref80" else None,
    )
    scheduler = Flux3KleinScheduler(16 if flow_map else 50)
    step = scheduler.step(0)
    reference = reference_velocity(dit, [request.sample()], scheduler, 0)

    model = device_transformer(mesh, flow_map=flow_map, depth=FAST_DEPTH)
    t0 = time.time()
    model.load_torch_state_dict(dit.state_dict(), strict=True)
    logger.info(f"random DiT ({len(dit.state_dict())} keys) loaded strict in {time.time() - t0:.1f}s")
    dev = upload_request(mesh, model, request)
    got = device_velocity(mesh, model, dev, step)
    check(
        f"fast_random[{case},{variant}] {mesh.tag} {request.tag} depth {FAST_DEPTH}",
        reference,
        got,
        pcc=VELOCITY_PCC,
        relative_rmse=VELOCITY_RMSE,
    )
    # A second forward on the same instance and plan (the step loop) must reproduce the first bit for bit.
    again = device_velocity(mesh, model, dev, step)
    assert torch.equal(got, again), "the second forward of the same step differs from the first"
    logger.info(f"RESULT fast_random[{case},{variant}] {mesh.tag}: second forward bit-equal to the first")
    dev.deallocate()


# --- T2: real base weights, full depth ---------------------------------------------------------------------------------


@mesh_rows
def test_base_real(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    """T2I, 1-reference and the CFG step of the base model at step 0 of the 50-step schedule (full depth)."""
    snapshot = variant_checkpoint_dir("base", required=("model.safetensors",))
    if snapshot is None:
        pytest.skip("FLUX 3 Klein base checkpoint not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    model = device_transformer(mesh, flow_map=False, depth=FULL_DEPTH)
    load_real_weights(mesh, model, "base", snapshot)
    scheduler = Flux3KleinScheduler(FLUX3_KLEIN_STEPS["base"])
    step = scheduler.step(0)
    guidance = FLUX3_KLEIN_GUIDANCE["base"]

    seed = 600
    image = random_image(seed, (64, 64))
    text320 = random_text(seed + 1, 320)
    text80 = random_text(seed + 2, EMPTY_PROMPT_TEXT_LENGTH)
    cond = random_cond(seed + 3, (64, 64))
    t2i = Request(image=image, text=text320)
    ref1 = Request(image=image, text=text320, image_cond=cond)
    uncond = Request(image=image, text=text80)
    oracle = lambda case, samples, compute: cached_oracle(  # noqa: E731
        snapshot=snapshot,
        variant="base",
        case=case,
        seed=seed,
        step_index=0,
        samples=samples,
        step=step,
        compute=compute,
    )

    # (a) T2I 1024 and (b) one 1024x1024 reference.
    for case, request in (("t2i", t2i), ("ref1", ref1)):
        reference = oracle(
            case,
            [request.sample()],
            lambda: {"velocity": reference_velocity(torch_dit("base", snapshot), [request.sample()], scheduler, 0)},
        )["velocity"]
        dev = upload_request(mesh, model, request)
        got = device_velocity(mesh, model, dev, step)
        check(
            f"base_real[{case}] {mesh.tag} {request.tag}", reference, got, pcc=VELOCITY_PCC, relative_rmse=VELOCITY_RMSE
        )
        dev.deallocate()

    # (c) CFG: one reference call on the [unconditional, conditional] pack vs two device forwards.
    def cfg_reference() -> dict[str, torch.Tensor]:
        both = reference_velocity(torch_dit("base", snapshot), [uncond.sample(), t2i.sample()], scheduler, 0)
        u, c = both.chunk(2)
        return {"velocity_uncond": u, "velocity_cond": c}

    reference = oracle("cfg_t2i", [uncond.sample(), t2i.sample()], cfg_reference)
    dev_uncond = upload_request(mesh, model, uncond)
    dev_cond = upload_request(mesh, model, t2i)
    got_uncond = device_velocity(mesh, model, dev_uncond, step)
    got_cond = device_velocity(mesh, model, dev_cond, step)
    name = f"base_real[cfg] {mesh.tag} uncond {uncond.tag} / cond {t2i.tag}"
    check(
        f"{name}: velocity_uncond",
        reference["velocity_uncond"],
        got_uncond,
        pcc=VELOCITY_PCC,
        relative_rmse=VELOCITY_RMSE,
    )
    check(f"{name}: velocity_cond", reference["velocity_cond"], got_cond, pcc=VELOCITY_PCC, relative_rmse=VELOCITY_RMSE)
    check(
        f"{name}: cfg_velocity(u, c, {guidance})",
        Flux3KleinScheduler.cfg_velocity(
            reference["velocity_uncond"].to(BF16), reference["velocity_cond"].to(BF16), guidance
        ).float(),
        Flux3KleinScheduler.cfg_velocity(got_uncond.to(BF16), got_cond.to(BF16), guidance).float(),
        pcc=VELOCITY_PCC,
        relative_rmse=VELOCITY_RMSE,
    )
    dev_uncond.deallocate()
    dev_cond.deallocate()


# --- T3: real distilled weights, the flow-map path ---------------------------------------------------------------------


@mesh_rows
def test_distilled_real_flow_map(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    """T2I 4096 + 320 at step 0 of the 16-step schedule with ``timesteps_ref = schedule[1]`` (full depth, 359 keys);
    plus the liveness check that ``ref_time_in`` is wired: the same forward with step 8's ``t_next`` must differ."""
    snapshot = variant_checkpoint_dir("distilled", required=("model.safetensors",))
    if snapshot is None:
        pytest.skip("FLUX 3 Klein distilled checkpoint not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    model = device_transformer(mesh, flow_map=True, depth=FULL_DEPTH)
    load_real_weights(mesh, model, "distilled", snapshot)
    scheduler = Flux3KleinScheduler(FLUX3_KLEIN_STEPS["distilled"])
    step = scheduler.step(0)

    seed = 700
    request = Request(image=random_image(seed, (64, 64)), text=random_text(seed + 1, 320))
    reference = cached_oracle(
        snapshot=snapshot,
        variant="distilled",
        case="t2i",
        seed=seed,
        step_index=0,
        samples=[request.sample()],
        step=step,
        compute=lambda: {
            "velocity": reference_velocity(torch_dit("distilled", snapshot), [request.sample()], scheduler, 0)
        },
    )["velocity"]
    dev = upload_request(mesh, model, request)
    got = device_velocity(mesh, model, dev, step)
    check(
        f"distilled_real[t2i flow_map] {mesh.tag} {request.tag} t_next=schedule[1]",
        reference,
        got,
        pcc=VELOCITY_PCC,
        relative_rmse=VELOCITY_RMSE,
    )

    # Liveness of the ref path: another step's t_next (same t) must move the velocity away from the oracle.
    other = scheduler.step(8)
    moved = device_velocity(mesh, model, dev, step, t_ref_step=other)
    a, b = reference.double().flatten(), moved.double().flatten()
    cov = torch.cov(torch.stack([a, b]))
    pcc_moved = (cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item()
    logger.info(
        f"RESULT distilled_real[liveness] {mesh.tag}: forward with t_next of step 8 (embedding_t_next="
        f"{other.embedding_t_next}) vs the step-0 oracle: pcc={pcc_moved:.6f} (must be < {LIVENESS_PCC_MAX})"
    )
    assert pcc_moved < LIVENESS_PCC_MAX, "ref_time_in has no effect on the velocity"
    dev.deallocate()


# --- T4: strict loading ------------------------------------------------------------------------------------------------


def parameter_names(module, prefix: str = "") -> list[str]:
    names = [f"{prefix}{name}" for name, _ in module.named_parameters()]
    for name, child in module.named_children():
        names.extend(parameter_names(child, f"{prefix}{name}."))
    return sorted(names)


FUSED_BLOCK_PARAMETERS = (
    "attn.norm_k.weight",
    "attn.norm_q.weight",
    "attn.to_qkv.weight",
    "proj_mlp.weight",
    "proj_out.weight",
)


def expected_parameters(flow_map: bool, depth: tuple[int, int]) -> list[str]:
    names = ["txt_in.weight", "emb_in_image.weight", "emb_in_image_cond.weight", "proj_out.weight"]
    names += ["modulation.lin.weight", "modulation.lin.bias", "modulation.time_in.linear_1.weight"]
    names += ["modulation.time_in.linear_2.weight"]
    if flow_map:
        names += ["modulation.ref_time_in.linear_1.weight", "modulation.ref_time_in.linear_2.weight"]
    names += [f"early.{m}.{i}.{p}" for m in FLUX3_KLEIN_MODES for i in range(depth[0]) for p in FUSED_BLOCK_PARAMETERS]
    names += [f"single_blocks.{i}.{p}" for i in range(depth[1]) for p in FUSED_BLOCK_PARAMETERS]
    return sorted(names)


@mesh_rows
def test_strict_load_negative(
    mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, monkeypatch, expect_error
) -> None:
    # The real checkpoints' key sets are exactly the reference DiT's (357 base / 359 distilled): the loader's input.
    for variant, flow_map, count in (("base", False, 357), ("distilled", True, 359)):
        snapshot = variant_checkpoint_dir(variant, required=("model.safetensors",))
        if snapshot is None:
            logger.info(f"{variant} checkpoint not found; key-set check skipped")
            continue
        with torch.device("meta"):
            expected = set(ref_dit.DiT(flow_map).state_dict())
        keys = ref_checkpoint.checkpoint_keys(str(snapshot))
        assert keys == expected and len(keys) == count, (len(keys), sorted(keys ^ expected)[:5])
        logger.info(f"RESULT keys[{variant}]: {snapshot.name} holds exactly the {count} reference DiT keys")

    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    patch_reference_depth(monkeypatch, ref_dit, FAST_DEPTH)
    base = dict(random_dit(False, seed=1).state_dict())
    distilled = dict(random_dit(True, seed=2).state_dict())
    assert len(base) == 53 and len(distilled) == 55, (len(base), len(distilled))
    fresh = lambda flow_map: device_transformer(mesh, flow_map=flow_map, depth=FAST_DEPTH)  # noqa: E731

    with expect_error(ValueError, r"missing Torch state keys: modulation\.ref_time_in\.linear_1\.weight"):
        fresh(True).load_torch_state_dict(base, strict=True)
    with expect_error(ValueError, r"unexpected Torch state keys: modulation\.ref_time_in\.in_layer\.weight"):
        fresh(False).load_torch_state_dict(distilled, strict=True)

    dropped = dict(base)
    del dropped["single_blocks.1.mlp_in.weight"]
    with expect_error(ValueError, r"missing Torch state keys: single_blocks\.1\.proj_mlp\.weight"):
        fresh(False).load_torch_state_dict(dropped, strict=True)

    extra = dict(base)
    extra["single_blocks.2.q_proj.weight"] = base["single_blocks.1.q_proj.weight"]
    with expect_error(ValueError, r"unexpected Torch state keys: single_blocks\.2\.q_proj\.weight"):
        fresh(False).load_torch_state_dict(extra, strict=True)

    # Positive controls: both key sets load strictly into exactly the expected parameters.
    for flow_map, state in ((False, base), (True, distilled)):
        model = fresh(flow_map)
        model.load_torch_state_dict(state, strict=True)
        assert parameter_names(model) == expected_parameters(flow_map, FAST_DEPTH), parameter_names(model)
        logger.info(
            f"RESULT strict_load flow_map={flow_map}: {len(state)} reference keys -> {len(parameter_names(model))} parameters"
        )
