# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M3 heads: ``Flux3KleinModulation`` (time path, merged modulation heads, per-token table) on device vs the
vendored torch reference (``reference/flux3_klein/dit.py``: ``timestep_embedding``, ``MLPEmbedder``,
``ModeModulation``, ``LastLayer.adaLN_modulation``) under ``torch.inference_mode()`` + CPU bf16 autocast on bf16
weights, exactly as the reference sampler runs them.

Rows (each on the (2,2) Linear submesh for iteration and the (4,8) Ring with 8 KB packets as the gate; mesh handling
copied from the M1 block test, see its module docstring for why the (2,2) mesh is opened as a submesh):
  H1 test_time_path[base|distilled]   real ``time_in`` (+ ``ref_time_in``) weights; every schedule point of the model
                                      (t_model of all steps and every t_next_model: 100 rows for the 50-step base
                                      schedule, 32 for the 16-step distilled one): (i) the device bf16 sinusoid is
                                      ``torch.equal`` to the reference ``timestep_embedding`` (hard assert); (ii) ``vec``
                                      vs the reference bf16 ``vec`` at the VEC gate; the distilled row adds the flow-map
                                      ``vec`` with ``ref_time_in(t_next)`` at steps 0 and 15.
  H2 test_table_and_gathers[base|distilled]  real heads + real time path at step 0: the 60-row table vs the torch
                                      heads (TABLE gate on the 40 real rows), pad rows EXACTLY 0 / 1 / 0 (hard assert:
                                      the fused +1 bias with the masked zero activation row), ``final_rows`` ==
                                      table rows 54 / 57 and accepted by ``DistributedLayerNorm`` as dynamic affine,
                                      ``gather_modulation`` for the early and single index sets an exact copy of the
                                      table rows with gate pad rows 0 and scale pad rows 1, ``rope_to_device`` round trip.
  H3 test_strict_load_negative        missing / unexpected reference keys are reported by the strict loader.

Thresholds (never loosened; measured floors in the constants below).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.normalization import DistributedLayerNorm
from ....models.transformers.flux3_klein.modulation_flux3_klein import (
    FLUX3_KLEIN_MOD_CHUNKS,
    FLUX3_KLEIN_MOD_FINAL_SCALE_CHUNK,
    FLUX3_KLEIN_MOD_FINAL_SHIFT_CHUNK,
    FLUX3_KLEIN_MOD_PAD_MODE_ID,
    FLUX3_KLEIN_MOD_SCALE_CHUNKS,
    FLUX3_KLEIN_MOD_SOURCES,
    FLUX3_KLEIN_MOD_TABLE_ROWS,
    FLUX3_KLEIN_MOD_VEC_ROWS,
    Flux3KleinModulation,
    gather_modulation,
    mod_chunk,
    modulation_table_indices,
    timestep_inputs,
    timesteps_to_device,
)
from ....models.transformers.flux3_klein.rope_flux3_klein import rope_cos_sin, rope_to_device
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.packing import image_ids, padded_length, text_ids
from ....pipelines.flux3_klein.scheduler import FLUX3_KLEIN_STEPS, Flux3KleinScheduler
from ....reference.flux3_klein.dit import HIDDEN, LastLayer, MLPEmbedder, ModeModulation, timestep_embedding
from ....utils import tensor
from ....utils.check import assert_quality
from .common import variant_checkpoint_dir
from .device_params import ring_params_8k_flux3_klein_transformer

D = HIDDEN
BF16 = torch.bfloat16

# H1 (ii): fp32 TimestepEmbedding on the bf16-cast sinusoid vs the reference bf16 MLPEmbedder. Measured (judge probe,
# real base time_in, 100 + 32 points, both meshes) 0.999997 / 0.0032 / max_abs 0.0078. Bug classes: a bf16 MLP path
# measures 0.0064 (design A), an un-rounded 1000 * t moves the sinusoid argument by up to 4 rad at t = 1.
VEC_PCC = 0.9999
VEC_RMSE = 0.01

# H2: the 40 real table rows (bf16 matmul with fp32 accumulation + fused bias vs the reference's bf16 head output,
# scale rows +1 in bf16). Measured (design A, real heads, both meshes) 0.999998 / 0.0021 (base) and 0.999997 / 0.0026
# (distilled). Bug calibration (M1 host sweep, block level): chunk order shifted by one head or scale without the +1
# measures 0.970 / 0.243.
TABLE_PCC = 0.9999
TABLE_RMSE = 0.012

# H2: the final layer's DistributedLayerNorm with the sliced rows as dynamic affine vs fp32 torch LN on the same bf16
# rows: a bf16-in / bf16-out LayerNorm floor (the M1 block gate; the measured value is logged and belongs in the ledger).
LN_PCC = 0.999
LN_RMSE = 0.032

TIME_KEYS = ("time_in.in_layer.weight", "time_in.out_layer.weight")
REF_TIME_KEYS = ("ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight")
HEAD_KEYS = tuple(f"{prefix}.weight" for prefix, _, _ in FLUX3_KLEIN_MOD_SOURCES)

mesh_rows = pytest.mark.parametrize(
    "mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, device_params",
    [
        [(4, 8), (2, 2), 0, 1, ttnn.Topology.Linear, 2, ring_params_8k_flux3_klein_transformer],
        [(4, 8), None, 0, 1, ttnn.Topology.Ring, 2, ring_params_8k_flux3_klein_transformer],
    ],
    ids=["bh_glx_2x2sub_linear", "bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)

variants = pytest.mark.parametrize("variant", ["base", "distilled"])


# --- mesh -----------------------------------------------------------------------------------------------------------


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

    @property
    def tp(self) -> int:
        return self.parallel_config.tensor_parallel.factor


def open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> Mesh:
    device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape)) if submesh_shape else mesh_device
    shape = tuple(device.shape)
    sp, tp = shape[sp_axis], shape[tp_axis]
    ccl = CCLManager(mesh_device=device, num_links=num_links, topology=topology)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp, mesh_axis=sp_axis),
    )
    tag = f"{shape[0]}x{shape[1]}_sp{sp}_tp{tp}_{'ring' if topology == ttnn.Topology.Ring else 'linear'}"
    logger.info(f"mesh {tag}: parent {tuple(mesh_device.shape)}, {num_links} links")
    return Mesh(device, sp_axis, tp_axis, ccl, parallel_config, tag)


# --- weights --------------------------------------------------------------------------------------------------------


def real_state(variant: str, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Only ``keys`` of ``variant``'s ``model.safetensors`` (11 GB; never loaded whole), bf16 as stored."""
    snapshot = variant_checkpoint_dir(variant, required=("model.safetensors",))
    if snapshot is None:
        pytest.skip(f"FLUX 3 Klein {variant} checkpoint not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    from safetensors import safe_open

    t0 = time.time()
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        state = {key: f.get_tensor(key) for key in keys}
    logger.info(f"loaded {len(state)} tensors of {snapshot.name} in {time.time() - t0:.1f}s")
    return state


def random_state(flow_map: bool, seed: int = 0) -> dict[str, torch.Tensor]:
    """Reference-named random weights of this part (N(0, 0.02), bf16)."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for embedder in ("time_in", "ref_time_in") if flow_map else ("time_in",):
        state[f"{embedder}.in_layer.weight"] = (0.02 * torch.randn(D, 256, generator=g)).to(BF16)
        state[f"{embedder}.out_layer.weight"] = (0.02 * torch.randn(D, D, generator=g)).to(BF16)
    for prefix, count, _ in FLUX3_KLEIN_MOD_SOURCES:
        state[f"{prefix}.weight"] = (0.02 * torch.randn(count * D, D, generator=g)).to(BF16)
    return state


def device_modulation(mesh: Mesh, state: dict[str, torch.Tensor], flow_map: bool) -> Flux3KleinModulation:
    mod = Flux3KleinModulation(
        flow_map=flow_map, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    t0 = time.time()
    mod.load_torch_state_dict(state, strict=True)
    logger.info(
        f"modulation weights ({len(state)} reference keys, flow_map={flow_map}) loaded strict in {time.time() - t0:.1f}s"
    )
    return mod


# --- reference ------------------------------------------------------------------------------------------------------


def torch_embedder(state: dict[str, torch.Tensor], prefix: str) -> MLPEmbedder:
    mlp = MLPEmbedder(256, D)
    mlp.load_state_dict(
        {"in_layer.weight": state[f"{prefix}.in_layer.weight"], "out_layer.weight": state[f"{prefix}.out_layer.weight"]}
    )
    return mlp.to(BF16).eval()


def reference_vec(state: dict[str, torch.Tensor], t_model: list[float], t_next_model: float | None) -> torch.Tensor:
    """``vec`` rows ``[len(t_model), D]`` (fp32 copy of the bf16 result) as ``dit.py:199-206`` computes them."""
    time_in = torch_embedder(state, "time_in")
    ref_time_in = torch_embedder(state, "ref_time_in") if t_next_model is not None else None
    with torch.inference_mode(), torch.autocast("cpu", dtype=BF16):
        vec = time_in(timestep_embedding(torch.tensor(t_model, dtype=BF16)))
        if ref_time_in is not None:
            vec = vec + ref_time_in(timestep_embedding(torch.tensor([t_next_model], dtype=BF16)))
    return vec.float().clone()


def reference_table(state: dict[str, torch.Tensor], vec_rows: torch.Tensor) -> torch.Tensor:
    """The expected device table ``[60, D]`` (fp32 copy of bf16 values): chunk ``c`` of head ``h`` at vec row ``r`` is
    the head's output row ``r`` (scale chunks: ``(1 + scale)`` rounded to bf16, the M1 convention); the third row of
    every chunk is the pad row 0 / 1 / 0. ``vec_rows`` ``[2, D]`` fp32 (image row, zero-timestep row)."""
    vec = vec_rows.to(BF16)
    rows = []
    with torch.inference_mode(), torch.autocast("cpu", dtype=BF16):
        for prefix, count, scale_offsets in FLUX3_KLEIN_MOD_SOURCES:
            weight = state[f"{prefix}.weight"]
            if count == 3:
                head = ModeModulation(D)
                head.lin.load_state_dict({"weight": weight})
                outs = head.to(BF16).eval()(vec)  # shift, scale, gate (dit.py:124)
            else:
                last = LastLayer(D, 128)
                last.adaLN_modulation[1].load_state_dict({"weight": weight})
                outs = last.to(BF16).eval().adaLN_modulation(vec).chunk(2, dim=-1)  # shift, scale (dit.py:134)
            for i, out in enumerate(outs):
                out = out.to(BF16)
                if i in scale_offsets:
                    out = (1.0 + out.float()).to(BF16)
                    pad_row = torch.ones(1, D, dtype=BF16)
                else:
                    pad_row = torch.zeros(1, D, dtype=BF16)
                rows.append(torch.cat([out, pad_row]))
    table = torch.cat(rows).float().clone()
    assert table.shape == (FLUX3_KLEIN_MOD_TABLE_ROWS, D)
    return table


# --- checks ---------------------------------------------------------------------------------------------------------


def check(name: str, reference: torch.Tensor, got: torch.Tensor, *, pcc: float, relative_rmse: float) -> None:
    """Log ``pcc / relative RMSE / max abs`` at full precision (the ledger takes these numbers), then gate."""
    a = reference.detach().double().flatten()
    b = got.detach().double().flatten()
    cov = torch.cov(torch.stack([a, b]))
    pcc_found = (cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item()
    rmse_found = (torch.nn.functional.mse_loss(a, b).sqrt() / cov[0, 0].sqrt()).item()
    logger.info(
        f"RESULT {name}: pcc={pcc_found:.6f} rel_rmse={rmse_found:.4f} max_abs={(a - b).abs().max().item():.4f} "
        f"(gate pcc>={pcc} rel_rmse<={relative_rmse})"
    )
    assert_quality(reference, got, pcc=pcc, relative_rmse=relative_rmse)


def bit_equal_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(BF16).view(torch.int16) == b.to(BF16).view(torch.int16)).float().mean().item()


def schedule_points(num_steps: int) -> tuple[list[float], list[float]]:
    """``(t_model, embedding_t)`` of every step's model timestep followed by every step's flow-map target: the 51 / 17
    distinct grid points plus every ``t_next``."""
    steps = Flux3KleinScheduler(num_steps).steps()
    t_model = [s.t_model for s in steps] + [s.t_next_model for s in steps]
    t_embed = [s.embedding_t for s in steps] + [s.embedding_t_next for s in steps]
    return t_model, t_embed


def indices_to_device(mesh: Mesh, idx: torch.Tensor) -> ttnn.Tensor:
    """``[N]`` -> uint32 ROW_MAJOR ``[1, N]`` fractured over SP (device d holds its own ``N / SP`` rows)."""
    assert idx.numel() % (32 * mesh.sp) == 0, idx.numel()
    return tensor.from_torch(
        idx.to(torch.int32)[None],
        device=mesh.device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_axes=[None, mesh.sp_axis],
    )


# --- H1: time path ----------------------------------------------------------------------------------------------------


@mesh_rows
@variants
def test_time_path(
    mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, variant: str, expect_error
) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    flow_map = variant == "distilled"
    keys = TIME_KEYS + (REF_TIME_KEYS if flow_map else ()) + HEAD_KEYS
    state = real_state(variant, keys)
    mod = device_modulation(mesh, state, flow_map)
    num_steps = FLUX3_KLEIN_STEPS[variant]

    # (i) the bf16 sinusoid on every schedule point of both grids (no weights involved: the claim covers both models),
    # (ii) vec on this model's own grid (base: the plain time_in rows; distilled: flow-map rows below).
    for steps in (50, 16):
        t_model, t_embed = schedule_points(steps)
        tt_t = tensor.from_torch(
            torch.tensor(t_embed, dtype=torch.float32)[:, None],
            device=mesh.device,
            dtype=ttnn.float32,
            mesh_axes=[None, None],
        )
        sinus_dev = tensor.to_torch(mod.sinusoid_bf16(tt_t)).reshape(-1, 256)
        sinus_ref = timestep_embedding(torch.tensor(t_model, dtype=BF16))
        assert sinus_dev.dtype == BF16 and sinus_ref.dtype == BF16
        equal = torch.equal(sinus_dev, sinus_ref)
        logger.info(
            f"RESULT time[{variant}] {mesh.tag} {steps}-step: device bf16 sinusoid == reference timestep_embedding: "
            f"{equal} ({len(t_model)} rows: {steps} t_model + {steps} t_next_model), max_abs="
            f"{(sinus_dev.float() - sinus_ref.float()).abs().max().item():.3e}"
        )
        assert equal, "device bf16 sinusoid differs from the reference timestep_embedding"
        if not flow_map:
            vec_dev = tensor.to_torch(mod.vec(tt_t)).reshape(-1, D).float()
            vec_ref = reference_vec(state, t_model, None)
            logger.info(f"  vec bf16 bit-equal fraction={bit_equal_fraction(vec_ref, vec_dev):.3f}")
            check(
                f"vec[{variant}] {mesh.tag} {steps}-step {len(t_model)} rows vs reference",
                vec_ref,
                vec_dev,
                pcc=VEC_PCC,
                relative_rmse=VEC_RMSE,
            )

    if flow_map:
        sched = Flux3KleinScheduler(num_steps)
        for index in (0, num_steps - 1):
            step = sched.step(index)
            tt_t, tt_ref = timesteps_to_device(*timestep_inputs(step, True), device=mesh.device)
            vec_dev = tensor.to_torch(mod.vec(tt_t, tt_ref)).reshape(-1, D).float()
            assert vec_dev.shape == (FLUX3_KLEIN_MOD_VEC_ROWS, D)
            vec_ref = reference_vec(state, [step.t_model, 0.0, 0.0], step.t_next_model)
            logger.info(
                f"  flow-map step {index}: t_model={step.t_model} t_next_model={step.t_next_model} "
                f"embedding_t={step.embedding_t} embedding_t_next={step.embedding_t_next} "
                f"bit-equal fraction={bit_equal_fraction(vec_ref, vec_dev):.3f}"
            )
            check(
                f"vec[{variant} flow_map] {mesh.tag} step {index} (3 rows, + ref_time_in(t_next))",
                vec_ref,
                vec_dev,
                pcc=VEC_PCC,
                relative_rmse=VEC_RMSE,
            )
            # Liveness of the ref path: the reference rows without the ref term differ from the rows with it.
            vec_no_ref = reference_vec(state, [step.t_model, 0.0, 0.0], None)
            delta = (vec_ref - vec_no_ref).abs().max().item()
            logger.info(f"  ref_time_in contribution max_abs={delta:.4f}")
            assert delta > 0.0

    # The ValueError of dit.py:189-190 in both directions (a host-side argument check).
    tt_t, tt_ref = timesteps_to_device(
        *timestep_inputs(Flux3KleinScheduler(num_steps).step(0), True), device=mesh.device
    )
    with expect_error(ValueError, "flow-map"):
        mod.vec(tt_t, None if flow_map else tt_ref)


# --- H2: table, final rows, gathers -------------------------------------------------------------------------------------


@mesh_rows
@variants
def test_table_and_gathers(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, variant: str) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    flow_map = variant == "distilled"
    keys = TIME_KEYS + (REF_TIME_KEYS if flow_map else ()) + HEAD_KEYS
    state = real_state(variant, keys)
    mod = device_modulation(mesh, state, flow_map)
    step = Flux3KleinScheduler(FLUX3_KLEIN_STEPS[variant]).step(0)
    logger.info(f"step 0: t_model={step.t_model} embedding_t={step.embedding_t} t_next_model={step.t_next_model}")

    # --- device: one forward = vec -> 20 chunks -> table + final rows (timed twice: cold JIT, then warm).
    tt_t, tt_ref = timesteps_to_device(*timestep_inputs(step, flow_map), device=mesh.device)
    for attempt in ("cold", "warm"):
        ttnn.synchronize_device(mesh.device)
        t0 = time.time()
        table, (final_shift, final_scale) = mod(tt_t, tt_ref)
        ttnn.synchronize_device(mesh.device)
        logger.info(f"modulation forward ({attempt}) {mesh.tag}: {time.time() - t0:.3f}s host wall")
    assert table.layout == ttnn.ROW_MAJOR_LAYOUT and table.dtype == ttnn.bfloat16
    assert tuple(table.shape) == (FLUX3_KLEIN_MOD_TABLE_ROWS, D // mesh.tp), tuple(table.shape)
    table_host = tensor.to_torch(table, mesh_axes=[None, mesh.tp_axis])
    assert table_host.shape == (FLUX3_KLEIN_MOD_TABLE_ROWS, D) and table_host.dtype == BF16

    # --- reference table from the reference vec rows (image at t, zero timestep) + ref for the flow map.
    vec_ref = reference_vec(state, [step.t_model, 0.0], step.t_next_model if flow_map else None)
    expected = reference_table(state, vec_ref)
    real_rows = [r for r in range(FLUX3_KLEIN_MOD_TABLE_ROWS) if r % FLUX3_KLEIN_MOD_VEC_ROWS != 2]
    for c in range(FLUX3_KLEIN_MOD_CHUNKS):
        rows = [3 * c, 3 * c + 1]
        a, b = expected[rows].double().flatten(), table_host[rows].double().flatten()
        cov = torch.cov(torch.stack([a, b]))
        logger.info(
            f"  chunk {c:2d}{' (scale)' if c in FLUX3_KLEIN_MOD_SCALE_CHUNKS else '        '}: "
            f"pcc={(cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item():.6f} "
            f"rel_rmse={(torch.nn.functional.mse_loss(a, b).sqrt() / cov[0, 0].sqrt()).item():.4f} "
            f"max_abs={(a - b).abs().max().item():.4f} bit_equal={bit_equal_fraction(expected[rows], table_host[rows]):.3f}"
        )
    check(
        f"table[{variant}] {mesh.tag} 40 real rows vs torch heads",
        expected[real_rows],
        table_host[real_rows].float(),
        pcc=TABLE_PCC,
        relative_rmse=TABLE_RMSE,
    )
    logger.info(
        f"  table bit-equal fraction (40 real rows)={bit_equal_fraction(expected[real_rows], table_host[real_rows]):.4f}"
    )

    # --- pad rows: EXACTLY 0 / 1 / 0 (the fused +1 bias on a masked zero activation row).
    for c in range(FLUX3_KLEIN_MOD_CHUNKS):
        pad_row = table_host[3 * c + 2]
        value = 1.0 if c in FLUX3_KLEIN_MOD_SCALE_CHUNKS else 0.0
        assert torch.equal(pad_row, torch.full_like(pad_row, value)), f"chunk {c} pad row is not exactly {value}"
    logger.info(f"RESULT table[{variant}] {mesh.tag} pad rows 2::3 exactly 0 / 1 / 0 over all 20 chunks: True")

    # --- final rows: slices of chunks 18 / 19 at row 0 == table rows 54 / 57, shaped [1, 1, D/TP] TILE.
    shift_rows = 3 * FLUX3_KLEIN_MOD_FINAL_SHIFT_CHUNK
    scale_rows = 3 * FLUX3_KLEIN_MOD_FINAL_SCALE_CHUNK
    assert (shift_rows, scale_rows) == (54, 57)
    for name, row, expected_row in (("shift", final_shift, shift_rows), ("scale_plus_one", final_scale, scale_rows)):
        assert tuple(row.shape) == (1, 1, D // mesh.tp), tuple(row.shape)
        assert row.layout == ttnn.TILE_LAYOUT and row.dtype == ttnn.bfloat16
        row_host = tensor.to_torch(row, mesh_axes=[None, None, mesh.tp_axis])
        assert row_host.shape == (1, 1, D)
        assert torch.equal(
            row_host[0, 0], table_host[expected_row]
        ), f"final {name} differs from table row {expected_row}"
    logger.info(
        f"RESULT final_rows[{variant}] {mesh.tag}: torch.equal to table rows 54 / 57, shape (1, 1, {D // mesh.tp}) TILE"
    )

    # --- the final rows drive DistributedLayerNorm as dynamic affine (the final layer's form; B's e2e used this).
    norm_out = DistributedLayerNorm(
        D,
        norm_eps=1e-6,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=mesh.tp_axis,
        mesh_device=mesh.device,
        ccl_manager=mesh.ccl,
    )
    torch.manual_seed(200)
    rows = 32 * mesh.sp * 2
    x = torch.randn(rows, D).to(BF16)
    tt_x = tensor.from_torch(x[None], device=mesh.device, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])
    y = norm_out(ttnn.unsqueeze(tt_x, 0), dynamic_weight=final_scale, dynamic_bias=final_shift)
    y = ttnn.squeeze(y, 0)
    y_host = tensor.to_torch(y, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0].float()
    y_ref = (
        table_host[scale_rows].float() * torch.nn.functional.layer_norm(x.float(), (D,), eps=1e-6)
        + table_host[shift_rows].float()
    )
    check(
        f"final LN[{variant}] {mesh.tag} (1+scale)*LN(x)+shift with the sliced rows, {rows} rows",
        y_ref,
        y_host,
        pcc=LN_PCC,
        relative_rmse=LN_RMSE,
    )

    # --- gathers: early index sets (one mode + pad tail) and single sets ([image | cond | text | pad]).
    def mode_ids(lengths: dict[str, int], padded: int) -> torch.Tensor:
        order = ("image", "image_cond", "text")
        parts = [torch.full((lengths[m],), i, dtype=torch.long) for i, m in enumerate(order) if lengths.get(m, 0)]
        total = sum(p.numel() for p in parts)
        return torch.cat(parts + [torch.full((padded - total,), FLUX3_KLEIN_MOD_PAD_MODE_ID, dtype=torch.long)])

    gather_cases = []
    for mode, length in (("image", 4096), ("image_cond", 80), ("text", 320), ("text", 80)):
        gather_cases.append(
            (f"early {mode} L={length}", "early", mode_ids({mode: length}, padded_length(length, mesh.sp)))
        )
    for lengths in ({"image": 4096, "image_cond": 80, "text": 320}, {"image": 256, "text": 80}):
        total = sum(lengths.values())
        gather_cases.append(
            (f"single {tuple(lengths.values())}", "single", mode_ids(lengths, padded_length(total, mesh.sp)))
        )
    for name, stage, ids in gather_cases:
        real = int((ids != FLUX3_KLEIN_MOD_PAD_MODE_ID).sum())
        for param, idx in zip(("shift", "scale_plus_one", "gate"), modulation_table_indices(ids, stage)):
            tt_idx = indices_to_device(mesh, idx)
            ttnn.synchronize_device(mesh.device)
            t0 = time.time()
            out = gather_modulation(table, tt_idx)
            ttnn.synchronize_device(mesh.device)
            wall = time.time() - t0
            assert tuple(out.shape) == (1, ids.numel() // mesh.sp, D // mesh.tp), tuple(out.shape)
            assert out.layout == ttnn.TILE_LAYOUT
            got = tensor.to_torch(out, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0]
            exact = torch.equal(got, table_host[idx])
            tail = got[real:]
            pad_value = 1.0 if param == "scale_plus_one" else 0.0
            tail_ok = torch.equal(tail, torch.full_like(tail, pad_value))
            logger.info(
                f"RESULT gather[{variant}] {mesh.tag} {name} {param}: exact copy of table rows={exact} shape={tuple(got.shape)} "
                f"pad rows ({tail.shape[0]}) exactly {pad_value}: {tail_ok} wall={wall * 1e3:.2f} ms"
            )
            assert exact and tail_ok
            # The real rows are the expected head rows (vs torch, at the table gate: same numbers as the table).
            if param == "gate" and real:
                check(
                    f"gather[{variant}] {mesh.tag} {name} gate real rows vs torch",
                    expected[idx[:real]],
                    got[:real].float(),
                    pcc=TABLE_PCC,
                    relative_rmse=TABLE_RMSE,
                )

    # --- rope_to_device round trip in the block's layout.
    cos, sin = rope_cos_sin(torch.cat([image_ids(16, 16), text_ids(80).float()]), padded_length(336, mesh.sp))
    tt_cos, tt_sin = rope_to_device(cos, sin, device=mesh.device, sp_axis=mesh.sp_axis)
    for name, host, dev in (("cos", cos, tt_cos), ("sin", sin, tt_sin)):
        assert dev.dtype == ttnn.float32 and tuple(dev.shape) == (1, 1, cos.shape[2] // mesh.sp, 128)
        assert torch.equal(tensor.to_torch(dev, mesh_axes=[None, None, mesh.sp_axis, None]), host), name

    # --- the heads' row mask is a device constant outside the parameters: freed with the weights (eviction hook),
    # rebuilt lazily on the next forward; a reload + forward reproduces the table bit for bit.
    assert mod._row_mask is not None
    mod.deallocate_weights()
    assert mod._row_mask is None and not mod.is_loaded()
    mod.load_torch_state_dict(real_state(variant, keys), strict=True)
    table_again, _ = mod(tt_t, tt_ref)
    assert mod._row_mask is not None
    assert torch.equal(table_host, tensor.to_torch(table_again, mesh_axes=[None, mesh.tp_axis]))
    logger.info(f"RESULT eviction[{variant}] {mesh.tag}: row mask freed with the weights; reload + forward bit-equal")


# --- H3: strict loading ---------------------------------------------------------------------------------------------


def _named_parameters_recursive(module, prefix: str = ""):
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter
    for name, child in module.named_children():
        yield from _named_parameters_recursive(child, f"{prefix}{name}.")


@mesh_rows
def test_strict_load_negative(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, expect_error) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    base, distilled = random_state(flow_map=False), random_state(flow_map=True)

    def fresh(flow_map: bool) -> Flux3KleinModulation:
        return Flux3KleinModulation(
            flow_map=flow_map, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
        )

    # 6 of 7 heads: no partial merge; lin.weight and lin.bias are reported missing.
    six = dict(base)
    del six["single_stream_modulations.txt.lin.weight"]
    with expect_error(ValueError, r"missing Torch state keys: lin\.weight, lin\.bias"):
        fresh(False).load_torch_state_dict(six, strict=True)

    # Base keys into a flow-map instance: ref_time_in is missing.
    with expect_error(
        ValueError, r"missing Torch state keys: ref_time_in\.linear_1\.weight, ref_time_in\.linear_2\.weight"
    ):
        fresh(True).load_torch_state_dict(base, strict=True)

    # Distilled keys into a base instance: ref_time_in is unexpected (reported under the reference's names).
    with expect_error(
        ValueError, r"unexpected Torch state keys: ref_time_in\.in_layer\.weight, ref_time_in\.out_layer\.weight"
    ):
        fresh(False).load_torch_state_dict(distilled, strict=True)

    # Positive controls: the reference key sets load strictly into exactly the expected parameters.
    for flow_map, state in ((False, base), (True, distilled)):
        mod = fresh(flow_map)
        mod.load_torch_state_dict(state, strict=True)
        loaded = sorted(name for name, _ in _named_parameters_recursive(mod))
        expected = ["lin.bias", "lin.weight", "time_in.linear_1.weight", "time_in.linear_2.weight"]
        if flow_map:
            expected += ["ref_time_in.linear_1.weight", "ref_time_in.linear_2.weight"]
        assert loaded == sorted(expected), loaded
        assert tuple(mod.lin.weight.data.shape) == (D, FLUX3_KLEIN_MOD_CHUNKS * D // mesh.tp), tuple(
            mod.lin.weight.data.shape
        )

    # mod_chunk / modulation_table_indices host contract (cheap; pinned next to the loader so the chunk order and the
    # table layout are asserted together).
    assert [
        mod_chunk("early", m, p) for m in ("image", "image_cond", "text") for p in ("shift", "scale", "gate")
    ] == list(range(9))
    assert [
        mod_chunk("single", m, p) for m in ("image", "image_cond", "text") for p in ("shift", "scale", "gate")
    ] == list(range(9, 18))
    shift, scale, gate = modulation_table_indices(torch.tensor([0, 1, 2, 3]), "single")
    assert (
        shift.tolist() == [27, 37, 46, 29] and scale.tolist() == [30, 40, 49, 32] and gate.tolist() == [33, 43, 52, 35]
    )
    shift, scale, gate = modulation_table_indices(torch.tensor([0, 1, 2, 3]), "early")
    assert shift.tolist() == [0, 10, 19, 2] and scale.tolist() == [3, 13, 22, 5] and gate.tolist() == [6, 16, 25, 8]
    with expect_error(ValueError, "stage must be"):
        modulation_table_indices(torch.tensor([0]), "late")
    with expect_error(ValueError, "mode ids must lie"):
        modulation_table_indices(torch.tensor([4]), "early")
