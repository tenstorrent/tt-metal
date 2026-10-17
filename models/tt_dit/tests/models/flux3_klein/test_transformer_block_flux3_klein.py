# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M1: one FLUX 3 Image Klein block on device against the vendored torch ``Block``.

Oracle: ``reference/flux3_klein/dit.py::Block`` in bf16 under ``torch.inference_mode()`` +
``torch.autocast("cpu", dtype=torch.bfloat16)``, exactly as the reference sampler runs it. CPU autocast keeps
``layer_norm``, the linears and the MATH SDPA in bf16, so the reference's own roundings set the comparison floor:
against the device's fp32-accumulated path the block outputs measure pcc 0.99995-0.99997 / relative RMSE
0.011-0.013, real and random weights alike (the measured per-row floors are listed at ``BLOCK_PCC`` below). The
gate ``pcc >= 0.999, relative_rmse <= 0.032`` sits 2-3x above that floor; every threshold below carries the
"worst plausible bug" it was calibrated against (host and device measurements made while this block was
designed).

Rows (each on the (2,2) Linear submesh for iteration and the (4,8) Ring with 8 KB packets as the gate):
  R0 early_image_random   L=4096, broadcast modulation, seeded random weights with randomized QK-norm scales.
  R1 early_image_real     L=4096, broadcast modulation, ``content_mode_blocks.image.0`` of the BASE checkpoint.
  R2 early_text_masked    L in (80, 320), ``txt_mode_blocks.0``: the production padding and a loud 512-row
                          phantom tail, each vs torch (the mask proof), plus loud tail vs production padding
                          on the real rows.
  R3 attention_branch     L=320, ``txt_mode_blocks.0``: the attention output alone (RoPE / QK-norm / mask proof,
                          where the same bugs the block gate is blind to measure pcc 0.69-0.996).
  R4 single_pertoken      [image | image_cond | text] with per-token three-run modulation, ``single_blocks.0``
                          (L=4496 and a short L=80 that lands 32 rows per device at SP4).
  R5 cfg_pair             two samples vs ONE reference call with ``bounds=[(0, L1), (L1, L1 + L2)]``.
  R6 strict_load_negative missing / unexpected reference keys are reported by the strict loader.

Only the (4,8) mesh is requested; the iteration row is a (2,2) submesh of the (4,8) parent (the root fixture
closes submeshes). The submesh route was chosen because, during the design probes on this Blackhole Galaxy,
direct (2,2), (2,4) and (1,8) opens timed out in the fabric router handshake ("Fabric Router Sync: Timeout
after 120000 ms", ethernet handshake incomplete); only the submesh route has been measured since.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import Flux3KleinTransformerBlock
from ....parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.packing import image_ids, padded_length, reference_ids, text_ids
from ....reference.flux3_klein.dit import (
    HEADS,
    HIDDEN,
    MLP_RATIO,
    Block,
    layernorm_modulation,
    packed_attention,
    rope_embed,
)
from ....utils import tensor
from ....utils.check import assert_quality
from .common import variant_checkpoint_dir
from .device_params import ring_params_8k_flux3_klein_transformer

D = HIDDEN
HEAD_DIM = HIDDEN // HEADS

# Whole-block gate (PLAN.md M1). Measured floors with the fp32-accumulating SDPA the block installs: random weights
# (N(0, 0.02) linears, 2x the default-init std, so the gated update is a larger share of the output than with the
# designers' default-init rows) 0.999958 / 0.0133 on the 2x2 submesh and 0.999954 / 0.0113 on the 4x8 ring; real
# weights 0.99996-0.99997 / 0.0106-0.0131 (2x2: 0.0125-0.0131, 4x8: 0.0106-0.0111). The shared bf16-accumulating
# SDPA default measured the same to four digits at block level (0.999957 / 0.0134 and 0.999952 / 0.0114): the block
# gate cannot see the accumulator width, only R3 can. Worst plausible bugs (host sweep, random weights): swiglu
# halves swapped 0.970 / 0.243,
# scale without the +1 fold 0.970 / 0.243, k heads rolled by one head 0.9956 / 0.094, mlp_out K offset by one
# tile 0.900 / 0.447, [attn | mlp] segments misaligned -0.003 / 1.30, QK-norm scale dropped 0.99956 / 0.0296
# (visible only because the random rows draw scales from U(0.5, 1.5); ones would hide it).
BLOCK_PCC = 0.999
BLOCK_RMSE = 0.032

# Masking proof at block level: real rows of a loud-tail padded run vs the production padding. Measured here
# 0.999999 / 0.0014-0.0017 (2x2) and 0.999995 / 0.0032-0.0033 (4x8); the same loud tail attended as keys (missing
# mask) measures 0.0067-0.0113 in torch, i.e. INSIDE the block gate above, which is why this bar exists.
MASK_PCC = 0.99998
MASK_RMSE = 0.006

# Attention branch (R3): measured 0.999987 / 0.0085 (2x2) and 0.999986 / 0.0083 (4x8) against the reference with the
# fp32-accumulating SDPA (bf16 accumulation, the shared default, measured 0.999967 / 0.0088-0.0091 here: the one
# place the accumulator width shows). Bugs: no RoPE 0.9927 / 0.12, half-split
# RoPE tables (channel i paired with i+64) 0.9878 / 0.16, QK-norm scale dropped 0.9960 / 0.090, q/k scales swapped
# 0.99945 / 0.033 (the RMSE bar catches it, the PCC bar barely: keep both), unmasked loud 512-row tail 0.690 / 0.74.
ATTN_PCC = 0.999
ATTN_RMSE = 0.032
# Attention loud-tail vs production-padding real rows: measured 0.999983 / 0.0059 (2x2) and 0.999984 / 0.0056 (4x8)
# with fp32 accumulation (0.999972 / 0.0075 with bf16 accumulation) vs 0.69 with the tail attended.
ATTN_MASK_PCC = 0.9999
ATTN_MASK_RMSE = 0.01

# fp32 tables remove one non-reference rounding for free; bf16 tables (the Flux2 pipeline convention) measured
# identical at block level (0.999999 / 0.0015 between the two). Flip for a Flux2-compatible comparison.
ROPE_TABLE_DTYPE = ttnn.float32

# A loud phantom tail: 8 * N(0, 1) rows with arbitrary RoPE angles, so an unmasked key corrupts the real rows
# visibly instead of averaging into noise.
LOUD_TAIL_ROWS = 512
LOUD_TAIL_SCALE = 8.0

BASE_BLOCK_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "mlp_in.weight",
    "attn_out.weight",
    "mlp_out.weight",
    "norm.query_norm.scale",
    "norm.key_norm.scale",
)

mesh_rows = pytest.mark.parametrize(
    "mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, device_params",
    [
        [(4, 8), (2, 2), 0, 1, ttnn.Topology.Linear, 2, ring_params_8k_flux3_klein_transformer],
        [(4, 8), None, 0, 1, ttnn.Topology.Ring, 2, ring_params_8k_flux3_klein_transformer],
    ],
    ids=["bh_glx_2x2sub_linear", "bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)


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


# --- reference ------------------------------------------------------------------------------------------------------


def random_block(seed: int) -> Block:
    """Seeded random weights: linears N(0, 0.02); QK-norm scales U(0.5, 1.5) so a dropped scale is visible."""
    torch.manual_seed(seed)
    block = Block(HIDDEN, HEADS, MLP_RATIO)
    with torch.no_grad():
        for linear in (block.q_proj, block.k_proj, block.v_proj, block.mlp_in, block.attn_out, block.mlp_out):
            linear.weight.normal_(0.0, 0.02)
        block.norm.query_norm.scale.uniform_(0.5, 1.5)
        block.norm.key_norm.scale.uniform_(0.5, 1.5)
    return block.to(torch.bfloat16).eval()


def real_block(prefix: str) -> Block:
    """One block of the BASE checkpoint, read key by key with ``safe_open`` (the file is 11 GB; never load it whole).
    The block weights are identical in layout for the distilled checkpoint; base is the first milestone target."""
    snapshot = variant_checkpoint_dir("base", required=("model.safetensors",))
    if snapshot is None:
        pytest.skip("FLUX 3 Klein base checkpoint not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    from safetensors import safe_open

    block = Block(HIDDEN, HEADS, MLP_RATIO)
    assert tuple(block.state_dict()) == BASE_BLOCK_KEYS
    state = {}
    t0 = time.time()
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        for key in BASE_BLOCK_KEYS:
            state[key] = f.get_tensor(f"{prefix}.{key}")
    block.load_state_dict(state, strict=True)
    logger.info(f"loaded {prefix}.* ({len(state)} tensors) from {snapshot.name} in {time.time() - t0:.1f}s")
    return block.to(torch.bfloat16).eval()


def reference_forward(block: Block, x: torch.Tensor, ids: torch.Tensor, bounds, mod) -> torch.Tensor:
    """The oracle: ``Block`` in bf16 under inference_mode + CPU bf16 autocast; fp32 copy of the output."""
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = block(x, rope_embed(ids[None].float()), bounds, mod).float()
    logger.info(f"reference forward L={x.shape[0]} bounds={bounds} in {time.time() - t0:.1f}s")
    return out.clone()


def reference_attention(block: Block, x: torch.Tensor, ids: torch.Tensor, mod) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_mod, attn)``: the modulated LN input (bf16) and the attention branch output ``[L, H*D]`` before
    ``attn_out`` (``dit.py:155-159``), under the same autocast as the block oracle."""
    shift, scale, _gate = mod
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        x_mod = layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], HEADS, -1).transpose(1, 2)  # noqa: E731
        q, k = block.norm(split(block.q_proj(x_mod)), split(block.k_proj(x_mod)), rope_embed(ids[None].float()))
        v = split(block.v_proj(x_mod))
        attn = packed_attention(q, k, v.to(q.dtype), [(0, x.shape[0])])
    return x_mod.to(torch.bfloat16), attn.float()


# --- host-side input preparation ------------------------------------------------------------------------------------


def mod_rows(seed: int, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(shift, scale, gate)``, each ``[count, D]`` bf16 ~ N(0, 1). ``scale`` is the raw reference scale."""
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(count, D, generator=g)).to(torch.bfloat16) for _ in range(3))


def scale_plus_one(scale: torch.Tensor) -> torch.Tensor:
    """The +1 folded on host in bf16: bit-identical to the reference's bf16 ``1.0 + scale``."""
    return (1.0 + scale.float()).to(torch.bfloat16)


def expand_runs(rows: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    """Row ``i`` of ``rows`` repeated ``lengths[i]`` times: the per-token expansion of a per-run modulation."""
    return torch.cat([rows[i].expand(n, -1) for i, n in enumerate(lengths) if n > 0])


def per_token_mod(rows, lengths, padded) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token ``(shift, scale_plus_one, gate)`` ``[padded, D]`` with the pad rows ``0 / 1 / 0``."""
    shift, scale, gate = (expand_runs(r, lengths) for r in rows)
    pad = padded - shift.shape[0]
    return (
        torch.nn.functional.pad(shift, (0, 0, 0, pad)),
        torch.nn.functional.pad(scale_plus_one(scale), (0, 0, 0, pad), value=1.0),
        torch.nn.functional.pad(gate, (0, 0, 0, pad)),
    )


def rope_tables(ids: torch.Tensor, padded: int, tail_ids: torch.Tensor | None = None):
    """``(cos, sin)`` ``[1, 1, padded, 128]`` fp32 for the fused RMSNorm+RoPE op (see the module docstring of the
    block for the conversion). Pad rows carry ``cos=1, sin=0`` unless ``tail_ids`` names their angles."""
    if tail_ids is not None:
        ids = torch.cat([ids, tail_ids])
    emb = rope_embed(ids[None].float())  # (1, 1, L, 64, 2, 2) with [[cos, -sin], [sin, cos]] per pair
    cos = emb[..., 0, 0].repeat_interleave(2, dim=-1)
    sin = emb[..., 1, 0].repeat_interleave(2, dim=-1)
    pad = padded - cos.shape[-2]
    assert pad >= 0
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad), value=1.0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad), value=0.0)
    return cos, sin


def loud_tail(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_tail, ids_tail)``: 8 * N(0, 1) bf16 rows and random RoPE ids."""
    g = torch.Generator().manual_seed(seed)
    x_tail = (LOUD_TAIL_SCALE * torch.randn(rows, D, generator=g)).to(torch.bfloat16)
    ids_tail = torch.randint(0, 64, (rows, 4), generator=g).float()
    return x_tail, ids_tail


def pad_zero(x: torch.Tensor, padded: int) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, 0, 0, padded - x.shape[0]))


@dataclass
class DeviceCase:
    """Everything one device forward needs, in the block's tensor contract."""

    x: torch.Tensor  # [L_pad, D] bf16
    cos: torch.Tensor  # [1, 1, L_pad, 128] fp32
    sin: torch.Tensor
    mod: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # [1, D] (broadcast) or [L_pad, D] (per token), +1 folded
    logical_length: int

    @property
    def padded(self) -> int:
        return self.x.shape[0]

    @property
    def per_token(self) -> bool:
        return self.mod[0].shape[0] != 1


def broadcast_case(x, ids, mod, padded, *, tail: tuple[torch.Tensor, torch.Tensor] | None = None) -> DeviceCase:
    """Early-stack usage: one modulation row; zero pad rows (production) or a loud tail (masking proof)."""
    shift, scale, gate = mod
    length = x.shape[0]
    if tail is None:
        x_pad, cos, sin = pad_zero(x, padded), *rope_tables(ids, padded)
    else:
        x_tail, ids_tail = tail
        assert x_tail.shape[0] == padded - length
        x_pad = torch.cat([x, x_tail])
        cos, sin = rope_tables(ids, padded, tail_ids=ids_tail)
    return DeviceCase(x_pad, cos, sin, (shift, scale_plus_one(scale), gate), length)


def per_token_case(x, ids, rows, lengths, padded) -> DeviceCase:
    """Single-stack usage: ``[image | image_cond | text]`` runs, per-token modulation, zero pad rows."""
    assert x.shape[0] == sum(lengths)
    cos, sin = rope_tables(ids, padded)
    return DeviceCase(pad_zero(x, padded), cos, sin, per_token_mod(rows, lengths, padded), x.shape[0])


# --- device ---------------------------------------------------------------------------------------------------------


def device_block(mesh: Mesh, torch_block: Block) -> Flux3KleinTransformerBlock:
    tt_block = Flux3KleinTransformerBlock(
        device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    t0 = time.time()
    tt_block.load_torch_state_dict(torch_block.state_dict(), strict=True)
    logger.info(f"block weights loaded strict in {time.time() - t0:.1f}s")
    return tt_block


def to_device(mesh: Mesh, case: DeviceCase):
    sp, tp = mesh.sp_axis, mesh.tp_axis
    assert case.padded == padded_length(case.padded, mesh.sp), f"{case.padded} rows do not fracture over SP={mesh.sp}"
    x = tensor.from_torch(case.x[None], device=mesh.device, mesh_axes=[None, sp, tp])
    rope = tuple(
        tensor.from_torch(t, device=mesh.device, dtype=ROPE_TABLE_DTYPE, mesh_axes=[None, None, sp, None])
        for t in (case.cos, case.sin)
    )
    mod_axes = [None, sp, tp] if case.per_token else [None, None, tp]
    mod = tuple(tensor.from_torch(t[None], device=mesh.device, mesh_axes=mod_axes) for t in case.mod)
    return x, rope, mod


def device_forward(mesh: Mesh, tt_block: Flux3KleinTransformerBlock, case: DeviceCase) -> torch.Tensor:
    """All ``L_pad`` output rows as fp32 ``[L_pad, D]`` (the caller slices the real rows)."""
    x, rope, mod = to_device(mesh, case)
    t0 = time.time()
    out = tt_block(x, rope=rope, mod=mod, logical_length=case.logical_length)
    ttnn.synchronize_device(mesh.device)
    logger.info(f"device forward L={case.logical_length} L_pad={case.padded} in {time.time() - t0:.2f}s")
    return tensor.to_torch(out, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0].float()


def device_attention(mesh: Mesh, tt_block: Flux3KleinTransformerBlock, case: DeviceCase) -> torch.Tensor:
    """The attention branch alone: ``case.x`` is the already-modulated input; concatenated heads ``[L_pad, D]``."""
    x, rope, _ = to_device(mesh, case)
    out, _ = tt_block.attn.forward(sequence_1=x, sequence_1_rope=rope, sequence_1_length=case.logical_length)
    return tensor.to_torch(out, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0].float()


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


def assert_finite(name: str, rows: torch.Tensor) -> None:
    assert torch.isfinite(rows).all(), f"{name}: non-finite values in {rows.shape}"
    logger.info(f"{name}: all {tuple(rows.shape)} rows finite, max_abs={rows.abs().max().item():.4f}")


# --- R0 / R1: early image block, broadcast modulation ---------------------------------------------------------------


@mesh_rows
@pytest.mark.parametrize("weights", ["random", "content_mode_blocks.image.0"], ids=["random", "real_image0"])
def test_early_image(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, weights: str) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block = random_block(seed=0) if weights == "random" else real_block(weights)
    torch.manual_seed(100)
    length = 4096
    x = torch.randn(length, D).to(torch.bfloat16)
    ids = image_ids(64, 64)
    mod = mod_rows(seed=110, count=1)
    reference = reference_forward(block, x, ids, [(0, length)], mod)

    tt_block = device_block(mesh, block)
    padded = padded_length(length, mesh.sp)
    got = device_forward(mesh, tt_block, broadcast_case(x, ids, mod, padded))
    check(
        f"early_image[{weights}] {mesh.tag} L={length}",
        reference,
        got[:length],
        pcc=BLOCK_PCC,
        relative_rmse=BLOCK_RMSE,
    )
    assert_finite("early_image output", got)


# --- R2: early text block, padded tail must not be attended ---------------------------------------------------------


@mesh_rows
@pytest.mark.parametrize("length", [80, 320], ids=["L80", "L320"])
def test_early_text_masked(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, length: int) -> None:
    """Text rows come in multiples of 80 and are all real; only the tile/SP alignment rows are phantom. Run A uses the
    production padding (``padded_length``: L itself only when L is already a multiple of 32 * SP, e.g. 320 at SP2,
    else the next multiple, e.g. 384 at SP4), run B appends a loud 512-row tail. The mask proof is each run vs
    torch; the third check (loud tail vs production padding on the real rows) bounds the tail's influence by the
    production padding's own noise. Every padded row must stay finite (broadcast modulation does not zero-gate
    the pad rows; see the block's module docstring)."""
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block = real_block("txt_mode_blocks.0")
    torch.manual_seed(102 + length)
    x = torch.randn(length, D).to(torch.bfloat16)
    ids = text_ids(length).float()
    mod = mod_rows(seed=120 + length, count=1)
    reference = reference_forward(block, x, ids, [(0, length)], mod)

    tt_block = device_block(mesh, block)
    padded_a = padded_length(length, mesh.sp)
    padded_b = padded_a + LOUD_TAIL_ROWS
    got_a = device_forward(mesh, tt_block, broadcast_case(x, ids, mod, padded_a))
    got_b = device_forward(
        mesh, tt_block, broadcast_case(x, ids, mod, padded_b, tail=loud_tail(padded_b - length, seed=999))
    )
    name = f"early_text L={length} {mesh.tag}"
    check(f"{name} pad->{padded_a} vs torch", reference, got_a[:length], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
    check(
        f"{name} pad->{padded_b} loud tail vs torch", reference, got_b[:length], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE
    )
    check(
        f"{name} real rows: loud tail pad->{padded_b} vs production pad->{padded_a}",
        got_a[:length],
        got_b[:length],
        pcc=MASK_PCC,
        relative_rmse=MASK_RMSE,
    )
    assert_finite(f"{name} run A all rows", got_a)
    assert_finite(f"{name} run B all rows", got_b)


# --- R3: attention branch alone (RoPE / QK-norm / mask proof) --------------------------------------------------------


@mesh_rows
def test_attention_branch(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    """``Attention.forward`` on the torch-modulated input vs ``packed_attention(*QKNorm(q, k, freqs), v)``. The block
    gate measured 0.99992-0.99997 for the RoPE / QK-norm / mask bugs listed at ``ATTN_PCC``; this row sees them."""
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block = real_block("txt_mode_blocks.0")
    torch.manual_seed(103)
    length = 320
    x = torch.randn(length, D).to(torch.bfloat16)
    ids = text_ids(length).float()
    mod = mod_rows(seed=130, count=1)
    x_mod, reference = reference_attention(block, x, ids, mod)

    tt_block = device_block(mesh, block)
    padded_a = padded_length(length, mesh.sp)
    padded_b = padded_a + LOUD_TAIL_ROWS
    got_a = device_attention(mesh, tt_block, broadcast_case(x_mod, ids, mod, padded_a))
    got_b = device_attention(
        mesh, tt_block, broadcast_case(x_mod, ids, mod, padded_b, tail=loud_tail(padded_b - length, seed=998))
    )
    name = f"attention_branch L={length} {mesh.tag}"
    check(f"{name} pad->{padded_a} vs torch", reference, got_a[:length], pcc=ATTN_PCC, relative_rmse=ATTN_RMSE)
    check(
        f"{name} pad->{padded_b} loud tail vs torch", reference, got_b[:length], pcc=ATTN_PCC, relative_rmse=ATTN_RMSE
    )
    check(
        f"{name} real rows: loud tail pad->{padded_b} vs production pad->{padded_a}",
        got_a[:length],
        got_b[:length],
        pcc=ATTN_MASK_PCC,
        relative_rmse=ATTN_MASK_RMSE,
    )


# --- R4: single block, per-token three-run modulation ----------------------------------------------------------------


SINGLE_LAYOUTS = {
    # (image grid, image_cond grid, text length): L = 4096 + 80 + 320 = 4496 is the canonical 1024x1024 case with
    # one 128x160 reference and a 320-token prompt; L = 32 + 16 + 32 = 80 pads to 128 rows, 32 per device at SP4.
    "L4496": ((64, 64), (8, 10), 320),
    "L80": ((4, 8), (4, 4), 32),
}


@mesh_rows
@pytest.mark.parametrize("layout", list(SINGLE_LAYOUTS), ids=list(SINGLE_LAYOUTS))
def test_single_pertoken(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, layout: str) -> None:
    """Three distinct (shift, scale, gate) rows expanded over ``[image | image_cond | text]``; pad rows 0 / 1 / 0 keep a
    zero input row exactly zero. Bug: gate rows shifted by one run measures 0.973 / 0.231 (host)."""
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    (img_h, img_w), (ref_h, ref_w), text_length = SINGLE_LAYOUTS[layout]
    block = real_block("single_blocks.0")
    torch.manual_seed(104)
    lengths = (img_h * img_w, ref_h * ref_w, text_length)
    length = sum(lengths)
    x = torch.randn(length, D).to(torch.bfloat16)
    ids = torch.cat([image_ids(img_h, img_w), reference_ids(ref_h, ref_w, 0), text_ids(text_length).float()])
    rows = mod_rows(seed=140, count=3)  # image, image_cond, text
    reference = reference_forward(block, x, ids, [(0, length)], tuple(expand_runs(r, lengths) for r in rows))

    tt_block = device_block(mesh, block)
    padded = padded_length(length, mesh.sp)
    got = device_forward(mesh, tt_block, per_token_case(x, ids, rows, lengths, padded))
    name = f"single_pertoken[{layout}] {mesh.tag} L={length} pad->{padded}"
    check(name, reference, got[:length], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
    start = 0
    for segment, n in zip(("image", "image_cond", "text"), lengths):
        check(
            f"{name} segment {segment}",
            reference[start : start + n],
            got[start : start + n],
            pcc=BLOCK_PCC,
            relative_rmse=BLOCK_RMSE,
        )
        start += n
    tail = got[length:]
    assert torch.equal(tail, torch.zeros_like(tail)), f"pad rows not exactly zero: max_abs={tail.abs().max().item()}"
    logger.info(f"{name}: {tail.shape[0]} pad rows exactly zero")


# --- R5: CFG pair, two samples that never attend to each other -------------------------------------------------------


@mesh_rows
def test_cfg_pair(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    """Sample 1 = image 4096 + text 320, sample 2 = image 4096 + image_cond 80, packed in ONE reference call with
    ``bounds=[(0, L1), (L1, L1 + L2)]``; the device runs two forwards on the same block instance. A packed
    single-sequence device run without a block-diagonal mask would land in the 0.69 class at attention level."""
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block = real_block("single_blocks.0")
    torch.manual_seed(105)
    lengths_1, lengths_2 = (4096, 0, 320), (4096, 80, 0)
    l1, l2 = sum(lengths_1), sum(lengths_2)
    x = torch.randn(l1 + l2, D).to(torch.bfloat16)
    ids_1 = torch.cat([image_ids(64, 64), text_ids(320).float()])
    ids_2 = torch.cat([image_ids(64, 64), reference_ids(8, 10, 0)])
    rows = mod_rows(seed=150, count=3)  # image (shared by both samples), image_cond, text
    mod = tuple(torch.cat([expand_runs(r, lengths_1), expand_runs(r, lengths_2)]) for r in rows)
    reference = reference_forward(block, x, torch.cat([ids_1, ids_2]), [(0, l1), (l1, l1 + l2)], mod)

    tt_block = device_block(mesh, block)
    got_1 = device_forward(mesh, tt_block, per_token_case(x[:l1], ids_1, rows, lengths_1, padded_length(l1, mesh.sp)))
    got_2 = device_forward(mesh, tt_block, per_token_case(x[l1:], ids_2, rows, lengths_2, padded_length(l2, mesh.sp)))
    name = f"cfg_pair {mesh.tag}"
    check(f"{name} sample 1 L1={l1}", reference[:l1], got_1[:l1], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
    check(f"{name} sample 2 L2={l2}", reference[l1:], got_2[:l2], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
    check(f"{name} both", reference, torch.cat([got_1[:l1], got_2[:l2]]), pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)


# --- R6: strict loading reports missing / unexpected reference keys -------------------------------------------------


@mesh_rows
def test_strict_load_negative(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, expect_error) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block = random_block(seed=0)

    missing = dict(block.state_dict())
    del missing["norm.key_norm.scale"]
    tt_block = Flux3KleinTransformerBlock(
        device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    with expect_error(ValueError, "missing Torch state keys"):
        tt_block.load_torch_state_dict(missing, strict=True)

    unexpected = dict(block.state_dict())
    unexpected["extra.weight"] = torch.zeros(D, dtype=torch.bfloat16)
    tt_block = Flux3KleinTransformerBlock(
        device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    with expect_error(ValueError, "unexpected Torch state keys"):
        tt_block.load_torch_state_dict(unexpected, strict=True)

    # And the positive control: the unmodified 8-key reference state loads strictly into the 5 fused parameters.
    tt_block = device_block(mesh, block)
    loaded = sorted(name for name, _ in _named_parameters_recursive(tt_block))
    assert loaded == [
        "attn.norm_k.weight",
        "attn.norm_q.weight",
        "attn.to_qkv.weight",
        "proj_mlp.weight",
        "proj_out.weight",
    ], loaded


def _named_parameters_recursive(module, prefix: str = ""):
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter
    for name, child in module.named_children():
        yield from _named_parameters_recursive(child, f"{prefix}{name}.")
