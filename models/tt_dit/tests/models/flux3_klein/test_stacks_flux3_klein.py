# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M2: the FLUX 3 Image Klein early (per-mode) stacks and the single-stage assembly on device.

Oracle (as in M1): the vendored ``reference/flux3_klein/dit.py::Block`` in bf16 under ``torch.inference_mode()`` +
``torch.autocast("cpu", dtype=torch.bfloat16)`` (SDPBackend.MATH), run per mode the way ``DiT.forward`` runs its
early loop (``dit.py:211-214``: one mode's blocks on that mode's rows with ``bounds=[(0, L_m)]`` and ONE broadcast
modulation row per mode). The device side gets the modulation PER TOKEN (the mode's row on the real rows, ``(0, 1, 0)``
on the alignment rows) built on host with M1's ``per_token_mod``, so this file does not depend on the modulation
heads of M3; the early-stack contract is in ``stacks_flux3_klein.py``'s module docstring.

Rows (each on the (2,2) Linear submesh for iteration and the (4,8) Ring with 8 KB packets as the gate; the mesh
handling, oracle, random-weight recipe and ``check`` wrapper are imported from the M1 test):
  S1 early_real_depth1    ``Flux3KleinEarlyStacks(depth=1)`` strict-loaded from the 24 reference keys of the BASE
                          checkpoint; image 4096, image_cond in {0 (empty), 4096}, text in {80, 240, 320, 560}; per
                          mode vs torch on the real rows; alignment tails exactly 0; the image/text outputs of the
                          empty-cond run bit-equal to the cond-4096 run's (the phantom-independence proof on device).
  S2 early_random_depth2  two chained random blocks per mode (N(0, 0.02) linears, QK scales U(0.5, 1.5)); image 1024,
                          cond 80, text 80 (32 rows per device at SP4, tails on two devices).
  S3 assembly_exact       ``assemble_single_stage`` of zero-padded random parts == the host concatenation of the real
                          rows (``torch.equal``) for (4096, 0, 320), (4096, 80, 80), (3920, 80, 320), (4096, 8192, 240);
                          ``gather_rows_over_sp`` with ``image_out_indices`` exact; then one random block on the
                          assembled (4096, 80, 80) sequence with per-token three-run modulation vs torch on the
                          host-concatenated rows (the judge's probe as a test row), tail exactly 0.
  S4 strict_load_negative missing / unexpected reference keys through the stacks' rename + the block's fusion.

Gate per mode / per sequence: ``assert_quality(pcc >= 0.999, relative_rmse <= 0.032)`` = M1's block gate (``BLOCK_PCC``
/ ``BLOCK_RMSE`` there, with the worst-plausible-bug calibration). Measured floors: one real block 0.99996-0.99997 /
0.0106-0.0131 (M1 R1/R2); two chained random blocks 0.99992-0.99994 / 0.0133-0.0154 (design probe B,
``logs/design23_B_probe_*``); the assembled (4096, 80, 80) sequence through one random block 0.999958 / 0.0133 on the
(2,2) submesh and 0.999954 / 0.0113 on the (4,8) ring (judge probe, ``logs/judge23_probe_*``), i.e. the M1 random
block floor. The numbers each run measured are logged as ``RESULT ...`` lines.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ....models.transformers.flux3_klein.layout_flux3_klein import Flux3KleinSequenceLayout
from ....models.transformers.flux3_klein.stacks_flux3_klein import (
    Flux3KleinEarlyStacks,
    assemble_single_stage,
    gather_rows_over_sp,
    indices_to_device,
)
from ....models.transformers.flux3_klein.transformer_block_flux3_klein import Flux3KleinTransformerBlock
from ....pipelines.flux3_klein.packing import (
    FLUX3_KLEIN_LATENT_CHANNELS,
    FLUX3_KLEIN_MODES,
    image_ids,
    padded_length,
    reference_ids,
    text_ids,
)
from ....reference.flux3_klein.dit import HEADS, HIDDEN, MLP_RATIO, Block, rope_embed
from ....utils import tensor
from .common import variant_checkpoint_dir
from .test_transformer_block_flux3_klein import (
    BASE_BLOCK_KEYS,
    BLOCK_PCC,
    BLOCK_RMSE,
    Mesh,
    assert_finite,
    check,
    expand_runs,
    mesh_rows,
    mod_rows,
    open_mesh,
    pad_zero,
    per_token_mod,
    random_block,
    reference_forward,
    rope_tables,
)

D = HIDDEN

# Reference state prefixes of the three early stacks (dit.py:172-176).
REFERENCE_PREFIXES = {
    "image": "content_mode_blocks.image",
    "image_cond": "content_mode_blocks.image_cond",
    "text": "txt_mode_blocks",
}
TEXT_LENGTHS = (80, 240, 320, 560)
IMAGE_LENGTH = 4096  # 64 x 64 (1024 x 1024)
# S3 shapes (image, image_cond, text): T2I 320, the unconditional prompt with a small reference, a 896x1120 canvas
# with a 320 prompt, two 1024 references with a 240 prompt.
ASSEMBLY_SHAPES = ((4096, 0, 320), (4096, 80, 80), (3920, 80, 320), (4096, 8192, 240))
BLOCK_SHAPE = (4096, 80, 80)


# --- reference state -------------------------------------------------------------------------------------------------


def real_early_state(depth: int = 1) -> dict[str, torch.Tensor]:
    """The reference-named keys of the first ``depth`` blocks of the three early stacks of the BASE checkpoint, read
    key by key with ``safe_open`` (the file is 11 GB; never load it whole)."""
    snapshot = variant_checkpoint_dir("base", required=("model.safetensors",))
    if snapshot is None:
        pytest.skip("FLUX 3 Klein base checkpoint not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    from safetensors import safe_open

    state = {}
    t0 = time.time()
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        for prefix in REFERENCE_PREFIXES.values():
            for i in range(depth):
                for key in BASE_BLOCK_KEYS:
                    state[f"{prefix}.{i}.{key}"] = f.get_tensor(f"{prefix}.{i}.{key}")
    logger.info(f"loaded {len(state)} early-stack tensors from {snapshot.name} in {time.time() - t0:.1f}s")
    return state


def random_early_state(depth: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, list[Block]]]:
    """``(reference-named state, torch blocks per mode)`` from ``random_block`` (distinct seeds per block)."""
    state, blocks = {}, {}
    for k, (mode, prefix) in enumerate(REFERENCE_PREFIXES.items()):
        blocks[mode] = [random_block(seed + 10 * k + i) for i in range(depth)]
        for i, block in enumerate(blocks[mode]):
            for key, value in block.state_dict().items():
                state[f"{prefix}.{i}.{key}"] = value
    return state, blocks


def reference_blocks(state: dict[str, torch.Tensor], depth: int) -> dict[str, list[Block]]:
    """Torch ``Block``s per mode from a reference-named state (bf16, eval)."""
    blocks = {}
    for mode, prefix in REFERENCE_PREFIXES.items():
        blocks[mode] = []
        for i in range(depth):
            block = Block(HIDDEN, HEADS, MLP_RATIO)
            block.load_state_dict({key: state[f"{prefix}.{i}.{key}"] for key in BASE_BLOCK_KEYS}, strict=True)
            blocks[mode].append(block.to(torch.bfloat16).eval())
    return blocks


def reference_stack(blocks: list[Block], x: torch.Tensor, ids: torch.Tensor, mod) -> torch.Tensor:
    """The early loop of ``DiT.forward`` for one mode (``dit.py:211-214``): the mode's blocks chained in bf16 under
    inference_mode + CPU bf16 autocast with ``bounds=[(0, L_m)]`` and the broadcast ``mod``; fp32 copy."""
    if len(blocks) == 1:
        return reference_forward(blocks[0], x, ids, [(0, x.shape[0])], mod)
    t0 = time.time()
    freqs = rope_embed(ids[None].float())
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for block in blocks:
            x = block(x, freqs, [(0, x.shape[0])], mod)
        out = x.float()
    logger.info(f"reference stack of {len(blocks)} blocks L={x.shape[0]} in {time.time() - t0:.1f}s")
    return out.clone()


# --- device ----------------------------------------------------------------------------------------------------------


def device_stacks(mesh: Mesh, state: dict[str, torch.Tensor], depth: int) -> Flux3KleinEarlyStacks:
    stacks = Flux3KleinEarlyStacks(
        depth=depth, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    t0 = time.time()
    stacks.load_torch_state_dict(state, strict=True)
    logger.info(f"early stacks depth={depth} loaded strict ({len(state)} reference keys) in {time.time() - t0:.1f}s")
    return stacks


def upload_hidden(mesh: Mesh, x: torch.Tensor) -> ttnn.Tensor:
    """``[L_pad, C]`` bf16 -> ``[1, L_pad, C]`` ``mesh_axes=[None, sp, tp]``."""
    assert x.shape[0] == padded_length(x.shape[0], mesh.sp), f"{x.shape[0]} rows do not fracture over SP={mesh.sp}"
    return tensor.from_torch(x[None], device=mesh.device, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])


def download_hidden(mesh: Mesh, x: ttnn.Tensor) -> torch.Tensor:
    return tensor.to_torch(x, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0].float()


def upload_per_token(mesh: Mesh, ids: torch.Tensor, rows, lengths: tuple[int, ...], padded: int):
    """``(rope, mod)`` of the block contract for a sequence of runs ``lengths`` (per-token modulation from one
    ``(shift, scale, gate)`` row per run, pad rows ``0 / 1 / 0``; RoPE pad rows cos 1 / sin 0)."""
    cos, sin = rope_tables(ids, padded)
    rope = tuple(
        tensor.from_torch(t, device=mesh.device, dtype=ttnn.float32, mesh_axes=[None, None, mesh.sp_axis, None])
        for t in (cos, sin)
    )
    mod = tuple(
        tensor.from_torch(t[None], device=mesh.device, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])
        for t in per_token_mod(rows, lengths, padded)
    )
    return rope, mod


def run_stack(mesh: Mesh, stacks: Flux3KleinEarlyStacks, mode: str, x: torch.Tensor, ids: torch.Tensor, rows):
    """One mode through its stack: ``x`` ``[L_m, D]`` bf16 real rows, ``rows`` one ``(shift, scale, gate)`` row each
    ``[1, D]``. Returns the device output ``[1, L_m_pad, D]`` (``[None, sp, tp]``); the inputs are deallocated. The
    caller downloads with :func:`download_hidden` AFTER the other modes of the pass have run, the production order,
    which is what catches a block output aliasing a CCL ping-pong buffer (see the stacks' ``forward``)."""
    length = x.shape[0]
    padded = padded_length(length, mesh.sp)
    x_dev = upload_hidden(mesh, pad_zero(x, padded))
    rope, mod = upload_per_token(mesh, ids, rows, (length,), padded)
    t0 = time.time()
    out = stacks(mode, x_dev, rope=rope, mod=mod, logical_length=length)
    ttnn.synchronize_device(mesh.device)
    logger.info(f"device {mode} stack L={length} L_pad={padded} in {time.time() - t0:.2f}s")
    for t in (x_dev, *rope, *mod):
        ttnn.deallocate(t)
    return out


def download_all(mesh: Mesh, outputs: dict[str, ttnn.Tensor]) -> dict[str, torch.Tensor]:
    """fp32 ``[L_m_pad, D]`` per key; the device tensors are deallocated (they are the caller's, by contract)."""
    got = {}
    for key, out in outputs.items():
        got[key] = download_hidden(mesh, out)
        ttnn.deallocate(out)
    return got


def assert_tail_zero(name: str, got: torch.Tensor, length: int) -> None:
    tail = got[length:]
    assert torch.equal(tail, torch.zeros_like(tail)), f"{name}: pad rows not exactly zero, max_abs={tail.abs().max()}"
    logger.info(f"{name}: {tail.shape[0]} pad rows exactly zero")


def assert_bit_equal(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.equal(a, b), f"{name}: not bit-equal, max_abs={(a - b).abs().max().item()}"
    logger.info(f"RESULT {name}: bit-equal ({tuple(a.shape)})")


# --- S1: real weights, depth 1, the per-mode early loop ----------------------------------------------------------------


@mesh_rows
def test_early_real_depth1(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    """Cases loop inside (the mesh fixture is function-scoped and the weights load once): image 4096 x cond in
    {4096, 0} x text in {80, 240, 320, 560}. The image and cond references are text-length independent and
    computed once. The cond-0 pass runs no cond stack; its image and text outputs must be bit-equal to the
    cond-4096 pass (same inputs), which is the device statement of the reference's phantom rule."""
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    state = real_early_state(depth=1)
    blocks = reference_blocks(state, depth=1)
    stacks = device_stacks(mesh, state, depth=1)

    torch.manual_seed(200)
    x = {
        "image": torch.randn(IMAGE_LENGTH, D).to(torch.bfloat16),
        "image_cond": torch.randn(IMAGE_LENGTH, D).to(torch.bfloat16),
        **{f"text{n}": torch.randn(n, D).to(torch.bfloat16) for n in TEXT_LENGTHS},
    }
    ids = {
        "image": image_ids(64, 64),
        "image_cond": reference_ids(64, 64, 0),
        **{f"text{n}": text_ids(n).float() for n in TEXT_LENGTHS},
    }
    rows = {mode: mod_rows(seed=210 + k, count=1) for k, mode in enumerate(FLUX3_KLEIN_MODES)}

    reference = {
        "image": reference_stack(blocks["image"], x["image"], ids["image"], rows["image"]),
        "image_cond": reference_stack(blocks["image_cond"], x["image_cond"], ids["image_cond"], rows["image_cond"]),
        **{
            f"text{n}": reference_stack(blocks["text"], x[f"text{n}"], ids[f"text{n}"], rows["text"])
            for n in TEXT_LENGTHS
        },
    }

    outputs = {}
    for cond_length in (IMAGE_LENGTH, 0):
        tag = f"early_real_depth1 {mesh.tag} cond={cond_length}"
        on_device = {"image": run_stack(mesh, stacks, "image", x["image"], ids["image"], rows["image"])}
        if cond_length:
            on_device["image_cond"] = run_stack(
                mesh, stacks, "image_cond", x["image_cond"], ids["image_cond"], rows["image_cond"]
            )
        for n in TEXT_LENGTHS:
            on_device[f"text{n}"] = run_stack(mesh, stacks, "text", x[f"text{n}"], ids[f"text{n}"], rows["text"])
        got = download_all(mesh, on_device)
        for key, out in got.items():
            length = x[key].shape[0]
            check(f"{tag} {key} L={length}", reference[key], out[:length], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
            assert_tail_zero(f"{tag} {key}", out, length)
            assert_finite(f"{tag} {key} all rows", out)
        outputs[cond_length] = got

    for key in ("image", *(f"text{n}" for n in TEXT_LENGTHS)):
        assert_bit_equal(
            f"early_real_depth1 {mesh.tag} {key}: cond 0 vs cond 4096", outputs[0][key], outputs[IMAGE_LENGTH][key]
        )


# --- S2: random weights, depth 2 -----------------------------------------------------------------------------------------


@mesh_rows
def test_early_random_depth2(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    state, blocks = random_early_state(depth=2, seed=300)
    stacks = device_stacks(mesh, state, depth=2)

    torch.manual_seed(301)
    lengths = {"image": 32 * 32, "image_cond": 8 * 10, "text": 80}
    ids = {"image": image_ids(32, 32), "image_cond": reference_ids(8, 10, 0), "text": text_ids(80).float()}
    x = {mode: torch.randn(n, D).to(torch.bfloat16) for mode, n in lengths.items()}
    rows = {mode: mod_rows(seed=310 + k, count=1) for k, mode in enumerate(FLUX3_KLEIN_MODES)}

    # Production order: all three modes run before anything is read back (cond 80 and text 80 share a shard shape).
    got = download_all(
        mesh, {mode: run_stack(mesh, stacks, mode, x[mode], ids[mode], rows[mode]) for mode in FLUX3_KLEIN_MODES}
    )
    for mode in FLUX3_KLEIN_MODES:
        reference = reference_stack(blocks[mode], x[mode], ids[mode], rows[mode])
        name = f"early_random_depth2 {mesh.tag} {mode} L={lengths[mode]}"
        check(name, reference, got[mode][: lengths[mode]], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
        assert_tail_zero(name, got[mode], lengths[mode])
        assert_finite(f"{name} all rows", got[mode])


# --- S3: single-stage assembly ---------------------------------------------------------------------------------------------


def zero_padded_parts(layout: Flux3KleinSequenceLayout, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    parts = {}
    for m in layout.modes:
        x = torch.randn(layout.lengths[m], D, generator=g).to(torch.bfloat16)
        parts[m] = pad_zero(x, layout.padded[m])
    return parts


@mesh_rows
def test_assembly_exact(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    block_inputs = None
    for shape in ASSEMBLY_SHAPES:
        layout = Flux3KleinSequenceLayout.from_lengths(dict(zip(FLUX3_KLEIN_MODES, shape)), mesh.sp)
        name = f"assembly {mesh.tag} {shape} L_real={layout.total} L_pad={layout.total_padded} R={layout.rows_total}"
        host = zero_padded_parts(layout, seed=sum(shape))
        parts = [upload_hidden(mesh, host[m]) for m in layout.modes]
        expected = F.pad(
            torch.cat([host[m][: layout.lengths[m]] for m in layout.modes]),
            (0, 0, 0, layout.total_padded - layout.total),
        )

        indices = indices_to_device(layout.assembly_indices(), device=mesh.device, sp_axis=mesh.sp_axis)
        t0 = time.time()
        assembled = assemble_single_stage(parts, indices, ccl_manager=mesh.ccl, sp_axis=mesh.sp_axis)
        ttnn.synchronize_device(mesh.device)
        dt = time.time() - t0
        got = tensor.to_torch(assembled, mesh_axes=[None, mesh.sp_axis, mesh.tp_axis])[0]
        assert got.shape == expected.shape, (got.shape, expected.shape)
        assert torch.equal(
            got, expected
        ), f"{name}: assembled sequence differs, max_abs={(got.float() - expected.float()).abs().max()}"
        logger.info(f"RESULT {name}: assembled {tuple(assembled.shape)} exact in {dt:.3f}s")

        # Reverse re-layout of a 128-wide G-order output into the image stack's own SP layout.
        g = torch.Generator().manual_seed(sum(shape) + 1)
        velocity = torch.randn(layout.total_padded, FLUX3_KLEIN_LATENT_CHANNELS, generator=g).to(torch.bfloat16)
        velocity_dev = tensor.from_torch(velocity[None], device=mesh.device, mesh_axes=[None, mesh.sp_axis, None])
        out_indices = indices_to_device(layout.image_out_indices(), device=mesh.device, sp_axis=mesh.sp_axis)
        t0 = time.time()
        image_out = gather_rows_over_sp(velocity_dev, out_indices, ccl_manager=mesh.ccl, sp_axis=mesh.sp_axis)
        ttnn.synchronize_device(mesh.device)
        dt = time.time() - t0
        got_v = tensor.to_torch(image_out, mesh_axes=[None, mesh.sp_axis, None])[0]
        length = layout.lengths["image"]
        assert got_v.shape == (layout.padded["image"], FLUX3_KLEIN_LATENT_CHANNELS), got_v.shape
        assert torch.equal(got_v[:length], velocity[:length]), f"{name}: image-layout rows differ"
        logger.info(f"RESULT {name}: reverse re-layout to {tuple(got_v.shape)} exact in {dt:.3f}s")

        if shape == BLOCK_SHAPE:
            block_inputs = (layout, host, assembled)
        else:
            ttnn.deallocate(assembled)
        for t in (*parts, velocity_dev, image_out, indices, out_indices):
            ttnn.deallocate(t)

    # One random block on the assembled (4096, 80, 80) sequence with per-token three-run modulation in G order vs
    # torch on the host-concatenated real rows (the judge's probe; floors in the module docstring).
    layout, host, assembled = block_inputs
    block = random_block(seed=11)
    tt_block = Flux3KleinTransformerBlock(
        device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    tt_block.load_torch_state_dict(block.state_dict(), strict=True)
    lengths = tuple(layout.lengths[m] for m in FLUX3_KLEIN_MODES)
    ids = torch.cat([image_ids(64, 64), reference_ids(8, 10, 0), text_ids(lengths[2]).float()])
    assert ids.shape[0] == layout.total
    x_real = torch.cat([host[m][: layout.lengths[m]] for m in layout.modes])
    rows = mod_rows(seed=5, count=3)  # image, image_cond, text
    reference = reference_forward(block, x_real, ids, [(0, layout.total)], tuple(expand_runs(r, lengths) for r in rows))
    rope, mod = upload_per_token(mesh, ids, rows, lengths, layout.total_padded)
    t0 = time.time()
    out = tt_block(assembled, rope=rope, mod=mod, logical_length=layout.total)
    ttnn.synchronize_device(mesh.device)
    logger.info(
        f"block on the assembled sequence L={layout.total} L_pad={layout.total_padded} in {time.time() - t0:.2f}s"
    )
    got = download_hidden(mesh, out)
    name = f"assembly+block {mesh.tag} {BLOCK_SHAPE} L_real={layout.total} L_pad={layout.total_padded}"
    check(name, reference, got[: layout.total], pcc=BLOCK_PCC, relative_rmse=BLOCK_RMSE)
    start = 0
    for mode, n in zip(FLUX3_KLEIN_MODES, lengths):
        check(
            f"{name} segment {mode}",
            reference[start : start + n],
            got[start : start + n],
            pcc=BLOCK_PCC,
            relative_rmse=BLOCK_RMSE,
        )
        start += n
    assert_tail_zero(name, got, layout.total)
    assert_finite(f"{name} all rows", got)


# --- S4: strict loading reports missing / unexpected reference keys ----------------------------------------------------------


def parameter_names(module, prefix: str = "") -> list[str]:
    names = [f"{prefix}{name}" for name, _ in module.named_parameters()]
    for name, child in module.named_children():
        names.extend(parameter_names(child, f"{prefix}{name}."))
    return sorted(names)


@mesh_rows
def test_strict_load_negative(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links, expect_error) -> None:
    mesh = open_mesh(mesh_device, submesh_shape, sp_axis, tp_axis, topology, num_links)
    state, _ = random_early_state(depth=1, seed=400)
    assert len(state) == 24

    missing = dict(state)
    del missing["txt_mode_blocks.0.norm.key_norm.scale"]
    stacks = Flux3KleinEarlyStacks(
        depth=1, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    with expect_error(ValueError, "missing Torch state keys"):
        stacks.load_torch_state_dict(missing, strict=True)

    unexpected = dict(state)
    unexpected["txt_mode_blocks.1.q_proj.weight"] = torch.zeros(D, D, dtype=torch.bfloat16)
    stacks = Flux3KleinEarlyStacks(
        depth=1, device=mesh.device, ccl_manager=mesh.ccl, parallel_config=mesh.parallel_config
    )
    with expect_error(ValueError, "unexpected Torch state keys"):
        stacks.load_torch_state_dict(unexpected, strict=True)

    # Positive control: the 24 reference keys load strictly into 3 x 5 fused parameters under the MODES names.
    stacks = device_stacks(mesh, state, depth=1)
    fused = ("attn.norm_k.weight", "attn.norm_q.weight", "attn.to_qkv.weight", "proj_mlp.weight", "proj_out.weight")
    assert parameter_names(stacks) == sorted(f"{mode}.0.{name}" for mode in FLUX3_KLEIN_MODES for name in fused)
