# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M6: the FLUX 3 Image Klein VAE decode on device (``pipelines/flux3_klein/vae_flux3_klein.py`` over
``models/vae/vae_flux2.py::Flux2VaeDecoder`` unchanged) against the vendored reference ``AutoEncoder.decode``
(fp32, ``ae.safetensors``) and the snapshot's own weight files. Design record ``docs/M6_M7_DESIGN.md`` section 3.

Host rows (no device):
  V1 test_ae_equals_vae            ``snap:ae.safetensors`` (BFL names) == ``snap:vae/diffusion_pytorch_model.safetensors``
                                   (diffusers names) tensor by tensor through :func:`map_bfl_name`: every float tensor
                                   ``torch.equal`` (the 8 attention 1x1 convs after a squeeze); the int64 training counter
                                   ``bn.num_batches_tracked`` is ignored by name. Measured 250 / 250 equal.
  V2 test_unpack_matches_reference  diffusers ``AutoencoderKLFlux2.decode`` fed through the TT unpack (inverse BatchNorm
                                   on the 128 token channels, ``reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)``)
                                   vs the reference ``AutoEncoder.decode`` on a random 256x256 latent: pcc >= UNPACK_PCC
                                   (measured 1.00000000 at 1024x1024, max |d| 4.4e-6 / 6.1e-6: fp32 op order only).
Device rows (``bh_4x8_ring`` = the (4, 8) ring parent, channels TP4 over axis 0, latent rows H8 over axis 1 = Flux2's
own 4x8 pipeline row; ``1x1`` = one device, no fabric, no TP, no H):
  V3 test_decode_golden_tokens[row, variant]  the golden ``final_tokens`` (base and distilled man_dog, 1024x1024)
                                   uploaded as the denoiser's fp32 state slot and decoded on device vs the CPU reference
                                   decode (cached under ``$FLUX3_KLEIN_VAE_ORACLE_CACHE``): ``assert_quality`` at
                                   Flux2's gate (pcc 0.9978 / rel_rmse 0.034); PSNR of the uint8 renders >= PSNR_DEC_FLOOR;
                                   traced == eager ``torch.equal``; replay == capture; then the trace captured on this
                                   variant's tokens is REPLAYED on the other variant's tokens and must ``torch.equal``
                                   that variant's eager decode (the tracer's input copy, not a stale capture input:
                                   pcc between the two renders is 0.35), and a replay back on the first tokens must
                                   equal the capture; TRACE / DRAM per bank logged.
  V4 test_decode_padded_layout[bh_4x8_ring]   a 1344(w)x768(h) random state (h 48, w 84: L 4032 -> L_pad 4096 at SP4, 64
                                   alignment rows; h % 8 == 0): the decoder INPUT of the sliced + H-partitioned path
                                   (``preprocess_and_unpatchify`` output, shards composed on the host) ``torch.equal``
                                   the same helper's replicated path on the same rows (slice and partition proven
                                   exactly, data movement only). Measured True, 64 alignment rows sliced.
  V4b test_decode_padded_size_full[bh_4x8_ring]  the full 1344x768 decode vs the CPU reference at Flux2's gate: KNOWN
                                   FAILURE in the shipped decoder's group-norm core-grid selection at this per-device
                                   shape (see the test docstring), carried as ``xfail(strict=True, raises=RuntimeError)``
                                   so the selector stays green and a decoder fix surfaces as XPASS; sizes other than
                                   1024x1024 are an M8/M10 item.
                                   The design's second (replicated-rows) decoder instance is NOT built: its decode would
                                   differ from the partitioned one in the cross-device GroupNorm reduction order, so a
                                   bit-equality of decoder OUTPUTS is not a property of the slice; the inputs are.

Oracle: ~11 s per 1024 image on the node CPU, 86-160 s on the login box (M6 design 2.2); the cache is a convenience.
Environment: ``FLUX3_KLEIN_MODEL_PATH`` / the HuggingFace cache (snapshot with ``ae.safetensors`` and ``vae/``),
``FLUX3_KLEIN_GOLDENS`` (V3), ``FLUX3_KLEIN_VAE_ORACLE_CACHE``, ``FLUX3_KLEIN_OUT`` (PNGs of the device renders,
never inside the tree), ``TT_DIT_CACHE_DIR``.
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

from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.pipeline_flux3_klein import to_pil
from ....pipelines.flux3_klein.vae_flux3_klein import (
    Flux3KleinVaeDecode,
    build_flux3_klein_vae_decoder,
    load_flux3_klein_vae_weights,
    vae_blob,
)
from ....reference.flux3_klein import autoencoder as ref_ae
from ....reference.flux3_klein import checkpoint as ref_ckpt
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    GOLDENS_ENV,
    GoldenCase,
    checkpoint_dir,
    golden_cases,
    load_golden_case,
    pcc,
    psnr_uint8,
    renders_dir,
    sha256_of,
    vae_oracle_cache,
)
from .device_params import TRACE_REGION_SIZE, ring_params_8k_flux3_klein_transformer, single_device_params_flux3_klein

# Flux2's decode gate (tests/models/flux2/test_vae_flux2.py:163, PLAN M6), kept unchanged. Measured on the golden
# final_tokens, 4x8 ring tp0/h1: pcc 0.999952 / rel_rmse 0.01133 (base), 0.999954 / 0.01096 (distilled)
# (logs/m67_designer_vae_decode_4x8.log, design 2.3).
DECODE_PCC = 0.9978
DECODE_RMSE = 0.034
# PSNR of the device render vs the CPU reference render of the SAME tokens, uint8 after the reference to_pil.
# Measured 51.26 / 51.97 dB (golden tokens, base / distilled) and 51.17 / 52.25 dB (the M4 device states) on the
# 4x8 ring; floor = 51.0 rounded down to 0.5 dB minus a 1.0 dB margin (design 3.3, same log).
PSNR_DEC_FLOOR = 50.0
# V2: measured pcc 1.00000000 (eight digits) at 1024x1024; fp32 op-order differences only.
UNPACK_PCC = 0.9999999
# V4 geometry: 1344 wide x 768 tall -> h 48, w 84, L 4032 (= 126 tiles) -> L_pad 4096 at SP4 (64 alignment rows), h % 8 == 0.
PADDED_SIZE = (768, 1344)  # (height, width): 768 tall, 1344 wide
ORACLE_THREADS = 16
IGNORED_KEYS = ("bn.num_batches_tracked",)


# --- snapshot --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot() -> Path:
    found = checkpoint_dir("distilled", required=("ae.safetensors", "vae/diffusion_pytorch_model.safetensors"))
    if found is None:
        pytest.skip("FLUX 3 Klein snapshot with ae.safetensors and vae/ not found (FLUX3_KLEIN_MODEL_PATH / HF cache)")
    return found


def map_bfl_name(key: str) -> str:
    """BFL ``autoencoder.py`` state key -> diffusers ``AutoencoderKLFlux2`` state key (design 2.2)."""
    if key.startswith("bn."):
        return key
    if key.startswith("decoder.post_quant_conv."):
        return key.removeprefix("decoder.")
    if key.startswith("encoder.quant_conv."):
        return key.removeprefix("encoder.")
    side, *rest = key.split(".")
    attn = {"q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0", "norm": "group_norm"}
    tail = lambda parts: ".".join(parts).replace("nin_shortcut", "conv_shortcut")  # noqa: E731
    if rest[0] == "norm_out":
        return f"{side}.conv_norm_out.{rest[1]}"
    if rest[0] == "mid":
        if rest[1] == "attn_1":
            return f"{side}.mid_block.attentions.0.{attn[rest[2]]}.{rest[3]}"
        return f"{side}.mid_block.resnets.{ {'block_1': 0, 'block_2': 1}[rest[1]] }.{tail(rest[2:])}"
    if rest[0] == "up":
        block = 3 - int(rest[1])  # diffusers up_blocks[0] is BFL up[3] (processed first)
        if rest[2] == "block":
            return f"decoder.up_blocks.{block}.resnets.{rest[3]}.{tail(rest[4:])}"
        if rest[2] == "upsample":
            return f"decoder.up_blocks.{block}.upsamplers.0.conv.{rest[4]}"
    if rest[0] == "down":
        if rest[2] == "block":
            return f"encoder.down_blocks.{rest[1]}.resnets.{rest[3]}.{tail(rest[4:])}"
        if rest[2] == "downsample":
            return f"encoder.down_blocks.{rest[1]}.downsamplers.0.conv.{rest[4]}"
    return key  # conv_in / conv_out


# --- V1 / V2: host --------------------------------------------------------------------------------------------------------------


def test_ae_equals_vae(snapshot: Path) -> None:
    from safetensors.torch import load_file

    ae = load_file(str(snapshot / "ae.safetensors"))
    vae = load_file(str(snapshot / "vae" / "diffusion_pytorch_model.safetensors"))
    mapped = {map_bfl_name(k): k for k in ae}
    assert len(mapped) == len(ae), "the name map is not injective"
    missing = sorted(set(vae) - set(mapped))
    extra = sorted(set(mapped) - set(vae))
    assert not missing and not extra, f"vae keys not covered {missing[:5]}; mapped keys absent {extra[:5]}"
    mismatches, compared, squeezed = [], 0, []
    for dk, ak in mapped.items():
        if dk in IGNORED_KEYS:
            assert not vae[dk].is_floating_point(), f"{dk} is not the int64 training counter"
            continue
        a, v = ae[ak], vae[dk]
        assert a.is_floating_point() and v.is_floating_point(), (ak, a.dtype, v.dtype)
        compared += 1
        if a.shape != v.shape:
            if a.numel() == v.numel() and torch.equal(a.reshape(v.shape), v):
                squeezed.append(dk)
                continue
            mismatches.append((ak, dk, tuple(a.shape), tuple(v.shape)))
        elif not torch.equal(a, v):
            mismatches.append((ak, dk, "values"))
    logger.info(
        f"RESULT ae == vae: {compared} float tensors compared, {len(squeezed)} equal after a shape squeeze "
        f"{sorted(squeezed)[:2]}..., ignored {list(IGNORED_KEYS)}; blob {vae_blob(snapshot)}"
    )
    assert not mismatches, mismatches[:5]
    assert compared == 250, compared


def test_unpack_matches_reference(snapshot: Path) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    torch.set_num_threads(ORACLE_THREADS)
    h = w = 16  # 256x256
    torch.manual_seed(0)
    tokens = torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    ae = reference_autoencoder(snapshot)
    with torch.inference_mode():
        reference = ae.decode(tokens, ref_ae.latent_ids(h, w))  # [3, H, W]
    dvae = AutoencoderKLFlux2.from_pretrained(str(snapshot), subfolder="vae").eval()
    assert dvae.config.latent_channels == 32 and list(dvae.config.patch_size) == [2, 2]
    assert dvae.bn.eps == 1e-4 and list(dvae.config.block_out_channels) == [128, 256, 512, 512]
    inp = tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2)
    s = (dvae.bn.running_var + dvae.bn.eps).sqrt()
    z = inp * s.view(1, -1, 1, 1) + dvae.bn.running_mean.view(1, -1, 1, 1)
    z = z.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
    with torch.inference_mode():
        got = dvae.decode(z).sample[0]
    p = pcc(reference, got)
    logger.info(
        f"RESULT unpack: diffusers decode through the TT unpack vs reference decode pcc {p:.8f} "
        f"max|d| {(reference - got).abs().max().item():.3e} equal {torch.equal(reference, got)}"
    )
    assert p >= UNPACK_PCC


# --- oracle ----------------------------------------------------------------------------------------------------------------------

_REFERENCE_AE: dict[Path, ref_ae.AutoEncoder] = {}


def reference_autoencoder(snapshot: Path) -> ref_ae.AutoEncoder:
    if snapshot not in _REFERENCE_AE:
        t0 = time.time()
        _REFERENCE_AE[snapshot] = ref_ckpt.load_checkpoint(
            ref_ae.AutoEncoder(), str(snapshot / "ae.safetensors"), "cpu", torch.float32
        )
        logger.info(f"reference AutoEncoder (fp32, ae.safetensors) loaded in {time.time() - t0:.1f}s")
    return _REFERENCE_AE[snapshot]


def cached_reference_decode(snapshot: Path, tokens: torch.Tensor, h: int, w: int, label: str) -> torch.Tensor:
    """``AutoEncoder.decode(tokens, latent_ids(h, w))`` fp32 ``[3, H, W]`` through ``$FLUX3_KLEIN_VAE_ORACLE_CACHE``."""
    from safetensors.torch import load_file, save_file

    blob = Path(os.path.realpath(snapshot / "ae.safetensors")).name[:16]
    fields = {"ae_blob": blob, "tokens_sha256": sha256_of(tokens), "h": h, "w": w, "torch": torch.__version__}
    digest = hashlib.sha256("|".join(str(fields[k]) for k in fields).encode()).hexdigest()
    root = vae_oracle_cache()
    if root is not None and (root / f"{digest}.safetensors").is_file():
        logger.info(f"ORACLE HIT {digest[:16]} ({label})")
        return load_file(str(root / f"{digest}.safetensors"))["image_f32"]
    logger.info(f"ORACLE MISS {digest[:16]} ({label}): CPU reference decode of {h}x{w} tokens")
    torch.set_num_threads(ORACLE_THREADS)
    t0 = time.time()
    with torch.inference_mode():
        image = reference_autoencoder(snapshot).decode(tokens.float(), ref_ae.latent_ids(h, w)).contiguous()
    logger.info(f"ORACLE {label}: CPU reference decode {time.time() - t0:.1f} s")
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        save_file({"image_f32": image}, str(root / f"{digest}.safetensors"))
        (root / f"{digest}.json").write_text(json.dumps({**fields, "label": label}, indent=1))
    return image


# --- device rows ----------------------------------------------------------------------------------------------------------------

vae_rows = pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, device_params",
    [
        [(4, 8), 0, 0, 1, ttnn.Topology.Ring, 2, ring_params_8k_flux3_klein_transformer],
        [
            (1, 1),
            None,
            None,
            None,
            None,
            0,
            {**single_device_params_flux3_klein, "trace_region_size": TRACE_REGION_SIZE},
        ],
    ],
    ids=["bh_4x8_ring", "1x1"],
    indirect=["mesh_device", "device_params"],
)


@dataclass
class VaeMesh:
    device: ttnn.MeshDevice
    sp_axis: int | None
    dit_ccl: CCLManager | None
    vae_ccl: CCLManager | None
    decoder: Flux2VaeDecoder
    helper: Flux3KleinVaeDecode
    tag: str


def open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot: Path) -> VaeMesh:
    shape = tuple(mesh_device.shape)
    multi = shape[0] * shape[1] > 1
    dit_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology) if multi else None
    vae_ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology) if multi else None
    decoder, parallel_config = build_flux3_klein_vae_decoder(
        mesh_device, tp_axis=tp_axis, h_axis=h_axis, ccl_manager=vae_ccl
    )
    t0 = time.time()
    state = load_flux3_klein_vae_weights(decoder, snapshot, parallel_config=parallel_config, mesh_device=mesh_device)
    tag = f"{shape[0]}x{shape[1]}_tp{tp_axis}_h{h_axis}"
    logger.info(f"RECORD {tag}: weights {state} in {time.time() - t0:.1f}s; {memory_line(mesh_device)}")
    helper = Flux3KleinVaeDecode(
        decoder,
        device=mesh_device,
        dit_ccl=dit_ccl,
        sp_axis=sp_axis,
        h_axis=h_axis,
        tp_axis=tp_axis,
        vae_ccl=vae_ccl,
        snapshot=snapshot,
    )
    return VaeMesh(mesh_device, sp_axis, dit_ccl, vae_ccl, decoder, helper, tag)


def memory_line(device) -> str:
    dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    return (
        f"DRAM {dram.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank; "
        f"TRACE {trace.total_bytes_allocated_per_bank} B per bank"
    )


def upload_state(mesh: VaeMesh, tokens: torch.Tensor) -> ttnn.Tensor:
    """``[L, 128]`` fp32 host tokens -> the denoiser's fp32 slot layout ``[1, L_pad, 128]`` (zero alignment rows)."""
    sp = int(mesh.device.shape[mesh.sp_axis]) if mesh.sp_axis is not None else 1
    padded = packing.pad_rows(tokens.float(), packing.padded_length(tokens.shape[0], sp))[0][None]
    return tensor.from_torch(
        padded,
        device=mesh.device,
        dtype=ttnn.float32,
        mesh_axes=[None, mesh.sp_axis, None] if mesh.sp_axis is not None else None,
    )


def device_decode(mesh: VaeMesh, tokens: torch.Tensor, height: int, width: int, *, traced: bool):
    """Returns ``([3, H, W] fp32 host image, wall)``."""
    h, w = packing.latent_shape(height, width)
    state = upload_state(mesh, tokens)
    ttnn.synchronize_device(mesh.device)
    t0 = time.time()
    out = mesh.helper.decode(state, num_image_rows=h * w, height=height, width=width, traced=traced)
    ttnn.synchronize_device(mesh.device)
    wall = time.time() - t0
    image = tensor.to_torch(out)[0].permute(2, 0, 1).float()
    if not traced:
        ttnn.deallocate(out)
    ttnn.deallocate(state)
    return image, wall


def golden_variants() -> list[tuple[str, Path]]:
    out = []
    for case_dir in golden_cases():
        record = case_dir / "record.json"
        if record.is_file():
            model = json.loads(record.read_text()).get("replay", {}).get("model")
            if model in ("base", "distilled") and not any(m == model for m, _ in out):
                out.append((model, case_dir))
    return out


GOLDEN_VARIANTS = golden_variants()


@vae_rows
@pytest.mark.parametrize(
    "variant, case_dir",
    GOLDEN_VARIANTS or [(None, None)],
    ids=[v for v, _ in GOLDEN_VARIANTS] or ["no_goldens"],
)
def test_decode_golden_tokens(
    mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot, variant: str | None, case_dir: Path | None
) -> None:
    if case_dir is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no case under it")
    golden: GoldenCase = load_golden_case(case_dir)
    tokens = golden.steps["final_tokens"]
    width, height = (int(v) for v in golden.replay["size"])
    h, w = packing.latent_shape(height, width)
    assert tokens.shape == (h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS)
    reference = cached_reference_decode(snapshot, tokens, h, w, f"{golden.name} final_tokens")
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    name = f"decode_golden[{golden.name}] {mesh.tag}"
    try:
        eager, wall_first = device_decode(mesh, tokens, height, width, traced=False)
        eager2, wall_warm = device_decode(mesh, tokens, height, width, traced=False)
        assert torch.equal(eager, eager2), "second eager decode differs from the first"
        assert eager.shape == (3, height, width) and torch.isfinite(eager).all()
        p, r = pcc(reference, eager), float(
            ((reference.double() - eager.double()) ** 2).mean().sqrt() / reference.double().std()
        )
        ref_png, dev_png = to_pil(reference), to_pil(eager)
        psnr_dec = psnr_uint8(dev_png, ref_png)
        golden_png = case_dir / "golden.png"
        psnr_golden = None
        if golden_png.is_file():
            from PIL import Image

            psnr_golden = psnr_uint8(dev_png, Image.open(golden_png).convert("RGB"))
        logger.info(
            f"RESULT {name} eager: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU reference decode pcc {p:.6f} "
            f"rel_rmse {r:.5f} max|d| {(reference - eager).abs().max().item():.4f}; PSNR vs CPU decode {psnr_dec:.2f} dB"
            + (f"; PSNR vs golden.png {psnr_golden:.2f} dB" if psnr_golden is not None else "")
        )
        out_dir = renders_dir()
        if out_dir is not None:
            dev_png.save(out_dir / f"vae_decode_{mesh.tag}_{variant}_{width}x{height}.png")
        traced1, wall_capture = device_decode(mesh, tokens, height, width, traced=True)
        traced2, wall_replay = device_decode(mesh, tokens, height, width, traced=True)
        logger.info(
            f"RECORD {name} traced: capture {wall_capture:.3f}s replay {wall_replay:.3f}s; traced == eager "
            f"{torch.equal(traced1, eager)}; replay == capture {torch.equal(traced1, traced2)}; {memory_line(mesh.device)}"
        )
        assert torch.equal(traced1, eager), "traced decode differs from eager"
        assert torch.equal(traced1, traced2), "replay differs from capture"
        # Input swap through the captured trace: replay on the OTHER variant's tokens (same size) must equal that
        # variant's eager decode, and a replay back must equal the capture. A trace replaying its stale capture
        # input would pass every row above (M6/M7 verifier finding).
        others = [(v, d) for v, d in GOLDEN_VARIANTS if v != variant]
        if others:
            other_variant, other_dir = others[0]
            other_tokens = load_golden_case(other_dir).steps["final_tokens"]
            assert other_tokens.shape == tokens.shape and not torch.equal(other_tokens, tokens)
            other_eager, _ = device_decode(mesh, other_tokens, height, width, traced=False)
            other_traced, wall_swap = device_decode(mesh, other_tokens, height, width, traced=True)
            back, _ = device_decode(mesh, tokens, height, width, traced=True)
            logger.info(
                f"RESULT {name} trace input swap: replay on {other_variant} tokens == its eager decode "
                f"{torch.equal(other_traced, other_eager)} (pcc vs the {variant} render {pcc(eager, other_traced):.3f}); "
                f"replay back == capture {torch.equal(back, traced1)}; swap replay {wall_swap:.3f}s"
            )
            assert torch.equal(other_traced, other_eager), "replay on swapped tokens differs from their eager decode"
            assert not torch.equal(other_traced, traced1), "replay ignored the swapped input"
            assert torch.equal(back, traced1), "replay back on the capture tokens differs from the capture"
        else:
            logger.warning(f"{name}: only one golden variant on disk; the trace input swap is not checked")
        assert psnr_dec >= PSNR_DEC_FLOOR, f"PSNR vs CPU decode {psnr_dec:.2f} dB < {PSNR_DEC_FLOOR}"
        assert_quality(reference, eager, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


@vae_rows
def test_decode_padded_layout(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot) -> None:
    if h_axis is None:
        pytest.skip("the partition row needs an H axis")
    height, width = PADDED_SIZE
    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    helper = mesh.helper
    sp = int(mesh.device.shape[sp_axis])
    assert packing.padded_length(h * w, sp) > h * w, "the size must have alignment rows"
    assert helper.partitioned(height, width)
    torch.manual_seed(1)
    tokens = torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    name = f"decode_padded[{width}x{height}] {mesh.tag}"
    try:
        # Decoder input, partitioned path: gather, slice, mesh_partition over the H axis, unpatchify (per-device H shard).
        state = upload_state(mesh, tokens)
        rows = helper.gather_real_rows(state, num_image_rows=h * w)
        gathered_host = tensor.to_torch(rows)[0]
        assert gathered_host.shape == (h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS)
        assert torch.equal(gathered_host.float(), tokens), "gathered + sliced rows differ from the uploaded tokens"
        parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=h_axis, memory_config=rows.memory_config())
        z_part = helper.decoder.preprocess_and_unpatchify(parted, height=height // 8, width=width // 8)
        z_part_host = tensor.to_torch(z_part, mesh_axes=[None, h_axis, None, None])
        # Replicated path on the same rows: unpatchify the whole latent on every device (the decoder's own helper
        # with h_factor 1 semantics is reproduced by composing per-device shards above; here the full tensor).
        z_full_host = reference_unpatchify_on_device(mesh, rows, height, width)
        logger.info(
            f"RESULT {name}: partitioned decoder input {tuple(z_part_host.shape)} == replicated "
            f"{torch.equal(z_part_host, z_full_host)}; {packing.padded_length(h * w, sp) - h * w} alignment rows sliced"
        )
        assert tuple(z_part_host.shape) == (1, height // 8, width // 8, 32)
        assert torch.equal(z_part_host, z_full_host)
        for t in (state, rows, parted, z_part):
            ttnn.deallocate(t)
    finally:
        helper.release_trace()
        ttnn.synchronize_device(mesh.device)


@vae_rows
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Flux2VaeDecoder group_norm core grid 8x8 invalid at the per-device shape Ht=504 (groupnorm.cpp:73); "
    "non-1024 sizes are M8/M10 (design section 9)",
)
def test_decode_padded_size_full(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot) -> None:
    """V4b: the FULL decode of the 1344(w)x768(h) random state vs the CPU reference at Flux2's gate. KNOWN FAILURE
    on the 4x8 H8 row (2026-10-17, logs/m67_builder_vae_4x8_v4.log:295, reproduced logs/m67_verifier_vae_4x8.log:140):
    ``Flux2VaeDecoder``'s group norm picks a core grid that is invalid for the per-device shape (``Ht=504`` tiles is
    not divisible by the 16 virtual rows of the 8x8 grid; ``TT_THROW`` -> ``RuntimeError``): a limitation of the
    shipped decoder's grid selection at this size, not of the slice / partition (proven by
    ``test_decode_padded_layout``). Non-1024 sizes are an M8/M10 item (design section 9). ``xfail(strict=True)``: a
    decoder fix makes this row XPASS = fail, which is the signal to drop the marker and gate the size."""
    if h_axis is None:
        pytest.skip("the partition row needs an H axis")
    height, width = PADDED_SIZE
    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    torch.manual_seed(1)
    tokens = torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    name = f"decode_padded_full[{width}x{height}] {mesh.tag}"
    try:
        reference = cached_reference_decode(snapshot, tokens, h, w, f"random {width}x{height}")
        image, wall = device_decode(mesh, tokens, height, width, traced=False)
        p = pcc(reference, image)
        logger.info(
            f"RESULT {name}: wall {wall:.3f}s; vs CPU reference decode pcc {p:.6f}; PSNR "
            f"{psnr_uint8(to_pil(image), to_pil(reference)):.2f} dB; {memory_line(mesh.device)}"
        )
        assert image.shape == (3, height, width) and torch.isfinite(image).all()
        assert_quality(reference, image, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


def reference_unpatchify_on_device(mesh: VaeMesh, rows: ttnn.Tensor, height: int, width: int) -> torch.Tensor:
    """The decoder's inverse BatchNorm + unpatchify on the FULL (replicated) row tensor, read back ``[1, H/8, W/8, 32]``:
    the same device ops as ``preprocess_and_unpatchify`` with ``h_factor`` 1."""
    decoder = mesh.decoder
    p = Flux2VaeDecoder._PATCH_SIZE
    z = decoder._inv_normalize(rows)
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    z = z.reshape([1, height // 8 // p, width // 8 // p, -1, p, p])
    z = ttnn.permute(z, [0, 1, 4, 2, 5, 3])
    z = z.reshape([1, height // 8, width // 8, -1])
    z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
    host = tensor.to_torch(z)
    ttnn.deallocate(z)
    return host
