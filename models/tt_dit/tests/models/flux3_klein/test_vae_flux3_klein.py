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
  V3 test_decode_golden_tokens[row, variant]  the golden ``final_tokens`` (base and distilled man_dog, 1024x1024: the
                                   first T2I 1024 case per model, ``golden_variants``; the reference cases of M8 are NOT
                                   eligible, a 3920-row figurine partner broke the swap assert, M8 verifier finding)
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
  V4b test_decode_padded_size_full[bh_4x8_ring]  the full 1344x768 decode vs the CPU reference at Flux2's gate. Was a
                                   strict xfail in M6/M7: the attention's PLAIN ``GroupNorm`` on the all-gathered full
                                   spatial tensor (96x168 = Ht 504 tiles, 128 channels per device) failed ``ttnn.group_norm``'s
                                   8x8 DRAM grid rule (16 virtual rows do not divide 504; logs/m67_builder_vae_4x8_v4.log:187,
                                   251, 295). M8 added the per-shape grid fallback to ``layers/normalization.py::GroupNorm``
                                   ((8, 7) for Ht 504; the shape the 8x8 grid accepts keeps it), so the row now gates.
                                   The design's second (replicated-rows) decoder instance is NOT built here: its decode would
                                   differ from the partitioned one in the cross-device GroupNorm reduction order, so a
                                   bit-equality of decoder OUTPUTS is not a property of the slice; the inputs are.
M8 rows (design ``docs/M8_DESIGN.md`` sections 4, 7, 9.2):
  V5 test_plain_grid_rule (host)   ``plain_group_norm_grid`` (the DRAM grid rule, pure host arithmetic) == ttnn's own
                                   ``_find_expected_dram_grid`` on every Klein VAE level of 1024x1024, 896x1120 and
                                   1344x768 (design 2.3 table: (8, 7) for Ht 490 / 1960 / 504, (8, 8) elsewhere) and
                                   raises where only an x < 8 grid exists (832x1248: (4, 3)).
  V5b test_klein_conv_slice_table (host)  every Klein ``Conv2d`` DRAM slice entry (``klein_conv_slices``) and every
                                   stride-2 downsample count (``klein_downsample_slices`` == the rule ``VaeDownsample2d``
                                   applies per instance) respects ttnn's ``num_slices <= ceil(output_width / 32)``
                                   (``op_slicing.cpp``) for 1024x1024, 896x1120, 1344x768, 2048x2048, the multi-reference
                                   fits 992x992 / 1312x784, 512x512, 256x256 and 64x64; the shipped 1024 decode entries
                                   are unchanged by the cap (M8 verifier finding: the shipped table threw
                                   ``TT_FATAL ... slices (8) exceeds the maximum allowed (4)`` at the level-2 downsample).
  V6 test_downsample_matches_torch[bh_4x8_ring]  ``VaeDownsample2d`` (``Conv2d`` k3 stride 2 padding (0, 1, 0, 1), channels
                                   TP4) on random [1, 64, 64, 128] and [1, 35, 35, 128] inputs vs ``F.conv2d(F.pad(x, (0, 1,
                                   0, 1)), w, b, stride=2)``: output 32x32 / 17x17 (floor((H + 1 - 3) / 2) + 1 = floor(H / 2);
                                   torch and the device agree, the first row asserted ceil and failed its own shape check
                                   while matching torch), pcc / max |d| gated at DOWNSAMPLE_PCC (design probe (a): the
                                   4-tuple padding through ttnn.conv2d).
  E1-E3 test_encode_reference[input]  the example inputs (2.png, 3.png 1024x1024; 1.png 896x1120) prepared by
                                   ``references.prepare_reference`` -> ``Flux3KleinVaeEncode.encode`` (Flux2VaeEncoder,
                                   channels TP4 over axis 0, spatial replicated over axis 1) vs the CPU reference
                                   ``AutoEncoder.encode`` (fp32): pcc > ENCODE_PCC (PLAN M6), rel_rmse <= ENCODE_RMSE
                                   (frozen at the first measurement), ids == ``latent_ids``; second encode bit-equal;
                                   traced == eager and a trace replayed on the other 1024 input == its eager encode (the
                                   trace input swap); for 1.png the encode on an encoder that first ran 1024 inputs ==
                                   the encode on a fresh encoder (prepared conv weights do not depend on the size).
  E4 test_encode_matches_golden_cond[case]  the device encode of the golden's prepared references vs the golden
                                   ``cond_tokens`` (the same oracle rounded to bf16): pcc > ENCODE_PCC; ids exact.
  D1 test_decode_896x1120[bh_4x8_ring]  random bf16-valued tokens (seed 1) and, when present, the ``distilled/1_figurine``
                                   golden ``final_tokens`` through ``Flux3KleinVaeDecode`` (h 70 % 8 != 0 -> the replicated
                                   TP4 decoder + the grid fallback at Ht 490 / 1960) vs the CPU decode: Flux2's gate,
                                   PSNR >= PSNR_DEC_FLOOR_896x1120 (frozen at the first measurement); PSNR vs golden.png
                                   logged (calibration).

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

from ....models.vae.encoder_flux2 import Flux2VaeEncoder, VaeDownsample2d
from ....models.vae.vae import VaeContext
from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.pipeline_flux3_klein import to_pil
from ....pipelines.flux3_klein.references import prepare_reference, prepare_references, reference_ids_for
from ....pipelines.flux3_klein.vae_encode_flux3_klein import (
    Flux3KleinVaeEncode,
    build_flux3_klein_vae_encoder,
    load_flux3_klein_vae_encoder_weights,
)
from ....pipelines.flux3_klein.vae_flux3_klein import (
    Flux3KleinVaeDecode,
    build_flux3_klein_vae_decoder,
    load_flux3_klein_vae_weights,
    plain_group_norm_grid,
    vae_blob,
)
from ....reference.flux3_klein import autoencoder as ref_ae
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    GOLDENS_ENV,
    GoldenCase,
    cached_reference_encode,
    checkpoint_dir,
    golden_cases,
    golden_reference_images,
    load_golden_case,
    pcc,
    psnr_uint8,
    reference_autoencoder,
    reference_golden_cases,
    rel_rmse_of,
    renders_dir,
    require_example_input,
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
# D1 geometry: the 1_figurine canvas, 896 wide x 1120 tall -> h 70, w 56 (70 % 8 != 0: the replicated decoder).
FIGURINE_SIZE = (1120, 896)  # (height, width)
ORACLE_THREADS = 16
IGNORED_KEYS = ("bn.num_batches_tracked",)
# --- M8 encoder gates ---------------------------------------------------------------------------------------------------
# Encode: PLAN M6 / docs/M6_M7_DESIGN.md 4.2: tokens pcc > 0.999 vs the CPU reference AutoEncoder.encode (fp32) of the SAME
# prepared image. ENCODE_RMSE has no prior measurement anywhere in the project: None until the first E1 RESULT line
# (logs/m8_builder_vae_4x8.log) sets it to the measured rel_rmse rounded up to two significant digits plus 25 % margin,
# with the log cite here; afterwards it may only tighten.
ENCODE_PCC = 0.999
ENCODE_RMSE: float | None = None
# D1: PSNR of the device render vs the CPU decode of the same tokens at 896x1120 (the replicated decoder + grid fallback):
# no prior measurement; None until the first D1 RESULT line sets it (measured rounded down to 0.5 dB minus 1.0 dB, the
# 1024 floor's rule). The 1024 floor PSNR_DEC_FLOOR (50.0) is never silently reused here.
PSNR_DEC_FLOOR_896x1120: float | None = None
# V6: the downsample conv vs torch fp32 on random data (bf16 operands, fp32 accumulation on device; the torch side runs
# the bf16-rounded inputs / weights in fp32). Measured 2026-10-17 on the 4x8 ring (M8 numerics verifier,
# logs/m8_verifier_numerics_vae_4x8.log:843 and :1494): 64x64 pcc 0.999995 rel_rmse 0.00340 max|d| 3.58e-2, second ==
# first; 35x35 pcc 0.999995 rel_rmse 0.00341 max|d| 4.60e-2 (output (1, 17, 17, 128) == torch). Gate: the measured
# 0.999995 rounded down to 0.9999 (a few bf16 ulps of the output); the block-level 0.999 placeholder is withdrawn.
DOWNSAMPLE_PCC = 0.9999
DOWNSAMPLE_RMSE = 0.005  # measured 0.00340 / 0.00341 (same lines), rounded up to one significant digit + margin
EXAMPLE_INPUTS = ("2.png", "3.png", "1.png")
REFERENCE_CASES = reference_golden_cases()


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
# ``reference_autoencoder`` (the vendored fp32 AutoEncoder, loaded once) lives in common.py since M8 (the host reference
# test shares it).


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
    """The first T2I 1024x1024 golden per model (no references): V3's rows and its trace input-swap partners. With the
    M8 reference goldens on disk the first sorted case was ``distilled/1_figurine`` (896x1120, 3920 rows) and the swap
    assert failed on the shape (logs/m8_verifier_numerics_vae_4x8.log:1702-1710); the filter keeps V3 the M6 row."""
    out = []
    for case_dir in golden_cases():
        record = case_dir / "record.json"
        if record.is_file():
            replay = json.loads(record.read_text()).get("replay", {})
            model = replay.get("model")
            t2i_1024 = not replay.get("references") and list(replay.get("size", [])) == [1024, 1024]
            if model in ("base", "distilled") and t2i_1024 and not any(m == model for m, _ in out):
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
def test_decode_padded_size_full(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot) -> None:
    """V4b: the FULL decode of the 1344(w)x768(h) random state vs the CPU reference at Flux2's gate on the shipped H8
    decoder (h 48 % 8 == 0). Until M8 this threw in ``ttnn.group_norm`` (2026-10-17, logs/m67_builder_vae_4x8_v4.log:295,
    reproduced logs/m67_verifier_vae_4x8.log:140): the attention's plain ``GroupNorm`` on the all-gathered full spatial
    tensor (Ht 504 tiles) rejected the 8x8 grid (16 virtual rows do not divide 504). ``GroupNorm._core_grid_for`` now
    takes the (8, 7) sub-grid the op's own rule suggests for that shape (every shape the 8x8 grid accepted keeps it)."""
    if h_axis is None:
        pytest.skip("the partition row needs an H axis")
    height, width = PADDED_SIZE
    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    assert mesh.helper.partitioned(height, width)
    torch.manual_seed(1)
    tokens = torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    name = f"decode_padded_full[{width}x{height}] {mesh.tag}"
    try:
        reference = cached_reference_decode(snapshot, tokens, h, w, f"random {width}x{height}")
        image, wall = device_decode(mesh, tokens, height, width, traced=False)
        image2, wall2 = device_decode(mesh, tokens, height, width, traced=False)
        p, r = pcc(reference, image), rel_rmse_of(reference, image)
        psnr = psnr_uint8(to_pil(image), to_pil(reference))
        logger.info(
            f"RESULT {name}: first {wall:.3f}s warm {wall2:.3f}s; vs CPU reference decode pcc {p:.6f} rel_rmse {r:.5f} "
            f"max|d| {(reference - image).abs().max().item():.4f}; PSNR vs CPU decode {psnr:.2f} dB; second eager == first "
            f"{torch.equal(image, image2)}; {memory_line(mesh.device)}"
        )
        out_dir = renders_dir()
        if out_dir is not None:
            to_pil(image).save(out_dir / f"vae_decode_{mesh.tag}_random_{width}x{height}.png")
        assert image.shape == (3, height, width) and torch.isfinite(image).all()
        assert torch.equal(image, image2), "second eager decode differs from the first"
        assert_quality(reference, image, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


@vae_rows
@pytest.mark.parametrize(
    "height, width", [(1024, 1024), (768, 1344), (2048, 2048)], ids=["1024x1024", "1344x768", "2048x2048"]
)
def test_decode_local_query_attention_bit_identical(
    mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot, height: int, width: int
) -> None:
    """M10: ``VaeAttention.local_queries`` (each chip attends for its own query rows: spatial shard, then its TP slice,
    with the TP group's rows all-gathered before ``to_out``) must be bit-identical to the replicated full-spatial
    SDPA it replaces (same K/V, same k chunking, per-row math), on the shipped TP4 x H8 decoder and on one chip
    (where it is a no-op). Logs the decode wall of both paths; the 2048 case is the one with the measurable gain."""
    from models.tt_dit.models.vae.vae import VaeAttention

    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    torch.manual_seed(3)
    tokens = torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    name = f"decode_local_query_attention[{width}x{height}] {mesh.tag}"
    saved = VaeAttention.local_queries
    try:
        VaeAttention.local_queries = False
        replicated, _ = device_decode(mesh, tokens, height, width, traced=False)
        replicated, wall_rep = device_decode(mesh, tokens, height, width, traced=False)
        VaeAttention.local_queries = True
        local, _ = device_decode(mesh, tokens, height, width, traced=False)
        local, wall_loc = device_decode(mesh, tokens, height, width, traced=False)
        same = torch.equal(replicated, local)
        logger.info(
            f"RESULT {name}: replicated SDPA warm decode {wall_rep:.3f}s, local-query {wall_loc:.3f}s; bit-identical "
            f"{same}; max|d| {(replicated - local).abs().max().item():.4g}; {memory_line(mesh.device)}"
        )
        assert local.shape == (3, height, width) and torch.isfinite(local).all()
        assert same, "local-query attention differs from the replicated full-spatial SDPA"
    finally:
        VaeAttention.local_queries = saved
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


# --- M8: V5 host grid rule, V6 downsample, E1-E4 encoder, D1 896x1120 ---------------------------------------------------------


def test_plain_grid_rule(expect_error) -> None:
    """V5: the host copy of ``ttnn.group_norm``'s DRAM grid rule == ttnn's own ``_find_expected_dram_grid`` for every
    Klein VAE level at TP4 (design 2.3 table)."""
    from ttnn.operations.normalization import _find_expected_dram_grid

    # (rows, channels per device, groups per device) -> expected (x, y)
    rows = {
        # 1024x1024: decoder x1..x8 / encoder /8../1
        (128 * 128, 128, 8): (8, 8),
        (256 * 256, 128, 8): (8, 8),
        (512 * 512, 64, 4): (8, 8),
        (1024 * 1024, 32, 2): (8, 8),
        # 896x1120 (figurine): Ht 490 / 1960 need (8, 7); 7840 / 31360 keep 8x8
        (140 * 112, 128, 8): (8, 7),
        (280 * 224, 128, 8): (8, 7),
        (560 * 448, 64, 4): (8, 8),
        (1120 * 896, 32, 2): (8, 8),
        # 1344x768 (V4b): only the full-spatial level x1 (Ht 504) needs (8, 7)
        (96 * 168, 128, 8): (8, 7),
        (192 * 336, 128, 8): (8, 8),
        (384 * 672, 64, 4): (8, 8),
        (768 * 1344, 32, 2): (8, 8),
        # tiny shapes (Ht 8 and 2)
        (16 * 16, 128, 8): (8, 4),
        (8 * 8, 128, 8): (8, 1),
    }
    for (n, c, g), expected in rows.items():
        got = plain_group_norm_grid(n, c, g)
        theirs = _find_expected_dram_grid(8, 8, c, g, n, 1)
        assert got == expected == (theirs.x, theirs.y), (n, c, g, got, expected, (theirs.x, theirs.y))
    # 832x1248 level x1: 156x104 = 16224 rows = Ht 507 = 3 x 13 x 13: no grid with x = 8 serves it; ttnn suggests (4, 3)
    theirs = _find_expected_dram_grid(8, 8, 128, 8, 156 * 104, 1)
    assert (theirs.x, theirs.y) == (4, 3)
    with expect_error(ValueError, "no core grid with x=8"):
        plain_group_norm_grid(156 * 104, 128, 8)
    assert packing.latent_shape(*FIGURINE_SIZE)[0] % 8 != 0 and packing.latent_shape(*PADDED_SIZE)[0] % 8 == 0
    logger.info(f"RESULT plain_grid_rule: {len(rows)} Klein VAE levels agree with ttnn's _find_expected_dram_grid")


def test_klein_conv_slice_table(monkeypatch) -> None:
    """V5b: every registered Klein conv slice count and every downsample count sits inside ttnn's bound (docstring)."""
    from ....layers.conv2d import Conv2d
    from ....pipelines.flux3_klein.vae_flux3_klein import (
        KLEIN_CONV_SLICES_PER_LEVEL,
        conv_slice_cap,
        downsample_extent,
        klein_conv_slices,
        klein_downsample_slices,
    )

    assert conv_slice_cap(128) == 4 and conv_slice_cap(129) == 5 and conv_slice_cap(1) == 1
    assert [downsample_extent(v) for v in (1024, 35, 17, 3, 2)] == [512, 17, 8, 1, 1]
    sizes = {
        "1024x1024": (1024, 1024),
        "896x1120": (1120, 896),
        "1344x768": (768, 1344),
        "2048x2048": (2048, 2048),
        "992x992": (992, 992),
        "1312x784": (784, 1312),
        "512x512": (512, 512),
        "256x256": (256, 256),
        "64x64": (64, 64),
    }
    checked, capped = 0, []
    for label, (height, width) in sizes.items():
        table = klein_conv_slices(height, width)
        assert len(table) == sum(len(level) for level in KLEIN_CONV_SLICES_PER_LEVEL), label
        for (h, w, cin, cout), count in table.items():
            assert 1 <= count <= conv_slice_cap(w), (label, (h, w, cin, cout), count, conv_slice_cap(w))
            checked += 1
        downs = klein_downsample_slices(height, width)
        assert len(downs) == 3, downs
        for (h, w, cin, cout), count in downs.items():
            out_w = downsample_extent(w)
            assert cin == cout and 1 <= count <= conv_slice_cap(out_w), (label, (h, w, cin), count, out_w)
            assert count == VaeDownsample2d.slice_count(table[(h, w, cin, cout)], out_w)
            if count < table[(h, w, cin, cout)]:
                capped.append((label, (h, w, cin), table[(h, w, cin, cout)], count))
            checked += 1
    # the shipped 1024 decode entries (slice_default) are inside the bound and untouched by the registration rule
    for (h, w, cin, cout), count in Conv2d.slice_default.items():
        if (h, w) in {(1024, 1024), (512, 512), (256, 256), (128, 128)}:
            assert count <= conv_slice_cap(w), ((h, w, cin, cout), count)
    table_1024 = klein_conv_slices(1024, 1024)
    for key, count in Conv2d.slice_default.items():
        if key in table_1024 and key[2:] in {
            (128, 128),
            (256, 256),
            (512, 512),
            (512, 256),
            (256, 128),
            (128, 3),
            (32, 512),
        }:
            assert table_1024[key] == count, (key, table_1024[key], count)
    # the verifier's violations: 1024 level-2 downsample 8 -> 4; 896x1120 level-0 downsample 16 -> 14 (out_w 448),
    # (560, 448, 512, 256) 16 -> 14, (280, 224, *, 512) 8 -> 7, level-2 downsample 8 -> 4
    assert klein_downsample_slices(1024, 1024)[(256, 256, 512, 512)] == 4
    fig = klein_conv_slices(1120, 896)
    assert fig[(1120, 896, 128, 128)] == 16 and fig[(560, 448, 512, 256)] == 14 and fig[(280, 224, 512, 512)] == 7
    assert klein_downsample_slices(1120, 896)[(1120, 896, 128, 128)] == 14
    assert klein_downsample_slices(1120, 896)[(280, 224, 512, 512)] == 4
    logger.info(
        f"RESULT klein_conv_slice_table: {checked} entries over {len(sizes)} sizes inside ceil(out_w / 32); downsample "
        f"counts capped below the stride-1 count: {capped}"
    )


@vae_rows
@pytest.mark.parametrize("spatial", [(64, 64), (35, 35)], ids=["64x64", "35x35"])
def test_downsample_matches_torch(
    mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot, spatial: tuple[int, int]
) -> None:
    """V6 (design probe (a)): ``VaeDownsample2d`` == ``F.conv2d(F.pad(x, (0, 1, 0, 1)), w, b, stride=2)``."""
    import torch.nn.functional as F

    shape = tuple(mesh_device.shape)
    multi = shape[0] * shape[1] > 1
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology) if multi else None
    ctx = VaeContext(device=mesh_device, tp_axis=tp_axis if multi else None, ccl_manager=ccl)
    channels = 128
    torch.manual_seed(11)
    weight = (torch.randn(channels, channels, 3, 3) * 0.05).to(torch.bfloat16).float()
    bias = (torch.randn(channels) * 0.1).to(torch.bfloat16).float()
    hh, ww = spatial
    x = torch.randn(1, hh, ww, channels).to(torch.bfloat16).float()
    expected = F.conv2d(F.pad(x.permute(0, 3, 1, 2), (0, 1, 0, 1)), weight, bias, stride=2).permute(0, 2, 3, 1)
    module = VaeDownsample2d(channels=channels, ctx=ctx)
    module.load_torch_state_dict({"conv.weight": weight, "conv.bias": bias})
    axes = [None, None, None, tp_axis] if multi else None
    t0 = time.time()
    xt = tensor.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=axes)
    out = module.forward(xt)
    ttnn.synchronize_device(mesh_device)
    wall = time.time() - t0
    got = tensor.to_torch(out, mesh_axes=axes).float()
    out2 = module.forward(xt)
    got2 = tensor.to_torch(out2, mesh_axes=axes).float()
    p, r, m = pcc(expected, got), rel_rmse_of(expected, got), (expected - got).abs().max().item()
    logger.info(
        f"RESULT downsample[{hh}x{ww}] tp{tp_axis}: out {tuple(got.shape)} expected {tuple(expected.shape)}; vs torch "
        f"F.pad(0,1,0,1)+conv stride 2: pcc {p:.6f} rel_rmse {r:.5f} max|d| {m:.4e}; second == first {torch.equal(got, got2)}; "
        f"first call {wall:.2f}s"
    )
    for t in (xt, out, out2):
        ttnn.deallocate(t)
    module.deallocate_weights()
    # floor((H + 1 - 3) / 2) + 1 = floor(H / 2): 64 -> 32, 35 -> 17 (torch's conv, the device and VaeDownsample2d.output_extent)
    assert tuple(got.shape) == tuple(expected.shape) == (1, hh // 2, ww // 2, channels)
    assert VaeDownsample2d.output_extent(hh) == hh // 2 and VaeDownsample2d.output_extent(ww) == ww // 2
    assert torch.equal(got, got2)
    assert p >= DOWNSAMPLE_PCC, p
    assert r <= DOWNSAMPLE_RMSE, r


@dataclass
class EncoderMesh:
    device: ttnn.MeshDevice
    ccl: CCLManager | None
    encoder: Flux2VaeEncoder
    helper: Flux3KleinVaeEncode
    tag: str


def open_encoder(mesh_device, tp_axis, topology, num_links, snapshot: Path) -> EncoderMesh:
    shape = tuple(mesh_device.shape)
    multi = shape[0] * shape[1] > 1
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology) if multi else None
    encoder, parallel_config = build_flux3_klein_vae_encoder(
        mesh_device, tp_axis=tp_axis if multi else None, ccl_manager=ccl
    )
    t0 = time.time()
    state = load_flux3_klein_vae_encoder_weights(
        encoder, snapshot, parallel_config=parallel_config, mesh_device=mesh_device
    )
    tag = f"{shape[0]}x{shape[1]}_tp{tp_axis}_replicated"
    logger.info(f"RECORD encoder {tag}: weights {state} in {time.time() - t0:.1f}s; {memory_line(mesh_device)}")
    return EncoderMesh(mesh_device, ccl, encoder, Flux3KleinVaeEncode(encoder, device=mesh_device), tag)


def example_prepared(name: str):
    from PIL import Image

    image, crop = prepare_reference(Image.open(require_example_input(name)))
    assert crop is None, (name, crop)
    return image


def encode_report(name: str, oracle: torch.Tensor, got: torch.Tensor, wall_first: float, wall_warm: float, mem: str):
    """pcc / rel_rmse / max |d| of the device bf16 tokens vs the fp32 oracle, plus the pcc vs the oracle rounded to bf16
    (the golden-equivalent comparison). Returns ``(pcc, rel_rmse)``."""
    got32 = got.float()
    p, r, m = pcc(oracle, got32), rel_rmse_of(oracle, got32), (oracle - got32).abs().max().item()
    p_bf16 = pcc(oracle.to(torch.bfloat16).float(), got32)
    equal_bf16 = torch.equal(oracle.to(torch.bfloat16), got)
    logger.info(
        f"RESULT {name}: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU reference encode (fp32) pcc {p:.6f} "
        f"rel_rmse {r:.5f} max|d| {m:.4f} (oracle std {oracle.std().item():.4f}); vs the oracle rounded to bf16 pcc "
        f"{p_bf16:.6f} bit-equal {equal_bf16}; {mem}"
    )
    return p, r


def gate_encode(p: float, r: float) -> None:
    assert p > ENCODE_PCC, f"encode pcc {p:.6f} <= {ENCODE_PCC}"
    if ENCODE_RMSE is not None:
        assert r <= ENCODE_RMSE, f"encode rel_rmse {r:.5f} > {ENCODE_RMSE}"


@vae_rows
@pytest.mark.parametrize("name", list(EXAMPLE_INPUTS))
def test_encode_reference(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot, name: str) -> None:
    image = example_prepared(name)
    width, height = image.size
    h, w = packing.latent_shape(height, width)
    oracle, oracle_ids = cached_reference_encode(snapshot, image, f"example {name}")
    assert torch.equal(oracle_ids, packing.latent_ids(h, w))
    mesh = open_encoder(mesh_device, tp_axis, topology, num_links, snapshot)
    row = f"encode[{name}] {width}x{height} {mesh.tag}"
    fresh = None
    try:
        if name == "1.png":
            # the size-change question: the same encoder first runs a 1024 input (its prepared conv weights come from
            # that shape), then the 896x1120 one; a FRESH encoder encodes 1.png as its first size for the comparison.
            warm = mesh.helper.encode(example_prepared("2.png"))
            assert warm.shape[0] == 64 * 64
        first = mesh.helper.encode(image)
        wall_first = mesh.helper.last_wall_s
        second = mesh.helper.encode(image)
        wall_warm = mesh.helper.last_wall_s
        assert first.shape == (h * w, 128) and first.dtype == torch.bfloat16 and torch.isfinite(first.float()).all()
        assert torch.equal(first, second), "second encode differs from the first"
        p, r = encode_report(row, oracle, first, wall_first, wall_warm, memory_line(mesh.device))
        if name == "1.png":
            fresh = open_encoder(mesh_device, tp_axis, topology, num_links, snapshot)
            fresh_tokens = fresh.helper.encode(image)
            logger.info(
                f"RESULT {row}: encode after a 1024 input == encode on a fresh encoder {torch.equal(first, fresh_tokens)} "
                f"(pcc {pcc(first.float(), fresh_tokens.float()):.6f})"
            )
            assert torch.equal(first, fresh_tokens), "prepared conv weights depend on the first input size"
        else:
            traced1 = mesh.helper.encode(image, traced=True)
            wall_capture = mesh.helper.last_wall_s
            traced2 = mesh.helper.encode(image, traced=True)
            wall_replay = mesh.helper.last_wall_s
            other_name = "3.png" if name == "2.png" else "2.png"
            other = example_prepared(other_name)
            other_eager = mesh.helper.encode(other)
            other_traced = mesh.helper.encode(other, traced=True)
            back = mesh.helper.encode(image, traced=True)
            logger.info(
                f"RESULT {row} traced: capture {wall_capture:.3f}s replay {wall_replay:.3f}s; traced == eager "
                f"{torch.equal(traced1, first)}; replay == capture {torch.equal(traced1, traced2)}; replay on {other_name} == "
                f"its eager encode {torch.equal(other_traced, other_eager)} (pcc vs {name} {pcc(first.float(), other_traced.float()):.3f}); "
                f"replay back == capture {torch.equal(back, traced1)}; {memory_line(mesh.device)}"
            )
            assert torch.equal(traced1, first) and torch.equal(traced1, traced2)
            assert torch.equal(other_traced, other_eager) and not torch.equal(other_traced, traced1)
            assert torch.equal(back, traced1)
        gate_encode(p, r)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)
        mesh.encoder.deallocate_weights()
        if fresh is not None:
            fresh.encoder.deallocate_weights()
        ttnn.synchronize_device(mesh.device)


@vae_rows
@pytest.mark.parametrize(
    "case_dir", REFERENCE_CASES or [None], ids=[f"{p.parent.name}-{p.name}" for p in REFERENCE_CASES] or ["no_goldens"]
)
def test_encode_matches_golden_cond(
    mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot, case_dir: Path | None
) -> None:
    if case_dir is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no reference case under it")
    golden = load_golden_case(case_dir)
    prepared = prepare_references(golden_reference_images(golden), width=None, height=None)
    assert [list(s) for s in prepared.sizes] == golden.replay["prepared_reference_sizes"]
    mesh = open_encoder(mesh_device, tp_axis, topology, num_links, snapshot)
    name = f"encode_golden_cond[{golden.name}] {mesh.tag}"
    try:
        tokens, ids = mesh.helper.encode_references(prepared)
        assert torch.equal(ids, golden.inputs["cond_ids"]), "reference ids differ from the golden cond_ids"
        assert torch.equal(ids, reference_ids_for(prepared))
        golden_cond = golden.inputs["cond_tokens"]
        assert tokens.shape == golden_cond.shape and tokens.dtype == golden_cond.dtype == torch.bfloat16
        start, worst = 0, 1.0
        for index, (h, w) in enumerate(prepared.latent_grids()):
            rows_dev, rows_gold = tokens[start : start + h * w].float(), golden_cond[start : start + h * w].float()
            start += h * w
            p, r = pcc(rows_gold, rows_dev), rel_rmse_of(rows_gold, rows_dev)
            worst = min(worst, p)
            logger.info(
                f"RESULT {name} reference {index} {prepared.sizes[index]}: device tokens vs golden cond_tokens pcc {p:.6f} "
                f"rel_rmse {r:.5f} max|d| {(rows_gold - rows_dev).abs().max().item():.4f}; encode wall {mesh.helper.last_wall_s:.3f}s"
            )
        logger.info(
            f"RESULT {name}: worst reference pcc {worst:.6f} (gate {ENCODE_PCC}); ids exact; {memory_line(mesh.device)}"
        )
        assert worst > ENCODE_PCC
    finally:
        ttnn.synchronize_device(mesh.device)
        mesh.encoder.deallocate_weights()
        ttnn.synchronize_device(mesh.device)


def figurine_golden() -> tuple[Path, GoldenCase] | None:
    for case_dir in REFERENCE_CASES:
        golden = load_golden_case(case_dir)
        if list(golden.replay["size"]) == [FIGURINE_SIZE[1], FIGURINE_SIZE[0]]:
            return case_dir, golden
    return None


@vae_rows
def test_decode_896x1120(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot) -> None:
    """D1: the figurine canvas (h 70 % 8 != 0) decodes on the replicated TP4 decoder with the GroupNorm grid fallback."""
    height, width = FIGURINE_SIZE
    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, sp_axis, tp_axis, h_axis, topology, num_links, snapshot)
    assert not mesh.helper.partitioned(height, width) or h_axis is None
    torch.manual_seed(1)
    inputs = [("random", torch.randn(h * w, packing.FLUX3_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float(), None)]
    found = figurine_golden()
    if found is not None:
        case_dir, golden = found
        inputs.append((golden.name, golden.steps["final_tokens"], case_dir / "golden.png"))
    try:
        for label, tokens, golden_png in inputs:
            name = f"decode_896x1120[{label}] {mesh.tag}"
            reference = cached_reference_decode(snapshot, tokens, h, w, f"{label} {width}x{height}")
            image, wall_first = device_decode(mesh, tokens, height, width, traced=False)
            image2, wall_warm = device_decode(mesh, tokens, height, width, traced=False)
            p, r = pcc(reference, image), rel_rmse_of(reference, image)
            dev_png, ref_png = to_pil(image), to_pil(reference)
            psnr_dec = psnr_uint8(dev_png, ref_png)
            extra = ""
            if golden_png is not None and golden_png.is_file():
                from PIL import Image

                extra = f"; PSNR vs golden.png {psnr_uint8(dev_png, Image.open(golden_png).convert('RGB')):.2f} dB (calibration only)"
            logger.info(
                f"RESULT {name}: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU reference decode pcc {p:.6f} rel_rmse "
                f"{r:.5f} max|d| {(reference - image).abs().max().item():.4f}; PSNR vs CPU decode {psnr_dec:.2f} dB (floor "
                f"{PSNR_DEC_FLOOR_896x1120}); second eager == first {torch.equal(image, image2)}{extra}; {memory_line(mesh.device)}"
            )
            out_dir = renders_dir()
            if out_dir is not None:
                dev_png.save(out_dir / f"vae_decode_{mesh.tag}_{label.replace('/', '_')}_{width}x{height}.png")
            assert image.shape == (3, height, width) and torch.isfinite(image).all()
            assert torch.equal(image, image2), "second eager decode differs from the first"
            assert_quality(reference, image, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
            if PSNR_DEC_FLOOR_896x1120 is not None:
                assert psnr_dec >= PSNR_DEC_FLOOR_896x1120, psnr_dec
        traced1, wall_capture = device_decode(mesh, inputs[0][1], height, width, traced=True)
        traced2, wall_replay = device_decode(mesh, inputs[0][1], height, width, traced=True)
        eager, _ = device_decode(mesh, inputs[0][1], height, width, traced=False)
        logger.info(
            f"RESULT decode_896x1120 traced {mesh.tag}: capture {wall_capture:.3f}s replay {wall_replay:.3f}s; traced == eager "
            f"{torch.equal(traced1, eager)}; replay == capture {torch.equal(traced1, traced2)}; {memory_line(mesh.device)}"
        )
        assert torch.equal(traced1, eager) and torch.equal(traced1, traced2)
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


def test_klein_partitioned_decoder_slices() -> None:
    """V5c (host): the per-device slice keys of the H-partitioned decoder (2026-10-20). Up to 1536x1536 every decoder
    conv keeps ttnn's auto slicer (count 0, the configuration the M6/M8 gates and the sweep measured); above it every
    conv of every level gets an explicit count inside ``conv_slice_cap`` whose bf16 input bytes per slice stay under
    the budget; the keys carry the halo rows (``H_level / 8 + 2``) and the registration writes them next to the
    full-image keys without touching a shipped entry."""
    from ....layers.conv2d import Conv2d
    from ....pipelines.flux3_klein.vae_flux3_klein import (
        AUTO_SLICE_MAX_AREA,
        DECODER_SLICE_BUDGET_BYTES,
        KLEIN_DECODER_CONVS_PER_LEVEL,
        conv_slice_cap,
        decoder_slice_count,
        klein_conv_slices,
        klein_partitioned_decoder_slices,
        partitioned_conv_rows,
        register_klein_conv_slices,
    )

    assert partitioned_conv_rows(2048, 8) == 258 and partitioned_conv_rows(1024, 8) == 130
    assert partitioned_conv_rows(2048, 1) == 2048 and partitioned_conv_rows(36 * 8, 8) == 38
    # the decoder conv list covers the shapes of every level once (upsamplers included) and nothing else
    assert KLEIN_DECODER_CONVS_PER_LEVEL[0][0] == (256, 256, True) and KLEIN_DECODER_CONVS_PER_LEVEL[1][0] == (
        512,
        512,
        True,
    )
    assert KLEIN_DECODER_CONVS_PER_LEVEL[0][-1] == (128, 3, False) and KLEIN_DECODER_CONVS_PER_LEVEL[3][0] == (
        32,
        512,
        True,
    )
    assert klein_partitioned_decoder_slices(1024, 1024, h_factor=1, tp_factor=4) == {}
    auto = klein_partitioned_decoder_slices(1024, 1024, h_factor=8, tp_factor=4)
    assert len(auto) == sum(len(level) for level in KLEIN_DECODER_CONVS_PER_LEVEL)
    assert set(auto.values()) == {0} and (130, 1024, 256, 256) in auto and (18, 128, 32, 512) in auto
    assert set(klein_partitioned_decoder_slices(1536, 1536, h_factor=8, tp_factor=4).values()) == {0}
    assert 1536 * 1536 == AUTO_SLICE_MAX_AREA
    # default mode (auto, 2026-10-21): every size gets ttnn's auto slicer; the aligned counts below need the env
    assert set(klein_partitioned_decoder_slices(4096, 4096, h_factor=8, tp_factor=4).values()) == {0}
    monkeypatch.setenv("FLUX3_KLEIN_VAE_SLICE_MODE", "aligned")
    explicit_sizes = {
        "2048x2048": (2048, 2048),
        "4096x2304": (2304, 4096),
        "4096x4096": (4096, 4096),
        "2560x1440": (1440, 2560),
    }
    assert set(klein_partitioned_decoder_slices(1152, 2048, h_factor=8, tp_factor=4).values()) == {0}  # = the bound
    for label, (height, width) in explicit_sizes.items():
        table = klein_partitioned_decoder_slices(height, width, h_factor=8, tp_factor=4)
        assert len(table) == sum(len(level) for level in KLEIN_DECODER_CONVS_PER_LEVEL), label
        for (rows, w, cin, cout), count in table.items():
            assert 1 <= count <= conv_slice_cap(w), (label, (rows, w, cin, cout), count)
            tensor_parallel = cout != 3
            cin_device = cin // 4 if (tensor_parallel and cin >= cout) else cin
            per_slice = rows * (-(-w // count)) * cin_device * 2
            assert per_slice <= DECODER_SLICE_BUDGET_BYTES or count == conv_slice_cap(w), (
                label,
                (rows, w, cin, cout),
                count,
                per_slice,
            )
            assert count == decoder_slice_count(rows, w, cin, cout, tensor_parallel, 4)
            assert (w // 32) % count == 0, (label, (rows, w, cin, cout), count)  # TILE-aligned slices
    t2048 = klein_partitioned_decoder_slices(2048, 2048, h_factor=8, tp_factor=4)
    assert (
        t2048[(258, 2048, 256, 256)] == 32 and t2048[(258, 2048, 128, 128)] == 16 and t2048[(258, 2048, 128, 3)] == 64
    )
    assert t2048[(130, 1024, 512, 512)] == 16 and t2048[(34, 256, 32, 512)] == 1
    t4096 = klein_partitioned_decoder_slices(4096, 4096, h_factor=8, tp_factor=4)
    assert t4096[(514, 4096, 128, 3)] == conv_slice_cap(4096) == 128  # the only budget-capped entry
    assert t4096[(514, 4096, 256, 256)] == 128 and t4096[(258, 2048, 512, 512)] == 64
    # registration: full-image + per-device keys side by side, shipped entries untouched, idempotent
    mesh = (97, 97)  # a shape no other test registers
    Conv2d.slice_params.pop(mesh, None)
    register_klein_conv_slices(mesh, 2048, 2048, h_factor=8, tp_factor=4)
    table = Conv2d.slice_params[mesh]
    for key, count in Conv2d.slice_default.items():
        assert table[key] == count, key
    for key, count in klein_conv_slices(2048, 2048).items():  # shipped (replicated 2048) entries win where both exist
        assert table[key] == Conv2d.slice_default.get(key, count), key
    for key, count in t2048.items():
        assert table[key] == count, key
    before = dict(table)
    register_klein_conv_slices(mesh, 2048, 2048, h_factor=8, tp_factor=4)
    register_klein_conv_slices(mesh, 2048, 2048, h_factor=1, tp_factor=4)
    assert Conv2d.slice_params[mesh] == before
    Conv2d.slice_params.pop(mesh, None)
    logger.info(f"partitioned decoder slice keys: 1024 auto {len(auto)} entries; 2048 {t2048}; 4096 {t4096}")


def test_set_conv_config_overrides() -> None:
    """V5d (host): ``set_conv_config_overrides`` reaches every conv under a module tree (per instance) and nothing
    else; the Klein VAE builders set ``config_tensors_in_dram=True`` through it."""
    import inspect

    from ....layers.module import Module
    from ....pipelines.flux3_klein import vae_encode_flux3_klein, vae_flux3_klein
    from ....pipelines.flux3_klein.vae_flux3_klein import set_conv_config_overrides

    class Leaf(Module):
        def forward(self, x):  # noqa: ANN001, ANN202 - stand-in
            return x

    class FakeConv(Leaf):
        def __init__(self) -> None:
            super().__init__()
            self.conv_config_overrides: dict[str, object] = {}

    class Tree(Leaf):
        def __init__(self) -> None:
            super().__init__()
            self.a = FakeConv()
            self.inner = Leaf()
            self.inner.b = FakeConv()
            self.inner.leaf = Leaf()

    tree = Tree()
    assert set_conv_config_overrides(tree, config_tensors_in_dram=True) == 2
    assert tree.a.conv_config_overrides == {"config_tensors_in_dram": True}
    assert tree.inner.b.conv_config_overrides == {"config_tensors_in_dram": True}
    assert not hasattr(tree.inner.leaf, "conv_config_overrides")
    assert set_conv_config_overrides(tree, act_block_h_override=64) == 2
    assert tree.a.conv_config_overrides == {"config_tensors_in_dram": True, "act_block_h_override": 64}
    for builder in (
        vae_flux3_klein.build_flux3_klein_vae_decoder,
        vae_encode_flux3_klein.build_flux3_klein_vae_encoder,
    ):
        assert "set_conv_config_overrides(" in inspect.getsource(
            builder
        ) and "config_tensors_in_dram=_config_tensors_in_dram()" in inspect.getsource(builder)
