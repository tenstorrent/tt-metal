# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 Klein VAE decode (``pipelines/flux2_klein/vae_flux2_klein.py::Flux2KleinVaeDecode``) and encode
(``vae_encode_flux2_klein.py::Flux2KleinVaeEncode``) on device vs diffusers ``AutoencoderKLFlux2`` on CPU in fp32
(``tests/models/flux3_klein/test_vae_flux3_klein.py`` pattern; the autoencoder is the same FLUX.2 model, so the
FLUX 3 gates apply unchanged).

Oracle: ``AutoencoderKLFlux2.from_pretrained(snapshot, subfolder="vae", torch_dtype=float32)`` driven exactly as
``Flux2KleinPipeline`` drives it (``pipeline_flux2_klein.py``, cited as ``P``), through the pipeline's own
staticmethods: decode = ``_unpack_latents_with_ids`` -> inverse BatchNorm ``z * sqrt(var + eps) + mean`` ->
``_unpatchify_latents`` -> ``vae.decode`` (``P:905-921``); encode = ``vae.encode(x).latent_dist.mode()``
(``retrieve_latents(sample_mode="argmax")``) -> ``_patchify_latents`` -> BatchNorm normalize -> ``_pack_latents``
(``P:462-477``, ``P:514-540``). The golden pipeline ran the VAE in bf16 (``generate_goldens``: ``torch_dtype=bf16``),
so ``golden.png`` is a calibration number here, not the reference: the device decoder is bf16 operands / fp32
accumulation and the fp32 oracle is the right yardstick for a 50 dB floor.

Host rows (no device):
  H1 test_unpack_matches_pipeline[case]   the port's host unpack (``packing.bn_denormalize`` + ``unpatchify_tokens``)
                                   ``torch.equal`` the pipeline staticmethod chain on the golden ``final_tokens``, and
                                   the port's ``patchify_latents`` + ``bn_normalize`` ``torch.equal`` the encode chain
                                   on the unpatchified latents (both fp32, both bit-exact).
  H2 test_plain_grid_rule          ``plain_group_norm_grid`` (the host copy of ``ttnn.group_norm``'s DRAM grid rule
                                   behind ``GroupNorm._core_grid_for``) == ttnn's own ``_find_expected_dram_grid`` for
                                   every encoder level (TP4) of the matrix canvases (all 8x8: the fused path is
                                   unchanged there) and of the unaligned reference sizes; the levels whose row count
                                   is not a multiple of 32, or that no x=8 grid serves, take ``_forward_unaligned``.
Device rows (``bh_4x8_ring`` = the (4, 8) ring, the port's default layout: DiT SP8 over axis 1, VAE channels TP4 over
axis 0, latent rows H8 over axis 1):
  D1 test_decode_golden_tokens[case]  the golden ``final_tokens`` (every 1024x1024 case under ``$FLUX2_KLEIN_GOLDENS``)
                                   uploaded as the denoiser's fp32 state slot and decoded on device vs the fp32 CPU
                                   decode: ``assert_quality`` at Flux2's gate (pcc >= DECODE_PCC, rel_rmse <=
                                   DECODE_RMSE); PSNR of the uint8 renders >= PSNR_DEC_FLOOR; second eager == first;
                                   traced == eager; replay == capture; the trace replayed on the OTHER case's tokens ==
                                   that case's eager decode and a replay back == the capture (the tracer's input copy);
                                   PSNR vs ``golden.png`` logged.
  E1 test_encode_golden_png[case]    ``golden.png`` through ``references.prepare_reference`` (1024x1024: unchanged) ->
                                   ``Flux2KleinVaeEncode.encode`` vs the fp32 CPU encode: pcc > ENCODE_PCC, rel_rmse <=
                                   ENCODE_RMSE when set; the reference ids == ``packing.reference_ids``; second encode
                                   bit-equal; traced == eager and a trace replayed on the other case's png == its eager
                                   encode; the pcc vs the oracle rounded to bf16 logged.
  E2 test_encode_unaligned_reference[size]  a synthetic reference of a size BFL's sizing produces for real photos
                                   (880x1168, 400x736, 1232x816: VAE levels with H * W not a multiple of 32, or
                                   tile-aligned but refused by the fused op's grid rule) encodes through the exact
                                   fp32-statistics GroupNorm path (``_forward_unaligned``) and matches the fp32 CPU
                                   encode at the E1 gate (pcc > ENCODE_PCC); second encode bit-equal.
  E3 test_encode_upload_replication_bit_identical[size]  the reference upload: one row band per mesh column + the
                                   on-device gather (``Flux2KleinVaeEncode.replicate_on_device``, the default; sizes
                                   with a ``sharded_upload_view``) vs every chip written from the host: tokens
                                   ``torch.equal`` (1024x1024 and 1344x768 sharded, 880x1168 falls back to the host
                                   path); walls of both logged.

  D2 test_decode_subgrid_canvas[size]  a canvas ``policy.admit`` accepts (every VAE level tile-aligned) whose x1 decoder
                                   level ``ttnn.group_norm``'s 8x8 DRAM grid rule rejects: 1344x768 (h 48 % 8 == 0: the
                                   shipped TP4 x H8 decoder; the attention's plain GroupNorm on the all-gathered 96x168 =
                                   Ht 504 tiles of 128 channels takes the (8, 7) sub-grid) and 896x1120 (h 70 % 8 != 0:
                                   the replicated TP4 fallback decoder; Ht 490 / 1960 at x1 / x2 take (8, 7)). Random
                                   bf16-valued tokens (seed 1) vs the fp32 CPU decode at the D1 gate (pcc >= DECODE_PCC,
                                   rel_rmse <= DECODE_RMSE); PSNR logged; second eager == first; the grids
                                   ``GroupNorm._core_grid_for`` returned during the decode are recorded and the expected
                                   (8, 7) levels asserted (ports of FLUX 3's V4b test_decode_padded_size_full and D1
                                   test_decode_896x1120; before the sub-grid both canvases threw in ``groupnorm.cpp``
                                   ``validate_dram_grid`` on the device, after admission).
  D3 test_decode_prepared_per_shape  ``layers/conv2d.py`` keeps one prepared weight / bias copy per input-shape key
                                   (FLUX 3 Klein's fix for its striped, saturated 2048+ decodes after the 1024 warm-up:
                                   ``ttnn.prepare_conv_weights`` lays the weight out for ONE conv program). The pipeline's
                                   order on ONE decoder instance: a 1024x1024 decode (the ``__init__`` warm-up), then a
                                   2048x2048 one (random bf16-valued tokens, seed 2), first with
                                   ``FLUX2_KLEIN_CONV_PREP_CACHE=single`` (the old single-copy path; its 2048 numbers
                                   vs the fp32 CPU decode are logged, pass or fail), then, after the prepared copies are
                                   dropped, with the default ``per_shape``: the 2048 decode must meet the D1 gate
                                   (pcc >= DECODE_PCC, rel_rmse <= DECODE_RMSE; PSNR logged), a 1024 decode after it
                                   must be bit-equal to the warm-up render (the cached 1024 copy is switched back to)
                                   and a second 2048 decode bit-equal to the first; every conv that ran holds one
                                   prepared copy per shape (the dict keeps every key seen).
Gates (FLUX 3 Klein ``test_vae_flux3_klein.py``, applied unchanged, never loosened): DECODE_PCC 0.9978 / DECODE_RMSE
0.034 (Flux2's decode gate), PSNR_DEC_FLOOR 50.0 dB, ENCODE_PCC 0.999, ENCODE_RMSE None (set from the first measured
line, then only tightened).

Environment: ``FLUX2_KLEIN_MODEL_PATH`` / the HuggingFace cache, ``FLUX2_KLEIN_GOLDENS``, ``FLUX2_KLEIN_ORACLE_CACHE``
(the fp32 CPU decodes / encodes between runs, ~10-20 s each on a node CPU), ``FLUX2_KLEIN_OUT`` (PNGs of the device
renders, never inside the tree), ``TT_DIT_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.encoder_flux2 import Flux2VaeEncoder
from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.manager import CCLManager
from ....pipelines.flux2_klein import packing
from ....pipelines.flux2_klein.pipeline_flux2_klein import to_pil
from ....pipelines.flux2_klein.references import prepare_reference, to_tensor
from ....pipelines.flux2_klein.vae_encode_flux2_klein import (
    Flux2KleinVaeEncode,
    build_flux2_klein_vae_encoder,
    load_flux2_klein_vae_encoder_weights,
    sharded_upload_view,
)
from ....pipelines.flux2_klein.vae_flux2_klein import (
    Flux2KleinVaeDecode,
    build_flux2_klein_vae_decoder,
    load_flux2_klein_vae_weights,
    plain_group_norm_grid,
    vae_blob,
)
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    GOLDENS_ENV,
    MESH_SHAPE,
    ORACLE_THREADS,
    cached_oracle,
    golden_cases,
    pcc,
    psnr_uint8,
    rel_rmse_of,
    renders_dir,
    require_checkpoint_dir,
    sha256_of,
)
from .device_params import ring_params_8k_flux2_klein_transformer

# tt-metal's pytest.ini default is 300 s per test; a COLD kernel cache (a box's first run) spends 230-270 s in the
# first eager decode / encode of a size alone (measured on b03u02 / b05u08, 2026-10-23), so the traced rows after it
# ran into the limit (job 146563: both iceberg rows "Timeout (>300.0s) from pytest-timeout", no device fault).
pytestmark = pytest.mark.timeout(1800)

# Flux2's decode gate (tests/models/flux2/test_vae_flux2.py, FLUX 3 Klein test_vae_flux3_klein.py DECODE_*), unchanged.
DECODE_PCC = 0.9978
DECODE_RMSE = 0.034
# PSNR of the device render vs the CPU reference render of the SAME tokens, uint8 after to_pil (FLUX 3: measured
# 51.3-52.3 dB on the 4x8 ring, floor 50.0).
PSNR_DEC_FLOOR = 50.0
# Encode: tokens pcc > 0.999 vs the fp32 CPU encode of the SAME prepared image (FLUX 3 ENCODE_PCC). ENCODE_RMSE: None
# until the first E1 RESULT line sets it (measured rounded up to two significant digits + 25 % margin); only tightens.
ENCODE_PCC = 0.999
ENCODE_RMSE: float | None = None
# The port's default layout on the (4, 8) ring: DiT SP8 over axis 1; VAE channels TP4 over axis 0, rows H8 over axis 1.
SP_AXIS, TP_AXIS, H_AXIS = 1, 0, 1
NUM_LINKS, TOPOLOGY = 2, ttnn.Topology.Ring
VAE_REQUIRED = ("vae/diffusion_pytorch_model.safetensors", "vae/config.json")

vae_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[MESH_SHAPE, ring_params_8k_flux2_klein_transformer]],
    ids=["bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)


# --- goldens (tolerant of both record layouts: the flat bin/ format and the tests-dir ``replay`` block) -------------------


@dataclass
class Golden:
    path: Path
    size: tuple[int, int]  # (width, height)
    model: str

    @property
    def name(self) -> str:
        return self.path.name

    def final_tokens(self) -> torch.Tensor:
        """``[h*w, 128]`` fp32: the state after the last step (a leading batch dim, if recorded, squeezed)."""
        from safetensors.torch import load_file

        tokens = load_file(str(self.path / "steps.safetensors"))["final_tokens"].float()
        if tokens.ndim == 3:
            assert tokens.shape[0] == 1, tokens.shape
            tokens = tokens[0]
        width, height = self.size
        h, w = packing.latent_shape(height, width)
        assert tokens.shape == (h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS), (self.name, tokens.shape)
        return tokens.contiguous()

    def png(self):
        from PIL import Image

        return Image.open(self.path / "golden.png").convert("RGB")


def golden_1024_cases() -> list[Golden]:
    out = []
    for case_dir in golden_cases():
        record = json.loads((case_dir / "record.json").read_text())
        replay = record.get("replay", record)
        size = replay.get("canvas") or replay.get("size")
        if size is None or replay.get("references") or not (case_dir / "golden.png").is_file():
            continue
        width, height = (int(v) for v in size)
        if (width, height) == (1024, 1024):
            out.append(Golden(case_dir, (width, height), str(replay.get("model", "?"))))
    return out


GOLDENS = golden_1024_cases()
golden_rows = pytest.mark.parametrize("golden", GOLDENS or [None], ids=[g.name for g in GOLDENS] or ["no_goldens"])


def other_golden(golden: Golden) -> Golden | None:
    return next((g for g in GOLDENS if g.path != golden.path), None)


@pytest.fixture(scope="module")
def snapshot() -> Path:
    return require_checkpoint_dir("distilled", "4b", required=VAE_REQUIRED)


# --- the CPU oracle: diffusers AutoencoderKLFlux2 in fp32, driven by the pipeline's own staticmethods --------------------

_REFERENCE_VAE: dict[Path, object] = {}


def reference_vae(snapshot: Path):
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    if snapshot not in _REFERENCE_VAE:
        torch.set_num_threads(ORACLE_THREADS)
        t0 = time.time()
        vae = AutoencoderKLFlux2.from_pretrained(str(snapshot), subfolder="vae", torch_dtype=torch.float32).eval()
        assert vae.config.latent_channels == 32 and list(vae.config.patch_size) == [2, 2]
        assert vae.config.batch_norm_eps == 1e-4 and list(vae.config.block_out_channels) == [128, 256, 512, 512]
        _REFERENCE_VAE[snapshot] = vae
        logger.info(f"reference AutoencoderKLFlux2 (fp32, vae/) loaded in {time.time() - t0:.1f}s")
    return _REFERENCE_VAE[snapshot]


def pipeline_class():
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    return Flux2KleinPipeline


def bn_stats(vae) -> tuple[torch.Tensor, torch.Tensor, float]:
    return vae.bn.running_mean.detach().float(), vae.bn.running_var.detach().float(), float(vae.config.batch_norm_eps)


def pipeline_unpack(vae, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``P:905-916``: ``[h*w, 128]`` tokens -> ``[1, 32, 2h, 2w]`` latents for ``vae.decode`` (fp32)."""
    P = pipeline_class()
    mean, var, eps = bn_stats(vae)
    latents = P._unpack_latents_with_ids(tokens[None], packing.latent_ids(h, w)[None], h, w)
    bn_mean = mean.view(1, -1, 1, 1).to(latents.dtype)
    bn_std = torch.sqrt(var.view(1, -1, 1, 1) + eps).to(latents.dtype)
    latents = latents * bn_std + bn_mean
    return P._unpatchify_latents(latents)


def pipeline_pack(vae, latents: torch.Tensor) -> torch.Tensor:
    """``P:466-477`` + ``_pack_latents``: ``[1, 32, 2h, 2w]`` latents -> ``[h*w, 128]`` normalized tokens (fp32)."""
    P = pipeline_class()
    mean, var, eps = bn_stats(vae)
    latents = P._patchify_latents(latents)
    bn_mean = mean.view(1, -1, 1, 1).to(latents.dtype)
    bn_std = torch.sqrt(var.view(1, -1, 1, 1) + eps).to(latents.dtype)
    latents = (latents - bn_mean) / bn_std
    return P._pack_latents(latents)[0]


def versions() -> str:
    import diffusers

    return f"diffusers={diffusers.__version__}|torch={torch.__version__}"


def cached_reference_decode(snapshot: Path, tokens: torch.Tensor, h: int, w: int, label: str) -> torch.Tensor:
    """fp32 ``[3, H, W]`` in [-1, 1]: the diffusers decode of ``tokens`` through ``$FLUX2_KLEIN_ORACLE_CACHE``."""
    fields = {
        "op": "vae_decode_fp32",
        "vae_blob": vae_blob(snapshot),
        "tokens_sha256": sha256_of(tokens),
        "h": h,
        "w": w,
        "versions": versions(),
    }

    def compute():
        vae = reference_vae(snapshot)
        t0 = time.time()
        with torch.inference_mode():
            image = vae.decode(pipeline_unpack(vae, tokens, h, w), return_dict=False)[0][0]
        logger.info(f"ORACLE {label}: CPU fp32 decode {time.time() - t0:.1f} s -> {tuple(image.shape)}")
        return {"image": image.float()}

    return cached_oracle(fields, label, compute)["image"]


def cached_reference_encode(snapshot: Path, image, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(tokens fp32 [h*w, 128], ids int64 [h*w, 4])``: the diffusers encode (posterior mean) of the prepared image."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    fields = {
        "op": "vae_encode_fp32",
        "vae_blob": vae_blob(snapshot),
        "image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "w": int(image.width),
        "h": int(image.height),
        "versions": versions(),
    }

    def compute():
        vae = reference_vae(snapshot)
        P = pipeline_class()
        t0 = time.time()
        with torch.inference_mode():
            x = to_tensor(image)[None]
            latents = vae.encode(x).latent_dist.mode()  # retrieve_latents(sample_mode="argmax"), P:468
            tokens = pipeline_pack(vae, latents)
            # P:525-527: the ids come from the PATCHIFIED (1, 128, h, w) latents _encode_vae_image returns.
            ids = P._prepare_image_ids([P._patchify_latents(latents)])[0]
        logger.info(f"ORACLE {label}: CPU fp32 encode {time.time() - t0:.1f} s -> {tuple(tokens.shape)}")
        return {"tokens": tokens.float(), "ids": ids}

    data = cached_oracle(fields, label, compute)
    return data["tokens"], data["ids"]


# --- H1: host ----------------------------------------------------------------------------------------------------------------


@golden_rows
def test_unpack_matches_pipeline(snapshot: Path, golden: Golden | None) -> None:
    if golden is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no 1024x1024 case under it")
    vae = reference_vae(snapshot)
    mean, var, eps = bn_stats(vae)
    tokens = golden.final_tokens()
    width, height = golden.size
    h, w = packing.latent_shape(height, width)
    theirs = pipeline_unpack(vae, tokens, h, w)
    ours = packing.unpatchify_tokens(packing.bn_denormalize(tokens, mean, var, eps), h, w)[None]
    assert theirs.shape == ours.shape == (1, 32, 2 * h, 2 * w)
    back_theirs = pipeline_pack(vae, theirs)
    back_ours = packing.bn_normalize(packing.patchify_latents(theirs[0]), mean, var, eps)
    logger.info(
        f"RESULT unpack[{golden.name}]: port unpack == pipeline unpack {torch.equal(ours, theirs)} "
        f"(max|d| {(ours - theirs).abs().max().item():.3e}); port pack == pipeline pack {torch.equal(back_ours, back_theirs)}; "
        f"round trip vs the tokens max|d| {(back_theirs - tokens).abs().max().item():.3e}"
    )
    assert torch.equal(ours, theirs), "the port's host unpack differs from the pipeline's"
    assert torch.equal(back_ours, back_theirs), "the port's host pack differs from the pipeline's"


# --- device: decode ---------------------------------------------------------------------------------------------------------


@dataclass
class VaeMesh:
    device: ttnn.MeshDevice
    dit_ccl: CCLManager
    vae_ccl: CCLManager
    decoder: Flux2VaeDecoder
    helper: Flux2KleinVaeDecode
    tag: str


# --- H2: the plain GroupNorm's DRAM grid rule (host) -------------------------------------------------------------------------

# Every FLUX.2 Klein VAE ENCODER GroupNorm at TP4: (rows = H * W at the level, channels per device, groups per device) ->
# the (x, y) core grid ``plain_group_norm_grid`` must return. The encoder's levels are /1 (128 channels -> 32 local), /2
# (256 -> 64), /4 (512 -> 128) and /8 (512 -> 128) with 32 groups -> 8 local. The matrix canvases keep the fixed 8x8 grid
# on every level, so the fused ``ttnn.group_norm`` call there is unchanged by ``_core_grid_for``; the unaligned reference
# sizes (UNALIGNED_SIZES) either lower y (the /1 level), or no x=8 grid serves them (ValueError -> the exact path).
PLAIN_GRID_ROWS = {
    # 1024x1024 (matrix 1K): /1 .. /8
    (1024 * 1024, 32, 8): (8, 8),
    (512 * 512, 64, 8): (8, 8),
    (256 * 256, 128, 8): (8, 8),
    (128 * 128, 128, 8): (8, 8),
    # 2048x2048 (matrix 2K): /1 .. /8
    (2048 * 2048, 32, 8): (8, 8),
    (1024 * 1024, 64, 8): (8, 8),
    (512 * 512, 128, 8): (8, 8),
    (256 * 256, 128, 8): (8, 8),
    # 880x1168: /1 Ht 32120 -> (8, 5); /2 Ht 8030 refused (below); /4 and /8 are not tile-aligned
    (1168 * 880, 32, 8): (8, 5),
    # 400x736: /1 Ht 9200 and /2 Ht 2300 -> (8, 5); /4 Ht 575 refused; /8 not tile-aligned
    (736 * 400, 32, 8): (8, 5),
    (368 * 200, 64, 8): (8, 5),
    # 1232x816: /1 Ht 31416 -> (8, 7); /2 Ht 7854 refused; /4 and /8 not tile-aligned
    (816 * 1232, 32, 8): (8, 7),
    # 768x1344: tile-aligned everywhere, only the /8 level (Ht 504) needs (8, 7)
    (1344 * 768, 32, 8): (8, 8),
    (672 * 384, 64, 8): (8, 8),
    (336 * 192, 128, 8): (8, 8),
    (168 * 96, 128, 8): (8, 7),
    # tiny shapes (Ht 8 and 2)
    (16 * 16, 128, 8): (8, 4),
    (8 * 8, 128, 8): (8, 1),
}
# Tile-aligned levels no grid with x = 8 serves (ttnn would suggest a smaller x, which the packed gamma / beta / mask
# layout of the module cannot follow): ``_core_grid_for`` raises and ``GroupNorm.forward`` takes ``_forward_unaligned``.
PLAIN_GRID_REFUSED = {
    (584 * 440, 64, 8): (4, 5),  # 880x1168 /2, Ht 8030
    (184 * 100, 128, 8): (4, 5),  # 400x736 /4, Ht 575
    (408 * 616, 64, 8): (6, 7),  # 1232x816 /2, Ht 7854
    (504 * 504, 64, 8): (6, 7),  # 1008x1008 /2, Ht 7938
}


def test_plain_grid_rule(expect_error) -> None:
    """H2: the host copy of ``ttnn.group_norm``'s DRAM grid rule == ttnn's own ``_find_expected_dram_grid`` for every
    FLUX.2 Klein VAE encoder level at TP4 of the matrix canvases and the unaligned reference sizes."""
    from ttnn.operations.normalization import _find_expected_dram_grid

    for (n, c, g), expected in PLAIN_GRID_ROWS.items():
        assert n % 32 == 0, (n, c, g)
        got = plain_group_norm_grid(n, c, g)
        theirs = _find_expected_dram_grid(8, 8, c, g, n, 1)
        assert got == expected == (theirs.x, theirs.y), (n, c, g, got, expected, (theirs.x, theirs.y))
    for (n, c, g), suggested in PLAIN_GRID_REFUSED.items():
        assert n % 32 == 0, (n, c, g)
        theirs = _find_expected_dram_grid(8, 8, c, g, n, 1)
        assert (theirs.x, theirs.y) == suggested, (n, c, g, (theirs.x, theirs.y))
        with expect_error(ValueError, "no core grid with x=8"):
            plain_group_norm_grid(n, c, g)
    # the matrix canvases are tile-aligned on every encoder / decoder level: the fused op keeps its 8x8 grid there
    for width, height in [(1024, 1024), (2048, 2048)]:
        for div in (1, 2, 4, 8):
            assert ((height // div) * (width // div)) % 32 == 0, (width, height, div)
    # the unaligned reference sizes: at least one level is not a multiple of 32
    for width, height in UNALIGNED_SIZES:
        assert any(((height // div) * (width // div)) % 32 != 0 for div in (1, 2, 4, 8)), (width, height)
    logger.info(
        f"RESULT plain_grid_rule: {len(PLAIN_GRID_ROWS)} encoder levels agree with ttnn's _find_expected_dram_grid; "
        f"{len(PLAIN_GRID_REFUSED)} tile-aligned levels refused at x=8 (exact path)"
    )


UNALIGNED_SIZES = [(880, 1168), (400, 736), (1232, 816)]
"""Prepared reference sizes BFL's own sizing (``references.prepare_reference``: the 1 MiP cap, then ``floor16``)
produces for real photos (1152x1536 -> 880x1168, 415x739 -> 400x736, a 3000x2000 photo -> 1232x816; FLUX 3's
UNALIGNED_SIZES) whose VAE encoder levels have row counts that are not multiples of 32 (16060 / 4600 / 15708 rows at
the /8 level) or that the fused op's grid rule refuses (Ht 8030 / 575 / 7854 at /2, /4, /2): the plain GroupNorm takes
``_forward_unaligned`` there."""


def memory_line(device) -> str:
    try:
        dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    except Exception as error:  # noqa: BLE001
        return f"memory view n/a ({type(error).__name__})"
    return (
        f"DRAM {dram.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank; "
        f"TRACE {trace.total_bytes_allocated_per_bank} B per bank"
    )


def open_vae(mesh_device, snapshot: Path) -> VaeMesh:
    shape = tuple(mesh_device.shape)
    dit_ccl = CCLManager(mesh_device=mesh_device, num_links=NUM_LINKS, topology=TOPOLOGY)
    vae_ccl = CCLManager(mesh_device=mesh_device, num_links=NUM_LINKS, topology=TOPOLOGY)
    decoder, parallel_config = build_flux2_klein_vae_decoder(
        mesh_device, tp_axis=TP_AXIS, h_axis=H_AXIS, ccl_manager=vae_ccl
    )
    t0 = time.time()
    state = load_flux2_klein_vae_weights(decoder, snapshot, parallel_config=parallel_config, mesh_device=mesh_device)
    tag = f"{shape[0]}x{shape[1]}_sp{SP_AXIS}_tp{TP_AXIS}_h{H_AXIS}"
    logger.info(f"RECORD decoder {tag}: weights {state} in {time.time() - t0:.1f}s; {memory_line(mesh_device)}")
    helper = Flux2KleinVaeDecode(
        decoder,
        device=mesh_device,
        dit_ccl=dit_ccl,
        sp_axis=SP_AXIS,
        h_axis=H_AXIS,
        tp_axis=TP_AXIS,
        vae_ccl=vae_ccl,
        snapshot=snapshot,
    )
    return VaeMesh(mesh_device, dit_ccl, vae_ccl, decoder, helper, tag)


def upload_state(mesh: VaeMesh, tokens: torch.Tensor) -> ttnn.Tensor:
    """``[L, 128]`` fp32 host tokens -> the denoiser's fp32 slot ``[1, L_pad, 128]`` sharded over the SP axis."""
    sp = int(mesh.device.shape[SP_AXIS])
    padded = packing.pad_rows(tokens.float(), packing.padded_length(tokens.shape[0], sp))[0][None]
    return tensor.from_torch(padded, device=mesh.device, dtype=ttnn.float32, mesh_axes=[None, SP_AXIS, None])


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


@vae_rows
@golden_rows
def test_decode_golden_tokens(mesh_device, snapshot: Path, golden: Golden | None) -> None:
    if golden is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no 1024x1024 case under it")
    tokens = golden.final_tokens()
    width, height = golden.size
    h, w = packing.latent_shape(height, width)
    reference = cached_reference_decode(snapshot, tokens, h, w, f"{golden.name} final_tokens")
    mesh = open_vae(mesh_device, snapshot)
    name = f"decode[{golden.name}] {mesh.tag}"
    try:
        eager, wall_first = device_decode(mesh, tokens, height, width, traced=False)
        eager2, wall_warm = device_decode(mesh, tokens, height, width, traced=False)
        assert eager.shape == (3, height, width) and torch.isfinite(eager).all()
        p, r = pcc(reference, eager), rel_rmse_of(reference, eager)
        ref_png, dev_png = to_pil(reference), to_pil(eager)
        psnr_dec = psnr_uint8(dev_png, ref_png)
        psnr_golden = psnr_uint8(dev_png, golden.png())
        psnr_ref_golden = psnr_uint8(ref_png, golden.png())
        logger.info(
            f"RESULT {name} eager: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU fp32 decode pcc {p:.6f} "
            f"rel_rmse {r:.5f} max|d| {(reference - eager).abs().max().item():.4f}; PSNR vs CPU decode {psnr_dec:.2f} dB; "
            f"PSNR vs golden.png {psnr_golden:.2f} dB (CPU fp32 decode vs golden.png {psnr_ref_golden:.2f} dB); "
            f"second eager == first {torch.equal(eager, eager2)}; {memory_line(mesh.device)}"
        )
        out_dir = renders_dir()
        if out_dir is not None:
            dev_png.save(out_dir / f"vae_decode_{mesh.tag}_{golden.name}.png")
            ref_png.save(out_dir / f"vae_decode_cpu_fp32_{golden.name}.png")
        assert torch.equal(eager, eager2), "second eager decode differs from the first"
        traced1, wall_capture = device_decode(mesh, tokens, height, width, traced=True)
        traced2, wall_replay = device_decode(mesh, tokens, height, width, traced=True)
        logger.info(
            f"RECORD {name} traced: capture {wall_capture:.3f}s replay {wall_replay:.3f}s; traced == eager "
            f"{torch.equal(traced1, eager)}; replay == capture {torch.equal(traced1, traced2)}; {memory_line(mesh.device)}"
        )
        assert torch.equal(traced1, eager), "traced decode differs from eager"
        assert torch.equal(traced1, traced2), "replay differs from capture"
        other = other_golden(golden)
        if other is not None:
            other_tokens = other.final_tokens()
            assert other_tokens.shape == tokens.shape and not torch.equal(other_tokens, tokens)
            other_eager, _ = device_decode(mesh, other_tokens, height, width, traced=False)
            other_traced, wall_swap = device_decode(mesh, other_tokens, height, width, traced=True)
            back, _ = device_decode(mesh, tokens, height, width, traced=True)
            logger.info(
                f"RESULT {name} trace input swap: replay on {other.name} tokens == its eager decode "
                f"{torch.equal(other_traced, other_eager)} (pcc vs the {golden.name} render {pcc(eager, other_traced):.3f}); "
                f"replay back == capture {torch.equal(back, traced1)}; swap replay {wall_swap:.3f}s"
            )
            assert torch.equal(other_traced, other_eager), "replay on swapped tokens differs from their eager decode"
            assert not torch.equal(other_traced, traced1), "replay ignored the swapped input"
            assert torch.equal(back, traced1), "replay back on the capture tokens differs from the capture"
        else:
            logger.warning(f"{name}: only one 1024 golden on disk; the trace input swap is not checked")
        assert psnr_dec >= PSNR_DEC_FLOOR, f"PSNR vs CPU decode {psnr_dec:.2f} dB < {PSNR_DEC_FLOOR}"
        assert_quality(reference, eager, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


@vae_rows
@pytest.mark.parametrize("height, width", [(1024, 1024), (2048, 2048)], ids=["1024x1024", "2048x2048"])
def test_decode_local_query_attention_bit_identical(mesh_device, snapshot: Path, height: int, width: int) -> None:
    """``VaeAttention.local_queries`` (each chip attends for its own query rows: spatial shard, then its TP slice, with
    the TP group's rows all-gathered before ``to_out``) must be bit-identical to the replicated full-spatial SDPA it
    replaces (same K/V, same k chunking, per-row math) on the port's TP4 x H8 decoder. Ported from the FLUX 3 Klein
    test of the same name (max |d| 0 at 1024x1024, 1344x768, 2048x2048 there). Logs the warm eager decode wall of both
    paths; the 2048 case is the one with the measurable gain."""
    from ....models.vae.vae import VaeAttention

    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, snapshot)
    torch.manual_seed(3)
    tokens = torch.randn(h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
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


# --- D3: prepared conv weights per input shape --------------------------------------------------------------------------------

PREP_SIZES = [
    (1024, 1024),
    (2048, 2048),
]  # (height, width): the pipeline's warm-up canvas, then the largest matrix canvas


def _convs(module) -> list:
    """Every ``Conv2d`` under ``module`` (depth first, in registration order)."""
    from ....layers.conv2d import Conv2d

    found = []
    stack = [module]
    while stack:
        current = stack.pop()
        if isinstance(current, Conv2d):
            found.append(current)
        stack.extend(child for _, child in reversed(list(current.named_children())))
    return found


def _drop_prepared(convs) -> int:
    """Release every prepared (device-laid-out) weight / bias copy of ``convs`` and forget the shape keys; the raw
    weights stay loaded, so the next forward prepares afresh exactly as a fresh decoder would. Test-only: it is
    ``Conv2d.deallocate_weights`` without the raw-weight part."""
    dropped = 0
    for conv in convs:
        copies = dict(conv._prepared_by_shape)
        if conv._prepared_weight is not None or conv._prepared_bias is not None:
            copies[conv._prepared_key] = (conv._prepared_weight, conv._prepared_bias)
        for weight, bias in copies.values():
            for t in (weight, bias):
                if t is not None:
                    ttnn.deallocate(t)
                    dropped += 1
        conv._prepared_weight = conv._prepared_bias = None
        conv._prepared_by_shape = {}
        conv._prepared_key = None
    return dropped


@vae_rows
def test_decode_prepared_per_shape(mesh_device, snapshot: Path, monkeypatch) -> None:
    """D3: one decoder instance decodes 1024x1024 first (the pipeline's warm-up) and 2048x2048 after it. With the old
    single prepared copy (``FLUX2_KLEIN_CONV_PREP_CACHE=single``) every 2048 conv runs with weights
    ``ttnn.prepare_conv_weights`` laid out for the 1024 program (FLUX 3 Klein: 5.2 dB); with the default per-shape
    cache the 2048 decode is prepared for 2048 and meets the D1 gate, switching back to 1024 reuses the cached copy
    bit for bit, and a second 2048 decode equals the first. The single-copy numbers are logged, not gated: whether
    the ttnn of this tree re-prepares a mismatched weight is exactly what the row records."""
    from ....layers.conv2d import CONV_PREP_CACHE_ENV

    mesh = open_vae(mesh_device, snapshot)
    convs = _convs(mesh.decoder)
    assert convs, "no Conv2d under the decoder"
    torch.manual_seed(2)
    tokens, reference = {}, {}
    for height, width in PREP_SIZES:
        h, w = packing.latent_shape(height, width)
        tokens[(height, width)] = torch.randn(h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
        reference[(height, width)] = cached_reference_decode(
            snapshot, tokens[(height, width)], h, w, f"prepared_per_shape {width}x{height} seed 2"
        )
    small, big = PREP_SIZES
    name = f"decode_prepared_per_shape {mesh.tag}"

    def numbers(size, image) -> dict:
        ref = reference[size]
        return {
            "pcc": pcc(ref, image),
            "rel_rmse": rel_rmse_of(ref, image),
            "max_abs": float((ref - image).abs().max()),
            "psnr": psnr_uint8(to_pil(image), to_pil(ref)),
            "saturated": float(((image <= -1) | (image >= 1)).float().mean()),
        }

    def fmt(n: dict) -> str:
        ok = n["pcc"] >= DECODE_PCC and n["rel_rmse"] <= DECODE_RMSE
        return (
            f"pcc {n['pcc']:.6f} rel_rmse {n['rel_rmse']:.5f} max|d| {n['max_abs']:.4f} PSNR {n['psnr']:.2f} dB "
            f"saturated {n['saturated']:.4f} (D1 gate {'ok' if ok else 'FAIL'})"
        )

    try:
        # (1) the pre-fix path: one copy, prepared by the 1024 warm-up, reused at 2048
        monkeypatch.setenv(CONV_PREP_CACHE_ENV, "single")
        warm_single, _ = device_decode(mesh, tokens[small], *small, traced=False)
        big_single, wall_single = device_decode(mesh, tokens[big], *big, traced=False)
        assert all(c._prepared_key is None and not c._prepared_by_shape for c in convs), "single mode touched the cache"
        n_warm_single, n_big_single = numbers(small, warm_single), numbers(big, big_single)
        dropped = _drop_prepared(convs)
        ttnn.synchronize_device(mesh.device)
        logger.info(
            f"RESULT {name} single copy: 1024 warm-up {fmt(n_warm_single)}; 2048 after it {fmt(n_big_single)} "
            f"({wall_single:.3f}s); {dropped} prepared tensors dropped from {len(convs)} convs"
        )
        # (2) the default: one prepared copy per input-shape key
        monkeypatch.setenv(CONV_PREP_CACHE_ENV, "per_shape")
        warm, _ = device_decode(mesh, tokens[small], *small, traced=False)
        used = [c for c in convs if c._prepared_key is not None]
        assert used, "no conv prepared anything"
        # a key's (h, w) is the conv's own per-device input at its level, not the canvas
        key_small = {id(c): c._prepared_key for c in used}
        assert all(not c._prepared_by_shape for c in used)
        big_first, wall_big = device_decode(mesh, tokens[big], *big, traced=False)
        key_big = {id(c): c._prepared_key for c in used}
        assert all(key_big[id(c)] != key_small[id(c)] for c in used)
        assert all(set(c._prepared_by_shape) == {key_small[id(c)]} for c in used)
        back, wall_back = device_decode(mesh, tokens[small], *small, traced=False)
        # the dict keeps every key seen (the current one included once it has been switched away from): 2 copies
        assert all(c._prepared_key == key_small[id(c)] for c in used)
        assert all(set(c._prepared_by_shape) == {key_small[id(c)], key_big[id(c)]} for c in used)
        big_second, wall_big2 = device_decode(mesh, tokens[big], *big, traced=False)
        n_warm, n_big = numbers(small, warm), numbers(big, big_first)
        logger.info(
            f"RESULT {name} per shape: 1024 warm-up {fmt(n_warm)}; 2048 after it {fmt(n_big)} ({wall_big:.3f}s, "
            f"second {wall_big2:.3f}s); 1024 again == warm-up {torch.equal(back, warm)} ({wall_back:.3f}s); "
            f"second 2048 == first {torch.equal(big_second, big_first)}; {len(used)}/{len(convs)} convs hold "
            f"one prepared copy per shape (2); single-copy 2048 == per-shape 2048 {torch.equal(big_single, big_first)}; "
            f"{memory_line(mesh.device)}"
        )
        out_dir = renders_dir()
        if out_dir is not None:
            to_pil(big_single).save(out_dir / f"vae_prep_single_2048_{mesh.tag}.png")
            to_pil(big_first).save(out_dir / f"vae_prep_per_shape_2048_{mesh.tag}.png")
            to_pil(reference[big]).save(out_dir / "vae_prep_cpu_fp32_2048.png")
        assert big_first.shape == (3, *big) and torch.isfinite(big_first).all()
        assert torch.equal(back, warm), "the 1024 decode after the 2048 one differs from the warm-up render"
        assert torch.equal(big_second, big_first), "the second 2048 decode differs from the first"
        assert_quality(reference[small], warm, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
        assert_quality(reference[big], big_first, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


# --- D2: canvases whose x1 decoder level needs the GroupNorm sub-grid -----------------------------------------------------------

# (height, width): 1344(w)x768(h) -> latent h 48, w 84 (h % 8 == 0: the shipped TP4 x H8 decoder); 896(w)x1120(h) -> h 70,
# w 56 (h % 8 != 0: the replicated TP4 fallback decoder). Both are admitted by ``policy.admit`` (every VAE level has H * W
# a multiple of 32), so before ``GroupNorm._core_grid_for`` they reached the device and ``ttnn.group_norm`` threw in
# ``validate_dram_grid`` (16 virtual rows do not divide Ht 504 / 490 / 1960). The levels and the sub-grids expected:
# (rows at the level, channels per device) -> (x, y), the same arithmetic as PLAIN_GRID_ROWS for the decoder's levels.
SUBGRID_CANVASES = [(768, 1344), (1120, 896)]
SUBGRID_LEVELS = {
    (768, 1344): {(96 * 168, 128): (8, 7), (192 * 336, 128): (8, 8), (384 * 672, 64): (8, 8), (768 * 1344, 32): (8, 8)},
    (1120, 896): {
        (140 * 112, 128): (8, 7),
        (280 * 224, 128): (8, 7),
        (560 * 448, 64): (8, 8),
        (1120 * 896, 32): (8, 8),
    },
}


@vae_rows
@pytest.mark.parametrize("height, width", SUBGRID_CANVASES, ids=[f"{w}x{h}" for h, w in SUBGRID_CANVASES])
def test_decode_subgrid_canvas(mesh_device, snapshot: Path, monkeypatch, height: int, width: int) -> None:
    """D2: a canvas whose x1 decoder level the fused op's 8x8 grid rule rejects decodes on the sub-grid
    ``GroupNorm._core_grid_for`` picks ((8, 7) for Ht 504 / 490 / 1960; every shape the 8x8 grid accepted keeps it),
    on the shipped H8 decoder (1344x768) and on the replicated fallback decoder (896x1120), vs the fp32 CPU decode at
    the D1 gate. The grids returned during the decode are recorded through ``_core_grid_for`` and asserted."""
    from ....layers.normalization import GroupNorm

    h, w = packing.latent_shape(height, width)
    mesh = open_vae(mesh_device, snapshot)
    partitioned = mesh.helper.partitioned(height, width)
    assert partitioned == (h % 8 == 0), (height, width, h, partitioned)
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    original = GroupNorm._core_grid_for

    def recording(self, rows, batch_size):
        grid = original(self, rows, batch_size)
        seen[(int(rows), int(self.num_padded_channels))] = (grid.x, grid.y)
        return grid

    monkeypatch.setattr(GroupNorm, "_core_grid_for", recording)
    torch.manual_seed(1)
    tokens = torch.randn(h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS).to(torch.bfloat16).float()
    name = f"decode_subgrid_canvas[{width}x{height}] {mesh.tag} {'H8' if partitioned else 'replicated'}"
    try:
        reference = cached_reference_decode(snapshot, tokens, h, w, f"random {width}x{height}")
        image, wall_first = device_decode(mesh, tokens, height, width, traced=False)
        image2, wall_warm = device_decode(mesh, tokens, height, width, traced=False)
        p, r = pcc(reference, image), rel_rmse_of(reference, image)
        dev_png, ref_png = to_pil(image), to_pil(reference)
        psnr_dec = psnr_uint8(dev_png, ref_png)
        grids = {k: v for k, v in sorted(seen.items())}
        logger.info(
            f"RESULT {name}: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU fp32 decode pcc {p:.6f} rel_rmse "
            f"{r:.5f} max|d| {(reference - image).abs().max().item():.4f}; PSNR vs CPU decode {psnr_dec:.2f} dB; second "
            f"eager == first {torch.equal(image, image2)}; plain GroupNorm grids (rows, channels) -> (x, y): {grids}; "
            f"{memory_line(mesh.device)}"
        )
        out_dir = renders_dir()
        if out_dir is not None:
            dev_png.save(out_dir / f"vae_decode_{mesh.tag}_random_{width}x{height}.png")
            ref_png.save(out_dir / f"vae_decode_cpu_fp32_random_{width}x{height}.png")
        assert image.shape == (3, height, width) and torch.isfinite(image).all()
        assert torch.equal(image, image2), "second eager decode differs from the first"
        expected = SUBGRID_LEVELS[(height, width)]
        for key, grid in expected.items():
            if grid != (8, 8) or not partitioned:
                # the H8 decoder's plain GroupNorm is only the attention's (x1 level, all-gathered rows); the
                # replicated decoder runs every level through the plain GroupNorm
                assert seen.get(key) == grid, (key, grid, seen.get(key))
        assert all(x == 8 for x, _ in seen.values()), seen
        assert_quality(reference, image, pcc=DECODE_PCC, relative_rmse=DECODE_RMSE)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)


# --- device: encode ---------------------------------------------------------------------------------------------------------


@dataclass
class EncoderMesh:
    device: ttnn.MeshDevice
    ccl: CCLManager
    encoder: Flux2VaeEncoder
    helper: Flux2KleinVaeEncode
    tag: str


def open_encoder(mesh_device, snapshot: Path) -> EncoderMesh:
    shape = tuple(mesh_device.shape)
    ccl = CCLManager(mesh_device=mesh_device, num_links=NUM_LINKS, topology=TOPOLOGY)
    encoder, parallel_config = build_flux2_klein_vae_encoder(mesh_device, tp_axis=TP_AXIS, ccl_manager=ccl)
    t0 = time.time()
    state = load_flux2_klein_vae_encoder_weights(
        encoder, snapshot, parallel_config=parallel_config, mesh_device=mesh_device
    )
    tag = f"{shape[0]}x{shape[1]}_tp{TP_AXIS}_replicated"
    logger.info(f"RECORD encoder {tag}: weights {state} in {time.time() - t0:.1f}s; {memory_line(mesh_device)}")
    return EncoderMesh(mesh_device, ccl, encoder, Flux2KleinVaeEncode(encoder, device=mesh_device), tag)


def prepared_png(golden: Golden):
    image = prepare_reference(golden.png())
    assert image.size == golden.size, (golden.name, image.size)
    return image


def encode_report(name: str, oracle: torch.Tensor, got: torch.Tensor, wall_first: float, wall_warm: float, mem: str):
    got32 = got.float()
    p, r, m = pcc(oracle, got32), rel_rmse_of(oracle, got32), (oracle - got32).abs().max().item()
    p_bf16 = pcc(oracle.to(torch.bfloat16).float(), got32)
    equal_bf16 = torch.equal(oracle.to(torch.bfloat16), got)
    logger.info(
        f"RESULT {name}: first {wall_first:.3f}s warm {wall_warm:.3f}s; vs CPU fp32 encode pcc {p:.6f} "
        f"rel_rmse {r:.5f} max|d| {m:.4f} (oracle std {oracle.std().item():.4f}); vs the oracle rounded to bf16 pcc "
        f"{p_bf16:.6f} bit-equal {equal_bf16}; {mem}"
    )
    return p, r


def gate_encode(p: float, r: float) -> None:
    assert p > ENCODE_PCC, f"encode pcc {p:.6f} <= {ENCODE_PCC}"
    if ENCODE_RMSE is not None:
        assert r <= ENCODE_RMSE, f"encode rel_rmse {r:.5f} > {ENCODE_RMSE}"


@vae_rows
@golden_rows
def test_encode_golden_png(mesh_device, snapshot: Path, golden: Golden | None) -> None:
    if golden is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no 1024x1024 case under it")
    image = prepared_png(golden)
    width, height = image.size
    h, w = packing.latent_shape(height, width)
    oracle, oracle_ids = cached_reference_encode(snapshot, image, f"{golden.name} golden.png")
    assert oracle.shape == (h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS)
    assert torch.equal(oracle_ids, packing.reference_ids(h, w, 0)), "pipeline image ids != packing.reference_ids"
    mesh = open_encoder(mesh_device, snapshot)
    row = f"encode[{golden.name}] {width}x{height} {mesh.tag}"
    try:
        first = mesh.helper.encode(image)
        wall_first = mesh.helper.last_wall_s
        second = mesh.helper.encode(image)
        wall_warm = mesh.helper.last_wall_s
        assert first.shape == (h * w, 128) and first.dtype == torch.bfloat16 and torch.isfinite(first.float()).all()
        p, r = encode_report(row, oracle, first, wall_first, wall_warm, memory_line(mesh.device))
        assert torch.equal(first, second), "second encode differs from the first"
        traced1 = mesh.helper.encode(image, traced=True)
        wall_capture = mesh.helper.last_wall_s
        traced2 = mesh.helper.encode(image, traced=True)
        wall_replay = mesh.helper.last_wall_s
        logger.info(
            f"RECORD {row} traced: capture {wall_capture:.3f}s replay {wall_replay:.3f}s; traced == eager "
            f"{torch.equal(traced1, first)}; replay == capture {torch.equal(traced1, traced2)}; {memory_line(mesh.device)}"
        )
        assert torch.equal(traced1, first) and torch.equal(traced1, traced2)
        other = other_golden(golden)
        if other is not None:
            other_image = prepared_png(other)
            other_eager = mesh.helper.encode(other_image)
            other_traced = mesh.helper.encode(other_image, traced=True)
            back = mesh.helper.encode(image, traced=True)
            logger.info(
                f"RESULT {row} trace input swap: replay on {other.name} == its eager encode "
                f"{torch.equal(other_traced, other_eager)} (pcc vs {golden.name} {pcc(first.float(), other_traced.float()):.3f}); "
                f"replay back == capture {torch.equal(back, traced1)}"
            )
            assert torch.equal(other_traced, other_eager) and not torch.equal(other_traced, traced1)
            assert torch.equal(back, traced1)
        else:
            logger.warning(f"{row}: only one 1024 golden on disk; the trace input swap is not checked")
        gate_encode(p, r)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)
        mesh.encoder.deallocate_weights()
        ttnn.synchronize_device(mesh.device)


@vae_rows
@pytest.mark.parametrize("size", UNALIGNED_SIZES, ids=[f"{w}x{h}" for w, h in UNALIGNED_SIZES])
def test_encode_unaligned_reference(mesh_device, snapshot: Path, size: tuple[int, int]) -> None:
    """E2: a reference whose VAE levels are NOT tile-aligned encodes through the exact unaligned GroupNorm path and
    matches the CPU fp32 encode at the E1 gate (FLUX 3 test_encode_unaligned_reference; the sizes are the ones
    ``prepare_reference`` keeps as they are: 1 MiP or less, both sides multiples of 16)."""
    from PIL import Image

    width, height = size
    rng = torch.Generator().manual_seed(width * 7919 + height)
    noise = torch.rand(height, width, 3, generator=rng)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    rgb = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=-1) * 0.7 + 0.3 * noise  # smooth gradient + noise
    image = Image.fromarray((rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy(), "RGB")
    assert prepare_reference(image).size == image.size, "the size is not what prepare_reference would keep"
    h, w = packing.latent_shape(height, width)
    oracle, oracle_ids = cached_reference_encode(snapshot, image, f"unaligned {width}x{height}")
    assert oracle.shape == (h * w, packing.FLUX2_KLEIN_LATENT_CHANNELS)
    assert torch.equal(oracle_ids, packing.reference_ids(h, w, 0)), "pipeline image ids != packing.reference_ids"
    mesh = open_encoder(mesh_device, snapshot)
    row = f"encode_unaligned[{width}x{height}] {mesh.tag}"
    try:
        first = mesh.helper.encode(image)
        wall_first = mesh.helper.last_wall_s
        second = mesh.helper.encode(image)
        wall_warm = mesh.helper.last_wall_s
        assert first.shape == (h * w, 128) and first.dtype == torch.bfloat16 and torch.isfinite(first.float()).all()
        assert torch.equal(first, second), "second encode differs from the first"
        p, r = encode_report(row, oracle, first, wall_first, wall_warm, memory_line(mesh.device))
        gate_encode(p, r)
    finally:
        mesh.helper.release_trace()
        ttnn.synchronize_device(mesh.device)
        mesh.encoder.deallocate_weights()
        ttnn.synchronize_device(mesh.device)


UPLOAD_SIZES = [(1024, 1024), (1344, 768), (880, 1168)]


@vae_rows
@pytest.mark.parametrize("size", UPLOAD_SIZES, ids=[f"{w}x{h}" for w, h in UPLOAD_SIZES])
def test_encode_upload_replication_bit_identical(mesh_device, snapshot: Path, size: tuple[int, int]) -> None:
    """E3: the sharded upload + on-device gather feeds the encoder the same bytes as the host replication: tokens
    ``torch.equal`` between ``replicate_on_device`` True and False (eager, warm walls of both logged). 1024x1024 and
    1344x768 have a ``sharded_upload_view``; 880x1168 has none and must report the ``replicated`` path either way."""
    from PIL import Image

    width, height = size
    rng = torch.Generator().manual_seed(width * 31 + height)
    noise = torch.rand(height, width, 3, generator=rng)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    rgb = torch.stack([0.5 * (xx + yy), yy, xx], dim=-1) * 0.6 + 0.4 * noise
    image = Image.fromarray((rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy(), "RGB")
    assert prepare_reference(image).size == image.size, "the size is not what prepare_reference would keep"
    h, w = packing.latent_shape(height, width)
    expected_path = (
        "sharded" if sharded_upload_view(height, width, tuple(mesh_device.shape)[1 - TP_AXIS]) else "replicated"
    )
    mesh = open_encoder(mesh_device, snapshot)
    helper = mesh.helper
    row = f"encode_upload[{width}x{height}] {mesh.tag}"
    try:
        helper.replicate_on_device = False
        helper.encode(image)  # compile
        host = [helper.encode(image) for _ in range(3)]
        host_walls = (helper.last_upload_s, helper.last_wall_s)
        assert helper.last_upload_path == "replicated"
        helper.replicate_on_device = True
        first = helper.encode(image)  # the gather's first call (program cache)
        assert helper.last_upload_path == expected_path, (helper.last_upload_path, expected_path)
        device = [helper.encode(image) for _ in range(3)]
        device_walls = (helper.last_upload_s, helper.last_wall_s)
        assert first.shape == (h * w, 128) and first.dtype == torch.bfloat16 and torch.isfinite(first.float()).all()
        assert all(torch.equal(t, host[0]) for t in host[1:]), "host-replicated encode is not deterministic"
        assert all(torch.equal(t, first) for t in device), "sharded-upload encode is not deterministic"
        assert torch.equal(first, host[0]), "sharded upload + on-device gather changed the encoder's input bytes"
        logger.info(
            f"RESULT {row}: path {helper.last_upload_path} (expected {expected_path}); tokens torch.equal True; "
            f"warm upload wall host-replicated {host_walls[0] * 1e3:.1f} ms / {expected_path} {device_walls[0] * 1e3:.1f} ms; "
            f"encode wall {host_walls[1]:.3f} s / {device_walls[1]:.3f} s; {memory_line(mesh.device)}"
        )
    finally:
        helper.release_trace()
        ttnn.synchronize_device(mesh.device)
        mesh.encoder.deallocate_weights()
        ttnn.synchronize_device(mesh.device)
