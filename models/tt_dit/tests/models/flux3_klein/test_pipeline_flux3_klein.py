# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M7: the FLUX 3 Image Klein text-to-image pipeline (``pipelines/flux3_klein/pipeline_flux3_klein.py``) end to end
on the Galaxy 4x8 ring: device text encoder -> traced denoise -> device VAE decode -> PIL. Design record
``docs/M6_M7_DESIGN.md`` sections 5-6.

Rows (mesh ``bh_4x8_ring`` only: the (4, 8) parent, SP4 axis 0 x TP8 axis 1, Ring, 2 links, 8 KB packets; both
DiTs, the text encoder and the VAE resident; goldens from ``$FLUX3_KLEIN_GOLDENS``; PNGs under ``$FLUX3_KLEIN_OUT``,
never inside the tree):
  P0 test_request_resolve (host)   ``Flux3KleinRequest.resolve`` defaults (50 / 3.5, 16 / None), the flow-map
                          guidance error, ``policy.admit`` rejections (side, references), ``make_noise(7, 64, 64)``
                          sha256 == the goldens' ``record.json`` ``noise_sha256`` (base AND distilled man_dog).
  P1 test_t2i_golden_text[base|distilled]   ``text_override`` = the golden's recorded text (and empty-prompt)
                          encodings; one ``generate``: ``final_tokens`` ``torch.equal`` the denoiser's own traced
                          ``denoise(request_from_golden)`` in the same process (bit-identical to the M4 R3 traced run);
                          image (H, W, 3) finite; PSNR vs ``golden.png`` >= PSNR_INTERIM[variant]; PSNR vs the CPU
                          reference decode of the same ``final_tokens`` >= PSNR_DEC_FLOOR (isolates the VAE); walls.
  P2 test_t2i_device_text[base|distilled]   the recorded prompt through the DEVICE text encoder; ``generate`` twice:
                          the second ``final_tokens`` and text encoding ``torch.equal`` the first (determinism across
                          replays and encodes); encode / denoise / decode / total walls = the STATUS section 1 e2e
                          row; the digest is written to ``$FLUX3_KLEIN_OUT/digests.json`` for P3.
                          base: PSNR vs ``golden.png`` >= PSNR_INTERIM["base"] (the device text costs 0.03 dB there,
                          inside the band's margin); final-state pcc vs the golden logged (0.981: the bf16 reference's
                          own sensitivity, M4). The base CPU reference with the device text (50 steps x 2 forwards, ~2 h
                          of node CPU) is not run: M8 item.
                          distilled: gated against its PROPER reference, the CPU reference sampler run on the SAME
                          device text (``cached_reference_sampling``, 9 min on the node CPU, cached): final pcc /
                          rel_rmse at the M4 distilled band (0.99 / 0.15, ``test_denoise_flux3_klein.final_gate``) and
                          PSNR of the device render vs that reference's render >= PSNR_PROPER_REFERENCE. Carried as
                          ``xfail(strict=True)``: measured 0.9855 / 0.170 / 29.99 dB (gate comment below), so the band
                          is NOT met for this input; an XPASS is the signal to drop the marker.
  P3 test_both_models_resident   base, distilled, base in one process (the P2 prompts): each ``final_tokens`` equals
                          its P2 digest when one is on disk (else logged) and the third call equals the first; TRACE /
                          DRAM per bank after each; whether the base trace was recaptured on the third call is LOGGED:
                          the pipeline releases every live trace before a prepare that allocates a new layout (the M4
                          aliasing rule across modules, pipeline docstring), so the distilled request in between
                          forces a base recapture by design. Quality: base calls at PSNR_INTERIM["base"]; the distilled
                          call's verdict is P2's (bit-equality with the P2 digest), no band of its own.
  P4 test_trace_alloc_tracking   opt-in (``TT_METAL_TRACE_ALLOC_TRACKING=1`` on the pytest process): the P2 base
                          request twice (capture, then the replay path incl. the VAE input copy); the log must carry 0
                          tracker lines (M4 precedent; checked by the operator's grep). M8: the second generate's first
                          DiT replay used to raise ``Found 2 device buffer(s) still alive before trace replay`` (the VAE
                          trace's captured input and its CCL gather buffer, allocated by the first generate's decode
                          AFTER the DiT capture; M7 had only ever run ONE generate under the tracker;
                          logs/m8_verifier_device_tracker_4x8.log:37-49,114,180-186,252-266). The pipeline now captures
                          the VAE trace of a canvas size BEFORE the DiT trace of a layout (pipeline docstring), and this
                          row is the gate: run it with the tracker env on the ring and cite its passing line.
  test_t2i_b300             the real >= 30 dB gate vs our own B300 render: skipped until ``$FLUX3_KLEIN_GOLDENS``
                          carries a ``record.json`` with ``"device": "cuda"`` (PLAN decisions 4 and 8).
M8 rows (references; design ``docs/M8_DESIGN.md`` section 9.4; cases = the goldens with references under
``$FLUX3_KLEIN_GOLDENS``: distilled / base ``3_bandana`` (1 ref, seed 22), ``2_3_mugprint`` (2 refs, seed 23),
``1_figurine`` (896x1120, seed 33); a case whose golden is not on disk is skipped by name):
  P5a test_edit_golden_inputs[case]  ``Flux3KleinRequest(images=<the golden's reference images>, width / height None,
                          text_override=<recorded text>, image_cond_override=<recorded cond_tokens / cond_ids>)``: the
                          canvas == ``replay.size`` (default_size of the prepared references); ``final_tokens``
                          ``torch.equal`` the denoiser's own traced ``denoise(request_from_golden)`` (the M4 path, now with
                          image_cond); teacher-forced per-step velocity floor ``teacher_forced_floor`` (0.998); final
                          ``(pcc, rel_rmse)`` vs the golden inside ``final_gate`` (0.99 / 0.15 distilled, 0.97 / 0.25
                          base 50); render vs the CPU decode of its own ``final_tokens`` >= PSNR_DEC_FLOOR (896x1120:
                          its own frozen floor); PSNR vs ``golden.png`` LOGGED only (calibration band: CPU golden vs the
                          shipped BFL PNG 39.2-39.5 dB on man_dog, 57.19 / 56.22 / 32.45 dB on bandana / mugprint /
                          figurine, logs/m8_designer_golden_distilled_*.log:31; recorded-text T2I rows 28.4 / 32.9
                          dB); the admission's ``total_tokens`` == the record's token sum. ``distilled/1_figurine`` is a
                          STRICT xfail at the unchanged band with its measurement (0.976069 / 0.2189, floor met at
                          0.998783: ``test_denoise_flux3_klein.FINAL_BAND_XFAIL_CASES``), never a lowered band.
  P5b test_edit_device_path[case]  the same request through the DEVICE text encoder and the DEVICE VAE encoder (no
                          overrides), ``generate`` twice: deterministic; device cond tokens vs the golden ``cond_tokens``
                          logged; the proper reference = the CPU sampler on the device's OWN text and cond rows
                          (``cached_reference_sampling``, key extended with the cond rows): final band ``final_gate`` and
                          render >= PSNR_PROPER_REFERENCE vs that reference's render. Distilled rows run the oracle by
                          default (28 / 55 / 27 min of node CPU once, cached); base rows need
                          ``FLUX3_KLEIN_BASE_PROPER_REFERENCE=1`` (2.5-5.5 h each) and otherwise log their numbers and
                          skip the band. A distilled row that misses the band is carried as a strict xfail WITH its
                          measurement (the P2 distilled precedent), never by a lowered band.
  P5c test_edit_crop_back    a 1016x1016 synthetic reference, distilled, ``steps=1``: canvas 1024x1024, image (1016, 1016),
                          ``crop == (4, 4, 1020, 1020)``, ``image_tokens == 4096`` (shape / crop only, no quality band;
                          the design's 1000x1000 example pads to a 1008 canvas whose /8 VAE level is not tile-aligned and
                          is refused by the policy, see test_references_flux3_klein.py).
  P6 test_admission_before_device  a structurally invalid request (a 4112 side, + 10 references + the prompt): ``generate``
                          raises ``Flux3KleinAdmissionError`` with ``budget == "side"`` and leaves the denoisers'
                          captured keys, the DRAM / TRACE per bank and the encoder untouched, AND makes zero calls into
                          any device entry point (``DeviceCallCounter``: ``ttnn.from_torch`` / ``to_device`` /
                          ``allocate_tensor_on_device`` / ``copy``, ``utils.tensor.from_torch``, the VAE encoder's
                          ``encode`` / ``upload``, the prompt encoder, the VAE decode, the denoiser's ``prepare`` /
                          ``denoise_on_device``); a refusal that uploaded and freed a tensor would pass the memory check
                          alone (M8 verifier finding; measured 0 calls on the ring for six refusals,
                          logs/m8_verifier_device_probe_4x8_run3.log:274-292, run2:305).
  P3 gains a distilled 1-reference call (bandana, device path) between the two base calls when that golden exists.

Environment: ``FLUX3_KLEIN_MODEL_PATH`` / the HuggingFace cache (base and distilled snapshots), ``FLUX3_KLEIN_GOLDENS``,
``FLUX3_KLEIN_VAE_ORACLE_CACHE``, ``FLUX3_KLEIN_SAMPLING_ORACLE_CACHE``, ``FLUX3_KLEIN_OUT``, ``TT_DIT_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.denoise_flux3_klein import Flux3KleinDenoiseRequest
from ....pipelines.flux3_klein.pipeline_flux3_klein import (
    MODELS,
    Flux3KleinAdmissionError,
    Flux3KleinPipeline,
    Flux3KleinRequest,
    Flux3KleinResult,
    caption_seed,
    make_noise,
    noise_sha256,
    resolve_caption,
    to_pil,
)
from ....pipelines.flux3_klein.policy import REFERENCES_REASON, admit
from .common import (
    GOLDENS_ENV,
    GoldenCase,
    golden_cases,
    golden_index,
    golden_prompt,
    golden_reference_images,
    load_golden_case,
    pcc,
    psnr_uint8,
    reference_golden_cases,
    renders_dir,
    sampling_oracle_cache,
    sha256_of,
    variant_checkpoint_dir,
)
from .device_params import ring_params_8k_flux3_klein_transformer
from .test_denoise_flux3_klein import (
    FinalBandNotMet,
    case_model,
    final_gate,
    golden_case_param,
    reference_final,
    request_from_golden,
    teacher_forced_floor,
    teacher_forced_table,
)
from .test_transformer_flux3_klein import checkpoint_blob, torch_dit
from .test_vae_flux3_klein import PSNR_DEC_FLOOR, PSNR_DEC_FLOOR_896x1120, cached_reference_decode

# Interim PSNR gates vs golden.png (the CPU bf16 reference render), design 6.2. Measured band: the M4 device final
# states decoded on CPU give 28.40 / 32.89 dB (logs/m4gate_decode_psnr.log:3,8); the same states through the DEVICE
# VAE give 28.40 / 32.86 dB (logs/m67_designer_vae_decode_4x8.log, RESULT "M4-device-state" lines). Gate = measured
# rounded down to 0.5 dB minus a 1.0 dB margin: 28.40 -> 28.0 - 1.0 = 27.0; 32.86 -> 32.5 - 1.0 = 31.5. These are
# calibration bands at the measured device trajectory, NOT quality acceptance: the ~0.98 final-state pcc is the bf16
# reference's own sensitivity (docs/STATUS.md M4). They gate the recorded-text rows (P1) and the base device-text rows
# (P2 / P3 base: measured 28.37 dB with the device text vs 28.40 recorded, -0.03 dB, inside the margin;
# logs/m67_builder_pipeline_4x8.log:1703, logs/m67_verifier_pipeline_4x8.log:321).
PSNR_INTERIM = {"base": 27.0, "distilled": 31.5}
# The distilled DEVICE-text row (P2 / P3 distilled) is NOT gated vs golden.png: golden.png is the trajectory of the
# RECORDED text, and the device text differs from it (concat pcc 0.999986, M5). Its proper reference is the CPU
# reference sampler run on the SAME device text (cached_reference_sampling below). Measured 2026-10-17 on the 4x8 ring
# (M6/M7 verifier, logs/m67_verifier_cpu_distilled_devtext.log RESULT, m67_verifier_compare_traj.log,
# m67_verifier_device_devtext_teacher.log:107,124):
#   - the text swap is benign through the reference math: CPU reference with the device text vs the golden
#     final_tokens pcc 0.999590 / rel_rmse 0.0286, its render 42.81 dB vs golden.png;
#   - the device trajectory vs THAT reference: final pcc 0.985549 / rel_rmse 0.1704, render 29.99 dB (vs golden.png
#     30.03 dB, pcc 0.985898); the recorded-text device row sits at 0.993042 / 0.1183 (M4 band 0.99 / 0.15 met);
#   - the per-step DiT floor is identical on both inputs: teacher-forced velocity pcc worst 0.998616 (device text)
#     vs 0.998640 (recorded text), equal to 1e-5 from step 2 on. The device-vs-reference gap grows x1.45 per step on
#     both inputs from a 1.4x larger start (rel_rmse 0.0015 vs 0.0011 at step 3): input-dependent trajectory
#     sensitivity of the bf16 device-vs-reference difference, not a per-step regression and not an amplification of
#     the text difference by the flow map (the builder's first explanation, withdrawn).
# So the M4 distilled band (0.99 / 0.15) is NOT met for this input against the proper reference; the row is carried as
# xfail(strict=True, raises=ProperReferenceBandNotMet) with these numbers until the B300 render (the acceptance gate,
# PLAN decisions 4 and 8) exists. The earlier 29.0 dB band vs golden.png (set from the 30.03 dB measurement) is
# withdrawn: it compared against the wrong reference. PSNR_PROPER_REFERENCE = the PLAN's 30 dB end-to-end number,
# applied to the device render vs the reference render of the same input (measured 29.99 dB).
PSNR_PROPER_REFERENCE = 30.0
# Oracle sanity (not a device gate): the reference run on the device text must stay near the golden (measured pcc
# 0.999590); anything below this means the oracle took a different input, not that the device changed.
REFERENCE_TEXT_SWAP_PCC = 0.999
# The real gate: PSNR vs our own B300 render of the same seed and recorded prompt (PLAN decisions 4 and 8; pending).
PSNR_B300 = 30.0
ORACLE_THREADS = 24
RESIDENT = ("base", "distilled")
DIGESTS_FILE = "digests.json"
# M8: the proper-reference CPU runs of the base reference rows cost 2.5-5.5 h each; opt-in.
BASE_PROPER_REFERENCE_ENV = "FLUX3_KLEIN_BASE_PROPER_REFERENCE"
REFERENCE_CASES = reference_golden_cases()
REFERENCE_CASE_IDS = [f"{p.parent.name}-{p.name}" for p in REFERENCE_CASES]
# P5a params: the recorded final-band misses are strict xfails with their measurement (test_denoise FINAL_BAND_XFAIL_CASES).
REFERENCE_CASE_PARAMS = [golden_case_param(p) for p in REFERENCE_CASES]

mesh_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), ring_params_8k_flux3_klein_transformer]],
    ids=["bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)


# --- goldens ------------------------------------------------------------------------------------------------------------------


def golden_for(variant: str) -> GoldenCase:
    for case_dir in golden_cases():
        if case_model(case_dir) == variant:
            return load_golden_case(case_dir)
    pytest.skip(f"no complete {variant} case under {GOLDENS_ENV}")


def golden_png(golden: GoldenCase):
    from PIL import Image

    path = golden.path / "golden.png"
    if not path.is_file():
        pytest.skip(f"{path} missing")
    return Image.open(path).convert("RGB")


def request_for(golden: GoldenCase, *, text_override: bool) -> Flux3KleinRequest:
    replay = golden.replay
    width, height = (int(v) for v in replay["size"])
    override = None
    if text_override:
        inputs = golden.inputs
        null = (inputs["text_null_tokens"], inputs["text_null_ids"]) if golden.model == "base" else None
        override = ((inputs["text_tokens"], inputs["text_ids"]), null)
    return Flux3KleinRequest(
        prompt=golden_prompt(golden),
        model=golden.model,
        width=width,
        height=height,
        seed=int(replay["seed"]),
        steps=golden.num_steps,
        guidance=golden.guidance,
        alpha=golden.alpha,
        text_override=override,
    )


def snapshots_or_skip() -> dict[str, Path]:
    out = {}
    for variant in RESIDENT:
        found = variant_checkpoint_dir(variant)
        if found is None:
            pytest.skip(f"FLUX 3 Klein {variant} snapshot not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
        out[variant] = found
    return out


def build_pipeline(mesh_device) -> Flux3KleinPipeline:
    t0 = time.time()
    pipeline = Flux3KleinPipeline(mesh_device, snapshots=snapshots_or_skip(), resident=RESIDENT)
    logger.info(f"RECORD pipeline init {time.time() - t0:.1f}s: {pipeline.timings_init}")
    return pipeline


def mesh_tag(mesh_device) -> str:
    shape = tuple(mesh_device.shape)
    return f"{shape[0]}x{shape[1]}_sp{shape[0]}_tp{shape[1]}_ring"


def save_render(result: Flux3KleinResult, mesh_device, label: str) -> None:
    out_dir = renders_dir()
    if out_dir is None:
        return
    width, height = result.image.size
    path = out_dir / f"m7_{mesh_tag(mesh_device)}_{result.model}_{width}x{height}_seed{result.seed}_{label}.png"
    result.image.save(path)
    logger.info(f"render saved: {path}")


def digests_path() -> Path | None:
    out_dir = renders_dir()
    return None if out_dir is None else out_dir / DIGESTS_FILE


def record_digest(key: str, digest: str) -> None:
    path = digests_path()
    if path is None:
        return
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[key] = digest
    path.write_text(json.dumps(data, indent=1))


def stored_digest(key: str) -> str | None:
    path = digests_path()
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text()).get(key)


class ProperReferenceBandNotMet(AssertionError):
    """The device trajectory misses the M4 band against the CPU reference run on the same input (the xfail reason of
    the distilled device-text row; any other assertion in that row is a real failure)."""


def rel_rmse(reference: torch.Tensor, got: torch.Tensor) -> float:
    a, b = reference.double(), got.double()
    return float(((a - b) ** 2).mean().sqrt() / a.std())


def cached_reference_sampling(
    snapshot: Path, variant: str, request: Flux3KleinDenoiseRequest, label: str
) -> torch.Tensor:
    """The vendored sampler (``sampling.denoise`` / ``denoise_flow_map``, bf16 autocast, CPU) with the real ``variant``
    DiT on ``request``: the proper reference of a device trajectory that consumed the same text. fp32 ``[L, 128]``
    final tokens through ``$FLUX3_KLEIN_SAMPLING_ORACLE_CACHE`` (key: DiT blob + every request tensor + schedule)."""
    from safetensors.torch import load_file, save_file

    fields = {
        "blob": checkpoint_blob(snapshot),
        "variant": variant,
        "noise": sha256_of(request.noise),
        "image_ids": sha256_of(request.image_ids),
        "text": sha256_of(request.text[0]),
        "text_ids": sha256_of(request.text[1]),
        "text_null": sha256_of(request.text_null[0]) if request.text_null is not None else "",
        "cond_tokens": sha256_of(request.image_cond[0]) if request.image_cond is not None else "",
        "cond_ids": sha256_of(request.image_cond[1]) if request.image_cond is not None else "",
        "num_steps": request.num_steps,
        "guidance": request.guidance,
        "alpha": request.alpha,
        "torch": torch.__version__,
    }
    digest = hashlib.sha256("|".join(f"{k}={fields[k]}" for k in fields).encode()).hexdigest()
    root = sampling_oracle_cache()
    if root is not None and (root / f"{digest}.safetensors").is_file():
        logger.info(f"SAMPLING ORACLE HIT {digest[:16]} ({label})")
        return load_file(str(root / f"{digest}.safetensors"))["final_tokens"]
    logger.info(
        f"SAMPLING ORACLE MISS {digest[:16]} ({label}): CPU reference {variant} {request.num_steps} steps "
        f"(cfg {request.cfg}) on {ORACLE_THREADS} threads"
    )
    torch.set_num_threads(ORACLE_THREADS)
    t0 = time.time()
    final = reference_final(torch_dit(variant, snapshot), request, request.num_steps).float().contiguous()
    logger.info(f"SAMPLING ORACLE {label}: {time.time() - t0:.0f} s")
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        save_file({"final_tokens": final}, str(root / f"{digest}.safetensors"))
        (root / f"{digest}.json").write_text(json.dumps({**fields, "label": label}, indent=1))
    return final


def report(name: str, result: Flux3KleinResult, golden: GoldenCase, png, gates=PSNR_INTERIM) -> dict[str, float]:
    """PSNR vs golden.png, pcc vs the golden final_tokens, digest; returns the numbers (the caller gates)."""
    psnr_golden = psnr_uint8(result.image, png)
    p = pcc(golden.steps["final_tokens"], result.final_tokens)
    digest = sha256_of(result.final_tokens)
    t = result.timings
    gate = gates.get(golden.model) if gates is not None else None
    logger.info(
        f"RESULT {name}: PSNR vs golden.png {psnr_golden:.2f} dB (gate {gate}); final_tokens pcc "
        f"vs golden {p:.6f}; sha256 {digest[:16]}; image {result.image.size} finite "
        f"{bool(torch.isfinite(result.final_tokens).all())}"
    )
    logger.info(
        f"RECORD {name} walls: encode {t['encode_s']:.3f}s prepare {t['prepare_s']:.3f}s denoise {t['denoise_s']:.3f}s "
        f"({t['denoise_s'] / result.steps:.3f} s/step incl. capture {t['trace_captured']}) decode {t['decode_s']:.3f}s "
        f"(vae capture {t['vae_trace_captured']}) to_host {t['to_host_s']:.3f}s total {t['total_s']:.3f}s"
    )
    return {"psnr_golden": psnr_golden, "pcc": p, "digest": digest}


# --- P0: host ----------------------------------------------------------------------------------------------------------------------


def test_request_resolve(expect_error) -> None:
    base = Flux3KleinRequest(prompt="x").resolve()
    assert (base.model, base.steps, base.guidance, base.flow_map) == ("base", 50, 3.5, False)
    distilled = Flux3KleinRequest(prompt="x", model="distilled").resolve()
    assert (distilled.steps, distilled.guidance, distilled.flow_map) == (16, None, True)
    assert Flux3KleinRequest(prompt="x", model="distilled", guidance=1.0).resolve().guidance is None
    assert Flux3KleinRequest(prompt="x", steps=8, guidance=2.5).resolve().guidance == 2.5
    assert MODELS["base"]["steps"] == 50 and MODELS["distilled"]["guidance"] is None
    with expect_error(ValueError, "flow map"):
        Flux3KleinRequest(prompt="x", model="distilled", guidance=2.0).resolve()
    with expect_error(ValueError, "unknown model"):
        Flux3KleinRequest(prompt="x", model="pup").resolve()
    with expect_error(ValueError, "output_format"):
        Flux3KleinRequest(prompt="x", output_format="webp").resolve()
    with expect_error(ValueError, "text_override"):
        Flux3KleinRequest(prompt="x", text_override=((torch.zeros(80, 20480), packing.text_ids(80)), None)).resolve()
    # generate.py:72 ``width or default_width``: 0 means "not given" (M8 verifier finding; was a ValueError).
    zero = Flux3KleinRequest(prompt="x", width=0, height=0).resolve()
    assert (zero.width, zero.height, zero.crop) == (1024, 1024, None)
    with expect_error(ValueError, "negative"):
        Flux3KleinRequest(prompt="x", width=-16).resolve()

    side = admit(Flux3KleinRequest(prompt="x", width=1000).resolve(), (4, 8))
    assert not side.ok and side.budget == "side", side
    big = admit(Flux3KleinRequest(prompt="x", height=4112).resolve(), (4, 8))
    assert not big.ok and big.budget == "side", big
    from PIL import Image

    with expect_error(ValueError, REFERENCES_REASON):
        Flux3KleinRequest(prompt="x", images=(Image.new("RGB", (64, 64)),) * 11).resolve()
    ok = admit(Flux3KleinRequest(prompt="x", width=1344, height=768).resolve(), (4, 8))
    assert ok.ok and ok.measured["image_tokens"] == 84 * 48, ok
    default = Flux3KleinRequest(prompt="x").resolve()
    assert (default.width, default.height, default.crop) == (1024, 1024, None) and default.prepared.images == ()
    # a 1000x1000 reference with width=0 / height=0: the 1008 canvas AND the crop-back, as upstream_glue(images, 0, 0)
    zero_ref = Flux3KleinRequest(prompt="x", width=0, height=0, images=(Image.new("RGB", (1000, 1000)),)).resolve()
    assert (zero_ref.width, zero_ref.height, zero_ref.crop) == (1008, 1008, (4, 4, 1004, 1004)), zero_ref
    assert Flux3KleinRequest(prompt="x", width=1008, images=(Image.new("RGB", (1000, 1000)),)).resolve().crop is None
    # no token budget (2026-10-19): the structural worst case is admitted, a side over MAX_SIDE is refused
    worst = admit(
        replace(
            Flux3KleinRequest(
                prompt="x", width=4096, height=4096, images=(Image.new("RGB", (1024, 1024)),) * 10
            ).resolve(),
            text_length=80,
        ),
        (4, 8),
    )
    assert worst.ok and worst.measured["total_tokens"] == 65536 + 40960 + 80, worst
    over = admit(replace(Flux3KleinRequest(prompt="x", width=4112, height=4096).resolve(), text_length=80), (4, 8))
    assert not over.ok and over.budget == "side", over
    with expect_error(Flux3KleinAdmissionError, "outside"):
        raise Flux3KleinAdmissionError(over)

    noise = make_noise(7, 64, 64)
    assert noise.dtype == torch.float32 and torch.equal(noise, noise.to(torch.bfloat16).float())
    digest = noise_sha256(noise)
    logger.info(f"RESULT noise(seed 7, 64x64) sha256 {digest}")
    checked = 0
    for case_dir in golden_cases():
        golden = load_golden_case(case_dir)
        if int(golden.replay["seed"]) == 7 and list(golden.replay["latent_hw"]) == [64, 64]:
            assert golden.replay["noise_sha256"] == digest, golden.name
            assert torch.equal(golden.inputs["noise"], noise), golden.name
            checked += 1
    logger.info(f"RESULT noise matches {checked} golden record(s)")
    if not checked:
        pytest.skip(f"no seed-7 64x64 golden under {GOLDENS_ENV} to compare the noise hash against")


def test_pipeline_upsampler_hook(expect_error) -> None:
    """P7 (M9, host): ``resolve_caption`` = ``generate.py:52-63`` through the request fields, with a stub upsampler
    (no device): a finished caption passes through untouched (0 calls), the recorded knitted no-pup table is
    reproduced byte for byte, an upsampled request carries the stub's caption, a ``ValueError`` from the upsampler
    falls back to the prompt as typed, ``prompt_upsampling=False`` is the identity, a malformed box fails
    ``resolve()`` before anything else, and the result dataclass carries the record fields."""
    from ....reference.flux3_klein import bbox
    from .common import load_record

    calls: list[tuple] = []

    def stub(prompt, images, boxes):
        calls.append((prompt, len(images), boxes))
        return "CAPTION"

    # a finished upsampler answer (balloons record: boxes cited as <id>s) -> never upsampled, byte-equal
    balloons = load_record("distilled", "balloons")
    assert bbox.is_finished(balloons["prompt"]) and not bbox.is_finished(balloons["input_prompt"])
    resolved = Flux3KleinRequest(prompt=balloons["prompt"], model="distilled").resolve()
    assert resolve_caption(resolved, stub) == (balloons["prompt"], False) and calls == []
    # the no-pup box table (generate.py:63): the knitted record IS input_prompt + to_canvas(boxes)
    knitted = load_record("base_vs_distilled", "knitted_distilled")
    resolved = Flux3KleinRequest(
        prompt=knitted["input_prompt"], model="distilled", boxes=tuple(knitted["boxes"]), prompt_upsampling=False
    ).resolve()
    assert resolve_caption(resolved, stub) == (knitted["prompt"], False) and calls == []
    assert resolved.boxes == tuple(knitted["boxes"])
    # upsampled with the stub: the stub sees the raw prompt, the images and the 0-1 boxes
    resolved = Flux3KleinRequest(prompt="A man walking with his dog, afternoon sun", model="distilled").resolve()
    assert resolve_caption(resolved, stub) == ("CAPTION", True)
    assert calls == [("A man walking with his dog, afternoon sun", 0, None)]
    from PIL import Image

    resolved = Flux3KleinRequest(
        prompt=balloons["input_prompt"],
        model="distilled",
        boxes=tuple(balloons["boxes"]),
        images=(Image.new("RGB", (1024, 1024)),),
    ).resolve()
    assert resolve_caption(resolved, stub) == ("CAPTION", True)
    assert calls[-1] == (balloons["input_prompt"], 1, list(balloons["boxes"]))
    # no upsampler attached (every recorded-caption gate) and prompt_upsampling=False: the identity
    assert resolve_caption(resolved, None) == (
        f"{balloons['input_prompt']} {json.dumps(bbox.to_canvas(list(balloons['boxes'])), ensure_ascii=False)}",
        False,
    )
    plain = Flux3KleinRequest(prompt="x", prompt_upsampling=False).resolve()
    assert resolve_caption(plain, stub) == ("x", False) and len(calls) == 2

    # a ValueError from the upsampler (generate.py:59-61): the prompt as typed, upsampled False
    def refusing(prompt, images, boxes):
        raise ValueError("the upsampler takes at most 10 references")

    assert resolve_caption(Flux3KleinRequest(prompt="raw").resolve(), refusing) == ("raw", False)
    # a malformed box fails resolve() on the host
    with expect_error(ValueError, "neither bbox"):
        Flux3KleinRequest(prompt="x", boxes=({"id": "obj_1", "desc": "d"},)).resolve()
    with expect_error(ValueError, "a box is"):
        Flux3KleinRequest(prompt="x", boxes=({"id": "obj_1", "bbox": [0.1, 0.2, 0.3]},)).resolve()
    # the record fields on the result
    fields = Flux3KleinResult.__dataclass_fields__
    assert {"input_prompt", "caption", "upsampled", "boxes"} <= set(fields)
    assert fields["upsampled"].default is False and fields["boxes"].default is None
    # the seed of an unseeded request (generate.py:73): sha256 of the CAPTION the DiT sees, derived after the
    # upsampler by generate(); resolve() keeps None, an explicit seed is kept as it is (M9 fixer)
    import hashlib

    assert Flux3KleinRequest(prompt="x").resolve().seed is None
    assert Flux3KleinRequest(prompt="x", seed=7).resolve().seed == 7
    man_dog = load_record("distilled", "man_dog")
    expected = int(hashlib.sha256(man_dog["prompt"].encode()).hexdigest(), 16) % 2**32
    assert caption_seed(man_dog["prompt"]) == expected and 0 <= expected < 2**32
    assert caption_seed(man_dog["prompt"]) != caption_seed(man_dog["input_prompt"])
    logger.info(
        f"RESULT upsampler hook: {len(calls)} stub calls, pass-through / table / fallback rows exact; unseeded "
        f"man_dog caption seed {expected}"
    )


# --- P1 / P2 / P3 / P4: device ------------------------------------------------------------------------------------------------------


@mesh_rows
@pytest.mark.parametrize("variant", list(RESIDENT))
def test_t2i_golden_text(mesh_device, variant: str) -> None:
    golden = golden_for(variant)
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device)
    name = f"t2i_golden_text[{golden.name}] {mesh_tag(mesh_device)}"
    try:
        result = pipeline.generate(request_for(golden, text_override=True), traced=True)
        save_render(result, mesh_device, "golden_text")
        numbers = report(name, result, golden, png)
        width, height = result.image.size
        assert result.image.size == tuple(int(v) for v in golden.replay["size"])
        assert torch.isfinite(result.final_tokens).all()
        # Bit-identical to M4: the denoiser's own traced run of the golden request on the same transformer instance.
        denoiser = pipeline.denoisers[variant]
        prepared = denoiser.prepare(request_from_golden(golden))
        own = denoiser.denoise(prepared, traced=True)
        logger.info(
            f"RESULT {name}: pipeline final_tokens == denoiser.denoise(request_from_golden) {torch.equal(own, result.final_tokens)}"
        )
        assert torch.equal(own, result.final_tokens), "the pipeline's denoise differs from the M4 traced run"
        # The VAE in isolation: the device render vs the CPU reference decode of the SAME final_tokens.
        h, w = packing.latent_shape(height, width)
        reference = cached_reference_decode(
            pipeline._snapshots[variant], result.final_tokens, h, w, f"{golden.name} pipeline final_tokens"
        )
        psnr_dec = psnr_uint8(result.image, to_pil(reference))
        logger.info(
            f"RESULT {name}: PSNR device render vs CPU decode of its own final_tokens {psnr_dec:.2f} dB (gate {PSNR_DEC_FLOOR})"
        )
        assert psnr_dec >= PSNR_DEC_FLOOR
        assert numbers["psnr_golden"] >= PSNR_INTERIM[variant], f"PSNR vs golden.png {numbers['psnr_golden']:.2f} dB"
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


DEVICE_TEXT_ROWS = [
    "base",
    pytest.param(
        "distilled",
        marks=pytest.mark.xfail(
            strict=True,
            raises=ProperReferenceBandNotMet,
            reason="device distilled trajectory vs the CPU reference on the same device text: measured final pcc "
            "0.9855 / rel_rmse 0.170 / 29.99 dB (M4 band 0.99 / 0.15, PSNR 30.0); see the gate comment",
        ),
    ),
]


@mesh_rows
@pytest.mark.parametrize("variant", DEVICE_TEXT_ROWS)
def test_t2i_device_text(mesh_device, variant: str) -> None:
    golden = golden_for(variant)
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device)
    name = f"t2i_device_text[{golden.name}] {mesh_tag(mesh_device)}"
    try:
        request = request_for(golden, text_override=False)
        first = pipeline.generate(request, traced=True)
        save_render(first, mesh_device, "device_text")
        numbers = report(f"{name} first", first, golden, png, PSNR_INTERIM if variant == "base" else None)
        second = pipeline.generate(request, traced=True)
        report(f"{name} second", second, golden, png, PSNR_INTERIM if variant == "base" else None)
        assert first.text_length == int(golden.inputs["text_tokens"].shape[0]), first.text_length
        assert first.text is not None and tuple(first.text[0].shape) == tuple(golden.inputs["text_tokens"].shape)
        assert torch.equal(second.text[0], first.text[0]), "second encode differs from the first"
        assert torch.equal(second.final_tokens, first.final_tokens), "second generate differs from the first"
        assert not second.timings["trace_captured"] and not second.timings["vae_trace_captured"]
        assert torch.isfinite(first.final_tokens).all()
        record_digest(f"{variant}/{golden.name}", numbers["digest"])
        text_pcc = pcc(golden.inputs["text_tokens"].float(), first.text[0].float())
        logger.info(f"RESULT {name}: device text vs the recorded text pcc {text_pcc:.6f} (L {first.text_length})")
        if variant == "base":
            assert numbers["psnr_golden"] >= PSNR_INTERIM["base"], f"PSNR vs golden.png {numbers['psnr_golden']:.2f} dB"
            return
        # distilled: the proper reference = the CPU reference sampler on the SAME device text (gate comment).
        snapshot = pipeline._snapshots[variant]
        oracle_request = replace(request_from_golden(golden), text=first.text)
        reference = cached_reference_sampling(snapshot, variant, oracle_request, f"{golden.name} device text")
        width, height = first.image.size
        h, w = packing.latent_shape(height, width)
        ref_png = to_pil(cached_reference_decode(snapshot, reference, h, w, f"{golden.name} reference(device text)"))
        out_dir = renders_dir()
        if out_dir is not None:
            ref_png.save(out_dir / f"cpu_reference_{variant}_{width}x{height}_seed{first.seed}_device_text.png")
        golden_final = golden.steps["final_tokens"]
        ref_pcc, ref_rr = pcc(golden_final, reference), rel_rmse(golden_final, reference)
        dev_pcc, dev_rr = pcc(reference, first.final_tokens), rel_rmse(reference, first.final_tokens)
        psnr_ref = psnr_uint8(first.image, ref_png)
        gate_pcc, gate_rmse = final_gate(variant, first.steps, smoke=False)
        logger.info(
            f"RESULT {name}: CPU reference with the device text vs golden final_tokens pcc {ref_pcc:.6f} rel_rmse "
            f"{ref_rr:.4f}, its render vs golden.png {psnr_uint8(ref_png, png):.2f} dB; DEVICE vs that reference: "
            f"final pcc {dev_pcc:.6f} rel_rmse {dev_rr:.4f} (gate {gate_pcc} / {gate_rmse}), render "
            f"{psnr_ref:.2f} dB (gate {PSNR_PROPER_REFERENCE})"
        )
        assert ref_pcc >= REFERENCE_TEXT_SWAP_PCC, f"oracle took a different input: pcc vs golden {ref_pcc:.6f}"
        if not (dev_pcc >= gate_pcc and dev_rr <= gate_rmse and psnr_ref >= PSNR_PROPER_REFERENCE):
            raise ProperReferenceBandNotMet(
                f"device vs the CPU reference on the same device text: pcc {dev_pcc:.6f} rel_rmse {dev_rr:.4f} "
                f"PSNR {psnr_ref:.2f} dB (band {gate_pcc} / {gate_rmse} / {PSNR_PROPER_REFERENCE})"
            )
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


@mesh_rows
def test_both_models_resident(mesh_device) -> None:
    goldens = {variant: golden_for(variant) for variant in RESIDENT}
    pngs = {variant: golden_png(goldens[variant]) for variant in RESIDENT}
    pipeline = build_pipeline(mesh_device)
    name = f"both_models_resident {mesh_tag(mesh_device)}"
    results: dict[str, list[Flux3KleinResult]] = {v: [] for v in RESIDENT}
    edit_golden = next((load_golden_case(d) for d in REFERENCE_CASES if case_model(d) == "distilled"), None)
    calls = ["base", "distilled", "base"] if edit_golden is None else ["base", "distilled", "distilled-edit", "base"]
    if edit_golden is None:
        logger.warning(f"{name}: no distilled reference golden on disk; the 1-reference call is not made")
    try:
        for i, call in enumerate(calls):
            if call == "distilled-edit":
                edit_request = request_for_edit(edit_golden, text_override=False, cond_override=False)
                edit_resolved = replace(edit_request.resolve(), text_length=pipeline.text_length(edit_request.prompt))
                if not admit(edit_resolved, pipeline.mesh_shape).ok:
                    logger.warning(f"{name}: the 1-reference bucket is not measured yet; the edit call is not made")
                    continue
                result = pipeline.generate(edit_request, traced=True)
                report(
                    f"{name} call {i} distilled edit {edit_golden.name}",
                    result,
                    edit_golden,
                    golden_png(edit_golden),
                    None,
                )
                stored = stored_digest(f"distilled/{edit_golden.name}/edit")
                logger.info(
                    f"RESULT {name} call {i} distilled edit: final_tokens == P5b digest "
                    f"{stored == sha256_of(result.final_tokens) if stored else 'n/a (no digest on disk)'}; "
                    f"denoise trace captured {result.timings['trace_captured']}; captured keys "
                    f"{ {v: d.captured_keys for v, d in pipeline.denoisers.items()} }"
                )
                if stored is not None:
                    assert stored == sha256_of(result.final_tokens), "edit final_tokens differ from the P5b digest"
                continue
            variant = call
            golden = goldens[variant]
            result = pipeline.generate(request_for(golden, text_override=False), traced=True)
            results[variant].append(result)
            numbers = report(
                f"{name} call {i} {variant}", result, golden, pngs[variant], PSNR_INTERIM if variant == "base" else None
            )
            captured = {v: d.captured_keys for v, d in pipeline.denoisers.items()}
            logger.info(
                f"RECORD {name} call {i} {variant}: denoise trace captured {result.timings['trace_captured']}, vae "
                f"trace captured {result.timings['vae_trace_captured']}; captured keys {captured}"
            )
            stored = stored_digest(f"{variant}/{golden.name}")
            if stored is not None:
                assert stored == numbers["digest"], f"{variant} final_tokens differ from the P2 digest"
                logger.info(f"RESULT {name} call {i} {variant}: final_tokens == P2 digest")
            else:
                logger.warning(f"{name}: no P2 digest for {variant} on disk; equality against P2 not checked")
            if variant == "base":
                assert numbers["psnr_golden"] >= PSNR_INTERIM["base"], numbers
            # distilled: no band of its own here; its verdict is P2's (xfail vs the proper reference) through the digest.
        assert torch.equal(results["base"][0].final_tokens, results["base"][1].final_tokens), "base runs differ"
        logger.info(
            f"RESULT {name}: base call 2 == base call 0; base trace recaptured on call 2: "
            f"{results['base'][1].timings['trace_captured']} (expected True: the distilled prepare released every trace)"
        )
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


@mesh_rows
def test_trace_alloc_tracking(mesh_device) -> None:
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        pytest.skip("opt-in: TT_METAL_TRACE_ALLOC_TRACKING=1 on the pytest process")
    golden = golden_for("base")
    pipeline = build_pipeline(mesh_device)
    try:
        request = request_for(golden, text_override=False)
        first = pipeline.generate(request, traced=True)  # capture path (DiT trace, VAE trace)
        second = pipeline.generate(request, traced=True)  # replay path (VAE input copy + deallocate)
        assert torch.isfinite(first.final_tokens).all()
        assert torch.equal(second.final_tokens, first.final_tokens)
        assert not second.timings["trace_captured"] and not second.timings["vae_trace_captured"]
        logger.info(
            "RESULT trace_alloc_tracking: two generates completed (capture + replay); grep the log for tracker lines "
            "(expected 0)"
        )
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


# --- M8: reference rows ---------------------------------------------------------------------------------------------------------------


def request_for_edit(golden: GoldenCase, *, text_override: bool, cond_override: bool) -> Flux3KleinRequest:
    """The golden's edit request: its reference images, width / height None (the canvas is ``default_size`` of the
    prepared references, as the generator rendered it), the recorded text and / or cond rows as overrides."""
    replay = golden.replay
    inputs = golden.inputs
    text = None
    if text_override:
        null = (inputs["text_null_tokens"], inputs["text_null_ids"]) if golden.model == "base" else None
        text = ((inputs["text_tokens"], inputs["text_ids"]), null)
    cond = (inputs["cond_tokens"], inputs["cond_ids"]) if cond_override else None
    return Flux3KleinRequest(
        prompt=golden_prompt(golden),
        model=golden.model,
        width=None,
        height=None,
        seed=int(replay["seed"]),
        steps=golden.num_steps,
        guidance=golden.guidance,
        images=tuple(golden_reference_images(golden)),
        alpha=golden.alpha,
        text_override=text,
        image_cond_override=cond,
    )


def golden_case_or_skip(case_dir: Path | None) -> GoldenCase:
    if case_dir is None:
        pytest.skip(f"{GOLDENS_ENV} not set or no reference case under it")
    golden = load_golden_case(case_dir)
    assert golden.replay["references"], golden.name
    return golden


def decode_floor(width: int, height: int) -> float | None:
    """The PSNR floor of the device render vs the CPU decode of its own tokens for this canvas (1024: 50.0; 896x1120:
    its own frozen floor, ``None`` = log only until measured)."""
    if (width, height) == (896, 1120):
        return PSNR_DEC_FLOOR_896x1120
    return PSNR_DEC_FLOOR


def check_decode(name: str, pipeline: Flux3KleinPipeline, result: Flux3KleinResult, golden: GoldenCase) -> float:
    """PSNR of the (uncropped) device render vs the CPU reference decode of the device's OWN final tokens."""
    width, height = (int(v) for v in golden.replay["size"])
    h, w = packing.latent_shape(height, width)
    reference = cached_reference_decode(
        pipeline._snapshots[golden.model], result.final_tokens, h, w, f"{golden.name} device final_tokens {name}"
    )
    ref_png = to_pil(reference)
    image = result.image if result.crop is None else result.image  # the goldens of these cases are not cropped
    psnr_dec = psnr_uint8(image, ref_png)
    floor = decode_floor(width, height)
    logger.info(
        f"RESULT {name}: PSNR device render vs CPU decode of its own final_tokens {psnr_dec:.2f} dB (floor {floor})"
    )
    if floor is not None:
        assert psnr_dec >= floor, f"PSNR vs CPU decode {psnr_dec:.2f} dB < {floor}"
    return psnr_dec


def expected_total_tokens(golden: GoldenCase) -> int:
    tokens = golden.replay["tokens"]
    return int(tokens["image"]) + int(tokens["image_cond"]) + int(tokens["text"])


def skip_unless_bucket_measured(pipeline: Flux3KleinPipeline, request: Flux3KleinRequest, name: str) -> None:
    """Every structurally valid request is admitted (no token budget since 2026-10-19); the row asserts the admission
    and logs the token sum before the device run."""
    resolved = request.resolve()
    resolved = replace(resolved, text_length=pipeline.text_length(resolved.prompt))
    admission = admit(resolved, pipeline.mesh_shape)
    logger.info(f"RESULT {name}: admitted {admission.measured}")
    assert admission.ok, admission


@mesh_rows
@pytest.mark.parametrize("case_dir", REFERENCE_CASE_PARAMS or [None], ids=REFERENCE_CASE_IDS or ["no_goldens"])
def test_edit_golden_inputs(mesh_device, case_dir: Path | None) -> None:
    golden = golden_case_or_skip(case_dir)
    variant = golden.model
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device)
    name = f"edit_golden_inputs[{golden.name}] {mesh_tag(mesh_device)}"
    try:
        request = request_for_edit(golden, text_override=True, cond_override=True)
        skip_unless_bucket_measured(pipeline, request, name)
        result = pipeline.generate(request, traced=True)
        save_render(result, mesh_device, "golden_inputs")
        numbers = report(name, result, golden, png, gates=None)
        assert result.image.size == tuple(int(v) for v in golden.replay["size"]) == png.size
        assert result.crop == (tuple(golden.replay["crop"]) if golden.replay["crop"] else None)
        assert [list(s) for s in result.references] == golden.replay["prepared_reference_sizes"]
        assert result.admission.measured["total_tokens"] == expected_total_tokens(golden), result.admission
        assert result.admission.measured["cond_tokens"] == int(golden.replay["tokens"]["image_cond"])
        assert torch.isfinite(result.final_tokens).all()
        logger.info(
            f"RECORD {name}: admission {result.admission.measured}; layout key {result.layout_key}; "
            f"encode_refs {result.timings['encode_refs_s']:.3f}s (override)"
        )
        # Bit-identical to M4: the denoiser's own traced run of the golden request (with image_cond) on the same transformer.
        denoiser = pipeline.denoisers[variant]
        prepared = denoiser.prepare(request_from_golden(golden))
        own = denoiser.denoise(prepared, traced=True)
        equal = torch.equal(own, result.final_tokens)
        logger.info(f"RESULT {name}: pipeline final_tokens == denoiser.denoise(request_from_golden) {equal}")
        assert equal, "the pipeline's denoise differs from the M4 traced run of the golden request"
        # The per-step floor (teacher forced, no accumulation) and the final band of the M4 method.
        worst = teacher_forced_table(name, denoiser, prepared, golden)
        floor = teacher_forced_floor(variant, smoke=False)
        final_pcc, final_rmse = numbers["pcc"], rel_rmse(golden.steps["final_tokens"], result.final_tokens)
        gate_pcc, gate_rmse = final_gate(variant, golden.num_steps, smoke=False)
        logger.info(
            f"RESULT {name}: teacher-forced worst velocity pcc {worst:.6f} (floor {floor}); final pcc {final_pcc:.6f} "
            f"rel_rmse {final_rmse:.4f} (band {gate_pcc} / {gate_rmse}); PSNR vs golden.png {numbers['psnr_golden']:.2f} dB "
            f"(calibration only: CPU golden vs BFL PNG 39.2-39.5 dB on man_dog, recorded-text T2I rows 28.4 / 32.9 dB)"
        )
        psnr_dec = check_decode(name, pipeline, result, golden)
        logger.info(
            f"RESULT {name} summary: final {final_pcc:.6f} / {final_rmse:.4f}, teacher-forced {worst:.6f}, decode "
            f"{psnr_dec:.2f} dB, vs golden.png {numbers['psnr_golden']:.2f} dB; {pipeline.timings_init}"
        )
        assert worst >= floor, f"teacher-forced velocity floor {worst:.6f} < {floor}"
        if not (final_pcc >= gate_pcc and final_rmse <= gate_rmse):
            raise FinalBandNotMet(
                f"{name}: final pcc {final_pcc:.6f} rel_rmse {final_rmse:.4f} outside the band {gate_pcc} / {gate_rmse} "
                f"(teacher-forced floor {worst:.6f} >= {floor})"
            )
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


@mesh_rows
@pytest.mark.parametrize("case_dir", REFERENCE_CASES or [None], ids=REFERENCE_CASE_IDS or ["no_goldens"])
def test_edit_device_path(mesh_device, case_dir: Path | None) -> None:
    golden = golden_case_or_skip(case_dir)
    variant = golden.model
    png = golden_png(golden)
    pipeline = build_pipeline(mesh_device)
    name = f"edit_device_path[{golden.name}] {mesh_tag(mesh_device)}"
    try:
        request = request_for_edit(golden, text_override=False, cond_override=False)
        skip_unless_bucket_measured(pipeline, request, name)
        first = pipeline.generate(request, traced=True)
        save_render(first, mesh_device, "device_path")
        numbers = report(f"{name} first", first, golden, png, gates=None)
        second = pipeline.generate(request, traced=True)
        report(f"{name} second", second, golden, png, gates=None)
        assert first.image.size == tuple(int(v) for v in golden.replay["size"])
        assert first.text_length == int(golden.inputs["text_tokens"].shape[0])
        assert first.image_cond is not None and first.image_cond[0].shape == golden.inputs["cond_tokens"].shape
        assert torch.equal(first.image_cond[1], golden.inputs["cond_ids"]), "device cond ids differ from the golden"
        assert torch.equal(second.text[0], first.text[0]) and torch.equal(second.image_cond[0], first.image_cond[0])
        assert torch.equal(second.final_tokens, first.final_tokens), "second generate differs from the first"
        assert not second.timings["trace_captured"] and not second.timings["vae_trace_captured"]
        assert torch.isfinite(first.final_tokens).all()
        record_digest(f"{variant}/{golden.name}/edit", numbers["digest"])
        text_pcc = pcc(golden.inputs["text_tokens"].float(), first.text[0].float())
        cond_pcc = pcc(golden.inputs["cond_tokens"].float(), first.image_cond[0].float())
        cond_rr = rel_rmse(golden.inputs["cond_tokens"].float(), first.image_cond[0].float())
        logger.info(
            f"RESULT {name}: device text vs recorded pcc {text_pcc:.6f} (L {first.text_length}); device cond tokens vs "
            f"golden cond_tokens pcc {cond_pcc:.6f} rel_rmse {cond_rr:.5f} ({first.image_cond[0].shape[0]} rows, "
            f"encode_refs {first.timings['encode_refs_s']:.3f}s first / {second.timings['encode_refs_s']:.3f}s second)"
        )
        psnr_dec = check_decode(name, pipeline, first, golden)
        if variant == "base" and os.environ.get(BASE_PROPER_REFERENCE_ENV) != "1":
            logger.info(
                f"RESULT {name}: base proper reference not run ({BASE_PROPER_REFERENCE_ENV} != 1; 2.5-5.5 h of node CPU); "
                f"final pcc vs golden {numbers['pcc']:.6f}, PSNR vs golden.png {numbers['psnr_golden']:.2f} dB, decode {psnr_dec:.2f} dB"
            )
            pytest.skip(f"{BASE_PROPER_REFERENCE_ENV} != 1: the base proper-reference band is opt-in (numbers logged)")
        snapshot = pipeline._snapshots[variant]
        oracle_request = replace(request_from_golden(golden), text=first.text, image_cond=first.image_cond)
        reference = cached_reference_sampling(snapshot, variant, oracle_request, f"{golden.name} device text + cond")
        width, height = (int(v) for v in golden.replay["size"])
        h, w = packing.latent_shape(height, width)
        ref_png = to_pil(cached_reference_decode(snapshot, reference, h, w, f"{golden.name} reference(device inputs)"))
        out_dir = renders_dir()
        if out_dir is not None:
            ref_png.save(out_dir / f"cpu_reference_{variant}_{width}x{height}_seed{first.seed}_device_inputs.png")
        golden_final = golden.steps["final_tokens"]
        ref_pcc, ref_rr = pcc(golden_final, reference), rel_rmse(golden_final, reference)
        dev_pcc, dev_rr = pcc(reference, first.final_tokens), rel_rmse(reference, first.final_tokens)
        psnr_ref = psnr_uint8(first.image, ref_png)
        gate_pcc, gate_rmse = final_gate(variant, first.steps, smoke=False)
        logger.info(
            f"RESULT {name}: CPU reference with the device text + cond vs golden final_tokens pcc {ref_pcc:.6f} rel_rmse "
            f"{ref_rr:.4f}, its render vs golden.png {psnr_uint8(ref_png, png):.2f} dB; DEVICE vs that reference: final pcc "
            f"{dev_pcc:.6f} rel_rmse {dev_rr:.4f} (gate {gate_pcc} / {gate_rmse}), render {psnr_ref:.2f} dB (gate "
            f"{PSNR_PROPER_REFERENCE})"
        )
        assert ref_pcc >= REFERENCE_TEXT_SWAP_PCC, f"oracle took a different input: pcc vs golden {ref_pcc:.6f}"
        if not (dev_pcc >= gate_pcc and dev_rr <= gate_rmse and psnr_ref >= PSNR_PROPER_REFERENCE):
            raise ProperReferenceBandNotMet(
                f"device vs the CPU reference on the same device inputs: pcc {dev_pcc:.6f} rel_rmse {dev_rr:.4f} "
                f"PSNR {psnr_ref:.2f} dB (band {gate_pcc} / {gate_rmse} / {PSNR_PROPER_REFERENCE})"
            )
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


def synthetic_reference(width: int, height: int, seed: int):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), np.uint8))


@mesh_rows
def test_edit_crop_back(mesh_device, expect_error) -> None:
    pipeline = build_pipeline(mesh_device)
    name = f"edit_crop_back {mesh_tag(mesh_device)}"
    try:
        request = Flux3KleinRequest(
            prompt="a photo", model="distilled", seed=3, steps=1, images=(synthetic_reference(1016, 1016, 3),)
        )
        resolved = request.resolve()
        assert (resolved.width, resolved.height, resolved.crop) == (1024, 1024, (4, 4, 1020, 1020))
        skip_unless_bucket_measured(pipeline, request, name)
        result = pipeline.generate(request, traced=True)
        save_render(result, mesh_device, "crop_back")
        logger.info(
            f"RESULT {name}: canvas 1024x1024, image {result.image.size}, crop {result.crop}, references {result.references}, "
            f"admission {result.admission.measured}; encode_refs {result.timings['encode_refs_s']:.3f}s total "
            f"{result.timings['total_s']:.3f}s"
        )
        assert result.image.size == (1016, 1016)
        assert result.crop == (4, 4, 1020, 1020) and result.references == ((1024, 1024),)
        assert result.admission.measured["image_tokens"] == 4096 and result.admission.measured["cond_tokens"] == 4096
        assert torch.isfinite(result.final_tokens).all() and result.final_tokens.shape == (4096, 128)
        # P6 in the same process (realistic residency): an over-capacity request leaves the device untouched.
        check_admission_before_device(pipeline, name, expect_error)
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


class DeviceCallCounter:
    """Counts the calls into the device entry points a refused request must never reach (every upload / allocation
    path of the pipeline and the four stage entry points), by wrapping them for the duration of the ``with`` block.
    "Memory unchanged" alone would pass a refusal that uploaded a tensor and freed it (M8 verifier finding)."""

    def __init__(self, pipeline: Flux3KleinPipeline) -> None:
        from ....utils import tensor as tensor_utils

        self.counts: dict[str, int] = {}
        self._patches: list[tuple[object, str, object]] = []
        targets = [
            (ttnn, "from_torch"),
            (ttnn, "to_device"),
            (ttnn, "allocate_tensor_on_device"),
            (ttnn, "copy"),
            (tensor_utils, "from_torch"),
            (type(pipeline.vae_encode), "encode"),
            (type(pipeline.vae_encode), "upload"),
            (type(pipeline.prompt_encoder), "encode_host"),
            (type(pipeline.vae), "decode"),
        ]
        denoiser = next(iter(pipeline.denoisers.values()))
        targets += [(type(denoiser), "prepare"), (type(denoiser), "denoise_on_device")]
        for owner, attr in targets:
            key = f"{getattr(owner, '__name__', owner)}.{attr}"
            if key in self.counts:
                continue
            self.counts[key] = 0
            self._patches.append((owner, attr, getattr(owner, attr)))

    def __enter__(self) -> DeviceCallCounter:
        for owner, attr, original in self._patches:
            setattr(owner, attr, self._wrap(f"{getattr(owner, '__name__', owner)}.{attr}", original))
        return self

    def __exit__(self, *_: object) -> None:
        for owner, attr, original in self._patches:
            setattr(owner, attr, original)

    def _wrap(self, key: str, original):
        def wrapped(*args, **kwargs):
            self.counts[key] += 1
            return original(*args, **kwargs)

        return wrapped

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def check_admission_before_device(pipeline: Flux3KleinPipeline, name: str, expect_error) -> None:
    from ....pipelines.flux3_klein.pipeline_flux3_klein import dram_per_bank

    before = dram_per_bank(pipeline.device)
    keys_before = {v: d.captured_keys for v, d in pipeline.denoisers.items()}
    wall_before = pipeline.vae_encode.last_wall_s
    request = Flux3KleinRequest(
        prompt="a photo",
        model="distilled",
        width=4112,  # over MAX_SIDE: the structural refusal (no token budget since 2026-10-19)
        height=4096,
        images=(synthetic_reference(1024, 1024, 9),) * 10,
    )
    t0 = time.time()
    with DeviceCallCounter(pipeline) as counter, expect_error(Flux3KleinAdmissionError, "outside") as info:
        pipeline.generate(request, traced=True)
    wall = time.time() - t0
    admission = info.value.admission
    ttnn.synchronize_device(pipeline.device)
    after = dram_per_bank(pipeline.device)
    keys_after = {v: d.captured_keys for v, d in pipeline.denoisers.items()}
    logger.info(
        f"RESULT {name} admission_before_device: refused budget {admission.budget} measured {admission.measured} in "
        f"{wall:.3f}s host wall; DRAM/TRACE before == after {before == after} ({after}); captured keys unchanged "
        f"{keys_before == keys_after}; encoder untouched {pipeline.vae_encode.last_wall_s == wall_before}; device "
        f"entry-point calls during the refused generate {counter.total} {counter.counts}"
    )
    assert admission.budget == "side" and admission.measured["width"] == 4112, admission
    assert before == after, "the refused request changed the device memory"
    assert keys_before == keys_after and pipeline.vae_encode.last_wall_s == wall_before
    assert counter.total == 0, f"the refused request reached a device entry point: {counter.counts}"


@mesh_rows
def test_admission_before_device(mesh_device, expect_error) -> None:
    pipeline = build_pipeline(mesh_device)
    try:
        check_admission_before_device(pipeline, f"admission_before_device {mesh_tag(mesh_device)}", expect_error)
    finally:
        pipeline.close()
        ttnn.synchronize_device(mesh_device)


@mesh_rows
@pytest.mark.parametrize("variant", list(RESIDENT))
def test_t2i_b300(mesh_device, variant: str) -> None:
    golden = golden_for(variant)
    device = golden.record.get("device") or golden_index(golden.path).get("device")
    if device != "cuda":
        pytest.skip(f"B300 goldens pending (golden device {device!r}); the >= {PSNR_B300} dB gate waits for them")
    pytest.fail("B300 goldens present: implement the PSNR_B300 gate row (PLAN decisions 4 and 8)")


@mesh_rows
def test_readback_fast_equals_legacy(mesh_device) -> None:
    """M10 readback (2026-10-20): the device-side uint8 conversion + single-replica read ("fast", the default) gives
    the SAME bytes as ``to_pil`` of the bf16 host copy of every replica ("legacy"), on a traced replay of the same
    request (deterministic: same seed, same captured trace). Logs both ``to_host_s``."""
    import numpy as np

    pipeline = build_pipeline(mesh_device)
    name = f"readback_fast_equals_legacy {mesh_tag(mesh_device)}"
    try:
        request = Flux3KleinRequest(
            prompt="a lighthouse on a cliff at dusk", model="distilled", width=768, height=768, seed=7
        )
        pipeline.readback = "fast"
        first = pipeline.generate(request, traced=True)  # capture
        fast = pipeline.generate(request, traced=True)  # replay
        pipeline.readback = "legacy"
        legacy = pipeline.generate(request, traced=True)  # replay
        a, b = np.asarray(fast.image), np.asarray(legacy.image)
        assert a.shape == b.shape == (768, 768, 3) and a.dtype == b.dtype == np.uint8, (
            a.shape,
            a.dtype,
            b.shape,
            b.dtype,
        )
        differing = int((a != b).sum())
        max_abs = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) if differing else 0
        logger.info(
            f"RESULT {name}: fast == legacy {differing == 0} (differing bytes {differing}, max |d| {max_abs}); "
            f"to_host fast {fast.timings['to_host_s']:.3f}s legacy {legacy.timings['to_host_s']:.3f}s "
            f"(capture call {first.timings['to_host_s']:.3f}s); image std {a.std():.2f}"
        )
        assert differing == 0, f"fast readback differs from to_pil in {differing} bytes (max |d| {max_abs})"
        assert np.array_equal(np.asarray(first.image), a), "capture-call render differs from the replay render"
    finally:
        pipeline.close()
