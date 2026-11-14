# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Digest-bound LTX 2.5 Ingredients reference conditioning with the conv VAE.

Native imports are lazy. Validated provenance is a loading contract, not proof
of correct fusion, native trace safety, identity quality or streaming performance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

ASSET_KINDS = {
    "transformer": "fused_ingredients09",
    "text_encoder": "gemma4",
    "video_vae": "conv",
    "audio_vae": "audio",
    "upsampler": "spatial_x2",
}
ADAPTER = {
    "repo": "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
    "revision": "5f83f6f7e5e6cd6eadfade02a57f03b9212d6434",
    "sha256": "ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95",
    "size_bytes": 1308787472,
}
S1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
S2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
# Version 1 is the existing baseline recipe. Refuse silent process-wide overrides.
ENVIRONMENT = {
    "LTX_REF_IMG0": "s2",
    "LTX_REF_STILL_ENCODE": "1",
    "LTX_KF_APPEND_TOKEN": "0",
    "LTX_REF_FL_END_HOLD": "0",
    "LTX_REF_FL_END_GUIDE": "",
    "LTX_S1_SIGMAS": "",
    "LTX_S2_SIGMAS": "",
    "LTX_KEYFRAME_S1_SIGMAS": "",
    "LTX_KEYFRAME_S1_STRENGTH": "0.5",
    "LTX_KEYFRAME_S1_STRENGTH_COARSE": "0.25",
    "LTX_KF_TRACE_PAD": "0",
    "LTX_KF_TRACE_ANCHORS": "",
    "LTX_QUANT": "",
    "LTX_VIDEO_ONLY": "0",
    "LTX_PROFILE_DENOISE_ONLY": "0",
    "LTX_ITER_FAST": "0",
    "LTX_GEN_EAGER_STAGES": "",
    "LTX_WARMUP_T2V_ONLY": "0",
    "LTX_WARMUP_ENCODERS": "1",
    "LTX_WARMUP_CAPTURE": "1",
    "LTX_S2_PROMPT_REUSE": "1",
    "LTX_YUV_EXPORT": "1",
    "LTX_ASYNC_EXPORT": "1",
    "LTX_AUDIO_OVERLAP": "1",
    "LTX_UPSAMPLER_RESIDENT": "1",
    "LTX_DEVICE_PROMPT_HANDOFF": "0",
    "LTX_EULER_TAIL_TRACE": "0",
    "LTX_ITER_DIT_RESIDENT": "0",
    "LTX_VAE_ENCODER_HOST": "0",
    "LTX25_VAE_FALLBACK_23": "0",
    "LTX25_TEXT_STACK": "gemma4",
    "RUN_I2V": "1",
}
RUNTIME_OPTIONS = frozenset({"sp_axis", "tp_axis", "num_links", "dynamic_load", "topology", "is_fsdp", "traced"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be recorded")


def _hash(value, label, lengths=(64,)):
    _require(isinstance(value, str) and len(value) in lengths and re.fullmatch(r"[0-9a-f]+", value), f"invalid {label}")


def file_digest(path):
    """Streaming SHA256; does not load a model file into host RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(binding, root):
    _hash(binding["sha256"], "file SHA256")
    _text(binding["path"], "file path")
    path = (root / binding["path"]).resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    _require(file_digest(path) == binding["sha256"], f"digest mismatch: {path}")
    binding["path"] = str(path)


def _still_image(path, geometry=None):
    from PIL import Image

    _require(Path(path).is_file(), f"not a regular image file: {path}")
    with Image.open(path) as image:
        _require(getattr(image, "n_frames", 1) == 1, f"expected a still image: {path}")
        if geometry is not None:
            _require(image.size == geometry, f"endpoint geometry {image.size} differs from {geometry}")
        image.verify()


def _environment():
    for name, value in os.environ.items():
        if name.startswith(("LTX_", "LTX25_")) or name == "RUN_I2V":
            _require(name in ENVIRONMENT and value == ENVIRONMENT[name], f"incompatible environment: {name}={value!r}")


def _native_factory():
    from . import pipeline_ltx_distilled as distilled
    from .pipeline_ltx25_distilled import LTX25DistilledPipeline

    # These globals are bound at import time. A process can have imported them
    # under another recipe before restoring its environment: check both layers.
    _require(list(distilled.DISTILLED_SIGMA_VALUES) == S1_SIGMAS, "imported stage-1 schedule differs")
    _require(list(distilled.STAGE_2_DISTILLED_SIGMA_VALUES) == S2_SIGMAS, "imported stage-2 schedule differs")
    _require(
        distilled.KEYFRAME_S1_STRENGTH == 0.5 and distilled.KEYFRAME_S1_STRENGTH_COARSE == 0.25,
        "imported keyframe strengths differ",
    )

    class ReferencePipeline(LTX25DistilledPipeline):
        def __init__(self, *args, ref_num_frames, **kwargs):
            # The base constructor forces warmup when traced, including its cold
            # kernel-manifest pass. Bind the reference recipe before either pass
            # can allocate or capture the default base trace families.
            self._iclora_ref_num_frames = ref_num_frames
            super().__init__(*args, **kwargs)

        def warmup_buffers(self, *args, **kwargs):
            kwargs.update(ref_num_frames=self._iclora_ref_num_frames, stages=("s1_ref", "s2_ref"))
            return super().warmup_buffers(*args, **kwargs)

    return ReferencePipeline.create_pipeline


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate fixture key: {key}")
        result[key] = value
    return result


def _keys(mapping, expected, label):
    _require(isinstance(mapping, dict) and set(mapping) == set(expected), f"unexpected or missing {label} fields")


def load_fixture(fixture_path):
    """Validate local assets and return a resolved manifest using only CPU imports.

    No unresolved example or absent real weights counts as a successful fixture.
    Relative asset paths are resolved against the manifest, not the process cwd.
    """
    path = Path(fixture_path).resolve(strict=True)
    _require(path.is_file(), "fixture must be a regular file")
    with path.open() as stream:
        raw = stream.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, "fixture exceeds 1 MiB")
    fixture = json.loads(raw, object_pairs_hook=_unique_object)
    try:
        _keys(
            fixture,
            ("schema_version", "resolved", "assets", "fusion", "reference", "generation", "environment", "acceptance"),
            "fixture",
        )
        _require(
            type(fixture["schema_version"]) is int and fixture["schema_version"] == 1,
            "unsupported fixture schema_version",
        )
        _require(fixture["resolved"] is True, "unresolved fixture: real asset and quality bindings are required")
        assets = fixture["assets"]
        _require(set(assets) == set(ASSET_KINDS), "fixture must bind all five split assets")
        for name, kind in ASSET_KINDS.items():
            binding = assets[name]
            _keys(binding, ("path", "sha256", "revision", "model_version", "kind"), name)
            _require(binding["model_version"] == "2.5" and binding["kind"] == kind, f"incompatible {name} kind/version")
            _hash(binding["revision"], f"{name} revision", (40, 64))
            _verify_file(binding, path.parent)
        fusion = fixture["fusion"]
        _keys(fusion, ("fused_sha256", "base", "adapter", "recipe"), "fusion")
        _require(
            fusion["fused_sha256"] == assets["transformer"]["sha256"], "fusion digest does not bind primary transformer"
        )
        _hash(fusion["base"]["sha256"], "base SHA256")
        _hash(fusion["base"]["revision"], "base revision", (40, 64))
        _require(fusion["base"]["sha256"] != fusion["fused_sha256"], "bare base transformer is not fused Ingredients")
        adapter = fusion["adapter"]
        _require(all(adapter[k] == v for k, v in ADAPTER.items()), "unsupported Ingredients adapter identity")
        _require(
            type(adapter["scale"]) in (int, float) and math.isfinite(adapter["scale"]) and adapter["scale"] > 0,
            "invalid adapter scale",
        )
        _text(fusion["recipe"]["tool"], "fusion tool")
        _text(fusion["recipe"]["description"], "fusion recipe")
        _hash(fusion["recipe"]["revision"], "fusion tool revision", (40, 64))
        g = fixture["generation"]
        _keys(
            g,
            (
                "mode",
                "quality_variant",
                "height",
                "width",
                "num_frames",
                "ref_frames",
                "fps",
                "reference_strength",
                "pin_s1_strength",
                "pin_s2_strength",
                "sampler",
                "s1_sigmas",
                "s2_sigmas",
            ),
            "generation",
        )
        _require(g["mode"] in ("r2v", "fl2v"), "mode must be r2v or fl2v")
        _text(g["quality_variant"], "quality_variant")
        for key in ("height", "width"):
            _require(type(g[key]) is int and g[key] >= 64 and g[key] % 64 == 0, f"invalid {key}")
        for key in ("num_frames", "ref_frames"):
            _require(type(g[key]) is int and g[key] >= 9 and (g[key] - 1) % 8 == 0, f"invalid {key}")
        _require(g["ref_frames"] <= g["num_frames"], "reference length exceeds output length")
        _require(type(g["fps"]) in (int, float) and math.isfinite(g["fps"]) and g["fps"] > 0, "invalid fps")
        for key in ("reference_strength", "pin_s1_strength", "pin_s2_strength"):
            _require(type(g[key]) in (int, float) and math.isfinite(g[key]) and 0 < g[key] <= 1, f"invalid {key}")
        _require(
            g["sampler"] == "deterministic_euler_s1_ref_s2_ref"
            and g["s1_sigmas"] == S1_SIGMAS
            and g["s2_sigmas"] == S2_SIGMAS,
            "unsupported sampler/sigma recipe",
        )
        if g["quality_variant"] == "ingredients09-full-reference-conv-default":
            _require(
                (
                    g["height"],
                    g["width"],
                    g["num_frames"],
                    g["ref_frames"],
                    g["fps"],
                    g["reference_strength"],
                    g["pin_s1_strength"],
                    g["pin_s2_strength"],
                    adapter["scale"],
                )
                == (512, 768, 121, 121, 24, 1, 1, 1, 1),
                "changed recipe requires a separately named quality variant",
            )
        _require(fixture["environment"] == ENVIRONMENT, "unsupported fixture environment recipe")
        reference = fixture["reference"]
        _text(reference["permission"], "reference use permission")
        _verify_file(reference, path.parent)
        _still_image(reference["path"])
        acceptance = fixture["acceptance"]
        _text(acceptance["prompt"], "acceptance prompt")
        _text(acceptance["criteria"], "acceptance criteria")
        _require(
            len(acceptance["seeds"]) == 3
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in acceptance["seeds"]),
            "acceptance requires three uint32 seeds",
        )
        for key in ("first_frame", "last_frame"):
            binding = acceptance[key]
            _text(binding["permission"], f"{key} use permission")
            _verify_file(binding, path.parent)
            _still_image(binding["path"], (g["width"], g["height"]))
        quality = acceptance["quality_reference"]
        if quality["kind"] == "primary_recipe":
            _hash(quality["revision"], "primary quality recipe revision", (40, 64))
            _text(quality["uri"], "primary quality recipe URI")
        elif quality["kind"] == "baseline_media":
            _verify_file(quality, path.parent)
        else:
            raise ValueError("quality_reference must bind primary_recipe or baseline_media")
    except (KeyError, TypeError) as error:
        raise ValueError(f"incomplete or malformed fixture: {error}") from error
    _environment()
    return fixture


class LTX25ICLoRAPipeline:
    """One immutable reference session; calls must be serialized by its owner.

    Only continuation endpoint caches are managed here. The underlying
    digest-keyed reference cache and native trace lifecycle stay unchanged.
    """

    def __init__(self, pipeline, fixture):
        self._pipeline = pipeline
        self._fixture = copy.deepcopy(fixture)
        self._endpoint_digests = {}
        self._owned_endpoint_keys = set()
        self._call_lock = threading.Lock()

    @classmethod
    def create_pipeline(cls, mesh_device, *, fixture_path, **runtime_options):
        _require(not (set(runtime_options) - RUNTIME_OPTIONS), "unsupported runtime option overrides fixture bindings")
        fixture = load_fixture(fixture_path)
        g = fixture["generation"]
        pipeline = _native_factory()(
            mesh_device,
            **{name: asset["path"] for name, asset in fixture["assets"].items()},
            diffusion_decoder=False,
            use_ancestral_sampler=False,
            run_warmup=False,
            ref_num_frames=g["ref_frames"],
            mode="av",
            num_frames=g["num_frames"],
            height=g["height"],
            width=g["width"],
            fps=g["fps"],
            **runtime_options,
        )
        return cls(pipeline, fixture)

    @property
    def resolved_fixture(self):
        return copy.deepcopy(self._fixture)

    def _validate_session(self):
        _environment()
        _verify_file(self._fixture["reference"], Path("/"))

    def warmup_buffers(self):
        _require(self._call_lock.acquire(blocking=False), "concurrent session call")
        try:
            self._validate_session()
            g = self._fixture["generation"]
            return self._pipeline.warmup_buffers(
                num_frames=g["num_frames"],
                height=g["height"],
                width=g["width"],
                ref_num_frames=g["ref_frames"],
                stages=("s1_ref", "s2_ref"),
            )
        finally:
            self._call_lock.release()

    def generate_r2v(self, prompt, *, first_frame=None, last_frame=None, seed, output_path=None, output_type="rgb"):
        _require(self._call_lock.acquire(blocking=False), "concurrent session call")
        try:
            return self._generate(prompt, first_frame, last_frame, seed, output_path, output_type)
        finally:
            self._call_lock.release()

    def _generate(self, prompt, first_frame, last_frame, seed, output_path, output_type):
        self._validate_session()
        _text(prompt, "prompt")
        _require(type(seed) is int and 0 <= seed < 2**32, "seed must be uint32")
        _require(output_type in ("rgb", "float", "yuv"), "unsupported output_type")
        g = self._fixture["generation"]
        _require(
            g["mode"] != "fl2v" or (first_frame is not None and last_frame is not None), "FL2V requires both endpoints"
        )
        images, active_digests, active_keys = [], {}, set()
        cache = self._pipeline._i2v_cond_cache
        for endpoint, index in ((first_frame, 0), (last_frame, g["num_frames"] - 1)):
            if endpoint is None:
                continue
            path = str(Path(endpoint).resolve(strict=True))
            _still_image(path, (g["width"], g["height"]))
            digest = file_digest(path)
            keys = {(path, g["height"] // 2, g["width"] // 2), (path, g["height"], g["width"])}
            # Compare before the underlying path-keyed cache can consume stale pixels.
            if self._endpoint_digests.get(path) != digest:
                for key in keys:
                    cache.pop(key, None)
            active_digests[path] = digest
            active_keys.update(keys)
            images.append((path, index, g["pin_s1_strength"], g["pin_s2_strength"]))
        self._owned_endpoint_keys.update(active_keys)
        success = False
        try:
            result = self._pipeline.generate(
                prompt,
                output_path=str(output_path) if output_path is not None else None,
                output_type=output_type,
                images=images or None,
                reference_video=(self._fixture["reference"]["path"], g["reference_strength"]),
                ref_frames=g["ref_frames"],
                num_frames=g["num_frames"],
                height=g["height"],
                width=g["width"],
                fps=g["fps"],
                seed=seed,
            )
            success = True
            return result
        finally:
            keep = active_keys if success else set()
            for key in self._owned_endpoint_keys - keep:
                cache.pop(key, None)
            self._owned_endpoint_keys = keep
            self._endpoint_digests = active_digests if success else {}
