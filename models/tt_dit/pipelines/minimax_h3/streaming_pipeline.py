# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit float-output bridge to the H3 base/Turbo ref2va call contract.

The caller creates/owns the pipeline inside its broker job and closes the mesh.
YUV feedback conversion is intentionally unsupported until its quality and cost
are measured. This bridge does not claim equality with the served YUV path.
"""

import hashlib

from .streaming import Media


def tensor_media(video, audio):
    """Own CPU TCHW video and stereo CN audio; context slices own their storage."""
    if video.device.type != "cpu" or audio.device.type != "cpu":
        raise ValueError("expected decoded host buffers; no implicit device transfer")
    if video.ndim != 4 or video.shape[1] != 3 or audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("expected TCHW RGB and stereo CN audio")
    if not video.is_floating_point() or not audio.is_floating_point():
        raise ValueError("expected decoded floating-point RGB and audio")

    def sliced(start, stop, audio_start, audio_stop):
        if any(type(value) is not int for value in (start, stop, audio_start, audio_stop)):
            raise ValueError("media slice bounds must be integers")
        if not (0 <= start <= stop <= len(video) and 0 <= audio_start <= audio_stop <= audio.shape[-1]):
            raise ValueError("media slice bounds exceed the decoded buffers")
        # A clone alone retains the autograd graph and therefore earlier full
        # outputs. Streaming owns decoded samples, never training history.
        return tensor_media(video[start:stop].detach().clone(), audio[:, audio_start:audio_stop].detach().clone())

    def repeated(previous, frames, samples):
        return video[:frames].equal(previous.video[-frames:]) and audio[:, :samples].equal(previous.audio[:, -samples:])

    return Media(video, audio, len(video), audio.shape[-1], sliced, repeated)


class PipelineRenderer:
    """Call an already loaded ref2va pipeline without changing its selected model.

    Identity is supplied by the owner's source/checkpoint receipt, not inferred
    from a class name. This bridge is provisional until tested on that exact
    served Turbo implementation. `on_event` preserves existing stage profiling.
    """

    def __init__(self, pipeline, *, width, height, identity, on_event=None, reference_factory=None):
        if getattr(pipeline, "task", None) != "ref2va":
            raise ValueError("requires an explicit ref2va pipeline")
        if getattr(pipeline, "vae_output_type", None) != "float":
            raise ValueError("reference feedback currently requires explicit float output")
        adapter_path = getattr(pipeline, "lora_path", None)
        if identity.variant == "h3-r2v-high":
            if adapter_path is not None:
                raise ValueError("base route received an adapted pipeline")
        else:
            if adapter_path is None or getattr(pipeline, "lora_strength", None) != 1.0:
                raise ValueError("speed route requires its adapter at strength 1.0")
            # Once at construction, never on every chunk. Hash the actual local
            # adapter bytes; documentation digests are not loaded-weight proof.
            digest = hashlib.sha256()
            with open(adapter_path, "rb") as adapter_file:
                for block in iter(lambda: adapter_file.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != identity.adapter_sha256:
                raise ValueError("loaded adapter path differs from identity digest")
            if (getattr(pipeline, "hyperflow", None) is not None) != (identity.variant == "h3-r2v-hyperflow8"):
                raise ValueError("adapter schedule family differs from selected route")
        expected_shift = 6 if identity.variant == "h3-r2v-turbo4" else 12
        if getattr(pipeline, "video_shift", 12) != expected_shift or getattr(pipeline, "audio_shift", 3) != 3:
            raise ValueError("loaded schedule shifts differ from selected route")
        if reference_factory is None:
            from .packing_ref2va import MiniMaxH3Reference

            reference_factory = MiniMaxH3Reference
        self.pipeline, self.width, self.height = pipeline, width, height
        self.identity, self.on_event, self.reference_factory = identity, on_event, reference_factory

    def __call__(self, request):
        if request.config.identity != self.identity:
            raise ValueError("request identity differs from the loaded pipeline receipt")
        refs = list(request.fixed_references)
        # Reserve a video/soundtrack slot even on the first request, so later
        # chunks do not fail after successful output has already been delivered.
        counts = {"image": 0, "video": 1, "audio": 0}
        for ref in refs:
            kind = (
                "video"
                if getattr(ref, "video", None) is not None
                else ("image" if getattr(ref, "image", None) is not None else "audio")
            )
            counts[kind] += 1
        if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("fixed references leave no supported rolling context slot")
        if request.previous is not None:
            refs.append(
                self.reference_factory(
                    video=request.previous.video,
                    fps=request.config.fps,
                    audio=request.previous.audio,
                    sample_rate=request.config.sample_rate,
                )
            )
        steps = {"h3-r2v-high": 50, "h3-r2v-turbo4": 5, "h3-r2v-hyperflow8": 9}[self.identity.variant]
        out = self.pipeline(
            request.prompt,
            references=refs,
            num_frames=request.config.raw_frames,
            height=self.height,
            width=self.width,
            seed=request.seed,
            num_inference_steps=steps,
            reference_resize_mode="match",
            on_event=self.on_event,
        )
        if out.fps != request.config.fps or out.sampling_rate != request.config.sample_rate:
            raise ValueError("pipeline media rates differ from the stream")
        if out.video_format != "rgb_float" or out.video.shape[:2] != (1, 3) or out.audio.shape[:2] != (1, 2):
            raise ValueError("unsupported H3 output format")
        if out.num_frames != out.video.shape[2]:
            raise ValueError("pipeline output frame metadata disagrees with buffer")
        # The output is fresh generation, not a repeated prefix. Do not invent
        # overlap from the duration of the conditioning reference.
        return tensor_media(out.video[0].permute(1, 0, 2, 3), out.audio[0])
