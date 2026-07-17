# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt_dit pipeline serving runner (SHM peer for SPRunner).

Replaces media-server ``video_runner`` orchestration inside tt_dit: opens the
mesh, creates a pipeline via ``PipelineFactory``, always runs a traced warmup,
then serves ``VideoRequest`` / ``VideoResponse`` over POSIX SHM.

Launch examples
---------------
Single-host (local mesh)::

    python -m models.tt_dit.pipelines.runner \\
      --model-id wan2.2 --mesh-shape 2x4 --mesh-topology linear

Multihost (tt-run, same binary on every rank)::

    tt-run ... bash -c "... python -m models.tt_dit.pipelines.runner \\
      --model-id wan2.2 --mesh-shape 4x32 --mesh-topology ring"

External model with device / pipeline param overrides::

    python -m models.tt_dit.pipelines.runner \\
      --model-id external:my_pkg.pipelines.my_pipe \\
      --mesh-shape 2x4 --mesh-topology ring \\
      --device_params trace_region_size:150000000 \\
      --pipeline_params height:720 width:1280

``INFERENCE_SERVER`` must point at the ``tt-inference-server`` repo root so the
living SHM module can be loaded as ``tt-media-server.ipc.video_shm`` (default:
``$CWD/tt-inference-server``). Fail hard if missing.

Smoke vs SPRunner: start this runner first (or in parallel under tt-run), then
run the media-server with ``SPRunner`` using the same ``TT_VIDEO_SHM_*`` names.
MP4 outputs land in ``/tmp/videos`` for co-location with SPRunner.

FFmpeg gap vs media-server ``VideoManager.export_to_mp4``
--------------------------------------------------------
This runner uses ``models.tt_dit.utils.video.export_to_video``. Relative to
VideoManager it lacks: ``-movflags +faststart``, ``-tune film``, ``-profile:v
high``, ``-level 4.2``, encode timeout; default CRF is 25 (vs 23); uses imageio
ffmpeg + per-frame stdin writes instead of system ``ffmpeg`` + bulk write.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import queue
import signal
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from models.tt_dit.pipelines.pipeline_device import close_device, get_device
from models.tt_dit.pipelines.pipeline_factory import PipelineFactory
from models.tt_dit.utils.video import export_to_video

LOG_PROMPT_PREVIEW_CHARS = 50
ENCODER_QUEUE_MAXSIZE = 2
_PER_ENCODE_BOUND_S = 10.0
ENCODER_JOIN_TIMEOUT_S = (ENCODER_QUEUE_MAXSIZE + 1) * _PER_ENCODE_BOUND_S
_VIDEO_OUTPUT_DIR = Path("/tmp/videos")

_shutdown = False


@dataclass
class _EncodeJob:
    task_id: str
    frames: Optional[Any] = None
    error: Optional[str] = None


class _SingleRankComm:
    """No-op collective bus for single-host (world size 1)."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, obj, root: int = 0):
        return obj


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _env_rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _env_world_size() -> int:
    for key in ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE", "PMI_SIZE"):
        v = os.environ.get(key)
        if v is not None:
            return int(v)
    return 1


def _attach_mpi_comm():
    """Attach mpi4py to the MPI context already initialized by tt-metal.

    Must be called AFTER ``get_device`` — tt-metal calls MPI_Init_thread during
    device init. ``rc.initialize=False`` prevents a second MPI_Init.
    """
    try:
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError as e:
        raise RuntimeError("mpi4py is required for multi-rank operation. Install with: pip install mpi4py") from e


def _import_video_shm():
    """Load living SHM module from inference-server (no copy into tt_dit)."""
    root = os.environ.get("INFERENCE_SERVER", os.path.join(os.getcwd(), "tt-inference-server"))
    video_shm_path = os.path.join(root, "tt-media-server", "ipc", "video_shm.py")
    if not os.path.isfile(video_shm_path):
        raise FileNotFoundError(
            f"INFERENCE_SERVER living reference missing: expected {video_shm_path}. "
            f"Set INFERENCE_SERVER to the tt-inference-server repo root "
            f"(default: $CWD/tt-inference-server)."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("tt-media-server.ipc.video_shm")


def _parse_mesh_shape(value: str) -> tuple[int, int]:
    text = value.strip().lower().replace(",", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"mesh-shape must look like 4x32 or 4,32; got {value!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid mesh-shape {value!r}") from e


def _parse_kv_pairs(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``key:val`` CLI pairs. Values via ``ast.literal_eval`` (int/bool/…)."""
    out: dict[str, Any] = {}
    for item in pairs or ():
        key, raw = item.split(":", 1)
        out[key.strip()] = ast.literal_eval(raw.strip())
    return out


def _default_wan_resolution(mesh_shape: tuple[int, int]) -> tuple[int, int]:
    """Match dit_runners: mesh_size >= 32 → 720p, else 480p."""
    if mesh_shape[0] * mesh_shape[1] >= 32:
        return 720, 1280
    return 480, 832


def _pipeline_create_params(model_id: str, mesh_shape: tuple[int, int]) -> dict[str, Any]:
    """Serving-oriented create kwargs (performance defaults via create_pipeline)."""
    if model_id in ("wan2.2", "wan2.2-i2v"):
        height, width = _default_wan_resolution(mesh_shape)
        return {"height": height, "width": width, "num_frames": 81}
    return {}


def _decode_image_data(image_data: str, width: int, height: int):
    """Decode SHM base64 image payload to a PIL RGB image sized to (width, height)."""
    import base64
    import io

    from PIL import Image

    raw = image_data
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height))
    return img


def _pipeline_image_size(pipeline) -> tuple[int, int]:
    width = int(getattr(pipeline, "_width", 832))
    height = int(getattr(pipeline, "_height", 480))
    return width, height


def _export_frames_to_mp4(frames: Any, fps: int = 16) -> str:
    """Encode frames to ``/tmp/videos/{uuid}.mp4`` via ``utils.video.export_to_video``."""
    if hasattr(frames, "frames"):
        frames = frames.frames
    elif isinstance(frames, (tuple, list)):
        frames = frames[0]
    # Wan uint8 output is (B, T, H, W, 3); export_to_video expects (T, H, W, 3).
    frames = frames[0]

    _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
    return export_to_video(frames, output_path, fps=fps)


def _write_response_to_shm(video_shm_mod, output_shm, task_id: str, mp4_path: str) -> None:
    output_shm.write_response(
        video_shm_mod.VideoResponse(
            task_id=task_id,
            status=video_shm_mod.VideoStatus.SUCCESS,
            file_path=mp4_path,
            error_message="",
        )
    )


def _write_error_to_shm(video_shm_mod, output_shm, task_id: str, error: str = "") -> None:
    output_shm.write_response(
        video_shm_mod.VideoResponse(
            task_id=task_id,
            status=video_shm_mod.VideoStatus.ERROR,
            file_path="",
            error_message=error[: video_shm_mod.VideoShm.MAX_ERROR_LEN],
        )
    )


def _encoder_loop(video_shm_mod, output_shm, encode_queue: queue.Queue) -> None:
    logger.info("Encoder thread: started")
    while True:
        job = encode_queue.get()
        if job is None:
            logger.info("Encoder thread: shutdown sentinel received, exiting")
            return

        if job.error is not None:
            try:
                _write_error_to_shm(video_shm_mod, output_shm, job.task_id, job.error)
            except Exception as write_err:
                logger.error(
                    f"Encoder thread: failed to write upstream-error response " f"for task {job.task_id}: {write_err}"
                )
            continue

        try:
            mp4_path = _export_frames_to_mp4(job.frames)
            logger.info(f"Encoder thread: encoded mp4 for task {job.task_id} at {mp4_path}")
            _write_response_to_shm(video_shm_mod, output_shm, job.task_id, mp4_path)
        except Exception as encode_err:
            logger.error(
                f"Encoder thread: encode failure for task {job.task_id}: " f"{encode_err}\n{traceback.format_exc()}"
            )
            try:
                _write_error_to_shm(video_shm_mod, output_shm, job.task_id, str(encode_err))
            except Exception as write_err:
                logger.error(f"Encoder thread: failed to write error response for task " f"{job.task_id}: {write_err}")


def _run_pipeline_from_request(
    pipeline,
    req,
    model_id: str,
    image_data: Optional[str] = None,
):
    """Map SHM VideoRequest → pipeline ``__call__`` (Wan T2V / I2V serving path)."""
    kwargs: dict[str, Any] = {
        "prompts": [req.prompt],
        "num_inference_steps": int(req.num_inference_steps),
        "seed": int(req.seed or 0),
        "traced": True,
    }
    if req.negative_prompt:
        kwargs["negative_prompts"] = [req.negative_prompt]

    if model_id in ("wan2.2", "wan2.2-i2v"):
        kwargs["guidance_scale"] = float(req.guidance_scale) if req.guidance_scale else 4.0
        kwargs["guidance_scale_2"] = float(req.guidance_scale_2) if req.guidance_scale_2 else 3.0

    if model_id == "wan2.2-i2v":
        if not image_data:
            raise ValueError("wan2.2-i2v requires an image (SHM image_path / image_data)")
        width, height = _pipeline_image_size(pipeline)
        kwargs["image_prompt"] = _decode_image_data(image_data, width, height)

    return pipeline(**kwargs)


def _run_inference_loop(
    comm,
    pipeline,
    model_id: str,
    video_shm_mod,
    input_shm,
    encode_queue: Optional[queue.Queue],
) -> None:
    rank = comm.Get_rank()

    while not _shutdown:
        raw_req = None
        image_data = None
        if rank == 0:
            raw_req = input_shm.read_request()
            if raw_req and getattr(raw_req, "image_path", None):
                try:
                    with open(raw_req.image_path, "r") as f:
                        image_data = f.read()
                except OSError as e:
                    logger.warning(f"Could not read image from {raw_req.image_path}: {e}")

        req, image_data = comm.bcast((raw_req, image_data), root=0)

        if req is None:
            logger.info(f"Rank {rank}: Shutdown signal received, exiting loop")
            break

        if rank == 0:
            logger.info(
                f"Rank 0: task_id={req.task_id}, "
                f"prompt='{req.prompt[:LOG_PROMPT_PREVIEW_CHARS]}...', "
                f"steps={req.num_inference_steps}, seed={req.seed}"
            )

        try:
            logger.info(f"Rank {rank}: Starting inference for task {req.task_id}")
            frames = _run_pipeline_from_request(pipeline, req, model_id, image_data=image_data)
            logger.info(f"Rank {rank}: Inference done for task {req.task_id}")
            if rank == 0 and encode_queue is not None:
                encode_queue.put(_EncodeJob(task_id=req.task_id, frames=frames))
        except Exception as e:
            logger.error(f"Rank {rank}: ERROR for task {req.task_id}: {e}\n{traceback.format_exc()}")
            if rank == 0 and encode_queue is not None:
                encode_queue.put(_EncodeJob(task_id=req.task_id, error=str(e)))


def run(
    *,
    model_id: str,
    mesh_shape: tuple[int, int],
    mesh_topology: str = "ring",
    device_params: dict | None = None,
    pipeline_params: dict | None = None,
) -> None:
    rank = _env_rank()
    logger.info(
        f"Rank {rank}: starting runner model_id={model_id} " f"mesh_shape={mesh_shape} mesh_topology={mesh_topology}"
    )

    video_shm_mod = _import_video_shm()

    mesh_device = get_device(mesh_shape, model_id, mesh_topology, **(device_params or {}))

    if _env_world_size() > 1:
        comm = _attach_mpi_comm()
        logger.info(f"Rank {comm.Get_rank()}/{comm.Get_size()}: MPI attached")
    else:
        comm = _SingleRankComm()
        logger.info("Single-host: using local request bus (no MPI)")

    create_params = _pipeline_create_params(model_id, mesh_shape)
    create_params.update(pipeline_params or {})
    pipeline, warmup_args = PipelineFactory.create(model_id, mesh_device, **create_params)

    # Serving contract: always traced warmup via shared pipeline __call__ API.
    # warmup_args carry model-specific extras (e.g. blank image_prompt for I2V).
    logger.info(f"Rank {rank}: traced warmup...")
    pipeline(
        prompts=["warmup"],
        num_inference_steps=2,
        traced=True,
        seed=0,
        **warmup_args,
    )
    logger.info(f"Rank {rank}: Model ready for inference")

    input_shm = None
    output_shm = None
    encode_queue: Optional[queue.Queue] = None
    encoder_thread: Optional[threading.Thread] = None

    if comm.Get_rank() == 0:
        input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
        output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")
        input_shm = video_shm_mod.VideoShm(input_name, mode="input", is_shutdown=_is_shutdown)
        output_shm = video_shm_mod.VideoShm(output_name, mode="output", is_shutdown=_is_shutdown)
        input_shm.open()
        output_shm.open()
        in_repair = input_shm.recover(side="reader")
        out_repair = output_shm.recover(side="writer")
        if any(in_repair.values()) or any(out_repair.values()):
            logger.warning(
                f"Rank 0: crash-recovery repaired prior inconsistency: " f"input={in_repair} output={out_repair}"
            )

        encode_queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        encoder_thread = threading.Thread(
            target=_encoder_loop,
            args=(video_shm_mod, output_shm, encode_queue),
            name="video-encoder",
            daemon=False,
        )
        encoder_thread.start()
        logger.info("Rank 0: SHM bridge ready, waiting for requests...")

    try:
        _run_inference_loop(comm, pipeline, model_id, video_shm_mod, input_shm, encode_queue)
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Interrupted by user")
    finally:
        if comm.Get_rank() == 0:
            if encode_queue is not None and encoder_thread is not None:
                encode_queue.put(None)
                encoder_thread.join(timeout=ENCODER_JOIN_TIMEOUT_S)
                if encoder_thread.is_alive():
                    logger.warning(
                        f"Rank 0: encoder thread did not drain within "
                        f"{ENCODER_JOIN_TIMEOUT_S}s "
                        f"(remaining queued={encode_queue.qsize()}); "
                        f"continuing shutdown"
                    )
            if input_shm is not None:
                input_shm.close()
            if output_shm is not None:
                output_shm.close()
        close_device(mesh_device)

    logger.info(f"Rank {rank}: Shutdown complete")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tt_dit pipeline runner (SHM serving path for SPRunner)")
    parser.add_argument(
        "--model-id",
        required=True,
        help=("Registered id (e.g. wan2.2), or " "external:<module.name> for an external pipeline module"),
    )
    parser.add_argument(
        "--mesh-shape",
        required=True,
        type=_parse_mesh_shape,
        help="Mesh shape as RxC (e.g. 4x32 or 4,32)",
    )
    parser.add_argument(
        "--mesh-topology",
        default="ring",
        choices=("linear", "ring"),
        help="Fabric topology (default: ring)",
    )
    parser.add_argument(
        "--device_params",
        nargs="*",
        default=[],
        metavar="KEY:VAL",
        help=(
            "Optional device param overrides as key:val pairs (default: none). "
            "Useful for external models, e.g. "
            "--device_params trace_region_size:150000000"
        ),
    )
    parser.add_argument(
        "--pipeline_params",
        nargs="*",
        default=[],
        metavar="KEY:VAL",
        help=(
            "Optional create_pipeline kwargs as key:val pairs (default: none). "
            "Useful for external models, e.g. "
            "--pipeline_params height:720 width:1280"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    args = parse_args(argv)
    run(
        model_id=args.model_id,
        mesh_shape=args.mesh_shape,
        mesh_topology=args.mesh_topology,
        device_params=_parse_kv_pairs(args.device_params),
        pipeline_params=_parse_kv_pairs(args.pipeline_params),
    )


if __name__ == "__main__":
    main()
