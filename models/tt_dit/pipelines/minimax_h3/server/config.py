# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Everything the MiniMax-H3 server decides before it opens a device.

One profile per mesh shape. ``config_defaults.json`` next to this file is the measured record for
both of them -- the working point a sweep chose, the quantization policy, which side runs the
conditioner, the Turbo adapter and its shifts -- so the server reads its serving configuration from
the same file the bring-up wrote rather than carrying a second copy that can drift from it.

Env overrides exist for every field a deployment might legitimately want to move (canvas, frames,
steps, queue depth, cache and output directories); the parts that are a property of the *build*
(quantization profile, adaLN residency, conditioner placement) are not overridable here, because
changing one of those without re-measuring is how a profile stops meaning what its numbers say.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DEFAULTS = Path(__file__).with_name("config_defaults.json")

HF_MODEL_DEFAULT = "MiniMaxAI/MiniMax-H3"
TURBO_REPO_DEFAULT = "lightx2v/Minimax-h3-Turbo"

#: EXACTLY the diffusers-format partitions this server loads: ~144 GB of a ~498 GB repo. The repo's
#: top-level ``FL2VA/`` and ``Ref2VA/`` trees are self-contained original-format bundles that
#: duplicate the root partitions and are never read, and ``transformer_ref`` serves only ref2va,
#: which this server does not expose.
#:
#: This list is not an optimization, it is what makes a mounted weight cache work at all. Without
#: it ``snapshot_download(..., local_files_only=True)`` can never be satisfied by a cache that was
#: populated from the same list, so every boot falls through to the network and re-downloads the
#: ~354 GB the server will never open. A missing entry here is a first-boot crash; an extra one is
#: a wasted download.
WEIGHT_ALLOW_PATTERNS: tuple[str, ...] = (
    "model_index.json",
    "modular_model_index.json",
    "text_encoder/*",
    "transformer/*",
    "vae/*",
    "audio_vae/*",
    "tokenizer/*",
    "processor/*",
    "scheduler/*",
    "audio_scheduler/*",
)

#: Mesh shape -> profile name. These are the only two shapes this server is measured on; a shape
#: that is not here is a configuration error rather than something to guess a preset for.
PROFILE_BY_MESH: dict[str, str] = {"1x1": "p150", "1x4": "p300x2"}
#: `tt-model` sets MESH_DEVICE rather than a shape, so accept its spelling too.
MESH_BY_DEVICE: dict[str, str] = {"P150": "1x1", "P300": "1x1", "P300X2": "1x4", "P150X4": "1x4"}

#: Canvas and frame grid, from the model (`packing.py`). Repeated here as *limits* the API reports;
#: the authority for snapping is still `align_num_frames` / the pipeline's own validation.
CANVAS_MULTIPLE = 32
LONG_EDGE_MAX = 1344
SHORT_EDGE_MIN = 256
FRAMES_MIN = 22
FRAME_GRID = "17n+5"
FPS = 24
SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2

ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

#: ~100 tokens of ordinary request, so the warm-up compiles the same programs a real first request
#: would. A short prompt would leave the conditioner's longest bucket cold and move the first real
#: request's latency into the compiler.
DEFAULT_WARMUP_PROMPT = (
    "A slow cinematic tracking shot through a quiet morning market: wooden stalls stacked with fruit, "
    "steam rising from a cart of noodles, an old vendor arranging oranges while a tabby cat threads "
    "between crates, warm low sunlight cutting through canvas awnings and dust in the air, shallow "
    "depth of field, gentle handheld drift. Audio: distant chatter, a kettle whistling, footsteps on "
    "wet stone and a bicycle bell ringing twice."
)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def align_frames(num_frames: int) -> int:
    """Snap up to the next ``17n + 5``.

    A local copy of `packing.align_num_frames` on purpose: request validation has to run in the HTTP
    worker, where importing the pipeline (and through it torch and ttnn) would make a 422 cost a
    model import. The two are pinned together by a unit test.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % 17 != 5:
        num_frames += 1
    return num_frames


def canvas_for_aspect(aspect: str, short_edge: int) -> tuple[int, int]:
    """``(width, height)`` for an ``"W:H"`` display ratio at a given short edge.

    Both axes land on a multiple of 32 and the long edge is capped at 1344, which is what makes the
    default `16:9` resolve to exactly the profile's own canvas rather than to a near miss of it.
    """
    try:
        aw, ah = (float(v) for v in aspect.split(":"))
    except Exception as exc:  # noqa: BLE001 - re-raised as a request error by the caller
        raise ValueError(f"aspect_ratio must look like '16:9', got {aspect!r}") from exc
    if aw <= 0 or ah <= 0:
        raise ValueError(f"aspect_ratio must be positive, got {aspect!r}")

    def snap(value: float) -> int:
        return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)

    if aw >= ah:
        width, height = snap(short_edge * aw / ah), snap(short_edge)
        width = min(width, LONG_EDGE_MAX)
    else:
        width, height = snap(short_edge), snap(short_edge * ah / aw)
        height = min(height, LONG_EDGE_MAX)
    return width, height


@dataclass(frozen=True)
class ServerConfig:
    """The resolved serving configuration for one profile."""

    profile: str
    mesh_shape: str
    mesh_rows: int
    mesh_cols: int

    model_id: str
    weights_dir: str | None
    weights_revision: str | None

    turbo_repo: str
    turbo_file: str | None
    turbo_revision: str | None
    lora_strength: float

    video_shift: float
    audio_shift: float
    dit_quant_profile: str | None
    precomputed_adaln: bool | None
    text_encoder_device: str | None
    vae_output_type: str
    prompt_cache: bool

    width: int
    height: int
    num_frames: int
    steps: int

    short_edge_max: int
    frames_max: int
    steps_max: int

    queue_max: int
    out_dir: str
    out_keep: int
    cache_dir: str | None
    api_key: str | None
    warmup: bool
    warmup_prompt: str
    job_history: int
    l1_small_size: int

    extra: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- views
    @property
    def defaults(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "steps": self.steps,
            "fps": FPS,
            "seed": 0,
        }

    @property
    def limits(self) -> dict:
        return {
            "short_edge_min": SHORT_EDGE_MIN,
            "short_edge_max": self.short_edge_max,
            "long_edge_max": LONG_EDGE_MAX,
            "multiple_of": CANVAS_MULTIPLE,
            "frames_min": FRAMES_MIN,
            "frames_max": self.frames_max,
            "frame_grid": FRAME_GRID,
            "steps_min": 1,
            "steps_max": self.steps_max,
        }

    @property
    def capabilities(self) -> dict:
        return {
            "modes": ["t2va", "fl2va"],
            "defaults": self.defaults,
            "limits": self.limits,
            "aspect_ratios": list(ASPECT_RATIOS),
            "queue_max": self.queue_max,
            "sample_rate": SAMPLE_RATE,
            "audio_channels": AUDIO_CHANNELS,
            "profile": self.profile,
            "mesh": self.mesh_shape,
            "model": self.model_id,
        }

    @property
    def num_inference_steps(self) -> int:
        """Sigma grid points for the default step count. See `steps_to_grid_points`."""
        return steps_to_grid_points(self.steps)


def steps_to_grid_points(steps: int) -> int:
    """API ``steps`` (forwards, i.e. NFE) -> the pipeline's ``num_inference_steps``.

    The scheduler's ``num_inference_steps`` counts points on the sigma grid and runs one model
    evaluation per *interval*, so a 4-step Turbo adapter is ``num_inference_steps=5``. Getting this
    off by one is a silent 25 % quality/latency error at NFE 4, which is why it is one named
    function rather than a ``+ 1`` at each call site.
    """
    return int(steps) + 1


def resolve_mesh_shape() -> str:
    shape = (os.environ.get("H3_MESH_SHAPE") or "").strip().lower()
    if shape:
        return shape
    device = (os.environ.get("MESH_DEVICE") or "").strip().upper()
    if device in MESH_BY_DEVICE:
        return MESH_BY_DEVICE[device]
    if device:
        raise ValueError(f"MESH_DEVICE={device!r} is not one of {sorted(MESH_BY_DEVICE)}")
    # A lone TT_METAL_VISIBLE_DEVICES naming one chip is the p150 bring-up environment.
    visible = (os.environ.get("TT_METAL_VISIBLE_DEVICES") or "").strip()
    if visible and len(visible.split(",")) == 1:
        return "1x1"
    raise ValueError("set H3_MESH_SHAPE (1x1|1x4) or MESH_DEVICE (P150|P300x2)")


def load_profile_record(profile: str, path: Path = CONFIG_DEFAULTS) -> dict:
    record = json.loads(Path(path).read_text())
    if profile not in record:
        raise ValueError(f"{path} has no record for profile {profile!r} (has {sorted(record)})")
    return record[profile]


def _resolve_weights_dir(model_id: str, revision: str | None) -> str | None:
    """A local snapshot if there is one, else let the pipeline download.

    Order: an explicit directory, then the HF cache with ``local_files_only`` (so a container with a
    mounted cache never reaches the network on a warm boot), then the network. Returning ``None``
    hands the decision to `resolve_weights_dir` in the pipeline, which is the behaviour a fresh
    container wants.
    """
    explicit = os.environ.get("H3_WEIGHTS_DIR") or os.environ.get("MINIMAX_H3_MODEL_PATH")
    if explicit:
        return explicit
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    for local_only in (True, False):
        if not local_only and os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0", "false"):
            break
        try:
            return snapshot_download(
                model_id,
                revision=revision,
                local_files_only=local_only,
                allow_patterns=list(WEIGHT_ALLOW_PATTERNS),
            )
        except Exception:  # noqa: BLE001 - a miss here is not fatal; the pipeline retries its own way
            continue
    return None


def _resolve_turbo_file(repo: str, filename: str | None, revision: str | None) -> str | None:
    if not filename:
        return None
    explicit = os.environ.get("H3_TURBO_PATH")
    if explicit:
        return explicit
    local_dir = os.environ.get("H3_TURBO_DIR")
    if local_dir and (Path(local_dir) / filename).is_file():
        return str(Path(local_dir) / filename)
    from huggingface_hub import hf_hub_download

    for local_only in (True, False):
        if not local_only and os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0", "false"):
            break
        try:
            return hf_hub_download(repo, filename, revision=revision, local_files_only=local_only)
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        f"Turbo adapter {filename!r} is not in the HF cache and could not be downloaded from {repo}. "
        "Mount a warm HF cache, set H3_TURBO_DIR, or allow network access."
    )


def build_config(resolve_paths: bool = True) -> ServerConfig:
    """Resolve the profile, its record and every env override into one frozen config."""
    mesh_shape = resolve_mesh_shape()
    if mesh_shape not in PROFILE_BY_MESH:
        raise ValueError(f"mesh shape {mesh_shape!r} has no served profile (have {sorted(PROFILE_BY_MESH)})")
    profile = PROFILE_BY_MESH[mesh_shape]
    rows, cols = (int(v) for v in mesh_shape.split("x"))
    record = load_profile_record(profile)

    model_id = os.environ.get("HF_MODEL") or os.environ.get("HF_MODEL_ID") or HF_MODEL_DEFAULT
    revision = os.environ.get("H3_WEIGHTS_REVISION") or None
    turbo_repo = os.environ.get("H3_TURBO_REPO") or TURBO_REPO_DEFAULT
    turbo_file = os.environ.get("H3_TURBO_FILE") or record.get("turbo_file")
    turbo_revision = os.environ.get("H3_TURBO_REVISION") or None

    # `H3_DEFAULTS` is one JSON object so a launcher can move the served point atomically; the four
    # keys it may carry are the four the capabilities endpoint reports as defaults.
    overrides = json.loads(os.environ.get("H3_DEFAULTS") or "{}")
    unknown = set(overrides) - {"width", "height", "num_frames", "steps"}
    if unknown:
        raise ValueError(f"H3_DEFAULTS may only set width/height/num_frames/steps, got {sorted(unknown)}")

    width = int(overrides.get("width", record["width"]))
    height = int(overrides.get("height", record["height"]))
    num_frames = align_frames(int(overrides.get("num_frames", record["num_frames"])))
    steps = int(overrides.get("steps", record["nfe"]))

    frames_max_record = record.get("frames_max") or {}
    frames_max = _env_int("H3_FRAMES_MAX", max([num_frames, *map(int, frames_max_record.values())]))

    cache_dir = os.environ.get("TT_DIT_CACHE_DIR")
    return ServerConfig(
        profile=profile,
        mesh_shape=mesh_shape,
        mesh_rows=rows,
        mesh_cols=cols,
        model_id=model_id,
        weights_dir=_resolve_weights_dir(model_id, revision) if resolve_paths else None,
        weights_revision=revision,
        turbo_repo=turbo_repo,
        turbo_file=_resolve_turbo_file(turbo_repo, turbo_file, turbo_revision) if resolve_paths else turbo_file,
        turbo_revision=turbo_revision,
        lora_strength=float(record.get("lora_strength", 1.0)),
        video_shift=float(record["video_shift"]),
        audio_shift=float(record["audio_shift"]),
        dit_quant_profile=record.get("dit_quant_profile"),
        precomputed_adaln=record.get("dit_dtype_policy", {}).get("precomputed_adaln"),
        text_encoder_device=record.get("text_encoder"),
        vae_output_type="uint8",
        prompt_cache=_env_flag("H3_PROMPT_CACHE", True),
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        short_edge_max=int(record.get("short_edge_max", 768)),
        frames_max=frames_max,
        steps_max=_env_int("H3_STEPS_MAX", 50),
        queue_max=_env_int("H3_MAX_QUEUE", 4),
        out_dir=os.environ.get("H3_OUT_DIR") or "/tmp/h3-out",
        out_keep=_env_int("H3_OUT_KEEP", 32),
        cache_dir=cache_dir,
        api_key=os.environ.get("H3_API_KEY") or None,
        warmup=_env_flag("H3_WARMUP", True),
        warmup_prompt=os.environ.get("H3_WARMUP_PROMPT") or DEFAULT_WARMUP_PROMPT,
        job_history=_env_int("H3_JOB_HISTORY", 64),
        l1_small_size=_env_int("H3_L1_SMALL_SIZE", 32768),
        extra={"choice_reason": record.get("choice_reason"), "e2e_s": record.get("e2e_s")},
    )
