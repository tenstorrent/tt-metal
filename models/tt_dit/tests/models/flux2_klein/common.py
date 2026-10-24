# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the FLUX.2 Klein correctness tests (the ``tests/models/flux3_klein/common.py`` pattern).

Imports no ``ttnn`` itself (the ``pipelines`` package ``__init__`` it goes through does; ``ttnn`` imports without a
device on a login box, where the host rows run); the mesh / device dicts live in ``device_params.py``. Resolves the
checkpoints and the goldens from the environment, owns the oracle cache and the comparison helpers, and builds the CPU
references: the diffusers ``Flux2KleinPipeline`` (``pipeline_flux2_klein.py``, cited as ``P``),
``Flux2Transformer2DModel`` (``transformer_flux2.py``, ``T``) and transformers' ``Qwen3ForCausalLM``, all in bf16 on
CPU, loaded from the same snapshot the device modules read. :func:`reference_prompt_embeds` is ``P:208-263``
(``_get_qwen3_prompt_embeds``) re-executed on the snapshot's own ``text_encoder`` / ``tokenizer``.

Environment:
  ``FLUX2_KLEIN_MODEL_PATH``    an explicit snapshot (``pipelines/flux2_klein/weights_flux2_klein.py``); else the
                                HuggingFace cache (``HF_HUB_CACHE`` / ``HF_HOME``). The four checkpoints are keyed by
                                ``(variant, size)``: ``distilled`` / ``base`` x ``4b`` / ``9b``.
  ``FLUX2_KLEIN_GOLDENS``       a directory holding golden cases (any ancestor of the case directories).
  ``FLUX2_KLEIN_ORACLE_CACHE``  where the CPU oracles (text taps, DiT velocities) are kept between runs: a directory
                                under ``/data`` (never ``/tmp`` or ``$HOME``); unset = recompute every time.
  ``FLUX2_KLEIN_OUT``           where the pipeline tests save their PNGs (never inside the tree); unset = nothing saved.
  ``TT_DIT_CACHE_DIR``          the converted-weight cache of the device modules.

Golden layout, the format ``generate_goldens_flux2_klein.py`` (this directory) writes and
``test_pipeline_flux2_klein.py`` reads (the FLUX 3 Klein ``generate_goldens.py`` format reduced to what this port
consumes), one directory per case under ``<root>/<model>/<case>/`` plus a root ``index.json``:
  ``record.json``         ``replay``: ``model`` (``distilled`` | ``base``), ``size`` (``4b`` | ``9b``), ``repo``,
                          ``snapshot``, ``prompt``, ``seed``, ``canvas`` [w, h], ``latent_hw`` [h, w], ``steps``,
                          ``guidance`` (``None`` for distilled), ``timesteps``, ``sigmas``, ``noise_sha256``,
                          ``noise_matches_port`` / ``schedule_matches_port`` / ``euler_matches_port`` (the
                          cross-checks against ``packing`` / ``scheduler``), library versions, the render wall and the
                          per-file sha256s.
  ``inputs.safetensors``  ``noise`` (h*w, 128) fp32 (bf16-valued: the packed state ENTERING step 0 exactly as
                          ``prepare_latents`` drew it, ``P:478-510``), ``image_ids`` (h*w, 4) int64, ``text_tokens``
                          (512, joint) bf16, ``text_ids`` (512, 4) int64, ``schedule`` (steps + 1,) fp32 (the sigmas
                          with the terminal 0); for the base model ``text_null_tokens`` / ``text_null_ids`` (the empty
                          prompt, ``P:749-760``).
  ``steps.safetensors``   per step k: ``step{k:02d}.x_t`` (h*w, 128) fp32, the state ENTERING step k
                          (``step00.x_t == noise``); ``step{k:02d}.velocity`` (h*w, 128) bf16, the velocity
                          ``scheduler.step`` consumed (CFG-combined for the base model, with ``.velocity_uncond`` /
                          ``.velocity_cond`` beside it); ``t`` (steps,) fp32 and ``t_next`` (steps,) fp32 (sigmas *
                          1000); ``final_tokens`` (h*w, 128) fp32, the state after the last step (what the VAE
                          decodes); ``image_f32`` (3, H, W) fp32, the decoded image after ``denormalize`` (in
                          ``[0, 1]``), before the uint8 rounding.
  ``golden.png``          ``round(image_f32 * 255)`` as uint8 RGB (``numpy_to_pil`` of ``pt_to_numpy``).
  ``index.json`` (root)   ``device`` (``cpu``), ``cases.<model>/<case>`` with the replay block and the file sha256s.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ....pipelines.flux2_klein.weights_flux2_klein import (
    FLUX2_KLEIN_MODEL_SIZES,
    FLUX2_KLEIN_VARIANTS,
    MODEL_PATH_ENV,
    WeightsNotFoundError,
    cached_snapshot,
    names_other_variant,
    repo_id,
    resolve_weights_dir,
)

GOLDENS_ENV = "FLUX2_KLEIN_GOLDENS"
ORACLE_CACHE_ENV = "FLUX2_KLEIN_ORACLE_CACHE"
RENDERS_ENV = "FLUX2_KLEIN_OUT"

SIZES = FLUX2_KLEIN_MODEL_SIZES
VARIANTS = FLUX2_KLEIN_VARIANTS
# DiT mesh layouts on the 4x8 Galaxy as ``(sp_axis, tp_axis)``: SP8 x TP4 is the port's default
# (``pipelines/flux2_klein/pipeline_flux2_klein.py`` ``sp_axis=1, tp_axis=0``), SP4 x TP8 the alternative.
LAYOUTS: dict[str, tuple[int, int]] = {"sp8tp4": (1, 0), "sp4tp8": (0, 1)}
DEFAULT_LAYOUT = "sp8tp4"
MESH_SHAPE = (4, 8)

TEXT_LENGTH = 512
TEXT_TAPS = (9, 18, 27)
ORACLE_THREADS = 24

DIT_CHECKPOINT = "transformer/diffusion_pytorch_model.safetensors"
TEXT_ENCODER_INDEX = "text_encoder/model.safetensors.index.json"

BF16 = torch.bfloat16

# The vendored-free references: one torch model per snapshot per process, built lazily on the first oracle miss.
_REFERENCE_TEXT_ENCODERS: dict[Path, tuple] = {}
_REFERENCE_TRANSFORMERS: dict[Path, object] = {}


# --- checkpoints ------------------------------------------------------------------------------------------------------


def variant_checkpoint_dir(variant: str, size: str, required: tuple[str, ...] | None = None) -> Path | None:
    """The snapshot of ``(variant, size)`` for a test that needs THAT checkpoint. ``resolve_weights_dir`` lets an
    explicit ``FLUX2_KLEIN_MODEL_PATH`` win unless it identifiably names another model; here such a path is passed
    over and the HuggingFace cache is searched instead. ``None`` when nothing resolves (callers skip)."""
    try:
        resolved = resolve_weights_dir(variant, size, required=required, allow_download=False)
    except WeightsNotFoundError:
        resolved = None
    if resolved is not None and not names_other_variant(resolved, variant, size):
        return resolved
    return cached_snapshot(variant, size, required)


def require_checkpoint_dir(variant: str, size: str, required: tuple[str, ...] | None = None) -> Path:
    """:func:`variant_checkpoint_dir` that skips the test instead of returning ``None``."""
    found = variant_checkpoint_dir(variant, size, required)
    if found is None:
        pytest.skip(
            f"FLUX.2 Klein '{variant}' {size} snapshot ({repo_id(variant, size)}) not found: set {MODEL_PATH_ENV} or "
            f"fill the HuggingFace cache (the weights are gated)"
        )
    return found


def blob_of(path: Path) -> str:
    """The HuggingFace blob name of a snapshot file: the basename of the symlink's OWN target (in a hub cache the
    file's sha256), not of the fully resolved path, which a shared blob store may rename; ``name:size`` for a plain
    file."""
    if path.is_symlink():
        return Path(os.readlink(path)).name
    return f"{path.name}:{path.stat().st_size}"


def dit_blob(snapshot: Path) -> str:
    return blob_of(Path(snapshot) / DIT_CHECKPOINT)


def text_encoder_blobs(snapshot: Path) -> list[str]:
    """The blob names of every ``text_encoder`` shard the index routes to, in shard order."""
    index = json.loads((Path(snapshot) / TEXT_ENCODER_INDEX).read_text())["weight_map"]
    return [blob_of(Path(snapshot) / "text_encoder" / shard) for shard in sorted(set(index.values()))]


# --- goldens ----------------------------------------------------------------------------------------------------------


def goldens_root() -> Path | None:
    """``$FLUX2_KLEIN_GOLDENS`` when it names an existing directory."""
    value = os.environ.get(GOLDENS_ENV)
    if not value:
        return None
    root = Path(value)
    return root if root.is_dir() else None


def golden_cases(root: Path | None = None) -> list[Path]:
    """Every case directory (one holding ``steps.safetensors`` and ``record.json``) under ``root``, sorted."""
    root = goldens_root() if root is None else root
    if root is None:
        return []
    return sorted(p.parent for p in root.rglob("steps.safetensors") if (p.parent / "record.json").is_file())


@dataclass
class GoldenCase:
    path: Path
    record: dict
    inputs: dict[str, torch.Tensor]
    steps: dict[str, torch.Tensor]

    @property
    def name(self) -> str:
        return f"{self.path.parent.name}/{self.path.name}"

    @property
    def replay(self) -> dict:
        """The generator's ``replay`` block. A golden without it is malformed: the model is never defaulted."""
        assert "replay" in self.record, f"{self.path}/record.json has no 'replay' block (generate_goldens format)"
        return self.record["replay"]

    @property
    def model(self) -> str:
        model = self.replay["model"]
        assert model in VARIANTS, f"{self.path}: unknown model {model!r}"
        return model

    @property
    def size(self) -> str:
        size = self.replay["size"]
        assert size in SIZES, f"{self.path}: unknown size {size!r}"
        return size

    @property
    def guidance(self) -> float | None:
        """CFG scale of the base model; ``None`` for the distilled model (and asserted to be so)."""
        guidance = self.replay["guidance"]
        assert (guidance is None) == (self.model == "distilled"), f"{self.path}: {self.model} with guidance {guidance}"
        return None if guidance is None else float(guidance)

    @property
    def prompt(self) -> str:
        return self.replay["prompt"]

    @property
    def seed(self) -> int:
        return int(self.replay["seed"])

    @property
    def canvas(self) -> tuple[int, int]:
        """``(width, height)``."""
        width, height = (int(v) for v in self.replay["canvas"])
        return width, height

    @property
    def num_steps(self) -> int:
        steps = int(self.steps["t"].shape[0])
        assert steps == int(
            self.replay["steps"]
        ), f"{self.path}: {steps} recorded steps vs replay.steps {self.replay['steps']}"
        return steps

    def x_t(self, i: int) -> torch.Tensor:
        """The fp32 latent state entering step ``i``; ``i == num_steps`` is the state after the last step."""
        return self.steps["final_tokens"] if i == self.num_steps else self.steps[f"step{i:02d}.x_t"]

    def velocity(self, i: int) -> torch.Tensor:
        """The velocity the Euler update of step ``i`` consumed (CFG-combined for the base model)."""
        return self.steps[f"step{i:02d}.velocity"]

    def text(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs["text_tokens"], self.inputs["text_ids"]

    def text_null(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The empty-prompt encoding of a CFG golden."""
        return self.inputs["text_null_tokens"], self.inputs["text_null_ids"]


def load_golden_case(case_dir: Path) -> GoldenCase:
    from safetensors.torch import load_file

    return GoldenCase(
        path=case_dir,
        record=json.loads((case_dir / "record.json").read_text()),
        inputs=load_file(str(case_dir / "inputs.safetensors")),
        steps=load_file(str(case_dir / "steps.safetensors")),
    )


def golden_index(root: Path | None = None) -> dict:
    """The nearest ``index.json`` at or above ``root`` (empty when none)."""
    root = goldens_root() if root is None else root
    for directory in (root, *root.parents) if root is not None else ():
        index = directory / "index.json"
        if index.is_file():
            return json.loads(index.read_text())
    return {}


def golden_for(model: str, size: str, *, canvas: tuple[int, int], steps: int | None = None) -> GoldenCase:
    """The first golden of ``model`` / ``size`` rendered at ``canvas`` (and ``steps`` when given); skips when none."""
    for case_dir in golden_cases():
        replay = json.loads((case_dir / "record.json").read_text()).get("replay", {})
        if replay.get("model") != model or replay.get("size") != size:
            continue
        if tuple(int(v) for v in replay.get("canvas", ())) != tuple(canvas):
            continue
        if steps is not None and int(replay.get("steps", -1)) != steps:
            continue
        return load_golden_case(case_dir)
    pytest.skip(
        f"no {model} {size} golden at {canvas[0]}x{canvas[1]}{'' if steps is None else f' with {steps} steps'} under "
        f"{GOLDENS_ENV} (run generate_goldens_flux2_klein.py)"
    )


def golden_png(golden: GoldenCase):
    from PIL import Image

    path = golden.path / "golden.png"
    if not path.is_file():
        pytest.skip(f"{path} missing")
    return Image.open(path).convert("RGB")


# --- oracle cache -----------------------------------------------------------------------------------------------------


def oracle_cache_dir() -> Path | None:
    """``$FLUX2_KLEIN_ORACLE_CACHE``; ``None`` recomputes every oracle."""
    value = os.environ.get(ORACLE_CACHE_ENV)
    return Path(value) if value else None


def oracle_digest(fields: dict) -> str:
    """sha256 over ``k=v`` of ``fields`` in insertion order (the cache key)."""
    return hashlib.sha256("|".join(f"{k}={fields[k]}" for k in fields).encode()).hexdigest()


def cached_oracle(fields: dict, label: str, compute) -> dict[str, torch.Tensor]:
    """``compute()`` -> ``{name: tensor}`` through the oracle cache: ``<digest>.safetensors`` + ``<digest>.json``
    under :func:`oracle_cache_dir`. A miss computes, writes (when a cache is set) and logs ``ORACLE MISS <digest>``;
    a hit logs ``ORACLE HIT <digest>``."""
    from loguru import logger
    from safetensors.torch import load_file, save_file

    digest = oracle_digest(fields)
    root = oracle_cache_dir()
    if root is not None and (root / f"{digest}.safetensors").is_file():
        logger.info(f"ORACLE HIT {digest[:16]} ({label})")
        return load_file(str(root / f"{digest}.safetensors"))
    logger.info(f"ORACLE MISS {digest[:16]} ({label})")
    torch.set_num_threads(ORACLE_THREADS)
    t0 = time.time()
    tensors = {k: v.detach().contiguous() for k, v in compute().items()}
    logger.info(f"ORACLE {label}: computed in {time.time() - t0:.1f} s")
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(root / f"{digest}.safetensors"))
        (root / f"{digest}.json").write_text(json.dumps({**fields, "label": label}, indent=1))
    return tensors


def renders_dir() -> Path | None:
    """``$FLUX2_KLEIN_OUT``; ``None`` saves nothing."""
    value = os.environ.get(RENDERS_ENV)
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- comparison helpers -----------------------------------------------------------------------------------------------


def pcc(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Pearson correlation of two tensors, in fp64."""
    a = reference.detach().double().flatten()
    b = test.detach().double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return 1.0 if denominator == 0 else float((a @ b) / denominator)


def rel_rmse_of(reference: torch.Tensor, got: torch.Tensor) -> float:
    """``rmse(reference - got) / std(reference)`` in fp64 (the ``assert_quality`` definition)."""
    a, b = reference.detach().double(), got.detach().double()
    return float(((a - b) ** 2).mean().sqrt() / a.std())


def measure(reference: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    """``(pcc, relative rmse, max abs)`` in fp64."""
    a, b = reference.detach().double().flatten(), got.detach().double().flatten()
    return pcc(a, b), rel_rmse_of(a, b), float((a - b).abs().max())


def psnr_uint8(a, b) -> float:
    """PSNR between two uint8 RGB images (PIL or arrays); ``inf`` when identical."""
    import numpy as np

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def sha256_of(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def image_f32_to_pil(image: torch.Tensor):
    """``numpy_to_pil(pt_to_numpy(x))`` of a denormalized ``[3, H, W]`` image: ``round(x * 255)`` to uint8 RGB."""
    from PIL import Image

    array = (image.detach().float().permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


# --- CPU references ---------------------------------------------------------------------------------------------------


def reference_text_encoder(snapshot: Path):
    """``(tokenizer, Qwen3ForCausalLM)`` of ``snapshot`` in bf16 on CPU, loaded once per process."""
    from loguru import logger

    snapshot = Path(snapshot)
    if snapshot not in _REFERENCE_TEXT_ENCODERS:
        import transformers

        torch.set_num_threads(ORACLE_THREADS)
        t0 = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), subfolder="tokenizer")
        model = transformers.Qwen3ForCausalLM.from_pretrained(
            str(snapshot), subfolder="text_encoder", torch_dtype=BF16
        ).eval()
        _REFERENCE_TEXT_ENCODERS[snapshot] = (tokenizer, model)
        logger.info(f"reference Qwen3ForCausalLM ({model.config.hidden_size} hidden) loaded in {time.time() - t0:.1f}s")
    return _REFERENCE_TEXT_ENCODERS[snapshot]


def reference_tokenize(
    tokenizer, prompt: str, sequence_length: int = TEXT_LENGTH
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """``P:226-239`` re-executed: ``(rendered text, input_ids [1, L], attention_mask [1, L])``."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=sequence_length)
    return text, inputs["input_ids"], inputs["attention_mask"]


def reference_prompt_embeds(
    snapshot: Path, prompt: str, *, sequence_length: int = TEXT_LENGTH, taps: tuple[int, ...] = TEXT_TAPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``P:208-263`` (``_get_qwen3_prompt_embeds``) on the CPU reference: ``(tokens [L, joint] bf16, input_ids [1, L],
    mask [1, L])`` with ``tokens[l] = [hidden_states[9] | hidden_states[18] | hidden_states[27]]`` of token ``l``
    (``torch.stack(..., dim=1).permute(0, 2, 1, 3).reshape``)."""
    tokenizer, model = reference_text_encoder(snapshot)
    _, input_ids, mask = reference_tokenize(tokenizer, prompt, sequence_length)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        out = torch.stack([output.hidden_states[k] for k in taps], dim=1).to(dtype=BF16)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        tokens = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    return tokens[0].contiguous(), input_ids, mask


def reference_transformer(snapshot: Path):
    """The ``Flux2Transformer2DModel`` of ``snapshot`` in bf16 on CPU (eval), loaded once per process."""
    from loguru import logger

    snapshot = Path(snapshot)
    if snapshot not in _REFERENCE_TRANSFORMERS:
        import diffusers

        torch.set_num_threads(ORACLE_THREADS)
        t0 = time.time()
        model = diffusers.Flux2Transformer2DModel.from_pretrained(
            str(snapshot), subfolder="transformer", torch_dtype=BF16, low_cpu_mem_usage=True
        ).eval()
        _REFERENCE_TRANSFORMERS[snapshot] = model
        logger.info(
            f"reference Flux2Transformer2DModel ({len(model.transformer_blocks)} dual + "
            f"{len(model.single_transformer_blocks)} single blocks) loaded in {time.time() - t0:.1f}s"
        )
    return _REFERENCE_TRANSFORMERS[snapshot]


def reference_code_sha() -> str:
    """sha256 of the installed diffusers ``transformer_flux2.py`` bytes plus the library versions: part of every DiT
    oracle key, so a diffusers upgrade can never serve a stale velocity."""
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as module

    h = hashlib.sha256(Path(module.__file__).read_bytes())
    h.update(f"|diffusers={diffusers.__version__}|torch={torch.__version__}".encode())
    return h.hexdigest()
