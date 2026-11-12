# SPDX-License-Identifier: Apache-2.0
"""h3/synth_gen.py: weight-free synthetic MiniMax-H3 ref2va generations (tray-test goal 48, reliability.sh h3 step).

h3.sh copies this file into the tt-metal tree as tests/models/minimax_h3/test_h3_synth_gen.py and runs it with pytest,
so the root conftest opens the mesh exactly as the H3 perf test does: (4, 8), FABRIC_1D_RING with the 8 KB router
payload, l1_small_size 16384 and dit_fsdp. Denoise flags are the 4x8 preset's (bucket_denoise on, trace_denoise off,
no persistent CCL buffers): the perf test's REF2VA_SERVING_FLAGS add trace_denoise=True, which this tt-metal commit
refuses on 4x8 (traced denoise needs persistent CCL buffers; the 4x8 preset has them off), so one galaxy serves untraced.

Weights: none are read. A weights tree is built under H3_SYNTH_WORK with the public configs (h3/config) and one empty
placeholder per shard. safetensors' load_file and safe_open are replaced by a generator that returns, for every tensor
of the public MiniMax-H3 checkpoint (names, shapes and dtypes from h3/config/shapes.json.gz), seeded random values
scaled so activations stay finite: 1/sqrt(fan_in) for matrices and convolutions, ones for norms and gains, 0.02 for
biases. They then go through the pipeline's own load path, so every op, shape, dtype, sharding and CCL is the real one.
The tokenizer is a stand-in that maps words to ids; the special tokens keep their real ids.

Setup in minutes (the 57 min before was 49 min of warmup and 7 min of weight conversion, see
results/h3-setup/breakdown.md):
- Device weights: every device parameter of the transformer (the bulk of the weights) is made on the device in its final dtype, layout and memory config with a
  seeded ttnn.rand (uniform, the host tensor's mean and spread) or ttnn.full (a constant host tensor, e.g. zero_k_bias,
  stays that constant), so no weight is converted, tilized or uploaded from the host. The host tensors still pass
  through the model's own load path first, so its shape checks and asserts run. Host parameters, integer tensors and
  any tensor the device ops refuse take the stock path; 'H3 WEIGHTS' counts both. H3_SYNTH_DEVICE_WEIGHTS=0 turns it off.
  The other models (video VAE, text encoder, audio models) keep the stock path: made-up values from a sampled spread
  turned the video VAE's reference latents non-finite (job 172950: 'video conditioning rows changed').
- No warmup: the pipeline compiles each bucket the first time a gen uses it, instead of every bucket, canvas, audio
  length and prompt length up front. Audio decode then runs untraced (a capture needs the warmup's compile first).
  H3_SYNTH_WARMUP=1 restores the served warmup.
'H3 PHASE <name> <start> <seconds>' lines time the setup phases.

Environment (set by h3.sh):
  H3_SYNTH_MODE     random | full
  H3_SYNTH_TABLE    random mode: TSV of 'gens<TAB>pause_s' rows
  H3_SYNTH_COOLDOWNS  full mode: TSV of 'offset_s<TAB>duration_s' zero-load cooldowns (h3/full-cooldowns.tsv), offsets
                    from the first gen; each starts at the first gen boundary at or after its offset, one gen at least
                    between two
  H3_SYNTH_TIME_SCALE  cooldown offsets and durations times this (reliability.sh --time-scale; default 1)
  H3_SYNTH_COOLDOWN_END_S  full mode: no cooldown starts at or after this many s from the first gen and none runs past it,
                    times H3_SYNTH_TIME_SCALE (h3.sh --cooldowns-end; reliability.sh 1800): continuous load from there.
                    Unset or 0: no limit
  H3_SYNTH_STOP_AT  epoch; no gen starts at or after it
  H3_SYNTH_GENLOG   file the 'H3 GEN' lines are appended to
  H3_SYNTH_CONFIG   h3/config
  H3_SYNTH_WORK     scratch dir for the placeholder weights tree
  H3_SYNTH_SEED     weight seed (default 4912)
  H3_SYNTH_WARMUP   1: the served warmup at setup (default 0)
  H3_SYNTH_DEVICE_WEIGHTS  0: convert host weights the stock way (default 1)
  H3_SYNTH_HOST_VIEWS      0: materialize every transformer and text encoder tensor on the host (default 1: large ones are views, synth_weights.py)
  H3_PCIE_TABLE     h3/pcie-traffic.tsv: the step's LTX-like PCIe traffic (h3/pcie_traffic.py)
  H3_PCIE_TRAFFIC   0: no PCIe traffic (h3.sh --no-pcie-traffic; 'H3 PCIE WARNING' at start and with every gen)

PCIe traffic (default on): before the pipeline is built, a weight reload of reload_mb_per_chip to every chip, each chunk
read back and checked ('H3 PCIE RELOAD', 'H3 PCIE CHIP'); around every gen, gen_mb_per_chip written before it and read
back and checked after it ('H3 PCIE GEN <n>'). A readback that differs is 'H3 PCIE FAIL <bdf> tray <t> ASIC <a>
direction=...' and fails the step after that gen's lines.

Stop signals: SIGTERM and SIGINT raise KeyboardInterrupt in the main thread, so pytest runs the fixture
teardown that closes the mesh device and the process ends with the device closed (a broker preemption then needs no
reset). The first one prints 'H3 SIGNAL <name> <epoch> closing the mesh device' and the exit prints
'H3 SIGNAL-EXIT <name> <seconds> after the signal'; any later stop signal is ignored so it cannot cut the teardown short.

Output, one line per gen: 'H3 GEN <n> <start> <end> <seconds> ok|fail:<why>', plus 'H3 CASE', 'H3 POWER', 'H3 COND' and
'H3 SETUP' info lines. Full mode with cooldowns: 'H3 COOLDOWN-START <k> <start> offset=<s> planned_s=<s>' as each starts, then
'H3 COOLDOWN <k> <start> <end> <seconds> offset=<s> planned_s=<s> late_s=<s> <chip power>'; a row due at or after H3_SYNTH_COOLDOWN_END_S: 'H3 COOLDOWN-SKIP <k> <epoch> offset=<s>
planned_s=<s> end_s=<s>', and no pause runs. 'H3 COND' counts the gen's conditioning rows and the non-finite ones; a non-finite one fails the gen
before denoising with that cause (see _guard_conditioning). A failed gen ends the run with a test failure.
"""

from __future__ import annotations

import atexit
import inspect
import math
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn

from ....layers.module import Parameter
from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import get_num_frames
from .common import create_fractal_image
from .synth_weights import SEED, SyntheticTokenizer, SyntheticWeights
from .pcie_traffic import OFF_WARNING as PCIE_OFF_WARNING, RELOAD_RESERVE_S, for_mesh as pcie_for_mesh
from .common_av import CALIBRATED_FOX_PROMPT, is_host
from .test_performance_minimax_h3 import REF2VA_MESHES

E = os.environ.get
# Served H3 (lightx2v Turbo, 4 forwards): 5 sigma grid points, video shift 6 for the 768p files.
NUM_INFERENCE_STEPS = int(E("H3_SYNTH_STEPS", "5"))
VIDEO_SHIFT = float(E("H3_SYNTH_VIDEO_SHIFT", "6"))
SUBFOLDERS = ("transformer_ref", "text_encoder", "vae", "audio_vae")

# The served request mix: name, duration s, references, weight. Mean about 31 s per gen.
CASES = (
    ("REF-5-1img", 5, "1img", 0.40),
    ("REF-5-large-img", 5, "large-img", 0.20),
    ("REF-10-9img", 10, "9img", 0.22),
    ("REF-15-vid15", 15, "vid15", 0.18),
)


def _install(weights: SyntheticWeights, tokenizer: SyntheticTokenizer) -> None:
    import safetensors.torch
    import transformers

    from ....encoders.qwen3vl import loader_minimax_h3
    from ....pipelines.minimax_h3 import pipeline_minimax_h3_turbo

    safetensors.torch.load_file = weights.load_file
    loader_minimax_h3.safe_open = weights.safe_open
    pipeline_minimax_h3_turbo.safe_open = weights.safe_open
    real = transformers.AutoTokenizer.from_pretrained

    def from_pretrained(path, *args, subfolder=None, **kwargs):
        if subfolder == "tokenizer" and Path(path) == weights.root:
            return tokenizer
        return real(path, *args, subfolder=subfolder, **kwargs)

    transformers.AutoTokenizer.from_pretrained = from_pretrained


# ----------------------------------------------------------------------------------------------- requests


class _DeviceWeights:
    """Makes each device parameter on the device instead of converting the host tensor (see the module docstring)."""

    FLOAT = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32)
    PHASES = ("_prepare_transformer",)  # setup steps whose parameters are made on the device

    def __init__(self, seed: int):
        self.seed = seed
        self.device = self.host = 0
        self.device_s = self.host_s = 0.0
        self.why: dict[str, int] = {}
        self.stack: list[str] = []  # the setup steps running now, innermost last

    @property
    def active(self) -> bool:
        return bool(self.stack) and self.stack[-1] in self.PHASES

    def _make(self, param, torch_tensor: torch.Tensor):
        if param.on_host or param.dtype not in self.FLOAT or not torch_tensor.is_floating_point():
            raise TypeError("host or integer parameter")
        row = torch_tensor
        while row.dim() > 1:
            row = row[0]
        sample = row[:4096].float()
        if sample.numel() == 0 or not torch.isfinite(sample).all():
            raise ValueError("empty or non-finite sample")
        shape = list(param.local_shape)
        dram = ttnn.DRAM_MEMORY_CONFIG
        lo, hi = float(sample.min()), float(sample.max())
        if lo == hi:
            data = ttnn.full(shape, lo, dtype=param.dtype, layout=param.layout, device=param.device, memory_config=dram)
        else:
            mean, half = float(sample.mean()), float(sample.std()) * math.sqrt(3.0)
            data = ttnn.rand(
                shape, param.device, dtype=param.dtype, layout=param.layout, memory_config=dram,
                low=mean - half, high=mean + half, seed=(self.seed * 100003 + self.device) % 2**31 + 1,
            )
        if param.memory_config != dram:
            try:
                moved = ttnn.to_memory_config(data, param.memory_config)
            except BaseException:
                ttnn.deallocate(data)  # do not leak device DRAM before the stock path allocates
                raise
            ttnn.deallocate(data)
            data = moved
        return data

    def install(self) -> None:
        original = Parameter.load_torch_tensor
        dw = self

        def load_torch_tensor(param, torch_tensor, /):
            start = time.time()
            if (dw.active and tuple(torch_tensor.shape) == param.total_shape
                    and E("H3_SYNTH_DEVICE_WEIGHTS", "1") != "0"):
                data = None
                try:
                    data = dw._make(param, torch_tensor)
                    param.data = data  # Parameter's own dtype, layout, memory config and shape checks
                except Exception as err:  # noqa: BLE001 - any refusal takes the stock path
                    if data is not None and not param.on_host:
                        ttnn.deallocate(data)
                    key = f"{type(err).__name__}: {str(err).splitlines()[0][:60] if str(err) else ''}"
                    dw.why[key] = dw.why.get(key, 0) + 1
                else:
                    dw.device += 1
                    dw.device_s += time.time() - start
                    return
            original(param, torch_tensor)
            dw.host += 1
            dw.host_s += time.time() - start

        Parameter.load_torch_tensor = load_torch_tensor

    def summary(self) -> str:
        why = ";".join(f"{k}={v}" for k, v in sorted(self.why.items(), key=lambda kv: -kv[1])[:4])
        return (f"device={self.device} device_s={self.device_s:.1f} host={self.host} host_s={self.host_s:.1f}"
                f" host_why={why or '-'}")


_PHASES = ("_read_safetensors", "_prepare_text_encoder", "_prepare_transformer", "_prepare_audio_decoder",
           "_warmup_on_init")


def _time_phases(emit, device_weights: _DeviceWeights) -> None:
    """Wraps the pipeline's setup steps so each logs 'H3 PHASE <name> <start> <seconds>' and device_weights knows
    which of them runs innermost."""
    for name in _PHASES:
        fn = getattr(MiniMaxH3Pipeline, name, None)
        if fn is None:
            continue

        def timed(*args, __fn=fn, __name=name, **kwargs):
            start = time.time()
            device_weights.stack.append(__name)
            try:
                return __fn(*args, **kwargs)
            finally:
                device_weights.stack.pop()
                emit(f"H3 PHASE {__name.lstrip('_')} {start:.0f} {time.time() - start:.1f}")

        setattr(MiniMaxH3Pipeline, name, timed)


def _guard_conditioning(emit) -> None:
    """Wraps the pipeline's _denoise so each gen logs 'H3 COND video=<rows> nonfinite_video=<n> audio=... ' and fails
    before the loop when a conditioning row is not finite. The loop never writes the conditioning rows; its own
    end-of-loop self-check compares them with a copy, and NaN never equals NaN, so non-finite reference latents read
    there as 'conditioning rows changed during denoising' (job 172950: 252 of 1008, after made-up device values
    reached the video VAE). This names the real cause; the self-check itself stays as it is."""
    fn = MiniMaxH3Pipeline._denoise
    sig = inspect.signature(fn)

    def guarded(*args, **kwargs):
        a = sig.bind(*args, **kwargs).arguments
        layout, parts, bad = a["layout"], [], []
        for name, rows, n in (("video", a["video_rows"], layout.num_condition_video_rows),
                              ("audio", a["audio_rows"], layout.num_condition_audio_rows)):
            nonfinite = int((~torch.isfinite(rows[:n])).any(dim=-1).sum()) if n else 0
            parts.append(f"{name}={n} nonfinite_{name}={nonfinite}")
            if nonfinite:
                bad.append(f"{nonfinite} of {n} {name}")
        emit("H3 COND " + " ".join(parts))
        if bad:
            raise RuntimeError(f"non-finite conditioning rows before denoising ({', '.join(bad)}): the synthetic "
                               "encoder weights made non-finite reference latents, not the loop's write mask")
        return fn(*args, **kwargs)

    MiniMaxH3Pipeline._denoise = guarded


VID15_SIZE = (1280, 720)  # REF-15-vid15's frame width and height (the offline test runs the case smaller)


def _references(kind: str, rng: np.random.Generator, audio_rate: int) -> list[MiniMaxH3Reference]:
    if kind == "1img":
        return [MiniMaxH3Reference(image=create_fractal_image(1024, 1024))]
    if kind == "large-img":
        return [MiniMaxH3Reference(image=create_fractal_image(2560, 1440))]
    if kind == "9img":
        out = []
        for i in range(9):
            noise = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
            base = np.asarray(create_fractal_image(512, 512), dtype=np.uint16)
            out.append(MiniMaxH3Reference(image=Image.fromarray(((base * (i + 1) + noise) % 256).astype(np.uint8))))
        return out
    if kind == "vid15":
        frames = 15 * 24
        width, height = VID15_SIZE
        x = np.linspace(0, 4 * np.pi, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 4 * np.pi, height, dtype=np.float32)[:, None]
        video = np.empty((frames, height, width, 3), dtype=np.uint8)
        for f in range(frames):
            phase = f * 0.11
            for c in range(3):
                video[f, :, :, c] = (127.5 + 127.5 * np.sin(x * (1 + c) + y + phase)).astype(np.uint8)
        # At the audio VAE's own rate, as served requests are: any other rate needs torchaudio's resample,
        # which the venv lacks, and every vid15 gen failed at once with ModuleNotFoundError.
        rate = int(audio_rate)
        t = torch.arange(15 * rate, dtype=torch.float32) / rate
        audio = torch.stack([0.3 * torch.sin(2 * math.pi * 220 * t), 0.3 * torch.sin(2 * math.pi * 330 * t)])
        return [MiniMaxH3Reference(video=video, fps=24.0, audio=audio, sample_rate=rate)]
    raise ValueError(kind)


def _plan(mode: str):
    """Yields ('gen', case_index) and ('pause', seconds)."""
    rng = np.random.default_rng(SEED)
    weights = np.array([c[3] for c in CASES]) / sum(c[3] for c in CASES)
    if mode == "full":
        while True:
            yield "gen", int(rng.choice(len(CASES), p=weights))
    rows = []
    for line in Path(E("H3_SYNTH_TABLE")).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            gens, pause = line.split("\t")[:2]
            rows.append((int(gens), float(pause)))
    while True:  # the table repeats if the stop time is beyond its end
        for gens, pause in rows:
            for _ in range(gens):
                yield "gen", int(rng.choice(len(CASES), p=weights))
            yield "pause", pause


def _cooldowns() -> list[tuple[float, float]]:
    """(offset_s, duration_s) rows of H3_SYNTH_COOLDOWNS, scaled by H3_SYNTH_TIME_SCALE, in offset order."""
    path = E("H3_SYNTH_COOLDOWNS", "")
    if not path:
        return []
    scale = float(E("H3_SYNTH_TIME_SCALE", "1") or 1)
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            offset, duration = line.split("\t")[:2]
            rows.append((float(offset) * scale, float(duration) * scale))
    return sorted(rows)


def _cooldown_due(cooldowns: list, elapsed: float, end_s: float):
    """-> (row, skipped). row: the (offset_s, planned_s) cooldown due `elapsed` s after the first gen, popped from
    `cooldowns`, else None. With end_s > 0 no cooldown runs past end_s and none starts at or after it: from there the
    rows left are popped into `skipped` and the load runs continuously to the step end."""
    if end_s > 0 and elapsed >= end_s:
        skipped = list(cooldowns); cooldowns.clear()
        return None, skipped
    if not cooldowns or elapsed < cooldowns[0][0]:
        return None, []
    offset, planned = cooldowns.pop(0)
    return (offset, min(planned, end_s - elapsed) if end_s > 0 else planned), []


class _Power:
    """Sum of every chip's hwmon power1_input while a gen runs (chip kW, not box kW)."""

    def __init__(self):
        self.paths = sorted(
            p for p in Path("/sys/class/hwmon").glob("hwmon*/power1_input")
            if os.path.basename(os.path.realpath(p.parent / "device" / "driver")) == "tenstorrent"
        )
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.wait(1.0):
            total = 0.0
            for p in self.paths:
                try:
                    total += int(p.read_text()) / 1e6
                except (OSError, ValueError):
                    pass
            self.samples.append(total / 1000.0)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        if self.paths:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return False

    def summary(self) -> str:
        if not self.samples:
            return f"chips={len(self.paths)} no-samples meter=chip_hwmon"
        s = sorted(self.samples)
        return (
            f"chips={len(self.paths)} n={len(s)} mean_kw={sum(s) / len(s):.2f} p95_kw={s[int(0.95 * (len(s) - 1))]:.2f}"
            f" max_kw={s[-1]:.2f} meter=chip_hwmon"
        )


def _emit(line: str) -> None:
    print(line, flush=True)
    with open(E("H3_SYNTH_GENLOG"), "a") as fh:
        fh.write(line + "\n")


def _fail_why(error: BaseException) -> str:
    # One token (power_profile.py's GEN_RE reads it as \S+): the type, then the message, so the cause is on the line.
    detail = "_".join(str(error).split())[:160]
    return f"fail:{type(error).__name__}" + (f":{detail}" if detail else "")


def _emit_error(n: int, error: BaseException) -> None:
    # The whole exception, uncut: one 'H3 ERROR' line in gens.log and h3.log, then the traceback in h3.log.
    text = str(error).replace("\\", "\\\\").replace("\n", "\\n")
    kind = type(error).__qualname__
    if type(error).__module__ != "builtins":
        kind = f"{type(error).__module__}.{kind}"
    _emit(f"H3 ERROR {n} {kind}: {text}")
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stdout)
    sys.stdout.flush()


def _pcie(emit, step):
    """Run a PCIe traffic step; an exception gets an 'H3 PCIE ERROR' line, then ends the run."""
    try:
        return step()
    except Exception as error:
        emit(f"H3 PCIE ERROR {type(error).__name__}: {' '.join(str(error).split())[:300]}")
        raise


def _gen_report(emit, host, traffic, per_gen, n, case, power, start, end, why) -> bool:
    """Gen n's CASE/POWER/GEN lines first, then its PCIe readback only after an ok gen: on a failed or hung chip the
    readback could block or raise before the GEN fail line is written. False only when the readback differs."""
    if host:
        _emit(f"H3 CASE {n} {case}")
        _emit(f"H3 POWER {n} {power}")
        _emit(f"H3 GEN {n} {start:.0f} {end:.0f} {end - start:.1f} {why}")
        if traffic is None:
            _emit(f"H3 PCIE GEN {n} {PCIE_OFF_WARNING}")
    if traffic is None or not per_gen:
        return True
    if why != "ok":
        emit(f"H3 PCIE GEN {n} skipped: gen failed")
        return True
    return _pcie(emit, lambda: traffic.after_gen(n))


def _fail(emit, traffic, why: str) -> None:
    """End the run; the PCIe totals go out first, so a failed step still logs the GB it moved."""
    if traffic is not None:
        emit(traffic.summary())
    pytest.fail(why)


def _finite(output) -> bool:
    if not torch.isfinite(output.audio).all():
        return False
    video = output.video
    if isinstance(video, torch.Tensor) and video.is_floating_point():
        return bool(torch.isfinite(video).all())
    return True


class _StopSignal(KeyboardInterrupt):
    """A stop signal: KeyboardInterrupt, so pytest tears its fixtures down (closing the mesh) and no 'except Exception'
    in the gen loop takes it for a failed gen."""


def _install_stop_handlers() -> None:
    """SIGTERM and SIGINT raise _StopSignal once; later ones are ignored so the mesh close is never cut short."""

    def on_stop(signum, _frame):
        name = signal.Signals(signum).name
        t0 = time.time()
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        print(f"H3 SIGNAL {name} {t0:.0f} closing the mesh device", file=sys.stderr, flush=True)
        atexit.register(lambda: print(f"H3 SIGNAL-EXIT {name} {time.time() - t0:.1f} after the signal",
                                      file=sys.stderr, flush=True))
        raise _StopSignal(name)

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_stop)


_install_stop_handlers()

_MESH_4X8 = [p for p in REF2VA_MESHES if p.id == "4x8"]


@pytest.mark.timeout(0)
@pytest.mark.parametrize(("mesh_device", "device_params"), _MESH_4X8, indirect=["mesh_device", "device_params"])
def test_h3_synth_gen(mesh_device):
    mode = E("H3_SYNTH_MODE", "random")
    stop_at = float(E("H3_SYNTH_STOP_AT"))
    config_dir = Path(E("H3_SYNTH_CONFIG"))
    host = is_host()
    t0 = time.time()
    emit = _emit if host else (lambda line: None)

    traffic = _pcie(emit, lambda: pcie_for_mesh(ttnn, torch, mesh_device, mode, emit))
    if traffic is not None and not _pcie(emit, lambda: traffic.reload(stop_at, RELOAD_RESERVE_S, min_chunks=1)):
        _fail(emit, traffic, "PCIe weight reload: readback differs from what was written (H3 PCIE FAIL lines)")

    weights = SyntheticWeights(config_dir, Path(E("H3_SYNTH_WORK")))
    _install(weights, SyntheticTokenizer(config_dir / "tokenizer" / "tokenizer_config.json"))
    device_weights = _DeviceWeights(SEED)
    device_weights.install()
    _time_phases(_emit if host else lambda line: None, device_weights)
    _guard_conditioning(_emit if host else lambda line: None)
    warmup = E("H3_SYNTH_WARMUP", "0") == "1"
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights.root,
        task="ref2va",
        dit_fsdp=True,
        vae_output_type="yuv420",
        video_shift=VIDEO_SHIFT,
        bucket_denoise=True,
        warmup=warmup,
        trace_audio=None if warmup else False,
    )
    if host:
        _emit(f"H3 WEIGHTS {device_weights.summary()} warmup={int(warmup)}")
        _emit(f"H3 SETUP {t0:.0f} {time.time():.0f} {time.time() - t0:.1f} elements={weights.elements} hf_rev={weights.revision}")

    per_gen = not warmup  # warmup captures traces; their replay may write over per-gen DRAM tensors made after it
    if traffic is not None and not per_gen:
        emit("H3 PCIE GEN skipped for every gen: H3_SYNTH_WARMUP=1 traces replay over DRAM made after capture; reload traffic only")
    rng = np.random.default_rng(SEED + 1)
    power = _Power()
    n = 0
    cooldowns = _cooldowns() if mode == "full" else []
    n_cool = len(cooldowns)
    cool_end = float(E("H3_SYNTH_COOLDOWN_END_S", "0") or 0) * float(E("H3_SYNTH_TIME_SCALE", "1") or 1)
    load_t0 = None
    gen_since = False  # a gen ran since the last cooldown (two cooldowns never back to back)
    for kind, value in _plan(mode):
        if time.time() >= stop_at:
            break
        due, skipped = _cooldown_due(cooldowns, time.time() - load_t0, cool_end) if cooldowns and gen_since else (None, [])
        for i, (offset, planned) in enumerate(skipped):
            if host:
                _emit(f"H3 COOLDOWN-SKIP {n_cool - len(skipped) + i + 1} {time.time():.0f} offset={offset:.0f}"
                      f" planned_s={planned:.0f} end_s={cool_end:.0f}")
        if due:
            offset, planned = due
            k = n_cool - len(cooldowns)
            start = time.time()
            if host:
                _emit(f"H3 COOLDOWN-START {k} {start:.0f} offset={offset:.0f} planned_s={planned:.0f}")
            with power:
                time.sleep(max(0.0, min(planned, stop_at - time.time())))
            end = time.time()
            gen_since = False
            if host:
                _emit(f"H3 COOLDOWN {k} {start:.0f} {end:.0f} {end - start:.1f} offset={offset:.0f} planned_s={planned:.0f}"
                      f" late_s={start - load_t0 - offset:.0f} {power.summary()}")
            continue
        if kind == "pause":
            time.sleep(max(0.0, min(value, stop_at - time.time())))
            continue
        name, duration_s, refs, _ = CASES[value]
        n += 1
        references = _references(refs, rng, pipeline.audio_sampling_rate)
        aspect = (16, 9) if rng.random() < 0.5 else (9, 16)
        if traffic is not None and per_gen:
            _pcie(emit, lambda: traffic.before_gen(n))
        start = time.time()
        if load_t0 is None:
            load_t0 = start
        gen_since = True
        why = "ok"
        try:
            with power:
                output = pipeline(
                    CALIBRATED_FOX_PROMPT,
                    references=references,
                    num_frames=get_num_frames(duration_s),
                    aspect_ratio=aspect,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    seed=SEED + n,
                )
            if not _finite(output):
                why = "fail:nonfinite"
        except Exception as error:  # noqa: BLE001 - the line names it, then the run fails
            why = _fail_why(error)
            if host:
                _emit_error(n, error)
            output = None
        end = time.time()
        case = f"{name} {aspect[0]}x{aspect[1]} {duration_s}s steps={NUM_INFERENCE_STEPS}"
        pcie_ok = _gen_report(emit, host, traffic, per_gen, n, case, power.summary(), start, end, why)
        if why != "ok":
            _fail(emit, traffic, f"gen {n}: {why}")
        if not pcie_ok:
            _fail(emit, traffic, f"gen {n}: PCIe readback differs from what was written (H3 PCIE FAIL lines)")
    if traffic is not None:
        emit(traffic.summary())
    written = weights.shared_writes()
    if host and written:  # a model write into a shared VAE pool changes the weights it shares with
        _emit(f"H3 SYNTH WARNING shared weight pools written in place ({' '.join(written)}): VAE weights differ from the seeded tables")
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()
    pipeline.release_traces()
