# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for GLM-5.3-Flash on the Blackhole Galaxy: ``GLM53MeshModel`` behind the TT vLLM plugin's
generator contract (``vllm-tt-plugin`` ``model_runner.py`` / ``async_decode.py``; design and status in
``~/ashai/glm53/_agents/vllm_glm53/{DESIGN,STATUS}.md``). DRAFT: written without device access; every
device-facing path carries a ``TODO(device)`` where the first server run has to confirm it.

Registration (no plugin edit needed for the first smoke; a ``_register_model_if_missing`` entry in
``platform.py::register_tt_models`` is the permanent form)::

    # bare + "TT"-prefixed names: the plugin prefixes every checkpoint architecture with "TT" before it resolves
    # the class (platform.py check_and_update_config), and the override map is consulted for both forms
    export TT_MODEL_CLASS_OVERRIDES="Glm5NextForConditionalGeneration=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM,TTGlm5NextForConditionalGeneration=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM,Glm5NextForCausalLM=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM,TTGlm5NextForCausalLM=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM"
    MESH_DEVICE=BH-Galaxy GLM53_MESH_SHAPE=8,4 vllm serve zai-org/GLM-5.3-Flash --max-model-len 1048576 --max-num-seqs 4 \\
        --block-size 32 --max-num-batched-tokens 16384 --no-enable-prefix-caching \\
        --additional-config '{"tt": {"trace_region_size": 536870912, "l1_small_size": 32768,
        "sample_on_device_mode": "all", "prefill_chunk_alignment": 256, "decode_interleave_enabled": false}}'

Contract mapping (the plugin's names on the left):

* ``initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, ...)`` builds the resident
  mesh model (``users_per_row = max_batch_size`` distinct requests; the rows replicate them, see
  ``ServingGeometry.rows_independent``), with the user-partitioned DSA cache and the heads->users prefill
  (the measured envelope: 32 x 1 M, 64 x 512 K, 128 x 256 K, 256 x 128 K slots on the 8x4) after the
  memory planner confirmed the fit.
* ``warmup_model_prefill`` / ``warmup_model_decode``: eager phase = persistent decode inputs + one eager
  decode step + one eager prefill (compiles); traced phase = the per-length prefill traces, then the
  decode trace (nothing persistent is allocated after a capture); ``tt_supported_decode_batch_sizes`` is
  narrowed to the one captured width (every decode is padded to it: fixed user slots).
* ``prefill_forward(tokens, page_table, kv_cache, enable_trace, prompt_lens, start_pos, empty_slots,
  sampling_params)``: one prompt at a time through the traced prefill (``start == 0`` and the chunk fits a
  captured length) or the eager sequence-parallel prefill in device chunks (``PrefillPlanner``); returns
  host logits ``[B, 1, vocab]`` fp32 or, with device sampling, greedy tokens ``[B, 1]``. Intermediate chunk
  rows (``prompt_lens < num_tokens``) emit nothing.
* ``decode_forward(tokens, start_pos, page_table, kv_cache, enable_trace, read_from_device, sampling_params,
  slot_remap, reload_inputs, ...)`` (``decode_input_update_contract = 1``): the plugin's ``[B_pad, 1]``
  rows are permuted onto the physical user slots (``SlotTable``), uploaded with ``load_decode_inputs``
  (``load_ids`` + ``set_positions``; with ``GLM53_DECODE_PACKED_INPUTS=1`` one packed host->device copy per
  step and one chip read per mesh row for the ids, ``tt/packed_inputs.py``), the captured decode step
  replayed, and the greedy ids / logits read back and un-permuted (``read_decode_output`` /
  ``process_decode_output_host``). Idle slots are parked at ``(pad_token, position 0)``.
* ``release_request(slot)`` / ``note_state_slots_moved(moves)``: slot ownership follows the plugin's logical
  slot through the adapter's indirection table; device state never moves.
* ``get_kv_cache_spec`` / ``allocate_kv_cache`` / ``get_max_tokens_all_users``: vLLM's block pool is logical
  (``LogicalBlockTable``); the latent cache is device-owned per slot.

Per-row prefill ownership (``GLM53_VLLM_ROWS_INDEPENDENT=1``, design ``_agents/vllm_glm53/ROW_OWNERSHIP.md``):
``max_num_seqs = rows x users_per_row``; the prefill rows of a step become waves (one chunk of a different
prompt per mesh row, ``vllm_config.WavePlanner``) and, while exact, sequence-parallel single-prompt steps;
needs ``GLM53MeshModel.prefill_wave`` / ``prefill_traced_wave`` (the device side of that design; refused at
construction until it exists). Knobs: ``GLM53_VLLM_WAVE_ROWS`` (rows per mesh row per wave, 2048),
``GLM53_VLLM_SP_FALLBACK`` (1).

Not supported in this draft (declared off): async decode overlap, prefix caching, stochastic device
sampling (greedy argmax only: ``max_device_top_k = 1``; other rows sample on host from the logits),
speculative decoding, multimodal inputs, lane-DP / standard DP (one engine = one mesh).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from models.demos.glm53_flash.tt.vllm_config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DEVICE_PREFILL_CHUNK,
    DEFAULT_WAVE_ROWS,
    PREFIX_BUCKET_ENV,
    TILE,
    DecodeRowMap,
    LogicalBlockTable,
    PrefillPlan,
    PrefillPlanner,
    ServingGeometry,
    SlotTable,
    SPStep,
    WaveCall,
    WaveMember,
    WavePlanner,
    WaveStep,
    greedy_rows,
    max_tokens_all_users,
)

HF_ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM"
LATENT_DIM = 512  # kv_lora_rank: the latent page row (the only "KV" vLLM accounts for)


def _ttnn():
    """ttnn, imported on first device use only (the host test imports this module without it)."""
    import ttnn

    return ttnn


def _mesh_shape_from_env() -> tuple[int, int]:
    """The mesh the plugin will open: ``GLM53_MESH_SHAPE`` (rows,cols) if set, else the plugin's ``MESH_DEVICE``
    presets this model runs on (``BH-Galaxy`` / ``TG`` = (8, 4), ``P150x8`` = (1, 8)) or a ``(r, c)`` tuple."""
    spec = os.environ.get("GLM53_MESH_SHAPE")
    if spec:
        r, c = (int(v) for v in spec.split(","))
        return r, c
    md = os.environ.get("MESH_DEVICE", "")
    if md.startswith("("):
        r, c = (int(v) for v in md.strip("() ").split(","))
        return r, c
    return {"BH-Galaxy": (8, 4), "TG": (8, 4), "P150x8": (1, 8), "P150x4": (1, 4)}.get(md, (8, 4))


def _fabric_capability() -> dict | None:
    """``model_capabilities["fabric_config"]``: the KEYWORD ARGUMENTS of ``ttnn.set_fabric_config`` the mesh model
    needs (``config``: 2D torus on the 8x4, ``reliability_mode``: relaxed init, ``router_config``: the maximum-payload
    router config), from ``tt/mesh.py::mesh_device_params`` (whose ``fabric_router_config`` entry is what the root
    conftest passes as ``set_fabric_config``'s ``router_config``). The plugin merges this dict into its own
    ``set_fabric_config(**kwargs)`` call, so the keys must be the call's parameter names.
    Built only in a process that already imported ttnn (the vLLM plugin does before it resolves the model
    class), so importing this module on a host without ttnn stays side-effect free."""
    if "ttnn" not in sys.modules:
        return None
    try:
        from models.demos.glm53_flash.tt.mesh import mesh_device_params

        rows, cols = _mesh_shape_from_env()
        params = mesh_device_params(rows, cols)
    except Exception as exc:  # pragma: no cover - device library detail
        logger.warning(f"GLM53 vLLM: fabric capability not resolved ({exc}); the plugin's hardware default applies")
        return None
    out: dict = {}
    if "fabric_config" in params:
        out["config"] = params["fabric_config"]
    if "reliability_mode" in params:
        out["reliability_mode"] = params["reliability_mode"]
    if "fabric_router_config" in params:
        out["router_config"] = params["fabric_router_config"]
    return out or None


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


DEFAULT_MESH_CACHE = "/home/ttuser/ashai/glm53/_cache/mesh"  # the model tests' tensorbin weight cache


def mesh_cache_root() -> Path | None:
    """The tensorbin weight cache the mesh model loads / writes: ``GLM53_MESH_CACHE`` (``none`` disables it), else
    ``$TT_CACHE_PATH/glm53_mesh`` when the serving stack sets ``TT_CACHE_PATH`` (tt-inference-server's per-model cache
    volume, so a served GLM lands its weight cache next to every other model's), else the model tests' default."""
    spec = os.environ.get("GLM53_MESH_CACHE")
    if spec is None:
        tt_cache = os.environ.get("TT_CACHE_PATH")
        spec = str(Path(tt_cache) / "glm53_mesh") if tt_cache else DEFAULT_MESH_CACHE
    return None if spec.strip().lower() == "none" else Path(spec)


# ------------------------------------------------------------------------------------------- outputs


COMPILE_CHECK_ENV = "GLM53_VLLM_COMPILE_CHECK"  # "1" (default): seal the program cache after the last capture
MEM_LOG_ENV = "GLM53_VLLM_MEM_LOG"  # "1": log the DRAM allocator's state after every eager prefill device chunk


def _dram_line(device) -> str:
    """One line of the DRAM allocator's state: per bank allocated / free / largest free block (the served 1 M-profile
    prefill ran out of DRAM at ~160 K positions on 2026-10-27 with 507 MB free but fragmented: what is resident after
    the warm-up and what each chunk adds is read from these lines)."""
    ttnn = sys.modules.get("ttnn")  # never imported here: the host tests run this adapter without ttnn or a device
    if device is None or ttnn is None:
        return "DRAM: no device"
    mv = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    alloc = mv.total_bytes_allocated_per_bank
    free = getattr(mv, "total_bytes_free_per_bank", mv.total_bytes_per_bank - alloc)
    largest = getattr(mv, "largest_contiguous_bytes_free_per_bank", -(2**20))
    try:  # the decode (and spec) traces live in the trace region: its use decides GLM53_MODEL_TRACE_MB
        tv = ttnn.device.get_memory_view(device, ttnn.BufferType.TRACE)
        trace = f"; trace region {tv.total_bytes_allocated_per_bank / 2**20:.0f} of {tv.total_bytes_per_bank / 2**20:.0f} MiB per bank"
    except Exception:  # noqa: BLE001 - informational
        trace = ""
    return (
        f"DRAM per bank: allocated {alloc / 2**20:.0f} MiB, free {free / 2**20:.0f} MiB, "
        f"largest free block {largest / 2**20:.0f} MiB (x{mv.num_banks} banks){trace}"
    )


def _mesh_device_of(model):
    return getattr(getattr(model, "mesh", None), "device", None) or getattr(model, "mesh_device", None)


def _install_step_profiler() -> None:
    """``GLM53_VLLM_STEP_PROFILE=N`` (debug, default off): wrap the plugin runner's ``execute_model`` so the engine
    log shows, every 32 engine steps, the mean wall of a whole step next to the adapter's own decode wall (the
    difference is the runner + vLLM share), and after N steps a cProfile top-40 (cumulative) of those steps. The
    served step is ~5x the device step (2026-10-27 smoke: 372 ms ITL vs 65 ms replay); this knob attributes it."""
    spec = os.environ.get("GLM53_VLLM_STEP_PROFILE", "")
    if not spec:
        return
    import cProfile
    import io
    import pstats

    try:
        from vllm_tt_plugin import model_runner as mr
    except ImportError as exc:  # pragma: no cover - plugin-less host runs
        logger.warning(f"GLM53 vLLM: step profiler not installed ({exc})")
        return
    runner_cls = getattr(mr, "TTModelRunner", None)
    if runner_cls is None or getattr(runner_cls.execute_model, "_glm53_profiled", False):
        return
    # "N" profiles engine steps [0, N); "S:N" profiles steps [S, S+N) (e.g. 600:200 = inside a 32-user benchmark)
    first, _, count = spec.partition(":")
    start, n_profile = (int(first), max(1, int(count))) if count else (0, max(1, int(first)))
    state = {"calls": 0, "wall": 0.0, "prof": cProfile.Profile(), "done": False}
    orig = runner_cls.execute_model

    def execute_model(self, *args, **kwargs):
        t0 = time.perf_counter()
        profiling = not state["done"] and start <= state["calls"] < start + n_profile
        if profiling:
            state["prof"].enable()
        try:
            return orig(self, *args, **kwargs)
        finally:
            if profiling:
                state["prof"].disable()
            state["calls"] += 1
            state["wall"] += time.perf_counter() - t0
            if state["calls"] % 32 == 0:
                model = getattr(self, "model", None)
                steps = getattr(model, "_perf_steps", 0) or 0
                adapter_ms = 1000 * getattr(model, "_perf_s", 0.0) / steps if steps else float("nan")
                logger.info(
                    f"[glm53-vllm-step] engine step {1000 * state['wall'] / state['calls']:.1f} ms mean over "
                    f"{state['calls']} execute_model calls; adapter decode wall {adapter_ms:.1f} ms over {steps} steps"
                )
            if not state["done"] and state["calls"] >= start + n_profile:
                state["done"] = True
                buf = io.StringIO()
                pstats.Stats(state["prof"], stream=buf).strip_dirs().sort_stats("cumulative").print_stats(40)
                logger.info(
                    f"[glm53-vllm-step] cProfile of engine steps [{start}, {start + n_profile}):\n{buf.getvalue()}"
                )

    execute_model._glm53_profiled = True  # type: ignore[attr-defined]
    runner_cls.execute_model = execute_model
    logger.info(f"GLM53 vLLM: step profiler installed (cProfile over engine steps [{start}, {start + n_profile}))")


@dataclass
class DeviceOwnedCache:
    """What ``allocate_kv_cache`` returns: a marker that the latent cache lives inside the mesh model (allocated at
    build per user slot); the plugin passes it back as ``kv_cache`` and the adapter ignores it."""

    num_blocks: int
    block_size: int
    num_layers: int


@dataclass
class GLM53DecodeOutput:
    """One decode submission: the trace's output tensors (device or host) + the row map to un-permute them."""

    ids: Any  # [1,1,B,1] uint32 per chip (device) or [rows*B] int64 (host) or None
    logits: Any  # [1,1,B,vocab] bf16 per chip (device) or [rows*B, vocab] fp32 (host, rows read) or None
    rows: DecodeRowMap
    is_tokens: bool
    on_host: bool = False


class GLM53FlashForCausalLM:
    """GLM-5.3-Flash text generation on a Blackhole Galaxy mesh for the TT vLLM plugin."""

    decode_input_update_contract = 1

    model_capabilities = {
        "supports_prefix_caching": False,
        # Every decode reloads tokens / positions from the host (no resident token feedback in the mesh model's
        # decode step yet); the plugin then runs synchronous scheduling. Flip with the device-side feedback.
        "supports_async_decode": False,
        # The head's greedy argmax is the only device sampler: greedy rows (and top_k == 1) sample on device,
        # the plugin routes any other row to host sampling from the returned logits.
        "supports_sample_on_device": True,
        "max_device_top_k": 1,
        "supports_device_penalties": False,
        # A prompt split across engine steps resumes through FunctionalDecoder's chunked prefill (``start``).
        "supports_chunked_prefill": True,
        "fabric_config": _fabric_capability(),
    }

    # Narrowed to the one captured decode width by ``warmup_model_decode`` (fixed user slots: no smaller bucket).
    tt_supported_decode_batch_sizes: tuple[int, ...] = ()
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    def __init__(self, model, geometry: ServingGeometry, *, pad_token_id: int | None = None, **vllm_kwargs):
        # ``**vllm_kwargs`` (``vllm_config`` / ``prefix``) is never used: the plugin constructs the adapter through
        # ``initialize_vllm_model``. vLLM >= 0.30 inspects the registered class with ``is_vllm_model`` (an ``__init__``
        # that accepts ``vllm_config``, ``embed_input_ids``, ``forward(input_ids, positions)``) before it admits the
        # generate runner, so the signature has to advertise the keyword like the tt_transformers adapters do.
        self.model = model
        self.geometry = geometry
        self.slots = SlotTable(geometry)
        self.blocks = LogicalBlockTable(geometry)
        self.prefill_planner = PrefillPlanner(geometry, traced_available=self._has_prefill_trace)
        # per-row prefill ownership (GLM53_VLLM_ROWS_INDEPENDENT): the step's prefill rows become waves (one chunk per
        # mesh row, different prompts) and, while exact, sequence-parallel single-prompt steps (ROW_OWNERSHIP.md)
        self.waves = WavePlanner(geometry, self.slots, self.prefill_planner)
        if geometry.rows_independent and not callable(getattr(model, "prefill_wave", None)):
            raise NotImplementedError(
                "GLM53_VLLM_ROWS_INDEPENDENT=1 needs a mesh model with the row-local wave prefill "
                "(GLM53MeshModel.prefill_wave / prefill_traced_wave; _agents/vllm_glm53/ROW_OWNERSHIP.md section 3): "
                "the sequence-parallel prefill replicates a prompt into slot `user` of every row, so with it the "
                "engine serves users_per_row distinct requests only"
            )
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else geometry.pad_token_id)
        self.vocab_size = int(getattr(getattr(model, "cfg", None), "vocab_size", 0))
        self.trace_id = None
        self._trace_outputs = None  # (streams, logits, ids) of the captured decode step
        self._decode_inputs_ready = False
        self.already_warmed_up_prefill = False
        self._decode_warmup_complete = False
        self._pending_moves: dict[int, int] | None = None
        self._perf_steps = 0
        self._perf_s = 0.0
        self._perf_log_every = max(1, int(os.environ.get("GLM53_VLLM_PERF_LOG_EVERY", "32")))

    # ================================================================================= construction
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
    ):
        """Build the resident mesh model for ``max_batch_size`` concurrent requests (``vllm --max-num-seqs``) of up
        to ``max_seq_len`` positions (``--max-model-len``). ``tt_data_parallel`` must be 1: the mesh rows are the
        model's own data parallelism, invisible to vLLM."""
        if int(tt_data_parallel) != 1:
            raise ValueError("GLM53FlashForCausalLM runs one engine per mesh (tt_data_parallel must be 1)")
        from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
        from models.demos.glm53_flash.tt.config import GLM53FlashConfig
        from models.demos.glm53_flash.tt.memory_plan import BF16_BYTES, BFP4_BYTES, BFP8_BYTES, MemoryPlan
        from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

        ttnn = _ttnn()
        snapshot = cls._snapshot_dir(hf_config, default_snapshot_dir)
        cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
        shape = list(mesh_device.shape)
        rows, cols = int(shape[0]), int(shape[1])
        # the serving envelope's cache layout: user-partitioned over the columns, heads->users prefill (no scratch slot)
        os.environ.setdefault("GLM53_PARTITION_USERS", "1")
        os.environ.setdefault("GLM53_PREFILL_PARTITION", "heads2users")
        partitioned = _env_flag("GLM53_PARTITION_USERS", "1") and cols > 1
        scratch = os.environ.get("GLM53_PREFILL_PARTITION", "heads2users") != "heads2users"
        # per-row prefill ownership: max_num_seqs = rows x users_per_row, prefills as row-local waves (the mesh model
        # must provide prefill_wave; phase 1 shares the slot index across a wave: wave_affinity "user")
        rows_independent = _env_flag("GLM53_VLLM_ROWS_INDEPENDENT", "0")
        if rows_independent and int(max_batch_size) % rows:
            raise ValueError(
                f"max_num_seqs {max_batch_size} must be a multiple of the {rows} mesh rows under "
                "GLM53_VLLM_ROWS_INDEPENDENT (users_per_row = max_num_seqs / rows)"
            )
        users_per_row = int(max_batch_size) // rows if rows_independent else int(max_batch_size)
        unit = TILE * rows
        chunk = int(os.environ.get("GLM53_VLLM_PREFILL_CHUNK", str(DEFAULT_DEVICE_PREFILL_CHUNK)))
        # the sparse prefill's prefix bucket (TtLatentCache.user_rows_rm): max(chunk, max_seq_len / 8 to the chunk)
        os.environ.setdefault(PREFIX_BUCKET_ENV, str(max(chunk, -(-(int(max_seq_len) // 8) // chunk) * chunk)))
        chunk = max(unit, chunk // unit * unit)
        geometry = ServingGeometry(
            rows=rows,
            cols=cols,
            users_per_row=users_per_row,
            max_seq_len=int(max_seq_len),
            partitioned=partitioned,
            rows_independent=rows_independent,
            block_size=int(os.environ.get("GLM53_VLLM_BLOCK_SIZE", str(DEFAULT_BLOCK_SIZE))),
            device_prefill_chunk=chunk,
            pad_token_id=cfg.pad_token_id,
            wave_rows=int(os.environ.get("GLM53_VLLM_WAVE_ROWS", str(DEFAULT_WAVE_ROWS))),
            wave_affinity="row" if getattr(GLM53MeshModel, "wave_per_row_users", False) else "user",
            sp_fallback=_env_flag("GLM53_VLLM_SP_FALLBACK", "1"),
        )
        geometry.check_max_num_seqs(max_batch_size)
        dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}
        expert_dtype = dt[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
        expert_down_dtype = dt[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
        eb = {ttnn.bfloat8_b: BFP8_BYTES, ttnn.bfloat4_b: BFP4_BYTES, ttnn.bfloat16: BF16_BYTES}[expert_dtype]
        layers = cls._layer_ids(n_layers, cfg.num_hidden_layers)
        plan = MemoryPlan.estimate(
            cfg,
            rows,
            cols,
            partition=partitioned,
            expert_bytes=eb,
            scratch_slot=scratch,
            trace_mb=int(os.environ.get("GLM53_MODEL_TRACE_MB", "512")),
            layers=layers,  # a GLM53_MODEL_LAYERS subset (block tests) is planned at its own depth
        )
        geometry.check_memory_plan(plan)
        logger.info(
            f"GLM53 vLLM: {rows}x{cols} mesh, {users_per_row} users per row ({geometry.request_capacity} requests, "
            f"{geometry.physical_slots} slots), max_seq_len {max_seq_len}, partitioned={partitioned}, planner headroom "
            f"{plan.headroom(int(max_seq_len)) / 2**30:.1f} GiB vs {plan.bytes_for(users_per_row, int(max_seq_len)) / 2**30:.1f} GiB of caches"
            f" ({len(layers)} layers)"
        )
        cache_root = mesh_cache_root()
        t0 = time.perf_counter()
        model = GLM53MeshModel(
            mesh_device,
            cfg,
            layers,
            max_batch=users_per_row,
            max_seq_len=int(max_seq_len),
            snapshot=snapshot,
            weight_cache_path=cache_root,
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
        )
        logger.info(f"GLM53 vLLM: mesh model built in {time.perf_counter() - t0:.0f} s ({len(layers)} layers)")
        geometry = dataclasses.replace(geometry, traced_prefill_lengths=tuple(model.trace_prefill_lengths()))
        _install_step_profiler()
        return cls(model, geometry, pad_token_id=cfg.pad_token_id)

    @staticmethod
    def _snapshot_dir(hf_config, default_snapshot_dir) -> Path:
        name = getattr(hf_config, "_name_or_path", "") or ""
        p = Path(name)
        if name and p.is_dir() and (p / "config.json").exists():
            return p
        return default_snapshot_dir()

    @staticmethod
    def _layer_ids(n_layers: int | None, num_hidden_layers: int) -> list[int]:
        spec = os.environ.get("GLM53_MODEL_LAYERS")
        if spec:
            out: list[int] = []
            for part in spec.split(","):
                a, _, b = part.partition("-")
                out.extend(range(int(a), int(b or a) + 1))
            return out
        return list(range(int(n_layers) if n_layers else num_hidden_layers))

    @property
    def cache_path(self):
        root = mesh_cache_root()
        return "" if root is None else str(root)

    # ================================================================================= KV accounting
    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """Every slot owns ``max_model_len`` latent pages on device, so the logical pool is ``max_num_seqs x
        max_model_len`` (the plugin adds one block per user on top): vLLM never refuses a request on blocks while a
        slot is free, and ``max_num_seqs`` is the admission limit."""
        if max_model_len is None or max_num_seqs is None:
            raise ValueError("get_max_tokens_all_users needs max_model_len and max_num_seqs")
        return max_tokens_all_users(max_num_seqs, max_model_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One logical full-attention group of latent pages: ``[block_size, 1, 512]`` bf16 per page. Only the page
        size and block count matter to vLLM (admission accounting); the device cache is model-owned."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cache_config = vllm_config.cache_config
        if int(cache_config.block_size) % TILE:
            raise ValueError(
                f"--block-size {cache_config.block_size} must be a multiple of {TILE}: the latent page is 32 rows"
            )
        return {
            "glm53_latent": FullAttentionSpec(
                block_size=int(cache_config.block_size),
                num_kv_heads=1,
                head_size=LATENT_DIM,
                dtype=torch.bfloat16,
            )
        }

    def allocate_kv_cache(self, shape, dtype, num_layers: int):
        """No device allocation: ``TtLatentCache`` (per DSA layer, per user slot) was built with the model."""
        num_blocks, _, block_size, _ = (int(v) for v in shape)
        if block_size != self.geometry.block_size:
            raise ValueError(f"vLLM block size {block_size} != the adapter's {self.geometry.block_size}")
        return DeviceOwnedCache(num_blocks=num_blocks, block_size=block_size, num_layers=int(num_layers))

    # ================================================================================= warm-up
    def warmup_model_prefill(self, kv_cache=None, enable_trace: bool = True, can_sample_on_device: bool = False, **kw):
        """Eager phase (``enable_trace=False``): the persistent decode inputs + one eager decode step (compiles,
        must precede every capture) and the eager prefill ladder (``GLM53_VLLM_WARMUP_PREFILL``: every listed prompt
        length walked in device chunks into slot 0, so the prefill programs of the short prompt AND of the long-prompt
        chunk shapes are compiled at boot rather than on the first request of that shape). Traced phase: capture the
        per-length prefill traces (every supported length, ``GLM53MeshModel.capture_prefill``; before the decode
        trace: a persistent tensor allocated after a capture may alias that trace's transient buffers)."""
        U = self.geometry.physical_slots
        if not enable_trace:
            self._ensure_decode_inputs(U)
            if not self.already_warmed_up_prefill:
                lengths = self._warmup_prefill_lengths()
                slots = self._warmup_slots(self.geometry.users_per_row)  # the model's per-row user index
                for n in lengths:
                    dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
                    plan = PrefillPlan("eager", PrefillPlanner.walk(0, n, self.geometry.device_prefill_chunk))
                    t0 = time.perf_counter()
                    into = slots if n == lengths[0] else slots[:1]
                    for u in into:
                        self._prefill_one(dummy, u, plan, want_logits=False)
                    self._sync()
                    logger.info(
                        f"GLM53 vLLM: eager prefill warm-up ({n} tokens in {len(plan.chunks)} device chunk(s) into "
                        f"{len(into)} slot(s)) in {time.perf_counter() - t0:.1f} s"
                    )
                self._warmup_prefix_starts(lengths)
                logger.info(f"GLM53 vLLM: after the prefill warm-up: {_dram_line(_mesh_device_of(self.model))}")
                n = lengths[0]
                dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
                if self.geometry.rows_independent:  # the wave path compiles its own programs: one wave on every row
                    g = self.geometry
                    members = tuple(WaveMember(logical=r, row=r, user=0, start=0, end=n) for r in range(g.rows))
                    call = WaveCall(rows=g.rows, t_pad=-(-n // TILE) * TILE, members=members, user=0)
                    for m in members:
                        call.ids[m.row] = dummy.clone()
                    t0 = time.perf_counter()
                    self.model.prefill_wave(call)
                    self._sync()
                    logger.info(
                        f"GLM53 vLLM: eager wave warm-up ({n} tokens x {g.rows} rows) in {time.perf_counter() - t0:.1f} s"
                    )
                self.already_warmed_up_prefill = True
            return
        if self.trace_id is not None:
            raise RuntimeError("prefill traces must be captured before the decode trace (allocation order)")
        self._ensure_decode_inputs(U)
        t0 = time.perf_counter()
        captured = self.model.capture_prefill()
        logger.info(f"GLM53 vLLM: prefill traces {captured} captured in {time.perf_counter() - t0:.0f} s")
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace: bool = True,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = False,
        **kw,
    ):
        """Eager phase: nothing beyond ``warmup_model_prefill``'s eager decode step. Traced phase: capture the one
        decode step for every user slot of the mesh (``GLM53MeshModel.capture``), then zero the KDA states the
        warm-up + capture advanced; publish that one width as the only supported decode batch."""
        if max_batch_size is not None:
            self.geometry.check_max_num_seqs(max_batch_size)
        U = self.geometry.physical_slots
        self._ensure_decode_inputs(U)
        if not enable_trace:
            return
        if self.trace_id is None:
            t0 = time.perf_counter()
            self.trace_id, self._trace_outputs = self.model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
            self.model.reset_kda_states()
            self._sync()
            logger.info(f"GLM53 vLLM: decode trace captured in {time.perf_counter() - t0:.1f} s ({U} slots)")
        self.tt_supported_decode_batch_sizes = (self.geometry.request_capacity,)
        self._decode_warmup_complete = True
        self._seal_program_cache()

    def _seal_program_cache(self) -> None:
        """After the last capture no program may be built: a request-time compile allocates its kernel binaries
        on the device under live traces (the 2026-10-27/28 served hangs, every one at the first op whose kernel
        variant the warm-up had not built). With ``GLM53_VLLM_COMPILE_CHECK=1`` (default) the device forbids
        program-cache misses from here on, so such a request fails with the op's name ("program cache miss
        occurred, but cache misses are forbidden") instead of hanging the mesh."""
        if os.environ.get(COMPILE_CHECK_ENV, "1") != "1":
            return
        dev = _mesh_device_of(self.model)
        dev.set_program_cache_misses_allowed(False)
        logger.info(f"GLM53 vLLM: program cache sealed after the captures ({dev.num_program_cache_entries()} programs)")

    def _warmup_prefix_starts(self, lengths: list[int]) -> None:
        """Continued chunks at every prefix bucket before the captures: the sparse prefill reads the user's prefix
        rows [0, start + T) rounded up to GLM53_PREFIX_BUCKET (``TtLatentCache.user_rows_rm``), so a continued chunk's
        kv shape depends on the prefix bucket, not on its exact start; one chunk of every lone bucket length (full and
        ragged, ``lengths`` up to the chunk) at start j x bucket for every j builds those programs, which a long
        prompt's chunk k >= 2 and a prefix-cache hit at a chunk-aligned snapshot then reuse (the served APC hit at
        1536 compiled its prefix read at request time under the sealed cache, 2026-10-28 8x4). The chunk cursor of
        slot 0 is rewound around every call; its cache rows are placeholders the next prompt overwrites.
        ``GLM53_VLLM_WARMUP_PREFIX_STARTS=0`` skips the walk."""
        if os.environ.get("GLM53_VLLM_WARMUP_PREFIX_STARTS", "1") != "1":
            return
        g = self.geometry
        chunk = g.device_prefill_chunk
        bucket = int(os.environ.get(PREFIX_BUCKET_ENV, "0")) or chunk
        if bucket % chunk:
            raise ValueError(f"{PREFIX_BUCKET_ENV}={bucket} is not a multiple of the device chunk {chunk}")
        tails = sorted({int(n) for n in lengths if n <= chunk})
        # the prefix read of a continued chunk (``TtLatentCache.user_rows`` / ``user_rows_rm``) is a slice whose
        # program keys on the slot's first block AND the read length: one program per (user, prefix bucket). The
        # length-dependent programs that do not depend on the user (the chunk's own attention over T rows) come
        # from slot 0's walk over every tail; the other slots walk every bucket with the shortest tail only.
        B = self.geometry.users_per_row
        t0 = time.perf_counter()
        n_calls = 0
        for j in range(1, g.max_seq_len // bucket):
            start = j * bucket
            for u in range(B):
                for n in tails if u == 0 else tails[:1]:
                    if start + n > g.max_seq_len:
                        continue
                    dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
                    self.model.rewind_user(u, start)
                    out = self.model.prefill(dummy, user=u, start=start)
                    self._free(out[0])
                    self._free(out[1])
                    n_calls += 1
        for u in range(B):
            self.model.rewind_user(u, 0)
        self._sync()
        logger.info(
            f"GLM53 vLLM: continued-chunk warm-up ({n_calls} chunks at {max(0, g.max_seq_len // bucket - 1)} prefix "
            f"bucket start(s) of {bucket} into {B} slot(s), tails {tails}) in {time.perf_counter() - t0:.1f} s"
        )

    def _warmup_slots(self, B: int) -> list[int]:
        """``GLM53_VLLM_WARMUP_SLOTS``: the per-row user slots the shortest warm-up prompt is prefilled into
        (default ``all``: every one of the ``B`` users a row holds — the mesh model's ``user`` index, replicated
        over the rows — so a program whose variant depends on that index, the eager per-user state writes, is
        built at boot, never at the first request into that slot; an int keeps the first that many)."""
        raw = os.environ.get("GLM53_VLLM_WARMUP_SLOTS", "all").strip().lower()
        n = B if raw in ("", "all") else max(1, min(B, int(raw)))
        return list(range(n))

    def _warmup_prefill_lengths(self) -> list[int]:
        """The prompt lengths the eager warm-up walks. ``GLM53_VLLM_WARMUP_PREFILL`` = ``auto`` (default): every
        prefill bucket up to the device chunk, full and ragged (``b - 1``), as a lone prompt (its first-chunk
        programs) and as the continuation of a full device chunk (``chunk + bucket``: the same programs with history, the conv / KDA / index
        maintenance variants of a second chunk), so no chunk shape of any prompt compiles after the captures (a
        1590-token prompt = a 1024 chunk + a 566-token continuation padded to the 1024 bucket: the 2026-10-28
        served APC seal hit). A comma-separated list adds its lengths to that set (the ``128,32768`` catalog form). Lengths are clamped to ``max_seq_len``; duplicates and empty entries dropped; the shortest comes
        first (it is the one prefilled into every slot)."""
        raw = os.environ.get("GLM53_VLLM_WARMUP_PREFILL", "auto").strip().lower()
        chunk = self.geometry.device_prefill_chunk
        try:  # the served branches' bucket set (GLM53_PREFILL_BUCKETS); the base pads to the traced lengths
            from models.demos.glm53_flash.tt.vllm_config import prefill_buckets

            buckets = list(prefill_buckets())
        except ImportError:
            buckets = list(self.geometry.traced_prefill_lengths)
        buckets = [b for b in buckets if b <= chunk]
        # every bucket full AND ragged (b - 1: the rows-past-the-prompt paths, e.g. the router's pad mask), lone
        # and as the continuation of a full chunk; an explicit list ADDS its lengths (it cannot drop this coverage:
        # the served APC test's "128,2048" walked 2048 full only and its 1590-token prompt compiled the ragged
        # pad-mask matmul at that bucket under the sealed cache)
        cand = {n for b in buckets for n in (b, max(1, b - 1), chunk + b, chunk + max(1, b - 1))} | {
            chunk,
            max(1, chunk - 1),
        }
        if raw not in ("", "auto"):
            cand |= {int(part) for part in raw.split(",") if part.strip()}
        cand = sorted(cand)
        out: list[int] = []
        for n in cand:
            n = min(int(n), self.geometry.max_seq_len)
            if n >= 1 and n not in out:
                out.append(n)
        out.sort()
        return out or [min(128, self.geometry.max_seq_len)]

    def _ensure_decode_inputs(self, U: int) -> None:
        if self._decode_inputs_ready:
            return
        t0 = time.perf_counter()
        self.model.warmup_decode(torch.zeros(U, dtype=torch.long), [0] * U)
        self.model.reset_kda_states()
        logger.info(f"GLM53 vLLM: eager decode warm-up in {time.perf_counter() - t0:.1f} s")
        self._decode_inputs_ready = True

    # ================================================================================= prefill
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """One vLLM prefill step: row ``i`` of ``tokens`` ``[B, L]`` is a request whose chunk ``[start_pos[i],
        prompt_lens[i])`` is prefilled into the physical slot of logical slot ``empty_slots[i]`` (allocated here when
        the chunk starts at 0). Prompts go one at a time through the mesh (the sequence-parallel prefill uses every
        chip). Returns ``logits [B, 1, vocab]`` fp32 (host sampling) or ``(tokens [B, 1], None)`` (device greedy)."""
        if kwargs.get("pixel_values") is not None or kwargs.get("image_grid_thw") is not None:
            raise NotImplementedError("multimodal prefill is not wired through the vLLM adapter yet")
        B = int(tokens.shape[0])
        if prompt_lens is None:
            raise ValueError("prefill_forward needs prompt_lens (the end position of every row's chunk)")
        starts = [0] * B if start_pos is None else [int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()]
        ends = [int(v) for v in prompt_lens]
        if empty_slots is None:
            empty_slots = list(range(B))
        if len(ends) != B or len(starts) < B or len(empty_slots) < B:
            raise ValueError(
                f"prefill rows {B} vs prompt_lens {len(ends)} start_pos {len(starts)} slots {len(empty_slots)}"
            )
        if sampling_params is not None:
            greedy_rows(sampling_params, range(B))
        # The plugin does not say which rows are intermediate chunks (it discards their outputs itself), so every
        # chunk's last token is scored; a slot stays in the ``prefill`` phase until its first decode consumes the
        # token at ``prefilled_to`` (a later chunk must start exactly there).
        wants_tokens = sampling_params is not None
        out_tokens = torch.zeros(B, 1, dtype=torch.int32)
        out_logits = None if wants_tokens else torch.zeros(B, 1, self.vocab_size, dtype=torch.float32)
        # placement / contiguity of every row first (a refused row leaves no half-run step behind)
        rows_of: dict[int, int] = {}  # logical slot -> row of the call
        items = []  # (i, logical, state, start, end, ids)
        for i in range(B):
            logical = int(empty_slots[i])
            start, end = starts[i], ends[i]
            if page_table is not None:
                self.blocks.check_row(page_table[i], end)
            self.prefill_planner.check_range(start, end)
            if start == 0:
                state = self.slots.allocate(logical)
            else:
                state = self.slots.physical(logical)
                if state is None or state.phase != "prefill" or state.prefilled_to != start:
                    raise RuntimeError(
                        f"slot {logical}: chunk [{start}, {end}) is not contiguous with the prefilled prefix "
                        f"({None if state is None else state.prefilled_to}); chunks must follow each other"
                    )
            ids = tokens[i, start:end].reshape(-1).to(torch.long)
            if int(ids.numel()) != end - start:
                raise ValueError(f"row {i}: tokens hold {ids.numel()} of the {end - start} chunk tokens")
            rows_of[logical] = i
            items.append((i, logical, state, start, end, ids))

        def record(i, state, end, first, last_logits):
            state.prefilled_to = end
            state.position = end
            state.phase = "prefill"
            if wants_tokens:
                out_tokens[i, 0] = int(first)
            else:
                out_logits[i, 0] = last_logits

        if not self.geometry.rows_independent:
            for i, logical, state, start, end, ids in items:
                plan = self.prefill_planner.plan(start, end, enable_trace=enable_trace)
                t0 = time.perf_counter()
                first, last_logits = self._prefill_one(ids, state.user, plan, want_logits=not wants_tokens)
                record(i, state, end, first, last_logits)
                logger.info(
                    f"GLM53 vLLM: prefill slot {logical} -> user {state.user} (row {state.row}) [{start}, {end}) via "
                    f"{plan.kind} ({len(plan.chunks)} chunk(s)) in {(time.perf_counter() - t0) * 1000:.0f} ms, "
                    f"chunk-end token {first}"
                )
        else:
            by_logical = {logical: (i, state, start, end, ids) for i, logical, state, start, end, ids in items}
            steps = self.waves.plan(
                [(logical, start, end) for _, logical, _, start, end, _ in items], enable_trace=enable_trace
            )
            for step in steps:
                t0 = time.perf_counter()
                if isinstance(step, SPStep):
                    i, state, start, end, ids = by_logical[step.logical]
                    first, last_logits = self._prefill_one(ids, state.user, step.plan, want_logits=not wants_tokens)
                    record(i, state, end, first, last_logits)
                    logger.info(
                        f"GLM53 vLLM: SP prefill slot {step.logical} -> user {state.user} (row {state.row}) [{start}, "
                        f"{end}) via {step.plan.kind} ({len(step.plan.chunks)} chunk(s)) in "
                        f"{(time.perf_counter() - t0) * 1000:.0f} ms, chunk-end token {first}"
                    )
                    continue
                assert isinstance(step, WaveStep), step
                call = step.call
                for m in call.members:
                    _, _, start, _, ids = by_logical[m.logical]
                    call.ids[m.row] = ids[m.start - start : m.end - start]
                results = self._prefill_wave(call, want_logits=not wants_tokens)
                for m in call.members:
                    i, state, _, end, _ = by_logical[m.logical]
                    state.sp_only = False  # the prefix of this request now exists on its own row only
                    first, last_logits = results[m.row]
                    if m.end == end:  # the member's last chunk of this step
                        record(i, state, m.end, first, last_logits)
                    else:
                        state.prefilled_to = m.end
                        state.position = m.end
                logger.info(
                    f"GLM53 vLLM: wave of {len(call.members)} ({'traced' if call.traced_length else 'eager'}, "
                    f"t_pad {call.t_pad}, slot index {call.user}) rows "
                    f"{[(m.row, m.start, m.end) for m in call.members]} in {(time.perf_counter() - t0) * 1000:.0f} ms"
                )
        if wants_tokens:
            return out_tokens, None
        return out_logits

    def _prefill_wave(self, call: WaveCall, *, want_logits: bool) -> dict:
        """One wave through the mesh model: the traced per-row replay when the planner found a captured length for
        it (every member at start 0), else the eager row-local wave. Returns ``{row: (first token, logits [vocab]
        fp32 or None)}``; a model that hands back host tensors or ints is accepted as is (the mesh model does its
        own per-row tile read). TODO(device): the first wave on the 8x4 confirms the per-row read."""
        if call.traced_length is not None and callable(getattr(self.model, "prefill_traced_wave", None)):
            res = self.model.prefill_traced_wave(call)
        else:
            res = self.model.prefill_wave(call)
        out = {}
        for m in call.members:
            first, logits = res[m.row]
            if logits is not None and not isinstance(logits, torch.Tensor):
                logits = self._to_torch(logits).float().reshape(-1)
            out[m.row] = (int(first), logits if want_logits else None)
        return out

    def _prefill_one(self, ids: torch.Tensor, user: int, plan, *, want_logits: bool):
        """Run ``plan`` for one prompt chunk into physical user ``user``; returns ``(first greedy token, last-token
        logits [vocab] fp32 or None)``. TODO(device): confirm the traced path's tile read and the eager owner-row
        read on the 8x4 (both follow tests/test_mesh_model.py / demo/text_demo.py)."""
        if plan.kind == "traced":
            res = self.model.prefill_traced(ids, user=user)
            first = int(self.model.first_token(res))
            logits = self._traced_last_logits(res) if want_logits else None
            return first, logits
        out = None
        mem_log = os.environ.get(MEM_LOG_ENV, "0") == "1"
        for s0, e in plan.chunks:
            if out is not None:
                self._free(out[0])
                self._free(out[1])
            out = self.model.prefill(ids[s0 - plan.start : e - plan.start], user=user, start=s0)
            if mem_log:
                logger.info(
                    f"GLM53 vLLM mem: user {user} chunk [{s0}, {e}) done: {_dram_line(_mesh_device_of(self.model))}"
                )
        streams, logits_last, owner_row = out
        first = int(self.model.first_token(logits_last, owner_row))
        logits = self._eager_last_logits(logits_last, owner_row) if want_logits else None
        self._free(streams)
        self._free(logits_last)
        return first, logits

    def _traced_last_logits(self, res) -> torch.Tensor:
        """``[vocab]`` fp32 of the prompt's last token out of the traced prefill's logits tile."""
        shard = self.model.mesh.device_tensors(res.logits)[res.owner_row * self.geometry.cols]
        t = self._to_torch(shard).float().reshape(-1, self.vocab_size)
        return t[res.row]

    def _eager_last_logits(self, logits_last, owner_row: int) -> torch.Tensor:
        shard = self.model.mesh.device_tensors(logits_last)[owner_row * self.geometry.cols]
        return self._to_torch(shard).float().reshape(-1, self.vocab_size)[-1]

    def _has_prefill_trace(self, T: int) -> bool:
        has = getattr(self.model, "has_prefill_trace", None)
        return bool(has(T)) if has is not None else False

    # ================================================================================= decode
    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        slot_remap=None,
        *,
        reload_inputs: bool = True,
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
        **kwargs,
    ):
        """One decode step for the whole mesh. ``tokens`` ``[B_pad, 1]`` / ``start_pos`` ``[B_pad]`` (``-1`` on
        padding rows): row ``i`` is logical slot ``i`` once ``slot_remap`` (row ``i`` reads slot ``remap[i]``) is
        applied to the indirection table. Version-1 reload contract: this adapter restages tokens and positions on
        every step, so ``reload_inputs`` must be true (``supports_async_decode`` is false, which is what the plugin
        sends); the sampling commands are accepted and need no state (greedy argmax has none)."""
        if "reset_batch" in kwargs:
            raise TypeError("legacy reset_batch keyword: this adapter declares decode_input_update_contract = 1")
        if kwargs.get("spec_mode") is not None or kwargs.get("num_valid_drafts") is not None:
            raise NotImplementedError("speculative decoding is not supported by the GLM53 adapter")
        if not reload_inputs:
            raise ValueError(
                "reload_inputs=False: the GLM53 adapter has no resident token / position state yet "
                "(supports_async_decode is False), every decode must reload its inputs"
            )
        if reload_page_table:
            raise ValueError("reload_page_table with reload_inputs is not a legal command combination")
        if self.trace_id is None and enable_trace:
            raise RuntimeError("decode trace not captured: warmup_model_decode(enable_trace=True) must run first")
        tokens = torch.as_tensor(tokens).reshape(-1)
        positions = torch.as_tensor(start_pos).reshape(-1)
        wire = int(tokens.shape[0])
        if int(positions.shape[0]) != wire:
            raise ValueError(f"tokens {wire} rows vs start_pos {int(positions.shape[0])}")
        if wire > self.slots.n:
            raise ValueError(f"{wire} decode rows exceed the {self.slots.n} logical slots")
        # the remap, applied exactly once; rolled back if the device refuses the step so the plugin's un-advanced
        # ownership map and this table still agree
        snapshot = list(self.slots._phys)
        applied_moves: dict[int, int] = {}
        if slot_remap is not None:
            remap = [int(v) for v in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            applied_moves = self.slots.apply_remap(remap)
        try:
            live = [i for i in range(wire) if int(positions[i]) >= 0]
            is_tokens = sampling_params is not None
            if is_tokens:
                greedy_rows(sampling_params, live)
            feed_tokens: dict[int, int] = {}
            feed_positions: dict[int, int] = {}
            rows = DecodeRowMap(wire_rows=wire, read_index=[-1] * wire)
            for i in live:
                state = self.slots.physical(i)
                if state is None:
                    raise RuntimeError(f"decode row {i} (slot {i}) was never prefilled: map {self.slots._phys}")
                p = int(positions[i])
                if state.phase == "prefill":
                    # the first decode after the prompt: it consumes the token at the prefilled end
                    if state.prefilled_to != p:
                        raise RuntimeError(
                            f"decode row {i}: slot {i} prefilled to {state.prefilled_to} but decodes at position {p} "
                            "(a prompt chunk is missing or out of order)"
                        )
                elif state.position != p:
                    logger.warning(
                        f"GLM53 vLLM: decode row {i} at position {p}, the adapter expected {state.position}; the host "
                        "position is authoritative (reload_inputs=True)"
                    )
                if page_table is not None:
                    self.blocks.check_row(page_table[i], p + 1)
                feed_tokens[i] = int(tokens[i])
                feed_positions[i] = p
                rows.read_index[i] = self.slots.read_index(i)
            # mid-prefill = a prefill-phase slot this decode does not advance (its next chunk is still to come)
            busy = [s for s in self.slots.in_prefill() if s not in feed_tokens]
            if busy and not getattr(self.model, "supports_slot_mask", False):
                # the decode step updates EVERY slot's KDA state in place; a slot between two prompt chunks would be
                # corrupted. The scheduler never schedules a decode while a continuation is pending when the decode
                # interleave is off (``decode_interleave_enabled: false``); fail loudly rather than corrupt.
                raise RuntimeError(
                    f"decode step while slots {busy} are mid-prefill: run with decode_interleave_enabled=false until "
                    "the mesh model's decode step takes an active-slot mask"
                )
            ids_feed, pos_feed = self.slots.device_feed(feed_tokens, feed_positions)
            t0 = time.perf_counter()
            # one call per step: with GLM53_DECODE_PACKED_INPUTS=1 the mesh model uploads ONE packed tensor (ids +
            # positions + sparse metadata, tt/packed_inputs.py) instead of the per-field refreshes
            load = getattr(self.model, "load_decode_inputs", None)
            if load is not None:
                load(torch.tensor(ids_feed, dtype=torch.long), pos_feed)
            else:
                self.model.load_ids(torch.tensor(ids_feed, dtype=torch.long))
                self.model.set_positions(pos_feed)
            set_active = getattr(self.model, "set_active_slots", None)
            if set_active is not None:  # future device hook: freeze idle / mid-prefill slots inside the traced step
                active = [False] * self.geometry.physical_slots
                for i in live:
                    for flat in self.slots.flat_indices(i):
                        active[flat] = True
                set_active(active)
            if enable_trace:
                self.model.replay(self.trace_id)
                _, logits_dev, ids_dev = self._trace_outputs
                transient = False
            else:
                _, logits_dev, ids_dev = self.model.decode_step(self.model.ids_dev, self.model.positions)
                transient = True
            for i in live:
                state = self.slots.physical(i)
                state.position = feed_positions[i] + 1
                state.phase = "decode"
        except Exception:
            self.slots._phys = snapshot
            raise
        self._pending_moves = applied_moves or None
        out = GLM53DecodeOutput(
            ids=ids_dev if is_tokens else None, logits=None if is_tokens else logits_dev, rows=rows, is_tokens=is_tokens
        )
        if read_from_device:
            out = self._read(out)
            if transient:
                self._free(logits_dev)
                self._free(ids_dev)
            self._perf(t0, len(live))
            return self.process_decode_output_host(out, is_tokens=is_tokens)
        if transient:
            out = self._read(out)
            self._free(logits_dev)
            self._free(ids_dev)
        self._perf(t0, len(live))
        return out

    def read_decode_output(self, tt_out, async_read: bool = False, sample_rows=None):
        """Host read of a submission (blocking today; the event list is empty). ``sample_rows`` (selective readback)
        is not declared and therefore refused."""
        if sample_rows is not None:
            raise NotImplementedError("selective host readback is not declared by the GLM53 adapter")
        if not isinstance(tt_out, GLM53DecodeOutput):
            raise TypeError(f"read_decode_output expects a GLM53DecodeOutput, got {type(tt_out).__name__}")
        out = self._read(tt_out)
        return (out, []) if async_read else out

    def process_decode_output_host(self, tt_out, is_tokens: bool = False, sample_rows=None):
        """``[B_pad, 1]`` int32 tokens (device greedy) or ``([B_pad, 1, vocab]`` fp32 logits, None)`` in the plugin's
        row order; padding rows are zeros."""
        if sample_rows is not None:
            raise NotImplementedError("compact host logits are not declared by the GLM53 adapter")
        if isinstance(tt_out, torch.Tensor) or (
            isinstance(tt_out, tuple) and all(isinstance(t, torch.Tensor) or t is None for t in tt_out)
        ):
            return tt_out
        if not isinstance(tt_out, GLM53DecodeOutput):
            raise TypeError(f"process_decode_output_host expects a GLM53DecodeOutput, got {type(tt_out).__name__}")
        out = tt_out if tt_out.on_host else self._read(tt_out)
        rows = out.rows
        if out.is_tokens:
            toks = torch.zeros(rows.wire_rows, 1, dtype=torch.int32)
            for i in rows.live_rows():
                toks[i, 0] = int(out.ids[rows.read_index[i]])
            return toks
        logits = torch.zeros(rows.wire_rows, 1, self.vocab_size, dtype=torch.float32)
        for i in rows.live_rows():
            logits[i, 0] = out.logits[rows.read_index[i]]
        return logits, None

    def _read(self, out: GLM53DecodeOutput) -> GLM53DecodeOutput:
        """Device -> host: the greedy ids of every slot (tiny) or the logits of the rows that hold live slots (one
        ``[B, vocab]`` bf16 shard per mesh row, the first column's replica). TODO(device): the logits shard read is
        the first place to measure (B x vocab x 2 B per row per step)."""
        if out.on_host:
            return out
        if out.is_tokens:
            host_ids = self.model.ids_to_host(out.ids).reshape(-1)
            return GLM53DecodeOutput(ids=host_ids, logits=None, rows=out.rows, is_tokens=True, on_host=True)
        B = self.geometry.users_per_row
        host = torch.zeros(self.geometry.physical_slots, self.vocab_size, dtype=torch.float32)
        shards = self.model.mesh.device_tensors(out.logits)
        for r in sorted({idx // B for idx in out.rows.read_index if idx >= 0}):
            t = self._to_torch(shards[r * self.geometry.cols]).float().reshape(-1, self.vocab_size)
            host[r * B : (r + 1) * B] = t[:B]
        return GLM53DecodeOutput(ids=None, logits=host, rows=out.rows, is_tokens=False, on_host=True)

    # ================================================================================= slot ownership
    def release_request(self, slot: int) -> None:
        """The request in logical ``slot`` finished: free its physical slot (the next prompt's ``start == 0``
        prefill resets the KDA state and refills the cache; idle slots are parked at position 0)."""
        try:
            self.slots.release(int(slot))
        except ValueError:
            logger.warning(f"GLM53 vLLM: release_request({slot}) outside the slot range")

    def note_state_slots_moved(self, moves) -> None:
        """The plugin's ``old -> new`` notification of the gather ``decode_forward`` already applied through
        ``slot_remap``: consumed as a consistency check. A plugin that never passes ``slot_remap`` to
        ``decode_forward`` reaches here with moves the table has not seen; they are applied then (exactly once)."""
        moves = {int(k): int(v) for k, v in dict(moves or {}).items()}
        if not moves:
            return
        pending = self._pending_moves or {}
        if pending == moves:
            self._pending_moves = None
            return
        if pending:
            raise RuntimeError(f"note_state_slots_moved {moves} disagrees with the applied remap {pending}")
        self.slots.apply_moves(moves)

    def release_persistent_capture(self) -> None:
        """Drop the captured traces (plugin teardown). A model without a device (the host test's fake) has no trace
        to release, so ttnn is only touched when the process already loaded it for a real mesh."""
        if self.trace_id is not None:
            if "ttnn" in sys.modules and getattr(self.model, "device", None) is not None:
                try:
                    sys.modules["ttnn"].release_trace(self.model.device, self.trace_id)
                except Exception as exc:  # pragma: no cover - device library detail
                    logger.warning(f"GLM53 vLLM: release_trace failed: {exc}")
            self.trace_id = None
            self._trace_outputs = None
        rel = getattr(self.model, "release_prefill_traces", None)
        if rel is not None:
            rel()
        self._decode_warmup_complete = False
        self.tt_supported_decode_batch_sizes = ()

    # ================================================================================= helpers
    def _perf(self, t0: float, live: int) -> None:
        self._perf_steps += 1
        self._perf_s += time.perf_counter() - t0
        if self._perf_steps % self._perf_log_every == 0 and self._perf_s > 0:
            ms = 1000 * self._perf_s / self._perf_steps
            logger.info(
                f"[glm53-vllm-perf] decode step {ms:.1f} ms = {1000 / ms:.2f} tok/s per user, "
                f"{live * 1000 / ms:.1f} tok/s aggregate over {live} live users ({self._perf_steps} steps, host wall incl. readback)"
            )

    @staticmethod
    def _to_torch(shard) -> torch.Tensor:
        if isinstance(shard, torch.Tensor):
            return shard
        return _ttnn().to_torch(shard)

    @staticmethod
    def _free(t) -> None:
        if t is None or isinstance(t, torch.Tensor):
            return
        if "ttnn" in sys.modules:
            ttnn = sys.modules["ttnn"]
            if isinstance(t, ttnn.Tensor) and t.is_allocated():
                ttnn.deallocate(t)

    def _sync(self) -> None:
        if "ttnn" in sys.modules and getattr(self.model, "device", None) is not None:
            try:
                sys.modules["ttnn"].synchronize_device(self.model.device)
            except Exception:  # pragma: no cover - fake models have no device
                pass

    # vLLM nn.Module protocol shims (the TT runner never calls these)
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models do not expose embed_input_ids")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models run through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("TT models return logits from decode_forward")
