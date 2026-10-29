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

Slot-level automatic prefix caching (``GLM53_VLLM_PREFIX_CACHING=1``, design ``doc/serving/prefix_caching.md``):
``supports_prefix_caching`` lets vLLM admit a request with ``num_computed_tokens = N > 0`` (its block hashes,
mm-aware, matched cached blocks); the adapter serves it only from a retained physical slot whose resident content
equals the request's first ``N`` tokens (``SlotPrefixCache``) and which holds a KDA snapshot at or below ``N``
(``tt/kda_snapshot.py``: taken at every 32-aligned decode crossing and at eager prefill chunk ends), restored
before the chunk ``[snapshot, end)`` is prefilled there; otherwise the hit is downgraded to a prefill from 0 (vLLM's
bookkeeping stays valid: the device ends in the state it expects). Knobs: ``GLM53_VLLM_APC_SNAPSHOTS`` (entries
per slot; 0 = what the planner fits), ``GLM53_VLLM_APC_SNAPSHOT_STORE`` (``device`` ring / ``host`` copies),
``GLM53_VLLM_APC_TAIL_SPLIT`` (eager prefills end on the prompt's last 32-boundary first).

Multimodal serving (image + video prompts; ``GLM53FlashForConditionalGeneration`` at the bottom of this module,
selected through ``TT_MODEL_CLASS_OVERRIDES``): the class adds vLLM's ``SupportsMultiModal`` and registers
upstream's ``Glm5NextMultiModalProcessor`` (``vllm.models.glm5next``), so ``vllm serve`` accepts ``image_url`` /
``video_url`` chat content and the plugin forwards every request's ``pixel_values`` / ``image_grid_thw`` /
``pixel_values_videos`` / ``video_grid_thw`` items plus their placeholder ranges to ``prefill_forward``
(``model_runner._gather_multi_modal_inputs``). The adapter encodes the items through the vision tower on device
(``tt/multimodal.py::VisionRuntime``: resident on the mesh, or built per request inside the prefill reserve),
scatters the features into the device chunk's input streams at the image-token rows (``tt/mm_inputs.py``: an
item straddling a 16 K chunk boundary is split at it) and runs ``GLM53MeshModel.prefill_streams`` per chunk. A
multimodal prompt always takes the EAGER prefill (the per-length traces embed the ids inside the trace, so no
stream can be injected): a 286-token image prompt pads to the 512 bucket. Knobs: ``GLM53_VLLM_IMAGES`` (auto =
the class's ``supports_multimodal``; 1 / 0 force), ``GLM53_VLLM_VISION_RESIDENT`` (1), ``GLM53_VLLM_VISION_BATCH_PATCHES``
(4096), ``GLM53_VLLM_WARMUP_MM`` (1). The text-only class is bit for bit the engine it was.

Speculative decoding (``GLM53_VLLM_SPEC=1``, the plugin's ``docs/SPEC_DECODE_CONTRACT.md``; policy in
``tt/vllm_spec.py``, graphs in ``tt/spec_decode.py``): the checkpoint's MTP layer drafts on device, the main model
verifies ``[anchor, d_1 .. d_K]`` per slot in one batched step (``B (K + 1) <= 32`` rows per mesh row), greedy
rows only (``argmax_ids``; a sampled request's steps run as the plain decode through ``supports_narrow_decode``).
Launch: ``--speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter",
"num_speculative_tokens": 3}'`` with ``GLM53_VLLM_SPEC=1 GLM53_PARTITION_USERS=0`` in the engine's env. The
adapter answers ``spec_plan`` (``GLM53SpecEngine`` / ``spec_plan``), ``decode_forward(spec_mode=...)``
(``_spec_verify``: graph V) and ``propose_draft_tokens`` (graphs C + D after a verify, D0 after a plain step);
every ordinary decode of a speculating launch is graph P (the plain step + MTP level 1, so the drafter's cache
follows every position); the prefill fills the MTP's cache from each chunk's streams (``_mtp_prefill_fill``).
v1 limits: replicated caches only (``GLM53_PARTITION_USERS=0``), the dense range (a slot beyond position
``2050 - K`` gets no drafts and makes its steps plain ones), ``rows_independent`` off.

Not supported (declared off): async decode overlap, stochastic device sampling (greedy argmax
only: ``max_device_top_k = 1``; other rows sample on host from the logits), multimodal prompts under
the per-row wave prefill, prefix caching together with speculative decode (one of the two per launch),
lane-DP / standard DP (one engine = one mesh).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from loguru import logger

from models.demos.glm53_flash.tt import vllm_spec as vs
from models.demos.glm53_flash.tt.mm_inputs import (
    IMAGE_TOKEN_ID,
    MMItem,
    chunk_scatter,
    feature_rows,
    items_in_range,
    mm_items,
    mm_present,
)
from models.demos.glm53_flash.tt.vllm_config import (
    APC_ALLOC_CHECK_ENV,
    APC_ENV,
    APC_MAX_SNAPSHOTS,
    APC_SNAPSHOT_STORE_ENV,
    APC_SNAPSHOTS_ENV,
    APC_TAIL_SPLIT_ENV,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DEVICE_PREFILL_CHUNK,
    DEFAULT_VISION_BATCH_PATCHES,
    DEFAULT_WAVE_ROWS,
    DENSE_PREFILL_ROWS,
    PREFIX_BUCKET_ENV,
    TILE,
    WAVE_CHUNK_UNIT,
    DecodeRowMap,
    LogicalBlockTable,
    PrefillPlan,
    PrefillPlanner,
    ServingGeometry,
    SlotPrefixCache,
    SlotTable,
    SPStep,
    WaveCall,
    WaveMember,
    WavePlanner,
    WaveStep,
    eager_pad_length,
    greedy_rows,
    max_tokens_all_users,
    mm_payload_missing,
    mm_spans_of_row,
    prefill_buckets,
)

HF_ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM"
MM_ADAPTER_TARGET = "models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForConditionalGeneration"
LATENT_DIM = 512  # kv_lora_rank: the latent page row (the only "KV" vLLM accounts for)

# multimodal serving knobs (read at initialize_vllm_model; the catalog sets them per profile)
IMAGES_ENV = "GLM53_VLLM_IMAGES"  # "auto" (default): the resolved class decides; "1" / "0" force the tower on / off
VISION_RESIDENT_ENV = "GLM53_VLLM_VISION_RESIDENT"  # "1": tower weights resident per chip; "0": built per request
VISION_BATCH_ENV = "GLM53_VLLM_VISION_BATCH_PATCHES"  # the tower's per-call patch budget (whole windows)
WARMUP_MM_ENV = "GLM53_VLLM_WARMUP_MM"  # "1": one dummy image prompt through the tower + eager prefill at boot


def _ttnn():
    """ttnn, imported on first device use only (the host test imports this module without it)."""
    import ttnn

    return ttnn


def _spec_upload(mesh, t: torch.Tensor, dtype, layout, rows_dim: int | None = None):
    """``spec_decode._upload`` (a persistent device tensor from host values, ``rows_dim`` split over the mesh rows): a
    device ALLOCATION, legal before the first trace capture only (``GLMMesh.allocations_frozen``). Imported on first
    device use; the host test substitutes a recorder."""
    from models.demos.glm53_flash.tt.spec_decode import _upload

    return _upload(mesh, t, dtype, layout, rows_dim)


def _spec_refresh(mesh, t: torch.Tensor, dst, rows_dim: int | None = None) -> None:
    """``spec_decode._refresh``: host values into the EXISTING device tensor ``dst`` (``copy_host_to_device_tensor``,
    no allocation): the only way a per-step / per-prompt value reaches the device once traces exist."""
    from models.demos.glm53_flash.tt.spec_decode import _refresh

    _refresh(mesh, t, dst, rows_dim)


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
WAVE_TRACE_LENGTHS_ENV = "GLM53_VLLM_WAVE_TRACE_LENGTHS"  # the per-row wave lengths captured at boot (rows_independent)
DEFAULT_WAVE_TRACE_LENGTHS = (
    "256,512"  # 256 = the smallest: the 45-layer wave gate validated it; a 128-token prompt pads to it
)
SP_TRACES_ENV = "GLM53_VLLM_SP_TRACES"  # "1": keep the sequence-parallel prefill traces under per-row ownership
SP_FIRST_ENV = "GLM53_VLLM_SP_FIRST"  # 0 = wave-affine placement for every prompt (the A/B of the placement policy)


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


class GLM53SpecEngine:
    """The device side of the served speculative decode: a ``SpecDecoder`` (``tt/spec_decode.py``) over the resident
    model and its MTP drafter, with the calls the adapter makes. The host test substitutes a torch fake with the
    same surface (``tests/test_serving_spec_host.py``); nothing here is imported without a device."""

    def __init__(self, model, mtp, draft_len: int, commit_mode: str):
        from models.demos.glm53_flash.tt.spec_decode import SpecDecoder

        self.model, self.mtp = model, mtp
        self.dec = SpecDecoder(model, mtp, draft_len, keep_logits=False, commit_mode=commit_mode, plain_draft=True)
        self.K = self.dec.K
        self.N = self.dec.N  # user slots over the mesh (rows x users per row)
        self.commit_mode = self.dec.commit_mode
        self.max_dense_pos = self.dec.vin.max_dense_pos
        self.hidden_size = int(model.cfg.hidden_size)
        self.pad_token_id = int(model.cfg.pad_token_id)
        # the prompt fill's pad-row mask expander: a [1,1,32,H] bf16 ones row (a mesh constant: allocated here, before
        # any capture; ``TtMoE.pad_ones`` keyed by its own width)
        ttnn = _ttnn()
        self.mask_ones = model.mesh.const(
            ("pad_mask_ones", self.hidden_size),
            lambda: model.mesh.upload(vs.pad_mask_ones_host(self.hidden_size), ttnn.bfloat16),
        )
        self.warmed_fill_lengths: list[int] = []
        # the prompt fill's per-length persistent inputs (allocated by ``warmup_prefill_fill`` before any capture,
        # refreshed in place per chunk): the next-token row [1,1,1,T] uint32 and the row mask [1,1,T,32] bf16
        self.fill_ids: dict[int, Any] = {}
        self.fill_mask: dict[int, Any] = {}

    # ---- the allocation discipline
    def frozen(self) -> bool:
        """True once any trace was captured on the mesh: no device allocation may outlive the next replay."""
        return bool(self.model.mesh.allocations_frozen[0])

    def dram_allocated(self) -> int:
        """The DRAM allocator's allocated bytes per bank (a host-side query, no device op): the footprint fingerprint
        the adapter compares across speculative steps (``GLM53_VLLM_SPEC_ALLOC_CHECK``), which must not change
        between two steps (a per-step allocation that survives a replay is what a replay corrupts)."""
        ttnn = _ttnn()
        if getattr(ttnn, "device", None) is None or getattr(self.model, "device", None) is None:
            return 0  # no allocator to ask (the host tests' fakes): a constant footprint
        mv = ttnn.device.get_memory_view(self.model.device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank)

    def _fill_buffers(self, T: int):
        """The persistent fill inputs of the padded length ``T``: allocated on first use BEFORE the freeze
        (``warmup_prefill_fill``), refused after it (a length the warm-up did not cover would allocate under live
        traces: fix the warm-up set, ``vllm_spec.mtp_fill_lengths``)."""
        T = int(T)
        if T not in self.fill_ids:
            if self.frozen():
                raise RuntimeError(
                    f"MTP prompt fill at T={T}: no persistent inputs for this chunk length and traces exist (warmed: "
                    f"{self.warmed_fill_lengths}); allocating now would be clobbered by the next replay. Add the "
                    "length to the warm-up set (vllm_spec.mtp_fill_lengths)"
                )
            ttnn = _ttnn()
            mesh = self.model.mesh
            rows = 3 if mesh.rows > 1 else None
            self.fill_ids[T] = _spec_upload(
                mesh,
                torch.full((1, 1, 1, T), self.pad_token_id, dtype=torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                rows,
            )
            self.fill_mask[T] = _spec_upload(
                mesh, vs.pad_row_mask_host(T, T), ttnn.bfloat16, ttnn.TILE_LAYOUT, 2 if mesh.rows > 1 else None
            )
        return self.fill_ids[T], self.fill_mask[T]

    # ---- warm-up / capture (the adapter's boot order: every eager run before every capture)
    def warmup_plain(self, ids: torch.Tensor, positions: list[int]) -> None:
        self.dec.warmup_plain(ids, positions)

    def warmup(self, positions: list[int]) -> None:
        self.dec.warmup(positions)

    def capture_plain(self, ids: torch.Tensor, positions: list[int]):
        """The P trace: ``(trace id, (logits, ids))``."""
        return self.dec.capture_plain(ids, positions)

    def capture(self) -> None:
        self.dec.capture()

    def release(self) -> None:
        self.dec.release()

    # ---- the ordinary step (P)
    def set_plain_positions(self, positions: list[int]) -> None:
        """The MTP's level-1 positions of the step (= the main positions the adapter loaded)."""
        self.model.set_mtp_positions(positions)

    def plain_step(self, eager: bool = False):
        """Replay P (eager before the capture, or on request): ``(logits, ids)``."""
        return self.dec.plain_step(eager=eager)

    def read_plain_d1(self) -> torch.Tensor:
        return self.dec.read_plain_d1()

    def draft_chain_plain(self, positions: list[int]) -> torch.Tensor:
        return self.dec.draft_chain_plain(positions)

    # ---- the verify / commit / draft graphs
    def verify(self, tokens: torch.Tensor, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dec.verify_forward(tokens, positions)

    def commit(self, m: torch.Tensor, positions: list[int]) -> None:
        self.dec.commit(m, positions)

    def draft_chain(self, m: torch.Tensor, d1: torch.Tensor, positions: list[int]) -> torch.Tensor:
        return self.dec.draft_chain(m, d1, positions)

    # ---- the prompt
    def prefill_fill(self, ids: torch.Tensor, first: int, streams, user: int, start: int, seq_len: int) -> None:
        """The MTP's cache / index rows ``[start, start + seq_len)`` of ``user`` from one prefill chunk: the main
        model's normed hidden of the chunk's streams (this row's slab ``[1,1,T_loc,4H]`` of the chunk's ``T_pad =
        rows x T_loc`` rows, eager or traced; ``head.collapse`` + ``head.norm``), its rows past ``seq_len`` zeroed
        (``MtpFillPlan.row_mask``: the main model leaves them unspecified), and the next token of every row
        (``MtpFillPlan.next_ids``: ``ids`` shifted by one, the last real row taking ``first`` = the chunk's greedy
        chunk-end token). ``spec_decode.prefill_with_mtp`` per chunk, with the mask."""
        ttnn = _ttnn()
        mesh = self.model.mesh
        plan = vs.mtp_fill_plan(
            ids,
            first,
            rows=mesh.rows,
            t_loc=int(streams.shape[-2]),
            seq_len=int(seq_len),
            start=int(start),
            pad_token_id=self.pad_token_id,
        )
        # the persistent per-length inputs (no allocation under live traces), refreshed in place for this chunk
        probe = lambda what: logger.info(f"GLM53 vLLM spec fill probe ({what}): {self.dram_allocated()} B/bank")
        probe("before")
        ids_dev, mask = self._fill_buffers(plan.t_pad)
        _spec_refresh(mesh, plan.next_ids, ids_dev, 3 if mesh.rows > 1 else None)
        probe("after the fill buffers + ids refresh")
        h = self.model.head.collapse(streams)  # this row's slab [1,1,T_loc,H] fp32
        hidden = self.model.head.norm(h)  # bf16: the LM-head input of every prompt row
        if h is not streams:
            ttnn.deallocate(h)
        probe("after collapse + norm (hidden live)")
        if plan.masked:
            # one exact 0/1 matmul (HiFi4, fp32 accumulation) + where: the rows past seq_len become 0 whatever they
            # held (NaN included), the real rows are copied bit for bit (``TtMoE._mask_pad_rows``); the matmul's
            # and where's outputs are this call's transients, freed before any replay
            _spec_refresh(mesh, plan.row_mask, mask, 2 if mesh.rows > 1 else None)
            full = ttnn.matmul(mask, self.mask_ones, dtype=ttnn.bfloat16, compute_kernel_config=self.mtp.norm_compute)
            clean = ttnn.where(full, hidden, 0.0)
            for t in (full, hidden):
                ttnn.deallocate(t)
            hidden = clean
        probe("after the mask path (hidden live)")
        out = self.mtp.prefill_cache(ids_dev, hidden, user=user, seq_len=plan.seq_len, start=plan.start)
        probe("after mtp.prefill_cache (out + hidden live)")
        for t in (out, hidden):
            ttnn.deallocate(t)
        probe("after the frees")

    def warmup_prefill_fill(self, lengths: Sequence[int]) -> None:
        """Run the prompt fill once at every padded chunk length the served prefill can produce (``vllm_spec.
        mtp_fill_lengths``), on a zero streams slab into slot 0, with rows past ``seq_len`` (the mask path): compiles
        the MTP's prefill programs and its row collectives at those shapes BEFORE any trace capture. The eager
        warm-up ladder covers only the buckets its prompts pad to (128 tokens -> 256), and the traced lengths'
        compile passes never run the drafter: on the 2026-10-27 (2,4) block run the first traced prompt's fill at
        T=128 compiled the MTP's gather / reduce-scatter after every capture and its routing came back garbage on
        every row (the late-collective trap). Writes placeholder rows into slot 0 (the next prompt overwrites them)."""
        ttnn = _ttnn()
        mesh = self.model.mesh
        for T in lengths:
            T = int(T)
            if T % (TILE * mesh.rows) or T < TILE:
                raise ValueError(f"fill length {T} is not a multiple of the row unit {TILE * mesh.rows}")
            if T in self.warmed_fill_lengths:
                continue
            t_loc = T // mesh.rows
            seq_len = T - TILE if T > TILE else T  # rows past seq_len: the mask path compiles too
            self._fill_buffers(T)  # the length's persistent inputs, allocated here (before any capture)
            streams = ttnn.zeros(
                (1, 1, t_loc, 4 * self.hidden_size),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dummy = torch.arange(seq_len, dtype=torch.long) % 1000 + 1000
            t0 = time.perf_counter()
            # the shortest length fills EVERY per-row user slot: the drafter's per-user constants (its index cache's
            # page-table / ring views, its KDA state writes) are built here, not at the first prompt into that slot
            # after the captures (the 2026-10-28 8x4 run: +1.43 MiB per bank after the eager chunk [0, 32)'s fill
            # into a slot the warm-up had not filled)
            users = range(int(getattr(self.model, "max_batch", 1))) if not self.warmed_fill_lengths else range(1)
            for u in users:
                self.prefill_fill(dummy, 1000, streams, u, 0, seq_len)
            ttnn.deallocate(streams)
            ttnn.synchronize_device(mesh.device)
            self.warmed_fill_lengths.append(T)
            logger.info(
                f"GLM53 vLLM: MTP prompt fill warmed at T={T} ({t_loc} rows per chip, seq_len {seq_len}) in "
                f"{time.perf_counter() - t0:.1f} s"
            )


class GLM53FlashForCausalLM:
    """GLM-5.3-Flash text generation on a Blackhole Galaxy mesh for the TT vLLM plugin."""

    decode_input_update_contract = 1

    model_capabilities = {
        # Slot-level APC (doc/serving/prefix_caching.md); off until device-validated. Read at import: the plugin
        # snapshots the class attribute in check_and_update_config.
        "supports_prefix_caching": _env_flag(APC_ENV, "0"),
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
        # The plugin's speculative contract (docs/SPEC_DECODE_CONTRACT.md): the model's own MTP drafter proposes on
        # device (``custom_class``), the target hidden state never leaves the device (the level-1 hidden of every
        # candidate row stays in the SpecDecoder's buffers), the verify answers ``argmax_ids``. ``spec_plan`` admits
        # a launch only with GLM53_VLLM_SPEC=1 (and replicated caches); without it the adapter is the plain one.
        "supports_spec_decode": True,
        "spec_requirements": (vs.SPEC_REQUIREMENT_DEVICE_PROPOSE,),
        "spec_hidden_handoff": (vs.HIDDEN_HANDOFF_ON_DEVICE,),
        "supports_async_spec_decode": False,
    }

    # Narrowed to the one captured decode width by ``warmup_model_decode`` (fixed user slots: no smaller bucket).
    tt_supported_decode_batch_sizes: tuple[int, ...] = ()
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False
    # vLLM's multimodal flag (``SupportsMultiModal.supports_multimodal``): False on the text class, True on
    # ``GLM53FlashForConditionalGeneration``; ``GLM53_VLLM_IMAGES=auto`` follows it
    supports_multimodal = False

    def __init__(
        self,
        model,
        geometry: ServingGeometry,
        *,
        pad_token_id: int | None = None,
        prefix_cache_entries: int | None = None,
        vision=None,
        spec=None,
        **vllm_kwargs,
    ):
        # ``**vllm_kwargs`` (``vllm_config`` / ``prefix``) is never used: the plugin constructs the adapter through
        # ``initialize_vllm_model``. vLLM >= 0.30 inspects the registered class with ``is_vllm_model`` (an ``__init__``
        # that accepts ``vllm_config``, ``embed_input_ids``, ``forward(input_ids, positions)``) before it admits the
        # generate runner, so the signature has to advertise the keyword like the tt_transformers adapters do.
        self.model = model
        if geometry.rows_independent:
            # the per-row lengths the mesh model captures waves at (the planner's traced-wave decision); a wave's
            # chunk pads to the next prefill bucket, so the wave width must be one (else every full chunk pads up)
            lengths = getattr(model, "wave_trace_lengths", None)
            if not geometry.traced_wave_lengths and callable(lengths):
                geometry = dataclasses.replace(geometry, traced_wave_lengths=tuple(int(T) for T in lengths()))
            if geometry.traced_prefill_lengths and not _env_flag(SP_TRACES_ENV, "0"):
                # the traced path of a fresh prompt is the wave (WavePlanner: a short prompt alone on its slot index
                # takes the traced wave too); the four SP prefill traces (~480 MB at 45 layers) left no room for any
                # wave trace in the 512 MB region (the 2026-10-29 8x4 runs). The SP path stays eager for the long lone
                # prompts (the 8 rows share that work).
                geometry = dataclasses.replace(geometry, traced_prefill_lengths=())
            padded = eager_pad_length(geometry.wave_rows, TILE)
            if padded != geometry.wave_rows:
                raise ValueError(
                    f"wave_rows {geometry.wave_rows} is not a prefill bucket ({prefill_buckets()}): a wave's chunk "
                    f"pads to {padded}; add it to GLM53_PREFILL_BUCKETS or pick a bucket width (GLM53_VLLM_WAVE_ROWS)"
                )
        self.geometry = geometry
        # the vision tower (``tt/multimodal.py::VisionRuntime`` or a test double with ``encode`` / ``chunk_streams``);
        # None = text-only: a prompt that arrives with pixel data is refused
        self.vision = vision
        # the speculative path (GLM53_VLLM_SPEC=1): a ``GLM53SpecEngine`` (or the host test's fake) over the model and
        # its MTP drafter; None = the plain adapter (every spec branch below is gated on it)
        self.spec = spec
        self.spec_policy: vs.ServingSpecPolicy | None = None
        if spec is not None:
            if geometry.rows_independent or geometry.partitioned:
                raise NotImplementedError(
                    "the speculative path needs replicated caches and today's row placement: GLM53_PARTITION_USERS=0 "
                    "and GLM53_VLLM_ROWS_INDEPENDENT=0 (tt/vllm_spec.py resolve_spec_decision)"
                )
            if int(spec.N) != geometry.physical_slots:
                raise ValueError(f"the spec engine serves {spec.N} slots, the geometry {geometry.physical_slots}")
            self.spec_policy = vs.ServingSpecPolicy(
                draft_len=int(spec.K), max_dense_pos=int(spec.max_dense_pos), max_seq_len=geometry.max_seq_len
            )
        self._spec_pending: vs.VerifyRecord | None = None  # a verify awaiting its commit (propose_draft_tokens)
        self._spec_last: vs.PlainRecord | None = None  # the last ordinary step of a speculating launch
        self._spec_stats = vs.SpecServingStats()
        self._spec_narrow_warned = False
        # GLM53_VLLM_SPEC_ALLOC_CHECK: the DRAM footprint right after the captures; every later speculative call must
        # leave it unchanged (``_spec_alloc_check``)
        self._spec_alloc_checking = spec is not None and vs.alloc_check_enabled()
        self._spec_alloc_baseline: int | None = None

        self.slots = SlotTable(geometry)
        self.blocks = LogicalBlockTable(geometry)
        self.prefill_planner = PrefillPlanner(geometry, traced_available=self._has_prefill_trace)
        # per-row prefill ownership (GLM53_VLLM_ROWS_INDEPENDENT): the step's prefill rows become waves (one chunk per
        # mesh row, different prompts) and, while exact, sequence-parallel single-prompt steps (ROW_OWNERSHIP.md)
        self.waves = WavePlanner(geometry, self.slots, self.prefill_planner, wave_captured=self._has_wave_trace)
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
        self._trace_outputs = None  # (logits, ids) of the captured decode step
        self._trace_capture_outputs = None  # every output of that capture, kept alive while the trace exists
        self._decode_inputs_ready = False
        self.already_warmed_up_prefill = False
        self._decode_warmup_complete = False
        self._pending_moves: dict[int, int] | None = None
        self._perf_steps = 0
        self._perf_s = 0.0
        self._perf_log_every = max(1, int(os.environ.get("GLM53_VLLM_PERF_LOG_EVERY", "32")))
        # slot-level prefix cache: the resident-content records + the KDA snapshot policy (None = APC off); the
        # model's snapshot store (``allocate_kda_snapshots``) must exist when entries > 0
        self.apc: SlotPrefixCache | None = None
        self.apc_tail_split = _env_flag(APC_TAIL_SPLIT_ENV, "0")
        if prefix_cache_entries is not None:
            self.apc = SlotPrefixCache(geometry, int(prefix_cache_entries))
            if self.apc.entries and not callable(getattr(model, "snapshot_kda", None)):
                raise NotImplementedError("the prefix cache needs a mesh model with snapshot_kda / restore_kda")
        # GLM53_VLLM_APC_ALLOC_CHECK: the DRAM footprint right after the decode capture; every later prefix-cache call
        # site must leave it unchanged (``_apc_alloc_check``, the served-spec audit's GLM53_VLLM_SPEC_ALLOC_CHECK shape)
        self._apc_alloc_checking = self.apc is not None and self.apc.entries > 0 and _env_flag(APC_ALLOC_CHECK_ENV, "0")
        self._apc_alloc_baseline: int | None = None

    def _spec_alloc_check(self, where: str) -> None:
        """The allocation discipline under live traces (``GLMMesh.allocations_frozen``): with the knob on, the
        allocator's footprint must equal the post-capture baseline after every speculative call (a persistent
        allocation made under live traces is clobbered by the next replay; a transient must have been freed)."""
        if not self._spec_alloc_checking or self._spec_alloc_baseline is None:
            return
        now = self.spec.dram_allocated()
        if now != self._spec_alloc_baseline:
            raise RuntimeError(
                f"GLM53 vLLM spec: the DRAM footprint changed after {where}: {now} vs {self._spec_alloc_baseline} "
                "bytes per bank at the captures (a device tensor allocated under live traces: the next replay "
                "clobbers it)"
            )

    # ================================================================================= construction
    @classmethod
    def images_enabled(cls) -> bool:
        """``GLM53_VLLM_IMAGES``: ``auto`` (default) = the resolved class's ``supports_multimodal`` (the multimodal
        class builds the tower, the text class does not); ``1`` / ``0`` force it."""
        raw = os.environ.get(IMAGES_ENV, "auto").strip().lower()
        if raw in ("", "auto"):
            return bool(getattr(cls, "supports_multimodal", False))
        return raw in ("1", "true", "yes", "on")

    @classmethod
    def vision_settings(cls) -> tuple[bool, int]:
        """``(resident, batch_patches)`` of the served tower: ``GLM53_VLLM_VISION_RESIDENT`` (1; the 1 M profile sets
        0: the tower is built per request inside the prefill reserve) and ``GLM53_VLLM_VISION_BATCH_PATCHES`` (4096;
        the 1 M profile sets 1024 so the per-request transient stays under its whole-prefix copy term)."""
        resident = _env_flag(VISION_RESIDENT_ENV, "1")
        batch = int(os.environ.get(VISION_BATCH_ENV, str(DEFAULT_VISION_BATCH_PATCHES)))
        if batch < 1:
            raise ValueError(f"{VISION_BATCH_ENV}={batch}: the tower's patch budget must be positive")
        return resident, batch

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
        from models.demos.glm53_flash.tt.memory_plan import (
            BF16_BYTES,
            BFP4_BYTES,
            BFP8_BYTES,
            MemoryPlan,
            vision_tower_bytes,
            vision_transient_bytes,
        )
        from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

        ttnn = _ttnn()
        snapshot = cls._snapshot_dir(hf_config, default_snapshot_dir)
        cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
        shape = list(mesh_device.shape)
        rows, cols = int(shape[0]), int(shape[1])
        # multimodal: the vision tower's bytes enter the planner (resident: static per chip; per request: inside the
        # prefill reserve together with the tower's transient for one batch of windows)
        multimodal = cls.images_enabled()
        vision_resident, vision_batch = cls.vision_settings()
        vision_static = vision_transient = 0.0
        if multimodal:
            from models.demos.glm53_flash.tt.vision import GLM53VisionConfig

            vc = GLM53VisionConfig.from_snapshot(snapshot)
            tower_bytes = vision_tower_bytes(
                vc.depth,
                vc.hidden_size,
                vc.intermediate_size,
                vc.out_hidden_size,
                vc.projection_intermediate_size,
                vc.patch_dim_padded,
                vc.head_dim,
            )
            vision_static = tower_bytes if vision_resident else 0.0
            vision_transient = vision_transient_bytes(vision_batch, vision_resident, tower_bytes)
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
            sp_first=os.environ.get(SP_FIRST_ENV, "1") == "1",
            sp_fallback=_env_flag("GLM53_VLLM_SP_FALLBACK", "1"),
            multimodal=multimodal,
            vision_resident=vision_resident,
            vision_batch_patches=vision_batch,
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
            vision_bytes=vision_static,
            vision_transient=vision_transient,
        )
        layers = cls._layer_ids(n_layers, cfg.num_hidden_layers)
        if len(layers) < cfg.num_hidden_layers:
            # a reduced-layer build (GLM53_MODEL_LAYERS: the (2,4) block tests): the planner's weights are the whole
            # model's, scaled to the subset so the gate does not refuse a 4-layer build on an 8-chip block
            plan = dataclasses.replace(plan, weights_bytes=plan.weights_bytes * len(layers) / cfg.num_hidden_layers)
            logger.info(f"GLM53 vLLM: {len(layers)}-layer build: the planner's weight bytes scaled to the subset")
        if rows_independent:
            plan = dataclasses.replace(plan, wave_rows=geometry.wave_rows)
        geometry.check_memory_plan(plan)
        # the speculative path: decided once from the env + geometry (the same rule ``spec_plan`` applied at config
        # time); the drafter and its buffers are built right after the model, before any warm-up or capture
        spec_decision = vs.resolve_spec_decision(
            os.environ,
            max_num_seqs=int(max_batch_size),
            requested_k=vs.spec_k_setting() if vs.spec_enabled() else vs.DEFAULT_SPEC_K,
            rows=rows,
            cols=cols,
            max_model_len=int(max_seq_len),
        )
        if vs.spec_enabled() and not spec_decision.admitted:
            raise ValueError(f"GLM53 vLLM: {vs.SPEC_ENV}=1 but the speculative path is refused: {spec_decision.reason}")
        if spec_decision.admitted:
            extra = vs.spec_resident_bytes(cfg, rows, cols, users_per_row, int(max_seq_len), eb)
            if plan.bytes_for(users_per_row, int(max_seq_len)) + extra > plan.headroom(int(max_seq_len)):
                raise ValueError(
                    f"GLM53 vLLM: the MTP drafter ({extra / 2**30:.2f} GiB per chip beside the caches) does not fit "
                    f"the planner's headroom at {users_per_row} users per row x {max_seq_len}: lower the context or "
                    f"drop {vs.SPEC_ENV}"
                )
        logger.info(
            f"GLM53 vLLM: {rows}x{cols} mesh, {users_per_row} users per row ({geometry.request_capacity} requests, "
            f"{geometry.physical_slots} slots), max_seq_len {max_seq_len}, partitioned={partitioned}, planner headroom "
            f"{plan.headroom(int(max_seq_len)) / 2**30:.1f} GiB vs {plan.bytes_for(users_per_row, int(max_seq_len)) / 2**30:.1f} GiB of caches"
            f" ({len(layers)} layers)"
            + (
                f"; vision tower {'resident' if vision_resident else 'per request'} ({vision_static / 2**30:.2f} GiB static, "
                f"{vision_transient / 2**20:.0f} MiB per-request reserve, {vision_batch} patches per call)"
                if multimodal
                else "; text only"
            )
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
        entries = None
        if cls.model_capabilities.get("supports_prefix_caching"):
            entries, store = cls._prefix_cache_plan(plan, users_per_row, int(max_seq_len))
            if entries:
                model.allocate_kda_snapshots(entries, store)  # before any capture: a persistent ring
            if store == "device" and entries:
                geometry.check_memory_plan(dataclasses.replace(plan, snapshot_entries=entries))
        vision = None
        if multimodal:
            # built here, before any warm-up capture: a resident tower allocated after a capture could alias that
            # trace's transient buffers; the per-request mode allocates only inside prefill_forward (eager, freed)
            from models.demos.glm53_flash.tt.multimodal import VisionRuntime

            vision = VisionRuntime(mesh_device, snapshot, resident=vision_resident, max_patches=vision_batch)
        spec = None
        if spec_decision.admitted:
            spec = cls._build_spec_engine(
                model,
                cfg,
                snapshot,
                cache_root,
                spec_decision,
                users_per_row,
                int(max_seq_len),
                expert_dtype,
                expert_down_dtype,
            )
        if spec is not None and entries:
            raise NotImplementedError(
                "prefix caching together with speculative decode is not wired (the verify path records no content "
                f"and takes no snapshots): launch with one of {APC_ENV}=1 and {vs.SPEC_ENV}=1"
            )
        _install_step_profiler()
        return cls(
            model, geometry, pad_token_id=cfg.pad_token_id, prefix_cache_entries=entries, vision=vision, spec=spec
        )

    @staticmethod
    def _prefix_cache_plan(plan, users_per_row: int, max_seq_len: int) -> tuple[int, str]:
        """``(entries per slot, store)`` of the KDA snapshot ring: ``GLM53_VLLM_APC_SNAPSHOTS`` (0 / unset = as many
        as the planner's spare DRAM fits, at most ``APC_MAX_SNAPSHOTS``) in the ``GLM53_VLLM_APC_SNAPSHOT_STORE``
        (``device``, the default; ``host``: torch copies, no DRAM). A device ring that fits nothing falls back to
        one host entry per slot (the multi-turn hit still works, over PCIe) with a warning."""
        want = int(os.environ.get(APC_SNAPSHOTS_ENV, "0"))
        store = os.environ.get(APC_SNAPSHOT_STORE_ENV, "device").strip().lower()
        fit = plan.max_snapshot_entries(users_per_row, max_seq_len, cap=APC_MAX_SNAPSHOTS)
        if store == "host":
            entries = want or 1
        else:
            entries = min(want, fit) if want else fit
            if want and entries < want:
                logger.warning(
                    f"GLM53 vLLM APC: {want} snapshot entries requested, the planner fits {fit}: using {entries}"
                )
            if entries == 0:
                store, entries = "host", 1
                logger.warning(
                    "GLM53 vLLM APC: no DRAM for a KDA snapshot ring at this profile (planner spare "
                    f"{plan.spare(users_per_row, max_seq_len) / 2**20:.0f} MiB per chip, one entry = "
                    f"{plan.snapshot_bytes(users_per_row, 1) / 2**20:.0f} MiB per chip): host store, 1 entry per slot"
                )
        logger.info(
            f"GLM53 vLLM APC: {entries} KDA snapshot entries per slot in the {store} store "
            f"({plan.snapshot_bytes(users_per_row, entries) / 2**20:.0f} MiB per chip when on device)"
        )
        return entries, store

    @staticmethod
    def _build_spec_engine(
        model,
        cfg,
        snapshot,
        cache_root,
        decision: vs.SpecDecision,
        users_per_row,
        max_seq_len,
        expert_dtype,
        expert_down_dtype,
    ) -> GLM53SpecEngine:
        """The MTP drafter (checkpoint layer 45: its routed experts from ``<cache>/layer45``, the model's shared
        embedding and head) attached to the model, and the ``SpecDecoder`` over both: every persistent tensor of the
        speculative graphs exists before the first warm-up / capture (the allocation discipline)."""
        from models.demos.glm53_flash.reference.checkpoint import shard_headers
        from models.demos.glm53_flash.tt.mtp import TtMTPBlock

        t0 = time.perf_counter()
        mtp = TtMTPBlock.from_checkpoint(
            model.device,
            cfg,
            snapshot,
            shard_headers(snapshot),
            embedding=model.embedding,
            head=model.head,
            weight_cache_root=cache_root,
            mesh=model.mesh,
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
            max_batch=users_per_row,
            max_seq_len=int(max_seq_len) + vs.MTP_CACHE_MARGIN,
        )
        model.attach_mtp(mtp)
        spec = GLM53SpecEngine(model, mtp, decision.k_cap, decision.commit_mode)
        logger.info(
            f"GLM53 vLLM: MTP drafter (layer {vs.MTP_LAYER}) + speculative graphs allocated in "
            f"{time.perf_counter() - t0:.0f} s: K={spec.K} (cap {decision.k_cap} for {users_per_row} users per row, "
            f"B(K+1)={users_per_row * (spec.K + 1)} verify rows per mesh row), KDA commit {spec.commit_mode}, "
            f"dense range: anchors <= {spec.max_dense_pos - spec.K}"
        )
        return spec

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        """The plugin's config-time admission (``docs/SPEC_DECODE_CONTRACT.md`` section 2): a ``SpecPlan`` with
        ``effective_k = min(requested_k, GLM53_VLLM_SPEC_K, 32 // users_per_row - 1)`` when ``GLM53_VLLM_SPEC=1``
        and the caches are replicated, else a ``SpecReject`` naming the knob. Never raises. The instance applies
        the same rule (``initialize_vllm_model``) so the two sides agree on K without sharing state; a launch whose
        ``num_speculative_tokens`` is below the adapter's K is served with its narrower block (the verify pads the
        missing columns and answers the plugin's width)."""
        max_model_len = None
        try:
            max_model_len = int(vllm_config.model_config.max_model_len)
        except Exception:  # the plugin passes a vLLM config; a test may pass anything
            pass
        rows, cols = _mesh_shape_from_env()
        decision = vs.resolve_spec_decision(
            os.environ,
            max_num_seqs=int(max_num_seqs),
            requested_k=int(requested_k),
            rows=rows,
            cols=cols,
            max_model_len=max_model_len,
        )
        extra = 0
        if decision.admitted and max_model_len:
            try:
                from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
                from models.demos.glm53_flash.tt.config import GLM53FlashConfig
                from models.demos.glm53_flash.tt.memory_plan import BFP8_BYTES

                cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
                extra = int(vs.spec_resident_bytes(cfg, rows, cols, decision.users_per_row, max_model_len, BFP8_BYTES))
            except Exception as exc:  # pragma: no cover - the byte estimate is advisory (not budgeted by the plugin)
                logger.warning(f"GLM53 vLLM: spec_plan could not estimate the drafter's bytes ({exc})")
        return vs.make_spec_plan(decision, extra_bytes_per_seq=extra // max(1, decision.users_per_row))

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
                if self.geometry.rows_independent:
                    self._warmup_waves()
                if self.vision is not None and _env_flag(WARMUP_MM_ENV, "1"):
                    self._warmup_multimodal()
                if self.spec is not None:
                    # the drafter's prompt fill at EVERY padded chunk length the served prefill can produce (the
                    # traced lengths + the eager buckets: the ladder above walked only the buckets its prompts pad
                    # to), then the speculative graphs, all eagerly and before ANY capture (a collective of a new
                    # shape compiled after a capture corrupts the replays: the 2026-10-26 block hang, and the
                    # 2026-10-27 served-spec block run, whose first traced prompt's fill was the first MTP prefill at
                    # T=128); placeholder rows at 0..K and in slot 0
                    t0 = time.perf_counter()
                    fill_lengths = self._spec_fill_lengths()
                    self.spec.warmup_prefill_fill(fill_lengths)
                    logger.info(f"GLM53 vLLM: MTP prompt fill warmed at the chunk lengths {fill_lengths}")
                    self.spec.warmup([0] * U)
                    self.model.reset_kda_states()
                    self._sync()
                    pol = self.spec_policy
                    logger.info(
                        f"GLM53 vLLM: SPECULATIVE DECODE ACTIVE: MTP drafter (layer {vs.MTP_LAYER}) K={self.spec.K}, "
                        f"B(K+1)={self.geometry.users_per_row * (self.spec.K + 1)} verify rows per mesh row, KDA commit "
                        f"{self.spec.commit_mode}, accept argmax_ids (greedy rows; sampled rows take the plain step), "
                        f"narrow decode on, drafts for anchors <= {pol.max_dense_pos - pol.draft_len} (dense range); "
                        f"graphs V/C/D/D0 compiled in {time.perf_counter() - t0:.1f} s"
                    )
                self.already_warmed_up_prefill = True
            return
        if self.trace_id is not None:
            raise RuntimeError("prefill traces must be captured before the decode trace (allocation order)")
        self._ensure_decode_inputs(U)
        wave_lengths = self._wave_trace_lengths() if self.geometry.rows_independent else []
        waves = bool(wave_lengths) and callable(getattr(self.model, "capture_waves", None))
        if waves:  # the wave traces' persistent inputs + their eager compiles precede EVERY capture
            t0 = time.perf_counter()
            prepared = self.model.prepare_waves(wave_lengths)
            logger.info(
                f"GLM53 vLLM: wave traces {prepared} (per-row lengths) prepared in {time.perf_counter() - t0:.0f} s"
            )
        if self.geometry.traced_prefill_lengths:
            t0 = time.perf_counter()
            captured = self.model.capture_prefill(list(self.geometry.traced_prefill_lengths))
            logger.info(f"GLM53 vLLM: prefill traces {captured} captured in {time.perf_counter() - t0:.0f} s")
        else:
            logger.info(
                "GLM53 vLLM: no sequence-parallel prefill traces (per-row ownership: the wave traces are the traced "
                f"path; {SP_TRACES_ENV}=1 keeps them)"
            )
        if waves:
            t0 = time.perf_counter()
            captured = self.model.capture_waves(wave_lengths)
            logger.info(
                f"GLM53 vLLM: wave traces {captured} (per-row lengths) captured in {time.perf_counter() - t0:.0f} s"
            )
        self.already_warmed_up_prefill = True

    def _warmup_apc_snapshots(self) -> None:
        """A KDA snapshot save + restore of EVERY entry of EVERY per-row user slot before any capture: the save reads
        the slot's live rows with a slice at the user's index and the restore reads the ring at the entry's index
        (a program per user and per entry), so the first snapshot into an entry the warm-up had not touched compiled
        at request time under the sealed program cache (SliceDeviceOperation, the 2026-10-28 served APC runs on the
        8x4: entry 0, then entry 1 at the hit's restore). The entries hold placeholder carries afterwards (a slot's
        real snapshots overwrite them); the live states are zeroed after the decode capture anyway."""
        store = getattr(self.model, "kda_snapshots", None)  # None on a text-only launch (and the host fakes)
        if self.apc is None or store is None or getattr(self, "_apc_snapshots_warmed", False):
            return
        t0 = time.perf_counter()
        B = self.geometry.users_per_row
        entries = int(getattr(self.model.kda_snapshots, "entries", self.apc.entries if self.apc is not None else 1))
        for u in range(B):
            for e in range(max(1, entries)):  # the ring reads slice at the entry's index: a program per entry too
                self.model.snapshot_kda(u, e)
                self.model.restore_kda(u, e, 0)
        self._sync()
        self._apc_snapshots_warmed = True
        logger.info(
            f"GLM53 vLLM APC: snapshot save + restore warmed for {B} slots x {max(1, entries)} entries in "
            f"{time.perf_counter() - t0:.1f} s"
        )

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
            self._warmup_apc_snapshots()
            return
        if self.trace_id is None:
            self._warmup_apc_snapshots()  # before the capture whatever the caller's eager / traced order
            t0 = time.perf_counter()
            if self.spec is None:
                self.trace_id, outs = self.model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
            else:
                # the speculating launch's ordinary step is P (the plain step + MTP level 1); V / C / D / D0 follow
                self.trace_id, outs = self.spec.capture_plain(torch.zeros(U, dtype=torch.long), [0] * U)
                self.spec.capture()
            # every output of the captured step stays referenced: the trace writes them on each replay, and a freed
            # one (the streams, when only logits / ids were kept) is a hole in the trace's address space the next
            # allocation reuses (the APC footprint check caught it: -64 KiB per bank after the baseline, 2026-10-28)
            self._trace_capture_outputs = tuple(outs)
            self._trace_outputs = (outs[-2], outs[-1])  # (logits, ids) of the captured step
            self.model.reset_kda_states()
            self._sync()
            logger.info(
                f"GLM53 vLLM: decode trace{'s (P, V, C, D, D0)' if self.spec is not None else ''} captured in "
                f"{time.perf_counter() - t0:.1f} s ({U} slots)"
            )
            if self._apc_alloc_checking:
                self._apc_alloc_baseline = self._dram_allocated()
                logger.info(
                    f"GLM53 vLLM APC: {APC_ALLOC_CHECK_ENV}=1: DRAM footprint baseline {self._apc_alloc_baseline} bytes "
                    "per bank after the decode capture; every prefix-cache call site is checked against it"
                )
            if self._spec_alloc_checking:
                self._spec_alloc_baseline = self.spec.dram_allocated()
                logger.info(
                    f"GLM53 vLLM spec: {vs.ALLOC_CHECK_ENV}=1: DRAM footprint baseline {self._spec_alloc_baseline} bytes "
                    "per bank after the captures; every speculative call is checked against it"
                )
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
        if dev is None or not hasattr(dev, "set_program_cache_misses_allowed"):
            return  # a model without a mesh device (the host tests' fakes): nothing to seal
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

    def _wave_trace_lengths(self) -> list[int]:
        """The per-row wave lengths captured at boot: ``GLM53_VLLM_WAVE_TRACE_LENGTHS``, a comma list drawn from the
        model's ``wave_trace_lengths`` (``all`` = every one, ``0`` = none: eager waves only; default 256,512: the
        45-layer wave gate validated the 256-row wave, to which a shorter prompt pads, as the eager path does).
        A wave trace of T rows per row costs about the SP trace of rows x T tokens: the 1024 / 2048-row waves are
        the 8 K / 16 K-token class serving never traced, and with the prefill traces they overflowed the 512 MB
        trace region on the 8x4 (2026-10-29: 550 MB at the first wave capture). Longer prompts take the eager wave;
        the traced wave's launch-bound gain is for the short ones."""
        lengths = getattr(self.model, "wave_trace_lengths", None)
        model_lengths = [int(T) for T in lengths()] if callable(lengths) else []
        raw = os.environ.get(WAVE_TRACE_LENGTHS_ENV, DEFAULT_WAVE_TRACE_LENGTHS).strip().lower()
        if raw in ("0", "", "none"):
            return []
        if raw == "all":
            return model_lengths
        want = sorted({int(v) for v in raw.split(",") if v.strip()})
        if WAVE_TRACE_LENGTHS_ENV in os.environ and any(T not in model_lengths for T in want):
            raise ValueError(
                f"{WAVE_TRACE_LENGTHS_ENV}={raw}: not among the model's wave trace lengths {model_lengths}"
            )
        return [T for T in want if T in model_lengths]

    def _warmup_waves(self) -> None:
        """The eager wave ladder (``GLM53_VLLM_ROWS_INDEPENDENT``): the wave path compiles its own programs, one set
        per per-row bucket (a wave's chunk pads to the next prefill bucket, up to the wave width) and, like the SP
        ladder, per prefix bucket of a continued chunk (the sparse read's kv shape) and per slot (the slot-keyed
        slices and state writes), so no wave shape compiles after the captures. Every row is a member (the idle-row
        form runs the same programs under masks); starts 0 (the dense first chunk), the wave width (a second chunk)
        and every prefix bucket (``GLM53_VLLM_WARMUP_PREFIX_STARTS=0`` drops these); slot 0 walks every bucket, the
        other warm-up slots the shortest; every bucket once ragged (the pad-row mask). The chunk cursors are rewound around every call; the caches and states hold
        placeholders afterwards (the next prompt overwrites them; the KDA states are zeroed after the captures)."""
        g = self.geometry
        W = g.wave_rows
        tails = sorted({b for b in prefill_buckets() if b <= W} | {W})
        starts = [0, W] if W <= g.max_seq_len else [0]
        if os.environ.get("GLM53_VLLM_WARMUP_PREFIX_STARTS", "1") == "1":
            bucket = int(os.environ.get(PREFIX_BUCKET_ENV, "0")) or g.device_prefill_chunk
            starts += [j * bucket for j in range(1, g.max_seq_len // bucket) if j * bucket != W]
        slots = self._warmup_slots(g.users_per_row)
        t0 = time.perf_counter()
        n_calls = 0
        for start in starts:
            for u in slots:
                for n in tails if u == slots[0] else tails[:1]:
                    if start + n > g.max_seq_len:
                        continue
                    dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
                    members = tuple(
                        WaveMember(logical=r, row=r, user=u, start=start, end=start + n) for r in range(g.rows)
                    )
                    call = WaveCall(rows=g.rows, t_pad=n, members=members, user=u)
                    for m in members:
                        call.ids[m.row] = dummy.clone()
                    self.model.rewind_user(u, start)
                    self.model.prefill_wave(call)
                    n_calls += 1
        # every bucket once RAGGED (row 0 one token short, the other rows full): a member whose chunk is shorter
        # than the wave's width (every prompt's last chunk, an idle row) takes the MoE's pad-row mask, a program the
        # full waves never build (the 2026-10-29 served needles: the last 2048-row waves compiled it under live
        # traces and the answers degenerated; the traced waves carry their mask); start 0, slot 0: the mask does not
        # depend on either
        u0 = slots[0]
        for n in tails:
            if n < 2 or n > g.max_seq_len:
                continue
            dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
            members = tuple(
                WaveMember(logical=r, row=r, user=u0, start=0, end=n - (1 if r == 0 else 0)) for r in range(g.rows)
            )
            call = WaveCall(rows=g.rows, t_pad=n, members=members, user=u0)
            for m in members:
                call.ids[m.row] = dummy[: m.n_tokens].clone()
            self.model.rewind_user(u0, 0)
            self.model.prefill_wave(call)
            n_calls += 1
        # every scorer-crop width a wave can reach: the sealed cache keys the crop on (sub-block rows, width), the width
        # follows the superset start max_r(start_r - r*T) rounded to the prefix bucket (``TtDSAIndexer.scored_width``),
        # and a member continued at a grid start between two bucket multiples reaches widths the starts above never
        # do when the bucket is smaller than the superset slack R*T/P (the block profile's 256-pool bucket: case (f),
        # 2026-10-29; the served buckets of >= 16384 tokens are not); per tail one lone row-0 wave per width still
        # unscored, at the first grid start that reaches it
        n_exact = 0
        dsa = next((d for d in getattr(self.model, "layers", []) if not d.is_kda and d.cache is not None), None)
        attn = getattr(getattr(dsa, "layer", None), "attn", None) if dsa is not None else None
        if attn is not None and getattr(attn, "indexer", None) is not None:
            from models.demos.glm53_flash.tt.dsa import prefill_blocks, sparse_block_plan

            idx, index, P = attn.indexer, dsa.cache.index, attn.indexer.P
            dense_rows = int(getattr(attn, "sparse_row_start", DENSE_PREFILL_ROWS))
            qb, ab = prefill_blocks(g.max_seq_len)
            for n in tails:
                subs = [(q0, q1) for _, _, ss in sparse_block_plan(0, n, qb, ab) for q0, q1 in ss]
                reach: dict[tuple[int, int], int] = {}  # (sub-block rows, width) -> the first grid start reaching it
                for s in range(0, g.max_seq_len - n + 1, WAVE_CHUNK_UNIT):
                    if s + n <= dense_rows:
                        continue  # all-dense: no scorer
                    for q0, q1 in subs:
                        rs = tuple(s + q0 if r == 0 else q0 for r in range(g.rows))
                        key = (q1 - q0, idx.scored_width(index, T=q1 - q0, row_starts=rs, num_pools_valid=(s + n) // P))
                        reach.setdefault(key, s)
                for key, s in sorted(reach.items(), key=lambda kv: kv[1]):
                    if key in idx.scored_shapes:
                        continue
                    dummy = torch.arange(n, dtype=torch.long) % 1000 + 1000
                    members = (WaveMember(logical=0, row=0, user=u0, start=s, end=s + n),)
                    call = WaveCall(rows=g.rows, t_pad=n, members=members, user=u0)
                    call.ids[0] = dummy.clone()
                    self.model.rewind_user(u0, s)
                    self.model.prefill_wave(call)
                    n_calls += 1
                    n_exact += 1
        for u in slots:
            self.model.rewind_user(u, 0)
        self._sync()
        logger.info(
            f"GLM53 vLLM: eager wave warm-up ({n_calls} waves x {g.rows} rows: per-row buckets {tails} at starts "
            f"{starts} into {len(slots)} slot(s), each bucket once ragged, {n_exact} lone wave(s) for the scorer widths "
            f"the starts missed) in {time.perf_counter() - t0:.1f} s"
        )

    def _warmup_slots(self, B: int) -> list[int]:
        """``GLM53_VLLM_WARMUP_SLOTS``: the per-row user slots the shortest warm-up prompt is prefilled into
        (default ``all``: every one of the ``B`` users a row holds — the mesh model's ``user`` index, replicated
        over the rows — so a program whose variant depends on that index, the eager per-user state writes, is
        built at boot, never at the first request into that slot; an int keeps the first that many)."""
        raw = os.environ.get("GLM53_VLLM_WARMUP_SLOTS", "all").strip().lower()
        n = B if raw in ("", "all") else max(1, min(B, int(raw)))
        return list(range(n))

    def _spec_fill_lengths(self) -> list[int]:
        """The padded chunk lengths the MTP prompt fill is warmed at (``vllm_spec.mtp_fill_lengths``): the traced
        lengths and the eager buckets up to the device chunk, within the caches."""
        g = self.geometry
        return vs.mtp_fill_lengths(
            g.traced_prefill_lengths,
            rows=g.rows,
            device_prefill_chunk=g.device_prefill_chunk,
            max_seq_len=g.max_seq_len,
        )

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

    def _warmup_multimodal(self) -> None:
        """One dummy image prompt (16 text + 256 image + 16 text tokens, the 448x448 geometry) through the tower and
        the eager stream prefill into slot 0 at boot: compiles the tower's programs (and exercises the per-request
        build + release), the 512-bucket eager prefill and the scatter ops, so the first image request compiles
        nothing. Eager phase only (before every capture); slot 0's state is overwritten by its next prompt."""
        t0 = time.perf_counter()
        self.vision.warmup()  # the tower at every patch bucket (a clip's 2048-patch call included)
        item = self.vision.dummy_item()
        text = torch.arange(16, dtype=torch.long) % 1000 + 1000
        ids = torch.cat([text, torch.full((item.n_tokens,), IMAGE_TOKEN_ID, dtype=torch.long), text])
        n = int(ids.numel())
        plan = PrefillPlan("eager", PrefillPlanner.walk(0, n, self.geometry.device_prefill_chunk))
        self._prefill_one_mm(ids, 0, plan, [item], want_logits=False)
        self._sync()
        logger.info(
            f"GLM53 vLLM: multimodal warm-up ({n} tokens, {item.n_tokens} image tokens through the "
            f"{'resident' if self.vision.resident else 'per-request'} tower) in {time.perf_counter() - t0:.1f} s"
        )

    def _ensure_decode_inputs(self, U: int) -> None:
        if self._decode_inputs_ready:
            return
        t0 = time.perf_counter()
        if self.spec is None:
            self.model.warmup_decode(torch.zeros(U, dtype=torch.long), [0] * U)
        else:
            self.spec.warmup_plain(torch.zeros(U, dtype=torch.long), [0] * U)  # P: the plain step + MTP level 1
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
        chip). Returns ``logits [B, 1, vocab]`` fp32 (host sampling) or ``(tokens [B, 1], None)`` (device greedy).

        Multimodal rows (the plugin's per-row ``pixel_values`` / ``image_grid_thw`` / ``pixel_values_videos`` /
        ``video_grid_thw`` item lists + ``mm_placeholder_offsets`` / ``mm_placeholder_lengths``): the items whose
        placeholder rows intersect the chunk are encoded through the vision tower and scattered into the eager
        prefill's input streams (``_prefill_one_mm``); a text row of the same batch takes the text path unchanged."""
        has_mm = mm_present(kwargs)
        if has_mm and self.vision is None:
            raise NotImplementedError(
                "multimodal prompt on a text-only GLM53 engine: serve GLM53FlashForConditionalGeneration "
                f"(TT_MODEL_CLASS_OVERRIDES) or set {IMAGES_ENV}=1 so the vision tower is built"
            )
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
        items = []  # (i, logical, state, start, end, ids, spans): start = the position the device run begins at
        apc = self.apc
        media: dict[int, tuple[list[MMItem], torch.Tensor]] = {}  # row -> (its multimodal items, prefix ids [0, end))
        for i in range(B):
            logical = int(empty_slots[i])
            start, end = starts[i], ends[i]
            if page_table is not None:
                self.blocks.check_row(page_table[i], end)
            self.prefill_planner.check_range(start, end)
            spans = mm_spans_of_row(kwargs, i) if apc is not None else []
            state = self.slots.physical(logical)
            row_items = mm_items(kwargs, i) if has_mm else []
            if row_items:
                prefix = tokens[i, :end].reshape(-1).to(torch.long)
                feature_rows(prefix, row_items)  # the payload must match the prompt before any slot is taken
                media[i] = (row_items, prefix)
            if start == 0:
                # a multimodal prompt takes the sequence-parallel path (the wave path has no vision scatter): under
                # per-row ownership it needs its slot index free on every row, which the SP-first placement provides
                state = self._allocate_slot(logical, end, sp_required=bool(row_items))
            elif state is not None and state.phase == "prefill" and state.prefilled_to == start:
                pass  # a chunked continuation of this slot's own prompt
            elif state is None and apc is not None:
                # a request vLLM admitted with num_computed_tokens = start (its first `start` tokens hashed to cached
                # blocks): served from the retained slot holding exactly that content + a KDA snapshot at or below
                # `start`, else downgraded to a prefill from 0 in a fresh slot (doc/serving/prefix_caching.md s.3)
                hit = apc.lookup(tokens[i], spans, start)
                if hit is not None:
                    state = self.slots.allocate(logical, at=(hit.row, hit.user))
                    apc.begin(hit.row, hit.user, hit.position)
                    mark = self._apc_alloc_mark()
                    self.model.restore_kda(state.user, hit.entry, hit.position)
                    self._apc_alloc_check(
                        f"the hit's restore of entry {hit.entry} at {hit.position} into user {state.user}", mark
                    )
                    apc.stats["hits"] += 1
                    logger.info(
                        f"[glm53-apc] hit: slot {logical} -> user {state.user} (row {state.row}) holds [0, {start}), "
                        f"snapshot at {hit.position}: prefilling [{hit.position}, {end}) ({apc.stats})"
                    )
                    start = hit.position
                else:
                    missing = mm_payload_missing(kwargs, i, start)
                    if missing:
                        apc.stats["refused_mm"] += 1
                        raise RuntimeError(
                            f"slot {logical}: prefix-cache hit of {start} tokens cannot be served (no retained slot "
                            f"holds that content) and multimodal items {missing} inside it arrived without their "
                            "pixels (vLLM strips fully cached items): the prefill from 0 is impossible. Needs the "
                            "plugin's hit confirmation hook (doc/serving/prefix_caching.md s.5) or APC off"
                        )
                    state = self._allocate_slot(logical, end)
                    apc.stats["downgraded"] += 1
                    logger.info(
                        f"[glm53-apc] miss: slot {logical}: no retained slot holds the {start}-token prefix; "
                        f"prefilling [0, {end}) in user {state.user} (row {state.row}) ({apc.stats})"
                    )
                    start = 0
            else:
                raise RuntimeError(
                    f"slot {logical}: chunk [{start}, {end}) is not contiguous with the prefilled prefix "
                    f"({None if state is None else state.prefilled_to}); chunks must follow each other"
                )
            ids = tokens[i, start:end].reshape(-1).to(torch.long)
            if int(ids.numel()) != end - start:
                raise ValueError(f"row {i}: tokens hold {ids.numel()} of the {end - start} chunk tokens")
            rows_of[logical] = i
            items.append((i, logical, state, start, end, ids, spans))

        def record(i, state, end, first, last_logits):
            state.prefilled_to = end
            state.position = end
            state.phase = "prefill"
            if wants_tokens:
                out_tokens[i, 0] = int(first)
            else:
                out_logits[i, 0] = last_logits

        def on_chunk(state):  # a KDA snapshot at every chunk end (the ring policy decides which to keep)
            if apc is None:
                return None
            return lambda pos, traced=False: self._snapshot(state, pos, "prefill", sync=traced)

        split_tail = apc is not None and self.apc_tail_split
        if not self.geometry.rows_independent:
            for i, logical, state, start, end, ids, spans in items:
                mm = media.get(i)
                # a multimodal prompt takes the eager stream prefill (the traces embed the ids inside the capture)
                plan = self.prefill_planner.plan(
                    start, end, enable_trace=enable_trace and mm is None, split_tail=split_tail
                )
                t0 = time.perf_counter()
                if mm is None:
                    first, last_logits = self._prefill_one(
                        ids, state.user, plan, want_logits=not wants_tokens, on_chunk=on_chunk(state)
                    )
                else:  # no chunk-end snapshots on the multimodal path (a later hit restores a decode snapshot)
                    first, last_logits = self._prefill_one_mm(
                        mm[1], state.user, plan, mm[0], want_logits=not wants_tokens
                    )
                if apc is not None:
                    apc.record_prefill(state.row, state.user, start, ids, spans)
                record(i, state, end, first, last_logits)
                logger.info(
                    f"GLM53 vLLM: prefill slot {logical} -> user {state.user} (row {state.row}) [{start}, {end}) via "
                    f"{plan.kind} ({len(plan.chunks)} chunk(s)) in {(time.perf_counter() - t0) * 1000:.0f} ms, "
                    f"chunk-end token {first}"
                    + (f", {len(mm[0])} multimodal item(s): {[it.kind for it in mm[0]]}" if mm is not None else "")
                )
                self._apc_alloc_check(f"the prefill [{start}, {end}) of slot {logical} (user {state.user})")
        else:
            by_logical = {logical: (i, state, start, end, ids) for i, logical, state, start, end, ids, _ in items}
            if apc is not None:  # the wave path records content; its snapshots are the decode crossings only
                for _, _, state, start, _, ids, spans in items:
                    apc.record_prefill(state.row, state.user, start, ids, spans)
            steps = self.waves.plan(
                [(logical, start, end) for _, logical, _, start, end, _, _ in items], enable_trace=enable_trace
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
                    self._apc_alloc_check(f"the SP prefill [{start}, {end}) of slot {step.logical}")
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
                self._apc_alloc_check(f"the wave {[(m.row, m.start, m.end) for m in call.members]}")
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

    def _prefill_one(self, ids: torch.Tensor, user: int, plan, *, want_logits: bool, on_chunk=None):
        """Run ``plan`` for one prompt chunk into physical user ``user``; returns ``(first greedy token, last-token
        logits [vocab] fp32 or None)``. ``on_chunk(end)`` is called after every device call (the prefix cache's
        chunk-end snapshots). TODO(device): confirm the traced path's tile read and the eager owner-row
        read on the 8x4 (both follow tests/test_mesh_model.py / demo/text_demo.py)."""
        if plan.kind == "traced":
            res = self.model.prefill_traced(ids, user=user)
            first = int(self.model.first_token(res))
            if self.spec is not None:  # the drafter's cache rows of the prompt, from the trace's streams
                self.spec.prefill_fill(ids, first, res.streams, user, 0, int(ids.numel()))
                self._spec_alloc_check("the traced prefill's MTP fill")
            logits = self._traced_last_logits(res) if want_logits else None
            if on_chunk is not None:
                on_chunk(plan.end, traced=True)
            return first, logits
        out = None
        mem_log = os.environ.get(MEM_LOG_ENV, "0") == "1"
        for s0, e in plan.chunks:
            if out is not None:
                self._free(out[0])
                self._free(out[1])
            chunk_ids = ids[s0 - plan.start : e - plan.start]
            out = self.model.prefill(chunk_ids, user=user, start=s0)
            if self.spec is not None:
                # every chunk fills the drafter's rows [s0, e): the last row takes the chunk's greedy chunk-end token
                # (the prompt's first generated token on the last chunk; on an intermediate chunk the model's own
                # guess of the next chunk's first token, which the plugin never hands this call)
                first_c = int(self.model.first_token(out[1], out[2]))
                self.spec.prefill_fill(chunk_ids, first_c, out[0], user, s0, e - s0)
            if on_chunk is not None:
                on_chunk(e)
            if mem_log:
                logger.info(
                    f"GLM53 vLLM mem: user {user} chunk [{s0}, {e}) done: {_dram_line(_mesh_device_of(self.model))}"
                )
        streams, logits_last, owner_row = out
        first = int(self.model.first_token(logits_last, owner_row))
        logits = self._eager_last_logits(logits_last, owner_row) if want_logits else None
        self._free(streams)
        self._free(logits_last)
        # the footprint check runs once the chunk outputs are freed: a check with the streams / logits still live
        # compared a transient against the post-capture baseline and always failed (the 2026-10-28 8x4 runs)
        self._spec_alloc_check(f"the eager prefill [{plan.start}, {plan.end}) (chunks + MTP fills, outputs freed)")
        return first, logits

    def _prefill_one_mm(self, prefix_ids: torch.Tensor, user: int, plan, mm: list[MMItem], *, want_logits: bool):
        """``_prefill_one`` for a prompt with multimodal items: ``prefix_ids`` ``[end]`` is the whole prefix up to
        the end of the chunk (``mm_inputs.feature_rows`` needs it: a video span may open in an earlier chunk),
        ``plan`` the EAGER device chunks of ``[start, end)``. The items whose placeholder rows intersect the range
        are encoded once (the tower on device, eager, temporaries freed); every device chunk then takes the text
        embedding of its ids with the features scattered at its image-token rows (``chunk_streams``), or the plain
        text ``prefill`` when no vision row falls into it (bit-identical to a text chunk). Rows past the chunk
        end are never written: a straddling item's remaining rows go with the next chunk."""
        if self.vision is None:
            raise NotImplementedError("multimodal prefill without a vision runtime")
        if plan.kind != "eager":
            raise ValueError("a multimodal prompt must take the eager prefill (the traces embed the ids)")
        rows = feature_rows(prefix_ids, mm)
        needed = items_in_range(rows, plan.start, plan.end)
        t0 = time.perf_counter()
        feats = dict(zip(needed, self.vision.encode([mm[k] for k in needed])))
        t_vision = time.perf_counter() - t0
        unit = TILE * self.geometry.rows
        mem_log = os.environ.get(MEM_LOG_ENV, "0") == "1"
        out = None
        for s0, e in plan.chunks:
            if out is not None:
                self._free(out[0])
                self._free(out[1])
            chunk_ids = prefix_ids[s0:e]
            entries = chunk_scatter(rows, s0, e)
            if not entries:
                out = self.model.prefill(chunk_ids, user=user, start=s0)
            else:
                pad_to = eager_pad_length(e - s0, unit)
                streams = self.vision.chunk_streams(
                    self.model, chunk_ids, [(en, feats[en.item]) for en in entries], pad_to
                )
                out = self.model.prefill_streams(streams, e - s0, user=user, start=s0)
            if mem_log:
                logger.info(
                    f"GLM53 vLLM mem: user {user} chunk [{s0}, {e}) done ({sum(en.n for en in entries)} vision rows): "
                    f"{_dram_line(_mesh_device_of(self.model))}"
                )
        streams, logits_last, owner_row = out
        first = int(self.model.first_token(logits_last, owner_row))
        logits = self._eager_last_logits(logits_last, owner_row) if want_logits else None
        self._free(streams)
        self._free(logits_last)
        logger.info(
            f"GLM53 vLLM: vision {len(needed)} of {len(mm)} item(s) ({sum(mm[k].n_tokens for k in needed)} tokens, "
            f"{sum(mm[k].n_patches for k in needed)} patches) encoded in {t_vision * 1000:.0f} ms for [{plan.start}, {plan.end})"
        )
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

    def _has_wave_trace(self, T: int) -> bool:
        has = getattr(self.model, "has_wave_trace", None)
        return bool(has(T)) if callable(has) else False

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
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid_drafts = kwargs.pop("num_valid_drafts", None)
        kwargs.pop("accepted_counts", None)  # the commit happened in propose_draft_tokens (graph C); nothing to select
        kwargs.pop("draft_token_ids", None)
        if spec_mode is not None or num_valid_drafts is not None:
            if self.spec is None:
                raise NotImplementedError(
                    f"a speculative step reached the GLM53 adapter without {vs.SPEC_ENV}=1 (spec_plan would have "
                    "refused the launch): the drafter is not built"
                )
            return self._spec_verify(
                tokens,
                start_pos,
                spec_mode,
                num_valid_drafts,
                page_table=page_table,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
                slot_remap=slot_remap,
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
            )
        self._check_reload(reload_inputs, reload_page_table, enable_trace)
        tokens = torch.as_tensor(tokens).reshape(-1)
        positions = torch.as_tensor(start_pos).reshape(-1)
        wire = int(tokens.shape[0])
        if int(positions.shape[0]) != wire:
            raise ValueError(f"tokens {wire} rows vs start_pos {int(positions.shape[0])}")
        if self.spec is not None and self._spec_pending is not None:
            raise RuntimeError(
                "a decode step follows a verify whose commit never ran (propose_draft_tokens was not called): the "
                "KDA states still hold the pre-verify context"
            )
        # the remap, applied exactly once; rolled back if the device refuses the step so the plugin's un-advanced
        # ownership map and this table still agree
        snapshot = list(self.slots._phys)
        applied_moves: dict[int, int] = {}
        if slot_remap is not None:
            remap = [int(v) for v in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            applied_moves = self.slots.apply_remap(remap)
        try:
            is_tokens = sampling_params is not None
            live, feed_tokens, feed_positions, rows = self._stage_rows(tokens, positions, page_table, sampling_params)
            park = self.apc.park_positions() if self.apc is not None else None
            ids_feed, pos_feed = self.slots.device_feed(feed_tokens, feed_positions, park=park)
            t0 = time.perf_counter()
            self._load_inputs(ids_feed, pos_feed, live)
            if self.spec is None:
                if enable_trace:
                    self.model.replay(self.trace_id)
                    logits_dev, ids_dev = self._trace_outputs
                    transient = False
                else:
                    _, logits_dev, ids_dev = self.model.decode_step(self.model.ids_dev, self.model.positions)
                    transient = True
            else:
                # the speculating launch's ordinary step: P (the plain step + MTP level 1 at the same positions, so
                # the drafter's cache follows every position and the next proposal starts from this step)
                self.spec.set_plain_positions(pos_feed)
                logits_dev, ids_dev = self.spec.plain_step(eager=not enable_trace)
                transient = not enable_trace
                self._spec_last = vs.PlainRecord(positions=list(pos_feed), read_index=list(rows.read_index), live=live)
                self._spec_stats.plain_steps += 1
            synced = False
            for i in live:
                state = self.slots.physical(i)
                state.position = feed_positions[i] + 1
                state.phase = "decode"
                if self.apc is not None:
                    self.apc.record_decode(state.row, state.user, feed_positions[i], feed_tokens[i])
                    if state.position % TILE == 0:
                        # the state now sits on a block boundary: the multi-turn hit position. The step's first save
                        # drains the device first (the replay just enqueued), the later ones queue behind it
                        if self._snapshot(state, state.position, "decode", sync=not synced):
                            synced = True
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
            self._spec_alloc_check("the ordinary step")
            return self.process_decode_output_host(out, is_tokens=is_tokens)
        if transient:
            out = self._read(out)
            self._free(logits_dev)
            self._free(ids_dev)
        self._perf(t0, len(live))
        return out

    def _check_reload(self, reload_inputs: bool, reload_page_table: bool, enable_trace: bool) -> None:
        if not reload_inputs:
            raise ValueError(
                "reload_inputs=False: the GLM53 adapter has no resident token / position state yet "
                "(supports_async_decode is False), every decode must reload its inputs"
            )
        if reload_page_table:
            raise ValueError("reload_page_table with reload_inputs is not a legal command combination")
        if self.trace_id is None and enable_trace:
            raise RuntimeError("decode trace not captured: warmup_model_decode(enable_trace=True) must run first")

    def _stage_rows(self, tokens: torch.Tensor, positions: torch.Tensor, page_table, sampling_params):
        """The per-row bookkeeping of one decode submission (plain or verify): the live wire rows (position >= 0),
        their anchor token / position per logical slot (phase and contiguity checks, the block-table check), the row
        map, and the mid-prefill guard. ``tokens`` / ``positions`` are the ``[B_pad]`` column-0 values."""
        wire = int(tokens.shape[0])
        if wire > self.slots.n:
            raise ValueError(f"{wire} decode rows exceed the {self.slots.n} logical slots")
        live = [i for i in range(wire) if int(positions[i]) >= 0]
        if sampling_params is not None:
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
        return live, feed_tokens, feed_positions, rows

    def _load_inputs(self, ids_feed: list[int], pos_feed: list[int], live: list[int]) -> None:
        """The step's token / position vectors onto the device (one call per step: with GLM53_DECODE_PACKED_INPUTS=1
        the mesh model uploads ONE packed tensor, ids + positions + sparse metadata, tt/packed_inputs.py) and the
        future active-slot hook."""
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

    # ================================================================================= speculative decode
    def _spec_verify(
        self,
        tokens,
        start_pos,
        spec_mode: str,
        num_valid_drafts,
        *,
        page_table,
        enable_trace: bool,
        sampling_params,
        slot_remap,
        reload_inputs: bool,
        reload_page_table: bool,
    ):
        """A decode step the plugin marked speculative: the ``[B_pad, 1+K']`` candidate block (column 0 the row's last
        committed token at ``start_pos[:, 0]``, columns ``1 .. num_valid_drafts[i]`` its drafts, the rest
        placeholders) -> ``VerifyOutput(spec_mode="argmax_ids", argmax_ids [B_pad, 1+K'])``: return column ``j`` is
        the main model's greedy choice at candidate position ``j`` (graph V over every slot's ``[anchor, d_1 ..
        d_K]`` rows, the padded columns carrying the anchor). The commit of the accepted prefix is graph C, run in
        ``propose_draft_tokens`` once the runner's accept walk has fixed the counts. When V cannot run for this
        batch (a live slot beyond the dense range, or no block to verify), the step is the plain one (P) on column 0
        and the answer is ONE column wide: the runner then commits exactly that token per row and counts every
        draft as rejected (``accept_greedy_drafts`` at width 1), which is what a plain step commits."""
        if spec_mode != vs.ACCEPT_MODE_ARGMAX_IDS:
            raise ValueError(
                f"the GLM53 adapter verifies in {vs.ACCEPT_MODE_ARGMAX_IDS!r} only, asked for {spec_mode!r}"
            )
        self._check_reload(reload_inputs, reload_page_table, enable_trace)
        if self._spec_pending is not None:
            raise RuntimeError(
                "a verify follows a verify whose commit never ran (propose_draft_tokens was not called in between)"
            )
        types = vs.contract_types()
        tokens = torch.as_tensor(tokens)
        positions = torch.as_tensor(start_pos)
        if tokens.dim() != 2 or positions.shape != tokens.shape:
            raise ValueError(
                f"a verify needs [B, 1+K] tokens and positions, got {tuple(tokens.shape)} / {tuple(positions.shape)}"
            )
        wire, width = (int(v) for v in tokens.shape)
        K = int(self.spec.K)
        if width - 1 > K:
            raise ValueError(f"the verify block carries {width - 1} drafts per row, the captured graphs verify {K}")
        if width - 1 < K and not self._spec_narrow_warned:
            self._spec_narrow_warned = True
            logger.warning(
                f"GLM53 vLLM: the launch verifies {width - 1} drafts per row, the adapter's graphs {K} "
                f"({vs.SPEC_K_ENV}): the missing columns are padded; set num_speculative_tokens={K} to use them"
            )
        valid = (
            [int(v) for v in torch.as_tensor(num_valid_drafts).reshape(-1).tolist()]
            if num_valid_drafts is not None
            else [0] * wire
        )
        if len(valid) != wire:
            raise ValueError(f"num_valid_drafts has {len(valid)} rows for a block of {wire}")
        snapshot = list(self.slots._phys)
        applied_moves: dict[int, int] = {}
        if slot_remap is not None:
            remap = [int(v) for v in torch.as_tensor(slot_remap).reshape(-1).tolist()]
            applied_moves = self.slots.apply_remap(remap)
        try:
            live, feed_tokens, feed_positions, rows = self._stage_rows(
                tokens[:, 0], positions[:, 0], page_table, sampling_params
            )
            drafted = [i for i in live if valid[i] > 0]
            can_verify = (
                bool(drafted) and width > 1 and self.spec_policy.step_can_verify([feed_positions[i] for i in live])
            )
            t0 = time.perf_counter()
            if not can_verify:
                # the plain answer: P on column 0, one column wide (every draft counts as rejected)
                ids_feed, pos_feed = self.slots.device_feed(feed_tokens, feed_positions)
                self._load_inputs(ids_feed, pos_feed, live)
                self.spec.set_plain_positions(pos_feed)
                logits_dev, ids_dev = self.spec.plain_step(eager=not enable_trace)
                host_ids = self.model.ids_to_host(ids_dev).reshape(-1)
                if not enable_trace:
                    self._free(logits_dev)
                    self._free(ids_dev)
                argmax = torch.zeros(wire, 1, dtype=torch.int32)
                for i in live:
                    argmax[i, 0] = int(host_ids[rows.read_index[i]])
                    state = self.slots.physical(i)
                    state.position = feed_positions[i] + 1
                    state.phase = "decode"
                self._spec_last = vs.PlainRecord(positions=list(pos_feed), read_index=list(rows.read_index), live=live)
                self._spec_stats.plain_steps += 1
                if drafted:
                    self._spec_stats.fallbacks += 1
                    beyond = [i for i in live if not self.spec_policy.anchor_ok(feed_positions[i])]
                    logger.info(
                        f"GLM53 vLLM spec: verify of rows {drafted} answered one column wide (plain step): rows "
                        f"{beyond} sit beyond the dense range (anchor + K > {self.spec_policy.max_dense_pos})"
                    )
                out = types["VerifyOutput"](spec_mode=vs.ACCEPT_MODE_ARGMAX_IDS, argmax_ids=argmax, hidden=None)
            else:
                blocks = {i: vs.candidate_block(tokens[i].tolist(), valid[i], K + 1) for i in live}
                ids_feed, pos_feed = self.slots.device_feed_block(blocks, feed_positions, K + 1)
                ids_v, d1_v = self.spec.verify(ids_feed, pos_feed)  # [N, K+1] each, user-major
                argmax = torch.zeros(wire, width, dtype=torch.int32)
                for i in live:
                    argmax[i] = ids_v[rows.read_index[i], :width].to(torch.int32)
                    state = self.slots.physical(i)
                    state.phase = "decode"  # the position advances with the accepted count, in propose_draft_tokens
                self._spec_pending = vs.VerifyRecord(
                    positions=list(pos_feed),
                    read_index=list(rows.read_index),
                    live=live,
                    ids=ids_v,
                    d1=d1_v,
                    width=width,
                    valid={i: valid[i] for i in live},
                )
                self._spec_stats.verifies += 1
                self._spec_stats.verify_s += time.perf_counter() - t0
                out = types["VerifyOutput"](spec_mode=vs.ACCEPT_MODE_ARGMAX_IDS, argmax_ids=argmax, hidden=None)
        except Exception:
            self.slots._phys = snapshot
            raise
        self._pending_moves = applied_moves or None
        self._perf(t0, len(live))
        self._spec_alloc_check("the verify step")
        return out

    def propose_draft_tokens(
        self, num_drafts: int, committed_tokens, committed_positions, accepted_counts, hidden=None
    ):
        """The plugin's proposal call after every decode step of a speculating launch (``docs/SPEC_DECODE_CONTRACT.md``
        section 4b): ``committed_tokens`` / ``committed_positions`` ``[B_pad, 1+K']`` the step's committed block,
        ``accepted_counts[i]`` how much of row ``i`` is real. After a verify: graph C commits candidates ``0 ..
        count - 1`` of every slot (the KDA states and the pooled-key index; the next anchor is the block's last
        committed token = the verify's bonus / correction) and graph D chains MTP levels ``2 .. K`` from the
        accepted row's level-1 output; after an ordinary step (count 1 everywhere): graph D0 chains them from P's
        level-1 output. ``hidden`` is None (the hidden state stays on device). A row is offered ``min(K, num_drafts)``
        drafts when its next anchor and every other live anchor are inside the dense range, else 0
        (``DraftOutput.num_valid``): the next step is then the plain decode."""
        if self.spec is None:
            raise NotImplementedError(f"propose_draft_tokens without {vs.SPEC_ENV}=1: the drafter is not built")
        types = vs.contract_types()
        committed = torch.as_tensor(committed_tokens)
        positions = torch.as_tensor(committed_positions)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        rows_n = int(committed.shape[0])
        K, N = int(self.spec.K), int(self.spec.N)
        k_req = int(num_drafts)
        t0 = time.perf_counter()
        rec, last = self._spec_pending, self._spec_last
        if rec is not None:
            if rows_n != len(rec.read_index):
                raise ValueError(f"propose_draft_tokens got {rows_n} rows, the verify ran on {len(rec.read_index)}")
            m = torch.zeros(N, dtype=torch.long)
            next_anchor: dict[int, int] = {}
            for i in rec.live:
                c = int(counts[i])
                if not 1 <= c <= rec.valid.get(i, 0) + 1:
                    raise ValueError(
                        f"row {i}: accepted count {c} outside [1, {rec.valid.get(i, 0) + 1}] (num_valid_drafts "
                        f"{rec.valid.get(i, 0)})"
                    )
                flat = rec.read_index[i]
                expect = int(rec.ids[flat, c - 1])
                got = int(committed[i, c - 1])
                if got != expect:
                    raise RuntimeError(
                        f"row {i}: the runner committed {got} at count {c}, the verify's choice there was {expect}: "
                        "the accept walk and the adapter's verify output disagree"
                    )
                for f in self.slots.flat_indices(i):
                    m[f] = c - 1
                next_anchor[i] = rec.positions[flat] + c
            # the record is consumed only once every check passed (a refused proposal leaves the adapter as it was)
            self._spec_pending = None
            self.spec.commit(m, rec.positions)
            for i in rec.live:
                c = int(counts[i])
                state = self.slots.physical(i)
                if state is not None:
                    state.position = next_anchor[i]
                self._spec_stats.committed += c
                self._spec_stats.accepted += c - 1
            d1_sel = rec.d1[torch.arange(N), m]
            rest = self.spec.draft_chain(m, d1_sel, rec.positions) if K > 1 else torch.zeros(N, 0, dtype=torch.long)
            read_index, live = rec.read_index, rec.live
        elif last is not None:
            if rows_n != len(last.read_index):
                raise ValueError(f"propose_draft_tokens got {rows_n} rows, the step ran on {len(last.read_index)}")
            self._spec_last = None
            d1_sel = self.spec.read_plain_d1().reshape(-1)
            rest = self.spec.draft_chain_plain(last.positions) if K > 1 else torch.zeros(N, 0, dtype=torch.long)
            next_anchor = {i: last.positions[last.read_index[i]] + 1 for i in last.live}
            read_index, live = last.read_index, last.live
        else:
            raise RuntimeError("propose_draft_tokens without a decode step to propose from")
        drafts = torch.cat([d1_sel.reshape(N, 1).long(), rest.reshape(N, K - 1).long()], dim=1)  # [N, K]
        out_ids = torch.zeros(rows_n, k_req, dtype=torch.int32)
        out_valid = torch.zeros(rows_n, dtype=torch.int32)
        all_ok = self.spec_policy.step_can_verify([next_anchor[i] for i in live])
        for i in live:
            # the runner's committed block starts one past the step's first column: its last real column is the
            # position the next anchor is consumed at (``_committed_positions``)
            if positions.dim() == 2 and int(positions[i, int(counts[i]) - 1]) != next_anchor[i]:
                raise RuntimeError(
                    f"row {i}: the runner's committed positions end at {int(positions[i, int(counts[i]) - 1])}, the "
                    f"adapter's next anchor is {next_anchor[i]}"
                )
            n = self.spec_policy.row_drafts(next_anchor[i], all_ok, k_req)
            out_ids[i, :n] = drafts[read_index[i], :n].to(torch.int32)
            out_valid[i] = n
        self._spec_stats.propose_s += time.perf_counter() - t0
        self._spec_alloc_check("the proposal (commit + draft chain)")
        return types["DraftOutput"](draft_token_ids=out_ids, num_valid=out_valid)

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
            state = self.slots.physical(int(slot))
            self.slots.release(int(slot))
        except ValueError:
            logger.warning(f"GLM53 vLLM: release_request({slot}) outside the slot range")
            return
        if self.apc is not None and state is not None:
            self.apc.release(state.row, state.user)  # the content stays resident (parked at its end) until reused
            self._apc_alloc_check(f"release_request({slot})")

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
        if self.spec is not None and self.trace_id is not None:
            # the P trace is the SpecDecoder's (released with V / C / D / D0)
            try:
                self.spec.release()
            except Exception as exc:  # pragma: no cover - device library detail
                logger.warning(f"GLM53 vLLM: releasing the speculative traces failed: {exc}")
            self.trace_id = None
            self._trace_outputs = None
            self._trace_capture_outputs = None
            self._spec_pending = self._spec_last = None
        if self.trace_id is not None:
            if "ttnn" in sys.modules and getattr(self.model, "device", None) is not None:
                try:
                    sys.modules["ttnn"].release_trace(self.model.device, self.trace_id)
                except Exception as exc:  # pragma: no cover - device library detail
                    logger.warning(f"GLM53 vLLM: release_trace failed: {exc}")
            self.trace_id = None
            self._trace_outputs = None
            self._trace_capture_outputs = None
        rel = getattr(self.model, "release_prefill_traces", None)
        if rel is not None:
            rel()
        store = getattr(self.model, "kda_snapshots", None)
        if store is not None and callable(getattr(store, "release", None)):
            store.release()
        self._decode_warmup_complete = False
        self.tt_supported_decode_batch_sizes = ()

    # ================================================================================= prefix cache
    def confirm_prefix_cache_hit(self, prompt_token_ids, mm_items=(), num_computed_tokens: int = 0) -> int:
        """The tokens of a cached prefix the adapter will honour (the plugin proposal of
        doc/serving/prefix_caching.md s.5 calls this from the scheduler before it commits vLLM's hit): the snapshot
        position of the best retained slot holding ``prompt_token_ids[:num_computed_tokens]`` (``mm_items``:
        ``(identifier, offset, length, modality)`` per multimodal item), 0 when nothing resident serves it."""
        if self.apc is None or num_computed_tokens <= 0:
            return 0
        from models.demos.glm53_flash.tt.vllm_config import MMSpan

        spans = [
            MMSpan(int(off), int(ln), "video" if str(mod).startswith("video") else str(mod or "image"), ident)
            for ident, off, ln, mod in mm_items
        ]
        ids = torch.as_tensor(list(prompt_token_ids), dtype=torch.int32)
        hit = self.apc.lookup(ids, spans, int(num_computed_tokens))
        return 0 if hit is None else int(hit.position)

    # ================================================================================= helpers
    def _allocate_slot(self, logical: int, end: int = 0, sp_required: bool = False):
        """A fresh prompt's physical slot: the column / row balance, then (APC) a slot without retained content
        before the least recently used resident prefix; a logical slot still mapped (never released) retains first.
        ``end``: the first chunk's end; under per-row ownership a prompt no traced wave holds (it prefills eagerly)
        takes a slot index live on no row, so its chunks run sequence-parallel (``SlotTable.allocate(prefer_sp)``);
        ``sp_required`` (a multimodal prompt) marks the slot so the planner never routes it through a wave."""
        prefer_sp = self.geometry.rows_independent and (
            sp_required or self.waves.captured_wave_length(int(end)) is None
        )
        if self.apc is None:
            state = self.slots.allocate(logical, prefer_sp=prefer_sp)
        else:
            old = self.slots.physical(logical)
            if old is not None:
                self.apc.release(old.row, old.user)
            state = self.slots.allocate(logical, slot_rank=self.apc.slot_rank, prefer_sp=prefer_sp)
            self.apc.begin(state.row, state.user, 0)
        state.sp_required = bool(sp_required)
        return state

    def _snapshot(self, state, position: int, kind: str, *, sync: bool = False) -> bool:
        """Snapshot slot ``state``'s KDA carries at ``position`` when the ring policy grants an entry; True when a
        save ran. ``sync``: the save follows a traced replay (a decode crossing, a traced prefill chunk): the device
        is drained first (``ttnn.synchronize_device``) so the save's reads of the live state and its scratch writes
        never overlap the replay. ``GLM53MeshModel.replay`` blocks until the replay finished anyway; the explicit
        sync keeps the order independent of that default (doc/serving/prefix_caching.md s.6)."""
        entry = self.apc.plan_snapshot(state.row, state.user, position, kind)
        if entry is None:
            return False
        if sync:
            self._sync()
        mark = self._apc_alloc_mark()
        self.model.snapshot_kda(state.user, entry)
        self._apc_alloc_check(
            f"the {kind} snapshot of user {state.user} (row {state.row}) at {position} into entry {entry}", mark
        )
        return True

    # ================================================================================= allocation discipline
    def _dram_allocated(self) -> int:
        fn = getattr(self.model, "dram_allocated", None)
        if fn is None:
            raise RuntimeError(f"{APC_ALLOC_CHECK_ENV}=1 needs a mesh model with dram_allocated() (GLM53MeshModel)")
        return int(fn())

    def _apc_alloc_mark(self) -> int | None:
        """The DRAM footprint before a save / restore (None: the check is off or no decode trace exists yet)."""
        if not self._apc_alloc_checking or self._apc_alloc_baseline is None:
            return None
        return self._dram_allocated()

    def _apc_alloc_check(self, where: str, mark: int | None = None) -> None:
        """The allocation discipline under live traces (doc/serving/prefix_caching.md s.6): with
        ``GLM53_VLLM_APC_ALLOC_CHECK=1`` the allocator's footprint must equal ``mark`` (a call that has to leave the
        footprint as it found it: a snapshot save or a hit's restore, possibly inside a chunk whose transients are
        live) or, without one, the post-capture baseline (a quiescent point: a prefill row done, a release). A
        difference means a device buffer allocated under live traces persists (a tensor, or the kernel binaries of
        a program first compiled after the capture), which the next replay clobbers; the call site is named."""
        if not self._apc_alloc_checking or self._apc_alloc_baseline is None:
            return
        expected = self._apc_alloc_baseline if mark is None else mark
        now = self._dram_allocated()
        if now != expected:
            raise RuntimeError(
                f"GLM53 vLLM APC: the DRAM footprint changed after {where}: {now} vs {expected} bytes per bank "
                f"({'before the call' if mark is not None else 'at the decode capture'}): a device buffer allocated "
                "under live traces persists (a tensor, or the kernel binaries of a program first compiled after the "
                "capture) and the next replay clobbers it; warm the shape up before the capture or make the buffer "
                "a preallocated scratch (doc/serving/prefix_caching.md s.6)"
            )
        logger.info(f"GLM53 vLLM APC alloc check: unchanged after {where} ({now} bytes per bank)")

    def _perf(self, t0: float, live: int) -> None:
        self._perf_steps += 1
        self._perf_s += time.perf_counter() - t0
        if self._perf_steps % self._perf_log_every == 0 and self._perf_s > 0:
            ms = 1000 * self._perf_s / self._perf_steps
            logger.info(
                f"[glm53-vllm-perf] decode step {ms:.1f} ms = {1000 / ms:.2f} tok/s per user, "
                f"{live * 1000 / ms:.1f} tok/s aggregate over {live} live users ({self._perf_steps} steps, host wall incl. readback)"
            )
            if self.spec is not None:
                logger.info(
                    f"[glm53-vllm-spec] K={self.spec.K} commit {self.spec.commit_mode}: {self._spec_stats.summary()}"
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


# ===================================================================================== multimodal class
def _vllm_multimodal_api():
    """vLLM's multimodal declaration pieces, or None in a process without vLLM (the host tests, the model
    harness): ``SupportsMultiModal`` and the processor registration decorator for the GLM-5.3-Flash upstream
    processor (``vllm.models.glm5next``'s ``Glm5NextMultiModalProcessor`` / ``Glm5NextProcessingInfo`` with the
    Glm4v dummy-inputs builder, exactly what upstream's ``Glm5NextForConditionalGeneration`` registers)."""
    try:
        from vllm.model_executor.models.glm4_1v import Glm4vDummyInputsBuilder
        from vllm.model_executor.models.interfaces import SupportsMultiModal
        from vllm.models.glm5next.nvidia.multimodal import Glm5NextMultiModalProcessor, Glm5NextProcessingInfo
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError as exc:
        logger.debug(f"GLM53 vLLM: multimodal class declared without vLLM ({exc})")
        return None
    from types import SimpleNamespace

    return SimpleNamespace(
        SupportsMultiModal=SupportsMultiModal,
        register_processor=MULTIMODAL_REGISTRY.register_processor(
            Glm5NextMultiModalProcessor, info=Glm5NextProcessingInfo, dummy_inputs=Glm4vDummyInputsBuilder
        ),
    )


def declare_multimodal_class(api=None):
    """``GLM53FlashForConditionalGeneration``: the adapter served as text + image + video. A SEPARATE class rather
    than making ``GLM53FlashForCausalLM`` multimodal (the Gemma 4 arrangement): the plugin registers the text class
    for every Glm5Next architecture and its text-only-ness is load-bearing (without ``SupportsMultiModal`` vLLM leaves
    ``multimodal_config`` unpopulated, no processor sits in front of the request path, no per-row ``pixel_values``
    lists reach ``prefill_forward``). Selected per deployment through the generic ``TT_MODEL_CLASS_OVERRIDES``.

    ``api``: ``_vllm_multimodal_api()`` (vLLM present: the class gains ``SupportsMultiModal`` and the upstream
    ``Glm5NextMultiModalProcessor`` registration, which is what makes ``vllm serve`` accept ``image_url`` /
    ``video_url`` content) or None (a plain subclass with the same flags, for hosts without vLLM)."""
    bases = (GLM53FlashForCausalLM,) + ((api.SupportsMultiModal,) if api is not None else ())

    class GLM53FlashForConditionalGeneration(*bases):
        """GLM-5.3-Flash text + image + video generation on a Blackhole Galaxy mesh for the TT vLLM plugin."""

        supports_multimodal = True  # the SupportsMultiModal ClassVar, restated for the vLLM-less declaration
        supports_multimodal_raw_input_only = False
        supports_encoder_tp_data = False
        supports_mm_device_do_normalize = False
        requires_raw_input_tokens = False

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> str | None:
            """The chat template's media markers (upstream ``Glm4vForConditionalGeneration``): the processor
            expands ``<|image|>`` to the image tokens and ``<|video|>`` to the per-frame structure."""
            if modality.startswith("image"):
                return "<|begin_of_image|><|image|><|end_of_image|>"
            if modality.startswith("video"):
                return "<|begin_of_video|><|video|><|end_of_video|>"
            raise ValueError("Only image or video modality is supported")

        # vLLM nn.Module multimodal protocol shims (the TT runner hands the raw items to prefill_forward instead)
        def embed_multimodal(self, **kwargs):  # pragma: no cover - protocol shim
            raise NotImplementedError("TT models encode multimodal items inside prefill_forward")

        def get_language_model(self):  # pragma: no cover - protocol shim
            raise NotImplementedError("TT models expose no nn.Module language model")

    GLM53FlashForConditionalGeneration.__qualname__ = GLM53FlashForConditionalGeneration.__name__
    GLM53FlashForConditionalGeneration.__module__ = __name__
    if api is not None:
        GLM53FlashForConditionalGeneration = api.register_processor(GLM53FlashForConditionalGeneration)
    return GLM53FlashForConditionalGeneration


GLM53FlashForConditionalGeneration = declare_multimodal_class(_vllm_multimodal_api())
