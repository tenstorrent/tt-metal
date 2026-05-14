# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable harness for the HostIoDecoderStage multi-turn sweep.

This module provides the configuration, schedule, result, and entry-point types
used by the CLI driver (`run_host_io_decoder_sweep.py`). It deliberately knows
nothing about argparse — all CLI ergonomics live in the driver.

Multi-turn semantics (the only scheduling policy in this harness)
-----------------------------------------------------------------
Every prompt is run through the SAME set of replicated slots
(``[0, num_replication_slots)``) in sequence. Position ids accumulate across
prompts: prompt ``i`` occupies positions
``[sum(L_0..L_{i-1}), sum(L_0..L_{i}))`` in every slot's KV cache. Conceptually
this models a single user (slot) continuing a multi-turn conversation, fanned
out across ``num_replication_slots`` identical-state replicas for cross-slot
determinism validation.

The persistent decoder + H2D/D2H sockets are launched ONCE before the first
prompt and torn down ONCE after the last prompt. The on-device KV cache thus
holds the concatenated state of every prompt at the end of the sweep, and can
be sliced by prompt for downstream consumption.

Validation gates (every gate is independently toggleable in the config)
-----------------------------------------------------------------------
1. ``validate_metadata_roundtrip`` — per-iteration assertions that the H2D
   page's slot_id / position_id / token_id / token_0_type are preserved on the
   D2H tail (catches socket / multi-upstream-D2H corruption).
2. ``validate_hidden_states_cross_slot`` — per-prompt ``torch.equal`` of every
   replicated slot's collected output against slot 0 (catches slot-dependent
   decoder nondeterminism; no-op in Mode A).
3. ``validate_kv_cache_cross_slot`` — post-teardown per-prompt ``torch.equal``
   on the KV cache slice (slot s, prompt p's position range) against slot 0
   (same property, checked through the KV-cache path; no-op in Mode A).
4. ``validate_hidden_states_cross_trace`` — per-(prompt, slot) PCC of the
   collected output against the prompt's reference ``trace["output"]``
   (numerical-correctness check; threshold from ``pcc_threshold``).

Dumps (independently toggleable, both off by default suppresses all disk I/O)
----------------------------------------------------------------------------
- ``dump_hidden_states`` — one file per (slot, prompt):
  ``output_hidden_states_slot_<id:02d>_<prompt>.pt`` containing
  ``(L_p, HIDDEN_SIZE)`` bf16.
- ``dump_kv_cache`` — one file per (slot, prompt):
  ``kv_cache_slot_<id:02d>_<prompt>.pt`` containing
  ``(1, L_p, kvpe_dim)`` bf16. Sliced from the global on-device KV cache by
  the prompt's position range.

Public API surface
------------------
- :class:`HostIoDecoderSweepConfig` — frozen dataclass with every knob.
- :class:`MultiTurnSchedule` — derived position-id schedule (returned in
  :class:`SweepResult`; built internally by the harness).
- :class:`SweepResult` — outputs (collected hidden states + KV cache +
  schedule + loaded traces).
- :func:`run_sweep` — main entry point; takes config + parent MeshDevice.
- :func:`open_mesh_device` — context manager that opens / closes the parent
  MeshDevice and applies fabric configuration. Replaces the
  ``bh_2d_mesh_device`` pytest fixture.

Stubs / not yet implemented
---------------------------
The helpers, preflight, and ``run_sweep`` body are scoped to subsequent
checkpoints of the migration plan. See ``host_io_decoder_stage_context.txt``
(or the per-PR notes) for which checkpoint lands what.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.decoder_stage import HostIoDecoderStage
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, StageContext
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.model import InputField, TokenType, parse_output_page
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.utils import float_to_uint32
from models.demos.deepseek_v3_b1.weights.prepare import NUM_ROUTED_EXPERTS

# Env var names — referenced by both the harness (preflight) and the CLI driver
# (for argparse defaults). Single source of truth so we don't drift.
HIDDEN_STATES_DIR_ENV = "DEEPSEEK_V3_HIDDEN_STATES_DIR"
KV_CACHE_DUMP_DIR_ENV = "DEEPSEEK_V3_KV_CACHE_DUMP_DIR"

# Default paths for weight loading. Either can be overridden through
# ``HostIoDecoderSweepConfig.hf_model_path`` / ``cache_path`` (or, transitively,
# via CLI flags).
DEFAULT_HF_MODEL_PATH = Path("/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized")
DEFAULT_CACHE_PATH = Path.home() / ".cache"

# DeepSeek V3 architecture: layer indices [0, NUM_DENSE_LAYERS) are dense
# (attention + dense MLP), and indices [NUM_DENSE_LAYERS, num_layers) are MoE
# (attention + routed-experts MLP). Matches ``demo/pipeline.py`` which maps
# pipeline stages 1/2/3 -> dense layer ids and stages 4+ -> MoE layer ids.
# The b1 demo does not re-export a configuration constant for this, so it's
# inlined here. (See ``DeepSeekV3Config.NUM_DENSE_LAYERS`` in deepseek_v3_d_p
# for the same constant in the other code path.)
NUM_DENSE_LAYERS = 3


def _is_moe_layer(layer_idx: int) -> bool:
    """Return True iff ``layer_idx`` is a MoE (routed-experts) layer.

    Used by :func:`run_sweep` to auto-select between dense / MoE weight loaders
    and the matching ``HostIoDecoderStage`` kwargs without forcing the caller
    to pass a redundant stage-type flag.
    """
    return layer_idx >= NUM_DENSE_LAYERS


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiTurnSchedule:
    """Position-id schedule for a multi-turn sweep.

    Each prompt occupies a disjoint, contiguous, monotonically-incrementing
    block of positions in every replicated slot's KV cache. Built by the
    harness's preflight step from the reference traces' ``"input"`` shapes.

    Attributes:
        prompt_lengths: Per-prompt sequence length L_i, parallel to
            ``HostIoDecoderSweepConfig.prompt_names`` (same order, same length).
    """

    prompt_lengths: tuple[int, ...]

    def offset_for(self, prompt_idx: int) -> int:
        """Position offset where prompt ``prompt_idx`` starts."""
        return sum(self.prompt_lengths[:prompt_idx])

    def range_for(self, prompt_idx: int) -> tuple[int, int]:
        """Half-open ``(start, end)`` position range for prompt ``prompt_idx``."""
        start = self.offset_for(prompt_idx)
        return (start, start + self.prompt_lengths[prompt_idx])

    def total_length(self) -> int:
        """Total positions consumed across all prompts (= ``sum(prompt_lengths)``)."""
        return sum(self.prompt_lengths)


@dataclass(frozen=True)
class HostIoDecoderSweepConfig:
    """All knobs for one ``run_sweep`` invocation.

    Field validation in ``__post_init__`` covers STATIC config invariants only.
    Trace-shape-dependent invariants (e.g. ``total_length() + 1 < max_seq_len``)
    and filesystem-existence checks are validated by the harness's preflight
    step once the traces are loaded.
    """

    # --- required: prompt source + identifying knobs ---
    # Positions are disjoint and incrementing in ``prompt_names`` order. Single
    # prompt is allowed; multi-prompt fans out into a multi-turn conversation
    # across the shared slot range.
    decoder_layer_idx: int
    hidden_states_dir: Path
    prompt_names: tuple[str, ...]

    # --- model shape ---
    max_seq_len: int = 128 * 1024
    num_slots: int = 64
    mesh_rows: int = 4
    mesh_cols: int = 2

    # --- replication / mode ---
    num_replication_slots: int = 8  # 1 = Mode A (single slot); >1 = Mode B (replicated)

    # --- validation knobs ---
    validate_metadata_roundtrip: bool = True
    validate_hidden_states_cross_slot: bool = True
    validate_kv_cache_cross_slot: bool = True
    validate_hidden_states_cross_trace: bool = False
    pcc_threshold: float = 0.97  # only consulted when validate_hidden_states_cross_trace is True

    # --- dump knobs ---
    # Disk dumps are OPT-IN: defaults are False so a validation-only run (or a
    # pure sweep) does not require ``--dump-dir`` and does not pay the cost of
    # the device-to-host KV cache pull. Set explicitly when you want artifacts.
    dump_hidden_states: bool = False
    dump_kv_cache: bool = False
    dump_dir: Path | None = None  # required if any dump knob is True

    # --- weights ---
    hf_model_path: Path = DEFAULT_HF_MODEL_PATH
    cache_path: Path = DEFAULT_CACHE_PATH

    # --- misc ---
    seed: int = 0
    log_per_iteration: bool = False

    def __post_init__(self) -> None:
        """Validate static invariants. Trace-dependent checks live in preflight."""
        if self.decoder_layer_idx < 0:
            raise ValueError(f"decoder_layer_idx must be >= 0, got {self.decoder_layer_idx}")
        if not self.prompt_names:
            raise ValueError("prompt_names must contain at least one entry")
        for name in self.prompt_names:
            if not isinstance(name, str) or not name:
                raise ValueError(f"prompt_names must be non-empty strings; got {self.prompt_names!r}")
        if self.num_replication_slots < 1:
            raise ValueError(f"num_replication_slots must be >= 1, got {self.num_replication_slots}")
        # num_slots must leave room for the termination dummy at slot=num_slots-1,
        # so num_replication_slots is strictly less than num_slots.
        if self.num_slots <= self.num_replication_slots:
            raise ValueError(
                f"num_slots ({self.num_slots}) must be > num_replication_slots "
                f"({self.num_replication_slots}) to reserve slot={self.num_slots - 1} "
                f"for the termination dummy"
            )
        if self.max_seq_len <= 1:
            raise ValueError(f"max_seq_len must be > 1, got {self.max_seq_len}")
        # mesh_rows and mesh_cols must each be >= 2 because run_sweep's pipeline_config
        # hard-codes entry_node_coord=MeshCoordinate(1, 0) and exit_node_coord=
        # MeshCoordinate(1, 1) (matching test_decoder_block.test_decoder's broadcast
        # source / reduce-to-one root layout). Smaller submeshes would trip out-of-bounds
        # coord errors at submesh construction.
        if self.mesh_rows < 2 or self.mesh_cols < 2:
            raise ValueError(
                f"mesh_rows and mesh_cols must each be >= 2 (the harness's single-stage "
                f"pipeline uses fixed entry=(1,0) / exit=(1,1) coordinates); got "
                f"({self.mesh_rows}, {self.mesh_cols})"
            )
        if not (0.0 < self.pcc_threshold <= 1.0):
            raise ValueError(f"pcc_threshold must be in (0, 1]; got {self.pcc_threshold}")
        if (self.dump_hidden_states or self.dump_kv_cache) and self.dump_dir is None:
            raise ValueError("dump_dir is required when dump_hidden_states or dump_kv_cache is True")


@dataclass
class SweepResult:
    """Outputs of one ``run_sweep`` invocation.

    Attributes:
        collected: Nested dict keyed by ``prompt_name`` -> ``slot_id`` ->
            ``(L_p, HIDDEN_SIZE)`` bf16 tensor of collected decoder outputs.
        kv_cache: Composed host KV cache tensor of shape
            ``(num_slots, 1, max_seq_len, kvpe_dim)`` dtype bfloat16, or
            ``None`` when the device-to-host pull was skipped because neither
            KV-cache validation nor KV-cache dump was requested.
        schedule: The :class:`MultiTurnSchedule` actually used.
        traces: Reference traces as loaded from disk; outer key is
            ``prompt_name``, inner keys are ``"input"`` / ``"output"`` -> bf16
            tensor of shape ``(L_p, HIDDEN_SIZE)``.
    """

    collected: dict[str, dict[int, torch.Tensor]]
    kv_cache: torch.Tensor | None
    schedule: MultiTurnSchedule
    traces: dict[str, dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _SinglePipelineStage:
    """Duck-typed stand-in for ``ttnn.experimental.BlitzDecodePipelineStage``.

    ``generate_blitz_decode_pipeline`` does not handle ``num_meshes == 1`` (it
    OOBs on ``hops[0]`` for the no-loopback path) and the C++
    ``BlitzDecodePipelineStage`` binding is read-only, so we construct the
    1-stage descriptor in Python. ``HostIoDecoderStage`` and
    ``PipelineBlock._init_combined_h2d_d2h_stage`` only read
    ``entry_node_coord`` and ``exit_node_coord`` from this object.
    """

    entry_node_coord: ttnn.MeshCoordinate
    exit_node_coord: ttnn.MeshCoordinate


# ---------------------------------------------------------------------------
# Trace I/O + H2D page construction + D2H page parsing (ported from the
# original pytest test_host_io_decoder_stage.py; semantics unchanged except
# for ``_build_per_iteration_input`` which now plumbs a single ``global_pos``).
# ---------------------------------------------------------------------------


def _load_reference_trace(trace_dir: Path, prompt_name: str) -> dict[str, torch.Tensor]:
    """Load and validate one prompt's reference trace.

    Expected on-disk format (per prompt):
        torch.save(
            {"input":  (L, HIDDEN_SIZE) bf16,
             "output": (L, HIDDEN_SIZE) bf16},
            f"{trace_dir}/{prompt_name}.pt",
        )

    Returns the dict unchanged. Raises ``FileNotFoundError`` / ``ValueError`` if
    the file's structure / shapes / dtypes don't match the contract.
    """
    path = trace_dir / f"{prompt_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Reference trace not found: {path}")
    # map_location="cpu" so traces saved on a CUDA host still load on a CPU host
    # (the harness only consumes the tensors on host before pushing to ttnn).
    trace = torch.load(path, map_location="cpu")
    if not isinstance(trace, dict):
        raise ValueError(f"{path}: expected dict, got {type(trace).__name__}")
    if not set(trace.keys()) >= {"input", "output"}:
        raise ValueError(f"{path}: missing required keys 'input'/'output'; got {sorted(trace.keys())}")
    inp, out = trace["input"], trace["output"]
    if not (isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor)):
        raise ValueError(f"{path}: 'input' and 'output' must be torch.Tensor")
    if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"{path}: expected bfloat16 tensors, got input={inp.dtype}, output={out.dtype}")
    if inp.ndim != 2 or out.ndim != 2:
        raise ValueError(
            f"{path}: expected 2D (seq_len, HIDDEN_SIZE) tensors, got input.shape={tuple(inp.shape)}, "
            f"output.shape={tuple(out.shape)}"
        )
    if inp.shape[-1] != D.HIDDEN_SIZE or out.shape[-1] != D.HIDDEN_SIZE:
        raise ValueError(
            f"{path}: last dim must equal D.HIDDEN_SIZE ({D.HIDDEN_SIZE}), got "
            f"input.shape={tuple(inp.shape)}, output.shape={tuple(out.shape)}"
        )
    if inp.shape[0] != out.shape[0]:
        raise ValueError(f"{path}: input and output seq_len mismatch: input={inp.shape[0]}, output={out.shape[0]}")
    return trace


def _to_hidden_state_input(
    hidden_state: torch.Tensor,
    *,
    token_id: int,
    prefill_token_id: int,
    user_id: int,
    position_id: int,
    token_type: int,
    temperature: float,
    top_k: int,
    probability_mass_threshold: float,
) -> ttnn.Tensor:
    """Build an H2D passthrough page = ``hidden_state || DeepseekMetadata`` (uint32-typed).

    Used with ``HostIoDecoderStage(inject_hidden_states=True)``. The activation
    half is the bf16 hidden state reinterpreted as int32 words; the metadata
    half mirrors ``model.to_spec_input``'s field placement so the decoder +
    multi-upstream D2H tail round-trip the input metadata fields the same way
    as the embedding-driven path.
    """
    assert hidden_state.dtype == torch.bfloat16, f"hidden_state must be bf16, got {hidden_state.dtype}"
    assert (
        hidden_state.numel() == D.HIDDEN_SIZE
    ), f"hidden_state must have {D.HIDDEN_SIZE} elements, got {hidden_state.numel()}"

    hidden_int32 = hidden_state.contiguous().view(torch.int32)  # (HIDDEN_SIZE / 2,) int32

    metadata_words = torch.zeros(DeepseekMetadata.aligned_size_bytes() // 4, dtype=torch.int32)
    metadata_words[InputField.TOKEN_ID] = token_id
    metadata_words[InputField.PREFILL_TOKEN_ID] = prefill_token_id
    metadata_words[InputField.TOKEN_TYPE] = token_type
    metadata_words[InputField.USER_ID] = user_id
    metadata_words[InputField.POSITION_ID] = position_id
    metadata_words[InputField.TOKEN0_POSITION_ID] = position_id
    metadata_words[InputField.TEMPERATURE] = float_to_uint32(temperature)
    metadata_words[InputField.TOP_K] = top_k
    metadata_words[InputField.PROBABILITY_MASS_THRESHOLD] = float_to_uint32(probability_mass_threshold)

    combined = torch.cat([hidden_int32, metadata_words]).reshape(1, -1)
    return ttnn.from_torch(combined, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _build_per_iteration_input(
    hidden_state: torch.Tensor,
    *,
    slot_id: int,
    global_pos: int,
) -> ttnn.Tensor:
    """Build a single H2D inject-mode page for one (slot, global_pos) sweep iteration.

    Thin wrapper over :func:`_to_hidden_state_input` that fixes the canonical
    decode-step defaults (matching ``demo/model_pipeline.py:_write_spec_pair``):
    ``prefill_token_id=-1``, ``token_type=TokenType.BASE``, ``temperature=0.6``,
    ``top_k=1``, ``probability_mass_threshold=1.0``.

    In multi-turn mode the same ``global_pos`` is written to ``token_id``,
    ``position_id`` (which also propagates to ``token0_pos``). This keeps the
    metadata round-trip assertions a single source of truth: the global
    position is the only "address" the decoder sees, no trace-local indices.
    """
    return _to_hidden_state_input(
        hidden_state,
        token_id=global_pos,
        prefill_token_id=-1,
        user_id=slot_id,
        position_id=global_pos,
        token_type=TokenType.BASE,
        temperature=0.6,
        top_k=1,
        probability_mass_threshold=1.0,
    )


def _extract_activation_from_d2h(output_tensor: ttnn.Tensor) -> torch.Tensor:
    """Return the activation half (first HIDDEN_SIZE bf16 elements) of a D2H page.

    The D2H page layout is ``activation (HIDDEN_SIZE bf16) || DeepseekMetadata
    (256 bytes)``. The leading slice IS the decoder's output hidden state for
    the corresponding ``(slot, global_pos)``.
    """
    return ttnn.to_torch(output_tensor).flatten()[: D.HIDDEN_SIZE]


def _extract_metadata_from_d2h(output_tensor: ttnn.Tensor):
    """Extract and parse the trailing DeepseekMetadata tail of a D2H page.

    Slices the last ``DeepseekMetadata.aligned_size_bytes()`` bytes off the
    bf16 receive buffer, reinterprets them bitwise as ``(64,) int32`` words,
    and parses via ``model.parse_output_page``.

    Returns:
        metadata_flat: torch.Tensor of shape ``(64,)`` int32 — raw idx reads
            (e.g. ``metadata_flat[InputField.POSITION_ID]`` for the input-side
            position_id field, which ``parse_output_page`` does not expose).
        parsed: ``DecodeResult`` — convenience accessors for ``slot_id`` and
            the ``token_0_*`` output fields.
    """
    metadata_bytes = DeepseekMetadata.aligned_size_bytes()  # 256
    metadata_uint32_count = metadata_bytes // 4  # 64
    metadata_bf16_count = metadata_bytes // dtype_size(ttnn.bfloat16)  # 128

    torch_full = ttnn.to_torch(output_tensor).flatten()
    metadata_words = torch_full[-metadata_bf16_count:].contiguous().view(torch.int32).reshape(1, metadata_uint32_count)
    metadata_tensor = ttnn.from_torch(metadata_words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    parsed = parse_output_page(metadata_tensor)
    return metadata_words.flatten(), parsed


# ---------------------------------------------------------------------------
# Preflight: trace loading + trace-dependent invariant validation
# ---------------------------------------------------------------------------


def _preflight(
    config: HostIoDecoderSweepConfig,
) -> tuple[MultiTurnSchedule, dict[str, dict[str, torch.Tensor]]]:
    """Load all reference traces and build the multi-turn schedule.

    Validates trace-dependent invariants that ``HostIoDecoderSweepConfig.__post_init__``
    can't catch: filesystem existence of ``hidden_states_dir`` and each prompt
    file, per-trace shape/dtype conformance (delegated to
    :func:`_load_reference_trace`), and the cumulative-position constraint
    ``total_length() + 1 < max_seq_len`` (the ``+1`` reserves the
    ``(slot=num_slots-1, pos=max_seq_len-1)`` corner for the termination
    dummy in :func:`run_sweep`'s phase 1).

    Returns:
        ``(schedule, traces)`` where ``traces`` is keyed by prompt_name and
        each value is the dict returned by :func:`_load_reference_trace`.

    Raises:
        ValueError: any trace-dependent invariant violated.
        AssertionError: per-trace shape/dtype contract violated (from
            :func:`_load_reference_trace`).
    """
    if not config.hidden_states_dir.is_dir():
        raise ValueError(f"hidden_states_dir does not exist or is not a directory: {config.hidden_states_dir}")

    traces: dict[str, dict[str, torch.Tensor]] = {}
    prompt_lengths: list[int] = []
    for prompt_name in config.prompt_names:
        trace = _load_reference_trace(config.hidden_states_dir, prompt_name)
        traces[prompt_name] = trace
        prompt_lengths.append(int(trace["input"].shape[0]))

    schedule = MultiTurnSchedule(prompt_lengths=tuple(prompt_lengths))

    total = schedule.total_length()
    if total + 1 >= config.max_seq_len:
        raise ValueError(
            f"Total positions consumed across all prompts ({total}) + 1 (termination dummy) "
            f">= max_seq_len ({config.max_seq_len}). Reduce prompt seq_lens or increase max_seq_len."
        )

    return schedule, traces


# ---------------------------------------------------------------------------
# Per-prompt inner sweep (used by run_sweep's prompt loop)
# ---------------------------------------------------------------------------


def _run_prompt_sweep(
    *,
    config: HostIoDecoderSweepConfig,
    pipeline_block,
    prompt_name: str,
    prompt_idx: int,
    schedule: MultiTurnSchedule,
    trace: dict[str, torch.Tensor],
    collected_for_prompt: dict[int, torch.Tensor],
    output_tensor: ttnn.Tensor,
) -> None:
    """Run one prompt's H2D / decoder / D2H sweep, writing into the supplied collector.

    Each ``(p_local, slot_id)`` iteration writes
    ``hidden_state || DeepseekMetadata`` to H2D, reads the D2H reply, optionally
    asserts the input metadata round-trips, and stashes the activation half in
    ``collected_for_prompt[slot_id][p_local, :]``. ``global_pos`` is the
    schedule offset plus ``p_local`` and is the only "position" the decoder
    sees.

    Both the collector and the ``output_tensor`` D2H receive buffer are
    allocated by the caller (in ``run_sweep``'s phase 0.5) so the sweep loop
    only writes/reads/stores, with no allocation work mixed in. ``output_tensor``
    is overwritten every iteration by ``read_output``; the per-iter activation /
    metadata extractors copy the bytes out to fresh CPU tensors so reuse is safe.

    Heartbeat logging is every ``log_every`` iterations of the outer p_local
    loop (1 if ``config.log_per_iteration`` is True, else 1000).
    """
    L_p = schedule.prompt_lengths[prompt_idx]
    offset = schedule.offset_for(prompt_idx)
    assert set(collected_for_prompt.keys()) == set(range(config.num_replication_slots)), (
        f"collected_for_prompt keys mismatch: expected {set(range(config.num_replication_slots))}, "
        f"got {set(collected_for_prompt.keys())}"
    )
    for slot, tensor in collected_for_prompt.items():
        assert tensor.shape == (L_p, D.HIDDEN_SIZE) and tensor.dtype == torch.bfloat16, (
            f"collected_for_prompt[{slot}] must be ({L_p}, {D.HIDDEN_SIZE}) bf16; "
            f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )
    log_every = 1 if config.log_per_iteration else 1000
    logger.info(
        f"sweep prompt={prompt_name!r} (idx={prompt_idx}): L_p={L_p} "
        f"global_pos_range=[{offset}, {offset + L_p}) "
        f"replication_slots={config.num_replication_slots}"
    )
    for p_local in range(L_p):
        global_pos = offset + p_local
        hidden_state = trace["input"][p_local]
        for slot_id in range(config.num_replication_slots):
            input_tensor = _build_per_iteration_input(hidden_state, slot_id=slot_id, global_pos=global_pos)
            pipeline_block.write_token(input_tensor)

            pipeline_block.read_output(output_tensor)

            if config.validate_metadata_roundtrip:
                metadata_flat, parsed = _extract_metadata_from_d2h(output_tensor)
                actual_position_id = int(metadata_flat[InputField.POSITION_ID].item())
                actual_token_id = int(metadata_flat[InputField.TOKEN_ID].item())
                assert parsed.slot_id == slot_id, (
                    f"prompt={prompt_name!r} slot={slot_id} global_pos={global_pos}: "
                    f"slot_id round-trip mismatch (got {parsed.slot_id})"
                )
                assert actual_position_id == global_pos, (
                    f"prompt={prompt_name!r} slot={slot_id} global_pos={global_pos}: "
                    f"position_id round-trip mismatch (got {actual_position_id})"
                )
                assert actual_token_id == global_pos, (
                    f"prompt={prompt_name!r} slot={slot_id} global_pos={global_pos}: "
                    f"token_id round-trip mismatch (got {actual_token_id})"
                )
                assert parsed.token_0_type == TokenType.BASE, (
                    f"prompt={prompt_name!r} slot={slot_id} global_pos={global_pos}: "
                    f"token_0_type round-trip mismatch (got {parsed.token_0_type})"
                )

            collected_for_prompt[slot_id][p_local, :] = _extract_activation_from_d2h(output_tensor)

        if (p_local + 1) % log_every == 0 or p_local + 1 == L_p:
            logger.info(f"sweep prompt={prompt_name!r}: {p_local + 1}/{L_p} positions complete")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def open_mesh_device(device_params: dict) -> Iterator[ttnn.MeshDevice]:
    """Open the parent ``MeshDevice`` and yield it; close on exit.

    Thin wrapper around :func:`conftest.bh_2d_mesh_device_context` so the
    fabric configuration / open / close lifecycle matches the production
    pytest path exactly. The parent mesh shape is determined by the system
    topology (1, 2, 4, 8, or 32 visible devices); the submesh of the shape
    requested by ``HostIoDecoderSweepConfig.mesh_rows / mesh_cols`` is
    created inside :func:`run_sweep`.

    Args:
        device_params: Fabric / worker-l1 / fabric-router config dict; same
            shape as the indirect ``device_params`` pytest fixture today (i.e.
            ``{"fabric_config": ..., "fabric_router_config": ...,
            "worker_l1_size": ...}``).

    Yields:
        The parent ``MeshDevice`` handle.

    Raises:
        ImportError: ``conftest.bh_2d_mesh_device_context`` is not importable
            (workspace root not on ``sys.path`` and/or ``pytest`` not installed).
            The harness module's other public types (dataclasses, ``run_sweep``,
            helpers) are importable independently of pytest; only this context
            manager pulls pytest in transitively.
    """
    # Imported lazily so the harness MODULE is importable without pytest on
    # PYTHONPATH (e.g. for tooling that only consumes the dataclasses). Calling
    # this context manager DOES require pytest to be installed and the
    # workspace root to be on sys.path, because conftest.py itself imports
    # pytest at module load. Raise an actionable error if either is missing.
    try:
        from conftest import bh_2d_mesh_device_context
    except ImportError as e:
        raise ImportError(
            "open_mesh_device requires conftest.bh_2d_mesh_device_context, which lives "
            "at the tt-metal repo root and imports pytest at module load. Run from the "
            "repo root (so conftest.py is on sys.path) and ensure pytest is installed."
        ) from e

    with bh_2d_mesh_device_context(device_params) as parent_mesh:
        yield parent_mesh


def run_sweep(
    config: HostIoDecoderSweepConfig,
    parent_mesh: ttnn.MeshDevice,
) -> SweepResult:
    """Run one multi-turn HostIoDecoderStage sweep end-to-end.

    Phases (see module docstring for full semantics):
        0.  Preflight: load traces, build :class:`MultiTurnSchedule`, validate
            trace-dependent invariants. Pre-allocate every prompt's per-slot
            ``collected`` tensors so the sweep loop has zero allocation work.
        0.5 Submesh + slow-dispatch + weights + HostIoDecoderStage + persistent
            kernels launch (runs ONCE for the whole multi-turn run).
        1.  Multi-turn prompt loop: per-prompt sweep into pre-allocated
            collectors. No validation in this phase — keeps the hot path
            (write/read/round-trip/store) free of any per-prompt host work
            other than the per-iter metadata round-trip asserts.
        2.  Termination (single teardown).
        3.  Hidden state validation (post-teardown):
            3a. Per-prompt cross-slot ``torch.equal`` across replicated slots.
            3b. Optional per-(prompt, slot) cross-trace PCC vs reference.
        4.  KV cache validation: conditional pull (fast-dispatch) + per-prompt
            KV cross-slot equality.
        5.  Per-(prompt, slot) dumps for hidden states and KV cache.

    Args:
        config: Frozen configuration. Validate-only and dump-only invocations
            are supported by toggling the relevant knobs.
        parent_mesh: Parent ``MeshDevice`` owned by the caller (e.g. by
            :func:`open_mesh_device`); not closed by ``run_sweep``.

    Returns:
        :class:`SweepResult` with collected outputs, full KV cache, schedule,
        and loaded reference traces.

    Raises:
        AssertionError: any enabled validation gate that fires.
        ValueError: preflight invariants violated (trace shape, sum of seq lens
            vs max_seq_len, missing dump_dir while a dump knob is set, etc.).
        RuntimeError: not invoked under slow dispatch (sets the env var
            ``TT_METAL_SLOW_DISPATCH_MODE=1`` upstream).
    """
    # =========================================================================
    # Phase 0: Preflight + environment checks
    # =========================================================================
    if not is_slow_dispatch():
        raise RuntimeError(
            "run_sweep requires slow dispatch (the H2D / D2H sockets do not work under fast "
            "dispatch). Set TT_METAL_SLOW_DISPATCH_MODE=1 before launching."
        )
    if not config.hf_model_path.exists():
        raise FileNotFoundError(f"HF model path does not exist: {config.hf_model_path}")

    torch.manual_seed(config.seed)
    schedule, traces = _preflight(config)
    logger.info(
        f"preflight: prompts={list(config.prompt_names)} "
        f"prompt_lengths={schedule.prompt_lengths} total_length={schedule.total_length()} "
        f"max_seq_len={config.max_seq_len}"
    )

    num_devices = config.mesh_rows * config.mesh_cols
    if parent_mesh.shape[0] * parent_mesh.shape[1] < num_devices:
        raise RuntimeError(
            f"parent_mesh has {parent_mesh.shape[0] * parent_mesh.shape[1]} devices but "
            f"config requires {num_devices} ({config.mesh_rows}x{config.mesh_cols})"
        )

    if config.dump_dir is not None:
        config.dump_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Phase 0.5: One-time setup (submesh, weights, persistent kernels)
    # =========================================================================
    logger.info(f"Creating {config.mesh_rows}x{config.mesh_cols} submesh ({num_devices} devices)")
    submesh = parent_mesh.create_submesh(ttnn.MeshShape((config.mesh_rows, config.mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    logger.info(f"Using HF model path: {config.hf_model_path}")
    logger.info(f"Using cache path:    {config.cache_path}")
    provider = CacheWeightProvider(cache_path=config.cache_path, model_path=config.hf_model_path)

    # Auto-detect dense vs MoE from layer_idx (see NUM_DENSE_LAYERS comment at the
    # top of this module). Dense layers use a different weight loader and pass
    # different MoE-related kwargs into HostIoDecoderStage; everything else about
    # the harness (broadcast / reduce / sockets / KV cache) is layer-type agnostic.
    is_moe_layer = _is_moe_layer(config.decoder_layer_idx)
    layer_type = "MoE" if is_moe_layer else "dense"
    logger.info(
        f"Layer type auto-detected: layer_idx={config.decoder_layer_idx} -> {layer_type} "
        f"(NUM_DENSE_LAYERS={NUM_DENSE_LAYERS})"
    )
    if is_moe_layer:
        layer_weights = provider.load_moe_layer(layer_id=config.decoder_layer_idx, device=submesh)
    else:
        layer_weights = provider.load_dense_layer(layer_id=config.decoder_layer_idx, device=submesh)
    logger.info(f"{layer_type} layer weights ready")

    # Single-stage pipeline (num_procs == 1, no loopback). Coords match
    # the original test_decoder_block.test_decoder layout:
    # broadcast source = (1, 0), reduce-to-one root = (1, 1).
    pipeline_config = [
        _SinglePipelineStage(
            entry_node_coord=ttnn.MeshCoordinate(1, 0),
            exit_node_coord=ttnn.MeshCoordinate(1, 1),
        )
    ]
    stages_metadata = {0: StageMetadata(rank=0, mesh_id=0)}
    ctx = StageContext(
        mesh_device=submesh,
        pipeline_config=pipeline_config,
        my_stage_idx=0,
        stages_metadata=stages_metadata,
    )

    # Production-shape decoder: is_torus=True, persistent_mode=True,
    # use_hardcoded_expert_index=False. MoE layers also set is_moe=True with
    # num_routed_experts=NUM_ROUTED_EXPERTS (256) and enable_routing=True; dense
    # layers set those to False/0/False. position_id=0 seeds the initial KV cache
    # state; runtime per-token positions come from each H2D page (global_pos in
    # multi-turn).
    if is_moe_layer:
        layer_type_kwargs = dict(
            is_moe=True,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            enable_routing=True,
        )
    else:
        layer_type_kwargs = dict(
            is_moe=False,
            num_routed_experts=0,
            enable_routing=False,
        )
    logger.info(
        f"Instantiating HostIoDecoderStage (layer_idx={config.decoder_layer_idx}, "
        f"layer_type={layer_type}, layer_type_kwargs={layer_type_kwargs})"
    )
    stage = HostIoDecoderStage(
        weights=layer_weights,
        layer_idx=config.decoder_layer_idx,
        metadata=DeepseekMetadata(position_id=0),
        max_seq_len=config.max_seq_len,
        num_slots=config.num_slots,
        persistent_mode=True,
        is_torus=True,
        use_hardcoded_expert_index=False,
        inject_hidden_states=True,
        **layer_type_kwargs,
    )

    logger.info("Building PipelineBlock + persistent kernels")
    pipeline_block = stage.create_pipeline_block(ctx)
    stage.setup(ctx, pipeline_block)
    pipeline_block.run()
    stage.launch_compute(ctx, pipeline_block)
    logger.info("All persistent kernels dispatched")

    out_words = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES // dtype_size(ttnn.bfloat16)

    # Pre-allocate ALL per-prompt collectors AND the shared D2H receive buffer
    # before any H2D write. Keeps the multi-turn sweep loop tight: only
    # writes/reads/stores live in the hot path, no allocation work between
    # iterations.
    #
    # output_tensor is reused across every (prompt, position, slot) iteration:
    # read_output overwrites it in place, and the activation/metadata extractors
    # copy the bytes out to fresh CPU tensors before the next read_output runs.
    collected: dict[str, dict[int, torch.Tensor]] = {
        prompt_name: {
            slot: torch.zeros(schedule.prompt_lengths[prompt_idx], D.HIDDEN_SIZE, dtype=torch.bfloat16)
            for slot in range(config.num_replication_slots)
        }
        for prompt_idx, prompt_name in enumerate(config.prompt_names)
    }
    output_tensor = ttnn.from_torch(
        torch.zeros(1, out_words, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    logger.info(
        f"Pre-allocated collected hidden states for {len(config.prompt_names)} prompt(s) x "
        f"{config.num_replication_slots} slot(s); reusing one D2H receive buffer "
        f"(out_words={out_words}) across all iterations"
    )

    # =========================================================================
    # Phase 1: Multi-turn prompt loop (sweep only — no validation here)
    # =========================================================================
    for prompt_idx, prompt_name in enumerate(config.prompt_names):
        _run_prompt_sweep(
            config=config,
            pipeline_block=pipeline_block,
            prompt_name=prompt_name,
            prompt_idx=prompt_idx,
            schedule=schedule,
            trace=traces[prompt_name],
            collected_for_prompt=collected[prompt_name],
            output_tensor=output_tensor,
        )

    # =========================================================================
    # Phase 2: Termination (single teardown for the entire multi-turn run)
    # =========================================================================
    logger.info("Phase 2: signalling persistent decoder termination")
    stage.terminate(ctx, pipeline_block)

    # Termination dummy targets (slot=num_slots-1, position=max_seq_len-1).
    # Preflight asserts schedule.total_length() + 1 < max_seq_len, so the dummy
    # never lands on a used position. __post_init__ asserts num_slots >
    # num_replication_slots, so it never lands on a used slot.
    zero_hidden_state = torch.zeros(D.HIDDEN_SIZE, dtype=torch.bfloat16)
    termination_dummy = _to_hidden_state_input(
        zero_hidden_state,
        token_id=0,
        prefill_token_id=-1,
        user_id=config.num_slots - 1,
        position_id=config.max_seq_len - 1,
        token_type=TokenType.BASE,
        temperature=0.6,
        top_k=1,
        probability_mass_threshold=1.0,
    )
    logger.info(f"Pushing termination dummy at slot={config.num_slots - 1} pos={config.max_seq_len - 1}")
    pipeline_block.write_token(termination_dummy)
    pipeline_block.drain_dummy_output()
    pipeline_block.terminate()
    ttnn.synchronize_device(submesh)
    logger.info("Phase 2 complete: HostIoDecoderStage teardown done")

    # =========================================================================
    # Phase 3: Hidden state validation (cross-slot equality + cross-trace PCC)
    # =========================================================================
    # Per-prompt cross-slot equality. The same hidden state is pushed through
    # every replicated slot at each position, so each slot's collected output
    # must be byte-identical for every prompt. (No-op for Mode A.)
    if config.validate_hidden_states_cross_slot and config.num_replication_slots > 1:
        logger.info(
            f"Phase 3a: hidden-state cross-slot equality across "
            f"{config.num_replication_slots} slots for {len(config.prompt_names)} prompt(s)"
        )
        for prompt_name in config.prompt_names:
            ref = collected[prompt_name][0]
            for s in range(1, config.num_replication_slots):
                assert torch.equal(collected[prompt_name][s], ref), (
                    f"Hidden-state cross-slot equality failed: prompt={prompt_name!r} " f"slot {s} diverged from slot 0"
                )
            logger.info(f"prompt={prompt_name!r}: hidden-state cross-slot equality OK")
    else:
        logger.info("Phase 3a skipped: validate_hidden_states_cross_slot=False or num_replication_slots == 1")

    # Cross-trace per-(prompt, slot) PCC against the reference output trace.
    # If 3a already proved cross-slot equality (i.e. all slots byte-identical to
    # slot 0), PCC of slot 0 is a sound proxy for every slot — skip the redundant
    # PCC computations for slots 1..N-1. Otherwise (Mode A, or 3a disabled) we
    # have no proof of cross-slot equality and must PCC each slot independently.
    if config.validate_hidden_states_cross_trace:
        cross_slot_proven = config.validate_hidden_states_cross_slot and config.num_replication_slots > 1
        slots_to_pcc: list[int] = [0] if cross_slot_proven else list(range(config.num_replication_slots))
        logger.info(
            f"Phase 3b: cross-trace PCC validation (threshold={config.pcc_threshold}) "
            f"for {len(config.prompt_names)} prompt(s); slots_to_pcc={slots_to_pcc} "
            f"(cross_slot_proven_in_3a={cross_slot_proven})"
        )
        for prompt_name in config.prompt_names:
            expected = traces[prompt_name]["output"]
            for slot in slots_to_pcc:
                passing, pcc = comp_pcc(
                    expected.flatten(),
                    collected[prompt_name][slot].flatten(),
                    config.pcc_threshold,
                )
                slot_label = (
                    f"slot={slot} (proxy for all {config.num_replication_slots} slots)"
                    if cross_slot_proven
                    else f"slot={slot}"
                )
                logger.info(
                    f"PCC: prompt={prompt_name!r} {slot_label} pcc={float(pcc):.6f} "
                    f"threshold={config.pcc_threshold} pass={passing}"
                )
                assert passing, (
                    f"Cross-trace validation FAILED: prompt={prompt_name!r} slot={slot} "
                    f"pcc={float(pcc):.6f} < threshold {config.pcc_threshold}"
                )
        logger.info("Phase 3b complete: cross-trace PCC OK for all prompts")
    else:
        logger.info("Phase 3b skipped: validate_hidden_states_cross_trace=False")

    # =========================================================================
    # Phase 4: KV cache pull (fast-dispatch) + per-prompt cross-slot validation
    # =========================================================================
    # Skip the device-to-host KV pull entirely if neither validation nor dump
    # needs it. The pull composes a ~9.6 GB host tensor (production shape) so
    # this is a meaningful win for validation-only or pure-sweep invocations.
    # Note: cross-slot validation is itself a no-op in Mode A (single slot), so
    # it can't justify pulling on its own when num_replication_slots == 1.
    kv_cache_torch: torch.Tensor | None = None
    need_kv_cache = config.dump_kv_cache or (config.validate_kv_cache_cross_slot and config.num_replication_slots > 1)
    if need_kv_cache:
        logger.info("Phase 4: pulling on-device KV cache to host (fast-dispatch context)")
        with ttnn.device.setup_fast_dispatch(submesh):
            kv_cache_torch = stage.get_kv_cache_host()
        assert kv_cache_torch is not None, "get_kv_cache_host returned None (setup not completed?)"
        logger.info(f"KV cache shape={tuple(kv_cache_torch.shape)} dtype={kv_cache_torch.dtype}")
    else:
        logger.info("Phase 4: KV cache pull skipped (no KV-cache validation, no KV-cache dump)")

    if config.validate_kv_cache_cross_slot and config.num_replication_slots > 1:
        assert kv_cache_torch is not None  # guaranteed by need_kv_cache above
        logger.info(f"Phase 4: per-prompt KV-cache cross-slot equality " f"({config.num_replication_slots} slots)")
        for prompt_idx, prompt_name in enumerate(config.prompt_names):
            start, end = schedule.range_for(prompt_idx)
            ref_kv = kv_cache_torch[0, :, start:end, :]
            for s in range(1, config.num_replication_slots):
                assert torch.equal(kv_cache_torch[s, :, start:end, :], ref_kv), (
                    f"KV-cache cross-slot equality failed: prompt={prompt_name!r} "
                    f"position_range=[{start}, {end}) slot {s} diverged from slot 0"
                )
            logger.info(f"prompt={prompt_name!r} KV cross-slot OK for positions [{start}, {end})")

    # =========================================================================
    # Phase 5: Per-(prompt, slot) dumps
    # =========================================================================
    if config.dump_hidden_states or config.dump_kv_cache:
        assert config.dump_dir is not None  # __post_init__ guarantees this
        logger.info(f"Phase 5: per-(prompt, slot) dumps to {config.dump_dir}")

    if config.dump_hidden_states:
        for prompt_name in config.prompt_names:
            for slot in range(config.num_replication_slots):
                out_path = config.dump_dir / f"output_hidden_states_slot_{slot:02d}_{prompt_name}.pt"
                torch.save(collected[prompt_name][slot], out_path)
                logger.info(f"Wrote {out_path}")

    if config.dump_kv_cache:
        assert kv_cache_torch is not None  # guaranteed by need_kv_cache above
        for prompt_idx, prompt_name in enumerate(config.prompt_names):
            start, end = schedule.range_for(prompt_idx)
            for slot in range(config.num_replication_slots):
                kv_slice = kv_cache_torch[slot, :, start:end, :]
                out_path = config.dump_dir / f"kv_cache_slot_{slot:02d}_{prompt_name}.pt"
                torch.save(kv_slice, out_path)
                logger.info(f"Wrote {out_path} shape={tuple(kv_slice.shape)}")

    logger.info("run_sweep complete")
    return SweepResult(
        collected=collected,
        kv_cache=kv_cache_torch,
        schedule=schedule,
        traces=traces,
    )
