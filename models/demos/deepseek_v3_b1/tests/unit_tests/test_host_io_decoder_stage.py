# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for HostIoDecoderStage with real DeepSeek V3 weights.

Runs a single ``HostIoDecoderStage`` (combined H2D + multi-upstream D2H pipeline
block) on a Blackhole 2D submesh, pushes tokens through the H2D socket, and reads
the assembled D2H pages back to host. The decoder is configured with the same
shape the production ``MoEDecoderStage`` uses (``num_routed_experts=256``,
``max_seq_len=128*1024``, ``num_slots=64``, ``persistent_mode=True``,
``is_torus=True``) and consumes real weights via ``CacheWeightProvider``.

The sweep covers a small subset of slots and positions (8 users x 32 token IDs)
and asserts that the input metadata fields (slot_id, position_id, token_id,
token_0_type) round-trip through the decoder to the D2H page tail.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import (  # noqa: F401  # comp_pcc kept for pipeclean re-enable
    comp_pcc,
    is_slow_dispatch,
)
from models.demos.deepseek_v3_b1.demo.decoder_stage import HostIoDecoderStage
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES, StageContext
from models.demos.deepseek_v3_b1.demo.weight_provider import CacheWeightProvider
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.model import InputField, TokenType, parse_output_page
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import ROUTED_EXPERT_LAYER_IDX
from models.demos.deepseek_v3_b1.utils import float_to_uint32
from models.demos.deepseek_v3_b1.weights.prepare import NUM_ROUTED_EXPERTS

# Default model + cache paths. Either env var overrides the corresponding default.
_DEFAULT_HF_MODEL_PATH = "/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized"
_DEFAULT_CACHE_PATH = str(Path.home() / ".cache")

# Reference hidden-state traces directory env var. No default — the test skips
# entirely if unset. Each ``*.pt`` under this directory is expected to contain
# a dict ``{"input": (L, HIDDEN_SIZE) bf16, "output": (L, HIDDEN_SIZE) bf16}``.
_HIDDEN_STATES_DIR_ENV = "DEEPSEEK_V3_HIDDEN_STATES_DIR"

# Standard PCC threshold (matches test_decoder_block.test_decoder's
# decoder-vs-golden MoE check).
_PCC_THRESHOLD = 0.97

# Sentinel parametrize value for prompt_name when the trace dir is unset/missing.
# The test body converts it to a pytest.skip so collection doesn't fail on an
# empty parametrize list.
_SKIP_PARAMETRIZE_TOKEN = "__skip__"


def _discover_prompt_names() -> list[str]:
    """Enumerate prompt trace files at pytest collection time.

    Reads ``DEEPSEEK_V3_HIDDEN_STATES_DIR``; returns the sorted list of file
    stems for every ``*.pt`` in that directory. If the env var is unset or
    points at a missing/empty directory, returns ``[_SKIP_PARAMETRIZE_TOKEN]``
    so pytest still collects exactly one parametrize row that the test body
    will then skip with a clean reason.
    """
    raw = os.getenv(_HIDDEN_STATES_DIR_ENV)
    if not raw:
        return [_SKIP_PARAMETRIZE_TOKEN]
    trace_dir = Path(raw)
    if not trace_dir.is_dir():
        return [_SKIP_PARAMETRIZE_TOKEN]
    stems = sorted(p.stem for p in trace_dir.glob("*.pt"))
    return stems if stems else [_SKIP_PARAMETRIZE_TOKEN]


def _load_reference_trace(trace_dir: Path, prompt_name: str) -> dict[str, torch.Tensor]:
    """Load and validate one prompt's reference trace.

    Expected on-disk format (per prompt):
        torch.save(
            {"input":  (L, HIDDEN_SIZE) bf16,
             "output": (L, HIDDEN_SIZE) bf16},
            f"{trace_dir}/{prompt_name}.pt",
        )

    Returns the dict unchanged. Raises AssertionError if the file's structure
    or shapes/dtypes don't match the contract.
    """
    path = trace_dir / f"{prompt_name}.pt"
    assert path.exists(), f"Reference trace not found: {path}"
    trace = torch.load(path)
    assert isinstance(trace, dict), f"{path}: expected dict, got {type(trace).__name__}"
    assert set(trace.keys()) >= {
        "input",
        "output",
    }, f"{path}: missing required keys 'input'/'output'; got {sorted(trace.keys())}"
    inp, out = trace["input"], trace["output"]
    assert isinstance(inp, torch.Tensor) and isinstance(
        out, torch.Tensor
    ), f"{path}: 'input' and 'output' must be torch.Tensor"
    assert (
        inp.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    ), f"{path}: expected bfloat16 tensors, got input={inp.dtype}, output={out.dtype}"
    assert inp.ndim == 2 and out.ndim == 2, (
        f"{path}: expected 2D (seq_len, HIDDEN_SIZE) tensors, got input.shape={tuple(inp.shape)}, "
        f"output.shape={tuple(out.shape)}"
    )
    assert inp.shape[-1] == D.HIDDEN_SIZE and out.shape[-1] == D.HIDDEN_SIZE, (
        f"{path}: last dim must equal D.HIDDEN_SIZE ({D.HIDDEN_SIZE}), got "
        f"input.shape={tuple(inp.shape)}, output.shape={tuple(out.shape)}"
    )
    assert (
        inp.shape[0] == out.shape[0]
    ), f"{path}: input and output seq_len mismatch: input={inp.shape[0]}, output={out.shape[0]}"
    return trace


@dataclass
class _SinglePipelineStage:
    """Duck-typed stand-in for ``ttnn.experimental.BlitzDecodePipelineStage``.

    ``generate_blitz_decode_pipeline`` does not handle ``num_meshes == 1`` (it OOBs on
    ``hops[0]`` for the no-loopback path) and the C++ ``BlitzDecodePipelineStage`` binding
    is read-only, so we construct the 1-stage descriptor in Python. ``HostIoDecoderStage``
    and ``PipelineBlock._init_combined_h2d_d2h_stage`` only read ``entry_node_coord`` and
    ``exit_node_coord`` from this object.
    """

    entry_node_coord: ttnn.MeshCoordinate
    exit_node_coord: ttnn.MeshCoordinate


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

    Used with ``HostIoDecoderStage(inject_hidden_states=True)``. The activation half is
    the bf16 hidden state reinterpreted as int32 words; the metadata half mirrors
    ``model.to_spec_input``'s field placement so the decoder + multi-upstream D2H tail
    round-trip the input metadata fields the same way as the embedding-driven path.
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
    pos_id: int,
    token_id: int,
) -> ttnn.Tensor:
    """Build a single H2D inject-mode page for a sweep iteration.

    Thin wrapper over ``_to_hidden_state_input`` that fixes the canonical
    decode-step defaults (matches ``demo/model_pipeline.py:_write_spec_pair``):
    ``prefill_token_id=-1``, ``token_type=TokenType.BASE``, ``temperature=0.6``,
    ``top_k=1``, ``probability_mass_threshold=1.0``.
    """
    return _to_hidden_state_input(
        hidden_state,
        token_id=token_id,
        prefill_token_id=-1,
        user_id=slot_id,
        position_id=pos_id,
        token_type=TokenType.BASE,
        temperature=0.6,
        top_k=1,
        probability_mass_threshold=1.0,
    )


def _extract_activation_from_d2h(output_tensor: ttnn.Tensor) -> torch.Tensor:
    """Return the activation half (first HIDDEN_SIZE bf16 elements) of a D2H page.

    The D2H page layout is ``activation (HIDDEN_SIZE bf16) || DeepseekMetadata
    (256 bytes)``. The leading slice IS the decoder's output hidden state for
    the corresponding (slot, position).
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


# Mode A (single_slot): one prompt fed through slot 0 only. Mode B (replicated_8slots):
# the same prompt fanned out across 8 slots, slot-by-slot per position. Cross-slot
# equality + per-slot PCC vs the reference output trace are checked downstream.
@pytest.mark.parametrize(
    "mode, num_replication_slots",
    [
        ("single_slot", 1),
        ("replicated_8slots", 8),
    ],
    ids=["mode_A_single_slot", "mode_B_replicated_8slots"],
)
# Discover prompt names at collection time. If DEEPSEEK_V3_HIDDEN_STATES_DIR is unset
# or empty, _discover_prompt_names returns [_SKIP_PARAMETRIZE_TOKEN] so pytest still
# collects exactly one row that the test body skips with a clean reason.
@pytest.mark.parametrize("prompt_name", _discover_prompt_names())
@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)])
# Production-shape decoder: max_seq_len=128*1024, num_slots=64 (matches MoEDecoderStage defaults).
@pytest.mark.parametrize("max_seq_len", [128 * 1024])
@pytest.mark.parametrize("position_id", [0])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "worker_l1_size": 1431568,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("decoder_layer_idx", [ROUTED_EXPERT_LAYER_IDX])
@pytest.mark.parametrize("num_slots", [64])
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(15000)
def test_host_io_decoder_stage(
    bh_2d_mesh_device,
    device_params,
    mode,
    num_replication_slots,
    prompt_name,
    mesh_rows,
    mesh_cols,
    max_seq_len,
    position_id,
    decoder_layer_idx,
    num_slots,
    tmp_path,
):
    """End-to-end test of HostIoDecoderStage: H2D → decoder → multi-upstream D2H."""
    torch.manual_seed(0)
    num_devices = mesh_rows * mesh_cols
    logger.info(f"Number of devices: {num_devices}")

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than available")
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode (H2D/D2H sockets require slow dispatch)")

    logger.info("Enabling asynchronous slow dispatch on the parent mesh device")

    logger.info(f"Creating {mesh_rows}x{mesh_cols} submesh ({num_devices} devices)")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    ttnn.enable_asynchronous_slow_dispatch(submesh)

    # Early skip if no reference traces were discovered at collection time.
    # _discover_prompt_names returns [_SKIP_PARAMETRIZE_TOKEN] when
    # DEEPSEEK_V3_HIDDEN_STATES_DIR is unset, missing, or empty.
    if prompt_name == _SKIP_PARAMETRIZE_TOKEN:
        pytest.skip(
            f"No reference traces discovered. Set ${_HIDDEN_STATES_DIR_ENV} to a directory "
            f"containing prompt traces (*.pt files of {{'input', 'output'}} bf16 tensors)."
        )
    trace_dir = Path(os.environ[_HIDDEN_STATES_DIR_ENV])
    trace = _load_reference_trace(trace_dir, prompt_name)
    seq_len = trace["input"].shape[0]
    logger.info(
        f"Reference trace loaded: prompt={prompt_name!r} mode={mode} "
        f"num_replication_slots={num_replication_slots} seq_len={seq_len}"
    )

    # Real weights via CacheWeightProvider. Either env var overrides the default;
    # if the resolved HF model path doesn't exist we skip rather than fail loudly.
    hf_model_path = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", _DEFAULT_HF_MODEL_PATH))
    cache_path = Path(os.getenv("DEEPSEEK_V3_CACHE_PATH", _DEFAULT_CACHE_PATH))
    if not hf_model_path.exists():
        pytest.skip(f"HF model path does not exist: {hf_model_path}")
    logger.info(f"Using HF model path: {hf_model_path}")
    logger.info(f"Using cache path: {cache_path}")
    provider = CacheWeightProvider(cache_path=cache_path, model_path=hf_model_path)

    logger.info(f"Loading real MoE layer {decoder_layer_idx} weights via CacheWeightProvider...")
    layer_weights = provider.load_moe_layer(layer_id=decoder_layer_idx, device=submesh)
    logger.info("MoE layer weights ready")

    # Single-stage pipeline (num_procs == 1, no loopback). Coords match
    # test_decoder_block.test_decoder: broadcast source = (1, 0), reduce-to-one root = (1, 1).
    logger.info("Building single-stage pipeline_config (entry=(1,0), exit=(1,1)) and StageContext")
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

    # Production-shape decoder, matching MoEDecoderStage's defaults:
    #   num_routed_experts = NUM_ROUTED_EXPERTS (256), enable_routing = True,
    #   use_hardcoded_expert_index = False, persistent_mode = True, is_torus = True.
    # max_seq_len and num_slots come from the parametrize (also production defaults).
    logger.info("Instantiating HostIoDecoderStage")
    stage = HostIoDecoderStage(
        weights=layer_weights,
        layer_idx=decoder_layer_idx,
        # Setup-time metadata only — position_id seeds the initial KV-cache state; the
        # runtime per-token slot_id / position_id come from each H2D page below.
        metadata=DeepseekMetadata(position_id=position_id),
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        persistent_mode=True,
        is_torus=True,
        is_moe=True,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        use_hardcoded_expert_index=False,
        enable_routing=True,
        inject_hidden_states=True,
    )

    logger.info("Building PipelineBlock (combined H2D + multi-upstream D2H branch)")
    pipeline_block = stage.create_pipeline_block(ctx)
    logger.info("Running stage.setup() — allocating decoder tensors and program context")
    stage.setup(ctx, pipeline_block)
    logger.info("Stage setup complete")

    logger.info("Dispatching H2D + multi-upstream D2H persistent kernels via pipeline_block.run()")
    pipeline_block.run()
    logger.info("Dispatching persistent decoder compute via stage.launch_compute()")
    stage.launch_compute(ctx, pipeline_block)
    logger.info("All persistent kernels dispatched")

    # Pre-allocate per-slot output collectors. Each entry is a (seq_len, HIDDEN_SIZE)
    # bf16 tensor that accumulates the decoder's activation output for the corresponding
    # replicated slot across the prompt sweep. After the sweep, Mode B asserts cross-slot
    # equality (steps 13) and both modes PCC the collected output against trace["output"]
    # (step 14).
    assert (
        num_replication_slots <= num_slots
    ), f"num_replication_slots ({num_replication_slots}) must be <= num_slots ({num_slots})"
    collected: dict[int, torch.Tensor] = {
        slot: torch.zeros(seq_len, D.HIDDEN_SIZE, dtype=torch.bfloat16) for slot in range(num_replication_slots)
    }

    # Reference-driven sweep. Outer loop = pos_id (steps the prompt); inner loop = slot_id
    # (Mode A: 1 slot, Mode B: 8 slots). For each (pos, slot) the SAME reference hidden
    # state from trace["input"][pos_id] is pushed; KV-cache contents at slot s/position p
    # should therefore match slot 0's. token_id mirrors pos_id to keep the metadata
    # round-trip exercised; in inject-hidden-states mode token_id does NOT drive an
    # embedding lookup.
    out_words = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES // dtype_size(ttnn.bfloat16)
    logger.info(
        f"Starting reference-driven sweep: prompt={prompt_name!r} mode={mode} "
        f"seq_len={seq_len} x num_replication_slots={num_replication_slots} "
        f"(hidden_state size = {D.HIDDEN_SIZE} bf16, out_words={out_words})"
    )
    for pos_id in range(seq_len):
        hidden_state = trace["input"][pos_id]
        for slot_id in range(num_replication_slots):
            token_id = pos_id
            input_tensor = _build_per_iteration_input(hidden_state, slot_id=slot_id, pos_id=pos_id, token_id=token_id)
            pipeline_block.write_token(input_tensor)

            torch_output = torch.zeros(1, out_words, dtype=torch.bfloat16)
            output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            pipeline_block.read_output(output_tensor)

            # Validate the metadata round-trip on every iteration (slot_id, position_id,
            # token_id, token_0_type). parse_output_page exposes slot_id and the output
            # token_0_* fields; the input-side fields (position_id, token_id) are read
            # directly off metadata_flat by InputField index.
            metadata_flat, parsed = _extract_metadata_from_d2h(output_tensor)
            actual_position_id = int(metadata_flat[InputField.POSITION_ID].item())
            actual_token_id = int(metadata_flat[InputField.TOKEN_ID].item())

            assert parsed.slot_id == slot_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata slot_id mismatch "
                f"(got {parsed.slot_id}, expected {slot_id})"
            )
            assert actual_position_id == pos_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata position_id mismatch "
                f"(got {actual_position_id}, expected {pos_id})"
            )
            assert actual_token_id == token_id, (
                f"slot={slot_id} pos={pos_id}: D2H metadata token_id mismatch "
                f"(got {actual_token_id}, expected {token_id})"
            )
            assert parsed.token_0_type == TokenType.BASE, (
                f"slot={slot_id} pos={pos_id}: D2H metadata token_0_type mismatch "
                f"(got {parsed.token_0_type}, expected TokenType.BASE = {TokenType.BASE})"
            )

            collected[slot_id][pos_id, :] = _extract_activation_from_d2h(output_tensor)
            logger.info(f"slot={slot_id} pos={pos_id} token_id={token_id} write/read/parse OK")

    logger.info(
        f"HostIoDecoderStage sweep complete: prompt={prompt_name!r} mode={mode} "
        f"seq_len={seq_len} x num_replication_slots={num_replication_slots}"
    )

    # =========================================================================
    # PHASE 1: Termination
    # =========================================================================
    # Tear down the persistent decoder + H2D/D2H kernels. Matches
    # Pipeline.terminate() in demo/pipeline.py minus the multi-process
    # distributed barriers (we're single-process here). Termination MUST happen
    # before KV-cache pull (phase 3) because the on-device read needs a
    # fast-dispatch context, and the H2D/D2H sockets run on slow dispatch.
    logger.info("Phase 1: signalling persistent decoder termination")
    stage.terminate(ctx, pipeline_block)

    # Termination dummy. We do NOT call pipeline_block.push_dummy_token() here:
    # that helper writes an all-zero H2D page, whose all-zero metadata trailer maps
    # to slot_id=0, position_id=0, token_id=0, token_0_type=TokenType.BASE(=0) — so
    # the dummy would land on slot 0 / position 0 of the prompt's KV-cache and stomp
    # the first cell BEFORE the KV-cache pull below reads it. Instead, build an
    # explicit inject-mode page that targets the (slot=num_slots-1,
    # position=max_seq_len-1) corner with a zero hidden state. The collected
    # (slot, pos) sweep only writes to slot ∈ [0, num_replication_slots) (≤ 8) and
    # pos ∈ [0, seq_len) (≪ max_seq_len), so this corner cell is guaranteed unused
    # and the KV cache pulled below remains uncontaminated.
    logger.info(
        f"Pushing termination dummy at slot={num_slots - 1} pos={max_seq_len - 1} "
        "(zero hidden state, guaranteed-unused KV-cache corner)"
    )
    zero_hidden_state = torch.zeros(D.HIDDEN_SIZE, dtype=torch.bfloat16)
    termination_dummy = _to_hidden_state_input(
        zero_hidden_state,
        token_id=0,
        prefill_token_id=-1,
        user_id=num_slots - 1,
        position_id=max_seq_len - 1,
        token_type=TokenType.BASE,
        temperature=0.6,
        top_k=1,
        probability_mass_threshold=1.0,
    )
    pipeline_block.write_token(termination_dummy)
    logger.info("Draining termination D2H output")
    pipeline_block.drain_dummy_output()
    logger.info("Terminating H2D + multi-upstream D2H kernels")
    pipeline_block.terminate()
    ttnn.synchronize_device(submesh)
    logger.info("Phase 1 complete: HostIoDecoderStage teardown done")

    # =========================================================================
    # PHASE 2: Hidden state validation
    # =========================================================================
    # (a) Mode B cross-slot equality. The same hidden state is pushed through every
    # replicated slot at each position, so the collected activation outputs must
    # be byte-identical across slots. (No-op for Mode A where num_replication_slots == 1.)
    logger.info("Phase 2: hidden state validation")
    if num_replication_slots > 1:
        logger.info(f"Checking hidden-state cross-slot equality across {num_replication_slots} slots")
        ref = collected[0]
        for s in range(1, num_replication_slots):
            assert torch.equal(collected[s], ref), (
                f"Hidden-state cross-slot equality failed: slot {s} diverged from slot 0 "
                f"(prompt={prompt_name!r}, seq_len={seq_len})"
            )
        logger.info("Hidden-state cross-slot equality OK")

    # (b) Per-slot PCC of the collected decoder activation outputs against the
    # reference output hidden states loaded from
    # ``$DEEPSEEK_V3_HIDDEN_STATES_DIR/<prompt_name>.pt``'s ``"output"`` field.
    # ``collected[slot]`` is (seq_len, HIDDEN_SIZE) bf16, accumulated over the
    # sweep at this prompt. PCC is computed on the flattened tensors against
    # the (seq_len, HIDDEN_SIZE) bf16 reference. Threshold is ``_PCC_THRESHOLD``
    # = 0.97 (matches the decoder-vs-golden bar in test_decoder_block.test_decoder).
    # Both modes run this gate; in Mode B every replicated slot must independently
    # clear it (cross-slot equality is already asserted above, so a single failure
    # implies all slots fail — the per-slot loop is kept for explicit per-slot
    # logging on failure).
    # TODO(pipeclean): PCC validation is temporarily disabled while we sanity-check
    # the infrastructure with synthetic traces (generate_pipeclean_traces.py). Synthetic
    # ``trace["output"]`` is unrelated to the decoder's real output and will fail PCC
    # at ~0.0. Re-enable this block as soon as real reference traces are available.
    logger.warning(
        f"PIPECLEAN: Reference-trace PCC validation is COMMENTED OUT "
        f"(prompt={prompt_name!r} mode={mode} threshold={_PCC_THRESHOLD}). "
        f"Re-enable once real GPU reference traces are available."
    )
    # logger.info(
    #     f"Reference-trace PCC validation: prompt={prompt_name!r} mode={mode} "
    #     f"threshold={_PCC_THRESHOLD}"
    # )
    # expected = trace["output"]
    # for slot in range(num_replication_slots):
    #     passing, pcc = comp_pcc(expected.flatten(), collected[slot].flatten(), _PCC_THRESHOLD)
    #     logger.info(
    #         f"Reference-trace PCC: slot={slot} prompt={prompt_name!r} pcc={float(pcc):.6f} "
    #         f"threshold={_PCC_THRESHOLD} pass={passing}"
    #     )
    #     assert passing, (
    #         f"Reference-trace validation FAILED: slot={slot} prompt={prompt_name!r} "
    #         f"mode={mode} pcc={float(pcc):.6f} below threshold {_PCC_THRESHOLD}"
    #     )
    # logger.info(f"Reference-trace PCC validation OK for all {num_replication_slots} slot(s)")
    logger.info("Phase 2 complete: hidden state validation done")

    # =========================================================================
    # PHASE 3: KV cache validation
    # =========================================================================
    # Pull the on-device KV cache to a single host tensor. Must run under a
    # fast-dispatch context — slow-dispatch (used by the H2D/D2H sockets in
    # phase 0) doesn't support arbitrary on-device reads, which is why this
    # phase has to follow phase 1 termination.
    logger.info("Phase 3: KV cache validation; pulling on-device KV cache to host")
    with ttnn.device.setup_fast_dispatch(submesh):
        kv_cache_torch = stage.get_kv_cache_host()
    assert kv_cache_torch is not None, "get_kv_cache_host returned None (setup not completed?)"

    # Mode B KV-cache cross-slot equality. Mirrors the hidden-state check above:
    # the same hidden state was pushed through every replicated slot at each
    # position, so each slot's KV-cache contents over the written range must be
    # byte-identical. ``kv_cache_torch`` shape is (num_slots, 1, max_seq_len, kvpe_dim);
    # we only wrote to slot ∈ [0, num_replication_slots) × position ∈ [0, seq_len),
    # so the slice we compare is [s, :, :seq_len, :] vs [0, :, :seq_len, :]. (No-op
    # for Mode A where num_replication_slots == 1.)
    if num_replication_slots > 1:
        logger.info(
            f"Checking KV-cache cross-slot equality across {num_replication_slots} slots "
            f"over positions [0, {seq_len}); kv_cache shape={tuple(kv_cache_torch.shape)}"
        )
        ref_kv = kv_cache_torch[0, :, :seq_len, :]
        for s in range(1, num_replication_slots):
            assert torch.equal(kv_cache_torch[s, :, :seq_len, :], ref_kv), (
                f"KV-cache cross-slot equality failed: slot {s} KV-cache diverged from slot 0 "
                f"(prompt={prompt_name!r}, seq_len={seq_len})"
            )
        logger.info("KV-cache cross-slot equality OK")
    logger.info("Phase 3 complete: KV cache validation done")

    # =========================================================================
    # PHASE 4: Hidden state and KV cache dump
    # =========================================================================
    # Dump dir is shared between the per-slot hidden state outputs and the KV cache.
    # Use $DEEPSEEK_V3_KV_CACHE_DUMP_DIR if set, else pytest's tmp_path so files
    # survive between test runs.
    dump_dir = Path(os.getenv("DEEPSEEK_V3_KV_CACHE_DUMP_DIR", str(tmp_path)))
    dump_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Phase 4: dumping hidden states + KV cache to {dump_dir}")

    # Per-slot collected output hidden states. One file per replicated slot,
    # plain (seq_len, HIDDEN_SIZE) bf16 tensor (no dict / no nesting). Mode A
    # writes 1 file (slot 00), Mode B writes 8 files (slots 00..07). Filename
    # includes the prompt name so different prompts coexist in the same dump
    # dir without colliding.
    logger.info(
        f"Dumping per-slot collected output hidden states for prompt={prompt_name!r} "
        f"({num_replication_slots} slot(s))"
    )
    for slot in range(num_replication_slots):
        out_path = dump_dir / f"output_hidden_states_slot_{slot:02d}_{prompt_name}.pt"
        torch.save(collected[slot], out_path)
        logger.info(f"Wrote slot {slot:02d} output trace to {out_path}")

    # KV cache dump. Mode A and Mode B for the SAME prompt collide on disk by
    # design (locked-in spec — only one KV file per prompt, regardless of mode).
    # Run one mode at a time per prompt if the per-mode KV file is needed.
    kv_cache_path = dump_dir / f"kv_cache_stage_00_layer_{decoder_layer_idx}_{prompt_name}.pt"
    logger.info(f"Writing KV cache shape={tuple(kv_cache_torch.shape)} to {kv_cache_path}")
    torch.save(kv_cache_torch, kv_cache_path)

    logger.info("Phase 4 complete: dumps written")
