# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 Performance Demo

Measures embedding throughput and latency on Tenstorrent hardware.

Metrics reported:
  - Compile time: time for the first forward pass
  - Forward time: wall-clock time to embed the full batch
  - Embeddings/s: batch_size / forward_time
  - Tokens/s:     total_tokens / forward_time

Usage (standalone, single device):
    python models/demos/wormhole/bge_m3/demo/perf_demo.py

Usage (pytest, picks device from MESH_DEVICE env):
    pytest models/demos/wormhole/bge_m3/demo/perf_demo.py -sv
"""

import json
import os
import time

import pytest
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import create_submeshes

try:
    import tracy
except ImportError:
    tracy = None

MODEL_NAME = "BAAI/bge-m3"

SAMPLE_TEXTS = [
    "Artificial intelligence is transforming how we interact with technology.",
    "AI is changing the way humans use computers and machines.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "The weather is sunny today with clear blue skies.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "Baking bread requires flour, water, yeast, and patience.",
    "Neural networks mimic the human brain's structure and function.",
    "Natural language processing enables computers to understand text.",
]

MESH_SHAPES = {
    1: (1, 1),
    2: (1, 2),
    8: (1, 8),
    32: (8, 4),
}

MESH_DEVICE_ENV_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _tracy_signpost(message: str) -> None:
    """Emit Tracy signposts when Tracy is available."""
    if tracy is not None:
        tracy.signpost(message)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _is_fixed_b1s512_case(
    *,
    batch_size: int,
    runtime: dict,
    tt_data_parallel: int,
    seq_len: int,
    max_seq_len: int,
) -> bool:
    return (
        batch_size == 1
        and tt_data_parallel == 1
        and runtime.get("local_data_parallel", 1) == 1
        and int(seq_len) == 512
        and int(max_seq_len) == 512
    )


def _is_fixed_s512_single_dp_case(
    *,
    runtime: dict,
    tt_data_parallel: int,
    seq_len: int,
    max_seq_len: int,
) -> bool:
    return (
        tt_data_parallel == 1
        and runtime.get("local_data_parallel", 1) == 1
        and int(seq_len) == 512
        and int(max_seq_len) == 512
    )


def _maybe_enable_async_slow_dispatch(mesh_device, enable: bool) -> bool:
    """
    Enable async slow dispatch when requested and return whether we need to restore it.
    """
    if not enable:
        return False
    try:
        already_enabled = bool(ttnn.device.is_asynchronous_slow_dispatch_enabled(mesh_device))
    except Exception as e:
        logger.warning(f"Unable to query async slow dispatch state: {e}. Continuing without async.")
        return False
    if already_enabled:
        logger.info("Async slow dispatch already enabled on mesh device; keeping current state.")
        return False
    try:
        ttnn.enable_asynchronous_slow_dispatch(mesh_device)
        logger.info("Auto-enabled async slow dispatch for this benchmark run.")
        return True
    except Exception as e:
        logger.warning(f"Failed to enable async slow dispatch: {e}. Continuing in synchronous mode.")
        return False


def _maybe_restore_async_slow_dispatch(mesh_device, restore_needed: bool) -> None:
    if not restore_needed:
        return
    try:
        ttnn.disable_asynchronous_slow_dispatch(mesh_device)
        logger.info("Restored synchronous slow dispatch mode after benchmark run.")
    except Exception as e:
        logger.warning(f"Failed to restore async slow dispatch state: {e}")


def load_input_texts(input_file, batch_size):
    """Load input texts from a JSON file or generate synthetic ones."""
    if input_file and os.path.exists(input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        texts = [item["text"] if isinstance(item, dict) else item for item in data]
    else:
        texts = SAMPLE_TEXTS[:]

    while len(texts) < batch_size:
        texts = texts * 2
    return texts[:batch_size]


def get_default_mesh_device_param():
    visible_devices = os.environ.get("TT_VISIBLE_DEVICES")
    if visible_devices:
        # Avoid import-time cluster probing on custom topologies by trusting explicit visibility.
        parsed_visible_devices = [d.strip() for d in visible_devices.split(",") if d.strip()]
        if parsed_visible_devices:
            n = len(parsed_visible_devices)
            return MESH_SHAPES.get(n, n)

    try:
        if ttnn.using_distributed_env():
            try:
                n = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
                return MESH_SHAPES.get(n, n)
            except Exception:
                pass
    except Exception:
        # On some custom clusters this probe can TT_FATAL during collection.
        pass

    try:
        n = len(ttnn.get_device_ids())
    except Exception:
        n = 1
    return MESH_SHAPES.get(n, n)


def resolve_mesh_device_param():
    """
    Resolve mesh parameter without eagerly invoking distributed-environment probing.
    This avoids import-time failures on custom cluster setups when MESH_DEVICE is set.
    """
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env in MESH_DEVICE_ENV_MAP:
        return MESH_DEVICE_ENV_MAP[mesh_device_env]
    return get_default_mesh_device_param()


def _submesh_has_local_devices(submesh):
    view = submesh.get_view()
    return any(
        view.is_local(ttnn.MeshCoordinate(row, col))
        for row in range(submesh.shape[0])
        for col in range(submesh.shape[1])
    )


def prepare_embedding_model(
    mesh_device,
    global_batch_size,
    max_seq_len,
    tt_data_parallel=1,
):
    """Build TT model(s) for embedding workloads.

    When tt_data_parallel > 1, creates independent model instances on submeshes.
    """
    if global_batch_size % tt_data_parallel != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by tt_data_parallel={tt_data_parallel}"
        )

    batch_per_dp = global_batch_size // tt_data_parallel

    all_submeshes = create_submeshes(mesh_device, tt_data_parallel)
    local_indices = (
        [i for i, s in enumerate(all_submeshes) if _submesh_has_local_devices(s)]
        if isinstance(mesh_device, ttnn.MeshDevice) and tt_data_parallel > 1
        else list(range(len(all_submeshes)))
    )
    submeshes = [all_submeshes[i] for i in local_indices]

    if not submeshes:
        raise RuntimeError("No local submeshes available on this host rank")

    if len(submeshes) != len(all_submeshes):
        logger.info(f"Distributed mode: using {len(submeshes)}/{len(all_submeshes)} local submeshes")

    models = []
    model_args_list = []
    state_dict = None

    for submesh in submeshes:
        model_args_i, model_i, state_dict = create_tt_model(
            mesh_device=submesh,
            max_batch_size=batch_per_dp,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            hf_model_name=MODEL_NAME,
        )
        models.append(model_i)
        model_args_list.append(model_args_i)

    if not model_args_list or model_args_list[0].tokenizer is None:
        raise RuntimeError("BGE-M3 model did not initialize model_args/tokenizer")

    runtime = {
        "models": models,
        "submeshes": submeshes,
        "batch_per_dp": batch_per_dp,
        "global_data_parallel": tt_data_parallel,
        "local_data_parallel": len(submeshes),
    }
    return runtime, model_args_list[0], model_args_list[0].tokenizer


def tokenize_and_pad(tokenizer, texts, max_seq_len):
    """Tokenize texts, returning padded input_ids and original lengths."""
    encoded = tokenizer(texts, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] != max_seq_len:
        raise RuntimeError(f"Tokenizer output length ({input_ids.shape[1]}) does not match max_seq_len ({max_seq_len})")
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    original_lens = attention_mask.sum(dim=1).tolist()
    return input_ids, attention_mask, token_type_ids, [int(l) for l in original_lens]


def generate_synthetic_inputs(tokenizer, batch_size, seq_len):
    """Generate random token sequences of exactly seq_len for ISL benchmarking."""
    vocab_size = tokenizer.vocab_size
    low = max(100, 0)
    high = min(vocab_size, 50000)
    input_ids = torch.randint(low, high, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, attention_mask, token_type_ids, prompt_lens


def generate_tokenized_small_isl_inputs(tokenizer, batch_size, seq_len, padded_seq_len=128):
    """
    Tokenize at short ISL, then pad execution tensors to padded_seq_len.
    Benchmark accounting remains at the original short ISL.
    """
    if seq_len >= padded_seq_len:
        raise ValueError(f"seq_len must be < padded_seq_len, got seq_len={seq_len}, padded_seq_len={padded_seq_len}")

    texts = load_input_texts(None, batch_size)
    short_input_ids, short_attention_mask, short_token_type_ids, _ = tokenize_and_pad(tokenizer, texts, seq_len)

    pad_len = padded_seq_len - seq_len
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.nn.functional.pad(short_input_ids, (0, pad_len), mode="constant", value=pad_token_id)
    attention_mask = torch.nn.functional.pad(short_attention_mask, (0, pad_len), mode="constant", value=0)
    token_type_ids = torch.nn.functional.pad(short_token_type_ids, (0, pad_len), mode="constant", value=0)

    prompt_lens = [seq_len] * batch_size
    return input_ids, attention_mask, token_type_ids, prompt_lens


def _to_ttnn_ids(ids: torch.Tensor, mesh_device, dtype=ttnn.uint32) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _to_ttnn_additive_attention_mask(
    additive_mask: torch.Tensor,
    mesh_device,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        additive_mask.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def build_position_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    XLM-RoBERTa-compatible padding-aware position IDs.

    This mirrors ``BgeM3PerformantRunner.build_position_ids`` in ``demo.py`` and the on-device
    ``BgeM3Model.create_position_ids_from_input_ids`` path.
    """
    mask = (input_ids != int(pad_token_id)).to(torch.int64)
    incremental_indices = torch.cumsum(mask, dim=1) * mask
    return (incremental_indices + int(pad_token_id)).to(torch.int64)


def build_additive_attention_mask(attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Build a rank-4 additive mask for SDPA from a HF ``{0, 1}`` 2-D mask.

    The additive-mask formula mirrors ``BgeM3PerformantRunner.build_additive_attention_mask`` in
    ``demo.py``. We expand to ``[B, 1, S, S]`` on host so the measured TT forward does not run the
    device-side rank-4 expansion.
    """
    keep = attention_mask.to(torch.bfloat16)
    additive = (1.0 - keep) * -100000.0
    return additive.unsqueeze(1).unsqueeze(1).expand(-1, -1, int(seq_len), -1).contiguous()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def capture_embedding_trace(
    runtime,
    input_ids,
    attention_mask,
    token_type_ids,
    *,
    position_ids=None,
    model_attention_mask=None,
):
    models = runtime["models"]
    submeshes = runtime["submeshes"]
    local_dp = runtime["local_data_parallel"]
    batch_per_dp = runtime["batch_per_dp"]

    if local_dp != 1:
        raise RuntimeError(f"Trace path currently supports local_data_parallel=1, got {local_dp}")

    local_batch = batch_per_dp * local_dp
    if input_ids.shape[0] != local_batch:
        raise RuntimeError(
            f"Input batch ({input_ids.shape[0]}) must match local batch ({local_batch}) for local_data_parallel={local_dp}"
        )

    submesh = submeshes[0]
    input_chunk = input_ids
    attention_chunk = attention_mask
    model_attention_chunk = model_attention_mask if model_attention_mask is not None else attention_chunk
    token_type_chunk = token_type_ids
    position_chunk = position_ids

    tt_input_ids = _to_ttnn_ids(input_chunk, mesh_device=submesh)
    if model_attention_chunk.dim() == 4:
        mask_dtype = getattr(models[0], "_mask_dtype", ttnn.bfloat16)
        tt_attention_mask = _to_ttnn_additive_attention_mask(
            model_attention_chunk, mesh_device=submesh, dtype=mask_dtype
        )
    else:
        tt_attention_mask = _to_ttnn_ids(model_attention_chunk, mesh_device=submesh)
    tt_token_type_ids = _to_ttnn_ids(token_type_chunk, mesh_device=submesh)
    tt_position_ids = _to_ttnn_ids(position_chunk, mesh_device=submesh) if position_chunk is not None else None

    tt_output = models[0].capture_trace(
        input_ids=tt_input_ids,
        attention_mask=tt_attention_mask,
        token_type_ids=tt_token_type_ids,
        position_ids=tt_position_ids,
        mesh_device=submesh,
        cq_id=0,
    )

    return {
        "model": models[0],
        "submesh": submesh,
        "tt_output": tt_output,
        "input_chunk": input_chunk,
        "attention_chunk": attention_chunk,
    }


def run_embedding_forward_trace(trace_state, profiler=None, step_name=None, collect_conversion_timing=False):
    torch_to_ttnn_time_s = 0.0
    ttnn_to_torch_time_s = 0.0
    ttnn_to_torch_step = f"{step_name}_ttnn_to_torch" if step_name else None
    use_profiler_for_conversion = collect_conversion_timing and profiler is not None and step_name is not None

    if profiler and step_name:
        profiler.start(step_name)

    trace_state["model"].execute_trace(blocking=False, synchronize=True)

    if profiler and step_name:
        profiler.end(step_name)

    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.start(ttnn_to_torch_step)
    elif collect_conversion_timing:
        ttnn_to_torch_start = time.perf_counter()

    hidden_states = to_torch_auto_compose(trace_state["tt_output"], device=trace_state["submesh"])
    if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)
    hidden_states = hidden_states[:, : trace_state["input_chunk"].shape[1], :].to(torch.float32)
    embeddings = _mean_pool(hidden_states, trace_state["attention_chunk"][:, : hidden_states.shape[1]])

    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.end(ttnn_to_torch_step)
        ttnn_to_torch_time_s = profiler.get_duration(ttnn_to_torch_step)
    elif collect_conversion_timing:
        ttnn_to_torch_time_s = time.perf_counter() - ttnn_to_torch_start

    if collect_conversion_timing:
        conversion_timing = {
            "torch_to_ttnn_s": torch_to_ttnn_time_s,
            "ttnn_to_torch_s": ttnn_to_torch_time_s,
        }
        return embeddings, conversion_timing

    return embeddings


def run_embedding_forward(
    runtime,
    input_ids,
    attention_mask,
    token_type_ids,
    profiler=None,
    step_name=None,
    collect_conversion_timing=False,
    *,
    position_ids=None,
    model_attention_mask=None,
):
    """Run one BGE-M3 TT forward pass and return dense sentence embeddings."""
    models = runtime["models"]
    submeshes = runtime["submeshes"]
    batch_per_dp = runtime["batch_per_dp"]
    local_dp = runtime["local_data_parallel"]

    local_batch = batch_per_dp * local_dp
    if input_ids.shape[0] != local_batch:
        raise RuntimeError(
            f"Input batch ({input_ids.shape[0]}) must match local batch ({local_batch}) for local_data_parallel={local_dp}"
        )

    input_chunks = torch.chunk(input_ids, local_dp, dim=0)
    attention_chunks = torch.chunk(attention_mask, local_dp, dim=0)
    model_attention_chunks = (
        torch.chunk(model_attention_mask, local_dp, dim=0) if model_attention_mask is not None else attention_chunks
    )
    token_type_chunks = torch.chunk(token_type_ids, local_dp, dim=0)
    position_chunks = torch.chunk(position_ids, local_dp, dim=0) if position_ids is not None else [None] * local_dp

    # Phase 1: Stage TT inputs for each submesh.
    staged_inputs = []
    torch_to_ttnn_time_s = 0.0
    ttnn_to_torch_time_s = 0.0
    torch_to_ttnn_step = f"{step_name}_torch_to_ttnn" if step_name else None
    use_profiler_for_conversion = collect_conversion_timing and profiler is not None and step_name is not None
    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.start(torch_to_ttnn_step)
    elif collect_conversion_timing:
        torch_to_ttnn_start = time.perf_counter()
    for i in range(local_dp):
        staged_inputs.append(
            (
                _to_ttnn_ids(input_chunks[i], mesh_device=submeshes[i]),
                (
                    _to_ttnn_additive_attention_mask(
                        model_attention_chunks[i],
                        mesh_device=submeshes[i],
                        dtype=getattr(models[i], "_mask_dtype", ttnn.bfloat16),
                    )
                    if model_attention_chunks[i].dim() == 4
                    else _to_ttnn_ids(model_attention_chunks[i], mesh_device=submeshes[i])
                ),
                _to_ttnn_ids(token_type_chunks[i], mesh_device=submeshes[i]),
                _to_ttnn_ids(position_chunks[i], mesh_device=submeshes[i]) if position_chunks[i] is not None else None,
            )
        )
    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.end(torch_to_ttnn_step)
        torch_to_ttnn_time_s = profiler.get_duration(torch_to_ttnn_step)
    elif collect_conversion_timing:
        torch_to_ttnn_time_s = time.perf_counter() - torch_to_ttnn_start

    # Phase 2: Dispatch forwards to all submeshes first.
    if profiler and step_name:
        profiler.start(step_name)

    tt_outputs = []
    for i in range(local_dp):
        tt_output = models[i](
            input_ids=staged_inputs[i][0],
            attention_mask=staged_inputs[i][1],
            token_type_ids=staged_inputs[i][2],
            position_ids=staged_inputs[i][3],
        )
        tt_outputs.append(tt_output)

    # We need to explicitly sync the device so the host waits for the computation
    # to finish before we stop the timer, otherwise we're just measuring Python dispatch time.
    for i in range(local_dp):
        ttnn.synchronize_device(submeshes[i])

    if profiler and step_name:
        profiler.end(step_name)

    # Phase 3: Read back and pool outputs.
    pooled_chunks = []
    ttnn_to_torch_step = f"{step_name}_ttnn_to_torch" if step_name else None
    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.start(ttnn_to_torch_step)
    elif collect_conversion_timing:
        ttnn_to_torch_start = time.perf_counter()
    for i in range(local_dp):
        hidden_states = to_torch_auto_compose(tt_outputs[i], device=submeshes[i])
        if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = hidden_states[:, : input_chunks[i].shape[1], :].to(torch.float32)
        pooled_chunks.append(_mean_pool(hidden_states, attention_chunks[i][:, : hidden_states.shape[1]]))
    if collect_conversion_timing and use_profiler_for_conversion:
        profiler.end(ttnn_to_torch_step)
        ttnn_to_torch_time_s = profiler.get_duration(ttnn_to_torch_step)
    elif collect_conversion_timing:
        ttnn_to_torch_time_s = time.perf_counter() - ttnn_to_torch_start

    embeddings = torch.cat(pooled_chunks, dim=0)
    if collect_conversion_timing:
        conversion_timing = {
            "torch_to_ttnn_s": torch_to_ttnn_time_s,
            "ttnn_to_torch_s": ttnn_to_torch_time_s,
        }
        return embeddings, conversion_timing

    return embeddings


def run_embedding_forward_two_pass(
    runtime,
    first_pass_inputs,
    second_pass_inputs,
    profiler=None,
    step_name=None,
    collect_conversion_timing=False,
):
    """
    Run two sequential forwards and treat them as one combined iteration.
    Used for large-shape fallback paths (e.g., ISL 8192 with 2x batch-16).
    """
    if profiler and step_name:
        profiler.start(step_name)

    emb_first, conv_first = run_embedding_forward(
        runtime,
        first_pass_inputs[0],
        first_pass_inputs[1],
        first_pass_inputs[2],
        profiler=None,
        step_name=None,
        collect_conversion_timing=collect_conversion_timing,
    )
    emb_second, conv_second = run_embedding_forward(
        runtime,
        second_pass_inputs[0],
        second_pass_inputs[1],
        second_pass_inputs[2],
        profiler=None,
        step_name=None,
        collect_conversion_timing=collect_conversion_timing,
    )

    if profiler and step_name:
        profiler.end(step_name)

    embeddings = torch.cat([emb_first, emb_second], dim=0)
    if not collect_conversion_timing:
        return embeddings

    conversion_timing = {
        "torch_to_ttnn_s": conv_first["torch_to_ttnn_s"] + conv_second["torch_to_ttnn_s"],
        "ttnn_to_torch_s": conv_first["ttnn_to_torch_s"] + conv_second["ttnn_to_torch_s"],
    }
    return embeddings, conversion_timing


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, max_seq_len, input_seq_len, num_iterations, tt_data_parallel",
    [
        (1, 512, 32, 10, 1),
        (1, 512, 64, 10, 1),
        (1, 512, 128, 10, 1),
        (1, 512, 256, 10, 1),
        (1, 512, 512, 10, 1),
        (32, 512, 512, 10, 1),
        # (1, 8192, None, 5, 1),
        # (8, 8192, None, 3, 1),
        # TG target: global batch 32, local batch 1 per chip (DP32).
        # (32, 8192, 512, 5, 32),
        # (32, 8192, 1024, 5, 32),
        # (32, 8192, 2048, 5, 32),
        # (32, 8192, 4096, 5, 32),
        # TG target: global batch 1024, local batch 32 per chip (DP32).
        # (1024, 8192, 512, 5, 32),
        # (1024, 8192, 1024, 5, 32),
        # (1024, 8192, 2048, 5, 32),
        # (1024, 8192, 4096, 5, 32),
        # Disabled: ISL 8192 is too large for local batch 32 per chip.
        # (1024, 8192, 8192, 5, 32),
    ],
    ids=[
        "batch1dp1-isl32",
        "batch1dp1-isl64",
        "batch1dp1-isl128",
        "batch1dp1-isl256",
        "batch1dp1-isl512",
        "batch32dp1-isl512",
        # "batch32dp32-isl512",
        # "batch32dp32-isl1024",
        # "batch32dp32-isl2048",
        # "batch32dp32-isl4096",
        # "batch1024-isl512",
        # "batch1024-isl1024",
        # "batch1024-isl2048",
        # "batch1024-isl4096",
        # "batch1024-isl8192",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [resolve_mesh_device_param()],
    indirect=True,
)
def test_embedding_perf(
    mesh_device,
    batch_size,
    max_seq_len,
    input_seq_len,
    num_iterations,
    tt_data_parallel,
    is_ci_env,
):
    """
    Embedding performance demo: measures compile time, forward latency, and throughput.

    max_seq_len:   model capacity
    input_seq_len: actual tokens per input (None = use real sample texts)
    """
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

    if tt_data_parallel > num_devices:
        pytest.skip(
            f"tt_data_parallel={tt_data_parallel} requires {tt_data_parallel} devices, only {num_devices} available"
        )
    if batch_size % max(tt_data_parallel, 1) != 0:
        pytest.skip(f"batch_size={batch_size} not evenly divisible by tt_data_parallel={tt_data_parallel}")
    if input_seq_len is not None and input_seq_len > max_seq_len:
        pytest.skip(f"input_seq_len={input_seq_len} exceeds max_seq_len={max_seq_len}")
    if batch_size < 1:
        pytest.skip("batch_size must be >= 1")

    profiler = BenchmarkProfiler()
    profiler.start("run")

    device_name_info = determine_device_name(mesh_device)
    tt_device_name = device_name_info[0] if isinstance(device_name_info, tuple) else str(device_name_info)

    # ---- Build model ----
    batch_per_dp = batch_size // max(tt_data_parallel, 1)
    logger.info(
        f"Building model: global_batch={batch_size}, batch_per_dp={batch_per_dp}, "
        f"tt_data_parallel={tt_data_parallel}, max_seq_len={max_seq_len}, input_seq_len={input_seq_len}, device={tt_device_name}"
    )
    profiler.start("build_model")
    runtime, model_args, tokenizer = prepare_embedding_model(
        mesh_device,
        global_batch_size=batch_size,
        max_seq_len=max_seq_len,
        tt_data_parallel=tt_data_parallel,
    )
    profiler.end("build_model")
    logger.info(
        f"Model built in {profiler.get_duration('build_model'):.1f}s "
        f"(global_dp={runtime['global_data_parallel']}, local_dp={runtime['local_data_parallel']})"
    )

    # ---- Prepare inputs ----
    profiler.start("loading_inputs")
    if input_seq_len is not None:
        if input_seq_len in (32, 64):
            input_ids, attention_mask, token_type_ids, prompt_lens = generate_tokenized_small_isl_inputs(
                tokenizer, batch_size, input_seq_len, padded_seq_len=128
            )
            logger.info(
                f"Short ISL tokenized path: requested ISL={input_seq_len}, padded execution length={input_ids.shape[1]}"
            )
        else:
            input_ids, attention_mask, token_type_ids, prompt_lens = generate_synthetic_inputs(
                tokenizer, batch_size, input_seq_len
            )
    else:
        texts = load_input_texts(None, batch_size)
        input_ids, attention_mask, token_type_ids, prompt_lens = tokenize_and_pad(tokenizer, texts, max_seq_len)
    profiler.end("loading_inputs")

    isl = input_seq_len if input_seq_len is not None else int(sum(prompt_lens) / len(prompt_lens))
    total_input_tokens = sum(prompt_lens)
    logger.info(f"Prepared {batch_size} inputs, ISL={isl}, total tokens = {total_input_tokens}")
    fixed_b1s512_case = _is_fixed_b1s512_case(
        batch_size=batch_size,
        runtime=runtime,
        tt_data_parallel=tt_data_parallel,
        seq_len=isl,
        max_seq_len=max_seq_len,
    )
    fixed_s512_single_dp_case = _is_fixed_s512_single_dp_case(
        runtime=runtime,
        tt_data_parallel=tt_data_parallel,
        seq_len=isl,
        max_seq_len=max_seq_len,
    )
    manual_trace_requested = _env_flag("BGE_M3_USE_TRACE") or _env_flag("BGE_M3_USE_TRACE_REPLAY")
    auto_trace_requested = _env_flag("BGE_M3_AUTO_TRACE", "1") and fixed_b1s512_case
    use_trace = manual_trace_requested or auto_trace_requested
    if auto_trace_requested and not manual_trace_requested:
        logger.info("Auto-enabled trace replay for fixed B1/S512 benchmark. Set BGE_M3_AUTO_TRACE=0 to disable.")
    if use_trace and runtime["local_data_parallel"] != 1:
        logger.warning(
            "BGE_M3_USE_TRACE=1 requested, but local_data_parallel != 1; falling back to non-trace forward path."
        )
        use_trace = False

    manual_async_requested = _env_flag("BGE_M3_USE_ASYNC_SLOW_DISPATCH")
    auto_async_requested = _env_flag("BGE_M3_AUTO_ASYNC_SLOW_DISPATCH", "0") and fixed_b1s512_case
    disable_async_requested = _env_flag("BGE_M3_DISABLE_ASYNC_SLOW_DISPATCH")
    enable_async_slow_dispatch = not disable_async_requested and (manual_async_requested or auto_async_requested)
    if auto_async_requested and not manual_async_requested and not disable_async_requested:
        logger.info(
            "Auto-enabled async slow dispatch policy for fixed B1/S512 benchmark. "
            "Set BGE_M3_AUTO_ASYNC_SLOW_DISPATCH=0 to disable."
        )

    position_ids = None
    use_precomputed_position_ids = _env_flag("BGE_M3_PRECOMPUTE_POSITION_IDS", "1") and fixed_s512_single_dp_case
    if use_precomputed_position_ids:
        position_ids = build_position_ids(input_ids, model_args.pad_token_id)
        logger.info(
            "Precomputed host position_ids for fixed single-DP S512 benchmark. "
            "Set BGE_M3_PRECOMPUTE_POSITION_IDS=0 to use the on-device builder."
        )

    model_attention_mask = None
    use_precomputed_attention_mask = _env_flag("BGE_M3_PRECOMPUTE_ATTENTION_MASK", "1") and fixed_s512_single_dp_case
    if use_precomputed_attention_mask:
        model_attention_mask = build_additive_attention_mask(attention_mask, seq_len=input_ids.shape[1])
        logger.info(
            "Precomputed host additive attention mask for fixed single-DP S512 benchmark. "
            "Set BGE_M3_PRECOMPUTE_ATTENTION_MASK=0 to use the on-device builder."
        )

    trace_state = None
    iteration_times = []
    torch_to_ttnn_times = []
    ttnn_to_torch_times = []
    embeddings = None
    dispatch_target = runtime["submeshes"][0] if runtime.get("submeshes") else mesh_device
    restore_async_dispatch = _maybe_enable_async_slow_dispatch(dispatch_target, enable_async_slow_dispatch)
    try:
        # ---- Warmup / compile ----
        logger.info("Compiling (first forward)...")
        _tracy_signpost("Compilation pass")
        _, compile_conversion_timing = run_embedding_forward(
            runtime,
            input_ids,
            attention_mask,
            token_type_ids,
            profiler,
            "compile_prefill",
            collect_conversion_timing=True,
            position_ids=position_ids,
            model_attention_mask=model_attention_mask,
        )
        logger.info(f"Compile forward: {profiler.get_duration('compile_prefill'):.2f}s")

        if use_trace:
            _tracy_signpost("Trace capture pass")
            logger.info("Capturing trace...")
            trace_state = capture_embedding_trace(
                runtime,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids=position_ids,
                model_attention_mask=model_attention_mask,
            )
            logger.info("Trace captured successfully.")

        # ---- Benchmark iterations ----
        benchmark_mode = "trace replay" if trace_state is not None else "regular forward"
        logger.info(f"Running {num_iterations} benchmark iterations ({benchmark_mode})...")

        for i in range(num_iterations):
            if i == 0:
                _tracy_signpost("Performance pass")
            if trace_state is not None:
                result, conversion_timing = run_embedding_forward_trace(
                    trace_state,
                    profiler,
                    f"inference_prefill_{i}",
                    collect_conversion_timing=True,
                )
            else:
                result, conversion_timing = run_embedding_forward(
                    runtime,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    profiler,
                    f"inference_prefill_{i}",
                    collect_conversion_timing=True,
                    position_ids=position_ids,
                    model_attention_mask=model_attention_mask,
                )

            t = profiler.get_duration(f"inference_prefill_{i}")
            iteration_times.append(t)
            torch_to_ttnn_times.append(conversion_timing["torch_to_ttnn_s"])
            ttnn_to_torch_times.append(conversion_timing["ttnn_to_torch_s"])
            logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

            if embeddings is None:
                embeddings = result
    finally:
        _maybe_restore_async_slow_dispatch(dispatch_target, restore_async_dispatch)

    # ---- Compute metrics ----
    avg_prefill_time = sum(iteration_times) / len(iteration_times)
    best_prefill_time = min(iteration_times)

    embeddings_per_sec_avg = batch_size / avg_prefill_time
    embeddings_per_sec_best = batch_size / best_prefill_time
    tokens_per_sec_avg = total_input_tokens / avg_prefill_time
    tokens_per_sec_best = total_input_tokens / best_prefill_time
    avg_torch_to_ttnn_time = sum(torch_to_ttnn_times) / len(torch_to_ttnn_times)
    best_torch_to_ttnn_time = min(torch_to_ttnn_times)
    avg_ttnn_to_torch_time = sum(ttnn_to_torch_times) / len(ttnn_to_torch_times)
    best_ttnn_to_torch_time = min(ttnn_to_torch_times)

    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "compile_torch_to_ttnn_time": compile_conversion_timing["torch_to_ttnn_s"],
        "compile_ttnn_to_torch_time": compile_conversion_timing["ttnn_to_torch_s"],
        "avg_prefill_time": avg_prefill_time,
        "best_prefill_time": best_prefill_time,
        "avg_torch_to_ttnn_time": avg_torch_to_ttnn_time,
        "best_torch_to_ttnn_time": best_torch_to_ttnn_time,
        "avg_ttnn_to_torch_time": avg_ttnn_to_torch_time,
        "best_ttnn_to_torch_time": best_ttnn_to_torch_time,
        "embeddings/s_avg": embeddings_per_sec_avg,
        "embeddings/s_best": embeddings_per_sec_best,
        "prefill_t/s_avg": tokens_per_sec_avg,
        "prefill_t/s_best": tokens_per_sec_best,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "data_parallel": tt_data_parallel,
        "input_seq_len": isl,
        "max_seq_len": max_seq_len,
        "total_input_tokens": total_input_tokens,
    }

    # ---- Print results ----
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {tt_data_parallel}")
    logger.info(f"  Global batch size:    {batch_size}")
    logger.info(f"  Batch per DP group:   {batch_per_dp}")
    logger.info(f"  Input seq length:     {isl}")
    logger.info(f"  Max seq length:       {max_seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_prefill_time * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_prefill_time * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {embeddings_per_sec_avg:.1f}")
    logger.info(f"  Best embeddings/s:    {embeddings_per_sec_best:.1f}")
    logger.info(f"  Avg tokens/s:         {tokens_per_sec_avg:.0f}")
    logger.info(f"  Best tokens/s:        {tokens_per_sec_best:.0f}")
    logger.info("-" * 60)
    logger.info(f"  Avg torch->ttnn:      {avg_torch_to_ttnn_time * 1000:.1f}ms")
    logger.info(f"  Best torch->ttnn:     {best_torch_to_ttnn_time * 1000:.1f}ms")
    logger.info(f"  Avg ttnn->torch:      {avg_ttnn_to_torch_time * 1000:.1f}ms")
    logger.info(f"  Best ttnn->torch:     {best_ttnn_to_torch_time * 1000:.1f}ms")
    logger.info("=" * 60)

    # ---- Cosine similarity sanity check (only for real text inputs) ----
    if input_seq_len is None and tt_data_parallel <= 1 and embeddings is not None and batch_size >= 2:
        emb_np = embeddings.float().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        elif emb_np.ndim > 2:
            emb_np = emb_np.reshape(batch_size, -1)

        sim = cosine_similarity(emb_np)
        logger.info(f"  Cosine similarity [0,1] = {sim[0, 1]:.4f} (should be high, both AI-related)")
        if batch_size >= 4:
            logger.info(f"  Cosine similarity [0,3] = {sim[0, 3]:.4f} (should be low, AI vs weather)")

    # ---- CI benchmark data ----
    profiler.end("run")

    if is_ci_env:
        model_name = model_args.hf_model_name if hasattr(model_args, "hf_model_name") else MODEL_NAME
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="embedding",
            num_layers=getattr(model_args, "n_layers", 0),
            batch_size=batch_size,
            config_params={
                "data_parallel": tt_data_parallel,
                "tensor_parallel": num_devices // max(tt_data_parallel, 1),
            },
            input_sequence_length=isl,
            output_sequence_length=0,
        )

    if trace_state is not None:
        trace_state["model"].release_trace()


@pytest.mark.parametrize(
    "tt_data_parallel",
    [1, 2, 3, 4],  # 8, 32]
    ids=[
        "dp1",
        "dp2",
        "dp3",
        "dp4",
        # "dp8",
        # "dp32",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [resolve_mesh_device_param()],
    indirect=True,
)
def test_embedding_perf_data_parallel(mesh_device, tt_data_parallel, is_ci_env):
    """
    DP throughput perf test at fixed sequence settings.
    Global batch is derived from per-device batch to keep load per chip constant.
    """
    batch_per_device = 32
    global_batch = batch_per_device * tt_data_parallel
    logger.info(
        f"DP perf case: tt_data_parallel={tt_data_parallel}, "
        f"batch_per_device={batch_per_device}, global_batch={global_batch}, "
        "input_seq_len=512, max_seq_len=512"
    )
    test_embedding_perf(
        mesh_device=mesh_device,
        batch_size=global_batch,
        max_seq_len=512,
        input_seq_len=512,
        num_iterations=10,
        tt_data_parallel=tt_data_parallel,
        is_ci_env=is_ci_env,
    )


@pytest.mark.parametrize(
    "global_batch",
    [96, 128],
    ids=[
        "dp4-gb96",
        "dp4-gb128",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [resolve_mesh_device_param()],
    indirect=True,
)
def test_embedding_perf_data_parallel_dp4_batch_sweep(mesh_device, global_batch, is_ci_env):
    """
    Fixed-DP=4 perf cases with requested global batches.
    """
    tt_data_parallel = 4
    if global_batch % tt_data_parallel != 0:
        pytest.skip(f"global_batch={global_batch} is not divisible by tt_data_parallel={tt_data_parallel}")

    batch_per_device = global_batch // tt_data_parallel
    logger.info(
        f"DP4 batch sweep case: tt_data_parallel={tt_data_parallel}, "
        f"batch_per_device={batch_per_device}, global_batch={global_batch}, "
        "input_seq_len=512, max_seq_len=512"
    )
    test_embedding_perf(
        mesh_device=mesh_device,
        batch_size=global_batch,
        max_seq_len=512,
        input_seq_len=512,
        num_iterations=10,
        tt_data_parallel=tt_data_parallel,
        is_ci_env=is_ci_env,
    )


@pytest.mark.parametrize(
    "input_seq_len",
    [1024, 2048, 4096, 8192],
    ids=[
        "batch32dp1-isl1024",
        "batch32dp1-isl2048",
        "batch32dp1-isl4096",
        "batch32dp1-isl8192-2x16",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
def test_embedding_perf_batch32_long_isl_single_device(mesh_device, input_seq_len, is_ci_env):
    """
    Single-device long-ISL perf sweep with fixed batch 32.
    ISL=8192 runs as two sequential batch-16 forwards, measured as one combined iteration.
    """
    batch_size = 32
    tt_data_parallel = 1
    num_iterations = 10
    max_seq_len = input_seq_len
    split_8192 = input_seq_len == 8192

    profiler = BenchmarkProfiler()
    profiler.start("run")

    device_name_info = determine_device_name(mesh_device)
    tt_device_name = device_name_info[0] if isinstance(device_name_info, tuple) else str(device_name_info)
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    if num_devices < 1:
        pytest.skip("No devices available")

    runtime_batch = 16 if split_8192 else batch_size
    logger.info(
        f"Long-ISL build: global_batch={runtime_batch}, tt_data_parallel={tt_data_parallel}, "
        f"max_seq_len={max_seq_len}, input_seq_len={input_seq_len}, device={tt_device_name}"
    )
    profiler.start("build_model")
    runtime, model_args, tokenizer = prepare_embedding_model(
        mesh_device,
        global_batch_size=runtime_batch,
        max_seq_len=max_seq_len,
        tt_data_parallel=tt_data_parallel,
    )
    profiler.end("build_model")
    logger.info(
        f"Model built in {profiler.get_duration('build_model'):.1f}s "
        f"(global_dp={runtime['global_data_parallel']}, local_dp={runtime['local_data_parallel']})"
    )

    profiler.start("loading_inputs")
    input_ids, attention_mask, token_type_ids, prompt_lens = generate_synthetic_inputs(
        tokenizer, batch_size, input_seq_len
    )
    profiler.end("loading_inputs")

    total_input_tokens = sum(prompt_lens)
    logger.info(f"Prepared {batch_size} inputs, ISL={input_seq_len}, total tokens = {total_input_tokens}")

    if split_8192:
        first_pass_inputs = (
            input_ids[:16],
            attention_mask[:16],
            token_type_ids[:16],
        )
        second_pass_inputs = (
            input_ids[16:],
            attention_mask[16:],
            token_type_ids[16:],
        )
        logger.info("ISL 8192 special path: two batch-16 forwards per iteration (combined latency).")

    logger.info("Compiling (first forward)...")
    _tracy_signpost("Compilation pass")
    if split_8192:
        _, compile_conversion_timing = run_embedding_forward_two_pass(
            runtime,
            first_pass_inputs,
            second_pass_inputs,
            profiler=profiler,
            step_name="compile_prefill",
            collect_conversion_timing=True,
        )
    else:
        _, compile_conversion_timing = run_embedding_forward(
            runtime,
            input_ids,
            attention_mask,
            token_type_ids,
            profiler,
            "compile_prefill",
            collect_conversion_timing=True,
        )
    logger.info(f"Compile forward: {profiler.get_duration('compile_prefill'):.2f}s")

    logger.info("Running %d benchmark iterations...", num_iterations)
    iteration_times = []
    torch_to_ttnn_times = []
    ttnn_to_torch_times = []
    embeddings = None

    for i in range(num_iterations):
        if i == 0:
            _tracy_signpost("Performance pass")
        if split_8192:
            result, conversion_timing = run_embedding_forward_two_pass(
                runtime,
                first_pass_inputs,
                second_pass_inputs,
                profiler=profiler,
                step_name=f"inference_prefill_{i}",
                collect_conversion_timing=True,
            )
        else:
            result, conversion_timing = run_embedding_forward(
                runtime,
                input_ids,
                attention_mask,
                token_type_ids,
                profiler,
                f"inference_prefill_{i}",
                collect_conversion_timing=True,
            )

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        torch_to_ttnn_times.append(conversion_timing["torch_to_ttnn_s"])
        ttnn_to_torch_times.append(conversion_timing["ttnn_to_torch_s"])
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")
        if embeddings is None:
            embeddings = result

    avg_prefill_time = sum(iteration_times) / len(iteration_times)
    best_prefill_time = min(iteration_times)
    embeddings_per_sec_avg = batch_size / avg_prefill_time
    embeddings_per_sec_best = batch_size / best_prefill_time
    tokens_per_sec_avg = total_input_tokens / avg_prefill_time
    tokens_per_sec_best = total_input_tokens / best_prefill_time
    avg_torch_to_ttnn_time = sum(torch_to_ttnn_times) / len(torch_to_ttnn_times)
    best_torch_to_ttnn_time = min(torch_to_ttnn_times)
    avg_ttnn_to_torch_time = sum(ttnn_to_torch_times) / len(ttnn_to_torch_times)
    best_ttnn_to_torch_time = min(ttnn_to_torch_times)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  BGE-M3 Long-ISL Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Data parallel:        {tt_data_parallel}")
    logger.info(f"  Global batch size:    {batch_size}")
    logger.info(f"  Input seq length:     {input_seq_len}")
    logger.info(f"  Max seq length:       {max_seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {profiler.get_duration('build_model'):.1f}s")
    logger.info(f"  Compile (1st run):    {profiler.get_duration('compile_prefill'):.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg prefill time:     {avg_prefill_time * 1000:.1f}ms")
    logger.info(f"  Best prefill time:    {best_prefill_time * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {embeddings_per_sec_avg:.1f}")
    logger.info(f"  Best embeddings/s:    {embeddings_per_sec_best:.1f}")
    logger.info(f"  Avg tokens/s:         {tokens_per_sec_avg:.0f}")
    logger.info(f"  Best tokens/s:        {tokens_per_sec_best:.0f}")
    logger.info("-" * 60)
    logger.info(f"  Avg torch->ttnn:      {avg_torch_to_ttnn_time * 1000:.1f}ms")
    logger.info(f"  Best torch->ttnn:     {best_torch_to_ttnn_time * 1000:.1f}ms")
    logger.info(f"  Avg ttnn->torch:      {avg_ttnn_to_torch_time * 1000:.1f}ms")
    logger.info(f"  Best ttnn->torch:     {best_ttnn_to_torch_time * 1000:.1f}ms")
    logger.info("=" * 60)

    profiler.end("run")
    if is_ci_env:
        model_name = model_args.hf_model_name if hasattr(model_args, "hf_model_name") else MODEL_NAME
        measurements = {
            "compile_prefill": profiler.get_duration("compile_prefill"),
            "compile_torch_to_ttnn_time": compile_conversion_timing["torch_to_ttnn_s"],
            "compile_ttnn_to_torch_time": compile_conversion_timing["ttnn_to_torch_s"],
            "avg_prefill_time": avg_prefill_time,
            "best_prefill_time": best_prefill_time,
            "avg_torch_to_ttnn_time": avg_torch_to_ttnn_time,
            "best_torch_to_ttnn_time": best_torch_to_ttnn_time,
            "avg_ttnn_to_torch_time": avg_ttnn_to_torch_time,
            "best_ttnn_to_torch_time": best_ttnn_to_torch_time,
            "embeddings/s_avg": embeddings_per_sec_avg,
            "embeddings/s_best": embeddings_per_sec_best,
            "prefill_t/s_avg": tokens_per_sec_avg,
            "prefill_t/s_best": tokens_per_sec_best,
            "build_model_time": profiler.get_duration("build_model"),
            "batch_size": batch_size,
            "data_parallel": tt_data_parallel,
            "input_seq_len": input_seq_len,
            "max_seq_len": max_seq_len,
            "total_input_tokens": total_input_tokens,
        }
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name=model_name,
            ml_model_type="embedding",
            num_layers=getattr(model_args, "n_layers", 0),
            batch_size=batch_size,
            config_params={
                "data_parallel": tt_data_parallel,
                "tensor_parallel": 1,
            },
            input_sequence_length=input_seq_len,
            output_sequence_length=0,
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BGE-M3 performance demo")
    parser.add_argument("--batch-size", type=int, default=1, help="Global batch size")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--input-seq-len", type=int, default=None, help="Synthetic input sequence length")
    parser.add_argument("--input-file", type=str, default=None, help="Optional JSON file with input texts")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID (single-device mode)")
    parser.add_argument(
        "--tt-data-parallel",
        type=int,
        default=None,
        help="Data parallel groups to build (default: auto from opened device)",
    )
    args = parser.parse_args()

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id, l1_small_size=32768, trace_region_size=50000000, num_command_queues=1
    )

    try:
        profiler = BenchmarkProfiler()
        profiler.start("run")

        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        tt_data_parallel = args.tt_data_parallel if args.tt_data_parallel is not None else num_devices
        if tt_data_parallel < 1:
            raise ValueError(f"--tt-data-parallel must be >= 1, got {tt_data_parallel}")
        if tt_data_parallel > num_devices:
            raise ValueError(
                f"--tt-data-parallel={tt_data_parallel} requires {tt_data_parallel} devices, but opened device exposes {num_devices}"
            )
        if args.batch_size % tt_data_parallel != 0:
            raise ValueError(
                f"--batch-size={args.batch_size} must be divisible by --tt-data-parallel={tt_data_parallel}"
            )
        logger.info(
            f"Standalone runtime config: global_batch={args.batch_size}, tt_data_parallel={tt_data_parallel}, num_devices={num_devices}"
        )

        runtime, model_args, tokenizer = prepare_embedding_model(
            device,
            global_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            tt_data_parallel=tt_data_parallel,
        )

        if args.input_seq_len is not None:
            input_ids, attention_mask, token_type_ids, prompt_lens = generate_synthetic_inputs(
                tokenizer, args.batch_size, args.input_seq_len
            )
        else:
            texts = load_input_texts(args.input_file, args.batch_size)
            input_ids, attention_mask, token_type_ids, prompt_lens = tokenize_and_pad(
                tokenizer, texts, args.max_seq_len
            )

        total_tokens = sum(prompt_lens)

        logger.info("Compile run...")
        _tracy_signpost("Compilation pass")
        _ = run_embedding_forward(runtime, input_ids, attention_mask, token_type_ids, profiler, "compile_prefill")

        logger.info(f"Benchmarking {args.iterations} iterations...")
        times = []
        for i in range(args.iterations):
            if i == 0:
                _tracy_signpost("Performance pass")
            _ = run_embedding_forward(
                runtime, input_ids, attention_mask, token_type_ids, profiler, f"inference_prefill_{i}"
            )
            t = profiler.get_duration(f"inference_prefill_{i}")
            times.append(t)
            logger.info(f"  Iter {i}: {t * 1000:.1f}ms")

        avg_t = sum(times) / len(times)
        best_t = min(times)
        logger.info("")
        logger.info(
            f"Avg: {avg_t * 1000:.1f}ms | {args.batch_size / avg_t:.1f} emb/s | {total_tokens / avg_t:.0f} tok/s"
        )
        logger.info(
            f"Best: {best_t * 1000:.1f}ms | {args.batch_size / best_t:.1f} emb/s | {total_tokens / best_t:.0f} tok/s"
        )

        profiler.end("run")

    finally:
        ttnn.close_device(device)
