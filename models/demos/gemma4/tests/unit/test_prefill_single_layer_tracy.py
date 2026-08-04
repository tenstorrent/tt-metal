# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-decoder-layer prefill Tracy profile for ``ops_perf_results_*.csv``.

Same Tracy workflow as ``test_prefill_trace_tracy_csv.py`` (load, capture, warm
replay, signposted measured replay), but the decoder stack is truncated to the
smallest prefix that contains the requested attention type. The point is a CSV
you can read: one decoder block's ops instead of the 30-60 identical repeats a
full-model profile emits, so per-op device time is attributable by eye.

``layer_type`` selects the profiled block:

* ``sliding`` -> ``num_layers=1``. Layer 0 is always ``sliding_attention``, so
  the CSV holds embedding, exactly one decoder layer, final norm, and lm_head.
* ``full`` -> ``num_layers = <index of first full_attention layer> + 1``. Gemma4
  interleaves N sliding layers before each global one (5+1 on 12B/26B/31B), and
  ``create_tt_model`` can only truncate from the front, so the profiled global
  layer arrives with its sliding prefix. The full-attention block is the *last*
  decoder block in the CSV; ops before it are the repeating sliding blocks.

Batch is fixed at 1 — batch sweeps belong in the full-model
``test_prefill_trace_tracy_csv.py``. Only trace replay is inference: filter the
CSV to rows between the ``start`` and ``stop`` signposts.

Example (31B blackhole 1x4, one sliding layer at ISL 4096):

    rm -rf generated/profiler/.logs generated/profiler/reports
    export HF_MODEL=google/gemma-4-31b-it
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

    python -m tracy -p -r -v -m pytest \\
        models/demos/gemma4/tests/unit/test_prefill_single_layer_tracy.py \\
        -k "sliding-prefill_4096-1x4" -v -s --timeout=1800

    python tools/tracy/process_ops_logs.py --date
    # Filter ops_perf_results_*.csv to rows between signposts "start" and "stop"
"""

import pytest
from loguru import logger

from models.demos.gemma4.demo.text_demo import _maybe_xfail_batch_prefill_dram
from models.demos.gemma4.tt.generator_trace import (
    GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
    GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
    GEMMA4_TRACE_PREFILL_SEQ_LENS,
    can_gemma4_enable_prefill_trace,
)
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from ..test_factory import (
    TestFactory,
    _get_model_path,
    find_layer_idx,
    num_layers_for_full_attention_group,
    parametrize_mesh_with_fabric,
)
from .test_prefill_trace_parity import _PREFILL_TRACE_BUCKETS
from .tracy_prefill_common import run_prefill_trace_tracy_session

# Single-layer profiling is about per-op device time inside one decoder block,
# which does not need the multi-user batching matrix.
_SINGLE_LAYER_BATCH_SIZE = 1

_LAYER_TYPES = {"sliding": "sliding_attention", "full": "full_attention"}


def _resolve_num_layers(hf_config, layer_type_id):
    """Smallest decoder-stack prefix that contains a layer of the requested type.

    Truncation is front-anchored (``create_tt_model`` only takes a layer count),
    so ``sliding`` resolves to 1 and ``full`` to the index of the first global
    layer plus one. Skips when the variant has no layer of that type.
    """
    layer_type = _LAYER_TYPES[layer_type_id]
    try:
        first_idx = find_layer_idx(hf_config, layer_type)
    except ValueError:
        pytest.skip(f"No {layer_type} layer in this model variant")
    if layer_type_id == "full":
        return num_layers_for_full_attention_group(hf_config)
    if first_idx != 0:
        pytest.skip(f"{layer_type} does not start the stack (first at index {first_idx}) — cannot isolate by prefix")
    return 1


@pytest.mark.gemma4_prefill_trace
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _PREFILL_TRACE_BUCKETS, ids=lambda n: f"prefill_{n}")
@pytest.mark.parametrize("layer_type", sorted(_LAYER_TYPES), ids=lambda t: t)
def test_prefill_single_layer_tracy_csv(layer_type, prefill_len, mesh_device, reset_seeds, request):
    """Tracy session over a truncated stack: load, capture, warm replay, signposted replay."""
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models disable prefill trace")
    # Truncating the stack leaves ``num_kv_shared_layers`` at its full-model value,
    # which drives ``first_shared_idx = num_hidden_layers - num_kv_shared`` negative
    # in Gemma4Model. Only E2B/E4B ship kv sharing and those are already PLI-skipped
    # above; keep the guard so a future kv-sharing variant skips instead of misbuilding.
    if int(getattr(hf_config, "num_kv_shared_layers", 0) or 0) > 0:
        pytest.skip("kv-shared variants cannot have their layer stack truncated")

    kernel_len = get_padded_prefill_len(prefill_len)
    if kernel_len not in GEMMA4_TRACE_PREFILL_SEQ_LENS:
        pytest.skip(f"kernel_len={kernel_len} not in trace ISL buckets {GEMMA4_TRACE_PREFILL_SEQ_LENS}")

    max_padded_batch = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= _SINGLE_LAYER_BATCH_SIZE)
    if not can_gemma4_enable_prefill_trace(kernel_len, batch_size=max_padded_batch):
        pytest.skip(
            f"prefill trace disabled for padded_batch={max_padded_batch} x kernel={kernel_len} "
            f"(ISL>{GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN} or "
            f"batch×kernel>={GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS})"
        )

    num_layers = _resolve_num_layers(hf_config, layer_type)
    profiled_idx = num_layers - 1
    logger.info(
        "Single-layer prefill Tracy profile: layer_type={} num_layers={} profiled_layer_idx={} "
        "layer_types={} kernel_len={}",
        _LAYER_TYPES[layer_type],
        num_layers,
        profiled_idx,
        list(hf_config.layer_types[:num_layers]),
        kernel_len,
    )
    if num_layers > 1:
        logger.info(
            "Stack has a {}-layer sliding prefix before the profiled {} block — the profiled "
            "block is the last decoder block in ops_perf_results_*.csv",
            num_layers - 1,
            _LAYER_TYPES[layer_type],
        )

    model_path = _get_model_path()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, _SINGLE_LAYER_BATCH_SIZE, prefill_len)

    generator, _kv_caches = run_prefill_trace_tracy_session(
        mesh_device,
        model_path,
        _SINGLE_LAYER_BATCH_SIZE,
        prefill_len,
        hf_config.vocab_size,
        emit_signposts=True,
        num_layers=num_layers,
    )

    # The whole point of this profile is a CSV holding one decoder block per
    # attention type; if truncation silently stopped working the CSV would be a
    # full-model profile and the numbers would be read as per-layer.
    built_layers = len(generator.model[0].layers)
    assert built_layers == num_layers, f"expected a {num_layers}-layer stack, built {built_layers}"
    assert (
        generator.model[0].hf_config.layer_types[profiled_idx] == _LAYER_TYPES[layer_type]
    ), f"layer {profiled_idx} is not {_LAYER_TYPES[layer_type]}"
