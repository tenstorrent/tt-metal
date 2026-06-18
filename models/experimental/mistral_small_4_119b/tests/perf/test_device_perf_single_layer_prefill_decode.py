# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill+decode device-perf test for Mistral-Small-4-119B.

Two tests live in this file:

  * ``test_profile_single_layer_prefill_decode`` — the Tracy *profile workload*. Builds a
    1-layer ``TtMistral4TextModel`` (text-only, partial Hub weights, no vision tower), prefills
    128 tokens, then runs one decode step at position 128. No PCC assertions — it exists purely
    so Tracy can capture per-op device timings for a single decoder layer. Each measured
    iteration runs **both** prefill and decode inside the ``start``/``stop`` signpost window
    (prefill before decode so phase-split tooling can separate the two in the ops log).

  * ``test_device_perf_single_layer_prefill_decode`` — the *device-perf wrapper*. Runs the
    profile workload above under Tracy via ``run_device_perf`` (which shells out to
    ``pytest <this file>::test_profile_single_layer_prefill_decode`` in a subprocess), then
    writes ``device_perf_*.csv`` plus a partial benchmark JSON via ``prep_device_perf_report``.
    No golden perf assertion — this is for collecting / inspecting reports.

Run the device-perf dump (the normal entry point)::

    pytest models/experimental/mistral_small_4_119b/tests/perf/test_device_perf_single_layer_prefill_decode.py \\
        -v -m models_device_performance_bare_metal

Capture the profile workload standalone under Tracy::

    python -m tracy -p -v -r --dump-device-data-mid-run \\
        pytest models/experimental/mistral_small_4_119b/tests/perf/test_device_perf_single_layer_prefill_decode.py \\
        ::test_profile_single_layer_prefill_decode -v

Analyze the generated ``ops_perf_results_*.csv`` with ``tt-perf-report``. The file contains
``start``/``stop`` signposts; you must pass both explicitly (default mode only anchors on the
last signpost and shows no device ops)::

    tt-perf-report generated/profiler/mistral_small_4_119b_L1_prefill_decode/reports/.../ops_perf_results_*.csv \\
        --start-signpost start --end-signpost stop

Or ignore signposts to include warmup + measured iteration (noisier)::

    tt-perf-report .../ops_perf_results_*.csv --ignore-signposts
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

NUM_LAYERS = 1
PREFILL_SEQ_LEN = 128
DECODE_POS = PREFILL_SEQ_LEN
NUM_WARMUP_ITERS = 1
MAX_SEQ_LEN = PREFILL_SEQ_LEN + 1 + 64

MODEL_NAME = "mistral_small_4_119b_L1_prefill128_decode1"
SUBDIR = "mistral_small_4_119b_L1_prefill_decode"
# Same-file profile workload, addressed by node id; run_device_perf shells out to it.
PROFILE_TEST = (
    "models/experimental/mistral_small_4_119b/tests/perf/test_device_perf_single_layer_prefill_decode.py"
    "::test_profile_single_layer_prefill_decode"
)
COLS = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30000000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


def _run_prefill_decode_step(tt_model, *, input_ids_prefill, input_ids_decode, decode_pos: int) -> int:
    tt_model.prefill_next_token(input_ids_prefill)
    return tt_model.decode_next_token(input_ids_decode, decode_pos)


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.timeout(0)
def test_profile_single_layer_prefill_decode(reset_seeds, mesh_device, batch_size):
    """Prefill 128 + decode 1 on a 1-layer ``TtMistral4TextModel`` (profile target for device perf)."""
    from transformers import AutoConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text_cfg = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text_cfg, attr):
            setattr(text_cfg, attr, "eager")

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(NUM_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    vocab = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].shape[0]

    logger.info(f"Building TtMistral4TextModel ({NUM_LAYERS} layer, max_seq={MAX_SEQ_LEN})...")
    tt_model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text_cfg,
        num_decoder_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )

    # Cache RoPE for every position prefill + decode will touch (one HF call).
    rotary = Mistral4RotaryEmbedding(text_cfg).eval().to(torch.bfloat16)
    total_positions = PREFILL_SEQ_LEN + 1
    full_position_ids = torch.arange(total_positions, dtype=torch.long).unsqueeze(0)
    cos_full, sin_full = rotary(torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16), full_position_ids)
    tt_model.cache_rope_tables(cos_full, sin_full)

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    input_ids_full = torch.randint(0, vocab, (batch_size, total_positions), dtype=torch.long, generator=gen)
    input_ids_prefill = input_ids_full[:, :PREFILL_SEQ_LEN]
    input_ids_decode = input_ids_full[:, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + 1]

    use_signpost = _tracy_signpost_available()
    if use_signpost:
        from tracy import signpost
    else:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    for _ in range(NUM_WARMUP_ITERS):
        _run_prefill_decode_step(
            tt_model,
            input_ids_prefill=input_ids_prefill,
            input_ids_decode=input_ids_decode,
            decode_pos=DECODE_POS,
        )
        ttnn.synchronize_device(mesh_device)

    if use_signpost:
        signpost("start")

    next_token = _run_prefill_decode_step(
        tt_model,
        input_ids_prefill=input_ids_prefill,
        input_ids_decode=input_ids_decode,
        decode_pos=DECODE_POS,
    )
    ttnn.synchronize_device(mesh_device)

    if use_signpost:
        signpost("stop")

    logger.info(
        f"Profile workload complete: prefill_seq_len={PREFILL_SEQ_LEN}, decode_pos={DECODE_POS}, "
        f"next_token={next_token}, signposts={'on' if use_signpost else 'off'}"
    )


@pytest.mark.timeout(0)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_single_layer_prefill_decode():
    """Capture Tracy device perf for 1-layer prefill+decode and dump CSV/JSON report.

    Shells out (via ``run_device_perf``) to the profile workload above by node id, so only
    that single test runs under Tracy in the subprocess.
    """
    command = f"pytest {PROFILE_TEST} -v"
    num_iterations = 1
    batch_size = 1

    post_processed_results = run_device_perf(
        command,
        subdir=SUBDIR,
        num_iterations=num_iterations,
        cols=COLS,
        batch_size=batch_size,
        has_signposts=True,
    )

    logger.info(f"Device perf results for {MODEL_NAME}:\n{post_processed_results}")

    prep_device_perf_report(
        model_name=MODEL_NAME,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="prefill128_decode1_partial_weights",
    )
