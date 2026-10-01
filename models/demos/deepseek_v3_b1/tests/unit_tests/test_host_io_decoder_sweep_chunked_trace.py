# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.tests.unit_tests import run_host_io_decoder_sweep
from models.demos.deepseek_v3_b1.tests.unit_tests.debug_trace_io import (
    ChunkedTraceReader,
    load_reference_kv,
    load_reference_trace,
    model_trace_id,
)

DEFAULT_TRACE_ROOT = Path("/mnt/models/ref_data/debug_traces")

TRACE_CASES = [
    pytest.param(
        "DeepSeek_R1_0528",
        "vllm-97854ebb-128000tok",
        id="ds-r1-0528-97854ebb-128k",
    )
]


def _trace_dir(trace_root: Path, model_id: str, prompt_id: str) -> Path:
    return trace_root / model_trace_id(model_id) / prompt_id


def _trace_root() -> Path:
    return Path(os.environ.get(run_host_io_decoder_sweep.TRACE_ROOT_ENV, str(DEFAULT_TRACE_ROOT)))


def _require_host_io_sweep_env() -> None:
    if not is_slow_dispatch():
        pytest.skip("HostIoDecoderStage sweep requires TT_METAL_SLOW_DISPATCH_MODE=1")
    if os.environ.get("TT_METAL_ALLOCATOR_MODE_HYBRID") != "1":
        pytest.skip("HostIoDecoderStage sweep requires TT_METAL_ALLOCATOR_MODE_HYBRID=1")


def _assert_chunked_trace_is_readable(trace_root: Path, model_id: str, prompt_id: str) -> None:
    trace_dir = _trace_dir(trace_root, model_id, prompt_id)
    if not (trace_dir / "index.json").is_file():
        logger.warning(f"chunked 128K trace is not present: {trace_dir}; this test can only be run on exabox machines")
        pytest.skip(f"chunked 128K trace is not present: {trace_dir} (only available on exabox machines)")

    reader = ChunkedTraceReader(trace_dir)
    assert reader.index["layout"] == "chunked_group_a_v1"
    assert reader.n_layers == 61

    row_count = 65 + 1024 - 1
    rows = slice(0, row_count)
    for layer in [0, 1, 3, 10, 30, 45, 55, 59, 60]:
        trace = load_reference_trace(
            trace_root,
            model_id=model_id,
            prompt_id=prompt_id,
            layer=layer,
            num_decode_steps=1024,
        )
        kv = reader.get_kv(layer, rows=rows)
        kv_trace = load_reference_kv(
            trace_root,
            model_id=model_id,
            prompt_id=prompt_id,
            layer=layer,
            num_decode_steps=1024,
            target_layout="trace",
        )
        kv_tt = load_reference_kv(
            trace_root,
            model_id=model_id,
            prompt_id=prompt_id,
            layer=layer,
            num_decode_steps=1024,
            target_layout="tt_device",
        )

        assert trace["input"].shape == (row_count, 7168)
        assert trace["output"].shape == (row_count, 7168)
        assert trace["input"].dtype == torch.bfloat16
        assert trace["output"].dtype == torch.bfloat16
        assert kv.shape == (row_count, 576)
        assert kv.dtype == torch.bfloat16
        assert kv_trace.shape == (1, row_count, 576)
        assert kv_trace.dtype == torch.bfloat16
        assert kv_tt.shape == (1, row_count, 576)
        assert kv_tt.dtype == torch.bfloat16
        assert torch.equal(kv_trace, kv.unsqueeze(0))
        assert torch.equal(kv_trace[:, :, :512], kv_tt[:, :, :512])
        assert torch.equal(kv_trace[:, :, 512:544], kv_tt[:, :, 512::2])
        assert torch.equal(kv_trace[:, :, 544:], kv_tt[:, :, 513::2])

        expected_input = (
            reader.get_decoder_input(0, rows=rows) if layer == 0 else reader.get_decoder_output(layer - 1, rows=rows)
        )
        assert torch.equal(trace["input"], expected_input)


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_load_reference_kv_cache_chunked_trace_layout(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    trace_dir = _trace_dir(trace_root, model_id, prompt_id)
    if not (trace_dir / "index.json").is_file():
        pytest.skip("chunked 128K trace is not present")

    num_decode_steps = 1024
    row_count = 65 + num_decode_steps - 1
    rows = slice(0, row_count)
    layer = 30
    reader = ChunkedTraceReader(trace_dir)

    kv = reader.get_kv(layer, rows=rows)
    kv_trace = load_reference_kv(
        trace_root,
        model_id=model_id,
        prompt_id=prompt_id,
        layer=layer,
        num_decode_steps=num_decode_steps,
        target_layout="trace",
    )
    kv_tt = load_reference_kv(
        trace_root,
        model_id=model_id,
        prompt_id=prompt_id,
        layer=layer,
        num_decode_steps=num_decode_steps,
        target_layout="tt_device",
    )

    assert kv.shape == (row_count, 576)
    assert kv.dtype == torch.bfloat16
    assert kv_trace.shape == (1, row_count, 576)
    assert kv_trace.dtype == torch.bfloat16
    assert kv_tt.shape == (1, row_count, 576)
    assert kv_tt.dtype == torch.bfloat16
    assert torch.equal(kv_trace, kv.unsqueeze(0))
    assert torch.equal(kv_trace[:, :, :512], kv_tt[:, :, :512])
    assert torch.equal(kv_trace[:, :, 512:544], kv_tt[:, :, 512::2])
    assert torch.equal(kv_trace[:, :, 544:], kv_tt[:, :, 513::2])


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_run_host_io_decoder_sweep_rank_parallel_config_contract(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    if not (_trace_dir(trace_root, model_id, prompt_id) / "index.json").is_file():
        pytest.skip("chunked 128K trace is not present")

    args = run_host_io_decoder_sweep._build_arg_parser().parse_args(
        [
            "--decoder-layer-indices",
            "4",
            "5",
            "6",
            "7",
            "--trace-root",
            str(trace_root),
            "--model-id",
            model_id,
            "--prompt",
            prompt_id,
        ]
    )

    for rank, expected_layer in enumerate([4, 5, 6, 7]):
        config = run_host_io_decoder_sweep._config_from_args_for_rank(args, rank=rank)
        assert config.decoder_layer_index == expected_layer
        assert config.trace_root == trace_root
        assert config.model_id == model_id
        assert config.prompt_names == (prompt_id,)
        assert config.num_replication_slots == 1
        assert config.validate_hidden_states_cross_trace
        assert not config.validate_hidden_states_cross_slot
        assert not config.validate_kv_cache_cross_slot
        assert not config.validate_kv_cache_cross_trace
        assert (
            run_host_io_decoder_sweep._build_device_params(config, layer_parallel=True)["fabric_config"]
            == ttnn.FabricConfig.FABRIC_2D_TORUS_Y
        )


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_run_host_io_decoder_sweep_multi_slot_stress_config_contract(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    if not (_trace_dir(trace_root, model_id, prompt_id) / "index.json").is_file():
        pytest.skip("chunked 128K trace is not present")

    args = run_host_io_decoder_sweep._build_arg_parser().parse_args(
        [
            "--decoder-layer-indices",
            "4",
            "5",
            "6",
            "7",
            "--trace-root",
            str(trace_root),
            "--model-id",
            model_id,
            "--prompt",
            prompt_id,
            "--num-decode-steps",
            "4096",
            "--num-replication-slots",
            "8",
            "--validate-hidden-states-cross-trace",
            "--pcc-threshold",
            "0.97",
            "--validate-hidden-states-cross-slot",
            "--validate-kv-cache-cross-slot",
            "--no-dump-hidden-states",
            "--no-dump-kv-cache",
        ]
    )

    for rank, expected_layer in enumerate([4, 5, 6, 7]):
        config = run_host_io_decoder_sweep._config_from_args_for_rank(args, rank=rank)
        assert config.decoder_layer_index == expected_layer
        assert config.trace_root == trace_root
        assert config.model_id == model_id
        assert config.prompt_names == (prompt_id,)
        assert config.num_decode_steps == 4096
        assert config.num_replication_slots == 8
        assert config.validate_hidden_states_cross_trace
        assert config.validate_hidden_states_cross_slot
        assert config.validate_kv_cache_cross_slot
        assert not config.validate_kv_cache_cross_trace
        assert not config.dump_hidden_states
        assert not config.dump_kv_cache


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_run_host_io_decoder_sweep_chunked_trace_smoke(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    _assert_chunked_trace_is_readable(trace_root, model_id, prompt_id)
    _require_host_io_sweep_env()
    if ttnn.get_num_devices() < 8:
        pytest.skip("HostIoDecoderStage sweep requires 8 devices (4x2 mesh)")

    exit_code = run_host_io_decoder_sweep.main(
        [
            "--decoder-layer-indices",
            "30",
            "--trace-root",
            str(trace_root),
            "--model-id",
            model_id,
            "--prompt",
            prompt_id,
            "--num-decode-steps",
            "1024",
            "--num-replication-slots",
            "1",
            "--validate-hidden-states-cross-trace",
            "--pcc-threshold",
            "0.97",
            "--validate-kv-cache-cross-trace",
            "--kv-cache-pcc-threshold",
            "0.97",
            "--no-validate-kv-cache-cross-slot",
            "--no-dump-hidden-states",
            "--no-dump-kv-cache",
        ]
    )
    assert exit_code == 0


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_run_host_io_decoder_sweep_chunked_trace_world_size_4(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    _assert_chunked_trace_is_readable(trace_root, model_id, prompt_id)
    _require_host_io_sweep_env()
    if ttnn.get_num_devices() < 8:
        pytest.skip("HostIoDecoderStage sweep requires 8 devices per rank (4x2 mesh)")

    hf_model_path = os.environ.get("HF_MODEL", str(run_host_io_decoder_sweep.DEFAULT_HF_MODEL_PATH))
    cache_path = os.environ.get("TT_MDDEL_CACHE", str(run_host_io_decoder_sweep.DEFAULT_CACHE_PATH))
    exit_code = run_host_io_decoder_sweep.main(
        [
            "--decoder-layer-indices",
            "4",
            "5",
            "6",
            "7",
            "--trace-root",
            str(trace_root),
            "--model-id",
            model_id,
            "--prompt",
            prompt_id,
            "--num-decode-steps",
            "1024",
            "--num-replication-slots",
            "1",
            "--validate-hidden-states-cross-trace",
            "--pcc-threshold",
            "0.97",
            "--validate-kv-cache-cross-trace",
            "--kv-cache-pcc-threshold",
            "0.97",
            "--no-validate-kv-cache-cross-slot",
            "--no-dump-hidden-states",
            "--no-dump-kv-cache",
            "--hf-model-path",
            hf_model_path,
            "--cache-path",
            cache_path,
        ]
    )
    assert exit_code == 0


@pytest.mark.parametrize(("model_id", "prompt_id"), TRACE_CASES)
def test_run_host_io_decoder_sweep_chunked_trace_multi_slot_stress(model_id: str, prompt_id: str) -> None:
    trace_root = _trace_root()
    _assert_chunked_trace_is_readable(trace_root, model_id, prompt_id)
    _require_host_io_sweep_env()
    if ttnn.get_num_devices() < 8:
        pytest.skip("HostIoDecoderStage sweep requires 8 devices per rank (4x2 mesh)")

    hf_model_path = os.environ.get("HF_MODEL", str(run_host_io_decoder_sweep.DEFAULT_HF_MODEL_PATH))
    cache_path = os.environ.get("TT_MDDEL_CACHE", str(run_host_io_decoder_sweep.DEFAULT_CACHE_PATH))
    exit_code = run_host_io_decoder_sweep.main(
        [
            "--decoder-layer-indices",
            "4",
            "5",
            "6",
            "7",
            "--trace-root",
            str(trace_root),
            "--model-id",
            model_id,
            "--prompt",
            prompt_id,
            "--num-decode-steps",
            "4096",
            "--num-replication-slots",
            "8",
            "--validate-hidden-states-cross-trace",
            "--pcc-threshold",
            "0.97",
            "--validate-hidden-states-cross-slot",
            "--validate-kv-cache-cross-slot",
            "--no-dump-hidden-states",
            "--no-dump-kv-cache",
            "--hf-model-path",
            hf_model_path,
            "--cache-path",
            cache_path,
        ]
    )
    assert exit_code == 0
