# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sampling1D module."""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig, _resolve_sampling1d_config

# ---------------------------------------------------------------------------
# Model name constants (match test_mlp_1d.py naming convention)
# ---------------------------------------------------------------------------
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN3_32B = "Qwen/Qwen3-32B"

_slow = pytest.mark.slow


def _list_collected_sampling_cases() -> list[pytest.param]:
    """
    Collected from TTTv1 demo runs (Phase B of test_case_collection.md).

    Each entry is:
        (mesh_shape, vocab_size, k, p, temp, force_argmax, hf_model_name)

    Source CSVs: sampling_generator_config_collected.csv,
                 sampling_generator_params_collected.csv
    Deduplicated by (topology, vocab, k, p, temp, force_argmax).
    """
    # fmt: off
    return [
        # --- (1,1) Mistral7B v32768 ---
        pytest.param((1, 1), 32768, 1, 1.0, 1.0, False, MISTRAL_7B, id="1x1-Mistral7B-v32768-k1-p1.0-t1.0"),
        pytest.param((1, 1), 32768, 10, 0.9, 1.0, False, MISTRAL_7B, id="1x1-Mistral7B-v32768-k10-p0.9-t1.0", marks=_slow),
        # --- (1,1) Llama8B v128256 ---
        pytest.param((1, 1), 128256, 1, 1.0, 1.0, True, LLAMA_8B, id="1x1-Llama8B-v128256-k1-p1.0-t1.0-argmax"),
        # --- (1,1) Llama1B v128256 ---
        pytest.param((1, 1), 128256, 1, 0.08, 1.0, False, LLAMA_1B, id="1x1-Llama1B-v128256-k1-p0.08-t1.0", marks=_slow),
        pytest.param((1, 1), 128256, 1, 1.0, 1.0, False, LLAMA_1B, id="1x1-Llama1B-v128256-k1-p1.0-t1.0", marks=_slow),
        pytest.param((1, 1), 128256, 10, 0.9, 1.0, False, LLAMA_1B, id="1x1-Llama1B-v128256-k10-p0.9-t1.0", marks=_slow),
        # --- (1,2) Mistral7B v32768 ---
        pytest.param((1, 2), 32768, 1, 1.0, 1.0, False, MISTRAL_7B, id="1x2-Mistral7B-v32768-k1-p1.0-t1.0"),
        pytest.param((1, 2), 32768, 10, 0.9, 1.0, False, MISTRAL_7B, id="1x2-Mistral7B-v32768-k10-p0.9-t1.0", marks=_slow),
        # --- (1,2) Llama8B v128256 ---
        pytest.param((1, 2), 128256, 1, 1.0, 1.0, True, LLAMA_8B, id="1x2-Llama8B-v128256-k1-p1.0-t1.0-argmax"),
        # --- (1,2) Llama1B v128256 ---
        pytest.param((1, 2), 128256, 1, 0.08, 1.0, False, LLAMA_1B, id="1x2-Llama1B-v128256-k1-p0.08-t1.0", marks=_slow),
        pytest.param((1, 2), 128256, 1, 1.0, 1.0, False, LLAMA_1B, id="1x2-Llama1B-v128256-k1-p1.0-t1.0", marks=_slow),
        pytest.param((1, 2), 128256, 10, 0.9, 1.0, False, LLAMA_1B, id="1x2-Llama1B-v128256-k10-p0.9-t1.0", marks=_slow),
        # --- (1,8) Mixtral8x7B v32000 ---
        pytest.param((1, 8), 32000, 1, 0.08, 1.0, False, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-k1-p0.08-t1.0"),
        pytest.param((1, 8), 32000, 1, 1.0, 1.0, False, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-k1-p1.0-t1.0", marks=_slow),
        pytest.param((1, 8), 32000, 10, 0.9, 1.0, False, MIXTRAL_8X7B, id="1x8-Mixtral8x7B-v32000-k10-p0.9-t1.0", marks=_slow),
        # --- (1,8) Mistral7B v32768 ---
        pytest.param((1, 8), 32768, 1, 1.0, 1.0, False, MISTRAL_7B, id="1x8-Mistral7B-v32768-k1-p1.0-t1.0"),
        pytest.param((1, 8), 32768, 10, 0.9, 1.0, False, MISTRAL_7B, id="1x8-Mistral7B-v32768-k10-p0.9-t1.0", marks=_slow),
        # --- (1,8) Llama8B v128256 ---
        pytest.param((1, 8), 128256, 1, 1.0, 1.0, True, LLAMA_8B, id="1x8-Llama8B-v128256-k1-p1.0-t1.0-argmax"),
        # --- (1,8) Llama1B v128256 ---
        pytest.param((1, 8), 128256, 1, 0.08, 1.0, False, LLAMA_1B, id="1x8-Llama1B-v128256-k1-p0.08-t1.0", marks=_slow),
        pytest.param((1, 8), 128256, 1, 1.0, 1.0, False, LLAMA_1B, id="1x8-Llama1B-v128256-k1-p1.0-t1.0", marks=_slow),
        pytest.param((1, 8), 128256, 10, 0.9, 1.0, False, LLAMA_1B, id="1x8-Llama1B-v128256-k10-p0.9-t1.0", marks=_slow),
        # --- (1,8) Qwen3-32B v151936 ---
        pytest.param((1, 8), 151936, 10, 0.9, 1.0, False, QWEN3_32B, id="1x8-Qwen3-32B-v151936-k10-p0.9-t1.0"),
        # --- (1,8) Qwen2.5-72B v152064 ---
        pytest.param((1, 8), 152064, 1, 0.08, 1.0, False, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-k1-p0.08-t1.0"),
        pytest.param((1, 8), 152064, 1, 1.0, 1.0, False, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-k1-p1.0-t1.0", marks=_slow),
        pytest.param((1, 8), 152064, 10, 0.9, 1.0, False, QWEN25_72B, id="1x8-Qwen2.5-72B-v152064-k10-p0.9-t1.0", marks=_slow),

    ]
    # fmt: on


# ==============================================================================
# Unit tests: Config (no device)
# ==============================================================================


class TestConfigUnit:
    def test_config_defaults(self):
        cfg = Sampling1DConfig(vocab_size=1024)
        assert cfg.max_batch_size == 32
        assert cfg.max_top_k == 32
        assert cfg.allow_force_argmax is False
        assert cfg.num_gather_links == 1
        assert cfg.mesh_device is None
        assert cfg.index_offsets is None
        assert cfg.seeds is None

    def test_config_custom(self):
        cfg = Sampling1DConfig(vocab_size=128256, max_top_k=64, allow_force_argmax=True)
        assert cfg.vocab_size == 128256
        assert cfg.max_top_k == 64
        assert cfg.allow_force_argmax is True

    def test_config_not_resolved_without_device(self):
        cfg = Sampling1DConfig(vocab_size=1024)
        assert not cfg.is_resolved()

    def test_config_not_resolved_multi_device_no_ccl(self):
        """is_resolved() returns False when multi-device mesh but tt_ccl is None (line 65)."""
        from unittest.mock import MagicMock

        mock_device = MagicMock()
        mock_device.get_num_devices.return_value = 2
        cfg = Sampling1DConfig(vocab_size=1024, mesh_device=mock_device, tt_ccl=None)
        assert not cfg.is_resolved()


# ==============================================================================
# Device tests
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestSampling1DDevice:
    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_config(self, ttnn_mesh_device, vocab_size):
        cfg = Sampling1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        resolved = _resolve_sampling1d_config(cfg)
        assert resolved.is_resolved()
        assert resolved.start_core is not None
        assert resolved.sampling_memory_config is not None
        assert resolved.index_offsets is not None
        assert resolved.seeds is not None

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers(self, ttnn_mesh_device, vocab_size):
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        sampler.load_device_buffers()
        assert sampler._device_buffers_loaded
        assert isinstance(sampler._index_offsets, ttnn.Tensor)
        assert isinstance(sampler._local_indices, ttnn.Tensor)
        assert isinstance(sampler._seeds, ttnn.Tensor)
        assert isinstance(sampler._user_ids, ttnn.Tensor)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_force_argmax(self, ttnn_mesh_device, vocab_size):
        """allow_force_argmax=True, k/p/temp=None → matches torch.argmax."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, allow_force_argmax=True)
        sampler.load_device_buffers()
        B = sampler.config.max_batch_size

        torch.manual_seed(42)
        logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

        tokens_tt, log_probs = sampler.decode_forward(logits_tt)
        tokens_host = to_torch_auto_compose(tokens_tt)

        expected_argmax = logits_host.float().argmax(dim=-1)
        tokens_flat = tokens_host.flatten()[:B]
        expected_flat = expected_argmax.flatten()[:B]

        assert torch.equal(
            tokens_flat, expected_flat
        ), f"Argmax mismatch: got {tokens_flat[:5]} vs expected {expected_flat[:5]}"

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_error_on_partial_params(self, ttnn_mesh_device, vocab_size):
        """k provided but not p/temp → ValueError."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        logits_host = torch.randn(1, 1, 32, vocab_size, dtype=torch.bfloat16)
        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)
        k_tt = ttnn.from_torch(torch.ones(32), device=ttnn_mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        with pytest.raises(ValueError, match="k, p, temp must all be provided"):
            sampler.decode_forward(logits_tt, k=k_tt)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_model_args(self, ttnn_mesh_device, vocab_size):
        """from_model_args backward compat factory."""

        class MockArgs:
            padded_vocab_size = vocab_size
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        sampler = Sampling1D.from_model_args(ttnn_mesh_device, None, MockArgs())
        assert sampler.config.vocab_size == vocab_size
        assert sampler.config.mesh_device is ttnn_mesh_device

    # ------------------------------------------------------------------
    # CCL introspection (_bind_strategy lines 116-126)
    # ------------------------------------------------------------------

    def test_bind_strategy_ccl_introspection_with_kwargs(self, ttnn_mesh_device):
        """_bind_strategy correctly detects buffer_key/dtype support on line_all_gather."""
        from dataclasses import replace

        sampler = Sampling1D(vocab_size=1024, mesh_device=ttnn_mesh_device)

        class MockCCL:
            def line_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
                return tensor

        sampler.config = replace(sampler.config, tt_ccl=MockCCL())
        sampler._bind_strategy()

        assert sampler._line_all_gather_supports_buffer_key
        assert sampler._line_all_gather_supports_dtype

    def test_bind_strategy_ccl_introspection_no_kwargs(self, ttnn_mesh_device):
        """_bind_strategy detects when line_all_gather does NOT support buffer_key/dtype."""
        from dataclasses import replace

        sampler = Sampling1D(vocab_size=1024, mesh_device=ttnn_mesh_device)

        class MockCCL:
            def line_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links):
                return tensor

        sampler.config = replace(sampler.config, tt_ccl=MockCCL())
        sampler._bind_strategy()

        assert not sampler._line_all_gather_supports_buffer_key
        assert not sampler._line_all_gather_supports_dtype

    def test_bind_strategy_ccl_introspection_exception(self, ttnn_mesh_device):
        """_bind_strategy handles TypeError from inspect.signature gracefully (lines 125-126)."""
        from dataclasses import replace
        from unittest.mock import patch

        sampler = Sampling1D(vocab_size=1024, mesh_device=ttnn_mesh_device)

        class MockCCL:
            def line_all_gather(self, *args, **kwargs):
                return args[0]

        sampler.config = replace(sampler.config, tt_ccl=MockCCL())

        with patch(
            "models.common.modules.sampling.sampling_1d.inspect.signature", side_effect=TypeError("Cannot inspect")
        ):
            sampler._bind_strategy()

        assert not sampler._line_all_gather_supports_buffer_key
        assert not sampler._line_all_gather_supports_dtype

    # ------------------------------------------------------------------
    # Error paths (lines 178, 186)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_error_all_none_no_force_argmax(self, ttnn_mesh_device, vocab_size):
        """decode_forward with all-None k/p/temp when allow_force_argmax=False → ValueError (line 178)."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        logits_host = torch.randn(1, 1, 32, vocab_size, dtype=torch.bfloat16)
        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

        with pytest.raises(ValueError, match="allow_force_argmax is False"):
            sampler.decode_forward(logits_tt)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_forward_dispatches_to_decode_forward(self, ttnn_mesh_device, vocab_size):
        """forward() delegates to decode_forward() (line 186)."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, allow_force_argmax=True)
        logits_host = torch.randn(1, 1, 32, vocab_size, dtype=torch.bfloat16)
        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

        result = sampler.forward(logits_tt)
        assert result is not None
        assert len(result) == 2  # (token_ids, log_probs)

    # ------------------------------------------------------------------
    # _perform_all_gather with line_all_gather (lines 367-377)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_perform_all_gather_with_mock_ccl(self, ttnn_mesh_device, vocab_size):
        """_perform_all_gather passes buffer_key/dtype kwargs when line_all_gather supports them."""
        B, K = 32, 32
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        sampler.load_device_buffers()

        captured_kwargs = {}

        def mock_line_ag(tensor, **kwargs):
            captured_kwargs.update(kwargs)
            return tensor

        sampler._line_all_gather = mock_line_ag
        sampler._line_all_gather_supports_buffer_key = True
        sampler._line_all_gather_supports_dtype = True

        test_tensor = ttnn.from_torch(
            torch.zeros(1, 1, B, K, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        result = sampler._perform_all_gather(
            test_tensor,
            dim=3,
            cluster_axis=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=1,
            buffer_key="TEST_KEY",
            dtype=ttnn.bfloat16,
        )

        assert result is test_tensor
        assert captured_kwargs.get("buffer_key") == "TEST_KEY"
        assert captured_kwargs.get("dtype") == ttnn.bfloat16

    # ------------------------------------------------------------------
    # from_model_args model_config branches (lines 406-408, 416-419)
    # ------------------------------------------------------------------

    def test_from_model_args_with_galaxy_num_links(self, ttnn_mesh_device):
        """from_model_args reads num_gather_links from GALAXY_NUM_LINKS in model_config (lines 406-408)."""

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        model_config = {"GALAXY_NUM_LINKS": 4}
        sampler = Sampling1D.from_model_args(ttnn_mesh_device, None, MockArgs(), model_config=model_config)
        # max_top_k=32 → 32//32=1, max_links=4 → min(1, 4) = 1
        assert sampler.config.num_gather_links == 1

    def test_from_model_args_with_sampling_ag_config(self, ttnn_mesh_device):
        """from_model_args reads allow_force_argmax/num_links/topology from SAMPLING_AG_CONFIG (lines 416-419)."""

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        model_config = {
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": True,
                "num_links": 3,
                "topology": ttnn.Topology.Linear,
            }
        }
        sampler = Sampling1D.from_model_args(ttnn_mesh_device, None, MockArgs(), model_config=model_config)
        assert sampler.config.allow_force_argmax is True
        assert sampler.config.num_argmax_gather_links == 3
        assert sampler.config.ag_topology == ttnn.Topology.Linear

    # ------------------------------------------------------------------
    # Buffer passthrough: _resolve_buf ttnn.Tensor path (lines 493-494)
    # and _materialize ttnn.Tensor path (line 554)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_tensor_passthrough_and_materialize(self, ttnn_mesh_device, vocab_size):
        """Pre-existing ttnn.Tensor passes through _resolve_buf (493-494) and _materialize (554)."""
        cluster_shape = tuple(ttnn_mesh_device.shape)
        num_devices_in_mesh = 2 if list(cluster_shape) == [1, 1] else max(cluster_shape)
        B, K = 32, 32

        replicate_mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape)
        offsets_host = torch.zeros(1, 1, B, K * num_devices_in_mesh, dtype=torch.int64)
        pre_tensor = ttnn.from_torch(
            offsets_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cfg = Sampling1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, index_offsets=pre_tensor)
        resolved = _resolve_sampling1d_config(cfg)
        assert resolved.index_offsets is pre_tensor  # ttnn.Tensor passthrough in _resolve_buf

        sampler = Sampling1D.from_config(cfg)
        sampler.load_device_buffers()
        assert sampler._index_offsets is pre_tensor  # ttnn.Tensor passthrough in _materialize

    # ------------------------------------------------------------------
    # Buffer passthrough: _resolve_buf LazyBuffer path (line 495)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_lazy_buffer_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing LazyBuffer with device=None → resolve_lazy_buffer fills in device (line 495)."""
        from models.common.modules.lazy_buffer import LazyBuffer

        cluster_shape = tuple(ttnn_mesh_device.shape)
        num_devices_in_mesh = 2 if list(cluster_shape) == [1, 1] else max(cluster_shape)
        B, K = 32, 32

        replicate_mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape)
        partial_lb = LazyBuffer(
            source=torch.zeros(1, 1, B, K * num_devices_in_mesh, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=None,  # device not set — resolve_lazy_buffer fills it in
            mesh_mapper=replicate_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cfg = Sampling1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, index_offsets=partial_lb)
        resolved = _resolve_sampling1d_config(cfg)
        assert isinstance(resolved.index_offsets, LazyBuffer)
        assert resolved.index_offsets.device is ttnn_mesh_device  # filled in by resolve_lazy_buffer

    def test_rejects_galaxy(self, ttnn_mesh_device):
        """from_model_args should reject 2D (Galaxy) topologies."""

        class FakeMesh:
            shape = (2, 4)

            def get_num_devices(self):
                return 8

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        with pytest.raises(ValueError, match="1D mesh topologies"):
            Sampling1D.from_model_args(FakeMesh(), None, MockArgs())


# ==============================================================================
# Shared helper: tie-break-aware mismatch assertion
# ==============================================================================


def _assert_no_true_mismatches(
    device_tokens: "torch.Tensor",
    ref_tokens: "torch.Tensor",
    logits_2d: "torch.Tensor",
    *,
    test_label: str,
    quant_tolerance: float = 0.0,
):
    """Classify index mismatches as tie-breaks vs true mismatches. Assert zero true mismatches.

    In low-precision formats (bfloat16, bfloat8_b), multiple elements can share the same
    representable value. When ttnn.topk picks a different index among tied elements, that's
    a TIE-BREAK (acceptable). Only a TRUE-MISMATCH (device picked a genuinely lower value)
    indicates a kernel bug.

    Args:
        device_tokens: [B] int tensor of device-chosen indices.
        ref_tokens:    [B] int tensor of reference indices.
        logits_2d:     [B, V] tensor for value lookups (same precision as device input).
        test_label:    Printed header, e.g. "top-k=1 vs argmax (V=32768, mesh=(1,1))".
        quant_tolerance: Max acceptable value delta for quantization-boundary effects
            (0.0 for bfloat16, ~0.032 for bfloat8_b block-float).
    """
    B = len(device_tokens)
    num_mismatches = (device_tokens != ref_tokens).sum().item()

    true_mismatches = 0
    tie_breaks = 0
    true_mismatch_details = []

    for b in range(B):
        dev_idx = int(device_tokens[b].item())
        ref_idx = int(ref_tokens[b].item())
        if dev_idx == ref_idx:
            continue
        dev_val = logits_2d[b, dev_idx].float().item()
        ref_val = logits_2d[b, ref_idx].float().item()
        delta = ref_val - dev_val

        if delta > quant_tolerance:
            true_mismatches += 1
            true_mismatch_details.append(
                f"  batch {b}: device idx={dev_idx} (val={dev_val:.6f}) "
                f"< ref idx={ref_idx} (val={ref_val:.6f}), delta={delta:.6f}"
            )
        else:
            tie_breaks += 1

    # Report
    print(f"\n--- {test_label} ---")
    print(f"  index mismatches: {num_mismatches}/{B}  (tie-breaks: {tie_breaks}, true: {true_mismatches})")

    if num_mismatches > 0:
        for b in range(B):
            dev_idx = int(device_tokens[b].item())
            ref_idx = int(ref_tokens[b].item())
            if dev_idx == ref_idx:
                continue
            dev_val = logits_2d[b, dev_idx].float().item()
            ref_val = logits_2d[b, ref_idx].float().item()
            delta = ref_val - dev_val
            if delta > quant_tolerance:
                label = "TRUE-MISMATCH"
            elif delta > 0:
                label = "QUANT-BOUNDARY"
            else:
                label = "TIE-BREAK"
            print(
                f"  batch {b} [{label}]: device idx={dev_idx} (val={dev_val:.6f}), "
                f"ref idx={ref_idx} (val={ref_val:.6f}), delta={delta:.6f}"
            )

    # Assert
    assert true_mismatches == 0, (
        f"{test_label}: {true_mismatches}/{B} TRUE mismatches "
        f"(device picked a lower value, not a tie-break)\n"
        f"  ({num_mismatches} total index disagreements, {tie_breaks} are tie-breaks)\n"
        + "\n".join(true_mismatch_details)
    )

    return num_mismatches, tie_breaks, true_mismatches


# ==============================================================================
# VS Reference tests — sampling correctness against torch golden
# ==============================================================================


def _make_logits_tt(logits_host, ttnn_mesh_device, *, shard_vocab=False):
    """Create logits on device. shard_vocab=True shards the last dim across devices (for top-k path)."""
    cluster_shape = tuple(ttnn_mesh_device.shape)
    if not shard_vocab or max(cluster_shape) == 1:
        shard_dims = (None, None)
    elif cluster_shape[-1] >= cluster_shape[-2]:
        shard_dims = (None, -1)
    else:
        shard_dims = (-1, None)
    return ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
    )


def _make_sampling_params(ttnn_mesh_device, B, *, k_val=1, p_val=0.0, temp_val=1.0, cluster_shape=(1, 1)):
    """Helper: create k/p/temp device tensors for Sampling1D.decode_forward()."""
    k = ttnn.from_torch(
        torch.full((B,), k_val, dtype=torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    p = ttnn.from_torch(
        torch.full((B,), p_val),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    temp = ttnn.from_torch(
        torch.full((B,), temp_val),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    return k, p, temp


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "mesh_shape,vocab_size,k_val,p_val,temp_val,force_argmax,hf_model_name",
    _list_collected_sampling_cases(),
)
def test_sampling1d_topk1_vs_argmax(
    ttnn_mesh_device, mesh_shape, vocab_size, k_val, p_val, temp_val, force_argmax, hf_model_name
):
    """
    Top-k=1, p=0.0, temp=1.0 should produce the same result as torch.argmax.

    This is the primary correctness test: with k=1 the sampling degenerates to argmax,
    giving us an exact reference to compare against.
    """
    torch.manual_seed(42)
    B = 32

    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)

    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=1, p_val=0.0, temp_val=1.0, cluster_shape=cluster_shape
    )

    tokens_tt, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    expected = logits_host.float().argmax(dim=-1).flatten()[:B]

    # Compute a bfloat16-aware sharded reference: shard the vocab the same way the device does,
    # find top-1 per shard, then pick the global winner. This accounts for bfloat16 precision
    # loss at shard boundaries that torch.argmax on float32 doesn't see.
    num_devices = max(ttnn_mesh_device.shape)
    if num_devices == 1:
        num_shards = 2  # single device splits vocab in half internally
    else:
        num_shards = num_devices
    logits_bf16 = logits_host.squeeze().bfloat16()  # [B, V] in bfloat16
    shard_size = vocab_size // num_shards
    # For each batch element, find the global argmax by comparing shard-local argmaxes
    bf16_expected = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        best_val = float("-inf")
        best_idx = 0
        for s in range(num_shards):
            shard = logits_bf16[b, s * shard_size : (s + 1) * shard_size]
            local_idx = shard.float().argmax().item()
            local_val = shard[local_idx].float().item()
            if local_val > best_val:
                best_val = local_val
                best_idx = s * shard_size + local_idx
        bf16_expected[b] = best_idx

    _assert_no_true_mismatches(
        tokens_host.long(),
        bf16_expected,
        logits_bf16,
        test_label=f"top-k=1 vs argmax (V={vocab_size}, mesh={tuple(ttnn_mesh_device.shape)})",
    )


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_argmax_vs_reference(ttnn_mesh_device):
    """
    Force-argmax path (k/p/temp=None, allow_force_argmax=True) vs torch.argmax.

    Tests the all-gather-free argmax path on single device.
    """
    torch.manual_seed(99)
    B = 32
    vocab_size = 1024

    sampler = Sampling1D(
        vocab_size=vocab_size,
        mesh_device=ttnn_mesh_device,
        allow_force_argmax=True,
    )

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

    tokens_tt, _ = sampler.decode_forward(logits_tt)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    expected = logits_host.float().argmax(dim=-1).flatten()[:B]

    assert torch.equal(
        tokens_host, expected
    ), f"argmax path mismatch:\n  got:      {tokens_host[:8]}\n  expected: {expected[:8]}"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_topk32_in_range(ttnn_mesh_device):
    """
    Top-k=32, p=1.0 → sampled token must be within the top-32 set for every batch element.

    This is a statistical correctness test: we don't know which token will be sampled
    (it's stochastic), but it MUST be one of the top-32 tokens by logit value.
    """
    torch.manual_seed(77)
    B = 32
    vocab_size = 1024

    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)

    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=32, p_val=1.0, temp_val=1.0, cluster_shape=cluster_shape
    )

    tokens_tt, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    # Compute the top-32 token set per batch element
    _, top32_indices = logits_host.float().squeeze().topk(32, dim=-1)  # [B, 32]

    for b in range(B):
        sampled_token = tokens_host[b].item()
        top32_set = set(top32_indices[b].tolist())
        assert sampled_token in top32_set, f"Batch {b}: sampled token {sampled_token} not in top-32 set"


def _hf_valid_token_set(logits_row: "torch.Tensor", k: int, p: float, temp: float) -> set:
    """Compute the set of tokens eligible under top-k / top-p / temperature filtering.

    Mirrors the pipeline inside ttnn.sampling:
      1. Temperature: divide logits by temp  (skipped if temp == 1.0)
      2. Top-k:       zero out all but top-k tokens
      3. Top-p:       zero out tokens outside the cumulative-probability nucleus

    Uses HuggingFace's LogitsWarper classes so this reference is auditable against
    the transformers library rather than a hand-rolled implementation.

    Returns the set of token ids that have finite logit after filtering — any
    sampled token MUST come from this set.
    """
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    # Warpers expect input_ids (unused here, pass None) and a [1, V] float32 scores tensor.
    scores = logits_row.float().unsqueeze(0)  # [1, V]
    if temp != 1.0:
        scores = TemperatureLogitsWarper(temperature=temp)(None, scores)
    if k > 0:
        scores = TopKLogitsWarper(top_k=k)(None, scores)
    if 0.0 < p < 1.0:
        scores = TopPLogitsWarper(top_p=p)(None, scores)
    # Tokens with -inf logit are filtered out; all others are valid candidates.
    return set(scores[0].isfinite().nonzero(as_tuple=False).squeeze(-1).tolist())


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "k, p, temp, max_boundary_violations",
    [
        # p=0.0 or p=1.0 → no nucleus boundary; token MUST be in top-k, zero tolerance.
        pytest.param(1, 0.0, 1.0, 0, id="k1-p0-t1"),  # degenerates to argmax
        pytest.param(8, 1.0, 1.0, 0, id="k8-p1-t1"),  # pure top-k, no nucleus cut
        # p ∈ (0, 1) → nucleus boundary may differ between bf16 (device) and f32 (HF ref).
        # ttnn.sampling computes softmax+cumsum in bf16; at the p-threshold, a token can
        # fall inside or outside depending on precision. max_boundary_violations is the
        # empirically-calibrated headroom for these boundary disagreements. A regression
        # (violations >> max) indicates a correctness issue beyond precision noise.
        pytest.param(32, 0.5, 1.0, 3, id="k32-p0.5-t1"),  # tight nucleus, neutral temp
        pytest.param(32, 0.9, 2.0, 2, id="k32-p0.9-t2"),  # loose nucleus, flat dist
        pytest.param(32, 0.9, 0.5, 6, id="k32-p0.9-t0.5"),  # loose nucleus, peaked dist
    ],
)
def test_sampling1d_token_in_valid_set(ttnn_mesh_device, k, p, temp, max_boundary_violations):
    """Sampled token must lie within the HF-derived valid candidate set (up to bf16 boundary).

    For each (k, p, temp), the HuggingFace pipeline
        TemperatureLogitsWarper → TopKLogitsWarper → TopPLogitsWarper
    defines which tokens are eligible. Any sampled token MUST come from this set.

    Precision note: ttnn.sampling runs its softmax/cumsum in bfloat16, while the HF
    reference uses float32. Tokens near the nucleus cutoff may fall on different sides
    of the cumulative-probability threshold. max_boundary_violations allows for this;
    it is zero when p ∈ {0.0, 1.0} (no nucleus threshold exists) and small-but-nonzero
    otherwise. Violations significantly above max indicate a real correctness regression.
    """
    torch.manual_seed(42)
    B = 32
    vocab_size = 1024

    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)

    cluster_shape = tuple(ttnn_mesh_device.shape)
    k_tt, p_tt, temp_tt = _make_sampling_params(
        ttnn_mesh_device, B, k_val=k, p_val=p, temp_val=temp, cluster_shape=cluster_shape
    )

    tokens_tt, _ = sampler.decode_forward(logits_tt, k=k_tt, p=p_tt, temp=temp_tt)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    # Build per-batch-element valid sets from bf16 logits (same precision as device input)
    logits_2d = logits_host.squeeze().bfloat16()  # [B, V]
    violations = []
    for b in range(B):
        valid = _hf_valid_token_set(logits_2d[b], k=k, p=p, temp=temp)
        token = tokens_host[b].item()
        if token not in valid:
            violations.append((b, token, len(valid)))

    assert len(violations) <= max_boundary_violations, (
        f"k={k} p={p} temp={temp}: {len(violations)}/{B} tokens outside valid set "
        f"(max allowed={max_boundary_violations} for bf16 boundary):\n"
        + "\n".join(f"  batch {b}: token {tok} not in {n}-token valid set" for b, tok, n in violations[:5])
    )


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_deterministic_with_same_seed(ttnn_mesh_device):
    """
    Two decode_forward calls with the same seed tensor should produce the same tokens.
    """
    torch.manual_seed(42)
    B = 32
    vocab_size = 1024

    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=32, p_val=0.9, temp_val=0.8, cluster_shape=cluster_shape
    )

    # Use explicit seed tensor
    seed_tensor = ttnn.from_torch(
        torch.arange(B, dtype=torch.int64).to(torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # First call
    logits_tt1 = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)
    tokens1, _ = sampler.decode_forward(logits_tt1, k=k, p=p, temp=temp, seeds=seed_tensor)
    tokens1_host = to_torch_auto_compose(tokens1).flatten()[:B]

    # Second call with same seed
    logits_tt2 = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)
    tokens2, _ = sampler.decode_forward(logits_tt2, k=k, p=p, temp=temp, seeds=seed_tensor)
    tokens2_host = to_torch_auto_compose(tokens2).flatten()[:B]

    assert torch.equal(
        tokens1_host, tokens2_host
    ), f"Same seed produced different tokens:\n  call1: {tokens1_host[:8]}\n  call2: {tokens2_host[:8]}"


# ==============================================================================
# Isolation tests — ttnn.topk + ttnn.all_gather without Sampling1D
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2), (1, 8)], ids=["1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "vocab_size",
    [
        pytest.param(1024, id="v1024"),
        pytest.param(32000, id="v32000"),
    ],
)
def test_topk_allgather_isolation(ttnn_mesh_device, vocab_size):
    """
    Minimal reproducer: ttnn.topk + ttnn.all_gather on multi-device, bypassing Sampling1D.

    Runs the raw op pipeline that Sampling1D._topk_multi_device performs:
      1. Shard logits across devices along the vocab dim
      2. ttnn.topk per device (local top-K)
      3. ttnn.all_gather values and indices across devices
      4. Add index offsets for global vocab indices
      5. Pick global top-1 from gathered results

    Compare against a bfloat16-sharded torch reference. This isolates whether
    mismatches come from topk+all_gather or from downstream ops (sampling, typecast, etc.).
    """
    torch.manual_seed(42)
    B = 32
    K = 32  # max_top_k, matches Sampling1D default

    cluster_shape = tuple(ttnn_mesh_device.shape)
    num_devices = max(cluster_shape)
    per_device_vocab = vocab_size // num_devices

    # -- 1. Shard logits across devices (same as _make_logits_tt with shard_vocab=True) --
    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

    shard_dims = (None, -1) if cluster_shape[-1] >= cluster_shape[-2] else (-1, None)
    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
    )

    # -- 2. Build local_indices buffer replicated on all devices --
    replicate_mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape)
    local_indices_host = torch.zeros(1, 1, B, per_device_vocab, dtype=torch.int32)
    for i in range(per_device_vocab):
        local_indices_host[:, :, :, i] = i

    local_indices_tt = ttnn.from_torch(
        local_indices_host,
        device=ttnn_mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # -- 3. ttnn.topk per device --
    topk_values, topk_indices = ttnn.topk(
        logits_tt,
        k=K,
        dim=-1,
        indices_tensor=local_indices_tt,
    )

    # -- 4. all_gather values and indices along the vocab dim --
    sampling_cluster_axis = None if 1 in cluster_shape else 0

    gathered_values = ttnn.all_gather(
        topk_values,
        dim=3,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=sampling_cluster_axis,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(topk_values)

    gathered_indices = ttnn.all_gather(
        topk_indices,
        dim=3,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=sampling_cluster_axis,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(topk_indices)

    # -- 5. Add per-device offsets to convert local → global vocab indices --
    offsets_host = torch.zeros(1, 1, B, K * num_devices, dtype=torch.int64)
    for d in range(num_devices):
        offsets_host[:, :, :, d * K : (d + 1) * K] = d * per_device_vocab

    index_offsets_tt = ttnn.from_torch(
        offsets_host,
        device=ttnn_mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gathered_indices_int32 = ttnn.typecast(gathered_indices, dtype=ttnn.int32)
    global_indices = ttnn.add(index_offsets_tt, gathered_indices_int32, dtype=ttnn.int32)

    # -- 6. Read back to host --
    values_host = to_torch_auto_compose(gathered_values).squeeze()[:B]  # [B, K*num_devices]
    indices_host = to_torch_auto_compose(global_indices).squeeze()[:B]  # [B, K*num_devices]

    # -- 7. Pick global top-1: find max value position then look up its global index --
    top1_pos = values_host.float().argmax(dim=-1)
    device_top1 = torch.tensor([indices_host[b, top1_pos[b]].item() for b in range(B)], dtype=torch.long)

    # -- 8. Bfloat16-sharded torch reference (same method as test_sampling1d_topk1_vs_argmax) --
    logits_bf16 = logits_host.squeeze().bfloat16()  # [B, V]
    bf16_expected = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        best_val = float("-inf")
        best_idx = 0
        for s in range(num_devices):
            shard = logits_bf16[b, s * per_device_vocab : (s + 1) * per_device_vocab]
            local_idx = shard.float().argmax().item()
            local_val = shard[local_idx].float().item()
            if local_val > best_val:
                best_val = local_val
                best_idx = s * per_device_vocab + local_idx
        bf16_expected[b] = best_idx

    # -- 9. Report and assert (tie-break-aware) --
    _assert_no_true_mismatches(
        device_top1,
        bf16_expected,
        logits_bf16,
        test_label=f"topk+all_gather isolation (V={vocab_size}, mesh={cluster_shape})",
    )


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2), (1, 8)], ids=["1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "vocab_size",
    [
        pytest.param(1024, id="v1024"),
        pytest.param(32000, id="v32000"),
    ],
)
def test_ttnn_sampling_isolation(ttnn_mesh_device, vocab_size):
    """
    Hypothesis 2: does ttnn.sampling introduce mismatches at k=1, p=0.0, temp=1.0?

    Builds correct gathered_values + global_indices via the topk+all_gather pipeline
    (confirmed 0 mismatches in test_topk_allgather_isolation), then runs the remaining
    steps from Sampling1D._sample_topk verbatim:
      - ttnn.typecast (uint16 → int32)
      - ttnn.add (index offsets)
      - ttnn.untilize (TILE → ROW_MAJOR, required by ttnn.sampling)
      - ttnn.manual_seed + ttnn.sampling(k=1, p=0.0, temp=1.0)

    Compares against the bfloat16-sharded torch argmax reference.
    Any mismatches here can be attributed to ttnn.sampling itself.

    Observed results (seed=42, B=32):
      v1024-1x2:  1/32 mismatches  ← ttnn.sampling
      v1024-1x8:  1/32 mismatches  ← same batch as 1x2, topology-independent
      v32000-1x2: 4/32 mismatches  ← ttnn.sampling
      v32000-1x8: 7/32 mismatches  ← 4 shared with 1x2 + 3 additional

    The growing mismatch count with more devices (1x2→1x8) is not caused by
    all_gather (test_topk_allgather_isolation confirms 0/32 there). The extra
    mismatches at 1x8 come from ttnn.sampling seeing a wider candidate buffer
    (K*8=256 entries vs K*2=64), causing its internal softmax reduction to
    diverge from argmax on more batches.
    """
    torch.manual_seed(42)
    B = 32
    K = 32  # max_top_k

    cluster_shape = tuple(ttnn_mesh_device.shape)
    num_devices = max(cluster_shape)
    per_device_vocab = vocab_size // num_devices
    replicate_mapper = ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape)

    # ---- Step A: topk + all_gather (confirmed correct, 0 mismatches) --------

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

    shard_dims = (None, -1) if cluster_shape[-1] >= cluster_shape[-2] else (-1, None)
    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
    )

    local_indices_host = torch.zeros(1, 1, B, per_device_vocab, dtype=torch.int32)
    for i in range(per_device_vocab):
        local_indices_host[:, :, :, i] = i
    local_indices_tt = ttnn.from_torch(
        local_indices_host,
        device=ttnn_mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    topk_values, topk_indices = ttnn.topk(logits_tt, k=K, dim=-1, indices_tensor=local_indices_tt)

    sampling_cluster_axis = None if 1 in cluster_shape else 0
    gathered_values = ttnn.all_gather(
        topk_values,
        dim=3,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=sampling_cluster_axis,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(topk_values)
    gathered_indices = ttnn.all_gather(
        topk_indices,
        dim=3,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=sampling_cluster_axis,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(topk_indices)

    # ---- Step B: index offset addition (same as _sample_topk lines 233-253) -

    gathered_indices_int32 = ttnn.typecast(gathered_indices, dtype=ttnn.int32)

    offsets_host = torch.zeros(1, 1, B, K * num_devices, dtype=torch.int64)
    for d in range(num_devices):
        offsets_host[:, :, :, d * K : (d + 1) * K] = d * per_device_vocab
    index_offsets_tt = ttnn.from_torch(
        offsets_host,
        device=ttnn_mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    global_indices_tiled = ttnn.add(
        index_offsets_tt, gathered_indices_int32, dtype=ttnn.int32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(gathered_indices_int32)

    global_indices_rm = ttnn.untilize(global_indices_tiled, use_multicore=True)
    ttnn.deallocate(global_indices_tiled)

    # ---- Step C: seed + ttnn.sampling(k=1, p=0.0, temp=1.0) -----------------

    seeds_host = torch.arange(B, dtype=torch.int32)
    seeds_tt = ttnn.from_torch(
        seeds_host,
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    user_ids_tt = ttnn.from_torch(
        seeds_host,
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.manual_seed(seeds=seeds_tt, user_ids=user_ids_tt)

    k_tt = ttnn.from_torch(
        torch.ones(B, dtype=torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p_tt = ttnn.from_torch(
        torch.zeros(B, dtype=torch.float32),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    temp_tt = ttnn.from_torch(
        torch.ones(B, dtype=torch.float32),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sampled_tokens = ttnn.sampling(gathered_values, global_indices_rm, k=k_tt, p=p_tt, temp=temp_tt)

    ttnn.deallocate(gathered_values)
    ttnn.deallocate(global_indices_rm)

    # ---- Step D: compare against bfloat16-sharded torch reference -----------

    tokens_host = to_torch_auto_compose(sampled_tokens).flatten()[:B]

    # Bfloat16-sharded reference (same as other tests in this file)
    logits_bf16 = logits_host.squeeze().bfloat16()
    bf16_expected = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        best_val, best_idx = float("-inf"), 0
        for s in range(num_devices):
            shard = logits_bf16[b, s * per_device_vocab : (s + 1) * per_device_vocab]
            local_idx = shard.float().argmax().item()
            local_val = shard[local_idx].float().item()
            if local_val > best_val:
                best_val, best_idx = local_val, s * per_device_vocab + local_idx
        bf16_expected[b] = best_idx

    _assert_no_true_mismatches(
        tokens_host.long(),
        bf16_expected,
        logits_bf16,
        test_label=f"ttnn.sampling isolation (V={vocab_size}, mesh={cluster_shape})",
    )


# ==============================================================================
# Minimal reproducer: ttnn.topk correctness at various input widths
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize(
    "input_width",
    [
        pytest.param(512, id="w512"),
        pytest.param(1024, id="w1024"),
        pytest.param(2048, id="w2048"),
        pytest.param(4096, id="w4096"),
        pytest.param(8192, id="w8192"),
        pytest.param(16384, id="w16384"),
        pytest.param(32768, id="w32768"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
def test_ttnn_topk_correctness(ttnn_mesh_device, input_width, dtype):
    """
    Minimal reproducer for ttnn.topk index disagreements at various input widths and dtypes.

    Isolates ttnn.topk on a SINGLE device with NO Sampling1D, NO all_gather,
    NO ttnn.sampling — just the raw topk op. We compare:
      1. Top-1 (argmax): device top-1 index vs torch.topk top-1
      2. Top-K set: whether the device's top-32 index SET matches torch's top-32 set
      3. Top-K values: whether the returned values match the expected values

    Note: ttnn.topk only supports BFLOAT16 and BFLOAT8_B inputs (enforced by the kernel
    at topk_device_operation.cpp:146). Float32 is not a valid input dtype.

    Observed results (seed=42, B=32, K=32, single device 1x1):

    bfloat16 (~128 distinct values in the typical randn range):
        Width   Top-1 mismatches   Top-K set mismatches   Top-1 value mismatches
        -----   ----------------   --------------------   ----------------------
          512          0/32                3/32                    0/32
         1024          1/32                3/32                    0/32
         2048          1/32               14/32                    0/32
         4096          3/32                7/32                    0/32
         8192          6/32               12/32                    0/32
        16384          8/32               13/32                    0/32
        32768         17/32               14/32                    0/32

        All top-1 mismatches are TIE-BREAKS (delta=0.0000). Zero true mismatches.

    bfloat8_b (even fewer distinct values → more ties expected):
        Serves as a confirmation of the tie-breaking hypothesis: with coarser
        quantization, ties are more frequent, so index disagreements should
        increase compared to bfloat16 at the same width.

    Key findings:
      - NOT a comparison bug. Every top-1 mismatch has delta=0.0000: the device picks a
        different index that has the SAME value as the reference's top-1.
      - Values are always correct (0/32 value mismatches at every width). ttnn.topk finds
        the right maximum value; it just returns a different index among tied elements.
      - Root cause: low-precision tie-breaking non-determinism. With few distinct
        representable values, ties become very common as width grows, explaining
        why mismatches scale with width.
      - Implication: the "10/32 mismatches" in Sampling1D with vocab=32768 are NOT a
        ttnn.topk kernel bug — they are precision tie-breaks. Any element with the max
        value is a valid argmax. Tests should only assert on true mismatches (device
        picked a genuinely lower value), not on tie-breaks.
    """
    # Both supported dtypes use bfloat16 as the host torch dtype; bfloat8_b is a
    # device-side block-float format that ttnn converts from bfloat16 on the host.
    torch_dtype = torch.bfloat16
    dtype_tag = "bf16" if dtype == ttnn.bfloat16 else "bf8b"

    torch.manual_seed(42)
    B = 32
    K = 32

    # -- 1. Create input tensor [1, 1, B, W] in the target dtype --
    input_host = torch.randn(1, 1, B, input_width, dtype=torch_dtype)

    input_tt = ttnn.from_torch(
        input_host,
        device=ttnn_mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # -- 2. Create local_indices [1, 1, B, W] with 0-based range --
    #    This matches the production code: indices_tensor[..., i] = i
    local_indices_host = torch.zeros(1, 1, B, input_width, dtype=torch.int32)
    for i in range(input_width):
        local_indices_host[:, :, :, i] = i

    local_indices_tt = ttnn.from_torch(
        local_indices_host,
        device=ttnn_mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # -- 3. Run ttnn.topk --
    topk_values_tt, topk_indices_tt = ttnn.topk(input_tt, k=K, dim=-1, indices_tensor=local_indices_tt)

    # -- 4. Read back to host --
    values_host = to_torch_auto_compose(topk_values_tt).squeeze()[:B]  # [B, K]
    indices_host = to_torch_auto_compose(topk_indices_tt).squeeze()[:B]  # [B, K]

    ttnn.deallocate(topk_values_tt)
    ttnn.deallocate(topk_indices_tt)
    ttnn.deallocate(input_tt)
    ttnn.deallocate(local_indices_tt)

    # -- 5. Torch reference on the SAME input (compare in native precision) --
    input_2d = input_host.squeeze()  # [B, W]
    ref_values, ref_indices = torch.topk(input_2d.float(), k=K, dim=-1)
    ref_values = ref_values.to(torch_dtype)  # compare in the same dtype

    # -- 6a. Check top-1 (argmax) correctness --
    device_top1 = indices_host[:, 0].long()  # first column = top-1 index
    ref_top1 = ref_indices[:, 0].long()
    top1_mismatches = (device_top1 != ref_top1).sum().item()

    # -- 6b. Check top-K set correctness (order doesn't matter) --
    set_mismatches = 0
    missing_details = []
    for b in range(B):
        device_set = set(indices_host[b].long().tolist())
        ref_set = set(ref_indices[b].tolist())
        missing_from_device = ref_set - device_set
        if missing_from_device:
            set_mismatches += 1
            if len(missing_details) < 5:
                extra_in_device = device_set - ref_set
                missing_details.append(
                    f"  batch {b}: missing {len(missing_from_device)} ref indices, "
                    f"has {len(extra_in_device)} wrong indices\n"
                    f"    missing (first 5): {sorted(missing_from_device)[:5]}\n"
                    f"    extra   (first 5): {sorted(extra_in_device)[:5]}"
                )

    # -- 6c. Check top-1 value correctness (did it at least get the max value right?) --
    device_top1_vals = values_host[:, 0].float()
    ref_top1_vals = ref_values[:, 0].float()
    val_mismatches = (device_top1_vals != ref_top1_vals).sum().item()

    # -- 6d. Tie-break-aware assertion on top-1 indices --
    #    bfloat8_b uses block-float quantization (shared exponent per block of 32 elements).
    #    Elements within a block can lose up to 2 ULPs (~0.03125 at typical magnitudes)
    #    relative to the bfloat16 host view.
    quant_tolerance = 0.032 if dtype == ttnn.bfloat8_b else 0.0

    # Print additional top-K diagnostics before the shared assertion prints its report
    print(f"\n  top-K set mismatches:   {set_mismatches}/{B}")
    print(f"  top-1 value mismatches: {val_mismatches}/{B}")

    _assert_no_true_mismatches(
        device_top1,
        ref_top1,
        input_2d,
        test_label=f"ttnn.topk isolation (width={input_width}, dtype={dtype_tag}, B={B}, K={K})",
        quant_tolerance=quant_tolerance,
    )
