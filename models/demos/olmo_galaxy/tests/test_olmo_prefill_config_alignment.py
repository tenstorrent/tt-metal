from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OLMO_CONFIG = REPO_ROOT / "models/demos/olmo_galaxy/tt/olmo_model_config.py"
OLMO_ATTENTION = REPO_ROOT / "models/demos/olmo_galaxy/tt/llama_attention.py"
OLMO_CCL = REPO_ROOT / "models/demos/olmo_galaxy/tt/llama_ccl.py"
OLMO_GENERATOR = REPO_ROOT / "models/demos/olmo_galaxy/tt/generator.py"
OLMO_GENERATOR_VLLM = REPO_ROOT / "models/demos/olmo_galaxy/tt/generator_vllm.py"
OLMO_MODEL = REPO_ROOT / "models/demos/olmo_galaxy/tt/llama_model.py"
OLMO_TEXT_DEMO = REPO_ROOT / "models/demos/olmo_galaxy/demo/text_olmo_demo.py"
VLLM_TT_PLATFORM = REPO_ROOT / "vllm/vllm/platforms/tt.py"
VLLM_TT_MODEL_RUNNER = REPO_ROOT / "vllm/vllm/v1/worker/tt_model_runner.py"
VLLM_TT_WORKER = REPO_ROOT / "vllm/vllm/v1/worker/tt_worker.py"
TT_VLLM_PLUGIN_MODEL_RUNNER = (
    REPO_ROOT / "tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/v1/worker/tt_model_runner.py"
)
TT_VLLM_PLUGIN_WORKER = REPO_ROOT / "tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/v1/worker/tt_worker.py"


def _source(path):
    return path.read_text()


def _between(source, start, end):
    start_idx = source.index(start)
    end_idx = source.index(end, start_idx)
    return source[start_idx:end_idx]


def _after(source, start):
    start_idx = source.index(start)
    return source[start_idx:]


def test_olmo_ff_configs_do_not_define_column_zero_worker_grid():
    source = _source(OLMO_CONFIG)
    init_grid_section = _between(source, "self.sub_core_grids = ttnn.CoreRangeSet", "self.sub_core_grid_topk")

    assert "self.ff_sub_core_grids = self.sub_core_grids" in init_grid_section
    assert "ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))" not in init_grid_section
    assert "self.start_core = ttnn.CoreCoord(1, 0)" in init_grid_section
    assert init_grid_section.index("self.start_core = ttnn.CoreCoord(1, 0)") < init_grid_section.index(
        "self.ff_start_core = self.start_core"
    )
    assert "self.ff_start_core = self.start_core" in init_grid_section


def test_olmo_ff2_input_memcfg_uses_padded_worker_grid():
    source = _source(OLMO_CONFIG)
    ff2_section = _between(source, "OLMO_FF2_IN_CORES", 'self.model_config["FF2_OUT_RING_MEMCFG"]')

    assert "OLMO_FF2_IN_CORES = 40" in ff2_section
    assert "self.start_core, OLMO_FF2_IN_CORES, self.sub_core_grids" in ff2_section
    assert "self.ff_start_core" not in ff2_section
    assert "self.ff_sub_core_grids" not in ff2_section
    assert "self.intermediate_dim_per_tp_padded_24_cores // OLMO_FF2_IN_CORES" in ff2_section


def test_olmo_long_prefill_ff1_ff3_matches_llama_style_low_pressure_schedule():
    source = _source(OLMO_CONFIG)
    ff1_section = _between(
        source,
        "def prefill_ff1_ff3_minimal_matmul_config(seq_len):",
        'self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"]',
    )

    long_section = _after(ff1_section, "else:")
    assert "subblock_h=4" in long_section
    assert "subblock_w=2" in long_section
    assert "compute_with_storage_grid_size=ttnn.CoreCoord(7, 8)" in long_section
    assert "compute_with_storage_grid_size=ttnn.CoreCoord(7, 9)" not in long_section


def test_olmo_long_prefill_ff2_uses_llama_style_64k_schedule():
    source = _source(OLMO_CONFIG)
    ff2_section = _between(
        source,
        "def prefill_ff2_minimal_matmul_config(seq_len):",
        'self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"]',
    )

    assert "elif seq_len <= 32768:" in ff2_section
    assert "elif seq_len <= 65536:" in ff2_section
    assert "subblock_h=2" in _between(ff2_section, "elif seq_len <= 65536:", "else:")
    assert "subblock_w=4" in _between(ff2_section, "elif seq_len <= 65536:", "else:")
    assert "compute_with_storage_grid_size=ttnn.CoreCoord(7, 8)" in _between(
        ff2_section, "elif seq_len <= 65536:", "else:"
    )


def test_olmo_qk_post_norm_stays_on_row8_and_uses_post_all_gather():
    config_source = _source(OLMO_CONFIG)
    attention_source = _source(OLMO_ATTENTION)

    assert "ttnn.CoreRange(ttnn.CoreCoord(5, 8), ttnn.CoreCoord(6, 8))" in config_source
    assert "ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 0)" not in _between(
        config_source,
        "# ==== Fused QK-norm configs",
        "# ==== Prefill MLP Configs",
    )

    q_norm_idx = attention_source.index("q_stats = ttnn.rms_norm_pre_all_gather")
    q_gather_idx = attention_source.index("q_stats_gathered = self._olmo_qk_norm_all_gather", q_norm_idx)
    q_post_idx = attention_source.index("q_normed_flat = ttnn.rms_norm_post_all_gather", q_gather_idx)
    k_norm_idx = attention_source.index("k_stats = ttnn.rms_norm_pre_all_gather")
    k_gather_idx = attention_source.index("k_stats_gathered = self._olmo_qk_norm_all_gather", k_norm_idx)
    k_post_idx = attention_source.index("k_heads_1KSD_pre_rot = ttnn.rms_norm_post_all_gather", k_gather_idx)

    assert q_norm_idx < q_gather_idx < q_post_idx
    assert k_norm_idx < k_gather_idx < k_post_idx


def test_olmo_text_demo_excludes_long_8k_batch32_case_for_now():
    source = _source(OLMO_TEXT_DEMO)
    param_section = _between(source, "# Test parametrization", '@pytest.mark.parametrize(\n    "optimizations"')

    assert "# long-8k-b32" not in param_section
    assert "input_data_long_8k_b32.json" not in param_section
    assert '"long-8k-b32"' not in param_section


def test_olmo_server_prefill_warmup_does_not_auto_prime_8k_to_64k_from_max_context():
    source = _source(OLMO_GENERATOR)
    warmup_section = _between(source, "def warmup_model_prefill", "## Destructor")

    assert "8K/16K/32K/64K" in warmup_section
    assert "self.long_isl_warmup_seqlens = []" in warmup_section
    assert "self.long_isl_warmup_seqlens = previous_long_isl_warmup" in warmup_section


def test_vllm_tt_platform_registers_olmo_to_olmo_galaxy_fork():
    source = _source(VLLM_TT_PLATFORM)
    olmo_section = _between(source, "# OLMo3 - Text", "# Optionally register test models")

    assert "models.demos.olmo_galaxy.tt.generator_vllm:OLMo3ForCausalLM" in olmo_section
    assert "models.demos.llama3_70b_galaxy.tt.generator_vllm:OLMo3ForCausalLM" not in olmo_section


def test_olmo_vllm_wrapper_keeps_llama_style_hot_path_without_sync_resets():
    source = _source(OLMO_GENERATOR_VLLM)
    olmo_section = _between(source, "class OLMo3ForCausalLM", "def allocate_kv_cache")

    assert "def prefill_forward" in olmo_section
    assert 'kwargs.pop("sampling_params", None)' in olmo_section
    assert "ttnn.synchronize_device" not in olmo_section
    assert "reset_gather_and_buffer_idx" not in olmo_section
    assert "def read_decode_output" not in olmo_section


def test_olmo_sequential_prefill_drains_after_output_processing():
    source = _source(OLMO_GENERATOR)
    prefill_loop_section = _between(
        source,
        "if not do_device_sampling:",
        "prefill_log_probs = None",
    )
    process_output_idx = prefill_loop_section.index("self.model.process_output_prefill(")
    after_process_output = prefill_loop_section[process_output_idx:]

    assert "ttnn.synchronize_device(self.mesh_device)" in after_process_output
    assert "self.model.tt_ccl.reset_gather_semaphores()" in after_process_output
    assert "self.model.tt_ccl.reset_gather_and_buffer_idx()" not in after_process_output


def test_olmo_mode_switches_use_targeted_gather_resets():
    source = _source(OLMO_MODEL)
    ccl_source = _source(OLMO_CCL)
    setup_prefill_section = _between(source, "def setup_prefill", "def setup_decode")
    setup_decode_section = _between(source, "def setup_decode", "def prepare_prefill_inputs_host")

    assert "def reset_gather_semaphores(" in ccl_source
    assert "reset_gather_semaphores()" in setup_prefill_section
    assert "reset_gather_semaphores()" in setup_decode_section


def test_tt_vllm_decode_uses_async_read_for_device_sampling():
    source = _source(VLLM_TT_MODEL_RUNNER)
    decode_section = _between(
        source,
        "# TODO: Add encoder-decoder support",
        "# tt_out can be a tuple of",
    )

    assert "async_read=perform_device_sampling" in decode_section
    assert "tt_out = self._finalize_async_decode_output(" in source
    assert "def _finalize_async_decode_output(" in source


def test_tt_vllm_plugin_decode_uses_async_read_for_device_sampling():
    source = _source(TT_VLLM_PLUGIN_MODEL_RUNNER)
    pure_decode_section = _between(
        source,
        "# Pure batch: use existing logic",
        "# tt_out is a tuple of",
    )
    mixed_decode_section = _between(
        source,
        "# TODO: Add encoder-decoder support",
        "# Combine outputs in original batch order",
    )

    assert "async_read=perform_device_sampling" in pure_decode_section
    assert "tt_out = self._finalize_async_decode_output(" in source
    assert 'decode_kwargs["async_read"] = decode_samples_on_device' in mixed_decode_section
    assert "decode_output = self._finalize_async_decode_output(" in source
    assert "def _finalize_async_decode_output(" in source


def test_tt_workers_preserve_explicit_num_gpu_blocks_override():
    for worker_path in (VLLM_TT_WORKER, TT_VLLM_PLUGIN_WORKER):
        source = _source(worker_path)
        determine_available_memory_section = _between(
            source,
            "def determine_available_memory",
            "def initialize_from_config",
        )

        assert "if self.cache_config.num_gpu_blocks_override is None:" in determine_available_memory_section
        assert "self.cache_config.num_gpu_blocks_override = num_tt_blocks" in determine_available_memory_section
