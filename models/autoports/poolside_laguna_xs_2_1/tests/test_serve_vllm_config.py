# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free launcher policy checks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "serve_vllm.sh"
SETUP = Path(__file__).resolve().parents[1] / "setup_vllm.sh"


def _config(tmp_path, **overrides):
    # A minimal environment makes the test independent of inherited bring-up/debug knobs.
    env = {
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
        "LAGUNA_PROFILE": "p150x2",
        "TT_VISIBLE_DEVICES": "0,1",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), "config"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_p150x2_defaults_to_qualified_prefix_caching(tmp_path):
    result = _config(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "hybrid_kv=0\n" in result.stdout
    assert "hybrid_kv_status=production_safe_disabled\n" in result.stdout
    assert "hybrid_kv_layout=uniform_forty_tensor_pairs\n" in result.stdout
    assert "prefix_cache=1\n" in result.stdout
    assert "prefix_cache_profile_default=1\n" in result.stdout
    assert "prefix_cache_profile_policy=qualified\n" in result.stdout
    assert "prefix_cache_env_explicit=0\n" in result.stdout
    assert "prefix_cache_status=production_qualified\n" in result.stdout
    assert "prefix_cache_cli_arg=--enable-prefix-caching\n" in result.stdout
    assert "context_status=profile_qualified_limit\n" in result.stdout
    assert "multi_seq_status=profile_qualified_sequence_limit\n" in result.stdout
    assert "prompt_tokens_details_cli_arg=--enable-prompt-tokens-details\n" in result.stdout
    assert ("prefix_cache_cli_args=--enable-prefix-caching " "--enable-prompt-tokens-details\n") in result.stdout
    assert "prefix_cache_quantum=8192\n" in result.stdout
    assert "prefix_cache_block_size=64\n" in result.stdout
    assert "prefix_cache_admission_policy=complete_canonical_prompt_chunks\n" in result.stdout
    assert "prefix_cache_kv_group_policy=single_uniform_full_attention\n" in result.stdout
    assert "prefix_cache_scheduler_policy=max_num_seqs_1_no_chunked_prefill\n" in result.stdout
    assert "prefix_cache_spec_decode_policy=disabled\n" in result.stdout
    assert "prefix_cache_external_kv_policy=disabled\n" in result.stdout
    assert "chunked_prefill_cli_arg=--no-enable-chunked-prefill\n" in result.stdout
    assert "streaming_prefill=1\n" in result.stdout
    assert "streaming_prefill_status=production_qualified\n" in result.stdout
    assert "moe_token_dispatch=0\n" in result.stdout
    assert "moe_token_dispatch_status=production_safe_disabled\n" in result.stdout
    assert "moe_prefill_tile_sparse=0\n" in result.stdout
    assert "moe_prefill_tile_sparse_status=production_safe_disabled\n" in result.stdout
    assert "experimental_overrides=<none>\n" in result.stdout
    assert "min_dram_free_fraction=0.10\n" in result.stdout
    assert "min_contiguous_mib=128\n" in result.stdout


def test_explicit_one_uses_qualified_policy_without_experimental_acknowledgement(tmp_path):
    result = _config(tmp_path, TT_LAGUNA_PREFIX_CACHE="1")

    assert result.returncode == 0, result.stderr
    assert "prefix_cache=1\n" in result.stdout
    assert "prefix_cache_profile_default=1\n" in result.stdout
    assert "prefix_cache_profile_policy=qualified\n" in result.stdout
    assert "prefix_cache_env_explicit=1\n" in result.stdout
    assert "prefix_cache_status=production_qualified\n" in result.stdout
    assert "prefix_cache_cli_arg=--enable-prefix-caching\n" in result.stdout
    assert "prompt_tokens_details_cli_arg=--enable-prompt-tokens-details\n" in result.stdout
    assert ("prefix_cache_cli_args=--enable-prefix-caching " "--enable-prompt-tokens-details\n") in result.stdout
    assert "chunked_prefill_cli_arg=--no-enable-chunked-prefill\n" in result.stdout
    assert "prefix_cache_quantum=8192\n" in result.stdout
    assert "experimental_overrides=<none>\n" in result.stdout


def test_explicit_zero_is_a_fail_closed_non_experimental_override(tmp_path):
    result = _config(tmp_path, TT_LAGUNA_PREFIX_CACHE="0")

    assert result.returncode == 0, result.stderr
    assert "prefix_cache=0\n" in result.stdout
    assert "prefix_cache_profile_default=1\n" in result.stdout
    assert "prefix_cache_profile_policy=qualified\n" in result.stdout
    assert "prefix_cache_env_explicit=1\n" in result.stdout
    assert "prefix_cache_status=operator_rollback_disabled\n" in result.stdout
    assert "prefix_cache_cli_arg=--no-enable-prefix-caching\n" in result.stdout
    assert "prompt_tokens_details_cli_arg=<none>\n" in result.stdout
    assert "chunked_prefill_cli_arg=<vllm-default>\n" in result.stdout
    assert "experimental_overrides=<none>\n" in result.stdout


def test_streaming_prefill_rollback_is_strict_and_experimental(tmp_path):
    invalid = _config(tmp_path, TT_LAGUNA_STREAMING_PREFILL="false")
    assert invalid.returncode == 2
    assert "TT_LAGUNA_STREAMING_PREFILL" in invalid.stderr
    assert "must be 0 or 1" in invalid.stderr

    rejected = _config(tmp_path, TT_LAGUNA_STREAMING_PREFILL="0")
    assert rejected.returncode == 2
    assert "TT_LAGUNA_STREAMING_PREFILL=0 (qualified=1)" in rejected.stderr

    accepted = _config(
        tmp_path,
        TT_LAGUNA_STREAMING_PREFILL="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "streaming_prefill=0\n" in accepted.stdout
    assert "streaming_prefill_status=experimental_monolithic_rollback\n" in accepted.stdout
    assert "TT_LAGUNA_STREAMING_PREFILL=0 (qualified=1)" in accepted.stdout


@pytest.mark.parametrize(
    ("flag", "value", "config_key", "status_key"),
    (
        (
            "TT_LAGUNA_MOE_TOKEN_DISPATCH",
            "1",
            "moe_token_dispatch",
            "moe_token_dispatch_status",
        ),
        (
            "TT_LAGUNA_MOE_PREFILL_TILE_SPARSE",
            "1",
            "moe_prefill_tile_sparse",
            "moe_prefill_tile_sparse_status",
        ),
    ),
)
def test_moe_prefill_optimizations_are_strict_default_off_experiments(tmp_path, flag, value, config_key, status_key):
    invalid = _config(tmp_path, **{flag: "yes"})
    assert invalid.returncode == 2
    assert "must be 0 or 1" in invalid.stderr

    rejected = _config(tmp_path, **{flag: value})
    assert rejected.returncode == 2
    assert f"{flag}=1 (qualified=0)" in rejected.stderr

    accepted = _config(
        tmp_path,
        **{
            flag: value,
            "TT_LAGUNA_PREFIX_CACHE": "0",
            "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1",
        },
    )
    assert accepted.returncode == 0, accepted.stderr
    assert f"{config_key}=1\n" in accepted.stdout
    assert f"{status_key}=experimental_unqualified\n" in accepted.stdout
    assert "prefix_cache=0\n" in accepted.stdout
    assert f"{flag}=1 (qualified=0)" in accepted.stdout


@pytest.mark.parametrize(
    ("flag", "overrides", "message"),
    (
        ("TT_LAGUNA_MOE_TOKEN_DISPATCH", {}, "requires TT_LAGUNA_PREFIX_CACHE=0"),
        ("TT_LAGUNA_MOE_PREFILL_TILE_SPARSE", {}, "requires TT_LAGUNA_PREFIX_CACHE=0"),
        (
            "TT_LAGUNA_MOE_TOKEN_DISPATCH",
            {"TT_LAGUNA_PREFIX_CACHE": "0", "TT_LAGUNA_HYBRID_KV": "1"},
            "requires TT_LAGUNA_HYBRID_KV=0 and TT_LAGUNA_DFLASH=0",
        ),
        (
            "TT_LAGUNA_MOE_PREFILL_TILE_SPARSE",
            {"TT_LAGUNA_PREFIX_CACHE": "0", "TT_LAGUNA_DFLASH": "1"},
            "requires TT_LAGUNA_HYBRID_KV=0 and TT_LAGUNA_DFLASH=0",
        ),
        (
            "TT_LAGUNA_MOE_TOKEN_DISPATCH",
            {"TT_LAGUNA_PREFIX_CACHE": "0", "TT_LAGUNA_MOE_PREFILL_TILE_SPARSE": "1"},
            "separate, unstacked qualification paths",
        ),
    ),
)
def test_moe_prefill_optimizations_reject_unqualified_feature_stacking(tmp_path, flag, overrides, message):
    result = _config(
        tmp_path,
        **{
            flag: "1",
            "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1",
            **overrides,
        },
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_hybrid_kv_is_explicit_cache_off_experimental_only(tmp_path):
    rejected = _config(
        tmp_path,
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
    )
    assert rejected.returncode == 2
    assert "unqualified inherited/debug override" in rejected.stderr
    assert "TT_LAGUNA_HYBRID_KV=1 (qualified=0)" in rejected.stderr

    accepted = _config(
        tmp_path,
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "hybrid_kv=1\n" in accepted.stdout
    assert "hybrid_kv_status=experimental_cache_off_qualification\n" in accepted.stdout
    assert "hybrid_kv_layout=four_groups_ten_aliased_tensor_pairs\n" in accepted.stdout
    assert "hybrid_kv_scheduler_chunk=8192\n" in accepted.stdout
    assert "chunked_prefill_cli_arg=--enable-chunked-prefill\n" in accepted.stdout
    assert ("chunked_prefill_cli_args=--enable-chunked-prefill " "--max-num-batched-tokens 8192\n") in accepted.stdout
    assert "TT_LAGUNA_HYBRID_KV=1 (qualified=0)" in accepted.stdout


def test_dflash_is_explicit_p150x2_batch_one_cache_off_experimental_only(tmp_path):
    rejected = _config(
        tmp_path,
        TT_LAGUNA_DFLASH="1",
        TT_LAGUNA_PREFIX_CACHE="0",
    )
    assert rejected.returncode == 2
    assert "unqualified inherited/debug override" in rejected.stderr
    assert "TT_LAGUNA_DFLASH=1 (qualified=0)" in rejected.stderr

    accepted = _config(
        tmp_path,
        TT_LAGUNA_DFLASH="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "dflash=1\n" in accepted.stdout
    assert "dflash_status=experimental_cache_off_serving\n" in accepted.stdout
    assert "dflash_envelope=p150x2_batch1_greedy_uniform_cache_off\n" in accepted.stdout
    assert "prefix_cache=0\n" in accepted.stdout
    assert "hybrid_kv=0\n" in accepted.stdout
    assert "max_num_seqs=1\n" in accepted.stdout
    assert "chunked_prefill_cli_arg=--no-enable-chunked-prefill\n" in accepted.stdout
    assert "chunked_prefill_cli_args=--no-enable-chunked-prefill\n" in accepted.stdout
    assert "TT_LAGUNA_DFLASH=1 (qualified=0)" in accepted.stdout


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "requires TT_LAGUNA_PREFIX_CACHE=0"),
        (
            {"TT_LAGUNA_PREFIX_CACHE": "0", "TT_LAGUNA_HYBRID_KV": "1"},
            "requires TT_LAGUNA_HYBRID_KV=0",
        ),
        (
            {"TT_LAGUNA_PREFIX_CACHE": "0", "TT_LAGUNA_SPEC_DECODE": "1"},
            "does not support TT_LAGUNA_SPEC_DECODE",
        ),
        (
            {
                "LAGUNA_PROFILE": "p150",
                "TT_VISIBLE_DEVICES": "0",
                "TT_LAGUNA_PREFIX_CACHE": "0",
            },
            "restricted to LAGUNA_PROFILE=p150x2",
        ),
    ],
)
def test_dflash_launcher_rejects_unqualified_feature_overlap(tmp_path, overrides, message):
    result = _config(
        tmp_path,
        TT_LAGUNA_DFLASH="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
        **overrides,
    )
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("plugin_filter", "missing"),
    (
        ("", "tt"),
        ("tt_model_registry,laguna_tt_ext", "tt"),
        ("tt,laguna_tt_ext", "tt_model_registry"),
        ("tt,tt_model_registry", "laguna_tt_ext"),
        ("tt,tt_model_registry,laguna_tt_ext_extra", "laguna_tt_ext"),
    ),
)
def test_hybrid_kv_rejects_incomplete_plugin_filter(tmp_path, plugin_filter, missing):
    result = _config(
        tmp_path,
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
        VLLM_PLUGINS=plugin_filter,
    )

    assert result.returncode == 2
    assert f"include the exact entry '{missing}'" in result.stderr


def test_hybrid_kv_accepts_plugin_filter_with_exact_local_entry(tmp_path):
    result = _config(
        tmp_path,
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
        VLLM_PLUGINS="tt,tt_model_registry,laguna_tt_ext",
    )

    assert result.returncode == 0, result.stderr
    assert "hybrid_kv=1\n" in result.stdout


@pytest.mark.parametrize(
    ("plugin_filter", "missing"),
    (
        ("tt_model_registry,laguna_tt_ext", "tt"),
        ("tt,laguna_tt_ext", "tt_model_registry"),
        ("tt,tt_model_registry", "laguna_tt_ext"),
    ),
)
def test_qualified_prefix_cache_rejects_incomplete_plugin_filter(tmp_path, plugin_filter, missing):
    result = _config(tmp_path, VLLM_PLUGINS=plugin_filter)

    assert result.returncode == 2
    assert f"include the exact entry '{missing}'" in result.stderr


def test_cache_off_launch_requires_complete_plugin_allowlist(tmp_path):
    rejected = _config(
        tmp_path,
        TT_LAGUNA_PREFIX_CACHE="0",
        VLLM_PLUGINS="tt,tt_model_registry",
    )
    assert rejected.returncode == 2
    assert "include the exact entry 'laguna_tt_ext'" in rejected.stderr

    accepted = _config(
        tmp_path,
        TT_LAGUNA_PREFIX_CACHE="0",
        VLLM_PLUGINS="tt,tt_model_registry,laguna_tt_ext",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "prefix_cache=0\n" in accepted.stdout


def test_setup_rejects_missing_local_extension_before_build(tmp_path):
    result = subprocess.run(
        ["bash", str(SETUP)],
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "VLLM_PLUGINS": "tt,tt_model_registry",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "include the exact entry 'laguna_tt_ext'" in result.stderr
    assert "First run builds tt-metal" not in result.stdout


def test_setup_rejects_missing_tt_runtime_plugin_before_build(tmp_path):
    result = subprocess.run(
        ["bash", str(SETUP)],
        env={
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "VLLM_PLUGINS": "tt_model_registry,laguna_tt_ext",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "include the exact entry 'tt'" in result.stderr
    assert "First run builds tt-metal" not in result.stdout


def test_hybrid_kv_rejects_prefix_overlap_or_non_d2_topology(tmp_path):
    overlap = _config(
        tmp_path,
        TT_LAGUNA_HYBRID_KV="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert overlap.returncode == 2
    assert "requires TT_LAGUNA_PREFIX_CACHE=0" in overlap.stderr

    non_d2 = _config(
        tmp_path,
        LAGUNA_PROFILE="p150",
        TT_VISIBLE_DEVICES="0",
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert non_d2.returncode == 2
    assert "restricted to LAGUNA_PROFILE=p150x2" in non_d2.stderr


def test_262k_context_probe_is_exact_explicit_hybrid_and_cache_off(tmp_path):
    unflagged = _config(tmp_path, LAGUNA_MAX_MODEL_LEN="262144")
    assert unflagged.returncode == 2
    assert "exceeds the verified p150x2 limit 131072" in unflagged.stderr

    wrong_value = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="200000",
        TT_LAGUNA_CONTEXT_PROBE="1",
    )
    assert wrong_value.returncode == 2
    assert "only fail-closed exception" in wrong_value.stderr

    unsafe_cache_mode = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="262144",
        TT_LAGUNA_CONTEXT_PROBE="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert unsafe_cache_mode.returncode == 2
    assert "requires cache-off hybrid KV" in unsafe_cache_mode.stderr

    accepted = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="262144",
        TT_LAGUNA_CONTEXT_PROBE="1",
        TT_LAGUNA_HYBRID_KV="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "max_model_len=262144\n" in accepted.stdout
    assert "max_num_seqs=1\n" in accepted.stdout
    assert "context_status=experimental_262144_probe\n" in accepted.stdout
    assert "multi_seq_status=profile_qualified_sequence_limit\n" in accepted.stdout
    assert "hybrid_kv=1\n" in accepted.stdout
    assert "prefix_cache=0\n" in accepted.stdout
    assert "TT_LAGUNA_ADVERTISED_CONTEXT=262144" in accepted.stdout
    assert "TT_LAGUNA_CONTEXT_PROBE=1 (qualified=0)" in accepted.stdout


def test_two_sequence_pool_probe_is_exact_uniform_cache_off_and_bounded(tmp_path):
    unflagged = _config(tmp_path, LAGUNA_MAX_NUM_SEQS="2")
    assert unflagged.returncode == 2
    assert "exceeds the p150x2 limit 1" in unflagged.stderr

    too_long = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="131072",
        LAGUNA_MAX_NUM_SEQS="2",
        TT_LAGUNA_MULTI_SEQ_POOL="1",
    )
    assert too_long.returncode == 2
    assert "max_model_len<=65536" in too_long.stderr

    unsafe_prefix = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="65536",
        LAGUNA_MAX_NUM_SEQS="2",
        TT_LAGUNA_MULTI_SEQ_POOL="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert unsafe_prefix.returncode == 2
    assert "uniform-KV cache-off path" in unsafe_prefix.stderr

    accepted = _config(
        tmp_path,
        LAGUNA_MAX_MODEL_LEN="65536",
        LAGUNA_MAX_NUM_SEQS="2",
        TT_LAGUNA_MULTI_SEQ_POOL="1",
        TT_LAGUNA_PREFIX_CACHE="0",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "max_model_len=65536\n" in accepted.stdout
    assert "max_num_seqs=2\n" in accepted.stdout
    assert "context_status=profile_qualified_limit\n" in accepted.stdout
    assert "multi_seq_status=experimental_two_sequence_pool\n" in accepted.stdout
    assert "hybrid_kv=0\n" in accepted.stdout
    assert "prefix_cache=0\n" in accepted.stdout
    assert "TT_LAGUNA_MULTI_SEQ_POOL=1 (qualified=0)" in accepted.stdout


@pytest.mark.parametrize(
    ("profile", "devices"),
    (("p150", "0"), ("p150x4", "0,1,2,3")),
)
def test_non_candidate_profiles_keep_clean_cache_off_defaults(tmp_path, profile, devices):
    result = _config(
        tmp_path,
        LAGUNA_PROFILE=profile,
        TT_VISIBLE_DEVICES=devices,
    )

    assert result.returncode == 0, result.stderr
    assert "prefix_cache=0\n" in result.stdout
    assert "prefix_cache_profile_default=0\n" in result.stdout
    assert "prefix_cache_profile_policy=experimental_only\n" in result.stdout
    assert "prefix_cache_env_explicit=0\n" in result.stdout
    assert "prefix_cache_status=production_safe_disabled\n" in result.stdout
    assert "prefix_cache_cli_arg=--no-enable-prefix-caching\n" in result.stdout
    assert "experimental_overrides=<none>\n" in result.stdout


@pytest.mark.parametrize(
    ("profile", "devices"),
    (("p150", "0"), ("p150x4", "0,1,2,3")),
)
def test_non_candidate_profiles_remain_experimental_only(tmp_path, profile, devices):
    overrides = {
        "LAGUNA_PROFILE": profile,
        "TT_VISIBLE_DEVICES": devices,
        "TT_LAGUNA_PREFIX_CACHE": "1",
        "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1",
    }
    if profile == "p150x4":
        overrides["LAGUNA_MAX_NUM_SEQS"] = "1"
    accepted = _config(
        tmp_path,
        **overrides,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert "prefix_cache_profile_policy=experimental_only\n" in accepted.stdout
    assert "prefix_cache_status=experimental_unqualified\n" in accepted.stdout
    assert "prompt_tokens_details_cli_arg=--enable-prompt-tokens-details\n" in accepted.stdout


def test_prefix_cache_rejects_multi_sequence_scheduler(tmp_path):
    rejected = _config(
        tmp_path,
        LAGUNA_PROFILE="p150x4",
        TT_VISIBLE_DEVICES="0,1,2,3",
        TT_LAGUNA_PREFIX_CACHE="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
    )

    assert rejected.returncode == 2
    assert "requires LAGUNA_MAX_NUM_SEQS=1" in rejected.stderr


@pytest.mark.parametrize(
    ("override", "value", "message"),
    (
        ("TT_LAGUNA_PREFILL_FAST", "0", "requires TT_LAGUNA_PREFILL_FAST=1"),
        (
            "TT_LAGUNA_PREFILL_FAST_CHUNK",
            "4096",
            "requires TT_LAGUNA_PREFILL_FAST_CHUNK=8192",
        ),
        (
            "TT_LAGUNA_PREFILL_SDPA_CHUNK",
            "4096",
            "requires TT_LAGUNA_PREFILL_SDPA_CHUNK=8192",
        ),
        (
            "TT_LAGUNA_SPEC_DECODE",
            "1",
            "does not support TT_LAGUNA_SPEC_DECODE",
        ),
    ),
)
def test_prefix_cache_rejects_incompatible_prefill_or_spec_policy(tmp_path, override, value, message):
    rejected = _config(
        tmp_path,
        TT_LAGUNA_PREFIX_CACHE="1",
        LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES="1",
        **{override: value},
    )

    assert rejected.returncode == 2
    assert message in rejected.stderr


def test_weaker_memory_margins_require_experimental_override_acknowledgement(tmp_path):
    cases = (
        ("TT_LAGUNA_MIN_DRAM_FREE_FRACTION", "0", "0.10", "min_dram_free_fraction=0\n"),
        ("TT_LAGUNA_MIN_CONTIGUOUS_MIB", "1", "128", "min_contiguous_mib=1\n"),
    )
    for name, value, qualified, config_line in cases:
        rejected = _config(tmp_path, **{name: value})

        assert rejected.returncode == 2
        assert "unqualified inherited/debug override" in rejected.stderr
        assert f"{name}={value} (qualified={qualified})" in rejected.stderr

        accepted = _config(
            tmp_path,
            **{
                name: value,
                "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1",
            },
        )
        assert accepted.returncode == 0, accepted.stderr
        assert config_line in accepted.stdout
        assert f"{name}={value} (qualified={qualified})" in accepted.stdout
