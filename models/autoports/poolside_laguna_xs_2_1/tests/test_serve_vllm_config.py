# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free launcher policy checks."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "serve_vllm.sh"


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
    assert "prefix_cache=1\n" in result.stdout
    assert "prefix_cache_profile_default=1\n" in result.stdout
    assert "prefix_cache_profile_policy=qualified\n" in result.stdout
    assert "prefix_cache_env_explicit=0\n" in result.stdout
    assert "prefix_cache_status=production_qualified\n" in result.stdout
    assert "prefix_cache_cli_arg=--enable-prefix-caching\n" in result.stdout
    assert "prompt_tokens_details_cli_arg=--enable-prompt-tokens-details\n" in result.stdout
    assert (
        "prefix_cache_cli_args=--enable-prefix-caching "
        "--enable-prompt-tokens-details\n"
    ) in result.stdout
    assert "prefix_cache_quantum=8192\n" in result.stdout
    assert "prefix_cache_block_size=64\n" in result.stdout
    assert (
        "prefix_cache_admission_policy=complete_canonical_prompt_chunks\n"
        in result.stdout
    )
    assert "prefix_cache_kv_group_policy=single_uniform_full_attention\n" in result.stdout
    assert (
        "prefix_cache_scheduler_policy=max_num_seqs_1_no_chunked_prefill\n"
        in result.stdout
    )
    assert "prefix_cache_spec_decode_policy=disabled\n" in result.stdout
    assert "prefix_cache_external_kv_policy=disabled\n" in result.stdout
    assert "chunked_prefill_cli_arg=--no-enable-chunked-prefill\n" in result.stdout
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
    assert (
        "prefix_cache_cli_args=--enable-prefix-caching "
        "--enable-prompt-tokens-details\n"
    ) in result.stdout
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


@pytest.mark.parametrize(
    ("profile", "devices"),
    (("p150", "0"), ("p150x4", "0,1,2,3")),
)
def test_non_candidate_profiles_keep_clean_cache_off_defaults(
    tmp_path, profile, devices
):
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
def test_prefix_cache_rejects_incompatible_prefill_or_spec_policy(
    tmp_path, override, value, message
):
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
