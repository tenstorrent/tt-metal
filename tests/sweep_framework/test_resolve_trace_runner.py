# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest

from model_tracer.resolve_trace_runner import arch_for, resolve_sku, weights_mount

SKUS = {
    "wh_n150": {"runs_on": ["N150", "in-service"]},
    "wh_n300": {"runs_on": ["N300", "in-service"]},
    "bh_p150": {"runs_on": ["arch-blackhole", "P150"], "weights-cache-mode": "cloud-mlperf"},
    "bh_loudbox": {"runs_on": ["arch-blackhole", "BH-LoudBox"], "weights-cache-mode": "cloud-mlperf"},
    "cpu_medium": {"runs_on": ["tt-ubuntu-2204-medium-stable"]},
    "bh_sc4": {"runs_on": ["exabox-multihost-ci-sc4-8kw"]},
    "bh_legacy_alloc": {"runs_on": ["arch-blackhole"], "allocation": {"hosts": 2}},
}


@pytest.fixture
def expect_error():
    # all-static-checks.yaml runs this file with --noconftest, so the root conftest's
    # fixture isn't available.
    @contextlib.contextmanager
    def expect_error_(error, message):
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: host job has no root conftest
            yield exc_info

    return expect_error_


def test_alias_resolves_to_sku():
    assert resolve_sku("BH-LoudBox", SKUS)[0] == "bh_loudbox"


@pytest.mark.parametrize("label", ["N150", "N300", "P100a", "P150b", "n300-llmbox"])
def test_retired_label_names_both_pools(label, expect_error):
    with expect_error(ValueError, r"no longer accepted.*CIv1.*CIv2"):
        resolve_sku(label, SKUS)


def test_sku_name_accepted_directly():
    assert resolve_sku("bh_p150", SKUS)[0] == "bh_p150"


def test_unknown_label_rejected(expect_error):
    with expect_error(ValueError, "unknown runner-label"):
        resolve_sku("N999", SKUS)


@pytest.mark.parametrize("sku", ["bh_sc4", "bh_legacy_alloc"])
def test_multihost_sku_rejected(sku, expect_error):
    with expect_error(ValueError, "multihost pool"):
        resolve_sku(sku, SKUS)


def test_arch_from_prefix_and_rejects_cpu(expect_error):
    assert arch_for("wh_n150", []) == "wormhole_b0"
    assert arch_for("bh_p150", []) == "blackhole"
    with expect_error(ValueError, "cannot tell the arch"):
        arch_for("cpu_medium", SKUS["cpu_medium"]["runs_on"])


@pytest.mark.parametrize(
    "mode, offline",
    [("", "1"), ("cloud-mlperf", "1"), ("yyz4-mnt-models", "1"), ("local-disk", "0"), ("lfc", "0")],
)
def test_weights_mount_lands_at_mlperf_huggingface(mode, offline):
    mount, off = weights_mount(mode)
    assert off == offline
    assert mount.split(":")[1] in ("/mnt/MLPerf", "/mnt/MLPerf/huggingface")


def test_unknown_mode_rejected(expect_error):
    with expect_error(ValueError, "unknown weights-cache-mode"):
        weights_mount("s3-direct")
