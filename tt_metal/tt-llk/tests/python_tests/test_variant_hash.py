# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.test_config import TestConfig
from helpers.test_variant_parameters import INPUT_DIMENSIONS


def _variant_id(*, templates=None, runtimes=None):
    config = TestConfig(
        test_name="sources/quasar/unpack_tilize_quasar_test.cpp",
        templates=templates or [],
        runtimes=runtimes or [],
    )
    config.generate_variant_hash()
    return config.variant_id


def test_runtime_dimensions_do_not_change_variant_hash(monkeypatch):
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False)

    first = _variant_id(runtimes=[INPUT_DIMENSIONS(1, 2, 2, 1)])
    second = _variant_id(runtimes=[INPUT_DIMENSIONS(4, 8, 4, 2)])

    assert first == second


def test_template_dimensions_change_variant_hash(monkeypatch):
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False)

    first = _variant_id(templates=[INPUT_DIMENSIONS(1, 2, 2, 1)])
    second = _variant_id(templates=[INPUT_DIMENSIONS(4, 8, 4, 2)])

    assert first != second
