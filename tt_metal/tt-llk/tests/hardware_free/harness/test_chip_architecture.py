# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test: a session whose architecture is not the hardware's does not start."""

import pytest
from helpers import llk_pytest_plugin as plugin
from helpers.chip_architecture import (
    CHIP_ARCH_SOURCE_CARD,
    CHIP_ARCH_SOURCE_ENV,
    ChipArchitecture,
)
from helpers.test_config import TestConfig


@pytest.fixture
def targeting_blackhole(monkeypatch):
    """A session on Blackhole; the test says what the card, the descriptor and the source are."""

    def configure(*, card=None, descriptor=None, run_simulator=False, source):
        monkeypatch.setattr(
            TestConfig, "CHIP_ARCH", ChipArchitecture.BLACKHOLE, raising=False
        )
        monkeypatch.setattr(TestConfig.TEST_TARGET, "run_simulator", run_simulator)
        monkeypatch.setattr(plugin, "pcie_architecture", lambda: card)
        monkeypatch.setattr(plugin, "simulator_architecture", lambda: descriptor)
        monkeypatch.setattr(plugin, "chip_architecture_source", lambda: source)

    return configure


def test_card_agrees(targeting_blackhole):
    targeting_blackhole(card=ChipArchitecture.BLACKHOLE, source=CHIP_ARCH_SOURCE_ENV)
    plugin._verify_chip_architecture()


def test_no_hardware_to_disagree(targeting_blackhole):
    """CHIP_ARCH is the only answer on a cardless host: the hardware-free suites run this way."""
    targeting_blackhole(card=None, source=CHIP_ARCH_SOURCE_ENV)
    plugin._verify_chip_architecture()


def test_chip_arch_against_the_card(targeting_blackhole):
    targeting_blackhole(card=ChipArchitecture.WORMHOLE, source=CHIP_ARCH_SOURCE_ENV)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        pytest.UsageError
    ) as error:
        plugin._verify_chip_architecture()
    assert "the card this harness runs on is wormhole" in str(error.value)
    assert "Unset CHIP_ARCH, or set it to wormhole." in str(error.value)


def test_chip_arch_against_the_descriptor(targeting_blackhole):
    """CHIP_ARCH chose the arch, so the remedy is CHIP_ARCH, not the flag already passed."""
    targeting_blackhole(
        descriptor=ChipArchitecture.QUASAR,
        run_simulator=True,
        source=CHIP_ARCH_SOURCE_ENV,
    )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        pytest.UsageError
    ) as error:
        plugin._verify_chip_architecture()
    assert "Unset CHIP_ARCH, or set it to quasar." in str(error.value)
    assert "--run-simulator" not in str(error.value)


def test_run_simulator_seen_too_late(targeting_blackhole):
    """--run-simulator from the ini file's addopts is invisible at import, so the card won."""
    targeting_blackhole(
        card=ChipArchitecture.BLACKHOLE,
        descriptor=ChipArchitecture.QUASAR,
        run_simulator=True,
        source=CHIP_ARCH_SOURCE_CARD,
    )
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        pytest.UsageError
    ) as error:
        plugin._verify_chip_architecture()
    assert "Set CHIP_ARCH=quasar, or pass --run-simulator" in str(error.value)
