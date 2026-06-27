"""M3 · environment_check tests (PLAN section 7.1)."""

from pathlib import Path

import pytest

from agent.environment import EnvironmentError_, environment_check, parse_env_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "tt_smi_snapshot.json"


def test_environment_check_parses_mock():
    facts = environment_check(probe=FIXTURE.read_text)
    assert facts["arch"] == "wormhole"
    assert facts["card"] == "n300"
    assert facts["grid_x"] == 8 and facts["grid_y"] == 8
    assert facts["worker_cores"] == 64
    assert facts["dram_bw_gbps"] == 288.0


def test_parse_top_level_card_and_arch():
    facts = parse_env_snapshot('{"card": "p150", "arch": "blackhole"}')
    assert facts["arch"] == "blackhole"
    assert facts["card"] == "p150"
    assert facts["dram_bw_gbps"] == 512.0


def test_unknown_arch_raises():
    with pytest.raises(EnvironmentError_):
        parse_env_snapshot('{"arch": "grayskull"}')


def test_unparseable_snapshot_raises():
    with pytest.raises(EnvironmentError_):
        parse_env_snapshot("not json")


def test_probe_required():
    with pytest.raises(ValueError):
        environment_check()
