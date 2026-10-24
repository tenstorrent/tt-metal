from agent.probes import board_to_arch


def test_board_to_arch_from_hardware_catalog():
    assert board_to_arch("n300 L") == "wormhole"
    assert board_to_arch("N150") == "wormhole"
    assert board_to_arch("p150b") == "blackhole"
    assert board_to_arch("p300a") == "blackhole"
    assert board_to_arch("p100") == "blackhole"
    assert board_to_arch("tt-galaxy-bh") == "blackhole"
    assert board_to_arch("tt-galaxy-wh") == "wormhole"
    assert board_to_arch("ubb_blackhole") == "blackhole"
    assert board_to_arch("ubb_wormhole") == "wormhole"
    assert board_to_arch("unknown-board") is None
    assert board_to_arch("") is None
