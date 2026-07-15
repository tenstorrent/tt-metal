import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_reset_arg_sets_is_board_aware(monkeypatch):
    """The wedge-recovery reset must match the host: a plain PCIe board resets with -r, a Galaxy host
    needs the galaxy-tray reset (else the board is left wedged and the next run can't fire), and an
    explicit override always wins. Detected once at healthy startup, never re-probed on a wedged board.

    Trust order: env override -> tt-smi galaxy-tray capability probe (authoritative) -> cheap hints
    (box/board name says 'galaxy', or >=32 chips)."""
    from agent import probes as P

    monkeypatch.delenv("TT_HW_PLANNER_RESET_ARGS", raising=False)
    monkeypatch.delenv("TT_HW_PLANNER_GALAXY", raising=False)

    # OBVIOUSLY non-galaxy (named small-chip PCIe board): the probe must NOT even run -> zero extra
    # commands, normal boards left untouched. (This probe would blow up if called.)
    def _boom(smi):
        raise AssertionError("capability probe ran on an obviously non-galaxy board")

    monkeypatch.setattr(P, "_galaxy_capability_probe", _boom)
    P.note_board("p300 L", 4)
    assert P._reset_arg_sets() == [["-r"]]

    # probe UNAVAILABLE (e.g. tt-smi missing) -> fall back to hints. never touch real hardware in tests.
    monkeypatch.setattr(P, "_galaxy_capability_probe", lambda smi: None)
    P.note_board("galaxy wormhole", 32)
    assert P._reset_arg_sets() == [["-glx_reset_auto"], ["-glx_reset"], ["-r"]]
    P.note_board("n300 L", 32)  # >=32 chips backstop
    assert P._reset_arg_sets()[0] == ["-glx_reset_auto"]
    P.note_board("qb2", 4, box="Galaxy")  # box name says galaxy even when chip count is small
    assert P._reset_arg_sets()[0] == ["-glx_reset_auto"]

    # probe AUTHORITATIVE (only runs when the board isn't obviously plain): overrides hints both ways
    monkeypatch.setattr(P, "_galaxy_capability_probe", lambda smi: True)
    P.note_board("mystery", 0)  # unknown chip count -> not obvious -> probe runs -> galaxy
    assert P._reset_arg_sets()[0] == ["-glx_reset_auto"]
    monkeypatch.setattr(P, "_galaxy_capability_probe", lambda smi: False)
    P.note_board("galaxy-ish", 64)  # name/count look galaxy -> probe runs -> probe says NO -> plain board
    assert P._reset_arg_sets() == [["-r"]]

    # explicit reset-args override wins over any detection
    monkeypatch.setattr(P, "_galaxy_capability_probe", lambda smi: True)
    P.note_board("p300 L", 4)
    monkeypatch.setenv("TT_HW_PLANNER_RESET_ARGS", "-glx_reset_tray 2")
    assert P._reset_arg_sets() == [["-glx_reset_tray", "2"]]
    monkeypatch.delenv("TT_HW_PLANNER_RESET_ARGS")

    # TT_HW_PLANNER_GALAXY forces galaxy without probing (guaranteed escape hatch on a real Galaxy host)
    monkeypatch.setenv("TT_HW_PLANNER_GALAXY", "1")
    P.note_board("p300 L", 4)
    assert P._reset_arg_sets()[0] == ["-glx_reset_auto"]
