"""I-2 tests: Checkpoint with WAL semantics (PLAN section 5)."""

from agent.checkpoint import Checkpoint


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    state = {"run_id": "r1", "state": "ROUTE", "iteration": 2}
    cp.save(state)
    assert cp.load() == state


def test_resume_returns_recorded_state(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.save({"state": "APPLY", "iteration": 4})
    # A fresh handle (simulating a resume) reads the recorded state, not START.
    resumed = Checkpoint(tmp_path / "state.json")
    assert resumed.load()["state"] == "APPLY"


def test_crash_between_intent_and_done_reverts(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.save({"state": "ROUTE", "iteration": 4})

    # Record intent for APPLY, then "crash" before mark_done().
    cp.mark_intent(state="APPLY", current_lever="mlp-fidelity-walk", git_sha_clean="abc123")

    resumed = Checkpoint(tmp_path / "state.json")
    assert resumed.is_in_flight() is True
    assert resumed.load()["git_sha_clean"] == "abc123"
    assert resumed.load()["current_lever"] == "mlp-fidelity-walk"


def test_mark_done_clears_in_flight(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.save({"state": "ROUTE"})
    cp.mark_intent(state="APPLY", git_sha_clean="abc123")
    assert cp.is_in_flight() is True
    cp.mark_done()
    assert cp.is_in_flight() is False


def test_is_in_flight_false_when_no_file(tmp_path):
    cp = Checkpoint(tmp_path / "missing.json")
    assert cp.is_in_flight() is False
