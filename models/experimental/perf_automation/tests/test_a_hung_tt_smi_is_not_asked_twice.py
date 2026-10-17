"""A tt-smi that hangs must not be re-asked on every probe.

A wedged ARC makes `tt-smi -s` never answer. Bounding it (which the code does) stops it hanging the
run, but every caller still pays the full timeout for a reading that is never going to arrive. On a
real wedged board this doubled the preflight suite. These pin the breaker -- and, more importantly,
pin that it costs no thermal safety."""
import time

import agent.probes as probes


def _reset():
    probes._TT_SMI_HUNG_AT = 0.0


def test_a_hang_is_not_asked_again_immediately(monkeypatch):
    _reset()
    calls = []

    def _hang(*a, **k):
        calls.append(1)
        raise probes.subprocess.TimeoutExpired(cmd="tt-smi", timeout=15)

    monkeypatch.setattr(probes.subprocess, "run", _hang)
    assert probes._tt_smi_asic_temp() is None
    assert probes._tt_smi_asic_temp() is None
    assert probes._tt_smi_asic_temp() is None
    assert len(calls) == 1, "a hung tt-smi was asked again while still hung"
    _reset()


def test_the_breaker_reopens_after_its_window(monkeypatch):
    _reset()
    calls = []

    def _hang(*a, **k):
        calls.append(1)
        raise probes.subprocess.TimeoutExpired(cmd="tt-smi", timeout=15)

    monkeypatch.setattr(probes.subprocess, "run", _hang)
    probes._tt_smi_asic_temp()
    # step past the window rather than sleeping through it
    probes._TT_SMI_HUNG_AT = time.time() - (probes._TT_SMI_BREAKER_S + 1)
    probes._tt_smi_asic_temp()
    assert len(calls) == 2, "the breaker never reopens -- tt-smi is dead for the rest of the run"
    _reset()


def test_an_answer_closes_the_breaker(monkeypatch):
    """Recovery must be immediate: one clean read and tt-smi is trusted again."""
    _reset()
    probes._TT_SMI_HUNG_AT = time.time() - (probes._TT_SMI_BREAKER_S + 1)

    class _R:
        stdout = "{}"

    monkeypatch.setattr(probes.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(probes, "_max_asic_temp", lambda _d: 61.0)
    assert probes._tt_smi_asic_temp() == 61.0
    assert probes._TT_SMI_HUNG_AT == 0.0, "a good reading did not clear the breaker"
    _reset()


def test_the_breaker_never_silences_sysfs(monkeypatch):
    """THE SAFETY PROPERTY. sysfs is the source that still works on a wedged board, so an open
    breaker must not stop a hot reading from being seen."""
    _reset()
    probes._TT_SMI_HUNG_AT = time.time()  # breaker wide open

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [95.0])
    assert probes._read_asic_temp() == 95.0, "an open tt-smi breaker hid a hot board"
    _reset()
