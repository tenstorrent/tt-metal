"""Round 3 eltwise binary: run the rotary_embedding_llama nightly tests on our single Blackhole card. They carry
skip_for_blackhole("... #12349"), a multi-device reason that none of their cases needs; this plugin drops that one mark so
main and the branch can be compared on the card. Every other skip stays."""
import os as _os_guard, sys as _sys_guard
if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.own_markers[:] = [m for m in item.own_markers if not (m.name == "skipif" and "#12349" in str(m.kwargs.get("reason", "")))]
