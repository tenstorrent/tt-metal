"""Round 3 eltwise binary (#58723 third review): drop test_post_sdpa's skip mark (#42714, PCC of the live multi-device cases in
an isolated fabric_2d run) so that its single-device case can be timed main against the opt-in. Every other mark stays."""
import os as _os_guard, sys as _sys_guard
if not (_os_guard.environ.get("HWLOCK_HELD") or _os_guard.environ.get("GITHUB_ACTIONS")):
    _sys_guard.exit("not under hwlock")


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.own_markers[:] = [m for m in item.own_markers if not (m.name == "skip" and "#42714" in str(m.kwargs.get("reason", "")))]
