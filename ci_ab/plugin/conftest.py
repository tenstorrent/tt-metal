# Round 3 tilize: keep only the node ids listed in FR_SELECT (one per line), also when pytest-repeat appended "-i-n".
import os
import re

import pytest


def _base(nodeid):
    return re.sub(r"\[(.*?)-?\d+-\d+\]$", lambda m: f"[{m.group(1)}]" if m.group(1) else "", nodeid)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    sel = os.environ.get("FR_SELECT")
    if not sel:
        return
    wanted = {l.strip() for l in open(sel) if l.strip()}
    keep = [it for it in items if it.nodeid in wanted or _base(it.nodeid) in wanted]
    dropped = [it for it in items if it not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
