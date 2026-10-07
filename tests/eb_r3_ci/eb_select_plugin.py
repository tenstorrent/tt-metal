# Round 3 eltwise binary: keep only the test items whose node ids are listed in $EB_NODES_FILE (one per line), so that ids
# with spaces or parentheses need not go through the profiler's shell command line.
import os


def pytest_collection_modifyitems(config, items):
    path = os.environ.get("EB_NODES_FILE")
    if not path:
        return
    want = {l.strip() for l in open(path) if l.strip()}
    keep = [it for it in items if it.nodeid in want]
    drop = [it for it in items if it.nodeid not in want]
    missing = want - {it.nodeid for it in keep}
    for m in sorted(missing):
        print(f"EB_SELECT missing: {m}")
    items[:] = keep
    if drop:
        config.hook.pytest_deselected(items=drop)
