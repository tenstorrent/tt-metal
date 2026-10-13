# pytest plugin (-p sel_plugin): keep only the items whose node id contains one of the lines of $SEL_FILE.
import os


def pytest_collection_modifyitems(session, config, items):
    f = os.environ.get("SEL_FILE")
    if not f:
        return
    keys = [l.strip() for l in open(f) if l.strip() and not l.startswith("#")]
    keep = [it for it in items if any(k in it.nodeid for k in keys)]
    drop = [it for it in items if it not in keep]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
