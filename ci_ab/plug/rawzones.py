# pytest plugin (-p rawzones, PYTHONPATH=tools/plug): append every profiler read's raw zone entries (thread, marker,
# type, timestamp) to $RAWZONES_OUT as JSON lines, tagged with the test node id and the variant id.
import json
import os

_node = {"id": None}


def pytest_runtest_setup(item):
    _node["id"] = item.nodeid


def pytest_configure(config):
    out = os.environ.get("RAWZONES_OUT")
    if not out:
        return
    from helpers import profiler as P

    orig = P.Profiler.get_data

    def get_data(test_name, variant_id, location="0,0"):
        d = orig(test_name, variant_id, location)
        try:
            df = d.df if hasattr(d, "df") else d
            cols = [c for c in ("thread", "marker", "type", "timestamp", "duration") if c in df.columns]
            rows = df[cols].astype(str).values.tolist()
            with open(out, "a") as fh:
                fh.write(json.dumps({"node": _node["id"], "variant": variant_id, "cols": cols, "rows": rows}) + "\n")
        except Exception as e:  # diagnostics only
            with open(out, "a") as fh:
                fh.write(json.dumps({"node": _node["id"], "variant": variant_id, "error": str(e)}) + "\n")
        return d

    P.Profiler.get_data = staticmethod(get_data)
