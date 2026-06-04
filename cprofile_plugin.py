import cProfile, pstats, io, os

_prof = cProfile.Profile()
_on = os.environ.get("CPROFILE") == "1"
_out = os.environ.get("CPROFILE_OUT", "/tmp/cprofile_out.txt")


def pytest_sessionstart(session):
    if _on:
        _prof.enable()


def pytest_sessionfinish(session, exitstatus):
    if not _on:
        return
    _prof.disable()
    buf = io.StringIO()
    for sort in ("tottime", "cumulative"):
        s = io.StringIO()
        pstats.Stats(_prof, stream=s).sort_stats(sort).print_stats(40)
        buf.write(f"\n===== CPROFILE sort={sort} =====\n{s.getvalue()}")
    with open(_out, "w") as f:
        f.write(buf.getvalue())
    print(f"\n[cprofile_plugin] wrote profile to {_out}", flush=True)
