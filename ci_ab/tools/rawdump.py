# pytest plugin: save every Profiler.get_data result (raw zone events of all threads) to RAWDUMP_DIR as csv,
# named <test_name>__<variant_id>__<n>.csv, so per-thread zone starts and ends can be compared across builds.
import itertools, os

_count = itertools.count()


def pytest_configure(config):
    out = os.environ.get("RAWDUMP_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    from helpers import profiler as P

    orig = P.Profiler.get_data

    def get_data(test_name, variant_id, location="0,0"):
        data = orig(test_name, variant_id, location)
        try:
            df = data.df if hasattr(data, "df") else data
            df.to_csv(os.path.join(out, f"{os.path.basename(str(test_name))}__{variant_id}__{next(_count)}.csv"), index=False)
        except Exception as e:  # never break the run
            print("rawdump failed:", e)
        return data

    P.Profiler.get_data = staticmethod(get_data)
