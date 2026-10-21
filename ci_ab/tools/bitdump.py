"""pytest plugin: append '<node id>\t<sha256 of the result buffer>' for every TestConfig.run to $BITDUMP_FILE."""
import hashlib
import os


def pytest_configure(config):
    out = os.environ.get("BITDUMP_FILE")
    if not out:
        return
    from helpers.test_config import TestConfig

    orig = TestConfig.run

    def run(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            h = hashlib.sha256(repr([float(x) for x in r.result]).encode()).hexdigest()
        except Exception as e:  # noqa: BLE001
            h = "ERR:" + type(e).__name__
        with open(out, "a") as f:
            f.write(os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0] + "\t" + h + "\n")
        return r

    TestConfig.run = run
