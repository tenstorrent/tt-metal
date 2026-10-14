# pytest plugin: log (node id, variant hash) for every variant a test builds; file from RA_HASHLOG.
import os
def pytest_configure(config):
    out = os.environ.get("RA_HASHLOG")
    if not out:
        return
    from helpers import test_config as tc
    orig = tc.TestConfig.generate_variant_hash
    def wrapped(self, *a, **k):
        r = orig(self, *a, **k)
        node = os.environ.get("PYTEST_CURRENT_TEST", "?").rsplit(" (", 1)[0]
        with open(out, "a") as f:
            f.write(f"{node}\t{self.variant_id}\n")
        return r
    tc.TestConfig.generate_variant_hash = wrapped
