from enum import Enum
from itertools import chain


class TestCase(str, Enum):
    LINK_UP = "LinkUp"
    BANDWIDTH = "Bandwidth"
    BANDWIDTH_BIDIR = "BandwidthBidir"
    DATA_INTEGRITY = "DataIntegrityDram"
    DATA_INTEGRITY_BIDIR = "DataIntegrityDramBidir"
    STRESS_TEST = "StressTest"


class DistTestCase(str, Enum):
    LINK_UP = "LinkUpDist"
    BANDWIDTH_BIDIR = "BandwidthBidirDist"


ALLTESTS = [t for t in chain(TestCase, DistTestCase)]


def prepare_filter(tests: list[TestCase | DistTestCase]) -> str:
    return ":".join(map(lambda x: f"*TensixDeploymentEthernet*{x}", tests))
