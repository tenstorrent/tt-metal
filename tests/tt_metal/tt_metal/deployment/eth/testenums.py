from enum import Enum
from itertools import chain
from dataclasses import dataclass


@dataclass
class TestInfo:
    tests_per_link: int = 1


class TestCase(str, Enum):
    LINK_UP = "LinkUp"
    BANDWIDTH = "Bandwidth"
    BANDWIDTH_BIDIR = "BandwidthBidir"
    DATA_INTEGRITY = "DataIntegrityDram"
    DATA_INTEGRITY_BIDIR = "DataIntegrityDramBidir"
    STRESS_TEST = "StressTest"


class DistTestCase(str, Enum):
    LINK_UP = "LinkUpDist"
    BANDWIDTH = "BandwidthDist"
    BANDWIDTH_BIDIR = "BandwidthBidirDist"
    DATA_INTEGRITY = "DataIntegrityDramDist"
    DATA_INTEGRITY_BIDIR = "DataIntegrityDramBidirDist"

    # TODO
    def get_test_info(t: "DistTestCase") -> TestInfo:
        global dist_test_info

        if t in dist_test_info:
            return dist_test_info[t]

        return TestInfo()


# TODO
# dist_test_info: dict[DistTestCase, TestInfo] = {
#     DistTestCase.LINK_UP: TestInfo(tests_per_link=2),
# }

ALLTESTS = [t for t in chain(TestCase, DistTestCase)]


def prepare_filter(tests: list[TestCase | DistTestCase]) -> str:
    return ":".join(map(lambda x: f"*TensixDeploymentEthernet*{x}", tests))
