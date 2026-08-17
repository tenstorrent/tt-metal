"""No-device control importing the repository's shared TT test utilities."""

from models.common.utility_functions import comp_pcc, run_for_blackhole

print(f"SHARED_TEST_UTILITY_CONTROL_OK symbols={comp_pcc.__name__},{run_for_blackhole.__name__}")
