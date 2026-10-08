# fifth pass, final code (#58726): native routing under the widened rule against main's routing
EB_R3_NO_NATIVE=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r5.py -k "test_nat5_dump or test_act5_dump"
EB_R3_NO_NATIVE=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_nat_dump
