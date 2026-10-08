# fourth pass, the PR code: the two-section operand pass (#58725) and native routing (#58726) against the head without them
EB_R3_NO_PRE_SECTIONS=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_mp_dump
EB_R3_NO_NATIVE=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_nat_dump
