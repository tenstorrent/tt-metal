# fourth pass: the multi-section operand pass (#58725) and native routing of block or width sharded broadcasts (#58726)
EB_R3_NONE=1|EB_R3_MULTI_PASS=2|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_mp_dump
EB_R3_NONE=1|EB_R3_NATIVE_BCAST=1|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_nat_dump
