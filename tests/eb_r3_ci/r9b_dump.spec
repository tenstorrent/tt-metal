# sixth pass, merged head: the operand pass against main's pass per section, at the planned sections and at 2, 3, 5, 6 and 7
# (the counts the L1 step-down can give); the native classes against main's routing
EB_R3_NO_PRE_SECTIONS=1|EB_R3_NONE=1;EB_R3_NO_PRE_SECTIONS=1|EB_R3_PRE_MAX=2;EB_R3_NO_PRE_SECTIONS=1|EB_R3_PRE_MAX=3;EB_R3_NO_PRE_SECTIONS=1|EB_R3_PRE_MAX=5;EB_R3_NO_PRE_SECTIONS=1|EB_R3_PRE_MAX=6;EB_R3_NO_PRE_SECTIONS=1|EB_R3_PRE_MAX=7|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_mp_dump
EB_R3_NO_NATIVE=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r5.py -k test_nat6_dump
