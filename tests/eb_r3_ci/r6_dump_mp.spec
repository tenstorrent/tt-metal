# fifth pass, final code (#58725): the operand pass over up to four sections and over one, against main's pass per section
# and against the fourth pass's two sections
EB_R3_NO_PRE_SECTIONS=1|EB_R3_NONE=1;EB_R3_PRE_K2=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r4.py -k test_mp_dump
EB_R3_NO_PRE_SECTIONS=1|EB_R3_NONE=1|||tests/eb_r3_ci/test_eb_dump_r5.py -k test_mp5_dump
