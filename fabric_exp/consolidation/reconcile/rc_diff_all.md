## glx_mesh_4352: 168 study router builds, 168 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## glx_torus_4352: 120 study router builds, 120 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (10), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=2, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## glx_vc1split4x4_4352: 296 study router builds, 296 port

slot rows (count of allocator logs): study VC0 sender slots {19, 9, 9, 9} receiver slots {16} (channel buffer 4448 B) x2; VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 4448 B) x10; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 4448 B) x10; VC1 sender slots {5, 5, 5} receiver slots {4} (channel buffer 4448 B) x2

port VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 4448 B) x10; VC0 sender slots {51, 4, 4, 4} receiver slots {8} (channel buffer 4448 B) x2; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 4448 B) x10; VC1 sender slots {2, 2, 2} receiver slots {4} (channel buffer 4448 B) x2

study: opt ['-O3'], positional CT args [65]
port: opt ['-O3'], positional CT args [65]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (10), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0,1, PEER_SENDER_CHANNEL_5_SERVICED=0,1, PEER_SENDER_CHANNEL_6_SERVICED=0,1, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=4, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=4, RX_CH_1_TRID_START=4, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=4

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 38 / 38 | CT[3] | 19 | 51 |
| active_erisc | E | 38 / 38 | CT[5] | 19 | 51 |
| active_erisc | E | 38 / 38 | CT[6] | 175424 | 317760 |
| active_erisc | E | 38 / 38 | CT[7] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[8] | 175424 | 317760 |
| active_erisc | E | 38 / 38 | CT[9] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[10] | 215456 | 335552 |
| active_erisc | E | 38 / 38 | CT[11] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[12] | 215456 | 335552 |
| active_erisc | E | 38 / 38 | CT[13] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[14] | 255488 | 353344 |
| active_erisc | E | 38 / 38 | CT[15] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[16] | 255488 | 353344 |
| active_erisc | E | 38 / 38 | CT[17] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[18] | 295520 | 371136 |
| active_erisc | E | 38 / 38 | CT[19] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[20] | 295520 | 371136 |
| active_erisc | E | 38 / 38 | CT[21] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[22] | 317760 | 380032 |
| active_erisc | E | 38 / 38 | CT[23] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[24] | 317760 | 380032 |
| active_erisc | E | 38 / 38 | CT[25] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[26] | 340000 | 388928 |
| active_erisc | E | 38 / 38 | CT[27] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[28] | 340000 | 388928 |
| active_erisc | E | 38 / 38 | CT[29] | 5 | 2 |
| active_erisc | E | 38 / 38 | CT[30] | 362240 | 397824 |
| active_erisc | E | 38 / 38 | CT[31] | 16 | 8 |
| active_erisc | E | 38 / 38 | CT[32] | 362240 | 397824 |
| active_erisc | E | 38 / 38 | CT[33] | 16 | 8 |
| active_erisc | E | 38 / 38 | CT[50] | 362240 | 397824 |
| active_erisc | E | 38 / 38 | CT[51] | 16 | 8 |
| active_erisc | E | 38 / 38 | CT[61] | 9 | 4 |
| active_erisc | E | 38 / 38 | CT[62] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[3] | 19 | 51 |
| active_erisc | N | 40 / 40 | CT[5] | 19 | 51 |
| active_erisc | N | 40 / 40 | CT[6] | 175424 | 317760 |
| active_erisc | N | 40 / 40 | CT[7] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[8] | 175424 | 317760 |
| active_erisc | N | 40 / 40 | CT[9] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[10] | 215456 | 335552 |
| active_erisc | N | 40 / 40 | CT[11] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[12] | 215456 | 335552 |
| active_erisc | N | 40 / 40 | CT[13] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[14] | 255488 | 353344 |
| active_erisc | N | 40 / 40 | CT[15] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[16] | 255488 | 353344 |
| active_erisc | N | 40 / 40 | CT[17] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[18] | 295520 | 371136 |
| active_erisc | N | 40 / 40 | CT[19] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[20] | 295520 | 371136 |
| active_erisc | N | 40 / 40 | CT[21] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[22] | 317760 | 380032 |
| active_erisc | N | 40 / 40 | CT[23] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[24] | 317760 | 380032 |
| active_erisc | N | 40 / 40 | CT[25] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[26] | 340000 | 388928 |
| active_erisc | N | 40 / 40 | CT[27] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[28] | 340000 | 388928 |
| active_erisc | N | 40 / 40 | CT[29] | 5 | 2 |
| active_erisc | N | 40 / 40 | CT[30] | 362240 | 397824 |
| active_erisc | N | 40 / 40 | CT[31] | 16 | 8 |
| active_erisc | N | 40 / 40 | CT[32] | 362240 | 397824 |
| active_erisc | N | 40 / 40 | CT[33] | 16 | 8 |
| active_erisc | N | 40 / 40 | CT[50] | 362240 | 397824 |
| active_erisc | N | 40 / 40 | CT[51] | 16 | 8 |
| active_erisc | N | 40 / 40 | CT[61] | 9 | 4 |
| active_erisc | N | 40 / 40 | CT[62] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[3] | 19 | 51 |
| active_erisc | S | 38 / 38 | CT[5] | 19 | 51 |
| active_erisc | S | 38 / 38 | CT[6] | 175424 | 317760 |
| active_erisc | S | 38 / 38 | CT[7] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[8] | 175424 | 317760 |
| active_erisc | S | 38 / 38 | CT[9] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[10] | 215456 | 335552 |
| active_erisc | S | 38 / 38 | CT[11] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[12] | 215456 | 335552 |
| active_erisc | S | 38 / 38 | CT[13] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[14] | 255488 | 353344 |
| active_erisc | S | 38 / 38 | CT[15] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[16] | 255488 | 353344 |
| active_erisc | S | 38 / 38 | CT[17] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[18] | 295520 | 371136 |
| active_erisc | S | 38 / 38 | CT[19] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[20] | 295520 | 371136 |
| active_erisc | S | 38 / 38 | CT[21] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[22] | 317760 | 380032 |
| active_erisc | S | 38 / 38 | CT[23] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[24] | 317760 | 380032 |
| active_erisc | S | 38 / 38 | CT[25] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[26] | 340000 | 388928 |
| active_erisc | S | 38 / 38 | CT[27] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[28] | 340000 | 388928 |
| active_erisc | S | 38 / 38 | CT[29] | 5 | 2 |
| active_erisc | S | 38 / 38 | CT[30] | 362240 | 397824 |
| active_erisc | S | 38 / 38 | CT[31] | 16 | 8 |
| active_erisc | S | 38 / 38 | CT[32] | 362240 | 397824 |
| active_erisc | S | 38 / 38 | CT[33] | 16 | 8 |
| active_erisc | S | 38 / 38 | CT[50] | 362240 | 397824 |
| active_erisc | S | 38 / 38 | CT[51] | 16 | 8 |
| active_erisc | S | 38 / 38 | CT[61] | 9 | 4 |
| active_erisc | S | 38 / 38 | CT[62] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[3] | 19 | 51 |
| active_erisc | W | 32 / 32 | CT[5] | 19 | 51 |
| active_erisc | W | 32 / 32 | CT[6] | 175424 | 317760 |
| active_erisc | W | 32 / 32 | CT[7] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[8] | 175424 | 317760 |
| active_erisc | W | 32 / 32 | CT[9] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[10] | 215456 | 335552 |
| active_erisc | W | 32 / 32 | CT[11] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[12] | 215456 | 335552 |
| active_erisc | W | 32 / 32 | CT[13] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[14] | 255488 | 353344 |
| active_erisc | W | 32 / 32 | CT[15] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[16] | 255488 | 353344 |
| active_erisc | W | 32 / 32 | CT[17] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[18] | 295520 | 371136 |
| active_erisc | W | 32 / 32 | CT[19] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[20] | 295520 | 371136 |
| active_erisc | W | 32 / 32 | CT[21] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[22] | 317760 | 380032 |
| active_erisc | W | 32 / 32 | CT[23] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[24] | 317760 | 380032 |
| active_erisc | W | 32 / 32 | CT[25] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[26] | 340000 | 388928 |
| active_erisc | W | 32 / 32 | CT[27] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[28] | 340000 | 388928 |
| active_erisc | W | 32 / 32 | CT[29] | 5 | 2 |
| active_erisc | W | 32 / 32 | CT[30] | 362240 | 397824 |
| active_erisc | W | 32 / 32 | CT[31] | 16 | 8 |
| active_erisc | W | 32 / 32 | CT[32] | 362240 | 397824 |
| active_erisc | W | 32 / 32 | CT[33] | 16 | 8 |
| active_erisc | W | 32 / 32 | CT[50] | 362240 | 397824 |
| active_erisc | W | 32 / 32 | CT[51] | 16 | 8 |
| active_erisc | W | 32 / 32 | CT[61] | 9 | 4 |
| active_erisc | W | 32 / 32 | CT[62] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[3] | 19 | 51 |
| subordinate_active_erisc | E | 38 / 38 | CT[5] | 19 | 51 |
| subordinate_active_erisc | E | 38 / 38 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | E | 38 / 38 | CT[7] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | E | 38 / 38 | CT[9] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | E | 38 / 38 | CT[11] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | E | 38 / 38 | CT[13] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | E | 38 / 38 | CT[15] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | E | 38 / 38 | CT[17] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | E | 38 / 38 | CT[19] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | E | 38 / 38 | CT[21] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | E | 38 / 38 | CT[23] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | E | 38 / 38 | CT[25] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | E | 38 / 38 | CT[27] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | E | 38 / 38 | CT[29] | 5 | 2 |
| subordinate_active_erisc | E | 38 / 38 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | E | 38 / 38 | CT[31] | 16 | 8 |
| subordinate_active_erisc | E | 38 / 38 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | E | 38 / 38 | CT[33] | 16 | 8 |
| subordinate_active_erisc | E | 38 / 38 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | E | 38 / 38 | CT[51] | 16 | 8 |
| subordinate_active_erisc | E | 38 / 38 | CT[61] | 9 | 4 |
| subordinate_active_erisc | E | 38 / 38 | CT[62] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[3] | 19 | 51 |
| subordinate_active_erisc | N | 40 / 40 | CT[5] | 19 | 51 |
| subordinate_active_erisc | N | 40 / 40 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | N | 40 / 40 | CT[7] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | N | 40 / 40 | CT[9] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | N | 40 / 40 | CT[11] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | N | 40 / 40 | CT[13] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | N | 40 / 40 | CT[15] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | N | 40 / 40 | CT[17] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | N | 40 / 40 | CT[19] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | N | 40 / 40 | CT[21] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | N | 40 / 40 | CT[23] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | N | 40 / 40 | CT[25] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | N | 40 / 40 | CT[27] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | N | 40 / 40 | CT[29] | 5 | 2 |
| subordinate_active_erisc | N | 40 / 40 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | N | 40 / 40 | CT[31] | 16 | 8 |
| subordinate_active_erisc | N | 40 / 40 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | N | 40 / 40 | CT[33] | 16 | 8 |
| subordinate_active_erisc | N | 40 / 40 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | N | 40 / 40 | CT[51] | 16 | 8 |
| subordinate_active_erisc | N | 40 / 40 | CT[61] | 9 | 4 |
| subordinate_active_erisc | N | 40 / 40 | CT[62] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[3] | 19 | 51 |
| subordinate_active_erisc | S | 38 / 38 | CT[5] | 19 | 51 |
| subordinate_active_erisc | S | 38 / 38 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | S | 38 / 38 | CT[7] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | S | 38 / 38 | CT[9] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | S | 38 / 38 | CT[11] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | S | 38 / 38 | CT[13] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | S | 38 / 38 | CT[15] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | S | 38 / 38 | CT[17] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | S | 38 / 38 | CT[19] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | S | 38 / 38 | CT[21] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | S | 38 / 38 | CT[23] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | S | 38 / 38 | CT[25] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | S | 38 / 38 | CT[27] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | S | 38 / 38 | CT[29] | 5 | 2 |
| subordinate_active_erisc | S | 38 / 38 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | S | 38 / 38 | CT[31] | 16 | 8 |
| subordinate_active_erisc | S | 38 / 38 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | S | 38 / 38 | CT[33] | 16 | 8 |
| subordinate_active_erisc | S | 38 / 38 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | S | 38 / 38 | CT[51] | 16 | 8 |
| subordinate_active_erisc | S | 38 / 38 | CT[61] | 9 | 4 |
| subordinate_active_erisc | S | 38 / 38 | CT[62] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[3] | 19 | 51 |
| subordinate_active_erisc | W | 32 / 32 | CT[5] | 19 | 51 |
| subordinate_active_erisc | W | 32 / 32 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | W | 32 / 32 | CT[7] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | W | 32 / 32 | CT[9] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | W | 32 / 32 | CT[11] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | W | 32 / 32 | CT[13] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | W | 32 / 32 | CT[15] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | W | 32 / 32 | CT[17] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | W | 32 / 32 | CT[19] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | W | 32 / 32 | CT[21] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | W | 32 / 32 | CT[23] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | W | 32 / 32 | CT[25] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | W | 32 / 32 | CT[27] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | W | 32 / 32 | CT[29] | 5 | 2 |
| subordinate_active_erisc | W | 32 / 32 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | W | 32 / 32 | CT[31] | 16 | 8 |
| subordinate_active_erisc | W | 32 / 32 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | W | 32 / 32 | CT[33] | 16 | 8 |
| subordinate_active_erisc | W | 32 / 32 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | W | 32 / 32 | CT[51] | 16 | 8 |
| subordinate_active_erisc | W | 32 / 32 | CT[61] | 9 | 4 |
| subordinate_active_erisc | W | 32 / 32 | CT[62] | 5 | 2 |

## glx_vc1split_3072: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 3168 B) x10; VC0 sender slots {53, 9, 9, 9} receiver slots {16} (channel buffer 3168 B) x2; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 3168 B) x10; VC1 sender slots {5, 5, 5} receiver slots {4} (channel buffer 3168 B) x2

port VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 3168 B) x10; VC0 sender slots {85, 4, 4, 4} receiver slots {8} (channel buffer 3168 B) x2; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 3168 B) x10; VC1 sender slots {2, 2, 2} receiver slots {4} (channel buffer 3168 B) x2

study: opt ['-O3'], positional CT args [65]
port: opt ['-O3'], positional CT args [65]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0,1, PEER_SENDER_CHANNEL_5_SERVICED=0,1, PEER_SENDER_CHANNEL_6_SERVICED=0,1, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=4, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=4, RX_CH_1_TRID_START=4, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=4

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 10 / 10 | CT[3] | 53 | 85 |
| active_erisc | E | 10 / 10 | CT[5] | 53 | 85 |
| active_erisc | E | 10 / 10 | CT[6] | 258816 | 360192 |
| active_erisc | E | 10 / 10 | CT[7] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[8] | 258816 | 360192 |
| active_erisc | E | 10 / 10 | CT[9] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[10] | 287328 | 372864 |
| active_erisc | E | 10 / 10 | CT[11] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[12] | 287328 | 372864 |
| active_erisc | E | 10 / 10 | CT[13] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[14] | 315840 | 385536 |
| active_erisc | E | 10 / 10 | CT[15] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[16] | 315840 | 385536 |
| active_erisc | E | 10 / 10 | CT[17] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[18] | 344352 | 398208 |
| active_erisc | E | 10 / 10 | CT[19] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[20] | 344352 | 398208 |
| active_erisc | E | 10 / 10 | CT[21] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[22] | 360192 | 404544 |
| active_erisc | E | 10 / 10 | CT[23] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[24] | 360192 | 404544 |
| active_erisc | E | 10 / 10 | CT[25] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[26] | 376032 | 410880 |
| active_erisc | E | 10 / 10 | CT[27] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[28] | 376032 | 410880 |
| active_erisc | E | 10 / 10 | CT[29] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[30] | 391872 | 417216 |
| active_erisc | E | 10 / 10 | CT[31] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[32] | 391872 | 417216 |
| active_erisc | E | 10 / 10 | CT[33] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[50] | 391872 | 417216 |
| active_erisc | E | 10 / 10 | CT[51] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[61] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[62] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[3] | 53 | 85 |
| active_erisc | N | 12 / 12 | CT[5] | 53 | 85 |
| active_erisc | N | 12 / 12 | CT[6] | 258816 | 360192 |
| active_erisc | N | 12 / 12 | CT[7] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[8] | 258816 | 360192 |
| active_erisc | N | 12 / 12 | CT[9] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[10] | 287328 | 372864 |
| active_erisc | N | 12 / 12 | CT[11] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[12] | 287328 | 372864 |
| active_erisc | N | 12 / 12 | CT[13] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[14] | 315840 | 385536 |
| active_erisc | N | 12 / 12 | CT[15] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[16] | 315840 | 385536 |
| active_erisc | N | 12 / 12 | CT[17] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[18] | 344352 | 398208 |
| active_erisc | N | 12 / 12 | CT[19] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[20] | 344352 | 398208 |
| active_erisc | N | 12 / 12 | CT[21] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[22] | 360192 | 404544 |
| active_erisc | N | 12 / 12 | CT[23] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[24] | 360192 | 404544 |
| active_erisc | N | 12 / 12 | CT[25] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[26] | 376032 | 410880 |
| active_erisc | N | 12 / 12 | CT[27] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[28] | 376032 | 410880 |
| active_erisc | N | 12 / 12 | CT[29] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[30] | 391872 | 417216 |
| active_erisc | N | 12 / 12 | CT[31] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[32] | 391872 | 417216 |
| active_erisc | N | 12 / 12 | CT[33] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[50] | 391872 | 417216 |
| active_erisc | N | 12 / 12 | CT[51] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[61] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[62] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[3] | 53 | 85 |
| active_erisc | S | 10 / 10 | CT[5] | 53 | 85 |
| active_erisc | S | 10 / 10 | CT[6] | 258816 | 360192 |
| active_erisc | S | 10 / 10 | CT[7] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[8] | 258816 | 360192 |
| active_erisc | S | 10 / 10 | CT[9] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[10] | 287328 | 372864 |
| active_erisc | S | 10 / 10 | CT[11] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[12] | 287328 | 372864 |
| active_erisc | S | 10 / 10 | CT[13] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[14] | 315840 | 385536 |
| active_erisc | S | 10 / 10 | CT[15] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[16] | 315840 | 385536 |
| active_erisc | S | 10 / 10 | CT[17] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[18] | 344352 | 398208 |
| active_erisc | S | 10 / 10 | CT[19] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[20] | 344352 | 398208 |
| active_erisc | S | 10 / 10 | CT[21] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[22] | 360192 | 404544 |
| active_erisc | S | 10 / 10 | CT[23] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[24] | 360192 | 404544 |
| active_erisc | S | 10 / 10 | CT[25] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[26] | 376032 | 410880 |
| active_erisc | S | 10 / 10 | CT[27] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[28] | 376032 | 410880 |
| active_erisc | S | 10 / 10 | CT[29] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[30] | 391872 | 417216 |
| active_erisc | S | 10 / 10 | CT[31] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[32] | 391872 | 417216 |
| active_erisc | S | 10 / 10 | CT[33] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[50] | 391872 | 417216 |
| active_erisc | S | 10 / 10 | CT[51] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[61] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[62] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[3] | 53 | 85 |
| active_erisc | W | 8 / 8 | CT[5] | 53 | 85 |
| active_erisc | W | 8 / 8 | CT[6] | 258816 | 360192 |
| active_erisc | W | 8 / 8 | CT[7] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[8] | 258816 | 360192 |
| active_erisc | W | 8 / 8 | CT[9] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[10] | 287328 | 372864 |
| active_erisc | W | 8 / 8 | CT[11] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[12] | 287328 | 372864 |
| active_erisc | W | 8 / 8 | CT[13] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[14] | 315840 | 385536 |
| active_erisc | W | 8 / 8 | CT[15] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[16] | 315840 | 385536 |
| active_erisc | W | 8 / 8 | CT[17] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[18] | 344352 | 398208 |
| active_erisc | W | 8 / 8 | CT[19] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[20] | 344352 | 398208 |
| active_erisc | W | 8 / 8 | CT[21] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[22] | 360192 | 404544 |
| active_erisc | W | 8 / 8 | CT[23] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[24] | 360192 | 404544 |
| active_erisc | W | 8 / 8 | CT[25] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[26] | 376032 | 410880 |
| active_erisc | W | 8 / 8 | CT[27] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[28] | 376032 | 410880 |
| active_erisc | W | 8 / 8 | CT[29] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[30] | 391872 | 417216 |
| active_erisc | W | 8 / 8 | CT[31] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[32] | 391872 | 417216 |
| active_erisc | W | 8 / 8 | CT[33] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[50] | 391872 | 417216 |
| active_erisc | W | 8 / 8 | CT[51] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[61] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[62] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[3] | 53 | 85 |
| subordinate_active_erisc | E | 10 / 10 | CT[5] | 53 | 85 |
| subordinate_active_erisc | E | 10 / 10 | CT[6] | 258816 | 360192 |
| subordinate_active_erisc | E | 10 / 10 | CT[7] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[8] | 258816 | 360192 |
| subordinate_active_erisc | E | 10 / 10 | CT[9] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[10] | 287328 | 372864 |
| subordinate_active_erisc | E | 10 / 10 | CT[11] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[12] | 287328 | 372864 |
| subordinate_active_erisc | E | 10 / 10 | CT[13] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[14] | 315840 | 385536 |
| subordinate_active_erisc | E | 10 / 10 | CT[15] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[16] | 315840 | 385536 |
| subordinate_active_erisc | E | 10 / 10 | CT[17] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[18] | 344352 | 398208 |
| subordinate_active_erisc | E | 10 / 10 | CT[19] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[20] | 344352 | 398208 |
| subordinate_active_erisc | E | 10 / 10 | CT[21] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[22] | 360192 | 404544 |
| subordinate_active_erisc | E | 10 / 10 | CT[23] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[24] | 360192 | 404544 |
| subordinate_active_erisc | E | 10 / 10 | CT[25] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[26] | 376032 | 410880 |
| subordinate_active_erisc | E | 10 / 10 | CT[27] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[28] | 376032 | 410880 |
| subordinate_active_erisc | E | 10 / 10 | CT[29] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[30] | 391872 | 417216 |
| subordinate_active_erisc | E | 10 / 10 | CT[31] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[32] | 391872 | 417216 |
| subordinate_active_erisc | E | 10 / 10 | CT[33] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[50] | 391872 | 417216 |
| subordinate_active_erisc | E | 10 / 10 | CT[51] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[61] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[62] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[3] | 53 | 85 |
| subordinate_active_erisc | N | 12 / 12 | CT[5] | 53 | 85 |
| subordinate_active_erisc | N | 12 / 12 | CT[6] | 258816 | 360192 |
| subordinate_active_erisc | N | 12 / 12 | CT[7] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[8] | 258816 | 360192 |
| subordinate_active_erisc | N | 12 / 12 | CT[9] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[10] | 287328 | 372864 |
| subordinate_active_erisc | N | 12 / 12 | CT[11] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[12] | 287328 | 372864 |
| subordinate_active_erisc | N | 12 / 12 | CT[13] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[14] | 315840 | 385536 |
| subordinate_active_erisc | N | 12 / 12 | CT[15] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[16] | 315840 | 385536 |
| subordinate_active_erisc | N | 12 / 12 | CT[17] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[18] | 344352 | 398208 |
| subordinate_active_erisc | N | 12 / 12 | CT[19] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[20] | 344352 | 398208 |
| subordinate_active_erisc | N | 12 / 12 | CT[21] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[22] | 360192 | 404544 |
| subordinate_active_erisc | N | 12 / 12 | CT[23] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[24] | 360192 | 404544 |
| subordinate_active_erisc | N | 12 / 12 | CT[25] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[26] | 376032 | 410880 |
| subordinate_active_erisc | N | 12 / 12 | CT[27] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[28] | 376032 | 410880 |
| subordinate_active_erisc | N | 12 / 12 | CT[29] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[30] | 391872 | 417216 |
| subordinate_active_erisc | N | 12 / 12 | CT[31] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[32] | 391872 | 417216 |
| subordinate_active_erisc | N | 12 / 12 | CT[33] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[50] | 391872 | 417216 |
| subordinate_active_erisc | N | 12 / 12 | CT[51] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[61] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[62] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[3] | 53 | 85 |
| subordinate_active_erisc | S | 10 / 10 | CT[5] | 53 | 85 |
| subordinate_active_erisc | S | 10 / 10 | CT[6] | 258816 | 360192 |
| subordinate_active_erisc | S | 10 / 10 | CT[7] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[8] | 258816 | 360192 |
| subordinate_active_erisc | S | 10 / 10 | CT[9] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[10] | 287328 | 372864 |
| subordinate_active_erisc | S | 10 / 10 | CT[11] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[12] | 287328 | 372864 |
| subordinate_active_erisc | S | 10 / 10 | CT[13] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[14] | 315840 | 385536 |
| subordinate_active_erisc | S | 10 / 10 | CT[15] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[16] | 315840 | 385536 |
| subordinate_active_erisc | S | 10 / 10 | CT[17] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[18] | 344352 | 398208 |
| subordinate_active_erisc | S | 10 / 10 | CT[19] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[20] | 344352 | 398208 |
| subordinate_active_erisc | S | 10 / 10 | CT[21] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[22] | 360192 | 404544 |
| subordinate_active_erisc | S | 10 / 10 | CT[23] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[24] | 360192 | 404544 |
| subordinate_active_erisc | S | 10 / 10 | CT[25] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[26] | 376032 | 410880 |
| subordinate_active_erisc | S | 10 / 10 | CT[27] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[28] | 376032 | 410880 |
| subordinate_active_erisc | S | 10 / 10 | CT[29] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[30] | 391872 | 417216 |
| subordinate_active_erisc | S | 10 / 10 | CT[31] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[32] | 391872 | 417216 |
| subordinate_active_erisc | S | 10 / 10 | CT[33] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[50] | 391872 | 417216 |
| subordinate_active_erisc | S | 10 / 10 | CT[51] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[61] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[62] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[3] | 53 | 85 |
| subordinate_active_erisc | W | 8 / 8 | CT[5] | 53 | 85 |
| subordinate_active_erisc | W | 8 / 8 | CT[6] | 258816 | 360192 |
| subordinate_active_erisc | W | 8 / 8 | CT[7] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[8] | 258816 | 360192 |
| subordinate_active_erisc | W | 8 / 8 | CT[9] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[10] | 287328 | 372864 |
| subordinate_active_erisc | W | 8 / 8 | CT[11] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[12] | 287328 | 372864 |
| subordinate_active_erisc | W | 8 / 8 | CT[13] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[14] | 315840 | 385536 |
| subordinate_active_erisc | W | 8 / 8 | CT[15] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[16] | 315840 | 385536 |
| subordinate_active_erisc | W | 8 / 8 | CT[17] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[18] | 344352 | 398208 |
| subordinate_active_erisc | W | 8 / 8 | CT[19] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[20] | 344352 | 398208 |
| subordinate_active_erisc | W | 8 / 8 | CT[21] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[22] | 360192 | 404544 |
| subordinate_active_erisc | W | 8 / 8 | CT[23] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[24] | 360192 | 404544 |
| subordinate_active_erisc | W | 8 / 8 | CT[25] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[26] | 376032 | 410880 |
| subordinate_active_erisc | W | 8 / 8 | CT[27] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[28] | 376032 | 410880 |
| subordinate_active_erisc | W | 8 / 8 | CT[29] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[30] | 391872 | 417216 |
| subordinate_active_erisc | W | 8 / 8 | CT[31] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[32] | 391872 | 417216 |
| subordinate_active_erisc | W | 8 / 8 | CT[33] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[50] | 391872 | 417216 |
| subordinate_active_erisc | W | 8 / 8 | CT[51] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[61] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[62] | 5 | 2 |

## glx_vc1split_4352: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {19, 9, 9, 9} receiver slots {16} (channel buffer 4448 B) x2; VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 4448 B) x10; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 4448 B) x10; VC1 sender slots {5, 5, 5} receiver slots {4} (channel buffer 4448 B) x2

port VC0 sender slots {4, 0, 0, 0} receiver slots {8} (channel buffer 4448 B) x10; VC0 sender slots {51, 4, 4, 4} receiver slots {8} (channel buffer 4448 B) x2; VC1 sender slots {0, 0, 0} receiver slots {4} (channel buffer 4448 B) x10; VC1 sender slots {2, 2, 2} receiver slots {4} (channel buffer 4448 B) x2

study: opt ['-O3'], positional CT args [65]
port: opt ['-O3'], positional CT args [65]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0,1, PEER_SENDER_CHANNEL_5_SERVICED=0,1, PEER_SENDER_CHANNEL_6_SERVICED=0,1, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=4, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=4, RX_CH_1_TRID_START=4, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=4

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 10 / 10 | CT[3] | 19 | 51 |
| active_erisc | E | 10 / 10 | CT[5] | 19 | 51 |
| active_erisc | E | 10 / 10 | CT[6] | 175424 | 317760 |
| active_erisc | E | 10 / 10 | CT[7] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[8] | 175424 | 317760 |
| active_erisc | E | 10 / 10 | CT[9] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[10] | 215456 | 335552 |
| active_erisc | E | 10 / 10 | CT[11] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[12] | 215456 | 335552 |
| active_erisc | E | 10 / 10 | CT[13] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[14] | 255488 | 353344 |
| active_erisc | E | 10 / 10 | CT[15] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[16] | 255488 | 353344 |
| active_erisc | E | 10 / 10 | CT[17] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[18] | 295520 | 371136 |
| active_erisc | E | 10 / 10 | CT[19] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[20] | 295520 | 371136 |
| active_erisc | E | 10 / 10 | CT[21] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[22] | 317760 | 380032 |
| active_erisc | E | 10 / 10 | CT[23] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[24] | 317760 | 380032 |
| active_erisc | E | 10 / 10 | CT[25] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[26] | 340000 | 388928 |
| active_erisc | E | 10 / 10 | CT[27] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[28] | 340000 | 388928 |
| active_erisc | E | 10 / 10 | CT[29] | 5 | 2 |
| active_erisc | E | 10 / 10 | CT[30] | 362240 | 397824 |
| active_erisc | E | 10 / 10 | CT[31] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[32] | 362240 | 397824 |
| active_erisc | E | 10 / 10 | CT[33] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[50] | 362240 | 397824 |
| active_erisc | E | 10 / 10 | CT[51] | 16 | 8 |
| active_erisc | E | 10 / 10 | CT[61] | 9 | 4 |
| active_erisc | E | 10 / 10 | CT[62] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[3] | 19 | 51 |
| active_erisc | N | 12 / 12 | CT[5] | 19 | 51 |
| active_erisc | N | 12 / 12 | CT[6] | 175424 | 317760 |
| active_erisc | N | 12 / 12 | CT[7] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[8] | 175424 | 317760 |
| active_erisc | N | 12 / 12 | CT[9] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[10] | 215456 | 335552 |
| active_erisc | N | 12 / 12 | CT[11] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[12] | 215456 | 335552 |
| active_erisc | N | 12 / 12 | CT[13] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[14] | 255488 | 353344 |
| active_erisc | N | 12 / 12 | CT[15] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[16] | 255488 | 353344 |
| active_erisc | N | 12 / 12 | CT[17] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[18] | 295520 | 371136 |
| active_erisc | N | 12 / 12 | CT[19] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[20] | 295520 | 371136 |
| active_erisc | N | 12 / 12 | CT[21] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[22] | 317760 | 380032 |
| active_erisc | N | 12 / 12 | CT[23] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[24] | 317760 | 380032 |
| active_erisc | N | 12 / 12 | CT[25] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[26] | 340000 | 388928 |
| active_erisc | N | 12 / 12 | CT[27] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[28] | 340000 | 388928 |
| active_erisc | N | 12 / 12 | CT[29] | 5 | 2 |
| active_erisc | N | 12 / 12 | CT[30] | 362240 | 397824 |
| active_erisc | N | 12 / 12 | CT[31] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[32] | 362240 | 397824 |
| active_erisc | N | 12 / 12 | CT[33] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[50] | 362240 | 397824 |
| active_erisc | N | 12 / 12 | CT[51] | 16 | 8 |
| active_erisc | N | 12 / 12 | CT[61] | 9 | 4 |
| active_erisc | N | 12 / 12 | CT[62] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[3] | 19 | 51 |
| active_erisc | S | 10 / 10 | CT[5] | 19 | 51 |
| active_erisc | S | 10 / 10 | CT[6] | 175424 | 317760 |
| active_erisc | S | 10 / 10 | CT[7] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[8] | 175424 | 317760 |
| active_erisc | S | 10 / 10 | CT[9] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[10] | 215456 | 335552 |
| active_erisc | S | 10 / 10 | CT[11] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[12] | 215456 | 335552 |
| active_erisc | S | 10 / 10 | CT[13] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[14] | 255488 | 353344 |
| active_erisc | S | 10 / 10 | CT[15] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[16] | 255488 | 353344 |
| active_erisc | S | 10 / 10 | CT[17] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[18] | 295520 | 371136 |
| active_erisc | S | 10 / 10 | CT[19] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[20] | 295520 | 371136 |
| active_erisc | S | 10 / 10 | CT[21] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[22] | 317760 | 380032 |
| active_erisc | S | 10 / 10 | CT[23] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[24] | 317760 | 380032 |
| active_erisc | S | 10 / 10 | CT[25] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[26] | 340000 | 388928 |
| active_erisc | S | 10 / 10 | CT[27] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[28] | 340000 | 388928 |
| active_erisc | S | 10 / 10 | CT[29] | 5 | 2 |
| active_erisc | S | 10 / 10 | CT[30] | 362240 | 397824 |
| active_erisc | S | 10 / 10 | CT[31] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[32] | 362240 | 397824 |
| active_erisc | S | 10 / 10 | CT[33] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[50] | 362240 | 397824 |
| active_erisc | S | 10 / 10 | CT[51] | 16 | 8 |
| active_erisc | S | 10 / 10 | CT[61] | 9 | 4 |
| active_erisc | S | 10 / 10 | CT[62] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[3] | 19 | 51 |
| active_erisc | W | 8 / 8 | CT[5] | 19 | 51 |
| active_erisc | W | 8 / 8 | CT[6] | 175424 | 317760 |
| active_erisc | W | 8 / 8 | CT[7] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[8] | 175424 | 317760 |
| active_erisc | W | 8 / 8 | CT[9] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[10] | 215456 | 335552 |
| active_erisc | W | 8 / 8 | CT[11] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[12] | 215456 | 335552 |
| active_erisc | W | 8 / 8 | CT[13] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[14] | 255488 | 353344 |
| active_erisc | W | 8 / 8 | CT[15] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[16] | 255488 | 353344 |
| active_erisc | W | 8 / 8 | CT[17] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[18] | 295520 | 371136 |
| active_erisc | W | 8 / 8 | CT[19] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[20] | 295520 | 371136 |
| active_erisc | W | 8 / 8 | CT[21] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[22] | 317760 | 380032 |
| active_erisc | W | 8 / 8 | CT[23] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[24] | 317760 | 380032 |
| active_erisc | W | 8 / 8 | CT[25] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[26] | 340000 | 388928 |
| active_erisc | W | 8 / 8 | CT[27] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[28] | 340000 | 388928 |
| active_erisc | W | 8 / 8 | CT[29] | 5 | 2 |
| active_erisc | W | 8 / 8 | CT[30] | 362240 | 397824 |
| active_erisc | W | 8 / 8 | CT[31] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[32] | 362240 | 397824 |
| active_erisc | W | 8 / 8 | CT[33] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[50] | 362240 | 397824 |
| active_erisc | W | 8 / 8 | CT[51] | 16 | 8 |
| active_erisc | W | 8 / 8 | CT[61] | 9 | 4 |
| active_erisc | W | 8 / 8 | CT[62] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[3] | 19 | 51 |
| subordinate_active_erisc | E | 10 / 10 | CT[5] | 19 | 51 |
| subordinate_active_erisc | E | 10 / 10 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | E | 10 / 10 | CT[7] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | E | 10 / 10 | CT[9] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | E | 10 / 10 | CT[11] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | E | 10 / 10 | CT[13] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | E | 10 / 10 | CT[15] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | E | 10 / 10 | CT[17] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | E | 10 / 10 | CT[19] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | E | 10 / 10 | CT[21] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | E | 10 / 10 | CT[23] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | E | 10 / 10 | CT[25] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | E | 10 / 10 | CT[27] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | E | 10 / 10 | CT[29] | 5 | 2 |
| subordinate_active_erisc | E | 10 / 10 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | E | 10 / 10 | CT[31] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | E | 10 / 10 | CT[33] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | E | 10 / 10 | CT[51] | 16 | 8 |
| subordinate_active_erisc | E | 10 / 10 | CT[61] | 9 | 4 |
| subordinate_active_erisc | E | 10 / 10 | CT[62] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[3] | 19 | 51 |
| subordinate_active_erisc | N | 12 / 12 | CT[5] | 19 | 51 |
| subordinate_active_erisc | N | 12 / 12 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | N | 12 / 12 | CT[7] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | N | 12 / 12 | CT[9] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | N | 12 / 12 | CT[11] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | N | 12 / 12 | CT[13] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | N | 12 / 12 | CT[15] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | N | 12 / 12 | CT[17] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | N | 12 / 12 | CT[19] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | N | 12 / 12 | CT[21] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | N | 12 / 12 | CT[23] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | N | 12 / 12 | CT[25] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | N | 12 / 12 | CT[27] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | N | 12 / 12 | CT[29] | 5 | 2 |
| subordinate_active_erisc | N | 12 / 12 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | N | 12 / 12 | CT[31] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | N | 12 / 12 | CT[33] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | N | 12 / 12 | CT[51] | 16 | 8 |
| subordinate_active_erisc | N | 12 / 12 | CT[61] | 9 | 4 |
| subordinate_active_erisc | N | 12 / 12 | CT[62] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[3] | 19 | 51 |
| subordinate_active_erisc | S | 10 / 10 | CT[5] | 19 | 51 |
| subordinate_active_erisc | S | 10 / 10 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | S | 10 / 10 | CT[7] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | S | 10 / 10 | CT[9] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | S | 10 / 10 | CT[11] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | S | 10 / 10 | CT[13] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | S | 10 / 10 | CT[15] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | S | 10 / 10 | CT[17] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | S | 10 / 10 | CT[19] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | S | 10 / 10 | CT[21] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | S | 10 / 10 | CT[23] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | S | 10 / 10 | CT[25] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | S | 10 / 10 | CT[27] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | S | 10 / 10 | CT[29] | 5 | 2 |
| subordinate_active_erisc | S | 10 / 10 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | S | 10 / 10 | CT[31] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | S | 10 / 10 | CT[33] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | S | 10 / 10 | CT[51] | 16 | 8 |
| subordinate_active_erisc | S | 10 / 10 | CT[61] | 9 | 4 |
| subordinate_active_erisc | S | 10 / 10 | CT[62] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[3] | 19 | 51 |
| subordinate_active_erisc | W | 8 / 8 | CT[5] | 19 | 51 |
| subordinate_active_erisc | W | 8 / 8 | CT[6] | 175424 | 317760 |
| subordinate_active_erisc | W | 8 / 8 | CT[7] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[8] | 175424 | 317760 |
| subordinate_active_erisc | W | 8 / 8 | CT[9] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[10] | 215456 | 335552 |
| subordinate_active_erisc | W | 8 / 8 | CT[11] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[12] | 215456 | 335552 |
| subordinate_active_erisc | W | 8 / 8 | CT[13] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[14] | 255488 | 353344 |
| subordinate_active_erisc | W | 8 / 8 | CT[15] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[16] | 255488 | 353344 |
| subordinate_active_erisc | W | 8 / 8 | CT[17] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[18] | 295520 | 371136 |
| subordinate_active_erisc | W | 8 / 8 | CT[19] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[20] | 295520 | 371136 |
| subordinate_active_erisc | W | 8 / 8 | CT[21] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[22] | 317760 | 380032 |
| subordinate_active_erisc | W | 8 / 8 | CT[23] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[24] | 317760 | 380032 |
| subordinate_active_erisc | W | 8 / 8 | CT[25] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[26] | 340000 | 388928 |
| subordinate_active_erisc | W | 8 / 8 | CT[27] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[28] | 340000 | 388928 |
| subordinate_active_erisc | W | 8 / 8 | CT[29] | 5 | 2 |
| subordinate_active_erisc | W | 8 / 8 | CT[30] | 362240 | 397824 |
| subordinate_active_erisc | W | 8 / 8 | CT[31] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[32] | 362240 | 397824 |
| subordinate_active_erisc | W | 8 / 8 | CT[33] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[50] | 362240 | 397824 |
| subordinate_active_erisc | W | 8 / 8 | CT[51] | 16 | 8 |
| subordinate_active_erisc | W | 8 / 8 | CT[61] | 9 | 4 |
| subordinate_active_erisc | W | 8 / 8 | CT[62] | 5 | 2 |

## glx_vc2_3072: 168 study router builds, 168 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {51, 14, 14, 14} receiver slots {16} (channel buffer 3168 B) x1; VC2 sender slots {2} receiver slots {4} (channel buffer 3168 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {77, 7, 7, 7} receiver slots {11} (channel buffer 3168 B) x1; VC2 sender slots {2} receiver slots {4} (channel buffer 3168 B) x1

study: opt ['-O3'], positional CT args [55]
port: opt ['-O3'], positional CT args [55]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0,1, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=4, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=4, RX_CH_1_TRID_START=4, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=4

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 24 / 24 | CT[3] | 51 | 77 |
| active_erisc | E | 24 / 24 | CT[5] | 51 | 77 |
| active_erisc | E | 24 / 24 | CT[6] | 252480 | 334848 |
| active_erisc | E | 24 / 24 | CT[7] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[8] | 252480 | 334848 |
| active_erisc | E | 24 / 24 | CT[9] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[10] | 296832 | 357024 |
| active_erisc | E | 24 / 24 | CT[11] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[12] | 296832 | 357024 |
| active_erisc | E | 24 / 24 | CT[13] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[14] | 341184 | 379200 |
| active_erisc | E | 24 / 24 | CT[15] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[16] | 341184 | 379200 |
| active_erisc | E | 24 / 24 | CT[17] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[18] | 385536 | 401376 |
| active_erisc | E | 24 / 24 | CT[20] | 385536 | 401376 |
| active_erisc | E | 24 / 24 | CT[22] | 391872 | 407712 |
| active_erisc | E | 24 / 24 | CT[23] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[24] | 391872 | 407712 |
| active_erisc | E | 24 / 24 | CT[25] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[40] | 391872 | 407712 |
| active_erisc | E | 24 / 24 | CT[41] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[51] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[3] | 51 | 77 |
| active_erisc | N | 18 / 18 | CT[5] | 51 | 77 |
| active_erisc | N | 18 / 18 | CT[6] | 252480 | 334848 |
| active_erisc | N | 18 / 18 | CT[7] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[8] | 252480 | 334848 |
| active_erisc | N | 18 / 18 | CT[9] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[10] | 296832 | 357024 |
| active_erisc | N | 18 / 18 | CT[11] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[12] | 296832 | 357024 |
| active_erisc | N | 18 / 18 | CT[13] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[14] | 341184 | 379200 |
| active_erisc | N | 18 / 18 | CT[15] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[16] | 341184 | 379200 |
| active_erisc | N | 18 / 18 | CT[17] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[18] | 385536 | 401376 |
| active_erisc | N | 18 / 18 | CT[20] | 385536 | 401376 |
| active_erisc | N | 18 / 18 | CT[22] | 391872 | 407712 |
| active_erisc | N | 18 / 18 | CT[23] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[24] | 391872 | 407712 |
| active_erisc | N | 18 / 18 | CT[25] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[40] | 391872 | 407712 |
| active_erisc | N | 18 / 18 | CT[41] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[51] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[3] | 51 | 77 |
| active_erisc | S | 18 / 18 | CT[5] | 51 | 77 |
| active_erisc | S | 18 / 18 | CT[6] | 252480 | 334848 |
| active_erisc | S | 18 / 18 | CT[7] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[8] | 252480 | 334848 |
| active_erisc | S | 18 / 18 | CT[9] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[10] | 296832 | 357024 |
| active_erisc | S | 18 / 18 | CT[11] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[12] | 296832 | 357024 |
| active_erisc | S | 18 / 18 | CT[13] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[14] | 341184 | 379200 |
| active_erisc | S | 18 / 18 | CT[15] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[16] | 341184 | 379200 |
| active_erisc | S | 18 / 18 | CT[17] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[18] | 385536 | 401376 |
| active_erisc | S | 18 / 18 | CT[20] | 385536 | 401376 |
| active_erisc | S | 18 / 18 | CT[22] | 391872 | 407712 |
| active_erisc | S | 18 / 18 | CT[23] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[24] | 391872 | 407712 |
| active_erisc | S | 18 / 18 | CT[25] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[40] | 391872 | 407712 |
| active_erisc | S | 18 / 18 | CT[41] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[51] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[3] | 51 | 77 |
| active_erisc | W | 24 / 24 | CT[5] | 51 | 77 |
| active_erisc | W | 24 / 24 | CT[6] | 252480 | 334848 |
| active_erisc | W | 24 / 24 | CT[7] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[8] | 252480 | 334848 |
| active_erisc | W | 24 / 24 | CT[9] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[10] | 296832 | 357024 |
| active_erisc | W | 24 / 24 | CT[11] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[12] | 296832 | 357024 |
| active_erisc | W | 24 / 24 | CT[13] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[14] | 341184 | 379200 |
| active_erisc | W | 24 / 24 | CT[15] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[16] | 341184 | 379200 |
| active_erisc | W | 24 / 24 | CT[17] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[18] | 385536 | 401376 |
| active_erisc | W | 24 / 24 | CT[20] | 385536 | 401376 |
| active_erisc | W | 24 / 24 | CT[22] | 391872 | 407712 |
| active_erisc | W | 24 / 24 | CT[23] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[24] | 391872 | 407712 |
| active_erisc | W | 24 / 24 | CT[25] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[40] | 391872 | 407712 |
| active_erisc | W | 24 / 24 | CT[41] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[51] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[3] | 51 | 77 |
| subordinate_active_erisc | E | 24 / 24 | CT[5] | 51 | 77 |
| subordinate_active_erisc | E | 24 / 24 | CT[6] | 252480 | 334848 |
| subordinate_active_erisc | E | 24 / 24 | CT[7] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[8] | 252480 | 334848 |
| subordinate_active_erisc | E | 24 / 24 | CT[9] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[10] | 296832 | 357024 |
| subordinate_active_erisc | E | 24 / 24 | CT[11] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[12] | 296832 | 357024 |
| subordinate_active_erisc | E | 24 / 24 | CT[13] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[14] | 341184 | 379200 |
| subordinate_active_erisc | E | 24 / 24 | CT[15] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[16] | 341184 | 379200 |
| subordinate_active_erisc | E | 24 / 24 | CT[17] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[18] | 385536 | 401376 |
| subordinate_active_erisc | E | 24 / 24 | CT[20] | 385536 | 401376 |
| subordinate_active_erisc | E | 24 / 24 | CT[22] | 391872 | 407712 |
| subordinate_active_erisc | E | 24 / 24 | CT[23] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[24] | 391872 | 407712 |
| subordinate_active_erisc | E | 24 / 24 | CT[25] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[40] | 391872 | 407712 |
| subordinate_active_erisc | E | 24 / 24 | CT[41] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[51] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[3] | 51 | 77 |
| subordinate_active_erisc | N | 18 / 18 | CT[5] | 51 | 77 |
| subordinate_active_erisc | N | 18 / 18 | CT[6] | 252480 | 334848 |
| subordinate_active_erisc | N | 18 / 18 | CT[7] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[8] | 252480 | 334848 |
| subordinate_active_erisc | N | 18 / 18 | CT[9] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[10] | 296832 | 357024 |
| subordinate_active_erisc | N | 18 / 18 | CT[11] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[12] | 296832 | 357024 |
| subordinate_active_erisc | N | 18 / 18 | CT[13] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[14] | 341184 | 379200 |
| subordinate_active_erisc | N | 18 / 18 | CT[15] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[16] | 341184 | 379200 |
| subordinate_active_erisc | N | 18 / 18 | CT[17] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[18] | 385536 | 401376 |
| subordinate_active_erisc | N | 18 / 18 | CT[20] | 385536 | 401376 |
| subordinate_active_erisc | N | 18 / 18 | CT[22] | 391872 | 407712 |
| subordinate_active_erisc | N | 18 / 18 | CT[23] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[24] | 391872 | 407712 |
| subordinate_active_erisc | N | 18 / 18 | CT[25] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[40] | 391872 | 407712 |
| subordinate_active_erisc | N | 18 / 18 | CT[41] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[51] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[3] | 51 | 77 |
| subordinate_active_erisc | S | 18 / 18 | CT[5] | 51 | 77 |
| subordinate_active_erisc | S | 18 / 18 | CT[6] | 252480 | 334848 |
| subordinate_active_erisc | S | 18 / 18 | CT[7] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[8] | 252480 | 334848 |
| subordinate_active_erisc | S | 18 / 18 | CT[9] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[10] | 296832 | 357024 |
| subordinate_active_erisc | S | 18 / 18 | CT[11] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[12] | 296832 | 357024 |
| subordinate_active_erisc | S | 18 / 18 | CT[13] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[14] | 341184 | 379200 |
| subordinate_active_erisc | S | 18 / 18 | CT[15] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[16] | 341184 | 379200 |
| subordinate_active_erisc | S | 18 / 18 | CT[17] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[18] | 385536 | 401376 |
| subordinate_active_erisc | S | 18 / 18 | CT[20] | 385536 | 401376 |
| subordinate_active_erisc | S | 18 / 18 | CT[22] | 391872 | 407712 |
| subordinate_active_erisc | S | 18 / 18 | CT[23] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[24] | 391872 | 407712 |
| subordinate_active_erisc | S | 18 / 18 | CT[25] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[40] | 391872 | 407712 |
| subordinate_active_erisc | S | 18 / 18 | CT[41] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[51] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[3] | 51 | 77 |
| subordinate_active_erisc | W | 24 / 24 | CT[5] | 51 | 77 |
| subordinate_active_erisc | W | 24 / 24 | CT[6] | 252480 | 334848 |
| subordinate_active_erisc | W | 24 / 24 | CT[7] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[8] | 252480 | 334848 |
| subordinate_active_erisc | W | 24 / 24 | CT[9] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[10] | 296832 | 357024 |
| subordinate_active_erisc | W | 24 / 24 | CT[11] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[12] | 296832 | 357024 |
| subordinate_active_erisc | W | 24 / 24 | CT[13] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[14] | 341184 | 379200 |
| subordinate_active_erisc | W | 24 / 24 | CT[15] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[16] | 341184 | 379200 |
| subordinate_active_erisc | W | 24 / 24 | CT[17] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[18] | 385536 | 401376 |
| subordinate_active_erisc | W | 24 / 24 | CT[20] | 385536 | 401376 |
| subordinate_active_erisc | W | 24 / 24 | CT[22] | 391872 | 407712 |
| subordinate_active_erisc | W | 24 / 24 | CT[23] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[24] | 391872 | 407712 |
| subordinate_active_erisc | W | 24 / 24 | CT[25] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[40] | 391872 | 407712 |
| subordinate_active_erisc | W | 24 / 24 | CT[41] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[51] | 14 | 7 |

## glx_vc2_4352: 168 study router builds, 168 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 14, 14, 14} receiver slots {16} (channel buffer 4448 B) x1; VC2 sender slots {2} receiver slots {4} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {43, 7, 7, 7} receiver slots {11} (channel buffer 4448 B) x1; VC2 sender slots {2} receiver slots {4} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [55]
port: opt ['-O3'], positional CT args [55]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=20,21,22,23,24,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0,1, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=4, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=4, RX_CH_1_TRID_START=4, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=4

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 24 / 24 | CT[3] | 17 | 43 |
| active_erisc | E | 24 / 24 | CT[5] | 17 | 43 |
| active_erisc | E | 24 / 24 | CT[6] | 166528 | 282176 |
| active_erisc | E | 24 / 24 | CT[7] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[8] | 166528 | 282176 |
| active_erisc | E | 24 / 24 | CT[9] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[10] | 228800 | 313312 |
| active_erisc | E | 24 / 24 | CT[11] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[12] | 228800 | 313312 |
| active_erisc | E | 24 / 24 | CT[13] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[14] | 291072 | 344448 |
| active_erisc | E | 24 / 24 | CT[15] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[16] | 291072 | 344448 |
| active_erisc | E | 24 / 24 | CT[17] | 14 | 7 |
| active_erisc | E | 24 / 24 | CT[18] | 353344 | 375584 |
| active_erisc | E | 24 / 24 | CT[20] | 353344 | 375584 |
| active_erisc | E | 24 / 24 | CT[22] | 362240 | 384480 |
| active_erisc | E | 24 / 24 | CT[23] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[24] | 362240 | 384480 |
| active_erisc | E | 24 / 24 | CT[25] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[40] | 362240 | 384480 |
| active_erisc | E | 24 / 24 | CT[41] | 16 | 11 |
| active_erisc | E | 24 / 24 | CT[51] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[3] | 17 | 43 |
| active_erisc | N | 18 / 18 | CT[5] | 17 | 43 |
| active_erisc | N | 18 / 18 | CT[6] | 166528 | 282176 |
| active_erisc | N | 18 / 18 | CT[7] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[8] | 166528 | 282176 |
| active_erisc | N | 18 / 18 | CT[9] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[10] | 228800 | 313312 |
| active_erisc | N | 18 / 18 | CT[11] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[12] | 228800 | 313312 |
| active_erisc | N | 18 / 18 | CT[13] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[14] | 291072 | 344448 |
| active_erisc | N | 18 / 18 | CT[15] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[16] | 291072 | 344448 |
| active_erisc | N | 18 / 18 | CT[17] | 14 | 7 |
| active_erisc | N | 18 / 18 | CT[18] | 353344 | 375584 |
| active_erisc | N | 18 / 18 | CT[20] | 353344 | 375584 |
| active_erisc | N | 18 / 18 | CT[22] | 362240 | 384480 |
| active_erisc | N | 18 / 18 | CT[23] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[24] | 362240 | 384480 |
| active_erisc | N | 18 / 18 | CT[25] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[40] | 362240 | 384480 |
| active_erisc | N | 18 / 18 | CT[41] | 16 | 11 |
| active_erisc | N | 18 / 18 | CT[51] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[3] | 17 | 43 |
| active_erisc | S | 18 / 18 | CT[5] | 17 | 43 |
| active_erisc | S | 18 / 18 | CT[6] | 166528 | 282176 |
| active_erisc | S | 18 / 18 | CT[7] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[8] | 166528 | 282176 |
| active_erisc | S | 18 / 18 | CT[9] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[10] | 228800 | 313312 |
| active_erisc | S | 18 / 18 | CT[11] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[12] | 228800 | 313312 |
| active_erisc | S | 18 / 18 | CT[13] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[14] | 291072 | 344448 |
| active_erisc | S | 18 / 18 | CT[15] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[16] | 291072 | 344448 |
| active_erisc | S | 18 / 18 | CT[17] | 14 | 7 |
| active_erisc | S | 18 / 18 | CT[18] | 353344 | 375584 |
| active_erisc | S | 18 / 18 | CT[20] | 353344 | 375584 |
| active_erisc | S | 18 / 18 | CT[22] | 362240 | 384480 |
| active_erisc | S | 18 / 18 | CT[23] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[24] | 362240 | 384480 |
| active_erisc | S | 18 / 18 | CT[25] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[40] | 362240 | 384480 |
| active_erisc | S | 18 / 18 | CT[41] | 16 | 11 |
| active_erisc | S | 18 / 18 | CT[51] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[3] | 17 | 43 |
| active_erisc | W | 24 / 24 | CT[5] | 17 | 43 |
| active_erisc | W | 24 / 24 | CT[6] | 166528 | 282176 |
| active_erisc | W | 24 / 24 | CT[7] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[8] | 166528 | 282176 |
| active_erisc | W | 24 / 24 | CT[9] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[10] | 228800 | 313312 |
| active_erisc | W | 24 / 24 | CT[11] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[12] | 228800 | 313312 |
| active_erisc | W | 24 / 24 | CT[13] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[14] | 291072 | 344448 |
| active_erisc | W | 24 / 24 | CT[15] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[16] | 291072 | 344448 |
| active_erisc | W | 24 / 24 | CT[17] | 14 | 7 |
| active_erisc | W | 24 / 24 | CT[18] | 353344 | 375584 |
| active_erisc | W | 24 / 24 | CT[20] | 353344 | 375584 |
| active_erisc | W | 24 / 24 | CT[22] | 362240 | 384480 |
| active_erisc | W | 24 / 24 | CT[23] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[24] | 362240 | 384480 |
| active_erisc | W | 24 / 24 | CT[25] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[40] | 362240 | 384480 |
| active_erisc | W | 24 / 24 | CT[41] | 16 | 11 |
| active_erisc | W | 24 / 24 | CT[51] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[3] | 17 | 43 |
| subordinate_active_erisc | E | 24 / 24 | CT[5] | 17 | 43 |
| subordinate_active_erisc | E | 24 / 24 | CT[6] | 166528 | 282176 |
| subordinate_active_erisc | E | 24 / 24 | CT[7] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[8] | 166528 | 282176 |
| subordinate_active_erisc | E | 24 / 24 | CT[9] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[10] | 228800 | 313312 |
| subordinate_active_erisc | E | 24 / 24 | CT[11] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[12] | 228800 | 313312 |
| subordinate_active_erisc | E | 24 / 24 | CT[13] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[14] | 291072 | 344448 |
| subordinate_active_erisc | E | 24 / 24 | CT[15] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[16] | 291072 | 344448 |
| subordinate_active_erisc | E | 24 / 24 | CT[17] | 14 | 7 |
| subordinate_active_erisc | E | 24 / 24 | CT[18] | 353344 | 375584 |
| subordinate_active_erisc | E | 24 / 24 | CT[20] | 353344 | 375584 |
| subordinate_active_erisc | E | 24 / 24 | CT[22] | 362240 | 384480 |
| subordinate_active_erisc | E | 24 / 24 | CT[23] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[24] | 362240 | 384480 |
| subordinate_active_erisc | E | 24 / 24 | CT[25] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[40] | 362240 | 384480 |
| subordinate_active_erisc | E | 24 / 24 | CT[41] | 16 | 11 |
| subordinate_active_erisc | E | 24 / 24 | CT[51] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[3] | 17 | 43 |
| subordinate_active_erisc | N | 18 / 18 | CT[5] | 17 | 43 |
| subordinate_active_erisc | N | 18 / 18 | CT[6] | 166528 | 282176 |
| subordinate_active_erisc | N | 18 / 18 | CT[7] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[8] | 166528 | 282176 |
| subordinate_active_erisc | N | 18 / 18 | CT[9] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[10] | 228800 | 313312 |
| subordinate_active_erisc | N | 18 / 18 | CT[11] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[12] | 228800 | 313312 |
| subordinate_active_erisc | N | 18 / 18 | CT[13] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[14] | 291072 | 344448 |
| subordinate_active_erisc | N | 18 / 18 | CT[15] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[16] | 291072 | 344448 |
| subordinate_active_erisc | N | 18 / 18 | CT[17] | 14 | 7 |
| subordinate_active_erisc | N | 18 / 18 | CT[18] | 353344 | 375584 |
| subordinate_active_erisc | N | 18 / 18 | CT[20] | 353344 | 375584 |
| subordinate_active_erisc | N | 18 / 18 | CT[22] | 362240 | 384480 |
| subordinate_active_erisc | N | 18 / 18 | CT[23] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[24] | 362240 | 384480 |
| subordinate_active_erisc | N | 18 / 18 | CT[25] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[40] | 362240 | 384480 |
| subordinate_active_erisc | N | 18 / 18 | CT[41] | 16 | 11 |
| subordinate_active_erisc | N | 18 / 18 | CT[51] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[3] | 17 | 43 |
| subordinate_active_erisc | S | 18 / 18 | CT[5] | 17 | 43 |
| subordinate_active_erisc | S | 18 / 18 | CT[6] | 166528 | 282176 |
| subordinate_active_erisc | S | 18 / 18 | CT[7] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[8] | 166528 | 282176 |
| subordinate_active_erisc | S | 18 / 18 | CT[9] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[10] | 228800 | 313312 |
| subordinate_active_erisc | S | 18 / 18 | CT[11] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[12] | 228800 | 313312 |
| subordinate_active_erisc | S | 18 / 18 | CT[13] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[14] | 291072 | 344448 |
| subordinate_active_erisc | S | 18 / 18 | CT[15] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[16] | 291072 | 344448 |
| subordinate_active_erisc | S | 18 / 18 | CT[17] | 14 | 7 |
| subordinate_active_erisc | S | 18 / 18 | CT[18] | 353344 | 375584 |
| subordinate_active_erisc | S | 18 / 18 | CT[20] | 353344 | 375584 |
| subordinate_active_erisc | S | 18 / 18 | CT[22] | 362240 | 384480 |
| subordinate_active_erisc | S | 18 / 18 | CT[23] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[24] | 362240 | 384480 |
| subordinate_active_erisc | S | 18 / 18 | CT[25] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[40] | 362240 | 384480 |
| subordinate_active_erisc | S | 18 / 18 | CT[41] | 16 | 11 |
| subordinate_active_erisc | S | 18 / 18 | CT[51] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[3] | 17 | 43 |
| subordinate_active_erisc | W | 24 / 24 | CT[5] | 17 | 43 |
| subordinate_active_erisc | W | 24 / 24 | CT[6] | 166528 | 282176 |
| subordinate_active_erisc | W | 24 / 24 | CT[7] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[8] | 166528 | 282176 |
| subordinate_active_erisc | W | 24 / 24 | CT[9] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[10] | 228800 | 313312 |
| subordinate_active_erisc | W | 24 / 24 | CT[11] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[12] | 228800 | 313312 |
| subordinate_active_erisc | W | 24 / 24 | CT[13] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[14] | 291072 | 344448 |
| subordinate_active_erisc | W | 24 / 24 | CT[15] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[16] | 291072 | 344448 |
| subordinate_active_erisc | W | 24 / 24 | CT[17] | 14 | 7 |
| subordinate_active_erisc | W | 24 / 24 | CT[18] | 353344 | 375584 |
| subordinate_active_erisc | W | 24 / 24 | CT[20] | 353344 | 375584 |
| subordinate_active_erisc | W | 24 / 24 | CT[22] | 362240 | 384480 |
| subordinate_active_erisc | W | 24 / 24 | CT[23] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[24] | 362240 | 384480 |
| subordinate_active_erisc | W | 24 / 24 | CT[25] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[40] | 362240 | 384480 |
| subordinate_active_erisc | W | 24 / 24 | CT[41] | 16 | 11 |
| subordinate_active_erisc | W | 24 / 24 | CT[51] | 14 | 7 |

## p150_line_3072: 72 study router builds, 72 port

slot rows (count of allocator logs): study VC0 sender slots {32, 0} receiver slots {32} (channel buffer 3120 B) x5; VC0 sender slots {43, 42} receiver slots {32} (channel buffer 3120 B) x1

port VC0 sender slots {32, 0} receiver slots {32} (channel buffer 3120 B) x5; VC0 sender slots {32, 32} receiver slots {32} (channel buffer 3120 B) x1

study: opt ['-O3'], positional CT args [30]
port: opt ['-O3'], positional CT args [30]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0, PEER_SENDER_CHANNEL_3_SERVICED=0, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 12 / 12 | CT[3] | 43 | 32 |
| active_erisc | E | 12 / 12 | CT[5] | 43 | 32 |
| active_erisc | E | 12 / 12 | CT[6] | 223664 | 189344 |
| active_erisc | E | 12 / 12 | CT[7] | 42 | 32 |
| active_erisc | E | 12 / 12 | CT[8] | 223664 | 189344 |
| active_erisc | E | 12 / 12 | CT[9] | 42 | 32 |
| active_erisc | E | 12 / 12 | CT[10] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[12] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[20] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[26] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[3] | 43 | 32 |
| active_erisc | N | 6 / 6 | CT[5] | 43 | 32 |
| active_erisc | N | 6 / 6 | CT[6] | 223664 | 189344 |
| active_erisc | N | 6 / 6 | CT[7] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[8] | 223664 | 189344 |
| active_erisc | N | 6 / 6 | CT[9] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[10] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[12] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[20] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[26] | 0,42 | 0,32 |
| active_erisc | S | 6 / 6 | CT[3] | 43 | 32 |
| active_erisc | S | 6 / 6 | CT[5] | 43 | 32 |
| active_erisc | S | 6 / 6 | CT[6] | 223664 | 189344 |
| active_erisc | S | 6 / 6 | CT[7] | 42 | 32 |
| active_erisc | S | 6 / 6 | CT[8] | 223664 | 189344 |
| active_erisc | S | 6 / 6 | CT[9] | 42 | 32 |
| active_erisc | S | 6 / 6 | CT[10] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[12] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[20] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[26] | 0,42 | 0,32 |
| active_erisc | W | 12 / 12 | CT[3] | 43 | 32 |
| active_erisc | W | 12 / 12 | CT[5] | 43 | 32 |
| active_erisc | W | 12 / 12 | CT[6] | 223664 | 189344 |
| active_erisc | W | 12 / 12 | CT[7] | 42 | 32 |
| active_erisc | W | 12 / 12 | CT[8] | 223664 | 189344 |
| active_erisc | W | 12 / 12 | CT[9] | 42 | 32 |
| active_erisc | W | 12 / 12 | CT[10] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[12] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[20] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[26] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[3] | 43 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[5] | 43 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | E | 12 / 12 | CT[7] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | E | 12 / 12 | CT[9] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[26] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[3] | 43 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[5] | 43 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | N | 6 / 6 | CT[7] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | N | 6 / 6 | CT[9] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[26] | 0,42 | 0,32 |
| subordinate_active_erisc | S | 6 / 6 | CT[3] | 43 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[5] | 43 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | S | 6 / 6 | CT[7] | 42 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | S | 6 / 6 | CT[9] | 42 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[26] | 0,42 | 0,32 |
| subordinate_active_erisc | W | 12 / 12 | CT[3] | 43 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[5] | 43 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | W | 12 / 12 | CT[7] | 42 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | W | 12 / 12 | CT[9] | 42 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[26] | 42 | 32 |

## p150_line_4352: 72 study router builds, 72 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0} receiver slots {32} (channel buffer 4400 B) x5; VC0 sender slots {26, 25} receiver slots {32} (channel buffer 4400 B) x1

port VC0 sender slots {16, 0} receiver slots {32} (channel buffer 4400 B) x5; VC0 sender slots {16, 16} receiver slots {32} (channel buffer 4400 B) x1

study: opt ['-O3'], positional CT args [30]
port: opt ['-O3'], positional CT args [30]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0, PEER_SENDER_CHANNEL_3_SERVICED=0, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 12 / 12 | CT[3] | 26 | 16 |
| active_erisc | E | 12 / 12 | CT[5] | 26 | 16 |
| active_erisc | E | 12 / 12 | CT[6] | 203904 | 159904 |
| active_erisc | E | 12 / 12 | CT[7] | 25 | 16 |
| active_erisc | E | 12 / 12 | CT[8] | 203904 | 159904 |
| active_erisc | E | 12 / 12 | CT[9] | 25 | 16 |
| active_erisc | E | 12 / 12 | CT[10] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[12] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[20] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[26] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[3] | 26 | 16 |
| active_erisc | N | 6 / 6 | CT[5] | 26 | 16 |
| active_erisc | N | 6 / 6 | CT[6] | 203904 | 159904 |
| active_erisc | N | 6 / 6 | CT[7] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[8] | 203904 | 159904 |
| active_erisc | N | 6 / 6 | CT[9] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[10] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[12] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[20] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[26] | 0,25 | 0,16 |
| active_erisc | S | 6 / 6 | CT[3] | 26 | 16 |
| active_erisc | S | 6 / 6 | CT[5] | 26 | 16 |
| active_erisc | S | 6 / 6 | CT[6] | 203904 | 159904 |
| active_erisc | S | 6 / 6 | CT[7] | 25 | 16 |
| active_erisc | S | 6 / 6 | CT[8] | 203904 | 159904 |
| active_erisc | S | 6 / 6 | CT[9] | 25 | 16 |
| active_erisc | S | 6 / 6 | CT[10] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[12] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[20] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[26] | 0,25 | 0,16 |
| active_erisc | W | 12 / 12 | CT[3] | 26 | 16 |
| active_erisc | W | 12 / 12 | CT[5] | 26 | 16 |
| active_erisc | W | 12 / 12 | CT[6] | 203904 | 159904 |
| active_erisc | W | 12 / 12 | CT[7] | 25 | 16 |
| active_erisc | W | 12 / 12 | CT[8] | 203904 | 159904 |
| active_erisc | W | 12 / 12 | CT[9] | 25 | 16 |
| active_erisc | W | 12 / 12 | CT[10] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[12] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[20] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[26] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[3] | 26 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[5] | 26 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | E | 12 / 12 | CT[7] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | E | 12 / 12 | CT[9] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[26] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[3] | 26 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[5] | 26 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | N | 6 / 6 | CT[7] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | N | 6 / 6 | CT[9] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[26] | 0,25 | 0,16 |
| subordinate_active_erisc | S | 6 / 6 | CT[3] | 26 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[5] | 26 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | S | 6 / 6 | CT[7] | 25 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | S | 6 / 6 | CT[9] | 25 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[26] | 0,25 | 0,16 |
| subordinate_active_erisc | W | 12 / 12 | CT[3] | 26 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[5] | 26 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | W | 12 / 12 | CT[7] | 25 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | W | 12 / 12 | CT[9] | 25 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[26] | 25 | 16 |

## p150_mesh4x2_4352: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_mesh_3072: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_mesh_4352: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_ne_4352: 72 study router builds, 72 port

slot rows (count of allocator logs): study VC0 sender slots {32} receiver slots {32} (channel buffer 4400 B) x6

port VC0 sender slots {32} receiver slots {32} (channel buffer 4400 B) x6

study: opt ['-O3'], positional CT args [25]
port: opt ['-O3'], positional CT args [25]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0, PEER_SENDER_CHANNEL_2_SERVICED=0, PEER_SENDER_CHANNEL_3_SERVICED=0, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=4, RX_CH_0_NUM_TRIDS=16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (1): NUM_TRANSACTION_IDS=16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_ring_3072: 72 study router builds, 72 port

slot rows (count of allocator logs): study VC0 sender slots {32, 0} receiver slots {32} (channel buffer 3120 B) x5; VC0 sender slots {43, 42} receiver slots {32} (channel buffer 3120 B) x1

port VC0 sender slots {32, 0} receiver slots {32} (channel buffer 3120 B) x5; VC0 sender slots {32, 32} receiver slots {32} (channel buffer 3120 B) x1

study: opt ['-O3'], positional CT args [30]
port: opt ['-O3'], positional CT args [30]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0, PEER_SENDER_CHANNEL_3_SERVICED=0, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=16, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 12 / 12 | CT[3] | 43 | 32 |
| active_erisc | E | 12 / 12 | CT[5] | 43 | 32 |
| active_erisc | E | 12 / 12 | CT[6] | 223664 | 189344 |
| active_erisc | E | 12 / 12 | CT[7] | 42 | 32 |
| active_erisc | E | 12 / 12 | CT[8] | 223664 | 189344 |
| active_erisc | E | 12 / 12 | CT[9] | 42 | 32 |
| active_erisc | E | 12 / 12 | CT[10] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[12] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[20] | 354704 | 289184 |
| active_erisc | E | 12 / 12 | CT[26] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[3] | 43 | 32 |
| active_erisc | N | 6 / 6 | CT[5] | 43 | 32 |
| active_erisc | N | 6 / 6 | CT[6] | 223664 | 189344 |
| active_erisc | N | 6 / 6 | CT[7] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[8] | 223664 | 189344 |
| active_erisc | N | 6 / 6 | CT[9] | 42 | 32 |
| active_erisc | N | 6 / 6 | CT[10] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[12] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[20] | 354704 | 289184 |
| active_erisc | N | 6 / 6 | CT[26] | 0,42 | 0,32 |
| active_erisc | S | 6 / 6 | CT[3] | 43 | 32 |
| active_erisc | S | 6 / 6 | CT[5] | 43 | 32 |
| active_erisc | S | 6 / 6 | CT[6] | 223664 | 189344 |
| active_erisc | S | 6 / 6 | CT[7] | 42 | 32 |
| active_erisc | S | 6 / 6 | CT[8] | 223664 | 189344 |
| active_erisc | S | 6 / 6 | CT[9] | 42 | 32 |
| active_erisc | S | 6 / 6 | CT[10] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[12] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[20] | 354704 | 289184 |
| active_erisc | S | 6 / 6 | CT[26] | 0,42 | 0,32 |
| active_erisc | W | 12 / 12 | CT[3] | 43 | 32 |
| active_erisc | W | 12 / 12 | CT[5] | 43 | 32 |
| active_erisc | W | 12 / 12 | CT[6] | 223664 | 189344 |
| active_erisc | W | 12 / 12 | CT[7] | 42 | 32 |
| active_erisc | W | 12 / 12 | CT[8] | 223664 | 189344 |
| active_erisc | W | 12 / 12 | CT[9] | 42 | 32 |
| active_erisc | W | 12 / 12 | CT[10] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[12] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[20] | 354704 | 289184 |
| active_erisc | W | 12 / 12 | CT[26] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[3] | 43 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[5] | 43 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | E | 12 / 12 | CT[7] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | E | 12 / 12 | CT[9] | 42 | 32 |
| subordinate_active_erisc | E | 12 / 12 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | E | 12 / 12 | CT[26] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[3] | 43 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[5] | 43 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | N | 6 / 6 | CT[7] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | N | 6 / 6 | CT[9] | 42 | 32 |
| subordinate_active_erisc | N | 6 / 6 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | N | 6 / 6 | CT[26] | 0,42 | 0,32 |
| subordinate_active_erisc | S | 6 / 6 | CT[3] | 43 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[5] | 43 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | S | 6 / 6 | CT[7] | 42 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | S | 6 / 6 | CT[9] | 42 | 32 |
| subordinate_active_erisc | S | 6 / 6 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | S | 6 / 6 | CT[26] | 0,42 | 0,32 |
| subordinate_active_erisc | W | 12 / 12 | CT[3] | 43 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[5] | 43 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[6] | 223664 | 189344 |
| subordinate_active_erisc | W | 12 / 12 | CT[7] | 42 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[8] | 223664 | 189344 |
| subordinate_active_erisc | W | 12 / 12 | CT[9] | 42 | 32 |
| subordinate_active_erisc | W | 12 / 12 | CT[10] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[12] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[20] | 354704 | 289184 |
| subordinate_active_erisc | W | 12 / 12 | CT[26] | 42 | 32 |

## p150_ring_4352: 72 study router builds, 72 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0} receiver slots {32} (channel buffer 4400 B) x5; VC0 sender slots {26, 25} receiver slots {32} (channel buffer 4400 B) x1

port VC0 sender slots {16, 0} receiver slots {32} (channel buffer 4400 B) x5; VC0 sender slots {16, 16} receiver slots {32} (channel buffer 4400 B) x1

study: opt ['-O3'], positional CT args [30]
port: opt ['-O3'], positional CT args [30]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0, PEER_SENDER_CHANNEL_3_SERVICED=0, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=16, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |
| active_erisc | E | 12 / 12 | CT[3] | 26 | 16 |
| active_erisc | E | 12 / 12 | CT[5] | 26 | 16 |
| active_erisc | E | 12 / 12 | CT[6] | 203904 | 159904 |
| active_erisc | E | 12 / 12 | CT[7] | 25 | 16 |
| active_erisc | E | 12 / 12 | CT[8] | 203904 | 159904 |
| active_erisc | E | 12 / 12 | CT[9] | 25 | 16 |
| active_erisc | E | 12 / 12 | CT[10] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[12] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[20] | 313904 | 230304 |
| active_erisc | E | 12 / 12 | CT[26] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[3] | 26 | 16 |
| active_erisc | N | 6 / 6 | CT[5] | 26 | 16 |
| active_erisc | N | 6 / 6 | CT[6] | 203904 | 159904 |
| active_erisc | N | 6 / 6 | CT[7] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[8] | 203904 | 159904 |
| active_erisc | N | 6 / 6 | CT[9] | 25 | 16 |
| active_erisc | N | 6 / 6 | CT[10] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[12] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[20] | 313904 | 230304 |
| active_erisc | N | 6 / 6 | CT[26] | 0,25 | 0,16 |
| active_erisc | S | 6 / 6 | CT[3] | 26 | 16 |
| active_erisc | S | 6 / 6 | CT[5] | 26 | 16 |
| active_erisc | S | 6 / 6 | CT[6] | 203904 | 159904 |
| active_erisc | S | 6 / 6 | CT[7] | 25 | 16 |
| active_erisc | S | 6 / 6 | CT[8] | 203904 | 159904 |
| active_erisc | S | 6 / 6 | CT[9] | 25 | 16 |
| active_erisc | S | 6 / 6 | CT[10] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[12] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[20] | 313904 | 230304 |
| active_erisc | S | 6 / 6 | CT[26] | 0,25 | 0,16 |
| active_erisc | W | 12 / 12 | CT[3] | 26 | 16 |
| active_erisc | W | 12 / 12 | CT[5] | 26 | 16 |
| active_erisc | W | 12 / 12 | CT[6] | 203904 | 159904 |
| active_erisc | W | 12 / 12 | CT[7] | 25 | 16 |
| active_erisc | W | 12 / 12 | CT[8] | 203904 | 159904 |
| active_erisc | W | 12 / 12 | CT[9] | 25 | 16 |
| active_erisc | W | 12 / 12 | CT[10] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[12] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[20] | 313904 | 230304 |
| active_erisc | W | 12 / 12 | CT[26] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[3] | 26 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[5] | 26 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | E | 12 / 12 | CT[7] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | E | 12 / 12 | CT[9] | 25 | 16 |
| subordinate_active_erisc | E | 12 / 12 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | E | 12 / 12 | CT[26] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[3] | 26 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[5] | 26 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | N | 6 / 6 | CT[7] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | N | 6 / 6 | CT[9] | 25 | 16 |
| subordinate_active_erisc | N | 6 / 6 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | N | 6 / 6 | CT[26] | 0,25 | 0,16 |
| subordinate_active_erisc | S | 6 / 6 | CT[3] | 26 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[5] | 26 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | S | 6 / 6 | CT[7] | 25 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | S | 6 / 6 | CT[9] | 25 | 16 |
| subordinate_active_erisc | S | 6 / 6 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | S | 6 / 6 | CT[26] | 0,25 | 0,16 |
| subordinate_active_erisc | W | 12 / 12 | CT[3] | 26 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[5] | 26 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[6] | 203904 | 159904 |
| subordinate_active_erisc | W | 12 / 12 | CT[7] | 25 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[8] | 203904 | 159904 |
| subordinate_active_erisc | W | 12 / 12 | CT[9] | 25 | 16 |
| subordinate_active_erisc | W | 12 / 12 | CT[10] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[12] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[20] | 313904 | 230304 |
| subordinate_active_erisc | W | 12 / 12 | CT[26] | 25 | 16 |

## p150_torus2x4_3072: 96 study router builds, 96 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=16, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_torus2x4_4352: 96 study router builds, 96 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=2, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_torus4x2_3072: 96 study router builds, 96 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {32} (channel buffer 3168 B) x5; VC0 sender slots {35, 16, 16, 16} receiver slots {32} (channel buffer 3168 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=16, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_torus4x2_4352: 96 study router builds, 96 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1, RX_CH_0_NUM_TRIDS=8, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=2, NUM_TRANSACTION_IDS=8

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

## p150_torusy_4352: 80 study router builds, 80 port

slot rows (count of allocator logs): study VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

port VC0 sender slots {16, 0, 0, 0} receiver slots {16} (channel buffer 4448 B) x5; VC0 sender slots {17, 16, 16, 16} receiver slots {16} (channel buffer 4448 B) x1

study: opt ['-O3'], positional CT args [40]
port: opt ['-O3'], positional CT args [40]

defines only on the study side: none; only on the port side: ['-DL1_ECC_HALT']

named args only the study passes (37): EXP_DRISC_DELAY=0, EXP_FLAGS=13129088, EXP_PUSH_DRISC_L1_BASE=0, EXP_PUSH_DRISC_L1_SIZE=0, EXP_PUSH_DRISC_X=0, EXP_PUSH_DRISC_Y=0, EXP_RX_DELAY=0, EXP_SEQ_UPDATE_TIMEOUT=1000, EXP_STACK_PAINT=0, EXP_TX_DELAY=0, PEER_ERISC_NOC_X=24,25,26,27,28,.. (8), PEER_ERISC_NOC_Y=25, PEER_SENDER_CHANNEL_0_SERVICED=0,1, PEER_SENDER_CHANNEL_1_SERVICED=0,1, PEER_SENDER_CHANNEL_2_SERVICED=0,1, PEER_SENDER_CHANNEL_3_SERVICED=0,1, PEER_SENDER_CHANNEL_4_SERVICED=0, PEER_SENDER_CHANNEL_5_SERVICED=0, PEER_SENDER_CHANNEL_6_SERVICED=0, PEER_SENDER_CHANNEL_7_SERVICED=0, PEER_SENDER_CHANNEL_8_SERVICED=0, PEER_SENDER_CHANNEL_9_SERVICED=0, PHASE_PROFILE_BUFFER_ADDR=0, PHASE_PROFILE_DRISC_MODE=0, PHASE_PROFILE_ITER_TIMING=1, PHASE_PROFILE_MASK=4095, PHASE_PROFILE_MODE=0, PHASE_PROFILE_TXQ_COUNTERS=0, RX_CH_0_COMPLETION_GROUP_SIZE=1,4, RX_CH_0_NUM_TRIDS=8,16, RX_CH_0_TRID_START=0, RX_CH_1_COMPLETION_GROUP_SIZE=1, RX_CH_1_NUM_TRIDS=0, RX_CH_1_TRID_START=0, RX_CH_2_COMPLETION_GROUP_SIZE=1, RX_CH_2_NUM_TRIDS=0, RX_CH_2_TRID_START=0

named args only the port passes (2): BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=2, NUM_TRANSACTION_IDS=8,16

| processor | dir | builds (study / port) | named arg | study values | port values |
| --- | --- | --- | --- | --- | --- |

