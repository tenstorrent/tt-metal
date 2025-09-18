# State Preservation in APIs
NOC APIs use registers in the command buffers to issue instructions. Some of these registers preserve their value over multiple calls to NOC APIs, unless instructed by the API. Others are modified in each API call. Below is a table of registers that preserve their value over API calls (stateful = preserves value).

## Stateful Registers

| Registers                       | Wormhole B0                                           | Blackhole                                             |
| --------------------------------| ----------------------------------------------------- | ----------------------------------------------------- |
| `NOC_TARG_ADDR_LO`              | stateful                                              | stateful                                              |
| `NOC_TARG_ADDR_MID`             | stateful                                              | stateful                                              |
| `NOC_TARG_ADDR_HI`              | unused                                                | stateful                                              |
| `NOC_RET_ADDR_LO`               | stateful                                              | stateful                                              |
| `NOC_RET_ADDR_MID`              | stateful                                              | stateful                                              |
| `NOC_RET_ADDR_HI`               | unused                                                | stateful                                              |
| `NOC_PACKET_TAG`                | stateful                                              | stateful                                              |
| `NOC_CTRL`                      | stateful                                              | stateful                                              |
| `NOC_AT_LEN_BE`                 | stateful                                              | stateful                                              |
| `NOC_AT_LEN_BE_1`               | -                                                     | stateful                                              |
| `NOC_AT_DATA`                   | stateful                                              | stateful                                              |
| `NOC_BRCST_EXCLUDE`             | -                                                     | stateful                                              |
| `NOC_L1_ACC_AT_INSTRN`          | -                                                     | stateful                                              |
| `NOC_SEC_CTRL`                  | -                                                     | stateful                                              |
| `NOC_CMD_CTRL`                  | not stateful (control flag for all other registers)   | not stateful (control flag for all other registers)   |
| `NOC_NODE_ID`                   | not stateful (control flag)                           | not stateful (control flag)                           |
| `NOC_ENDPOINT_ID`               | not stateful (control flag)                           | not stateful (control flag)                           |
| `NUM_MEM_PARITY_ERR`            | stateful                                              | stateful                                              |
| `NUM_HEADER_1B_ERR`             | stateful                                              | stateful                                              |
| `NUM_HEADER_2B_ERR`             | stateful                                              | stateful                                              |
| `ECC_CTRL`                      | not stateful (control flag for errors)                | not stateful (control flag for errors)                |
| `NOC_CLEAR_OUTSTANDING_REQ_CNT` | not stateful (control flag)                           | not stateful (control flag)                           |
| `RISC_IF_STATUS_OFFSET`         | not stateful (control flag)                           | -                                                     |
| `CMD_BUF_AVAIL`                 | -                                                     | not stateful (control flag)                           |
| `CMD_BUF_OVFL`                  | -                                                     | not stateful (control flag)                           |
