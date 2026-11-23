# TT / TTI counts (static, per ELF)

Columns: text bytes = `.text` size; TTI = `tt*` lines in elfray `.text` output; TT elfray = stores elfray marks `tensix issue` (lower bound); TT linear = harness linear register tracking; TT all paths = stores whose address is in `0xFFE40000`-`0xFFE4FFFF` on every path (dataflow analysis); unresolved = stores whose address could not be settled.

## assert_on

`/proj_sw/user_dev/ctr-sbratic/tools/asserts_off/out/8d/assert_on`

Builds: original 10, HAL 10. Varying build parameters: BROADCAST_TYPE, EN_DEST_REUSE, REUSE_DEST_TYPE, is_fp32_dest_acc_en.

### Average per build

| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved | TTI + TT all paths |
|---|---|---|---|---|---|---|---|---|
| original | unpack | 3235.2 | 21 | 11 | 11 | 11 | 2 | 32 |
| original | math | 2146.4 | 36 | 8 | 8.8 | 12 | 2 | 48 |
| original | pack | 2682.4 | 30 | 14 | 20 | 20 | 2 | 50 |
| original | sum | 8064 | 87 | 33 | 39.8 | 43 | 6 | 130 |
| hal | unpack | 2020 | 67 | 29 | 33 | 34 | 2 | 101 |
| hal | math | 812.8 | 29.6 | 1 | 2.2 | 2.2 | 2 | 31.8 |
| hal | pack | 1401.6 | 55 | 19 | 23 | 23 | 2 | 78 |
| hal | sum | 4234.4 | 151.6 | 49 | 58.2 | 59.2 | 6 | 210.8 |

### Total over all builds

| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |
|---|---|---|---|---|---|---|---|
| original | unpack | 32352 | 210 | 110 | 110 | 110 | 20 |
| original | math | 21464 | 360 | 80 | 88 | 120 | 20 |
| original | pack | 26824 | 300 | 140 | 200 | 200 | 20 |
| original | sum | 80640 | 870 | 330 | 398 | 430 | 60 |
| hal | unpack | 20200 | 670 | 290 | 330 | 340 | 20 |
| hal | math | 8128 | 296 | 10 | 22 | 22 | 20 |
| hal | pack | 14016 | 550 | 190 | 230 | 230 | 20 |
| hal | sum | 42344 | 1516 | 490 | 582 | 592 | 60 |

### Per build

| Side | Build | Variant | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |
|---|---|---|---|---|---|---|---|---|---|
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | unpack | 3184 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | math | 2064 | 35 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | pack | 2680 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | unpack | 3112 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | math | 2076 | 38 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | unpack | 3444 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | math | 1660 | 24 | 8 | 9 | 9 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | pack | 2680 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | unpack | 3184 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | math | 2868 | 48 | 8 | 8 | 12 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | unpack | 3252 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | math | 2064 | 35 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | unpack | 3444 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | math | 1660 | 24 | 8 | 9 | 9 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | unpack | 3184 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | math | 2868 | 48 | 8 | 8 | 12 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | unpack | 3184 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | math | 2064 | 35 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | pack | 2684 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | unpack | 3112 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | math | 2076 | 38 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | pack | 2680 | 30 | 14 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | unpack | 3252 | 21 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | math | 2064 | 35 | 8 | 9 | 13 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | pack | 2680 | 30 | 14 | 20 | 20 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | unpack | 2016 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | math | 728 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | pack | 1380 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | unpack | 2008 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | math | 720 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | unpack | 2068 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | math | 728 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | pack | 1380 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | unpack | 1920 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | math | 724 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | unpack | 1928 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | math | 732 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | pack | 1380 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | unpack | 2008 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | math | 1128 | 51 | 1 | 3 | 3 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | unpack | 2096 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | math | 764 | 25 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | pack | 1380 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | unpack | 2088 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | math | 756 | 25 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | unpack | 2060 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | math | 720 | 24 | 1 | 2 | 2 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | unpack | 2008 | 67 | 29 | 33 | 34 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | math | 1128 | 51 | 1 | 3 | 3 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | pack | 1416 | 55 | 19 | 23 | 23 | 2 |

Build pairs (same build.h parameters): `9e07982a7828` / `a99a0d75c2a7`; `44d374ddf4cd` / `72b30c912e52`; `b1506941e701` / `6dc33cc98f59`; `4fd5d4792074` / `f18dd740dbcc`; `e26aa3cfa1bd` / `2f48c3b3e8b1`; `1074b1db653b` / `2a6ac3b9f370`; `81a9c30f13e1` / `aa26734539e9`; `f86b1bd5b50b` / `3b2f6d555285`; `2b6d5972200a` / `462a3dc5d8d5`; `e5f7c70179b2` / `5fe1dad62958`

Checks: hal tt_op_assumed = 118, hal tt_op_proven = 474, original tt_op_assumed = 58, original tt_op_proven = 372.
Mnemonics without a value model (result treated as unknown): none.

