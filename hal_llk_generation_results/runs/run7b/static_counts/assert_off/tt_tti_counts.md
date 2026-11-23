# TT / TTI counts (static, per ELF)

Columns: text bytes = `.text` size; TTI = `tt*` lines in elfray `.text` output; TT elfray = stores elfray marks `tensix issue` (lower bound); TT linear = harness linear register tracking; TT all paths = stores whose address is in `0xFFE40000`-`0xFFE4FFFF` on every path (dataflow analysis); unresolved = stores whose address could not be settled.

## assert_off

`/proj_sw/user_dev/ctr-sbratic/tools/asserts_off/out/7b/assert_off`

Builds: original 10, HAL 10. Varying build parameters: BROADCAST_TYPE, EN_DEST_REUSE, REUSE_DEST_TYPE, is_fp32_dest_acc_en.

### Average per build

| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved | TTI + TT all paths |
|---|---|---|---|---|---|---|---|---|
| original | unpack | 2371.2 | 20 | 11 | 11 | 11 | 2 | 31 |
| original | math | 1075.2 | 28 | 7.6 | 10 | 10 | 2.8 | 38 |
| original | pack | 2354.4 | 30 | 15 | 20 | 20 | 2 | 50 |
| original | sum | 5800.8 | 78 | 33.6 | 41 | 41 | 6.8 | 119 |
| hal | unpack | 1912 | 56 | 36 | 40 | 40 | 2 | 96 |
| hal | math | 1011.2 | 25 | 9.8 | 13.8 | 13.8 | 2 | 38.8 |
| hal | pack | 1572.8 | 42 | 28.4 | 42 | 42 | 2 | 84 |
| hal | sum | 4496 | 123 | 74.2 | 95.8 | 95.8 | 6 | 218.8 |

### Total over all builds

| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |
|---|---|---|---|---|---|---|---|
| original | unpack | 23712 | 200 | 110 | 110 | 110 | 20 |
| original | math | 10752 | 280 | 76 | 100 | 100 | 28 |
| original | pack | 23544 | 300 | 150 | 200 | 200 | 20 |
| original | sum | 58008 | 780 | 336 | 410 | 410 | 68 |
| hal | unpack | 19120 | 560 | 360 | 400 | 400 | 20 |
| hal | math | 10112 | 250 | 98 | 138 | 138 | 20 |
| hal | pack | 15728 | 420 | 284 | 420 | 420 | 20 |
| hal | sum | 44960 | 1230 | 742 | 958 | 958 | 60 |

### Per build

| Side | Build | Variant | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |
|---|---|---|---|---|---|---|---|---|---|
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | unpack | 2392 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | math | 976 | 23 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `1074b1db653b` | pack | 2352 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | unpack | 2280 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | math | 980 | 24 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2b6d5972200a` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | unpack | 2416 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | math | 1048 | 25 | 7 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `44d374ddf4cd` | pack | 2352 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | unpack | 2392 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | math | 1396 | 45 | 10 | 12 | 12 | 3 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `4fd5d4792074` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | unpack | 2376 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | math | 976 | 23 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `81a9c30f13e1` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | unpack | 2416 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | math | 1048 | 25 | 7 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `9e07982a7828` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | unpack | 2392 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | math | 1396 | 45 | 10 | 12 | 12 | 3 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `b1506941e701` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | unpack | 2392 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | math | 976 | 23 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `e26aa3cfa1bd` | pack | 2356 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | unpack | 2280 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | math | 980 | 24 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `e5f7c70179b2` | pack | 2352 | 30 | 15 | 20 | 20 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | unpack | 2376 | 20 | 11 | 11 | 11 | 2 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | math | 976 | 23 | 7 | 9 | 9 | 3 |
| original | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `f86b1bd5b50b` | pack | 2352 | 30 | 15 | 20 | 20 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | unpack | 1900 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | math | 912 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `2a6ac3b9f370` | pack | 1592 | 42 | 29 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | unpack | 1892 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | math | 904 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `2f48c3b3e8b1` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | unpack | 1980 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | math | 912 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `3b2f6d555285` | pack | 1592 | 42 | 29 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | unpack | 1832 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | math | 932 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `462a3dc5d8d5` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | unpack | 1840 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | math | 940 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=SCALAR EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `5fe1dad62958` | pack | 1592 | 42 | 29 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | unpack | 1892 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | math | 1348 | 41 | 13 | 21 | 21 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCA is_fp32_dest_acc_en=false | `6dc33cc98f59` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | unpack | 1964 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | math | 960 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=true | `72b30c912e52` | pack | 1592 | 42 | 29 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | unpack | 1956 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | math | 952 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=COL EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `a99a0d75c2a7` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | unpack | 1972 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | math | 904 | 21 | 9 | 12 | 12 | 2 |
| hal | BROADCAST_TYPE=ROW EN_DEST_REUSE=- REUSE_DEST_TYPE=- is_fp32_dest_acc_en=false | `aa26734539e9` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | unpack | 1892 | 56 | 36 | 40 | 40 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | math | 1348 | 41 | 13 | 21 | 21 | 2 |
| hal | BROADCAST_TYPE=NONE EN_DEST_REUSE= REUSE_DEST_TYPE=DEST_TO_SRCB is_fp32_dest_acc_en=false | `f18dd740dbcc` | pack | 1560 | 42 | 28 | 42 | 42 | 2 |

Build pairs (same build.h parameters): `9e07982a7828` / `a99a0d75c2a7`; `44d374ddf4cd` / `72b30c912e52`; `b1506941e701` / `6dc33cc98f59`; `4fd5d4792074` / `f18dd740dbcc`; `e26aa3cfa1bd` / `2f48c3b3e8b1`; `1074b1db653b` / `2a6ac3b9f370`; `81a9c30f13e1` / `aa26734539e9`; `f86b1bd5b50b` / `3b2f6d555285`; `2b6d5972200a` / `462a3dc5d8d5`; `e5f7c70179b2` / `5fe1dad62958`

Checks: hal tt_op_assumed = 204, hal tt_op_proven = 754, original tt_op_assumed = 56, original tt_op_proven = 354.
Mnemonics without a value model (result treated as unknown): none.

