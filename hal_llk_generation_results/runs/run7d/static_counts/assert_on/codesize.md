Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2318.8 | -1144.4 |
| unpack | instructions total | 865.8 | 579.7 | -286.1 |
| unpack | Tensix instructions | 21 | 47 | +26 |
| unpack | RISC-V instructions | 844.8 | 532.7 | -312.1 |
| math | text section (B) | 2374.4 | 1278.4 | -1096.0 |
| math | instructions total | 593.6 | 319.6 | -274.0 |
| math | Tensix instructions | 36 | 31.2 | -4.8 |
| math | RISC-V instructions | 557.6 | 288.4 | -269.2 |
| pack | text section (B) | 2910.4 | 1907.2 | -1003.2 |
| pack | instructions total | 727.6 | 476.8 | -250.8 |
| pack | Tensix instructions | 30 | 63 | +33 |
| pack | RISC-V instructions | 697.6 | 413.8 | -283.8 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 63.1 |
| `lw` | 198.6 | 87.0 |
| `beq` | 108.0 | 12.0 |
| `bne` | 111.0 | 34.0 |
| `sw` | 287.2 | 215.6 |
| `add` | 35.8 | 83.2 |
| `seqz` | 50.6 | 4.0 |
| `j` | 70.4 | 24.0 |
| `ret` | 60.0 | 15.0 |
| `addi` | 280.2 | 236.5 |
| `mv` | 76.2 | 38.0 |
| `bltu` | 38.8 | 6.6 |
| `lui` | 159.8 | 129.8 |
| `ebreak` | 31.6 | 3.8 |
| `zext.b` | 37.4 | 11.2 |
| `ttrmwcib2` | 0.0 | 24.0 |
| `zext.h` | 2.0 | 22.6 |
| `andi` | 46.0 | 25.6 |
| `jal` | 26.4 | 6.0 |
| `ttrmwcib1` | 0.0 | 20.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
