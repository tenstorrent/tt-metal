Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2248 | -1215.2 |
| unpack | instructions total | 865.8 | 562 | -303.8 |
| unpack | Tensix instructions | 21 | 67 | +46 |
| unpack | RISC-V instructions | 844.8 | 495 | -349.8 |
| math | text section (B) | 2374.4 | 1040.8 | -1333.6 |
| math | instructions total | 593.6 | 260.2 | -333.4 |
| math | Tensix instructions | 36 | 29.6 | -6.4 |
| math | RISC-V instructions | 557.6 | 230.6 | -327.0 |
| pack | text section (B) | 2910.4 | 1629.6 | -1280.8 |
| pack | instructions total | 727.6 | 407.4 | -320.2 |
| pack | Tensix instructions | 30 | 55 | +25 |
| pack | RISC-V instructions | 697.6 | 352.4 | -345.2 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 59.2 |
| `lw` | 198.6 | 81.6 |
| `beq` | 108.0 | 12.2 |
| `sw` | 287.2 | 192.2 |
| `bne` | 111.0 | 30.4 |
| `addi` | 280.2 | 201.0 |
| `seqz` | 50.6 | 1.4 |
| `j` | 70.4 | 26.0 |
| `ret` | 60.0 | 15.6 |
| `mv` | 76.2 | 35.0 |
| `lui` | 159.8 | 126.4 |
| `ebreak` | 31.6 | 4.4 |
| `bltu` | 38.8 | 12.8 |
| `andi` | 46.0 | 22.2 |
| `zext.b` | 37.4 | 15.0 |
| `ttsetadc` | 0.0 | 22.0 |
| `jal` | 26.4 | 6.0 |
| `beqz` | 30.8 | 11.8 |
| `ttrmwcib0` | 0.0 | 19.0 |
| `ttrmwcib1` | 0.0 | 17.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
