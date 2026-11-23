Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2244 | -1219.2 |
| unpack | instructions total | 865.8 | 561 | -304.8 |
| unpack | Tensix instructions | 21 | 66 | +45 |
| unpack | RISC-V instructions | 844.8 | 495 | -349.8 |
| math | text section (B) | 2374.4 | 1130.4 | -1244.0 |
| math | instructions total | 593.6 | 282.6 | -311.0 |
| math | Tensix instructions | 36 | 30 | -6 |
| math | RISC-V instructions | 557.6 | 252.6 | -305.0 |
| pack | text section (B) | 2910.4 | 1612.8 | -1297.6 |
| pack | instructions total | 727.6 | 403.2 | -324.4 |
| pack | Tensix instructions | 30 | 70 | +40 |
| pack | RISC-V instructions | 697.6 | 333.2 | -364.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 51.8 |
| `lw` | 198.6 | 81.0 |
| `sw` | 287.2 | 185.6 |
| `beq` | 108.0 | 12.6 |
| `bne` | 111.0 | 30.2 |
| `addi` | 280.2 | 205.4 |
| `j` | 70.4 | 22.2 |
| `seqz` | 50.6 | 6.4 |
| `ret` | 60.0 | 16.0 |
| `mv` | 76.2 | 41.6 |
| `bltu` | 38.8 | 11.4 |
| `ebreak` | 31.6 | 4.4 |
| `zext.b` | 37.4 | 10.4 |
| `ttrmwcib1` | 0.0 | 27.0 |
| `ttrmwcib0` | 0.0 | 24.0 |
| `andi` | 46.0 | 22.2 |
| `lui` | 159.8 | 136.6 |
| `ttrmwcib2` | 0.0 | 21.0 |
| `jal` | 26.4 | 6.2 |
| `beqz` | 30.8 | 13.6 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
