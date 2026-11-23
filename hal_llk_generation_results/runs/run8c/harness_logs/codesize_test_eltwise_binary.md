Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2002.4 | -1460.8 |
| unpack | instructions total | 865.8 | 500.6 | -365.2 |
| unpack | Tensix instructions | 21 | 48.5 | +27.5 |
| unpack | RISC-V instructions | 844.8 | 452.1 | -392.7 |
| math | text section (B) | 2374.4 | 1174 | -1200.4 |
| math | instructions total | 593.6 | 293.5 | -300.1 |
| math | Tensix instructions | 36 | 27.8 | -8.2 |
| math | RISC-V instructions | 557.6 | 265.7 | -291.9 |
| pack | text section (B) | 2910.4 | 1768.8 | -1141.6 |
| pack | instructions total | 727.6 | 442.2 | -285.4 |
| pack | Tensix instructions | 30 | 73 | +43 |
| pack | RISC-V instructions | 697.6 | 369.2 | -328.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 53.0 |
| `lw` | 198.6 | 86.1 |
| `beq` | 108.0 | 9.9 |
| `sw` | 287.2 | 193.3 |
| `bne` | 111.0 | 29.4 |
| `addi` | 280.2 | 205.7 |
| `j` | 70.4 | 22.3 |
| `seqz` | 50.6 | 3.4 |
| `ret` | 60.0 | 15.8 |
| `mv` | 76.2 | 38.2 |
| `ttrmwcib1` | 0.0 | 30.0 |
| `ttrmwcib0` | 0.0 | 26.2 |
| `bltu` | 38.8 | 12.8 |
| `lui` | 159.8 | 135.0 |
| `ebreak` | 31.6 | 7.4 |
| `zext.b` | 37.4 | 13.2 |
| `andi` | 46.0 | 22.2 |
| `jal` | 26.4 | 6.0 |
| `beqz` | 30.8 | 12.0 |
| `bgeu` | 27.6 | 11.6 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
