Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 1944.8 | -1518.4 |
| unpack | instructions total | 865.8 | 486.2 | -379.6 |
| unpack | Tensix instructions | 21 | 36 | +15 |
| unpack | RISC-V instructions | 844.8 | 450.2 | -394.6 |
| math | text section (B) | 2374.4 | 1181.2 | -1193.2 |
| math | instructions total | 593.6 | 295.3 | -298.3 |
| math | Tensix instructions | 36 | 22.2 | -13.8 |
| math | RISC-V instructions | 557.6 | 273.1 | -284.5 |
| pack | text section (B) | 2910.4 | 1608 | -1302.4 |
| pack | instructions total | 727.6 | 402 | -325.6 |
| pack | Tensix instructions | 30 | 66 | +36 |
| pack | RISC-V instructions | 697.6 | 336 | -361.6 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 51.2 |
| `lw` | 198.6 | 75.3 |
| `sw` | 287.2 | 175.8 |
| `beq` | 108.0 | 11.2 |
| `bne` | 111.0 | 30.8 |
| `addi` | 280.2 | 210.1 |
| `seqz` | 50.6 | 3.0 |
| `j` | 70.4 | 24.7 |
| `ret` | 60.0 | 15.0 |
| `mv` | 76.2 | 37.2 |
| `ttrmwcib1` | 0.0 | 29.0 |
| `ttrmwcib0` | 0.0 | 27.2 |
| `zext.b` | 37.4 | 11.6 |
| `bltu` | 38.8 | 13.6 |
| `ebreak` | 31.6 | 7.4 |
| `andi` | 46.0 | 22.1 |
| `lui` | 159.8 | 137.0 |
| `jal` | 26.4 | 6.0 |
| `beqz` | 30.8 | 11.8 |
| `bgeu` | 27.6 | 11.8 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
