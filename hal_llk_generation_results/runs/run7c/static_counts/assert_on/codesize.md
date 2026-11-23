Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2144 | -1319.2 |
| unpack | instructions total | 865.8 | 536 | -329.8 |
| unpack | Tensix instructions | 21 | 37 | +16 |
| unpack | RISC-V instructions | 844.8 | 499 | -345.8 |
| math | text section (B) | 2374.4 | 1320 | -1054.4 |
| math | instructions total | 593.6 | 330 | -263.6 |
| math | Tensix instructions | 36 | 17.8 | -18.2 |
| math | RISC-V instructions | 557.6 | 312.2 | -245.4 |
| pack | text section (B) | 2910.4 | 1895.2 | -1015.2 |
| pack | instructions total | 727.6 | 473.8 | -253.8 |
| pack | Tensix instructions | 30 | 63 | +33 |
| pack | RISC-V instructions | 697.6 | 410.8 | -286.8 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 48.6 |
| `lw` | 198.6 | 79.6 |
| `beq` | 108.0 | 8.0 |
| `bne` | 111.0 | 29.2 |
| `j` | 70.4 | 19.0 |
| `sw` | 287.2 | 239.6 |
| `ret` | 60.0 | 15.8 |
| `mv` | 76.2 | 36.0 |
| `seqz` | 50.6 | 11.8 |
| `bltu` | 38.8 | 9.8 |
| `ebreak` | 31.6 | 3.6 |
| `addi` | 280.2 | 255.0 |
| `zext.b` | 37.4 | 12.6 |
| `andi` | 46.0 | 22.8 |
| `add` | 35.8 | 57.0 |
| `jal` | 26.4 | 6.0 |
| `ttrmwcib2` | 0.0 | 20.0 |
| `ttrmwcib1` | 0.0 | 18.0 |
| `ttsetc16` | 23.6 | 6.0 |
| `beqz` | 30.8 | 14.4 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
