Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2140 | -1323.2 |
| unpack | instructions total | 865.8 | 535 | -330.8 |
| unpack | Tensix instructions | 21 | 56 | +35 |
| unpack | RISC-V instructions | 844.8 | 479 | -365.8 |
| math | text section (B) | 2374.4 | 1316 | -1058.4 |
| math | instructions total | 593.6 | 329 | -264.6 |
| math | Tensix instructions | 36 | 25 | -11 |
| math | RISC-V instructions | 557.6 | 304 | -253.6 |
| pack | text section (B) | 2910.4 | 1840.8 | -1069.6 |
| pack | instructions total | 727.6 | 460.2 | -267.4 |
| pack | Tensix instructions | 30 | 42 | +12 |
| pack | RISC-V instructions | 697.6 | 418.2 | -279.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 51.6 |
| `lw` | 198.6 | 91.8 |
| `beq` | 108.0 | 12.4 |
| `bne` | 111.0 | 31.4 |
| `sw` | 287.2 | 235.6 |
| `j` | 70.4 | 19.4 |
| `addi` | 280.2 | 234.4 |
| `ret` | 60.0 | 15.0 |
| `seqz` | 50.6 | 8.2 |
| `mv` | 76.2 | 39.0 |
| `bltu` | 38.8 | 8.8 |
| `ebreak` | 31.6 | 4.6 |
| `zext.b` | 37.4 | 10.4 |
| `ttrmwcib0` | 0.0 | 24.2 |
| `andi` | 46.0 | 24.0 |
| `add` | 35.8 | 56.6 |
| `jal` | 26.4 | 6.0 |
| `ttrmwcib1` | 0.0 | 20.0 |
| `ttrmwcib2` | 0.0 | 17.0 |
| `bgeu` | 27.6 | 10.6 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
