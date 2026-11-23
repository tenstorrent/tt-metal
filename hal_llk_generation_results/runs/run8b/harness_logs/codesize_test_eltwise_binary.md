Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2104.8 | -1358.4 |
| unpack | instructions total | 865.8 | 526.2 | -339.6 |
| unpack | Tensix instructions | 21 | 78 | +57 |
| unpack | RISC-V instructions | 844.8 | 448.2 | -396.6 |
| math | text section (B) | 2374.4 | 1079.2 | -1295.2 |
| math | instructions total | 593.6 | 269.8 | -323.8 |
| math | Tensix instructions | 36 | 20.6 | -15.4 |
| math | RISC-V instructions | 557.6 | 249.2 | -308.4 |
| pack | text section (B) | 2910.4 | 1616.8 | -1293.6 |
| pack | instructions total | 727.6 | 404.2 | -323.4 |
| pack | Tensix instructions | 30 | 56 | +26 |
| pack | RISC-V instructions | 697.6 | 348.2 | -349.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 49.4 |
| `sw` | 287.2 | 179.2 |
| `lw` | 198.6 | 95.4 |
| `beq` | 108.0 | 10.2 |
| `addi` | 280.2 | 191.2 |
| `bne` | 111.0 | 31.8 |
| `j` | 70.4 | 17.2 |
| `seqz` | 50.6 | 4.4 |
| `ret` | 60.0 | 15.8 |
| `mv` | 76.2 | 38.2 |
| `lui` | 159.8 | 123.2 |
| `bltu` | 38.8 | 10.8 |
| `ttrmwcib1` | 0.0 | 27.0 |
| `ebreak` | 31.6 | 5.4 |
| `ttrmwcib0` | 0.0 | 25.0 |
| `zext.b` | 37.4 | 14.2 |
| `jal` | 26.4 | 6.0 |
| `andi` | 46.0 | 26.0 |
| `ttrmwcib2` | 0.0 | 19.0 |
| `beqz` | 30.8 | 14.4 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
