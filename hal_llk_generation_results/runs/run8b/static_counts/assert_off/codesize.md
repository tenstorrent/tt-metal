Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2052.8 | -546.4 |
| unpack | instructions total | 649.8 | 513.2 | -136.6 |
| unpack | Tensix instructions | 20 | 78 | +58 |
| unpack | RISC-V instructions | 629.8 | 435.2 | -194.6 |
| math | text section (B) | 1303.2 | 1016 | -287.2 |
| math | instructions total | 325.8 | 254 | -71.8 |
| math | Tensix instructions | 28 | 20.6 | -7.4 |
| math | RISC-V instructions | 297.8 | 233.4 | -64.4 |
| pack | text section (B) | 2582.4 | 1556.8 | -1025.6 |
| pack | instructions total | 645.6 | 389.2 | -256.4 |
| pack | Tensix instructions | 30 | 56 | +26 |
| pack | RISC-V instructions | 615.6 | 333.2 | -282.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 48.2 |
| `sw` | 250.0 | 174.0 |
| `addi` | 246.4 | 188.2 |
| `beq` | 57.8 | 8.0 |
| `lw` | 126.8 | 82.0 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 4.4 |
| `ttrmwcib1` | 0.0 | 27.0 |
| `lui` | 145.4 | 119.2 |
| `mv` | 64.4 | 38.8 |
| `ttrmwcib0` | 0.0 | 25.0 |
| `ttrmwcib2` | 0.0 | 19.0 |
| `j` | 34.0 | 16.8 |
| `bltu` | 22.0 | 5.4 |
| `ttsetdmareg` | 2.0 | 16.0 |
| `and` | 20.6 | 8.6 |
| `beqz` | 25.0 | 13.2 |
| `ttrmwcib3` | 0.0 | 11.0 |
| `add` | 30.2 | 39.6 |
| `andi` | 37.2 | 28.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
