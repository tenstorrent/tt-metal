Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2190.4 | -408.8 |
| unpack | instructions total | 649.8 | 547.6 | -102.2 |
| unpack | Tensix instructions | 20 | 66 | +46 |
| unpack | RISC-V instructions | 629.8 | 481.6 | -148.2 |
| math | text section (B) | 1303.2 | 1068 | -235.2 |
| math | instructions total | 325.8 | 267 | -58.8 |
| math | Tensix instructions | 28 | 30 | +2 |
| math | RISC-V instructions | 297.8 | 237 | -60.8 |
| pack | text section (B) | 2582.4 | 1616.8 | -965.6 |
| pack | instructions total | 645.6 | 404.2 | -241.4 |
| pack | Tensix instructions | 30 | 71 | +41 |
| pack | RISC-V instructions | 615.6 | 333.2 | -282.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 51.6 |
| `sw` | 250.0 | 184.4 |
| `lw` | 126.8 | 74.2 |
| `beq` | 57.8 | 11.4 |
| `addi` | 246.4 | 203.6 |
| `ret` | 58.0 | 15.4 |
| `seqz` | 43.0 | 6.4 |
| `ttrmwcib1` | 0.0 | 27.0 |
| `mv` | 64.4 | 40.2 |
| `ttrmwcib0` | 0.0 | 24.0 |
| `ttrmwcib2` | 0.0 | 21.0 |
| `add` | 30.2 | 46.6 |
| `bltu` | 22.0 | 7.0 |
| `andi` | 37.2 | 23.2 |
| `ttrmwcib3` | 0.0 | 14.0 |
| `ttsetdmareg` | 2.0 | 16.0 |
| `beqz` | 25.0 | 12.4 |
| `slli` | 23.4 | 35.6 |
| `j` | 34.0 | 22.8 |
| `lui` | 145.4 | 134.6 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
