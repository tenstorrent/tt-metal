Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2318.8 | -280.4 |
| unpack | instructions total | 649.8 | 579.7 | -70.1 |
| unpack | Tensix instructions | 20 | 47 | +27 |
| unpack | RISC-V instructions | 629.8 | 532.7 | -97.1 |
| math | text section (B) | 1303.2 | 1092 | -211.2 |
| math | instructions total | 325.8 | 273 | -52.8 |
| math | Tensix instructions | 28 | 31 | +3 |
| math | RISC-V instructions | 297.8 | 242 | -55.8 |
| pack | text section (B) | 2582.4 | 1891.2 | -691.2 |
| pack | instructions total | 645.6 | 472.8 | -172.8 |
| pack | Tensix instructions | 30 | 63 | +33 |
| pack | RISC-V instructions | 615.6 | 409.8 | -205.8 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 53.5 |
| `lw` | 126.8 | 75.8 |
| `add` | 30.2 | 79.4 |
| `beq` | 57.8 | 10.2 |
| `ret` | 58.0 | 15.6 |
| `sw` | 250.0 | 208.0 |
| `seqz` | 43.0 | 4.6 |
| `mv` | 64.4 | 38.0 |
| `slli` | 23.4 | 48.8 |
| `ttrmwcib2` | 0.0 | 24.0 |
| `ttrmwcib1` | 0.0 | 20.0 |
| `zext.h` | 2.0 | 20.8 |
| `bltu` | 22.0 | 4.2 |
| `ttrmwcib0` | 0.0 | 17.0 |
| `lui` | 145.4 | 129.4 |
| `srli` | 7.0 | 23.0 |
| `addi` | 246.4 | 232.1 |
| `ttrmwcib3` | 0.0 | 13.0 |
| `andi` | 37.2 | 24.6 |
| `j` | 34.0 | 21.6 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
