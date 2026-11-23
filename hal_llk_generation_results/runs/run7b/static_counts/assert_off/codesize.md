Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2140 | -459.2 |
| unpack | instructions total | 649.8 | 535 | -114.8 |
| unpack | Tensix instructions | 20 | 56 | +36 |
| unpack | RISC-V instructions | 629.8 | 479 | -150.8 |
| math | text section (B) | 1303.2 | 1239.2 | -64.0 |
| math | instructions total | 325.8 | 309.8 | -16.0 |
| math | Tensix instructions | 28 | 25 | -3 |
| math | RISC-V instructions | 297.8 | 284.8 | -13.0 |
| pack | text section (B) | 2582.4 | 1800.8 | -781.6 |
| pack | instructions total | 645.6 | 450.2 | -195.4 |
| pack | Tensix instructions | 30 | 42 | +12 |
| pack | RISC-V instructions | 615.6 | 408.2 | -207.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 46.4 |
| `beq` | 57.8 | 12.4 |
| `ret` | 58.0 | 15.0 |
| `lw` | 126.8 | 86.6 |
| `seqz` | 43.0 | 8.2 |
| `add` | 30.2 | 56.4 |
| `mv` | 64.4 | 38.6 |
| `ttrmwcib0` | 0.0 | 24.2 |
| `slli` | 23.4 | 44.6 |
| `ttrmwcib1` | 0.0 | 20.0 |
| `j` | 34.0 | 16.0 |
| `bltu` | 22.0 | 4.4 |
| `ttrmwcib2` | 0.0 | 17.0 |
| `sw` | 250.0 | 233.8 |
| `addi` | 246.4 | 230.4 |
| `andi` | 37.2 | 24.0 |
| `beqz` | 25.0 | 13.2 |
| `zext.h` | 2.0 | 13.6 |
| `ttrmwcib3` | 0.0 | 10.0 |
| `srli` | 7.0 | 16.8 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
