Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2164 | -435.2 |
| unpack | instructions total | 649.8 | 541 | -108.8 |
| unpack | Tensix instructions | 20 | 64 | +44 |
| unpack | RISC-V instructions | 629.8 | 477 | -152.8 |
| math | text section (B) | 1303.2 | 1022.4 | -280.8 |
| math | instructions total | 325.8 | 255.6 | -70.2 |
| math | Tensix instructions | 28 | 34.6 | +6.6 |
| math | RISC-V instructions | 297.8 | 221 | -76.8 |
| pack | text section (B) | 2582.4 | 1629.6 | -952.8 |
| pack | instructions total | 645.6 | 407.4 | -238.2 |
| pack | Tensix instructions | 30 | 55 | +25 |
| pack | RISC-V instructions | 615.6 | 352.4 | -263.2 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 57.8 |
| `sw` | 250.0 | 190.2 |
| `lw` | 126.8 | 75.2 |
| `addi` | 246.4 | 198.2 |
| `beq` | 57.8 | 11.2 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 1.4 |
| `mv` | 64.4 | 34.0 |
| `ttsetadc` | 0.0 | 22.0 |
| `lui` | 145.4 | 124.6 |
| `ttrmwcib0` | 0.0 | 19.0 |
| `add` | 30.2 | 49.0 |
| `ttrmwcib1` | 0.0 | 17.0 |
| `ttrmwcib2` | 0.0 | 17.0 |
| `bltu` | 22.0 | 7.6 |
| `beqz` | 25.0 | 10.8 |
| `andi` | 37.2 | 23.2 |
| `slli` | 23.4 | 34.2 |
| `ttrmwcib3` | 0.0 | 10.0 |
| `and` | 20.6 | 10.8 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
