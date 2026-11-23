Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 1828.8 | -770.4 |
| unpack | instructions total | 649.8 | 457.2 | -192.6 |
| unpack | Tensix instructions | 20 | 36 | +16 |
| unpack | RISC-V instructions | 629.8 | 421.2 | -208.6 |
| math | text section (B) | 1303.2 | 1123.6 | -179.6 |
| math | instructions total | 325.8 | 280.9 | -44.9 |
| math | Tensix instructions | 28 | 22.2 | -5.8 |
| math | RISC-V instructions | 297.8 | 258.7 | -39.1 |
| pack | text section (B) | 2582.4 | 1608 | -974.4 |
| pack | instructions total | 645.6 | 402 | -243.6 |
| pack | Tensix instructions | 30 | 66 | +36 |
| pack | RISC-V instructions | 615.6 | 336 | -279.6 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 48.8 |
| `sw` | 250.0 | 176.6 |
| `lw` | 126.8 | 70.1 |
| `beq` | 57.8 | 11.8 |
| `addi` | 246.4 | 203.0 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 3.0 |
| `ttrmwcib1` | 0.0 | 29.0 |
| `ttrmwcib0` | 0.0 | 27.2 |
| `mv` | 64.4 | 38.0 |
| `andi` | 37.2 | 21.1 |
| `bltu` | 22.0 | 7.0 |
| `beqz` | 25.0 | 10.6 |
| `j` | 34.0 | 20.1 |
| `and` | 20.6 | 9.2 |
| `lui` | 145.4 | 134.3 |
| `add` | 30.2 | 39.7 |
| `slli` | 23.4 | 32.7 |
| `bgeu` | 18.6 | 10.2 |
| `srli` | 7.0 | 15.4 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
