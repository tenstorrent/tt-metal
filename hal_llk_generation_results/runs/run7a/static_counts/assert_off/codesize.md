Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 1966.4 | -632.8 |
| unpack | instructions total | 649.8 | 491.6 | -158.2 |
| unpack | Tensix instructions | 20 | 44 | +24 |
| unpack | RISC-V instructions | 629.8 | 447.6 | -182.2 |
| math | text section (B) | 1303.2 | 1021.6 | -281.6 |
| math | instructions total | 325.8 | 255.4 | -70.4 |
| math | Tensix instructions | 28 | 35.6 | +7.6 |
| math | RISC-V instructions | 297.8 | 219.8 | -78.0 |
| pack | text section (B) | 2582.4 | 1557.6 | -1024.8 |
| pack | instructions total | 645.6 | 389.4 | -256.2 |
| pack | Tensix instructions | 30 | 35 | +5 |
| pack | RISC-V instructions | 615.6 | 354.4 | -261.2 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 37.0 |
| `sw` | 250.0 | 182.8 |
| `lw` | 126.8 | 70.2 |
| `beq` | 57.8 | 9.0 |
| `ret` | 58.0 | 15.0 |
| `addi` | 246.4 | 204.2 |
| `seqz` | 43.0 | 8.0 |
| `mv` | 64.4 | 35.6 |
| `lui` | 145.4 | 121.2 |
| `slli` | 23.4 | 46.0 |
| `j` | 34.0 | 12.6 |
| `add` | 30.2 | 51.0 |
| `bltu` | 22.0 | 4.0 |
| `ttrmwcib1` | 0.0 | 16.0 |
| `beqz` | 25.0 | 9.6 |
| `bne` | 38.8 | 25.6 |
| `ttrmwcib2` | 0.0 | 13.0 |
| `ttrmwcib0` | 0.0 | 12.0 |
| `zext.h` | 2.0 | 13.2 |
| `andi` | 37.2 | 26.2 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
