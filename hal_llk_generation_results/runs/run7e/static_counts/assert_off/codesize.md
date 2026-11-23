Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2278.4 | -320.8 |
| unpack | instructions total | 649.8 | 569.6 | -80.2 |
| unpack | Tensix instructions | 20 | 44 | +24 |
| unpack | RISC-V instructions | 629.8 | 525.6 | -104.2 |
| math | text section (B) | 1303.2 | 1040.8 | -262.4 |
| math | instructions total | 325.8 | 260.2 | -65.6 |
| math | Tensix instructions | 28 | 24 | -4 |
| math | RISC-V instructions | 297.8 | 236.2 | -61.6 |
| pack | text section (B) | 2582.4 | 1696.8 | -885.6 |
| pack | instructions total | 645.6 | 424.2 | -221.4 |
| pack | Tensix instructions | 30 | 30 | +0 |
| pack | RISC-V instructions | 615.6 | 394.2 | -221.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 43.4 |
| `beq` | 57.8 | 7.2 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 3.4 |
| `lw` | 126.8 | 88.8 |
| `add` | 30.2 | 62.8 |
| `mv` | 64.4 | 36.8 |
| `sw` | 250.0 | 230.2 |
| `addi` | 246.4 | 226.6 |
| `j` | 34.0 | 14.2 |
| `slli` | 23.4 | 40.6 |
| `bltu` | 22.0 | 5.4 |
| `ttrmwcib1` | 0.0 | 15.0 |
| `andi` | 37.2 | 24.8 |
| `beqz` | 25.0 | 12.6 |
| `ttrmwcib0` | 0.0 | 12.0 |
| `zext.h` | 2.0 | 13.8 |
| `bne` | 38.8 | 27.4 |
| `srli` | 7.0 | 17.8 |
| `ttrmwcib2` | 0.0 | 10.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
