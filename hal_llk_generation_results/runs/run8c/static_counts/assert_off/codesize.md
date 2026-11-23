Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 1926.4 | -672.8 |
| unpack | instructions total | 649.8 | 481.6 | -168.2 |
| unpack | Tensix instructions | 20 | 50 | +30 |
| unpack | RISC-V instructions | 629.8 | 431.6 | -198.2 |
| math | text section (B) | 1303.2 | 1107.6 | -195.6 |
| math | instructions total | 325.8 | 276.9 | -48.9 |
| math | Tensix instructions | 28 | 27.8 | -0.2 |
| math | RISC-V instructions | 297.8 | 249.1 | -48.7 |
| pack | text section (B) | 2582.4 | 1720.8 | -861.6 |
| pack | instructions total | 645.6 | 430.2 | -215.4 |
| pack | Tensix instructions | 30 | 73 | +43 |
| pack | RISC-V instructions | 615.6 | 357.2 | -258.4 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 50.0 |
| `sw` | 250.0 | 192.3 |
| `addi` | 246.4 | 195.7 |
| `lw` | 126.8 | 78.9 |
| `beq` | 57.8 | 10.2 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 3.4 |
| `ttrmwcib1` | 0.0 | 30.0 |
| `ttrmwcib0` | 0.0 | 26.2 |
| `mv` | 64.4 | 38.8 |
| `bltu` | 22.0 | 5.4 |
| `j` | 34.0 | 18.4 |
| `lui` | 145.4 | 129.8 |
| `andi` | 37.2 | 23.2 |
| `beqz` | 25.0 | 11.2 |
| `add` | 30.2 | 43.4 |
| `ttrmwcib2` | 0.0 | 12.0 |
| `bne` | 38.8 | 28.6 |
| `ttrmwcib3` | 0.0 | 9.0 |
| `slli` | 23.4 | 31.8 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
