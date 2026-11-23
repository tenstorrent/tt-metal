Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 2599.2 | 2141.6 | -457.6 |
| unpack | instructions total | 649.8 | 535.4 | -114.4 |
| unpack | Tensix instructions | 20 | 37 | +17 |
| unpack | RISC-V instructions | 629.8 | 498.4 | -131.4 |
| math | text section (B) | 1303.2 | 1255.2 | -48.0 |
| math | instructions total | 325.8 | 313.8 | -12.0 |
| math | Tensix instructions | 28 | 17.8 | -10.2 |
| math | RISC-V instructions | 297.8 | 296 | -1.8 |
| pack | text section (B) | 2582.4 | 1867.2 | -715.2 |
| pack | instructions total | 645.6 | 466.8 | -178.8 |
| pack | Tensix instructions | 30 | 63 | +33 |
| pack | RISC-V instructions | 615.6 | 403.8 | -211.8 |

Largest differences in mnemonics (average count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 135.2 | 43.2 |
| `lw` | 126.8 | 76.6 |
| `beq` | 57.8 | 9.8 |
| `ret` | 58.0 | 15.0 |
| `seqz` | 43.0 | 11.8 |
| `mv` | 64.4 | 35.4 |
| `add` | 30.2 | 55.6 |
| `lui` | 145.4 | 170.6 |
| `ttrmwcib2` | 0.0 | 20.0 |
| `ttrmwcib1` | 0.0 | 18.0 |
| `ttsetc16` | 23.6 | 6.0 |
| `ttrmwcib0` | 0.0 | 16.0 |
| `j` | 34.0 | 18.4 |
| `slli` | 23.4 | 39.0 |
| `bltu` | 22.0 | 6.4 |
| `andi` | 37.2 | 21.8 |
| `beqz` | 25.0 | 13.0 |
| `bne` | 38.8 | 27.4 |
| `sw` | 250.0 | 239.6 |
| `ttrmwcib3` | 0.0 | 10.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI ratio is an estimate; confirm with the elfray tool._
