Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 1973.6 | -1489.6 |
| unpack | instructions total | 865.8 | 493.4 | -372.4 |
| unpack | Tensix instructions | 21 | 44 | +23 |
| unpack | RISC-V instructions | 844.8 | 449.4 | -395.4 |
| math | text section (B) | 2374.4 | 1063.2 | -1311.2 |
| math | instructions total | 593.6 | 265.8 | -327.8 |
| math | Tensix instructions | 36 | 32.4 | -3.6 |
| math | RISC-V instructions | 557.6 | 233.4 | -324.2 |
| pack | text section (B) | 2910.4 | 1585.6 | -1324.8 |
| pack | instructions total | 727.6 | 396.4 | -331.2 |
| pack | Tensix instructions | 30 | 35 | +5 |
| pack | RISC-V instructions | 697.6 | 361.4 | -336.2 |

Largest mnemonic differences (mean count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 42.4 |
| `lw` | 198.6 | 73.8 |
| `sw` | 287.2 | 182.2 |
| `beq` | 108.0 | 7.2 |
| `bne` | 111.0 | 26.4 |
| `addi` | 280.2 | 205.8 |
| `j` | 70.4 | 13.6 |
| `ret` | 60.0 | 15.6 |
| `seqz` | 50.6 | 8.0 |
| `mv` | 76.2 | 35.2 |
| `lui` | 159.8 | 121.2 |
| `bltu` | 38.8 | 8.8 |
| `zext.b` | 37.4 | 10.2 |
| `ebreak` | 31.6 | 4.8 |
| `jal` | 26.4 | 6.0 |
| `andi` | 46.0 | 26.2 |
| `beqz` | 30.8 | 11.0 |
| `bgeu` | 27.6 | 9.4 |
| `slli` | 29.0 | 46.2 |
| `add` | 35.8 | 52.2 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI split is an estimate; confirm it with elfray._
