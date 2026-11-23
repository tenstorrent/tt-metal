Mode: not paired (different variant_id) - the aggregate of all variants is compared. Variants measured: original 10 / 10, HAL 10 / 10.

| Kernel | Metric (mean per variant) | Original | HAL | Difference |
|---|---|---|---|---|
| unpack | text section (B) | 3463.2 | 2394.4 | -1068.8 |
| unpack | instructions total | 865.8 | 598.6 | -267.2 |
| unpack | Tensix instructions | 21 | 44 | +23 |
| unpack | RISC-V instructions | 844.8 | 554.6 | -290.2 |
| math | text section (B) | 2374.4 | 1152 | -1222.4 |
| math | instructions total | 593.6 | 288 | -305.6 |
| math | Tensix instructions | 36 | 25.6 | -10.4 |
| math | RISC-V instructions | 557.6 | 262.4 | -295.2 |
| pack | text section (B) | 2910.4 | 1720.8 | -1189.6 |
| pack | instructions total | 727.6 | 430.2 | -297.4 |
| pack | Tensix instructions | 30 | 30 | +0 |
| pack | RISC-V instructions | 697.6 | 400.2 | -297.4 |

Largest mnemonic differences (mean count per kernel triple):

| Mnemonic | Original | HAL |
|---|---|---|
| `li` | 205.4 | 50.0 |
| `lw` | 198.6 | 97.0 |
| `beq` | 108.0 | 7.8 |
| `bne` | 111.0 | 28.2 |
| `sw` | 287.2 | 229.8 |
| `j` | 70.4 | 18.2 |
| `seqz` | 50.6 | 3.8 |
| `addi` | 280.2 | 234.6 |
| `ret` | 60.0 | 15.8 |
| `mv` | 76.2 | 36.0 |
| `add` | 35.8 | 65.6 |
| `bltu` | 38.8 | 12.8 |
| `zext.b` | 37.4 | 13.0 |
| `ebreak` | 31.6 | 8.2 |
| `jal` | 26.4 | 6.0 |
| `andi` | 46.0 | 25.8 |
| `beqz` | 30.8 | 13.8 |
| `ttrmwcib1` | 0.0 | 15.0 |
| `bgeu` | 27.6 | 13.0 |
| `slli` | 29.0 | 42.0 |

_Tensix instruction = mnemonic matching `^(tt[a-z0-9_]*|\.insn|\.word)$`. The TT/TTI split is an estimate; confirm it with elfray._
