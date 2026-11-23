# TT / TTI (elfray) — 20261106_020508_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 522.7 | 21 / 47 | 11 / 33 | 11 / 53 | 32 / 100 |
| math | 536.6 / 262.6 | 36 / 31.2 | 8 / 3.2 | 8.8 / 8.4 | 44.8 / 39.6 |
| pack | 670.6 / 419.8 | 30 / 63 | 14 / 24 | 20 / 38 | 50 / 101 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 12, `ttrmwcib0` 9, `ttrmwcib2` 7, `ttrmwcib3` 7, `ttstallwait` 4, `ttunpacr` 4, `ttsetc16` 2, `ttsetadcxy` 1, `ttsetadczw` 1

TT: `RMWCIB2` 9, `RMWCIB3` 9, `RMWCIB0` 8, `RMWCIB1` 5, `SETADCXX` 1, `SETADC` 1

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 21.4, `ttstallwait` 5.6, `ttsetrwc` 1.2, `ttseminit` 1, `ttsemwait` 1, `ttsempost` 1

TT: `SETC16` 1.2, `RMWCIB3` 1, `RMWCIB0` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib2` 17, `ttsetc16` 9, `ttrmwcib0` 8, `ttrmwcib1` 8, `ttstallwait` 7, `ttrmwcib3` 6, `ttsetadcxy` 2, `ttsetadczw` 2, `ttsetadcxx` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB0` 9, `RMWCIB1` 8, `RMWCIB2` 3, `RMWCIB3` 3, `ZEROACC` 1
