# TT / TTI (elfray) — 20261106_095649_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 541.6 | 21 / 44 | 11 / 37 | 11 / 51 | 32 / 95 |
| math | 536.6 / 231 | 36 / 25.6 | 8 / 2 | 8.8 / 6 | 44.8 / 31.6 |
| pack | 670.6 / 373.2 | 30 / 30 | 14 / 23.4 | 20 / 38 | 50 / 68 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 11, `ttrmwcib0` 7, `ttrmwcib2` 7, `ttrmwcib3` 5, `ttsetc16` 4, `ttstallwait` 3, `ttsetadczw` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1

TT: `RMWCIB3` 11, `RMWCIB2` 10, `RMWCIB0` 8, `RMWCIB1` 4, `SETADCXX` 4

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 13.2, `ttsetrwc` 3, `ttstallwait` 2.4, `ttmop` 1.4, `ttseminit` 1, `ttrmwcib0` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.8, `ttmovd2b` 0.8

TT: `SETC16` 1, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttstallwait` 7, `ttrmwcib0` 4, `ttrmwcib1` 4, `ttrmwcib2` 3, `ttsetc16` 3, `ttrmwcib3` 2, `ttsetadcxx` 2, `ttsetadcxy` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB0` 7.4, `RMWCIB1` 7, `RMWCIB2` 4, `RMWCIB3` 4, `ZEROACC` 1
