# TT / TTI (elfray) — 20261105_102435_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 478 | 21 / 56 | 11 / 36 | 11 / 40 | 32 / 96 |
| math | 536.6 / 272 | 36 / 25 | 8 / 9.8 | 8.8 / 13.8 | 44.8 / 38.8 |
| pack | 670.6 / 403.2 | 30 / 42 | 14 / 27.4 | 20 / 42 | 50 / 84 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 13, `ttrmwcib0` 12, `ttrmwcib2` 11, `ttrmwcib3` 5, `ttsetadcxx` 5, `ttstallwait` 3, `ttsetc16` 3, `ttsetadczw` 2, `ttsetadcxy` 1, `ttmop` 1

TT: `RMWCIB2` 10, `RMWCIB3` 10, `RMWCIB0` 9, `RMWCIB1` 7

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 10.8, `ttstallwait` 4.4, `ttsetrwc` 2.4, `ttmop` 2.4, `ttrmwcib0` 1.2, `ttseminit` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `SETC16` 8.8, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib0` 11, `ttrmwcib1` 7, `ttrmwcib2` 6, `ttrmwcib3` 5, `ttstallwait` 4, `ttsetc16` 2, `ttsetadczw` 2, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB1` 11, `RMWCIB0` 7.4, `RMWCIB2` 4, `RMWCIB3` 4, `ZEROACC` 1
