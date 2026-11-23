# TT / TTI (elfray) — 20261105_122703_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 479 | 21 / 37 | 11 / 45 | 11 / 52 | 32 / 89 |
| math | 536.6 / 273 | 36 / 17.8 | 8 / 22.4 | 8.8 / 27.8 | 44.8 / 45.6 |
| pack | 670.6 / 416.8 | 30 / 63 | 14 / 22.6 | 20 / 36 | 50 / 99 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 9, `ttrmwcib0` 7, `ttrmwcib2` 5, `ttstallwait` 3, `ttrmwcib3` 3, `ttsetadczw` 2, `ttatgetm` 2, `ttatrelm` 2, `ttsetc16` 2, `ttsetadcxy` 1, `ttmop` 1

TT: `RMWCIB0` 13, `RMWCIB3` 12, `RMWCIB2` 11, `RMWCIB1` 9

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttatgetm` 3.4, `ttstallwait` 3.2, `ttsetrwc` 2.8, `ttatrelm` 2.2, `ttsemwait` 1.2, `ttmop` 1.2, `ttseminit` 1, `ttsetc16` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `SETC16` 20.2, `RMWCIB0` 1.2, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib2` 15, `ttrmwcib0` 9, `ttrmwcib1` 9, `ttstallwait` 8, `ttrmwcib3` 7, `ttsetc16` 3, `ttsetadcxy` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsemwait` 1, `ttwrcfg` 1, `ttpacr` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB0` 7, `RMWCIB1` 7, `RMWCIB2` 4.6, `RMWCIB3` 4
