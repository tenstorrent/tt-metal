# TT / TTI (elfray) — 20261119_163437_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 504 | 21 / 66 | 11 / 29 | 11 / 33 | 32 / 99 |
| math | 536.6 / 225.6 | 36 / 30 | 8 / 3 | 8.8 / 5.2 | 44.8 / 35.2 |
| pack | 670.6 / 346.2 | 30 / 70 | 14 / 12.4 | 20 / 17 | 50 / 87 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 18, `ttrmwcib0` 15, `ttrmwcib2` 13, `ttrmwcib3` 7, `ttsetc16` 3, `ttstallwait` 2, `ttsetadc` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttsetadczw` 1

TT: `RMWCIB3` 9, `RMWCIB0` 8, `RMWCIB2` 7, `RMWCIB1` 4, `SETADC` 1

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 20.4, `ttstallwait` 3.2, `ttsetrwc` 1.4, `ttmop` 1.2, `ttseminit` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `SETC16` 1, `RMWCIB3` 1, `RMWCIB0` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttsetdmareg` 16, `ttrmwcib0` 9, `ttrmwcib1` 9, `ttrmwcib2` 8, `ttrmwcib3` 7, `ttstallwait` 4, `ttwrcfg` 4, `ttsetc16` 3, `ttsetadc` 2, `ttzeroacc` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttmop` 1

TT: `RMWCIB0` 3.4, `RMWCIB2` 3, `RMWCIB3` 3, `RMWCIB1` 3
