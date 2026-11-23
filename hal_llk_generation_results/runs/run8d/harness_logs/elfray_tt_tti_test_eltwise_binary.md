# TT / TTI (elfray) — 20261122_005259_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary file), average per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores that elfray marks are listed.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 505 | 21 / 67 | 11 / 29 | 11 / 33 | 32 / 100 |
| math | 536.6 / 203.2 | 36 / 29.6 | 8 / 1 | 8.8 / 2.2 | 44.8 / 31.8 |
| pack | 670.6 / 350.4 | 30 / 55 | 14 / 19 | 20 / 23 | 50 / 78 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttsetadc` 20, `ttrmwcib1` 9, `ttrmwcib0` 7, `ttrmwcib2` 7, `ttstallwait` 6, `ttsetc16` 5, `ttrmwcib3` 3, `ttmop` 3, `ttsemget` 3, `ttsetadcxy` 2, `ttseminit` 1, `ttsetadczw` 1

TT: `RMWCIB3` 9, `RMWCIB0` 8, `RMWCIB2` 7, `RMWCIB1` 5

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 20, `ttstallwait` 2.2, `ttmop` 1.4, `ttsetrwc` 1.2, `ttseminit` 1, `ttrmwcib0` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib0` 11, `ttrmwcib2` 10, `ttrmwcib1` 8, `ttrmwcib3` 7, `ttstallwait` 4, `ttsetc16` 3, `ttsetadc` 2, `ttzeroacc` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttwrcfg` 1, `ttnop` 1, `ttmop` 1

TT: `RMWCIB1` 7, `RMWCIB2` 4, `RMWCIB3` 4, `RMWCIB0` 4
