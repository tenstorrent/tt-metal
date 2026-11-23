# TT / TTI (elfray) — 20261121_205834_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary file), average per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores that elfray marks are listed.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 469.2 | 21 / 78 | 11 / 26 | 11 / 29 | 32 / 107 |
| math | 536.6 / 212.8 | 36 / 20.6 | 8 / 2 | 8.8 / 4.2 | 44.8 / 24.8 |
| pack | 670.6 / 347.2 | 30 / 56 | 14 / 13 | 20 / 16 | 50 / 72 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 22, `ttrmwcib0` 19, `ttrmwcib2` 16, `ttrmwcib3` 8, `ttsetc16` 4, `ttstallwait` 2, `ttsetadczw` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1

TT: `RMWCIB3` 8, `RMWCIB0` 7, `RMWCIB2` 7, `RMWCIB1` 3, `SETADCXX` 1

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 9.6, `ttsetrwc` 2.8, `ttstallwait` 2.2, `ttmop` 1.2, `ttseminit` 1, `ttrmwcib0` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `SETC16` 1, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttsetdmareg` 16, `ttstallwait` 8, `ttrmwcib0` 5, `ttrmwcib1` 5, `ttrmwcib2` 3, `ttrmwcib3` 3, `ttsetc16` 3, `ttwrcfg` 3, `ttsemget` 3, `ttzeroacc` 2, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttmop` 1

TT: `RMWCIB2` 4, `RMWCIB3` 3, `RMWCIB0` 3, `RMWCIB1` 3
