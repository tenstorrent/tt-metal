# TT / TTI (elfray) — 20261122_005225_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary file), average per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores that elfray marks are listed.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 443.6 | 21 / 48.5 | 11 / 26 | 11 / 29 | 32 / 77.5 |
| math | 536.6 / 236.5 | 36 / 27.8 | 8 / 8.9 | 8.8 / 13.6 | 44.8 / 41.4 |
| pack | 670.6 / 385.2 | 30 / 73 | 14 / 12.4 | 20 / 17 | 50 / 90 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 13, `ttrmwcib0` 8, `ttrmwcib2` 8, `ttrmwcib3` 6, `ttsetc16` 4, `ttstallwait` 3.5, `ttsetadczw` 2, `ttmop` 1.5, `ttsemget` 1.5, `ttsetadcxy` 1

TT: `RMWCIB3` 7, `RMWCIB0` 7, `RMWCIB2` 6, `RMWCIB1` 3, `SETADCXX` 3

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 13.8, `ttstallwait` 5.4, `ttsetrwc` 2.4, `ttrmwcib0` 1.2, `ttmop` 1.2, `ttseminit` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `SETC16` 7.9, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib0` 17, `ttrmwcib1` 17, `ttsetc16` 12, `ttstallwait` 9, `ttrmwcib2` 4, `ttrmwcib3` 3, `ttsetadcxy` 2, `ttzeroacc` 2, `ttsemget` 2, `ttsetadcxx` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttwrcfg` 1, `ttmop` 1

TT: `RMWCIB0` 3.4, `RMWCIB2` 3, `RMWCIB3` 3, `RMWCIB1` 3
