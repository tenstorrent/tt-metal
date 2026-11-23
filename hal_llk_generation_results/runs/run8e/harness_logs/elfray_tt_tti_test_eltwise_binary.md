# TT / TTI (elfray) — 20261122_005334_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary file), average per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores that elfray marks are listed.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 429.2 | 21 / 36 | 11 / 25 | 11 / 29 | 32 / 65 |
| math | 536.6 / 238.3 | 36 / 22.2 | 8 / 6.8 | 8.8 / 10.1 | 44.8 / 32.3 |
| pack | 670.6 / 345 | 30 / 66 | 14 / 14 | 20 / 19 | 50 / 85 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 12, `ttrmwcib0` 8, `ttrmwcib2` 3, `ttrmwcib3` 3, `ttsetc16` 3, `ttstallwait` 2, `ttsetadczw` 2, `ttsetadcxy` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB3` 8, `RMWCIB0` 7, `RMWCIB2` 6, `RMWCIB1` 3, `SETADCXX` 1

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 7.4, `ttstallwait` 5.4, `ttsetrwc` 2.8, `ttrmwcib0` 2.2, `ttsemwait` 1.2, `ttmop` 1.2, `ttseminit` 1, `ttsempost` 1

TT: `SETC16` 5.8, `RMWCIB3` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttrmwcib0` 17, `ttrmwcib1` 17, `ttsetc16` 9, `ttstallwait` 6, `ttrmwcib2` 4, `ttrmwcib3` 2, `ttsetadczw` 2, `ttzeroacc` 2, `ttsemget` 2, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttwrcfg` 1, `ttmop` 1

TT: `RMWCIB0` 5, `RMWCIB1` 4, `RMWCIB2` 3, `RMWCIB3` 2
