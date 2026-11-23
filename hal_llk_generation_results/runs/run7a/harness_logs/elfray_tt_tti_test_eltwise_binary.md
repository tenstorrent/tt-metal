# TT / TTI (elfray) — 20261104_211905_B_c0e38f1f_claude-opus-5-5 / test_eltwise_binary

Variants: original 10, HAL 10. Static count (in the binary), mean per variant. TT elfray = stores that elfray marks (`tensix issue`), a lower bound; TT estimate = all stores into the instruction buffer found by tracking the register that holds the buffer address (0xffe40000). Per opcode (below) only the stores elfray marks.

| Kernel | Instructions total (orig / HAL) | TTI (orig / HAL) | TT elfray, lower bound (orig / HAL) | TT estimate (orig / HAL) | TTI + TT estimate (orig / HAL) |
|---|---|---|---|---|---|
| unpack | 808.8 / 436.4 | 21 / 44 | 11 / 33 | 11 / 36 | 32 / 80 |
| math | 536.6 / 208.8 | 36 / 32.4 | 8 / 2 | 8.8 / 3.2 | 44.8 / 35.6 |
| pack | 670.6 / 339.4 | 30 / 35 | 14 / 16 | 20 / 25 | 50 / 60 |

**unpack, original** — TTI: `ttsetadcxx` 6, `ttsetc16` 4, `ttsetadczw` 2, `ttstallwait` 2, `ttmop` 2, `ttsemget` 2, `ttsetadcxy` 1, `ttatgetm` 1, `ttatrelm` 1

TT: `RMWCIB2` 4, `RMWCIB0` 2, `RMWCIB1` 2, `SETDMAREG` 2, `RMWCIB3` 1

**unpack, hal** — TTI: `ttrmwcib1` 11, `ttrmwcib0` 7, `ttrmwcib2` 7, `ttrmwcib3` 7, `ttstallwait` 3, `ttsetadczw` 2, `ttatgetm` 2, `ttatrelm` 2, `ttsetadcxy` 1, `ttsetc16` 1, `ttmop` 1

TT: `RMWCIB2` 9, `RMWCIB3` 9, `RMWCIB0` 9, `RMWCIB1` 5, `SETADCXX` 1

**math, original** — TTI: `ttsetc16` 16.6, `ttstallwait` 5.6, `ttsetrwc` 4.8, `ttmop` 2.8, `ttsemwait` 2.2, `ttsempost` 2.2, `ttseminit` 1, `ttmovd2b` 0.4, `ttmovd2a` 0.4

TT: `RMWCIB0` 3, `RMWCIB3` 3, `RMWCIB1` 1, `SETC16` 1

**math, hal** — TTI: `ttsetc16` 20.2, `ttsetrwc` 3, `ttstallwait` 2.2, `ttmop` 1.2, `ttseminit` 1, `ttatgetm` 1, `ttatrelm` 1, `ttsemwait` 1, `ttsempost` 1, `ttmovd2a` 0.4, `ttmovd2b` 0.4

TT: `RMWCIB3` 1, `RMWCIB0` 1

**pack, original** — TTI: `ttstallwait` 6, `ttdmanop` 5, `ttwrcfg` 3, `ttsetc16` 3, `ttnop` 2, `ttsetdmareg` 2, `ttsetadczw` 2, `ttatgetm` 1, `ttatrelm` 1, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `SETDMAREG` 4, `RMWCIB2` 4, `RMWCIB3` 2, `RMWCIB1` 2, `RMWCIB0` 1, `WRCFG` 1

**pack, hal** — TTI: `ttstallwait` 6, `ttrmwcib2` 6, `ttrmwcib0` 5, `ttrmwcib1` 5, `ttrmwcib3` 4, `ttsetc16` 3, `ttsetadcxx` 1, `ttsetadcxy` 1, `ttsetadczw` 1, `ttsemwait` 1, `ttmop` 1, `ttsemget` 1

TT: `RMWCIB0` 5, `RMWCIB2` 4, `RMWCIB1` 4, `RMWCIB3` 3
