# Regime-A critical-path ablation matrix

Test-only compile-gated, program-cache-hashed 6-bit diagnostic (`diag_mask`, TT_REGIME_A_DIAG_MASK env). Public API + mask-0 binaries unchanged. Bits: 1=in0_all, 2=in0_redun, 4=ring_fwd, 8=compute, 16=reduction, 32=output. Skips preserve all unaffected CB reserve/push/pop, pointers, waits, semaphores, loop structure; outputs for masks!=0 are intentionally invalid (PCC only for baseline).

Modes: baseline + 6 singles + 15 pairs (22). One persistent session/relaunch; 2 warmup + 12 timed iters/mode; >=2 relaunches with mode order reversed on odd relaunches; a 3rd added when a delta is near noise or relaunch distributions overlap. Kernel wall + per-RISC via run-host-id demux. `gain(S)=Tbase-T(S)`; `interaction(A,B)=gain(A+B)-gain(A)-gain(B)`. Commit + diagnostic; BH p150b, 1.35 GHz, fw 19.5.0; peak DRAM 512 GB/s. **Single-ablation %s are NOT summed as a forecast.**

## 256x2048x2048  cfg (Ns,Pk,Sm,kb,nsb)=(2, 2, 3, 4, 4)  baseline 36.85 us  (PCC 0.999995, replay True, 3 relaunches)

### Baseline + singles

| mode | median us | gain us | gain % | IQR | spread% | crit RISC | B/N/T us |
|---|---|---|---|---|---|---|---|
| baseline | 36.85 | +0.00 | +0.0% | 0.93 | 8.9 | BRISC | 36.7/35.64/34.93 |
| in0_all | 33.78 | +3.07 | +8.3% | 1.45 | 8.7 | NCRISC | 32.78/33.6/32.73 |
| in0_redun | 36.30 | +0.56 | +1.5% | 1.14 | 6.9 | BRISC | 36.21/33.53/34.48 |
| ring_fwd | 32.01 | +4.84 | +13.1% | 0.57 | 7.3 | BRISC | 32.14/29.94/30.4 |
| compute | 35.27 | +1.58 | +4.3% | 0.97 | 7.3 | BRISC | 35.24/34.27/33.67 |
| reduction | 38.76 | -1.90 | -5.2% | 1.47 | 12.4 | BRISC | 38.79/36.85/37.43 |
| output | 35.15 | +1.71 | +4.6% | 0.87 | 9.2 | NCRISC | 34.51/35.02/34.9 |

Theoretical DRAM us @512GB/s: in0_all=4.1, in0_redun=2.05, in1=16.38, output=2.05.

### Pair-interaction matrix (us)  `interaction=gain(A+B)-gain(A)-gain(B)`

| A\B | in0_all | in0_redun | ring_fwd | compute | reduction | output |
|---|---|---|---|---|---|---|
| in0_all | · | -0.08 | -0.17 | +0.24 | +0.36 | +0.19 |
| in0_redun | -0.08 | · | -0.78 | -0.04 | -0.86 | +0.20 |
| ring_fwd | -0.17 | -0.78 | · | -0.46 | -2.16 | +0.24 |
| compute | +0.24 | -0.04 | -0.46 | · | -1.01 | +0.38 |
| reduction | +0.36 | -0.86 | -2.16 | -1.01 | · | +2.35 |
| output | +0.19 | +0.20 | +0.24 | +0.38 | +2.35 | · |

### Fastest combinations

| rank | mode | median us | vs baseline |
|---|---|---|---|
| 1 | in0_all+ring_fwd | 29.11 | +21.0% |
| 2 | ring_fwd+output | 30.07 | +18.4% |
| 3 | ring_fwd+compute | 30.89 | +16.2% |
| 4 | in0_all+output | 31.89 | +13.5% |
| 5 | in0_all+compute | 31.96 | +13.3% |
| 6 | ring_fwd | 32.01 | +13.1% |

### Critical-RISC transitions (which RISC bounds the wall)

- baseline critical RISC: **BRISC** (B/N/T = 36.7/35.64/34.93 us)
- in0_all shifts the critical RISC to **NCRISC**
- output shifts the critical RISC to **NCRISC**

### Interpretation

- **in0_all**: gain +3.07 us (+8.3%) (theo DRAM 4.1 us; exposed ~0.75) -> exposed.
- **in0_redun**: gain +0.56 us (+1.5%) (theo DRAM 2.05 us; exposed ~0.27) -> hidden.
- **ring_fwd**: gain +4.84 us (+13.1%) -> exposed.
- **compute**: gain +1.58 us (+4.3%) -> exposed.
- **reduction**: gain -1.90 us (-5.2%) -> NEGATIVE (removing work worsened phasing).
- **output**: gain +1.71 us (+4.6%) (theo DRAM 2.05 us; exposed ~0.83) -> exposed.

## 256x2048x6144  cfg (Ns,Pk,Sm,kb,nsb)=(3, 2, 2, 2, 4)  baseline 84.63 us  (PCC 1.00006, replay True, 2 relaunches)

### Baseline + singles

| mode | median us | gain us | gain % | IQR | spread% | crit RISC | B/N/T us |
|---|---|---|---|---|---|---|---|
| baseline | 84.63 | +0.00 | +0.0% | 2.33 | 7.3 | NCRISC | 83.79/84.57/83.55 |
| in0_all | 77.75 | +6.88 | +8.1% | 1.75 | 9.1 | NCRISC | 77.17/77.47/76.69 |
| in0_redun | 79.38 | +5.25 | +6.2% | 3.31 | 8.8 | NCRISC | 79.0/79.05/78.28 |
| ring_fwd | 81.73 | +2.90 | +3.4% | 1.52 | 4.6 | BRISC | 81.63/80.9/80.52 |
| compute | 81.61 | +3.02 | +3.6% | 1.65 | 5.0 | NCRISC | 80.84/81.47/80.73 |
| reduction | 108.34 | -23.71 | -28.0% | 1.52 | 3.5 | NCRISC | 105.67/108.39/107.39 |
| output | 69.98 | +14.65 | +17.3% | 1.21 | 3.5 | BRISC | 69.88/69.29/69.84 |

Theoretical DRAM us @512GB/s: in0_all=6.14, in0_redun=4.1, in1=49.15, output=6.14.

### Pair-interaction matrix (us)  `interaction=gain(A+B)-gain(A)-gain(B)`

| A\B | in0_all | in0_redun | ring_fwd | compute | reduction | output |
|---|---|---|---|---|---|---|
| in0_all | · | -4.93 | +0.45 | +0.14 | -0.77 | +0.11 |
| in0_redun | -4.93 | · | -0.05 | -0.10 | +1.49 | +1.02 |
| ring_fwd | +0.45 | -0.05 | · | -1.85 | +4.70 | +1.84 |
| compute | +0.14 | -0.10 | -1.85 | · | +2.04 | -0.38 |
| reduction | -0.77 | +1.49 | +4.70 | +2.04 | · | +23.65 |
| output | +0.11 | +1.02 | +1.84 | -0.38 | +23.65 | · |

### Fastest combinations

| rank | mode | median us | vs baseline |
|---|---|---|---|
| 1 | in0_all+output | 62.99 | +25.6% |
| 2 | in0_redun+output | 63.71 | +24.7% |
| 3 | ring_fwd+output | 65.24 | +22.9% |
| 4 | compute+output | 67.34 | +20.4% |
| 5 | output | 69.98 | +17.3% |
| 6 | reduction+output | 70.05 | +17.2% |

### Critical-RISC transitions (which RISC bounds the wall)

- baseline critical RISC: **NCRISC** (B/N/T = 83.79/84.57/83.55 us)
- ring_fwd shifts the critical RISC to **BRISC**
- output shifts the critical RISC to **BRISC**

### Interpretation

- **in0_all**: gain +6.88 us (+8.1%) (theo DRAM 6.14 us; exposed ~1.12) -> exposed.
- **in0_redun**: gain +5.25 us (+6.2%) (theo DRAM 4.1 us; exposed ~1.28) -> exposed.
- **ring_fwd**: gain +2.90 us (+3.4%) -> exposed.
- **compute**: gain +3.02 us (+3.6%) -> exposed.
- **reduction**: gain -23.71 us (-28.0%) -> NEGATIVE (removing work worsened phasing).
- **output**: gain +14.65 us (+17.3%) (theo DRAM 6.14 us; exposed ~2.39) -> exposed.

## 512x6144x2304  cfg (Ns,Pk,Sm,kb,nsb)=(2, 6, 1, 2, 1)  baseline 170.36 us  (PCC 1.000024, replay True, 3 relaunches)

### Baseline + singles

| mode | median us | gain us | gain % | IQR | spread% | crit RISC | B/N/T us |
|---|---|---|---|---|---|---|---|
| baseline | 170.36 | +0.00 | +0.0% | 2.97 | 5.2 | BRISC | 170.73/166.0/169.84 |
| in0_all | 145.33 | +25.03 | +14.7% | 4.66 | 6.1 | BRISC | 145.91/141.05/145.01 |
| in0_redun | 164.02 | +6.34 | +3.7% | 3.86 | 9.8 | NCRISC | 159.26/163.88/163.67 |
| ring_fwd | 117.65 | +52.71 | +30.9% | 0.87 | 2.8 | BRISC | 117.66/116.98/117.17 |
| compute | 149.14 | +21.22 | +12.5% | 1.66 | 4.8 | BRISC | 149.3/146.24/148.39 |
| reduction | 207.64 | -37.28 | -21.9% | 8.89 | 9.8 | BRISC | 207.02/204.51/205.75 |
| output | 169.64 | +0.72 | +0.4% | 3.33 | 4.0 | BRISC | 169.64/166.06/169.56 |

Theoretical DRAM us @512GB/s: in0_all=24.58, in0_redun=12.29, in1=55.3, output=4.61.

### Pair-interaction matrix (us)  `interaction=gain(A+B)-gain(A)-gain(B)`

| A\B | in0_all | in0_redun | ring_fwd | compute | reduction | output |
|---|---|---|---|---|---|---|
| in0_all | · | -6.29 | +4.56 | -0.69 | -1.71 | -0.70 |
| in0_redun | -6.29 | · | -6.95 | +3.75 | -2.13 | -0.99 |
| ring_fwd | +4.56 | -6.95 | · | -6.56 | -21.91 | +0.08 |
| compute | -0.69 | +3.75 | -6.56 | · | +7.16 | +3.65 |
| reduction | -1.71 | -2.13 | -21.91 | +7.16 | · | +41.76 |
| output | -0.70 | -0.99 | +0.08 | +3.65 | +41.76 | · |

### Fastest combinations

| rank | mode | median us | vs baseline |
|---|---|---|---|
| 1 | in0_all+ring_fwd | 88.06 | +48.3% |
| 2 | ring_fwd+compute | 102.99 | +39.5% |
| 3 | ring_fwd+output | 116.85 | +31.4% |
| 4 | ring_fwd | 117.65 | +30.9% |
| 5 | in0_redun+ring_fwd | 118.26 | +30.6% |
| 6 | in0_all+compute | 124.80 | +26.7% |

### Critical-RISC transitions (which RISC bounds the wall)

- baseline critical RISC: **BRISC** (B/N/T = 170.73/166.0/169.84 us)
- in0_redun shifts the critical RISC to **NCRISC**

### Interpretation

- **in0_all**: gain +25.03 us (+14.7%) (theo DRAM 24.58 us; exposed ~1.02) -> exposed.
- **in0_redun**: gain +6.34 us (+3.7%) (theo DRAM 12.29 us; exposed ~0.52) -> exposed.
- **ring_fwd**: gain +52.71 us (+30.9%) -> exposed.
- **compute**: gain +21.22 us (+12.5%) -> exposed.
- **reduction**: gain -37.28 us (-21.9%) -> NEGATIVE (removing work worsened phasing).
- **output**: gain +0.72 us (+0.4%) (theo DRAM 4.61 us; exposed ~0.16) -> hidden.

## 512x6144x4608  cfg (Ns,Pk,Sm,kb,nsb)=(2, 6, 1, 4, 1)  baseline 224.06 us  (PCC 1.000072, replay True, 3 relaunches)

### Baseline + singles

| mode | median us | gain us | gain % | IQR | spread% | crit RISC | B/N/T us |
|---|---|---|---|---|---|---|---|
| baseline | 224.06 | +0.00 | +0.0% | 2.52 | 2.8 | BRISC | 223.6/220.12/222.76 |
| in0_all | 199.97 | +24.10 | +10.8% | 1.89 | 4.8 | BRISC | 199.97/192.82/199.04 |
| in0_redun | 229.27 | -5.21 | -2.3% | 5.00 | 6.5 | NCRISC | 221.35/229.42/228.47 |
| ring_fwd | 182.56 | +41.51 | +18.5% | 1.84 | 2.9 | NCRISC | 178.8/182.38/181.44 |
| compute | 202.80 | +21.26 | +9.5% | 3.01 | 4.0 | NCRISC | 201.15/202.84/201.88 |
| reduction | 347.77 | -123.71 | -55.2% | 14.36 | 12.2 | NCRISC | 340.94/346.69/345.73 |
| output | 222.95 | +1.11 | +0.5% | 1.85 | 3.4 | BRISC | 223.03/217.98/223.0 |

Theoretical DRAM us @512GB/s: in0_all=24.58, in0_redun=12.29, in1=110.59, output=9.22.

### Pair-interaction matrix (us)  `interaction=gain(A+B)-gain(A)-gain(B)`

| A\B | in0_all | in0_redun | ring_fwd | compute | reduction | output |
|---|---|---|---|---|---|---|
| in0_all | · | +5.68 | +14.21 | +1.22 | -0.03 | -0.33 |
| in0_redun | +5.68 | · | -1.89 | -0.49 | +2.90 | +0.80 |
| ring_fwd | +14.21 | -1.89 | · | -6.96 | -23.14 | +3.33 |
| compute | +1.22 | -0.49 | -6.96 | · | +11.38 | +13.49 |
| reduction | -0.03 | +2.90 | -23.14 | +11.38 | · | +128.62 |
| output | -0.33 | +0.80 | +3.33 | +13.49 | +128.62 | · |

### Fastest combinations

| rank | mode | median us | vs baseline |
|---|---|---|---|
| 1 | in0_all+ring_fwd | 144.24 | +35.6% |
| 2 | ring_fwd+compute | 168.26 | +24.9% |
| 3 | in0_all+compute | 177.49 | +20.8% |
| 4 | ring_fwd+output | 178.11 | +20.5% |
| 5 | ring_fwd | 182.56 | +18.5% |
| 6 | compute+output | 188.20 | +16.0% |

### Critical-RISC transitions (which RISC bounds the wall)

- baseline critical RISC: **BRISC** (B/N/T = 223.6/220.12/222.76 us)
- in0_redun shifts the critical RISC to **NCRISC**
- ring_fwd shifts the critical RISC to **NCRISC**
- compute shifts the critical RISC to **NCRISC**
- reduction shifts the critical RISC to **NCRISC**

### Interpretation

- **in0_all**: gain +24.10 us (+10.8%) (theo DRAM 24.58 us; exposed ~0.98) -> exposed.
- **in0_redun**: gain -5.21 us (-2.3%) (theo DRAM 12.29 us; exposed ~-0.42) -> NEGATIVE (removing work worsened phasing).
- **ring_fwd**: gain +41.51 us (+18.5%) -> exposed.
- **compute**: gain +21.26 us (+9.5%) -> exposed.
- **reduction**: gain -123.71 us (-55.2%) -> NEGATIVE (removing work worsened phasing).
- **output**: gain +1.11 us (+0.5%) (theo DRAM 9.22 us; exposed ~0.12) -> hidden.
