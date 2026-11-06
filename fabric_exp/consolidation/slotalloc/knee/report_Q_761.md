| arm | row | 256 B cycles (fraction of full) | 4352 B cycles (fraction) | pass spread |
| --- | --- | --- | --- | --- |
| full | Sls1HopEW | 114.7 (1.000) | 125.8 (1.000) | 0.2 |
| full | Sls1HopEWBidir | 114.8 (1.000) | 126.6 (1.000) | 0.2 |
| full | Sls3HopEW | 145.2 (1.000) | 155.2 (1.000) | 0.2 |
| full | SlsTurn4 | 145.8 (1.000) | 160.1 (1.000) | 0.2 |
| w10 | Sls1HopEW | 184.4 (0.622) | 202.1 (0.623) | 1.3 |
| w10 | Sls1HopEWBidir | 184.4 (0.623) | 212.8 (0.595) | 1.3 |
| r8 | Sls1HopEW | 157.4 (0.728) | 179.7 (0.700) | 0.2 |
| r8 | Sls1HopEWBidir | 157.3 (0.730) | 190.3 (0.666) | 0.2 |
| w10r8 | Sls1HopEW | 167.2 (0.686) | 180.7 (0.696) | 0.2 |
| w10r8 | Sls1HopEWBidir | 166.9 (0.688) | 192.4 (0.658) | 0.2 |
| w14r4 | Sls1HopEW | 270.0 (0.425) | 332.6 (0.378) | 0.2 |
| w14r4 | Sls1HopEWBidir | 269.7 (0.426) | 345.6 (0.366) | 0.2 |
| f8 | Sls3HopEW | 239.9 (0.605) | 266.8 (0.582) | 0.4 |
| f8 | SlsTurn4 | 253.1 (0.576) | 274.0 (0.584) | 0.4 |
| f8r8 | Sls3HopEW | 216.9 (0.670) | 245.2 (0.633) | 0.2 |
| f8r8 | SlsTurn4 | 224.0 (0.651) | 249.6 (0.641) | 0.2 |
| w10f8r8 | Sls1HopEW | 167.3 (0.685) | 180.7 (0.696) | 0.3 |
| w10f8r8 | Sls1HopEWBidir | 167.0 (0.687) | 192.3 (0.659) | 0.3 |
| w10f8r8 | Sls3HopEW | 216.9 (0.670) | 245.3 (0.632) | 0.3 |
| w10f8r8 | SlsTurn4 | 224.1 (0.651) | 249.8 (0.641) | 0.3 |
| r8q3 | Sls3HopEW | 167.8 (0.865) | 185.9 (0.835) | 0.3 |
| r8q3 | SlsTurn4 | 168.2 (0.866) | 186.4 (0.859) | 0.3 |
| w10q3 | Sls3HopEW | 192.2 (0.756) | 206.9 (0.750) | 0.1 |
| w10q3 | SlsTurn4 | 193.5 (0.753) | 207.3 (0.772) | 0.1 |

Pairs: measured fraction against min(single fractions) and their product
| pair | row | size | single fractions | both squeezed | min | product |
| --- | --- | --- | --- | --- | --- | --- |
| w10r8 = w10 + r8 | Sls1HopEW | 256 | 0.622, 0.728 | 0.686 | 0.622 | 0.453 |
| w10r8 = w10 + r8 | Sls1HopEW | 4352 | 0.623, 0.700 | 0.696 | 0.623 | 0.436 |
| w10r8 = w10 + r8 | Sls1HopEWBidir | 256 | 0.623, 0.730 | 0.688 | 0.623 | 0.454 |
| w10r8 = w10 + r8 | Sls1HopEWBidir | 4352 | 0.595, 0.666 | 0.658 | 0.595 | 0.396 |
| f8r8 = f8 + r8q3 | Sls3HopEW | 256 | 0.605, 0.865 | 0.670 | 0.605 | 0.524 |
| f8r8 = f8 + r8q3 | Sls3HopEW | 4352 | 0.582, 0.835 | 0.633 | 0.582 | 0.485 |
| f8r8 = f8 + r8q3 | SlsTurn4 | 256 | 0.576, 0.866 | 0.651 | 0.576 | 0.499 |
| f8r8 = f8 + r8q3 | SlsTurn4 | 4352 | 0.584, 0.859 | 0.641 | 0.584 | 0.502 |
| w10f8r8 = f8r8 + w10q3 | Sls3HopEW | 256 | 0.670, 0.756 | 0.670 | 0.670 | 0.506 |
| w10f8r8 = f8r8 + w10q3 | Sls3HopEW | 4352 | 0.633, 0.750 | 0.632 | 0.633 | 0.474 |
| w10f8r8 = f8r8 + w10q3 | SlsTurn4 | 256 | 0.651, 0.753 | 0.651 | 0.651 | 0.490 |
| w10f8r8 = f8r8 + w10q3 | SlsTurn4 | 4352 | 0.641, 0.772 | 0.641 | 0.641 | 0.495 |

Unequal against equal at the same total (worker + receiver 18 slots): w14r4 vs w10r8
- Sls1HopEW 256 B: equal 0.686, unequal 0.425 (-38.1% rate)
- Sls1HopEW 4352 B: equal 0.696, unequal 0.378 (-45.7% rate)
- Sls1HopEWBidir 256 B: equal 0.688, unequal 0.426 (-38.1% rate)
- Sls1HopEWBidir 4352 B: equal 0.658, unequal 0.366 (-44.3% rate)
