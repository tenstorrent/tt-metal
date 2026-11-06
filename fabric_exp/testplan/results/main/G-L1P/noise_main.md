# Pass-to-pass noise: main, router main

| group/step | kind | key | per pass | spread |
| --- | --- | --- | --- | ---: |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=6875 slots=1 | 0.349 / 0.362 | 3.66% |
| G-L1P/G1-hbw | metric | cap=1720 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=1720 slots=1 | 0.513 / 0.526 | 2.50% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=6875 slots=1 _n=2 | 2.035 / 2.085 | 2.43% |
| G-L1P/G1-hbw | metric | cap=1720 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=1720 slots=1 | 0.376 / 0.384 | 2.11% |
| G-L1P/G1-hbw | metric | cap=1720 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=1720 slots=1 | 0.527 / 0.534 | 1.32% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=6875 slots=1 | 0.403 / 0.407 | 0.99% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=6875 slots=4 | 0.38 / 0.383 | 0.79% |
| G-L1P/G1-hbw | metric | cap=31250 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=16384 slots=2 | 4.205 / 4.236 | 0.73% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=6875 slots=1 | 0.63 / 0.634 | 0.63% |
| G-L1P/G1-hbw | metric | cap=16384 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=16384 slots=1 | 1.838 / 1.848 | 0.54% |
| G-L1P/G1-hbw | metric | cap=16384 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=16384 slots=1 | 3.293 / 3.276 | 0.52% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=6875 slots=1 _n=2 | 1.285 / 1.29 | 0.39% |
| G-L1P/G1-hbw | metric | cap=16384 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=16384 slots=1 | 2.087 / 2.08 | 0.34% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=6875 slots=1 | 0.63 / 0.632 | 0.32% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=6875 slots=4 | 0.338 / 0.337 | 0.30% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=65536 slots=1 | 3.435 / 3.445 | 0.29% |
| G-L1P/G1-hbw | metric | cap=1720 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=1720 slots=1 | 0.4 / 0.401 | 0.25% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=65536 slots=1 _n=2 | 12.62 / 12.59 | 0.20% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=6875 slots=4 | 0.609 / 0.608 | 0.16% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=6875 slots=4 | 0.61 / 0.611 | 0.16% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=65536 slots=1 | 5.598 / 5.589 | 0.16% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=65536 slots=1 _n=2 | 7.776 / 7.787 | 0.14% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=65536 slots=1 _n=2 | 6.781 / 6.772 | 0.13% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=65536 slots=4 | 2.976 / 2.973 | 0.10% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=65536 slots=1 | 12.66 / 12.67 | 0.09% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=65536 slots=1 | 2.997 / 2.999 | 0.07% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=6875 slots=1 _n=2 | 1.825 / 1.826 | 0.05% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=2048B rows=65536 slots=4 | 5.577 / 5.575 | 0.04% |
| G-L1P/G1-hbw | metric | cap=125000 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=65536 slots=2 | 5.607 / 5.609 | 0.04% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.BFLOAT16 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=1152B rows=65536 slots=4 | 5.586 / 5.587 | 0.02% |
| G-L1P/G1-hbw | metric | cap=6875 dtype=DataType.BFLOAT8_B fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.TILE links=2 page=1088B rows=6875 slots=1 _n=2 | 1.043 / 1.043 | 0.00% |
| G-L1P/G1-hbw | metric | cap=65536 dtype=DataType.FP8_E4M3 fabric=FabricConfig.FABRIC_2D_TORUS_XY layout=Layout.ROW_MAJOR links=2 page=704B rows=65536 slots=4 | 3.421 / 3.421 | 0.00% |

32 rows with >= 2 passes: median spread 0.29%, max 3.66%
