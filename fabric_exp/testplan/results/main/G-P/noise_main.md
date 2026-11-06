# Pass-to-pass noise: main, router main

| group/step | kind | key | per pass | spread |
| --- | --- | --- | --- | ---: |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_WRITE Torus [32] 2 4096 | 18.97 / 18.82 | 0.81% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_FUSED_UNICAST_ATOMIC_INC Torus [32] 2 4096 | 18.18 / 18.27 | 0.46% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_WRITE Torus [32] 1 4096 | 23.18 / 23.21 | 0.14% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_FUSED_UNICAST_ATOMIC_INC Torus [32] 1 2048 | 11.1 / 11.11 | 0.09% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_SCATTER_WRITE Torus [32] 1 2048 | 10.6 / 10.59 | 0.07% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_SCATTER_WRITE Torus [32] 1 4096 | 21.18 / 21.17 | 0.04% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_SCATTER_WRITE Torus [32] 2 2048 | 10.59 / 10.59 | 0.04% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_FUSED_UNICAST_ATOMIC_INC Torus [32] 1 4096 | 21.21 / 21.21 | 0.03% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_SCATTER_WRITE Torus [32] 2 4096 | 18.91 / 18.91 | 0.03% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_FUSED_UNICAST_ATOMIC_INC Torus [32] 2 2048 | 11.14 / 11.14 | 0.03% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_WRITE Torus [32] 2 2048 | 13.18 / 13.18 | 0.01% |
| G-P/G0-bench | fabric | bw TorusAllToAllMulticast CHIP_MULTICAST NOC_UNICAST_WRITE Torus [32] 1 2048 | 13.21 / 13.21 | 0.00% |

12 rows with >= 2 passes: median spread 0.04%, max 0.81%
