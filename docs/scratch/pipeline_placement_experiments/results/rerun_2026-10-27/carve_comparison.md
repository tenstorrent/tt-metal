### encodings

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8x8: 1x1, 14x 2x2, 1x1 | 1 x 8x8 | 14x 2x2, 2x 1x1 (ring) | 58/64 | feasible | yes 0.005 | yes 0.005 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| 8x8: 1x1, 15x 2x2, 3x 1x1 (fills) | 1 x 8x8 | 15x 2x2, 4x 1x1 (ring) | 64/64 | feasible | yes 0.256 | yes 0.026 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| 8x8 ring of 64 x 1x1 (fills) | 1 x 8x8 | 64x 1x1 (ring) | 64/64 | feasible | yes 0.014 | yes 0.014 | yes 0.004 | yes 0.013 | no: budget-exhausted | no: budget-exhausted | - | - |
| 8x16: 1x1, 30x 2x2, 1x1 | 1 x 8x16 | 30x 2x2, 2x 1x1 (ring) | 122/128 | feasible | yes 0.024 | yes 0.024 | yes 0.001 | yes 0.003 | no: budget-exhausted | no: budget-exhausted | - | - |
| 8x16: 1x1, 2x4 x14, 2x2 x3, 1x1 (fills) | 1 x 8x16 | 14x 2x4, 3x 2x2, 3x 1x1 (ring) | 127/128 | feasible | yes 2.83 | yes 2.78 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| 8x16 ring of 128 x 1x1 (fills) | 1 x 8x16 | 128x 1x1 (ring) | 128/128 | feasible | yes 0.07 | yes 0.072 | yes 0.029 | yes 0.059 | no: budget-exhausted | no: budget-exhausted | - | - |
| 16x16: 1x1, 62x 2x2, 1x1 | 1 x 16x16 | 62x 2x2, 2x 1x1 (ring) | 250/256 | feasible | yes 0.134 | yes 0.132 | yes 0.004 | yes 0.01 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x8: 30 stages | 1 x 8x8 | 7x 2x2, 8x 1x2, 5x 2x1, 10x 1x1 (ring) | 64/64 | feasible | cap | yes 0.593 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x8: 29 stages | 1 x 8x8 | 9x 2x2, 6x 1x2, 2x 2x1, 12x 1x1 (ring) | 64/64 | feasible | cap | yes 0.123 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x8: 31 stages | 1 x 8x8 | 8x 2x2, 4x 1x2, 4x 2x1, 15x 1x1 (ring) | 63/64 | feasible | yes 1.16 | yes 1.16 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x8: 34 stages | 1 x 8x8 | 7x 2x2, 6x 1x2, 3x 2x1, 18x 1x1 (ring) | 64/64 | feasible | yes 1.43 | yes 0.037 | yes 0.001 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x8: 37 stages | 1 x 8x8 | 5x 2x2, 6x 1x2, 6x 2x1, 20x 1x1 (ring) | 64/64 | feasible | yes 0.212 | yes 0.105 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x16 s1.0: 37 stages | 1 x 8x16 | 5x 2x4, 14x 2x2, 4x 1x2, 3x 2x1, 11x 1x1 (ring) | 121/128 | feasible | yes 5.71 | yes 5.64 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x16 s1.1: 32 stages | 1 x 8x16 | 8x 2x4, 10x 2x2, 5x 1x2, 4x 2x1, 5x 1x1 (ring) | 127/128 | open | cap | cap | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| random 8x16 s1.2: 32 stages | 1 x 8x16 | 6x 2x4, 13x 2x2, 4x 1x2, 5x 2x1, 4x 1x1 (ring) | 122/128 | feasible | yes 0.36 | yes 0.36 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| 7x7: 1x1, 10x 2x2 (only 9 fit) | 1 x 7x7 | 10x 2x2, 1x1 (ring) | 41/49 | infeasible | no 2.14 | no 2.14 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| 5x5 ring of 25 x 1x1 (odd: none) | 1 x 5x5 | 25x 1x1 (ring) | 25/25 | infeasible | no 0.003 | no 0.003 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |

### sweep

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep 4x8 fill60 s10: 8 stages 21/32 | 1 x 4x8 | 4x 2x2, 1x2, 3x 1x1 (ring) | 21/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 4x8 fill60 s11: 6 stages 21/32 | 1 x 4x8 | 5x 2x2, 1x1 (ring) | 21/32 | feasible | yes 0.001 | yes 0.002 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 4x8 fill80 s10: 10 stages 29/32 | 1 x 4x8 | 6x 2x2, 1x2, 3x 1x1 (ring) | 29/32 | feasible | yes 0.015 | yes 0.015 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 4x8 fill80 s11: 8 stages 27/32 | 1 x 4x8 | 6x 2x2, 1x2, 1x1 (ring) | 27/32 | feasible | yes 0.001 | yes 0.002 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 4x8 fill95 s10: 11 stages 31/32 | 1 x 4x8 | 6x 2x2, 1x2, 2x1, 3x 1x1 (ring) | 31/32 | feasible | yes 0.011 | yes 0.011 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 4x8 fill95 s11: 9 stages 29/32 | 1 x 4x8 | 6x 2x2, 2x 1x2, 1x1 (ring) | 29/32 | feasible | yes 0.01 | yes 0.009 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill60 s10: 8 stages 21/32 | 1 x 8x4 | 4x 2x2, 1x2, 3x 1x1 (ring) | 21/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill60 s11: 6 stages 21/32 | 1 x 8x4 | 5x 2x2, 1x1 (ring) | 21/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill80 s10: 10 stages 29/32 | 1 x 8x4 | 6x 2x2, 1x2, 3x 1x1 (ring) | 29/32 | feasible | yes 0.009 | yes 0.008 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill80 s11: 8 stages 27/32 | 1 x 8x4 | 6x 2x2, 1x2, 1x1 (ring) | 27/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill95 s10: 11 stages 31/32 | 1 x 8x4 | 6x 2x2, 1x2, 2x1, 3x 1x1 (ring) | 31/32 | infeasible | no 0.032 | no 0.032 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x4 fill95 s11: 9 stages 29/32 | 1 x 8x4 | 6x 2x2, 2x 1x2, 1x1 (ring) | 29/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill60 s10: 8 stages 21/32 | 1 x 2x16 | 4x 2x2, 1x2, 3x 1x1 (ring) | 21/32 | infeasible | no 0.009 | no 0.009 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill60 s11: 6 stages 21/32 | 1 x 2x16 | 5x 2x2, 1x1 (ring) | 21/32 | infeasible | no 0.005 | no 0.006 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill80 s10: 10 stages 29/32 | 1 x 2x16 | 6x 2x2, 1x2, 3x 1x1 (ring) | 29/32 | infeasible | no 0.01 | no 0.009 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill80 s11: 8 stages 27/32 | 1 x 2x16 | 6x 2x2, 1x2, 1x1 (ring) | 27/32 | infeasible | no 0.007 | no 0.007 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill95 s10: 11 stages 31/32 | 1 x 2x16 | 6x 2x2, 1x2, 2x1, 3x 1x1 (ring) | 31/32 | infeasible | no 0.008 | no 0.008 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 2x16 fill95 s11: 9 stages 29/32 | 1 x 2x16 | 6x 2x2, 2x 1x2, 1x1 (ring) | 29/32 | infeasible | no 0.007 | no 0.008 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill60 s10: 12 stages 39/64 | 1 x 8x8 | 2x4, 6x 2x2, 1x2, 2x1, 3x 1x1 (ring) | 39/64 | feasible | yes 0.016 | yes 0.016 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill60 s11: 12 stages 45/64 | 1 x 8x8 | 2x4, 8x 2x2, 2x 1x2, 1x1 (ring) | 45/64 | feasible | yes 0.005 | yes 0.005 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill80 s10: 17 stages 52/64 | 1 x 8x8 | 2x4, 8x 2x2, 2x 1x2, 2x 2x1, 4x 1x1 (ring) | 52/64 | feasible | yes 0.007 | yes 0.007 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill80 s11: 15 stages 52/64 | 1 x 8x8 | 2x4, 9x 2x2, 3x 1x2, 2x 1x1 (ring) | 52/64 | feasible | yes 0.021 | yes 0.021 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill95 s10: 20 stages 63/64 | 1 x 8x8 | 2x 2x4, 8x 2x2, 3x 1x2, 2x 2x1, 5x 1x1 (ring) | 63/64 | infeasible | no 37.6 | no 37.7 | - | - | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x8 fill95 s11: 19 stages 61/64 | 1 x 8x8 | 2x4, 10x 2x2, 4x 1x2, 2x1, 3x 1x1 (ring) | 61/64 | feasible | yes 0.331 | yes 0.333 | yes 0.001 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill60 s10: 26 stages 78/128 | 1 x 8x16 | 2x 2x4, 10x 2x2, 4x 1x2, 4x 2x1, 6x 1x1 (ring) | 78/128 | feasible | yes 0.024 | yes 0.024 | yes 0 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill60 s11: 22 stages 81/128 | 1 x 8x16 | 3x 2x4, 11x 2x2, 4x 1x2, 2x1, 3x 1x1 (ring) | 81/128 | feasible | yes 0.068 | yes 0.067 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill80 s10: 33 stages 104/128 | 1 x 8x16 | 3x 2x4, 13x 2x2, 5x 1x2, 6x 2x1, 6x 1x1 (ring) | 104/128 | feasible | yes 0.981 | yes 0.988 | yes 0.001 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill80 s11: 28 stages 110/128 | 1 x 8x16 | 5x 2x4, 14x 2x2, 4x 1x2, 2x1, 4x 1x1 (ring) | 110/128 | feasible | yes 0.087 | yes 0.086 | yes 0 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill95 s10: 36 stages 122/128 | 1 x 8x16 | 5x 2x4, 13x 2x2, 5x 1x2, 7x 2x1, 6x 1x1 (ring) | 122/128 | feasible | yes 14 | yes 14 | yes 0.001 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| sweep 8x16 fill95 s11: 30 stages 122/128 | 1 x 8x16 | 6x 2x4, 15x 2x2, 4x 1x2, 2x1, 4x 1x1 (ring) | 122/128 | feasible | yes 9.23 | yes 9.28 | yes 0.001 | yes 0.001 | no: budget-exhausted | no: budget-exhausted | - | - |

### min2

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-1x1 4x8 fill60 s20: 3 stages 20/32 | 1 x 4x8 | 2x 2x4, 2x2 (ring) | 20/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: input-infeasible | yes 0.005 | yes 0 | - |
| no-1x1 4x8 fill60 s21: 5 stages 22/32 | 1 x 4x8 | 2x4, 3x 2x2, 2x1 (ring) | 22/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 4x8 fill80 s20: 6 stages 28/32 | 1 x 4x8 | 2x 2x4, 2x 2x2, 1x2, 2x1 (ring) | 28/32 | feasible | yes 0.007 | yes 0.007 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 4x8 fill80 s21: 6 stages 26/32 | 1 x 4x8 | 2x4, 4x 2x2, 2x1 (ring) | 26/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 4x8 fill95 s20: 7 stages 30/32 | 1 x 4x8 | 2x 2x4, 2x 2x2, 1x2, 2x 2x1 (ring) | 30/32 | infeasible | no 0.019 | no 0.019 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 4x8 fill95 s21: 7 stages 30/32 | 1 x 4x8 | 2x4, 5x 2x2, 2x1 (ring) | 30/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x4 fill60 s20: 5 stages 24/32 | 1 x 8x4 | 2x 2x4, 2x2, 2x 2x1 (ring) | 24/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x4 fill60 s21: 4 stages 20/32 | 1 x 8x4 | 2x 2x4, 1x2, 2x1 (ring) | 20/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x4 fill80 s20: 6 stages 28/32 | 1 x 8x4 | 2x 2x4, 2x 2x2, 2x 2x1 (ring) | 28/32 | infeasible | no 0.006 | no 0.006 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x4 fill80 s21: 5 stages 28/32 | 1 x 8x4 | 3x 2x4, 1x2, 2x1 (ring) | 28/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x4 fill95 s20: 7 stages 32/32 | 1 x 8x4 | 2x 2x4, 3x 2x2, 2x 2x1 (ring) | 32/32 | feasible | yes 0.001 | yes 0.001 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x4 fill95 s21: 7 stages 32/32 | 1 x 8x4 | 3x 2x4, 3x 1x2, 2x1 (ring) | 32/32 | infeasible | no 0.006 | no 0.001 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 2x16 fill60 s20: 5 stages 24/32 | 1 x 2x16 | 2x 2x4, 2x2, 1x2, 2x1 (ring) | 24/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 2x16 fill60 s21: 5 stages 22/32 | 1 x 2x16 | 2x 2x4, 3x 2x1 (ring) | 22/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 2x16 fill80 s20: 6 stages 28/32 | 1 x 2x16 | 2x 2x4, 2x 2x2, 1x2, 2x1 (ring) | 28/32 | infeasible | no 0.006 | no 0.006 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 2x16 fill80 s21: 7 stages 26/32 | 1 x 2x16 | 2x 2x4, 1x2, 4x 2x1 (ring) | 26/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 2x16 fill95 s20: 7 stages 30/32 | 1 x 2x16 | 2x 2x4, 2x 2x2, 1x2, 2x 2x1 (ring) | 30/32 | infeasible | no 0.001 | no 0.001 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 2x16 fill95 s21: 9 stages 32/32 | 1 x 2x16 | 2x 2x4, 2x2, 1x2, 5x 2x1 (ring) | 32/32 | infeasible | no 0.001 | no 0.002 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill60 s20: 11 stages 40/64 | 1 x 8x8 | 2x 2x4, 3x 2x2, 4x 1x2, 2x 2x1 (ring) | 40/64 | feasible | yes 0.004 | yes 0.004 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill60 s21: 9 stages 40/64 | 1 x 8x8 | 3x 2x4, 2x 2x2, 1x2, 3x 2x1 (ring) | 40/64 | feasible | yes 0.004 | yes 0.004 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill80 s20: 14 stages 54/64 | 1 x 8x8 | 3x 2x4, 4x 2x2, 4x 1x2, 3x 2x1 (ring) | 54/64 | feasible | yes 0.005 | yes 0.005 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill80 s21: 12 stages 54/64 | 1 x 8x8 | 4x 2x4, 3x 2x2, 2x 1x2, 3x 2x1 (ring) | 54/64 | feasible | yes 0.005 | yes 0.005 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill95 s20: 16 stages 62/64 | 1 x 8x8 | 3x 2x4, 6x 2x2, 4x 1x2, 3x 2x1 (ring) | 62/64 | feasible | yes 0.819 | yes 0.826 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 fill95 s21: 15 stages 62/64 | 1 x 8x8 | 4x 2x4, 4x 2x2, 4x 1x2, 3x 2x1 (ring) | 62/64 | feasible | yes 0.059 | yes 0.059 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill60 s20: 20 stages 82/128 | 1 x 8x16 | 4x 2x4, 9x 2x2, 3x 1x2, 4x 2x1 (ring) | 82/128 | feasible | yes 0.019 | yes 0.019 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill60 s21: 22 stages 80/128 | 1 x 8x16 | 3x 2x4, 9x 2x2, 8x 1x2, 2x 2x1 (ring) | 80/128 | feasible | yes 0.054 | yes 0.054 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill80 s20: 24 stages 106/128 | 1 x 8x16 | 6x 2x4, 11x 2x2, 3x 1x2, 4x 2x1 (ring) | 106/128 | feasible | yes 0.148 | yes 0.148 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill80 s21: 28 stages 104/128 | 1 x 8x16 | 4x 2x4, 12x 2x2, 9x 1x2, 3x 2x1 (ring) | 104/128 | feasible | yes 0.027 | yes 0.027 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill95 s20: 30 stages 122/128 | 1 x 8x16 | 6x 2x4, 13x 2x2, 4x 1x2, 7x 2x1 (ring) | 122/128 | feasible | yes 2.79 | yes 2.76 | yes 0.001 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 fill95 s21: 32 stages 126/128 | 1 x 8x16 | 6x 2x4, 13x 2x2, 10x 1x2, 3x 2x1 (ring) | 126/128 | open | cap | cap | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 4x8 ring of 16 x 1x2 | 1 x 4x8 | 16x 1x2 (ring) | 32/32 | feasible | yes 0.037 | yes 0.002 | yes 0 | yes 0 | no: input-infeasible | yes 0.036 | yes 0 | - |
| no-1x1 4x8 ring of 16 x 2x1 | 1 x 4x8 | 16x 2x1 (ring) | 32/32 | feasible | yes 0.002 | yes 0.003 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 4x8 ring: 8x 1x2, 8x 2x1 | 1 x 4x8 | 8x 1x2, 8x 2x1 (ring) | 32/32 | feasible | yes 0.011 | yes 0.002 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 8x8 ring of 32 x 1x2 | 1 x 8x8 | 32x 1x2 (ring) | 64/64 | feasible | yes 0.317 | yes 0.009 | yes 0.001 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 ring of 16 x 2x2 | 1 x 8x8 | 16x 2x2 (ring) | 64/64 | feasible | yes 0.125 | yes 0.005 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 ring: 8x 2x2, 16x 1x2 | 1 x 8x8 | 8x 2x2, 16x 1x2 (ring) | 64/64 | feasible | yes 1.6 | yes 0.03 | yes 0 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x8 ring: 4x 2x4, 8x 2x2, 8x 1x2 | 1 x 8x8 | 4x 2x4, 8x 2x2, 8x 1x2 (ring) | 80/64 | infeasible | no 1.26 | no 1.24 | - | - | no: input-infeasible | no: input-infeasible | - | - |
| no-1x1 8x16 ring of 64 x 1x2 | 1 x 8x16 | 64x 1x2 (ring) | 128/128 | feasible | cap | yes 0.044 | yes 0.004 | yes 0.012 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 ring of 32 x 2x2 | 1 x 8x16 | 32x 2x2 (ring) | 128/128 | feasible | yes 9.46 | yes 0.024 | yes 0.001 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 8x16 ring: 8x 2x4, 16x 2x2, 16x 1x2 | 1 x 8x16 | 8x 2x4, 16x 2x2, 16x 1x2 (ring) | 160/128 | infeasible | cap | cap | - | - | no: input-infeasible | no: input-infeasible | - | - |
| no-1x1 16x16 ring of 64 x 2x2 | 1 x 16x16 | 64x 2x2 (ring) | 256/256 | feasible | cap | yes 0.128 | yes 0.004 | yes 0.012 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 16x16 ring of 128 x 1x2 | 1 x 16x16 | 128x 1x2 (ring) | 256/256 | feasible | cap | yes 0.228 | yes 0.029 | yes 0.058 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 5x8 ring of 20 x 1x2 (odd rows) | 1 x 5x8 | 20x 1x2 (ring) | 40/40 | feasible | yes 0.354 | yes 0.003 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 6x6 ring: 9x 2x2 (area 36, 3x3 of 2x2: odd cycle) | 1 x 6x6 | 9x 2x2 (ring) | 36/36 | infeasible | no 0.109 | no 0.02 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 7x8: 14x 2x2 (only 12 fit) | 1 x 7x8 | 14x 2x2 (ring) | 56/56 | infeasible | no 9.32 | no 0.028 | - | - | no: input-infeasible | no: budget-exhausted | - | - |

### hetero

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hetero 2 meshes 4x8, 2 cables/pair, 16 stages 56/64, seed 0 | 2 x 4x8 | 3x 2x4, 4x 2x2, 2x 1x2, 5x 2x1, 2x 1x1 (ring) | 56/64 | feasible | yes 0.026 | yes 0.026 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 2 meshes 4x8, 2 cables/pair, 20 stages 56/64, seed 1 | 2 x 4x8 | 2x 2x4, 5x 2x2, 5x 1x2, 2x 2x1, 6x 1x1 (ring) | 56/64 | feasible | yes 0.007 | yes 0.007 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 2 meshes 4x8, 2 cables/pair, 23 stages 58/64, seed 2 | 2 x 4x8 | 8x 2x2, 8x 1x2, 3x 2x1, 4x 1x1 (ring) | 58/64 | feasible | yes 0.246 | yes 0.248 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 2 meshes 4x8, 4 cables/pair, 12 stages 58/64, seed 0 | 2 x 4x8 | 4x 2x4, 5x 2x2, 2x 1x2, 2x1 (ring) | 58/64 | feasible | yes 0.022 | yes 0.021 | yes 0 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| hetero 2 meshes 4x8, 4 cables/pair, 19 stages 57/64, seed 1 | 2 x 4x8 | 2x4, 9x 2x2, 1x2, 3x 2x1, 5x 1x1 (ring) | 57/64 | feasible | yes 0.007 | yes 0.007 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 2 meshes 4x8, 4 cables/pair, 13 stages 58/64, seed 2 | 2 x 4x8 | 5x 2x4, 2x 2x2, 1x2, 3x 2x1, 2x 1x1 (ring) | 58/64 | feasible | yes 0.004 | yes 0.004 | yes 0.001 | yes 0 | no: budget-exhausted | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 0 | 4 x 4x8 | 5x 2x4, 13x 2x2, 5x 1x2, 3x 2x1, 8x 1x1 (ring) | 116/128 | feasible | yes 0.834 | yes 0.841 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 1 | 4 x 4x8 | 5x 2x4, 12x 2x2, 3x 1x2, 8x 2x1, 6x 1x1 (ring) | 116/128 | feasible | yes 1.34 | yes 1.35 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 2 cables/pair, 34 stages 116/128, seed 2 | 4 x 4x8 | 4x 2x4, 15x 2x2, 7x 1x2, 2x 2x1, 6x 1x1 (ring) | 116/128 | feasible | yes 2.2 | yes 2.22 | yes 0.001 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 4 cables/pair, 33 stages 111/128, seed 0 | 4 x 4x8 | 5x 2x4, 11x 2x2, 7x 1x2, 3x 2x1, 7x 1x1 (ring) | 111/128 | feasible | yes 0.458 | yes 0.459 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 4 cables/pair, 28 stages 113/128, seed 1 | 4 x 4x8 | 7x 2x4, 8x 2x2, 7x 1x2, 5x 2x1, 1x1 (ring) | 113/128 | feasible | yes 0.44 | yes 0.439 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| hetero 4 meshes 4x8, 4 cables/pair, 29 stages 111/128, seed 2 | 4 x 4x8 | 5x 2x4, 12x 2x2, 6x 1x2, 5x 2x1, 1x1 (ring) | 111/128 | feasible | yes 0.126 | yes 0.125 | yes 0.001 | yes 0 | no: input-infeasible | no: no-alignment | - | - |

### multi4

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-1x1 domino 2 meshes 4x8, 1 cables/pair, 32 stages 64/64, seed 0 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | infeasible | cap | no 8.04 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 2 meshes 4x8, 1 cables/pair, 32 stages 64/64, seed 1 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | infeasible | cap | no 8.12 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 2 meshes 4x8, 2 cables/pair, 32 stages 64/64, seed 0 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | feasible | yes 19.9 | yes 0.035 | yes 0.002 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 2 meshes 4x8, 2 cables/pair, 32 stages 64/64, seed 1 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | feasible | yes 35.7 | yes 1.09 | yes 0.001 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 2 meshes 4x8, 4 cables/pair, 32 stages 64/64, seed 0 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | feasible | yes 1.22 | yes 0.009 | yes 0.001 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 2 meshes 4x8, 4 cables/pair, 32 stages 64/64, seed 1 | 2 x 4x8 | 32x 1x2 (ring) | 64/64 | feasible | yes 1.99 | yes 0.009 | yes 0.001 | yes 0.003 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 1 cables/pair, 64 stages 128/128, seed 0 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | open | cap | cap | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 1 cables/pair, 64 stages 128/128, seed 1 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | infeasible | cap | no 50.7 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 2 cables/pair, 64 stages 128/128, seed 0 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | open | cap | cap | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 2 cables/pair, 64 stages 128/128, seed 1 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | feasible | cap | yes 11.3 | yes 0.005 | yes 0.012 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 4 cables/pair, 64 stages 128/128, seed 0 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | feasible | cap | yes 0.277 | yes 0.004 | yes 0.012 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 4 meshes 4x8, 4 cables/pair, 64 stages 128/128, seed 1 | 4 x 4x8 | 64x 1x2 (ring) | 128/128 | feasible | cap | yes 0.751 | yes 0.004 | yes 0.012 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 8 meshes 4x8, 4 cables/pair, 128 stages 256/256, seed 0 | 8 x 4x8 | 128x 1x2 (ring) | 256/256 | feasible | cap | yes 2.61 | yes 0.03 | yes 0.057 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 domino 8 meshes 4x8, 4 cables/pair, 128 stages 256/256, seed 1 | 8 x 4x8 | 128x 1x2 (ring) | 256/256 | feasible | cap | yes 21.1 | yes 0.036 | yes 0.057 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 1 cables/pair, 15 stages 60/64, seed 0 | 2 x 4x8 | 4x 2x4, 3x 2x2, 3x 1x2, 5x 2x1 (ring) | 60/64 | feasible | yes 0.031 | yes 0.031 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 1 cables/pair, 16 stages 64/64, seed 1 | 2 x 4x8 | 3x 2x4, 7x 2x2, 2x 1x2, 4x 2x1 (ring) | 64/64 | infeasible | no 0.808 | no 0.101 | - | - | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 2 cables/pair, 17 stages 58/64, seed 0 | 2 x 4x8 | 2x4, 9x 2x2, 7x 2x1 (ring) | 58/64 | feasible | yes 0.023 | yes 0.023 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 2 cables/pair, 17 stages 58/64, seed 1 | 2 x 4x8 | 2x 2x4, 6x 2x2, 4x 1x2, 5x 2x1 (ring) | 58/64 | feasible | yes 0.05 | yes 0.05 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 4 cables/pair, 13 stages 58/64, seed 0 | 2 x 4x8 | 3x 2x4, 7x 2x2, 1x2, 2x 2x1 (ring) | 58/64 | feasible | yes 0.004 | yes 0.004 | yes 0 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 2 meshes 4x8, 4 cables/pair, 19 stages 60/64, seed 1 | 2 x 4x8 | 2x 2x4, 5x 2x2, 5x 1x2, 7x 2x1 (ring) | 60/64 | feasible | yes 0.035 | yes 0.035 | yes 0.001 | yes 0 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 mixed 4 meshes 4x8, 1 cables/pair, 30 stages 118/128, seed 0 | 4 x 4x8 | 6x 2x4, 11x 2x2, 9x 1x2, 4x 2x1 (ring) | 118/128 | feasible | yes 4.89 | yes 4.87 | yes 0.001 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 4 meshes 4x8, 1 cables/pair, 26 stages 118/128, seed 1 | 4 x 4x8 | 8x 2x4, 9x 2x2, 5x 1x2, 4x 2x1 (ring) | 118/128 | feasible | yes 0.343 | yes 0.347 | yes 0 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 4 meshes 4x8, 2 cables/pair, 30 stages 118/128, seed 0 | 4 x 4x8 | 6x 2x4, 11x 2x2, 8x 1x2, 5x 2x1 (ring) | 118/128 | feasible | yes 1.9 | yes 1.89 | yes 0.002 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 4 meshes 4x8, 2 cables/pair, 28 stages 116/128, seed 1 | 4 x 4x8 | 7x 2x4, 9x 2x2, 5x 1x2, 7x 2x1 (ring) | 116/128 | feasible | yes 0.786 | yes 0.783 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 4 meshes 4x8, 4 cables/pair, 30 stages 122/128, seed 0 | 4 x 4x8 | 6x 2x4, 13x 2x2, 5x 1x2, 6x 2x1 (ring) | 122/128 | feasible | yes 1.27 | yes 1.27 | yes 0.001 | yes 0 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 4 meshes 4x8, 4 cables/pair, 30 stages 116/128, seed 1 | 4 x 4x8 | 4x 2x4, 16x 2x2, 4x 1x2, 6x 2x1 (ring) | 116/128 | feasible | yes 0.548 | yes 0.547 | yes 0.001 | yes 0.001 | no: input-infeasible | no: no-alignment | - | - |
| no-1x1 mixed 8 meshes 4x8, 4 cables/pair, 62 stages 232/256, seed 0 | 8 x 4x8 | 10x 2x4, 24x 2x2, 15x 1x2, 13x 2x1 (ring) | 232/256 | feasible | yes 24.7 | yes 24.4 | yes 0.004 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |
| no-1x1 mixed 8 meshes 4x8, 4 cables/pair, 56 stages 232/256, seed 1 | 8 x 4x8 | 12x 2x4, 24x 2x2, 10x 1x2, 10x 2x1 (ring) | 232/256 | feasible | yes 35.7 | yes 35.7 | yes 0.003 | yes 0.001 | no: input-infeasible | no: budget-exhausted | - | - |

### designed

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4x4: 1x1, 3x 2x2 (blaze: no-cycle-13) | 1 x 4x4 | 3x 2x2, 1x1 (ring) | 13/16 | feasible | yes 0 | - | yes 0 | yes 0 | no: no-cycle-of-length-13 | - | - | - |
| 4x4: 1x1, 3x 2x2, 1x1 | 1 x 4x4 | 3x 2x2, 2x 1x1 (ring) | 14/16 | feasible | yes 0 | - | yes 0 | yes 0 | yes 0 | - | yes 0 | yes 0 |
| 4x4: 1x1, 2x 2x2, 1x1 | 1 x 4x4 | 2x 2x2, 2x 1x1 (ring) | 10/16 | feasible | yes 0 | - | yes 0 | yes 0 | yes 0 | - | yes 0 | yes 0 |
| 4x6: 1x1, 4x 2x2, 1x1 (blaze: no-alignment) | 1 x 4x6 | 4x 2x2, 2x 1x1 (ring) | 18/24 | feasible | yes 0.001 | - | yes 0 | yes 0 | no: no-alignment | - | - | - |
| 4x8: 1x1, 6x 2x2, 1x1 (blaze: no-alignment) | 1 x 4x8 | 6x 2x2, 2x 1x1 (ring) | 26/32 | feasible | yes 0.001 | - | yes 0 | yes 0 | no: no-alignment | - | - | - |
| 4x8: 1x1, 2x4, 2x4, 2x2, 1x1 | 1 x 4x8 | 2x 2x4, 2x2, 2x 1x1 (ring) | 22/32 | feasible | yes 0.001 | - | yes 0 | yes 0 | yes 0.009 | - | yes 0 | yes 0 |
| 5x5: 1x1, 5x 2x2 (only 4 fit) | 1 x 5x5 | 5x 2x2, 1x1 (ring) | 21/25 | infeasible | no 0.011 | - | - | - | no: budget-exhausted | - | - | - |
| 6x6: 1x1, 9x 2x2 (area 37 > 36) | 1 x 6x6 | 9x 2x2, 1x1 (ring) | 37/36 | infeasible | no 0.093 | - | - | - | no: input-infeasible | - | - | - |
| 6x6 fork: r 1x1, r-a1-a2-r, r-b1-b2-r (2x2 branches) | 1 x 6x6 | 4x 2x2, 1x1 (other) | 17/36 | feasible | yes 0.008 | - | - | - | n/a | - | - | - |

### repo

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4x8 ring of 32 x 1x1 (fills) | 1 x 4x8 | 32x 1x1 (ring) | 32/32 | feasible | yes 0.004 | - | yes 0.001 | yes 0.003 | yes 0.038 | - | yes 0.001 | yes 0.003 |
| 2x16 ring of 32 x 1x1 (fills) | 1 x 2x16 | 32x 1x1 (ring) | 32/32 | feasible | yes 0.003 | - | yes 0.001 | yes 0.003 | no: budget-exhausted | - | - | - |
| 5x5 ring of 25 x 1x1 (odd: none) | 1 x 5x5 | 25x 1x1 (ring) | 25/25 | infeasible | no 0.003 | - | - | - | no: budget-exhausted | - | - | - |
| 4x8: 1x1, 1x2 x6, 2x2 x4, 1x1 | 1 x 4x8 | 4x 2x2, 6x 1x2, 2x 1x1 (ring) | 30/32 | feasible | yes 0.011 | - | yes 0 | yes 0 | no: no-alignment | - | - | - |
| 8x4: 1x1, 2x1 x6, 2x2 x4, 1x1 | 1 x 8x4 | 4x 2x2, 6x 2x1, 2x 1x1 (ring) | 30/32 | feasible | yes 0.002 | - | yes 0 | yes 0 | no: no-alignment | - | - | - |
| 2x16: 1x1, 1x2 x15, 1x1 | 1 x 2x16 | 15x 1x2, 2x 1x1 (ring) | 32/32 | feasible | yes 0.002 | - | yes 0 | yes 0 | no: budget-exhausted | - | - | - |
| 2x16: 1x1, 2x2 x7, 1x1 (fills 30/32) | 1 x 2x16 | 7x 2x2, 2x 1x1 (ring) | 30/32 | infeasible | no 0.007 | - | - | - | no: budget-exhausted | - | - | - |
| 4x8: 2x4 x4 (fills, ring of 4) | 1 x 4x8 | 4x 2x4 (ring) | 32/32 | feasible | yes 0.001 | - | yes 0 | yes 0 | no: input-infeasible | - | - | - |
| 8x8: 1x1, 14x 2x2, 1x1 | 1 x 8x8 | 14x 2x2, 2x 1x1 (ring) | 58/64 | feasible | yes 0.005 | - | yes 0 | yes 0 | no: budget-exhausted | - | - | - |

### multi

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed4 (P,R): cables A0-B0, A2-B2 | 2 x 1x4 | 2x 1x2, 2x 1x1 (ring) | 6/8 | feasible | - | - | - | - | yes 0 | - | yes 0 | yes 0.001 |
| mixed4 (P,Q): cables A0-B0, A3-B3 | 2 x 1x4 | 2x 1x2, 2x 1x1 (ring) | 6/8 | open | - | - | - | - | no: no-cycle-of-length-6 | - | - | - |
| mixed4 Case B: crossings only on A0-B0 | 2 x 1x4 | 2x 1x2, 2x 1x1 (ring) | 6/8 | open | - | - | - | - | no: no-cycle-of-length-6 | - | - | - |
| two 4x4: 1x1, 6x 2x2, 1x1, cables (0,0),(0,1) | 2 x 4x4 | 6x 2x2, 2x 1x1 (ring) | 26/32 | open | - | - | - | - | no: no-alignment | - | - | - |
| two 4x4 ring of 32 x 1x1, cables (0,0),(0,1) | 2 x 4x4 | 32x 1x1 (ring) | 32/32 | feasible | - | - | - | - | yes 0.033 | - | yes 0.001 | yes 0.009 |

### rings

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4x8 ring of 32 x 1x1 | 1 x 4x8 | 32x 1x1 (ring) | 32/32 | feasible | - | - | - | - | yes 0.036 | - | yes 0.001 | yes 0.003 |
| 8x8 ring of 64 x 1x1 | 1 x 8x8 | 64x 1x1 (ring) | 64/64 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| 8x16 ring of 128 x 1x1 | 1 x 8x16 | 128x 1x1 (ring) | 128/128 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| 16x16 ring of 256 x 1x1 | 1 x 16x16 | 256x 1x1 (ring) | 256/256 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| 5x5 ring of 25 x 1x1 (odd: none) | 1 x 5x5 | 25x 1x1 (ring) | 25/25 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| 7x7 ring of 49 x 1x1 (odd: none) | 1 x 7x7 | 49x 1x1 (ring) | 49/49 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| 8x8 ring of 60 x 1x1 (4 spare) | 1 x 8x8 | 60x 1x1 (ring) | 60/64 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| two 4x4 meshes, cables (0,0),(0,1) | 2 x 4x4 | 32x 1x1 (ring) | 32/32 | feasible | - | - | - | - | yes 0.033 | - | yes 0.001 | yes 0.003 |
| two 4x4 meshes, cables same colour | 2 x 4x4 | 32x 1x1 (ring) | 32/32 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| two 8x8 meshes, cables (0,0),(0,1) | 2 x 8x8 | 128x 1x1 (ring) | 128/128 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| two 8x8 meshes, cables same colour | 2 x 8x8 | 128x 1x1 (ring) | 128/128 | open | - | - | - | - | no: budget-exhausted | - | - | - |
| two 4x8 meshes, cables same colour | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | open | - | - | - | - | no: budget-exhausted | - | - | - |

### multi2

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve | topology per-slice Auto, direct | resolver, direct (one slice = one submesh) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | open | cap | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | yes 0.846 | yes 0.079 | yes 0.005 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.067 | yes 0.271 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | yes 0.014 | yes 0.015 | yes 0.005 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.013 | yes 0.038 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | yes 0.015 | yes 0.016 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.013 | yes 0.005 |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | cap | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | cap | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | yes 0.079 | yes 0.074 | yes 0.031 | yes 0.057 | no: budget-exhausted | - | - | - | yes 0.059 | yes 25 |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | cap | yes 0.59 | yes 0.03 | yes 0.057 | no: budget-exhausted | - | - | - | yes 0.413 | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 0 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | cap | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 1 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | cap | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 0 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | feasible | cap | yes 52.7 | yes 0.258 | yes 0.325 | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 1 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | feasible | cap | cap | - | - | no: budget-exhausted | - | - | - | yes 34.7 | cap |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 0 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | yes 4.08 | yes 0.342 | yes 0.005 | yes 0.012 | n/a | - | - | - | yes 0.119 | yes 0.005 |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 0 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | yes 0.014 | yes 0.014 | yes 0.004 | yes 0.012 | n/a | - | - | - | yes 0.013 | yes 0.005 |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 0 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | open | cap | cap | - | - | n/a | - | - | - | cap | cap |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 0 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | feasible | cap | yes 0.32 | yes 0.032 | yes 0.056 | n/a | - | - | - | yes 0.328 | yes 21.9 |
| 2 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 0 | 2 x 4x8 | 48x 1x1 (ring) | 48/64 | feasible | yes 0.011 | yes 0.011 | yes 0.003 | yes 0.006 | no: budget-exhausted | - | - | - | yes 0.009 | yes 0.003 |
| 4 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 0 | 4 x 4x8 | 96x 1x1 (ring) | 96/128 | feasible | yes 0.05 | yes 0.05 | yes 0.019 | yes 0.029 | no: budget-exhausted | - | - | - | yes 0.044 | yes 30.2 |

### multi3

| Case | Meshes | Stages | Area / slices | Verdict | footprint Auto | footprint Auto + completeness | resolver after footprint carve | topology Auto after footprint carve | planner (C++ port) | planner, stage-0 rule off | resolver after planner carve | topology Auto after planner carve | topology per-slice Auto, direct | resolver, direct (one slice = one submesh) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.062 | yes 0.005 | yes 0.013 | no: budget-exhausted | - | - | - | yes 0.058 | yes 0.069 |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.053 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.049 | yes 0.009 |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.053 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.049 | yes 16 |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.015 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.013 | yes 0.005 |
| 2 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.015 | yes 0.005 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.012 | yes 0.006 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.018 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.015 | cap |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.015 | yes 0.004 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.013 | yes 0.005 |
| 2 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 2 x 4x8 | 64x 1x1 (ring) | 64/64 | feasible | - | yes 0.14 | yes 0.005 | yes 0.012 | no: budget-exhausted | - | - | - | yes 0.069 | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | - | yes 0.071 | yes 0.029 | yes 0.058 | no: budget-exhausted | - | - | - | yes 0.059 | yes 0.032 |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | - | yes 1.24 | yes 0.031 | yes 0.059 | no: budget-exhausted | - | - | - | yes 2.73 | cap |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | - | yes 0.071 | yes 0.03 | yes 0.058 | no: budget-exhausted | - | - | - | yes 0.06 | yes 0.031 |
| 4 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 4 x 4x8 | 128x 1x1 (ring) | 128/128 | feasible | - | yes 0.07 | yes 0.03 | yes 0.057 | no: budget-exhausted | - | - | - | yes 0.059 | yes 24.5 |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 32/32 per mesh, seed 12 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 32/32 per mesh, seed 12 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | feasible | - | yes 14 | yes 0.258 | yes 0.323 | no: budget-exhausted | - | - | - | yes 2.18 | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | feasible | - | yes 2.2 | yes 0.247 | yes 0.324 | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 4 cable/pair, 32/32 per mesh, seed 12 | 8 x 4x8 | 256x 1x1 (ring) | 256/256 | feasible | - | cap | - | - | no: budget-exhausted | - | - | - | yes 14.3 | cap |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | - | yes 7.51 | yes 0.004 | yes 0.012 | n/a | - | - | - | yes 2.81 | yes 0.829 |
| 2 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | - | yes 0.146 | yes 0.004 | yes 0.013 | n/a | - | - | - | yes 0.156 | yes 0.007 |
| 2 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 2 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | - | yes 0.049 | yes 0.004 | yes 0.012 | n/a | - | - | - | yes 0.047 | yes 0.006 |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | - | yes 0.114 | yes 0.004 | yes 0.014 | n/a | - | - | - | yes 0.066 | yes 0.006 |
| 2 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | 2 x 4x8 | 64x 1x1 (chain) | 64/64 | feasible | - | yes 0.052 | yes 0.004 | yes 0.012 | n/a | - | - | - | yes 0.049 | yes 5.9 |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 4 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | feasible | - | cap | - | - | n/a | - | - | - | cap | yes 1.05 |
| 4 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | feasible | - | cap | - | - | n/a | - | - | - | yes 2.15 | yes 0.077 |
| 4 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | feasible | - | yes 0.067 | yes 0.03 | yes 0.057 | n/a | - | - | - | yes 0.057 | yes 0.032 |
| 4 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | 4 x 4x8 | 128x 1x1 (chain) | 128/128 | feasible | - | yes 5.55 | yes 0.031 | yes 0.059 | n/a | - | - | - | yes 1.02 | cap |
| 8 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 8 32-rank meshes line, 1 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 8 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 8 32-rank meshes line, 2 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 8 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 10 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | feasible | - | cap | - | - | n/a | - | - | - | yes 1.76 | cap |
| 8 32-rank meshes line, 4 cable/pair, 32/32 per mesh, seed 11 | 8 x 4x8 | 256x 1x1 (chain) | 256/256 | open | - | cap | - | - | n/a | - | - | - | cap | cap |
| 2 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | 2 x 4x8 | 48x 1x1 (ring) | 48/64 | feasible | - | yes 0.011 | yes 0.002 | yes 0.007 | no: budget-exhausted | - | - | - | yes 0.01 | yes 0.31 |
| 2 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | 2 x 4x8 | 56x 1x1 (ring) | 56/64 | feasible | - | yes 0.251 | yes 0.003 | yes 0.009 | no: budget-exhausted | - | - | - | yes 0.312 | yes 0.316 |
| 2 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | 2 x 4x8 | 48x 1x1 (ring) | 48/64 | feasible | - | yes 0.012 | yes 0.002 | yes 0.006 | no: budget-exhausted | - | - | - | yes 0.01 | cap |
| 2 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | 2 x 4x8 | 56x 1x1 (ring) | 56/64 | feasible | - | yes 0.013 | yes 0.003 | yes 0.01 | no: budget-exhausted | - | - | - | yes 0.011 | cap |
| 4 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | 4 x 4x8 | 96x 1x1 (ring) | 96/128 | feasible | - | yes 1.45 | yes 0.013 | yes 0.028 | no: budget-exhausted | - | - | - | yes 2.88 | yes 0.394 |
| 4 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | 4 x 4x8 | 112x 1x1 (ring) | 112/128 | feasible | - | yes 9.9 | yes 0.02 | yes 0.042 | no: budget-exhausted | - | - | - | yes 25.2 | yes 0.415 |
| 4 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | 4 x 4x8 | 96x 1x1 (ring) | 96/128 | feasible | - | yes 0.051 | yes 0.017 | yes 0.029 | no: budget-exhausted | - | - | - | yes 0.043 | yes 0.022 |
| 4 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | 4 x 4x8 | 112x 1x1 (ring) | 112/128 | feasible | - | yes 0.489 | yes 0.02 | yes 0.043 | no: budget-exhausted | - | - | - | yes 0.408 | yes 0.022 |
| 8 32-rank meshes ring, 1 cable/pair, 24/32 per mesh, seed 10 | 8 x 4x8 | 192x 1x1 (ring) | 192/256 | feasible | - | yes 18.2 | yes 0.102 | yes 0.148 | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 1 cable/pair, 28/32 per mesh, seed 10 | 8 x 4x8 | 224x 1x1 (ring) | 224/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 24/32 per mesh, seed 10 | 8 x 4x8 | 192x 1x1 (ring) | 192/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
| 8 32-rank meshes ring, 2 cable/pair, 28/32 per mesh, seed 10 | 8 x 4x8 | 224x 1x1 (ring) | 224/256 | open | - | cap | - | - | no: budget-exhausted | - | - | - | cap | cap |
