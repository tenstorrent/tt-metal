# OP2OP SUMMARY — 3-layer (1 dense + 2 MoE), WARM pass, slowest device
# kernel = total device-op compute time; op2op = total inter-op gap (warm/non-compile)

| run                  | logical_n | dev |  ops | kernel_ms | op2op_ms | k+gap_ms | gap% |
|----------------------|-----------|-----|------|-----------|----------|----------|------|
| STANDALONE first     |      5120 |  20 |  240 |     56.10 |    28.38 |    84.49 |   34% |
| STANDALONE last      |     56320 |  30 |  240 |     91.12 |    13.02 |   104.14 |   13% |
| SERVICE first        |      5120 |  20 |  240 |     54.68 |    82.88 |   137.56 |   60% |
| SERVICE last         |     56320 |   2 |  240 |     90.38 |    42.08 |   132.46 |   32% |
