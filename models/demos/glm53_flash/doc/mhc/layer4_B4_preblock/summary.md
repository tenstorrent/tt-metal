# layer 4, B=4, fast + mhc_post_mix custom op (preblock)

traced decode window: 117 ops, 7.796 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 2 | 134 | 1.7 % |
| ATTN | 75 | 2203 | 28.3 % |
| FFN_HC | 2 | 132 | 1.7 % |
| ROUTER | 17 | 168 | 2.2 % |
| EXPERTS | 15 | 5078 | 65.1 % |
| SHARED | 5 | 79 | 1.0 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 2 | 134 |
| FFN_HC | 2 | 132 |
| both | 4 | 267 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| GenericOpDeviceOperation | 4 | 267 | 66.6 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | GenericOpDeviceOperation | 120 | 101 |
| 1 | GenericOpDeviceOperation | 120 | 33 |
