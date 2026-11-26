# layer 4, B=4, wrap

traced decode window: 466 ops, 10.630 ms device time

| module (innermost label) | ops | device us | % |
|---|---|---|---|
| ATTN_HC | 110 | 935 | 8.8 % |
| NORM | 2 | 93 | 0.9 % |
| ATTN | 75 | 2382 | 22.4 % |
| FFN_HC | 110 | 936 | 8.8 % |
| ROUTER | 139 | 700 | 6.6 % |
| EXPERTS | 21 | 5450 | 51.3 % |
| SHARED | 8 | 132 | 1.2 % |
| MOE | 1 | 3 | 0.0 % |

## Hyper-connection wraps (their own ops, sublayers excluded)

| wrap | ops | device us |
|---|---|---|
| ATTN_HC | 110 | 935 |
| FFN_HC | 110 | 936 |
| both | 220 | 1871 |

### Wrap ops by kind (both wraps)

| op | count | us total | us each |
|---|---|---|---|
| TernaryDeviceOperation | 38 | 608 | 16.0 |
| MatmulDeviceOperation | 2 | 263 | 131.6 |
| BinaryNgDeviceOperation | 18 | 233 | 12.9 |
| ReduceDeviceOperation | 2 | 229 | 114.5 |
| SliceDeviceOperation | 64 | 139 | 2.2 |
| TilizeWithValPaddingDeviceOperation | 42 | 133 | 3.2 |
| MhcSplitSinkhornDeviceOperation | 2 | 104 | 52.2 |
| UntilizeWithUnpaddingDeviceOperation | 42 | 82 | 2.0 |
| FillPadDeviceOperation | 2 | 30 | 15.2 |
| ConcatDeviceOperation | 2 | 30 | 15.1 |
| TypecastDeviceOperation | 4 | 13 | 3.2 |
| UnaryDeviceOperation | 2 | 6 | 3.2 |

### ATTN_HC ops in order

| # | op | cores | us |
|---|---|---|---|
| 0 | MatmulDeviceOperation 32 x 16384 x 32 | 1 | 131 |
| 1 | BinaryNgDeviceOperation | 120 | 24 |
| 2 | FillPadDeviceOperation | 120 | 15 |
| 3 | ReduceDeviceOperation | 1 | 115 |
| 4 | BinaryNgDeviceOperation | 120 | 5 |
| 5 | BinaryNgDeviceOperation | 120 | 5 |
| 6 | UnaryDeviceOperation | 120 | 3 |
| 7 | BinaryNgDeviceOperation | 120 | 7 |
| 8 | MhcSplitSinkhornDeviceOperation | 1 | 52 |
| 9 | SliceDeviceOperation | 120 | 3 |
| 10 | SliceDeviceOperation | 120 | 3 |
| 11 | SliceDeviceOperation | 120 | 3 |
| 12 | SliceDeviceOperation | 120 | 3 |
| 13 | SliceDeviceOperation | 1 | 1 |
| 14 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 15 | SliceDeviceOperation | 4 | 2 |
| 16 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 17 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 18 | SliceDeviceOperation | 4 | 2 |
| 19 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 20 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 21 | SliceDeviceOperation | 4 | 2 |
| 22 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 23 | BinaryNgDeviceOperation | 120 | 16 |
| 24 | TernaryDeviceOperation | 120 | 16 |
| 25 | TernaryDeviceOperation | 120 | 16 |
| 26 | TernaryDeviceOperation | 120 | 16 |
| 27 | TypecastDeviceOperation | 120 | 3 |
| 28 | TypecastDeviceOperation | 120 | 3 |
| 29 | SliceDeviceOperation | 120 | 3 |
| 30 | SliceDeviceOperation | 120 | 3 |
| 31 | SliceDeviceOperation | 120 | 3 |
| 32 | SliceDeviceOperation | 120 | 3 |
| 33 | SliceDeviceOperation | 1 | 1 |
| 34 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 35 | SliceDeviceOperation | 4 | 2 |
| 36 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 37 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 38 | SliceDeviceOperation | 4 | 2 |
| 39 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 40 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 41 | SliceDeviceOperation | 4 | 2 |
| 42 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 43 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 44 | SliceDeviceOperation | 4 | 2 |
| 45 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 46 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 47 | SliceDeviceOperation | 4 | 2 |
| 48 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 49 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 50 | SliceDeviceOperation | 4 | 2 |
| 51 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 52 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 53 | SliceDeviceOperation | 4 | 2 |
| 54 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 55 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 56 | SliceDeviceOperation | 4 | 2 |
| 57 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 58 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 59 | SliceDeviceOperation | 4 | 2 |
| 60 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 61 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 62 | SliceDeviceOperation | 4 | 2 |
| 63 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 64 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 65 | SliceDeviceOperation | 4 | 2 |
| 66 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 67 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 68 | SliceDeviceOperation | 4 | 2 |
| 69 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 70 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 71 | SliceDeviceOperation | 4 | 2 |
| 72 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 73 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 74 | SliceDeviceOperation | 4 | 2 |
| 75 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 76 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 77 | SliceDeviceOperation | 4 | 2 |
| 78 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 79 | SliceDeviceOperation | 1 | 1 |
| 80 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 81 | SliceDeviceOperation | 4 | 2 |
| 82 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 83 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 84 | SliceDeviceOperation | 4 | 2 |
| 85 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 86 | UntilizeWithUnpaddingDeviceOperation | 1 | 2 |
| 87 | SliceDeviceOperation | 4 | 2 |
| 88 | TilizeWithValPaddingDeviceOperation | 1 | 3 |
| 89 | BinaryNgDeviceOperation | 120 | 15 |
| 90 | BinaryNgDeviceOperation | 120 | 15 |
| 91 | BinaryNgDeviceOperation | 120 | 15 |
| 92 | BinaryNgDeviceOperation | 120 | 15 |
| 93 | TernaryDeviceOperation | 120 | 16 |
| 94 | TernaryDeviceOperation | 120 | 16 |
| 95 | TernaryDeviceOperation | 120 | 17 |
| 96 | TernaryDeviceOperation | 120 | 16 |
| 97 | TernaryDeviceOperation | 120 | 16 |
| 98 | TernaryDeviceOperation | 120 | 16 |
| 99 | TernaryDeviceOperation | 120 | 17 |
| 100 | TernaryDeviceOperation | 120 | 16 |
| 101 | TernaryDeviceOperation | 120 | 17 |
| 102 | TernaryDeviceOperation | 120 | 16 |
| 103 | TernaryDeviceOperation | 120 | 17 |
| 104 | TernaryDeviceOperation | 120 | 16 |
| 105 | TernaryDeviceOperation | 120 | 16 |
| 106 | TernaryDeviceOperation | 120 | 16 |
| 107 | TernaryDeviceOperation | 120 | 17 |
| 108 | TernaryDeviceOperation | 120 | 16 |
| 109 | ConcatDeviceOperation | 120 | 15 |
