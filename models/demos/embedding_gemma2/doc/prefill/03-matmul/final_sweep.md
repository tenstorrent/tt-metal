
#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 128: control 4.878 ms (drift +0.012 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 4.877 | -0.01 % | - | |
| only_ple_gate | ple_gate@128 family=2d, in0_block_w=16 | yes | 4.874 | -0.08 % | bitwise | |
| only_ple_wide | ple_wide@128 family=1d | yes | 4.869 | -0.18 % | bitwise | |
| only_qkv_full | qkv_full@128 family=1d | yes | 4.870 | -0.16 % | bitwise | |
| only_qkv_sliding | qkv_sliding@128 family=1d, grid=[8, 8] | yes | 4.871 | -0.13 % | bitwise | |
| all_kept | qkv_sliding@128 family=1d, grid=[8, 8]; qkv_full@128 family=1d; ple_wide@128 family=1d; ple_gate@128 family=2d, in0_block_w=16 | yes | 4.840 | -0.78 % | bitwise | |
| ctl_end | (code default) | yes | 4.878 | +0.01 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 256: control 5.875 ms (drift +0.013 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 5.874 | -0.01 % | - | |
| only_gate | gate@256 family=2d, in0_block_w=16, gelu=mm | yes | 5.864 | -0.18 % | 0.99525 | |
| only_ple_gate | ple_gate@256 family=2d, in0_block_w=16 | yes | 5.867 | -0.13 % | bitwise | |
| all_kept | gate@256 family=2d, in0_block_w=16, gelu=mm; ple_gate@256 family=2d, in0_block_w=16 | yes | 5.855 | -0.34 % | 0.99525 | |
| ctl_end | (code default) | yes | 5.875 | +0.01 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 512: control 7.214 ms (drift -0.011 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 7.215 | +0.01 % | - | |
| ctl_end | (code default) | yes | 7.214 | -0.01 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 1024: control 11.647 ms (drift -0.003 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 11.647 | +0.00 % | - | |
| only_gate | gate@1024 family=2d, in0_block_w=16, gelu=mm | yes | 11.548 | -0.85 % | 0.99649 | |
| only_qkv_sliding | qkv_sliding@1024 family=2d, in0_block_w=16 | yes | 11.639 | -0.07 % | bitwise | |
| only_up | up@1024 family=2d, in0_block_w=16 | yes | 11.628 | -0.16 % | bitwise | |
| all_kept | up@1024 family=2d, in0_block_w=16; qkv_sliding@1024 family=2d, in0_block_w=16; gate@1024 family=2d, in0_block_w=16, gelu=mm | yes | 11.508 | -1.19 % | 0.99649 | |
| ctl_end | (code default) | yes | 11.647 | -0.00 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 2048: control 25.086 ms (drift -0.010 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 25.088 | +0.01 % | - | |
| only_down | down@2048 family=minimal, sb=[2, 2], blocks=[4, 8, 2] | yes | 24.993 | -0.37 % | bitwise | |
| only_gate | gate@2048 family=2d, in0_block_w=16, gelu=mul | yes | 24.798 | -1.15 % | 0.99645 | |
| only_o_sliding | o_sliding@2048 family=2d | yes | 24.790 | -1.18 % | 0.99994 | |
| only_ple_gate | ple_gate@2048 family=2d, gelu=mul | yes | 25.048 | -0.15 % | 0.99614 | |
| only_ple_proj | ple_proj@2048 family=2d | yes | 24.779 | -1.23 % | 0.99782 | |
| only_qkv_full | qkv_full@2048 family=2d, in0_block_w=16 | yes | 25.058 | -0.11 % | bitwise | |
| only_qkv_sliding | qkv_sliding@2048 family=2d, in0_block_w=16 | yes | 24.968 | -0.47 % | bitwise | |
| only_up | up@2048 family=2d, in0_block_w=16 | yes | 24.943 | -0.57 % | bitwise | |
| all_kept | up@2048 family=2d, in0_block_w=16; qkv_sliding@2048 family=2d, in0_block_w=16; qkv_full@2048 family=2d, in0_block_w=16; gate@2048 family=2d, in0_block_w=16, gelu=mul; ple_proj@2048 family=2d; down@2048 family=minimal, sb=[2, 2], blocks=[4, 8, 2]; o_sliding@2048 family=2d; ple_gate@2048 family=2d, gelu=mul | yes | 23.797 | -5.14 % | 0.99611 | |
| ctl_end | (code default) | yes | 25.085 | -0.01 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 4096: control 43.925 ms (drift -0.012 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 43.927 | +0.01 % | - | |
| only_down | down@4096 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 43.645 | -0.64 % | bitwise | |
| only_gate | gate@4096 family=2d, in0_block_w=16, gelu=mul | yes | 43.360 | -1.29 % | 0.99676 | |
| only_o_sliding | o_sliding@4096 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 43.724 | -0.46 % | bitwise | |
| only_ple_gate | ple_gate@4096 family=2d, in0_block_w=16 | yes | 43.880 | -0.10 % | bitwise | |
| only_ple_proj | ple_proj@4096 family=2d | yes | 43.512 | -0.94 % | 0.99780 | |
| only_qkv_full | qkv_full@4096 family=2d, in0_block_w=16 | yes | 43.845 | -0.18 % | 1.00000 | |
| only_qkv_sliding | qkv_sliding@4096 family=2d, in0_block_w=16 | yes | 43.578 | -0.79 % | bitwise | |
| only_up | up@4096 family=2d, in0_block_w=16 | yes | 43.507 | -0.95 % | bitwise | |
| all_kept | up@4096 family=2d, in0_block_w=16; qkv_sliding@4096 family=2d, in0_block_w=16; qkv_full@4096 family=2d, in0_block_w=16; gate@4096 family=2d, in0_block_w=16, gelu=mul; ple_proj@4096 family=2d; down@4096 family=minimal, sb=[2, 2], blocks=[4, 8, 4]; o_sliding@4096 family=minimal, sb=[2, 2], blocks=[4, 8, 4]; ple_gate@4096 family=2d, in0_block_w=16 | yes | 41.614 | -5.26 % | 0.99649 | |
| ctl_end | (code default) | yes | 43.922 | -0.01 % | bitwise | |

#### final_128-256-512-1024-2048-4096-8192.jsonl bucket 8192: control 100.204 ms (drift -0.028 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 100.218 | +0.01 % | - | |
| only_down | down@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 2] | yes | 100.003 | -0.20 % | bitwise | |
| only_gate | gate@8192 family=minimal, gelu=mul | yes | 96.699 | -3.50 % | 0.99698 | |
| only_o_sliding | o_sliding@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 100.090 | -0.11 % | bitwise | |
| only_ple_gate | ple_gate@8192 family=minimal, gelu=mul | yes | 98.848 | -1.35 % | 0.99689 | |
| only_ple_layer | ple_layer@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 100.129 | -0.07 % | bitwise | |
| only_up | up@8192 family=2d, in0_block_w=8, chunk=4096 | yes | 99.936 | -0.27 % | bitwise | |
| all_kept | gate@8192 family=minimal, gelu=mul; ple_gate@8192 family=minimal, gelu=mul; down@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 2]; ple_layer@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 4]; o_sliding@8192 family=minimal, sb=[2, 2], blocks=[4, 8, 4]; up@8192 family=2d, in0_block_w=8, chunk=4096 | yes | 94.737 | -5.46 % | 0.99699 | |
| ctl_end | (code default) | yes | 100.189 | -0.01 % | bitwise | |

#### final_16384.jsonl bucket 16384: control 211.762 ms (drift -0.008 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 211.770 | +0.00 % | - | |
| only_down | down@16384 family=minimal, sb=[2, 2], blocks=[4, 8, 2] | yes | 211.316 | -0.21 % | bitwise | |
| only_gate | gate@16384 family=minimal, gelu=mul | yes | 205.625 | -2.90 % | 0.99741 | |
| only_ple_gate | ple_gate@16384 family=minimal, gelu=mul | yes | 208.483 | -1.55 % | 0.99761 | |
| only_ple_layer | ple_layer@16384 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 211.524 | -0.11 % | bitwise | |
| only_up | up@16384 family=2d, in0_block_w=8, chunk=4096 | yes | 211.403 | -0.17 % | 0.99963 | |
| all_kept | gate@16384 family=minimal, gelu=mul; ple_gate@16384 family=minimal, gelu=mul; down@16384 family=minimal, sb=[2, 2], blocks=[4, 8, 2]; ple_layer@16384 family=minimal, sb=[2, 2], blocks=[4, 8, 4]; up@16384 family=2d, in0_block_w=8, chunk=4096 | yes | 201.325 | -4.93 % | 0.99779 | |
| ctl_end | (code default) | yes | 211.754 | -0.00 % | bitwise | |

#### final_32768.jsonl bucket 32768: control 520.678 ms (drift +0.190 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 520.184 | -0.09 % | - | |
| only_down | down@32768 family=minimal, sb=[2, 2], blocks=[4, 8, 2] | yes | 520.004 | -0.13 % | bitwise | |
| only_gate | gate@32768 family=minimal, gelu=mul | yes | 512.652 | -1.54 % | 0.99766 | |
| only_ple_gate | ple_gate@32768 family=minimal, gelu=mul | yes | 501.296 | -3.72 % | 0.99781 | |
| only_ple_layer | ple_layer@32768 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 520.002 | -0.13 % | bitwise | |
| all_kept | gate@32768 family=minimal, gelu=mul; ple_gate@32768 family=minimal, gelu=mul; down@32768 family=minimal, sb=[2, 2], blocks=[4, 8, 2]; ple_layer@32768 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 492.999 | -5.32 % | 0.99768 | |
| ctl_end | (code default) | yes | 521.171 | +0.09 % | bitwise | |

#### final_65536.jsonl bucket 65536: control 1570.951 ms (drift +0.709 %)

| candidate | spec | legal | burst median ms | vs control | PCC last hidden vs control | note |
|---|---|---|---|---|---|---|
| ctl_start | (code default) | yes | 1565.405 | -0.35 % | - | |
| only_down | down@65536 family=minimal, sb=[2, 2], blocks=[4, 8, 2] | yes | 1573.728 | +0.18 % | bitwise | |
| only_gate | gate@65536 family=2d, in0_block_w=8, chunk=4096, gelu=mm | yes | 1560.362 | -0.67 % | 0.99837 | |
| only_ple_gate | ple_gate@65536 family=minimal, gelu=mul | yes | 1540.519 | -1.94 % | 0.99838 | |
| only_ple_layer | ple_layer@65536 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 1573.780 | +0.18 % | bitwise | |
| all_kept | gate@65536 family=2d, in0_block_w=8, chunk=4096, gelu=mm; ple_gate@65536 family=minimal, gelu=mul; down@65536 family=minimal, sb=[2, 2], blocks=[4, 8, 2]; ple_layer@65536 family=minimal, sb=[2, 2], blocks=[4, 8, 4] | yes | 1529.620 | -2.63 % | 0.99829 | |
| ctl_end | (code default) | yes | 1576.497 | +0.35 % | bitwise | |
