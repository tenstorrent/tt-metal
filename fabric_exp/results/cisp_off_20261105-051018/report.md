## Per arm and pass (ratio to the profiler-off run of the same pass)

| arm | pass | tests | median | worst | worst test | median >= 0.996 | worst >= 0.989 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |

## NeighborExchangeUnicast_size_4096_ntype_unicast_write_num_links_2


## Per test (ratio per pass)

| test | on a | off b / off a |
| --- | ---: | ---: |
--- router images: off vs on
active_erisc               {'unpaired': 36, 'different': 100}; largest A 8072 B 11720 B; changed B - A +3588 to +3720 B; B LINK_SYNC_ROLE counts {'2': 51, '1': 49}
subordinate_active_erisc   {'different': 100, 'unpaired': 36}; largest A 6268 B 7312 B; changed B - A +1036 to +1044 B; B LINK_SYNC_ROLE counts {'0': 100}
